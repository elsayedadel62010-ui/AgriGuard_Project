"""Irrigation prediction (advisory only - the AI never drives the pump).

Three layers, in order of preference:

1. Trained logistic regression loaded from ``models/irrigation_model.json``.
   Ten inputs: the four current sensor readings, three historical features and
   three crop indicators. Gives a ``prediction_score`` in 0..1. Consulted only
   when every client-supplied reading lies inside the range recorded at training
   time - see `DECISION_SOURCE_OUT_OF_RANGE`.
2. Safety guards: if the pump is already running or water is already moving, the
   answer is always "Irrigation Not Needed" so the backend can never be talked
   into a double irrigation.
3. Rule-based fallback: crop-specific soil-moisture thresholds with an
   evaporative-demand adjustment. Fully interpretable. Used when no trained model
   file is present, and when the model's own answer would be an extrapolation -
   the fitted ``light`` coefficient is negative, so on weather it has never seen
   the model scores a dry plot in the wrong direction with high confidence.

The model file stays JSON on purpose: the backend never has to understand any
ML internals, and inference needs nothing beyond numpy.
"""

import json
import math
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import IRRIGATION_MODEL_PATH
from .contracts import IrrigationRequest
from .preprocess import (
    CROP_FEATURES,
    FEATURE_COLUMNS,
    MAX_MINUTES_SINCE_IRRIGATION,
    build_features_from_request,
    fill_historical_defaults,
    scale_features,
)
from .version import IRRIGATION_MODEL_VERSION

#: Soil-moisture thresholds used by the rule-based fallback only.
#: These are placeholders from agricultural extension ranges, NOT calibrated
#: for the AgriGuard plots. See docs/limitations.md.
CROP_THRESHOLDS = {
    "tomato": 25.0,
    "maize": 27.0,
    "wheat": 22.0,
    "rice": 60.0,
    "potato": 30.0,
    "soybean": 24.0,
    "onion": 28.0,
    "chili": 26.0,
    "pepper": 24.0,
    "default": 25.0,
}

PUMP_RUNNING_GUARD = "pump already running (pump_status=1)"
WATER_RUNNING_GUARD = "water already flowing"

#: The crops the trained model actually has indicator columns for, i.e.
#: `CROP_FEATURES` in preprocess.py. Only these may be scored by the model.
TRAINED_CROPS = frozenset(
    column[len("crop_is_") :] for column in CROP_FEATURES
)

#: Crop names the fallback rule knows, i.e. every key of `CROP_THRESHOLDS`
#: except the "default" placeholder. A crop in here but not in
#: `TRAINED_CROPS` is answered by the rule, never by the model.
RULE_ONLY_CROPS = frozenset(CROP_THRESHOLDS) - TRAINED_CROPS - {"default"}

#: Every crop name the service accepts, and the union the two constants above
#: partition. Anything outside this is an invalid crop and is rejected.
KNOWN_CROPS = TRAINED_CROPS | RULE_ONLY_CROPS

#: The model is a linear function of the sensor readings, so outside the range it
#: was fitted on it extrapolates, and the weather features are the dangerous ones:
#: in the synthetic data the label marks the instant a threshold scheduler fired,
#: so daytime is correlated with "already watered" and the fitted `light`
#: coefficient came out negative. Given a reading it has never seen, the model
#: therefore extrapolates that trend hard - a tomato plot at 16 % soil moisture,
#: well past that crop's trigger, was told "do not irrigate" at
#: -10 C / 0 % RH / 200 000 lux. No synthetic row is anywhere near that.
#:
#: The fix is not to override the model's decision, which would also flatten the
#: history sensitivity that makes the model worth having. The fix is to admit
#: when the model cannot answer: if any reading it consumes falls outside the
#: range recorded at training time, its opinion is not reported. The crop
#: threshold rule answers instead, which is conservative by construction
#: (recommended whenever moisture is at or below the crop's reference
#: threshold), and the response says which of the two produced the advice.
#:
#: See `models/irrigation_model.json -> training_feature_ranges` for the recorded
#: bounds and docs/limitations.md for the measured failure this prevents.
DECISION_SOURCE_MODEL = "model"
DECISION_SOURCE_OUT_OF_RANGE = "rule_fallback_out_of_range"
DECISION_SOURCE_PUMP_GUARD = "pump_guard"
DECISION_SOURCE_WATER_GUARD = "water_guard"
DECISION_SOURCE_RULE_FALLBACK = "rule_fallback"

#: Every prediction carries this flag. The AI layer recommends; the backend and
#: the ESP32 decide whether and how to act, with their own safety limits.
ADVISORY_ONLY = True


def _advisory(result: Dict[str, object]) -> Dict[str, object]:
    result["advisory_only"] = ADVISORY_ONLY
    return result


def _normalise_crop(crop_type: str) -> str:
    """Fold a crop name to the form CROP_THRESHOLDS is keyed by.

    The threshold table and the one-hot encoder are both case-insensitive, so
    every crop-dependent decision must fold case before looking a crop up -
    otherwise "Maize" and "maize" silently take different paths.
    """
    return (crop_type or "").strip().lower()


def evaporative_demand_factor(req: IrrigationRequest) -> float:
    """~1.0 in mild conditions, higher when the crop loses water faster."""
    heat = max(1.0, (req.temperature - 28.0) * 0.05)
    dryness = max(1.0, (55.0 - req.humidity) * 0.02)
    sun = max(1.0, req.light / 60000.0)
    return float(min(2.5, max(0.7, heat * dryness * sun)))


def rule_based_recommendation(
    req: IrrigationRequest,
    threshold: float = 0.5,
) -> Dict[str, object]:
    """Interpretable fallback used when the model cannot be trusted to answer.

    Called both when no trained model file is present and when the model is
    loaded but the request falls outside the range the model was fitted on, so
    it is not merely a "no model available" path.

    `prediction_score` is on the SAME scale as the model's: an estimate of how
    strongly the evidence points towards irrigation, crossing `threshold` at
    exactly the crop's reference moisture. That matters because the API
    documents `irrigation_needed == (prediction_score >= decision_threshold)` as
    the way for a backend to re-derive the decision. An earlier version scored
    both branches between 0.5 and 0.95, which made that check true even while
    reporting "do not water" - harmless while this was only reachable with no
    model file, a live contract violation once out-of-range requests started
    routing here.

    The decision itself is moisture-only, and the evaporative demand factor
    never moved it; it is reported in the reason for a human reader rather than
    folded into the score, where a multiplier would smear the crossing.
    """
    # Crop names arrive as free text from a request body, so the same crop can
    # be spelled "maize", "Maize" or " maize ". The one-hot encoder in
    # preprocess.crop_indicators has always folded case, and the guard below
    # normalises before deciding whether the model may answer, but the
    # threshold lookup here used the raw string: a differently-cased name
    # missed CROP_THRESHOLDS and silently took the default 25% reference
    # instead of maize's 27%, which inverted the decision in the 25-27% band.
    # Normalising once keeps the lookup and the reported reason in agreement.
    crop = _normalise_crop(req.crop_type)
    reference = CROP_THRESHOLDS.get(crop, CROP_THRESHOLDS["default"])
    deficit = reference - req.soil_moisture
    needed = deficit >= 0.0

    # Monotone in moisture, crossing `threshold` exactly at `reference`:
    # score == threshold when soil_moisture == reference, and 0.5 at moisture 0.
    score = threshold + deficit / (2.0 * max(reference, 1e-6))
    score = min(1.0, max(0.0, score))

    if needed:
        reason = (
            f"soil moisture {req.soil_moisture:g} is {deficit:g} below the "
            f"{reference:g} reference threshold for {crop}; "
            f"evaporative demand factor {evaporative_demand_factor(req):.2f}"
        )
    else:
        reason = (
            f"soil moisture {req.soil_moisture:g} is at or above the "
            f"{reference:g} reference threshold for {crop}; "
            f"evaporative demand factor {evaporative_demand_factor(req):.2f}"
        )
    return _advisory({
        "irrigation_needed": needed,
        "prediction": "Irrigation Recommended" if needed else "Irrigation Not Needed",
        "prediction_score": round(float(score), 3),
        "irrigation_duration_min": None,
        # null, not false: no model was consulted at all, so the question
        # "were the historical features real?" does not even apply.
        "history_used": None,
        "decision_source": DECISION_SOURCE_RULE_FALLBACK,
        "reason": reason,
    })


def _not_needed(
    reason: str,
    score: float = 0.0,
    decision_source: str = DECISION_SOURCE_PUMP_GUARD,
) -> Dict[str, object]:
    """Safety-guard response.

    `score` defaults to 0.0 so it sits on the same scale as the model branch:
    low means "no evidence of need". A guard is a certainty, not a hunch, and
    reporting e.g. 0.75 next to "do not water" would break the documented
    `irrigation_needed == (prediction_score >= decision_threshold)` rule in the
    one case where following it wrongly would water a plot whose pump is already
    running.
    """
    return _advisory(
        {
            "irrigation_needed": False,
            "prediction": "Irrigation Not Needed",
            "prediction_score": round(float(score), 3),
            "irrigation_duration_min": None,
            "history_used": None,
            "decision_source": decision_source,
            "reason": reason,
        }
    )


class IrrigationModel:
    """Loads models/irrigation_model.json and scores a request."""

    def __init__(self, model_path=None):
        self.model_path = model_path or IRRIGATION_MODEL_PATH
        self.coefs: Optional[np.ndarray] = None
        self.intercept: float = 0.0
        self.means: Optional[np.ndarray] = None
        self.stds: Optional[np.ndarray] = None
        self.feature_medians: Dict[str, float] = {}
        #: Per-feature [min, max] observed at training time. Absent from older
        #: model files, in which case `out_of_range_features` always returns {}
        #: and the guard is simply inactive - the model then answers every
        #: request, which is the pre-guard behaviour.
        self.feature_ranges: Dict[str, List[float]] = {}
        self.feature_columns: List[str] = list(FEATURE_COLUMNS)
        # Operating point chosen on a held-out slice of the TRAINING timeline.
        # 0.5 is the plain logistic default and stays the fallback for old files.
        self.decision_threshold: float = 0.5
        self.metadata: Dict[str, object] = {}
        self.load_error: Optional[str] = None
        self._loaded = False
        if self.model_path.exists():
            self._load()

    # -- loading ---------------------------------------------------------
    def _load(self) -> None:
        try:
            with open(self.model_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception as exc:  # noqa: BLE001 - reported through /v1/health
            self.load_error = f"could not read {self.model_path.name}: {exc}"
            return

        try:
            features = list(data["features"])
            if features != list(FEATURE_COLUMNS):
                # A model file is only usable if it was trained on exactly the
                # features this version uses. Exact equality, not a subset test:
                # the coefficients, the scaler means/stds and the medians are all
                # aligned to one ordered feature vector, so a model naming a
                # different set of columns is in a different basis and cannot be
                # interpreted here.
                #
                # This matters in both directions. v0.1.0 files were trained on
                # the four current readings only; those are a SUBSET of today's
                # feature list, so a subset check would have accepted them and
                # silently scored with four coefficients against a ten-column
                # vector. Rejecting surfaces a visible load_error in /v1/health
                # instead of quiet nonsense. Retrain with
                # scripts/train_irrigation.py.
                if set(features) == set(FEATURE_COLUMNS):
                    self.feature_columns = features
                else:
                    missing = sorted(set(FEATURE_COLUMNS) - set(features))
                    extra = sorted(set(features) - set(FEATURE_COLUMNS))
                    raise ValueError(
                        "model file was not trained on this version's feature "
                        f"set. missing={missing} unexpected={extra}. "
                        f"Retrain it with scripts/train_irrigation.py. "
                        f"expected={sorted(FEATURE_COLUMNS)}"
                    )
            self.coefs = np.asarray(data["coefs"], dtype=float)
            self.intercept = float(data["intercept"])
            self.means = np.asarray(data["means"], dtype=float)
            self.stds = np.asarray(data["stds"], dtype=float)
            self.feature_medians = dict(data.get("feature_medians", {}))
            ranges = data.get("training_feature_ranges") or {}
            self.feature_ranges = {
                str(name): [float(bounds[0]), float(bounds[1])]
                for name, bounds in ranges.items()
                if isinstance(bounds, (list, tuple)) and len(bounds) == 2
            }
            self.decision_threshold = float(data.get("decision_threshold", 0.5))
            self.metadata = {
                key: data[key]
                for key in (
                    "model_version",
                    "trained_at",
                    "training_dataset",
                    "training_rows",
                    "test_rows",
                    "split",
                    "label_meaning",
                    "limitations",
                )
                if key in data
            }
            if len(self.coefs) != len(self.feature_columns):
                raise ValueError("coefs length does not match the feature list")
            self._loaded = True
            self.load_error = None
        except Exception as exc:  # noqa: BLE001
            self.load_error = f"invalid model file: {exc}"
            self.coefs = None

    @property
    def is_trained(self) -> bool:
        return self._loaded

    @property
    def model_version(self) -> str:
        return str(self.metadata.get("model_version", IRRIGATION_MODEL_VERSION))

    def describe(self) -> Dict[str, object]:
        return {
            "trained": self._loaded,
            "model_version": self.model_version,
            "features": list(self.feature_columns),
            "decision_threshold": self.decision_threshold,
            "training_feature_ranges": {
                name: list(bounds) for name, bounds in self.feature_ranges.items()
            },
            "trained_at": self.metadata.get("trained_at"),
            "training_dataset": self.metadata.get("training_dataset"),
            "load_error": self.load_error,
        }

    # -- scoring ---------------------------------------------------------
    def score_features(self, features: np.ndarray) -> float:
        """Logistic score that irrigation is needed, given a full feature vector."""
        order = [FEATURE_COLUMNS.index(name) for name in self.feature_columns]
        subset = np.asarray(features, dtype=float)[order]
        subset = fill_historical_defaults(subset, self.feature_medians)
        z = scale_features(subset, self.means, self.stds)
        return float(1.0 / (1.0 + math.exp(-(float(np.dot(self.coefs, z)) + self.intercept))))

    def predict(self, req: IrrigationRequest) -> Dict[str, object]:
        # Safety guards run before everything, trained model or not.
        if req.pump_status == 1:
            return _not_needed(PUMP_RUNNING_GUARD, decision_source=DECISION_SOURCE_PUMP_GUARD)
        if req.water_flow > 0:
            return _not_needed(WATER_RUNNING_GUARD, decision_source=DECISION_SOURCE_WATER_GUARD)

        crop = _normalise_crop(req.crop_type)
        if crop not in TRAINED_CROPS:
            # The crop has no indicator column, so scoring it would run the
            # model against an all-zero crop block and report the result as if
            # it were in-domain. The backend resolves crop_type from the plot
            # configuration, so reaching this branch means the caller sent a
            # crop the model was not trained on. Answer with the rule instead
            # and say so: the decision source is the only thing a client needs
            # to tell a validated model answer from an interpolated one.
            result = rule_based_recommendation(req, self.decision_threshold)
            if crop in RULE_ONLY_CROPS:
                result["reason"] = (
                    f"[rule fallback] crop_type '{crop}' has no indicator column in "
                    f"the trained model ({', '.join(sorted(TRAINED_CROPS))}), so the "
                    f"model was not consulted. {result['reason']}"
                )
            else:
                result["reason"] = (
                    f"[rule fallback] crop_type '{crop or 'default'}' is not a crop "
                    f"this service recognises, so the model was not consulted. "
                    f"{result['reason']}"
                )
            return result

        if not self._loaded:
            result = rule_based_recommendation(req, self.decision_threshold)
            result["reason"] = f"[rule fallback] {result['reason']}"
            return result

        features, notes, _placeholder = self._request_features(req)
        threshold = self.decision_threshold

        # Can the model answer this request at all? See DECISION_SOURCE_* above
        # for the measured failure that motivates the check. The score is
        # computed either way and quoted in the reason, because a reading that
        # falls outside the training range is exactly when a caller most wants
        # to see what the model thought and how little it is worth.
        score = self.score_features(features)
        history_used = not any("placeholder" in note for note in notes)

        out_of_range = self.out_of_range_features(features)
        if out_of_range:
            result = rule_based_recommendation(req, threshold)
            result["decision_source"] = DECISION_SOURCE_OUT_OF_RANGE
            result["history_used"] = history_used
            listed = ", ".join(
                f"{name}={value:g} (trained {self.feature_ranges[name][0]:g}"
                f"..{self.feature_ranges[name][1]:g})"
                for name, value in out_of_range.items()
            )
            # The model's score is quoted here, but `prediction_score` carries the
            # rule's. The wording keeps the two apart on purpose: an earlier
            # phrasing called the model score just "score", which read as
            # contradicting the field it sits next to.
            result["reason"] = (
                f"OUT OF TRAINING RANGE: {listed}. The model extrapolates "
                f"outside the readings it was fitted on; it would have scored "
                f"{score:.2f} against a threshold of {threshold:.2f}, and that "
                f"is reported here for diagnosis only. prediction_score "
                f"({result['prediction_score']}) and the decision both come from "
                f"the crop threshold rule."
            )
            return _add_caveats(result, req, notes, history_used)

        if score >= threshold:
            result = _advisory({
                "irrigation_needed": True,
                "prediction": "Irrigation Recommended",
                "prediction_score": round(score, 3),
                "irrigation_duration_min": None,
                "history_used": history_used,
                "decision_source": DECISION_SOURCE_MODEL,
                "reason": (
                    f"logistic-regression score {score:.2f} >= "
                    f"{threshold:.2f}; "
                ) + _feature_summary(req),
            })
        else:
            result = _advisory({
                "irrigation_needed": False,
                "prediction": "Irrigation Not Needed",
                # The raw logistic score, NOT 1 - score. A backend is documented
                # to be able to re-derive the decision with
                # `prediction_score >= decision_threshold`; inverting the score
                # here would report 1.0 for "do not water" and make that check
                # water a wet plot. Both branches report the same quantity, so
                # the score and the reason string can never disagree.
                "prediction_score": round(score, 3),
                "irrigation_duration_min": None,
                "history_used": history_used,
                "decision_source": DECISION_SOURCE_MODEL,
                "reason": (
                    f"logistic-regression score {score:.2f} < "
                    f"{threshold:.2f}; "
                    f"{_feature_summary(req)}"
                ),
            })
        return _add_caveats(result, req, notes, history_used)

    def out_of_range_features(
        self,
        features: np.ndarray,
    ) -> Dict[str, float]:
        """Client-supplied readings the model was not fitted on, as ``{name: value}``.

        Empty when every reading lies inside the range recorded at training
        time. Only `CLIENT_SUPPLIED_FEATURES` are checked; see that constant for
        why the derived historical features are deliberately excluded.

        Ranges are per feature, not a Mahalanobis-style joint region. A joint
        test would be tighter, but with three correlated weather inputs the
        honest statement is the weaker one: a point inside every per-feature
        range may still be an unusual combination, and the model should not be
        read as more reliable than that.
        """
        if not self.feature_ranges:
            return {}
        row = np.asarray(features, dtype=float)
        out: Dict[str, float] = {}
        for index, name in enumerate(self.feature_columns):
            if name not in CLIENT_SUPPLIED_FEATURES:
                continue
            bounds = self.feature_ranges.get(name)
            if not bounds or index >= len(row):
                continue
            low, high = float(bounds[0]), float(bounds[1])
            value = float(row[index])
            # Tolerate float noise so a value sitting exactly on a recorded
            # bound is not reported as outside it.
            slack = 1e-9 * max(1.0, abs(low), abs(high))
            if value < low - slack or value > high + slack:
                out[name] = value
        return out

    def _request_features(self, req: IrrigationRequest) -> tuple:
        """Build the feature vector for a live request, honouring optional history."""
        recent: Optional[List[tuple]] = None
        last_minutes: Optional[float] = None
        current_ts: Optional[pd.Timestamp] = None
        notes_to_add: List[str] = []

        if req.timestamp:
            try:
                current_ts = pd.Timestamp(req.timestamp)
                if current_ts.tzinfo is not None:
                    current_ts = current_ts.tz_convert("UTC").tz_localize(None)
            except Exception:  # noqa: BLE001 - contracts validate the format separately
                current_ts = None

        if req.history is not None:
            readings = req.history.recent_soil_moisture or []
            recent = [(r.timestamp, r.soil_moisture) for r in readings]
            if req.history.last_irrigation_at and current_ts is None:
                # Silent-drop guard. last_irrigation_at is an absolute instant,
                # so turning it into "minutes ago" needs a reference time. With
                # no request timestamp there is nothing to subtract from, and
                # falling back to wall-clock time would make every response
                # depend on when it was called, which no test can pin down. So
                # the value is dropped - but it is reported, because a client
                # that sent it and got back "no irrigation history" would
                # otherwise have no way to tell a dropped field from a missing
                # one.
                notes_to_add = [
                    "last_irrigation_at was supplied but ignored: it is an "
                    "absolute time and this request has no timestamp, so "
                    "minutes-since-last-irrigation cannot be computed; send "
                    "timestamp to use it"
                ]
            elif req.history.last_irrigation_at and current_ts is not None:
                try:
                    stamp = pd.Timestamp(req.history.last_irrigation_at)
                    if stamp.tzinfo is not None:
                        stamp = stamp.tz_convert("UTC").tz_localize(None)
                    last_minutes = max(
                        0.0,
                        min(
                            MAX_MINUTES_SINCE_IRRIGATION,
                            (current_ts - stamp).total_seconds() / 60.0,
                        ),
                    )
                except Exception:  # noqa: BLE001
                    last_minutes = None

        features, notes = build_features_from_request(
            req,
            previous_soil_moisture=recent[-1][1] if recent else None,
            recent_soil_moisture=recent,
            last_irrigation_minutes_ago=last_minutes,
            current_timestamp=current_ts,
        )

        # Which historical features the service invented rather than read. Tracked
        # for the reason text only; the out-of-range guard does not consult it,
        # since it never range-checks the derived features in the first place.
        placeholder: List[str] = []
        if not recent:
            placeholder.extend(["soil_moisture_change", "soil_moisture_avg_30min"])
        if last_minutes is None:
            placeholder.append("time_since_last_irrigation_min")

        return features, notes + notes_to_add, placeholder


#: The features a client supplies directly, as opposed to the ones the service
#: derives. Only these are range-checked at inference, and the choice is
#: deliberate rather than a shortcut.
#:
#: The three historical features are derived - `soil_moisture_change` is the
#: gap between the current reading and the previous one - so a single legitimate
#: current reading moved against a slightly stale history pushes the change
#: outside the marginal range the model was fitted on. Checking it turned the
#: documented per-crop trigger into an artifact of that arithmetic: sweeping
#: current moisture necessarily sweeps the change with it, so the boundary came
#: out as the crop's reference threshold regardless of what the model thought,
#: and the history sensitivity that makes the model worth having became
#: unreachable above ~26 % soil moisture.
#:
#: Nothing is given up by skipping them. The direction each historical
#: coefficient extrapolates is the safe one: `soil_moisture_change` and
#: `soil_moisture_avg_30min` both carry negative coefficients, so a reading
#: further outside than any seen drives the score UP (more irrigation) or
#: coincides with soil that really is wetter. The features that can talk a
#: farmer out of watering dry soil are the three weather readings, and all
#: three are supplied directly. What remains unguarded is a client feeding a
#: bogus history, which is stated in docs/limitations.md rather than papered
#: over here.
CLIENT_SUPPLIED_FEATURES = (
    "soil_moisture",
    "temperature",
    "humidity",
    "light",
)


def _add_caveats(
    result: Dict[str, object],
    req: IrrigationRequest,
    notes: List[str],
    history_used: bool,
) -> Dict[str, object]:
    """Append the caveats that apply whichever way the decision was reached.

    Shared by the model branch and the out-of-range branch on purpose. A client
    that reads `history_used: false` deserves the same explanation regardless of
    which rule then made the call, and a caveat that appears in only one branch
    is a caveat some clients will never see.

    There is deliberately no "crop was not seen in training" caveat here. Both
    callers are only reachable once the crop guard has confirmed the crop has a
    trained indicator column, so such a note could never fire; crops without one
    are diverted to `rule_based_recommendation` before these branches run, and
    that path already says so in its own reason.
    """
    reason = str(result["reason"])
    if not history_used:
        reason += (
            " | NOTE: no usable history was supplied, so the three "
            "historical features are placeholders and the score is not "
            "trustworthy (see models/irrigation_metrics.json -> "
            "test_metrics_without_history)"
        )
    if notes:
        reason += " | " + "; ".join(notes)
    result["reason"] = reason
    return result


def _feature_summary(req: IrrigationRequest) -> str:
    return (
        f"soil_moisture={req.soil_moisture:g}, temperature={req.temperature:g}, "
        f"humidity={req.humidity:g}, light={req.light:g}"
    )


irrigation_model = IrrigationModel()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
