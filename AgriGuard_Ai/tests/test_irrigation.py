"""Tests for the irrigation model and its preprocessing.

Coverage required by the brief:
  valid input, dry soil, wet soil, invalid sensor values, missing required
  fields, prediction output shape, model loading.
"""

import json
import re

import numpy as np
import pandas as pd
import pytest

from agriguard.config import IRRIGATION_CSV, IRRIGATION_MODEL_PATH
from agriguard.contracts import IrrigationRequest
from agriguard.irrigation import (
    CROP_THRESHOLDS,
    TRAINED_CROPS,
    IrrigationModel,
    evaporative_demand_factor,
    rule_based_recommendation,
)
from agriguard.preprocess import (
    CROP_FEATURES,
    FEATURE_COLUMNS,
    HISTORICAL_FEATURES,
    MAX_MINUTES_SINCE_IRRIGATION,
    NUMERIC_FEATURE_COLUMNS,
    UNUSED_WEATHER_COLUMNS,
    WEATHER_FEATURES,
    MissingColumnsError,
add_historical_features,
build_feature_matrix,
build_features_from_request,
chronological_split,
clean_irrigation_frame,
crop_indicators,
load_irrigation_csv,
scale_features,
)
from scripts import decision_boundaries  # noqa: E402


def make_req(**overrides):
    base = dict(
        timestamp="2026-09-27T18:30:00Z",
        device_id="ESP32_01",
        plot_id="PLOT_A",
        soil_moisture=30.0,
        temperature=29.0,
        humidity=58.0,
        light=700.0,
        npk_n=None,
        npk_p=None,
        npk_k=None,
        water_flow=0.0,
        pump_status=0,
        crop_type="tomato",
    )
    base.update(overrides)
    return IrrigationRequest(**base)


# ---------------------------------------------------------------- model input
def test_request_requires_the_four_sensor_readings():
    for field in ("soil_moisture", "temperature", "humidity", "light"):
        payload = {
            "soil_moisture": 35,
            "temperature": 29,
            "humidity": 58,
            "light": 700,
        }
        payload.pop(field)
        with pytest.raises(Exception):
            IrrigationRequest(**payload)


@pytest.mark.parametrize("field", ["soil_moisture", "humidity", "light"])
def test_invalid_sensor_values_are_rejected(field):
    values = {"soil_moisture": -1, "humidity": 140, "light": -5}
    with pytest.raises(Exception):
        make_req(**{field: values[field]})


def test_pump_status_must_be_zero_or_one():
    with pytest.raises(Exception):
        make_req(pump_status=7)


def test_valid_request_has_device_and_plot():
    req = make_req()
    assert req.device_id == "ESP32_01"
    assert req.plot_id == "PLOT_A"


# ---------------------------------------------------------------- predictions
def test_dry_soil_recommends_irrigation():
    pred = IrrigationModel(model_path=IRRIGATION_MODEL_PATH).predict(
        make_req(soil_moisture=12.0)
    )
    assert pred["irrigation_needed"] is True
    assert pred["prediction"] == "Irrigation Recommended"


def test_wet_soil_does_not_recommend_irrigation():
    pred = IrrigationModel(model_path=IRRIGATION_MODEL_PATH).predict(
        make_req(soil_moisture=70.0)
    )
    assert pred["irrigation_needed"] is False
    assert pred["prediction"] == "Irrigation Not Needed"


def test_pump_already_running_blocks_a_second_command():
    model = IrrigationModel(model_path=IRRIGATION_MODEL_PATH)
    pred = model.predict(make_req(soil_moisture=8.0, pump_status=1, water_flow=5.0))
    assert pred["irrigation_needed"] is False
    assert "pump" in pred["reason"].lower()


def test_water_flowing_blocks_a_second_command():
    model = IrrigationModel(model_path=IRRIGATION_MODEL_PATH)
    pred = model.predict(make_req(soil_moisture=8.0, water_flow=6.5))
    assert pred["irrigation_needed"] is False
    assert "flow" in pred["reason"].lower()


def test_prediction_score_is_a_fraction_not_a_guarantee():
    model = IrrigationModel(model_path=IRRIGATION_MODEL_PATH)
    for moisture in np.linspace(1.0, 99.0, 25):
        pred = model.predict(make_req(soil_moisture=float(moisture)))
        assert 0.0 <= pred["prediction_score"] <= 1.0


def test_irrigation_duration_is_always_null():
    """AgriGuard has no defensible duration model, so it must never guess one."""
    model = IrrigationModel(model_path=IRRIGATION_MODEL_PATH)
    for moisture in (5.0, 30.0, 80.0):
        pred = model.predict(make_req(soil_moisture=moisture))
        assert pred["irrigation_duration_min"] is None


def test_prediction_has_every_contract_key():
    model = IrrigationModel(model_path=IRRIGATION_MODEL_PATH)
    pred = model.predict(make_req(soil_moisture=12.0))
    for key in (
        "irrigation_needed",
        "prediction",
        "prediction_score",
        "irrigation_duration_min",
        "reason",
        "history_used",
        "advisory_only",
    ):
        assert key in pred
    assert pred["advisory_only"] is True


def test_history_used_is_false_without_history_and_true_with_it():
    """A score computed from placeholder features must be flagged, not trusted."""
    model = IrrigationModel(model_path=IRRIGATION_MODEL_PATH)

    without = model.predict(make_req(soil_moisture=12.0))
    assert without["history_used"] is False
    assert "not trustworthy" in without["reason"]

    with_history = model.predict(
        make_req(
            soil_moisture=12.0,
            timestamp="2026-09-27T18:30:00Z",
            history={
                "recent_soil_moisture": [
                    {"timestamp": "2026-09-27T18:10:00Z", "soil_moisture": 36.4},
                    {"timestamp": "2026-09-27T18:20:00Z", "soil_moisture": 35.6},
                ],
                "last_irrigation_at": "2026-09-27T15:00:00Z",
            },
        )
    )
    assert with_history["history_used"] is True
    assert "not trustworthy" not in with_history["reason"]


def test_a_dropped_last_irrigation_at_is_reported_not_silently_ignored():
    """last_irrigation_at needs a request timestamp; if it is dropped, say so.

    A client that sends history without a timestamp gets
    time_since_last_irrigation_min filled with a placeholder. The response
    must distinguish that from never having sent the field at all,
    otherwise an integrator cannot tell why their history was ignored.
    """
    model = IrrigationModel(model_path=IRRIGATION_MODEL_PATH)

    dropped = model.predict(
        make_req(
            soil_moisture=12.0,
            timestamp=None,
            history={
                "recent_soil_moisture": [
                    {"timestamp": "2026-09-27T18:20:00Z", "soil_moisture": 35.6},
                ],
                "last_irrigation_at": "2026-09-27T15:00:00Z",
            },
        )
    )
    assert "last_irrigation_at was supplied but ignored" in dropped["reason"]
    assert "no timestamp" in dropped["reason"]
    # The score is still untrustworthy, because the feature is a placeholder.
    assert dropped["history_used"] is False

    honoured = model.predict(
        make_req(
            soil_moisture=12.0,
            timestamp="2026-09-27T18:30:00Z",
            history={
                "recent_soil_moisture": [
                    {"timestamp": "2026-09-27T18:20:00Z", "soil_moisture": 35.6},
                ],
                "last_irrigation_at": "2026-09-27T15:00:00Z",
            },
        )
    )
    assert "last_irrigation_at was supplied but ignored" not in honoured["reason"]
    assert honoured["history_used"] is True


def test_history_used_is_null_when_the_model_is_not_consulted():
    model = IrrigationModel(model_path=IRRIGATION_MODEL_PATH)
    guard = model.predict(make_req(soil_moisture=6.0, pump_status=1))
    assert guard["irrigation_needed"] is False
    assert guard["history_used"] is None

    fallback = rule_based_recommendation(make_req(soil_moisture=6.0))
    # No model was consulted, so history_used is null rather than false: the
    # field answers "were real historical features used?", which is not a
    # meaningful question when the model never ran.
    assert fallback["history_used"] is None


def test_advisory_wording_never_orders_the_hardware():
    model = IrrigationModel(model_path=IRRIGATION_MODEL_PATH)
    pred = model.predict(make_req(soil_moisture=6.0))
    assert pred["prediction"] in {"Irrigation Recommended", "Irrigation Not Needed"}
    text = json.dumps(pred).lower()
    assert "must turn on" not in text
    assert "activate pump" not in text


# ---------------------------------------------------------------- rule fallback
def test_rule_fallback_uses_crop_thresholds():
    assert (
        rule_based_recommendation(make_req(crop_type="rice", soil_moisture=55.0))[
            "irrigation_needed"
        ]
        is True
    )
    assert (
        rule_based_recommendation(make_req(crop_type="tomato", soil_moisture=55.0))[
            "irrigation_needed"
        ]
        is False
    )


def test_rule_fallback_runs_without_any_model_file():
    model = IrrigationModel(model_path=IRRIGATION_MODEL_PATH.parent / "does_not_exist.json")
    assert model.is_trained is False
    pred = model.predict(make_req(soil_moisture=10.0))
    assert pred["irrigation_needed"] is True
    assert "rule fallback" in pred["reason"]


def test_evaporative_demand_rises_with_heat_and_sun():
    mild = evaporative_demand_factor(
        make_req(temperature=20.0, humidity=80.0, light=500.0)
    )
    harsh = evaporative_demand_factor(
        make_req(temperature=38.0, humidity=30.0, light=90000.0)
    )
    assert harsh > mild


def test_the_documented_decision_boundaries_are_current(irrigation_trained):
    """The per-crop boundaries quoted in the docs must match the shipped model.

    The measurement code is imported from `scripts/decision_boundaries.py` rather
    than reimplemented here, on purpose. The docs quote what that script prints;
    a second copy of the sweep in the test suite is a second thing to keep in
    sync, and it already drifted once - the test's own version of the history
    context differed from the script's by an irrigation timestamp and reported a
    boundary 0.6 points off. If the script changes, this test follows it and the
    documented numbers fail here instead of going stale in the README.
    """
    model = IrrigationModel(model_path=IRRIGATION_MODEL_PATH)

    # the "recently watered" context from the script's CONTEXTS table
    window = decision_boundaries.feasible_window(35.0, -2.0)

    def measured(crop):
        ask = decision_boundaries.make_asker(model, crop, 35.0, -2.0, 60.0)
        served, _ = decision_boundaries.boundary(ask, model.decision_threshold, window)
        return served

    # Rice is deliberately absent: its trigger sits above the range where that
    # history context is representable, so there is no boundary to pin. The docs
    # quote "> 51.5 %" for it rather than a number taken from the search limit.
    documented = {"tomato": 20.8, "pepper": 20.5}
    for crop, expected in documented.items():
        flip = measured(crop)
        assert flip == expected, (
            f"{crop} boundary is {flip}, docs say {expected}. "
            "Re-run scripts/decision_boundaries.py and update the docs."
        )

    rice = measured("rice")
    assert decision_boundaries.is_truncated(rice, window), (
        f"rice boundary is now {rice}, which is inside the window. If a real "
        "boundary was found, quote it in the docs and add it to `documented`."
    )


def test_the_score_always_agrees_with_the_decision(irrigation_trained):
    """`prediction_score >= decision_threshold` must equal `irrigation_needed`.

    The documented contract tells the backend it can re-derive the decision from
    the score, so the invariant has to hold on every path - not just the happy
    one. Two things used to break it:

    - the "Irrigation Not Needed" branch reported `1 - score`, so a wet plot came
      back as `prediction_score: 1.0` next to "irrigation_needed: false" and a
      reason quoting 0.00. A backend following the documented check would have
      watered a soaked plot.
    - the rule-based fallback scored both of its branches between 0.5 and 0.95,
      so the check was true even while reporting "do not water". That was
      invisible while the fallback was only reachable with no model file on disk;
      it became a live contract violation once out-of-range requests started
      routing through it.
    """
    model = IrrigationModel(model_path=IRRIGATION_MODEL_PATH)
    threshold = model.decision_threshold

    cases = [
        (12.0, "tomato"), (18.0, "tomato"), (20.0, "tomato"),
        (25.0, "tomato"), (45.0, "tomato"), (68.0, "tomato"),
        (20.0, "rice"), (55.0, "rice"), (68.0, "rice"),
        (18.0, "pepper"), (30.0, "pepper"),
    ]
    checked = {"water": 0, "no water": 0}
    sources = set()

    for moisture, crop in cases:
        for send_history in (False, True):
            kwargs = {}
            if send_history:
                kwargs["history"] = {
                    "recent_soil_moisture": [
                        {"timestamp": "2026-09-27T18:00:00Z", "soil_moisture": moisture + 2},
                    ],
                    "last_irrigation_at": "2026-09-27T12:00:00Z",
                }
            pred = model.predict(
                make_req(soil_moisture=moisture, crop_type=crop, **kwargs)
            )
            score = pred["prediction_score"]
            needed = pred["irrigation_needed"]
            source = pred["decision_source"]
            sources.add(source)
            label = f"{crop} at {moisture} % (history={send_history}, {source})"

            assert 0.0 <= score <= 1.0, label
            # No exceptions, and that is the point: `decision_source` tells a
            # caller what produced the advice, but the score and the decision
            # agree on their own, so nothing needs special-casing.
            assert needed == (score >= threshold), (
                f"{label}: prediction_score={score} but "
                f"irrigation_needed={needed} at threshold {threshold}"
            )
            if source == "rule_fallback_out_of_range":
                # `prediction_score` is the rule's score here, not the model's,
                # so the reason quoting the model's number must be labelled as
                # such or an integrator reads it as a contradiction.
                assert "for diagnosis only" in pred["reason"], label
                assert f"prediction_score ({score})" in pred["reason"], (
                    f"{label}: the reason should point at the field it quotes"
                )
            else:
                # The reason must quote the same number the score reports, so the
                # human-readable explanation cannot disagree with the field a
                # backend acts on. `reason` prints 2dp and `prediction_score`
                # 3dp, so half a unit in the last printed place is the whole
                # expected difference.
                quoted = float(
                    re.search(r"score ([\d.]+)\s*[<>]", pred["reason"]).group(1)
                )
                assert abs(quoted - score) <= 0.005 + 1e-9, (
                    f"{label}: reason quotes {quoted} but prediction_score is {score}"
                )
            checked["water" if needed else "no water"] += 1

    # the sweep must actually exercise both outcomes, or it proves nothing
    assert checked["water"] >= 5, checked
    assert checked["no water"] >= 5, checked
    # 12 % is below the recorded soil_moisture range, so the fallback path is
    # covered here too. If the guard ever stops firing, this fails and the
    # invariant stops being tested on that branch.
    assert "rule_fallback_out_of_range" in sources, sources
    assert "model" in sources, sources


def test_safety_guards_stay_on_the_same_score_scale(irrigation_trained):
    """A guard response must also satisfy score >= threshold <=> irrigation_needed.

    The guards used to report 0.75 beside "do not water". A backend that
    re-derived the decision from the score would then have started a second pump
    while one was already running - the exact thing the guard exists to prevent.
    """
    model = IrrigationModel(model_path=IRRIGATION_MODEL_PATH)
    threshold = model.decision_threshold

    for kwargs in (
        {"pump_status": 1},
        {"water_flow": 1.5},
        {"pump_status": 1, "water_flow": 1.5},
    ):
        for crop in ("tomato", "rice", "pepper"):
            pred = model.predict(
                make_req(soil_moisture=12.0, crop_type=crop, **kwargs)
            )
            assert pred["irrigation_needed"] is False
            assert pred["history_used"] is None, "a guard means no model was consulted"
            assert pred["prediction_score"] < threshold, (
                f"guard {kwargs} on {crop} reported score "
                f"{pred['prediction_score']} >= threshold {threshold}; a backend "
                "following the documented check would start a second pump"
            )


def test_crop_thresholds_cover_the_dataset_crops():
    for crop in ("tomato", "rice", "pepper"):
        assert crop in CROP_THRESHOLDS


# ---------------------------------------------------------------- model file
def test_model_file_exists_and_is_json(irrigation_trained):
    assert IRRIGATION_MODEL_PATH.exists()
    with open(IRRIGATION_MODEL_PATH, encoding="utf-8") as handle:
        data = json.load(handle)
    assert data["features"] == FEATURE_COLUMNS
    assert len(data["coefs"]) == len(FEATURE_COLUMNS)
    assert len(data["means"]) == len(FEATURE_COLUMNS)
    assert len(data["stds"]) == len(FEATURE_COLUMNS)


def test_model_file_records_versioning_and_limitations(irrigation_trained):
    with open(IRRIGATION_MODEL_PATH, encoding="utf-8") as handle:
        data = json.load(handle)
    for key in (
        "model_version",
        "trained_at",
        "training_dataset",
        "split",
        "test_metrics",
        "limitations",
    ):
        assert key in data
    assert data["split"]["shuffled"] is False
    assert data["split"]["method"] == "chronological"


def test_a_broken_model_file_falls_back_instead_of_crashing(tmp_path):
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    model = IrrigationModel(model_path=broken)
    assert model.is_trained is False
    assert model.load_error
    pred = model.predict(make_req(soil_moisture=10.0))
    assert pred["irrigation_needed"] is True


def test_a_v0_1_0_model_file_is_rejected_rather_than_mis_scored(irrigation_trained):
    """A stale artefact must fail loudly, not answer with nonsense.

    v0.1.0 models were fitted on the four current readings. temperature/humidity/
    light are no longer model inputs, so those coefficients sit in a different
    basis and reusing them would produce a confident, meaningless score. The
    model has to refuse the file and fall back.
    """
    with open(IRRIGATION_MODEL_PATH, encoding="utf-8") as handle:
        data = json.load(handle)

    old_features = [
        "soil_moisture",
        "temperature",
        "humidity",
        "light",
        "soil_moisture_change",
        "soil_moisture_avg_30min",
        "time_since_last_irrigation_min",
    ]
    stale = {
        **data,
        "model_version": "0.1.0",
        "features": old_features,
        "coefs": data["coefs"][:4] + data["coefs"][4:],
    }
    path = IRRIGATION_MODEL_PATH.parent / "stale_v010_model.json"
    path.write_text(json.dumps(stale), encoding="utf-8")
    try:
        model = IrrigationModel(model_path=path)
        assert model.is_trained is False
        assert "Retrain" in (model.load_error or "")
        # and it must not silently return a model score for the stale basis
        pred = model.predict(make_req(soil_moisture=12.0))
        assert pred["history_used"] is None
    finally:
        path.unlink(missing_ok=True)


def test_score_features_is_monotonic_in_drying():
    """Wetter soil must never increase the irrigation score."""
    model = IrrigationModel(model_path=IRRIGATION_MODEL_PATH)
    scores = []
    for moisture in (10.0, 20.0, 30.0, 40.0):
        # Built by name against FEATURE_COLUMNS so this test cannot silently
        # drift out of step with the feature list the way a hard-coded
        # 7-element vector did when weather was added.
        values = {
            "soil_moisture": moisture,
            "soil_moisture_change": -0.5,
            "soil_moisture_avg_30min": moisture,
            "time_since_last_irrigation_min": 300.0,
            "temperature": 28.0,
            "humidity": 55.0,
            "light": 20000.0,
        }
        ordered = [values[name] for name in FEATURE_COLUMNS if name in values]
        ordered += [0.0] * (len(FEATURE_COLUMNS) - len(ordered))
        features = np.array(ordered, dtype=float)
        assert len(features) == len(FEATURE_COLUMNS)
        scores.append(model.score_features(features))
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------- preprocessing
def test_required_columns_are_validated():
    with pytest.raises(MissingColumnsError):
        clean_irrigation_frame(pd.DataFrame({"soil_moisture": [1.0]}))


def test_npk_columns_may_be_empty():
    rows = pd.DataFrame(
        {
            "timestamp": ["2026-01-01T00:00:00Z", "2026-01-01T00:10:00Z"],
            "soil_moisture": [30.0, 29.0],
            "temperature": [25.0, 25.0],
            "humidity": [60.0, 60.0],
            "light": [100.0, 100.0],
            "npk_n": [None, None],
            "irrigation_event": [0, 0],
        }
    )
    frame, _ = clean_irrigation_frame(rows)
    assert len(frame) == 2
    assert frame["npk_n"].isna().all() or frame["npk_n"].notna().all()


def test_impossible_rows_are_dropped_and_values_clipped():
    rows = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:10:00Z",
                "2026-01-01T00:20:00Z",
                "not-a-timestamp",
            ],
            "soil_moisture": [30.0, 29.0, 400.0, 30.0],
            "temperature": [25.0, 25.0, 25.0, 25.0],
            "humidity": [60.0, 60.0, 60.0, 60.0],
            "light": [100.0, 100.0, 100.0, 100.0],
            "irrigation_event": [0, 0, 0, 0],
        }
    )
    frame, report = clean_irrigation_frame(rows)
    assert report["rows_dropped"] == 1
    assert report["values_clipped"] == 1
    assert frame["soil_moisture"].max() <= 100.0


def test_frame_is_sorted_by_timestamp():
    rows = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-01T00:20:00Z",
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:10:00Z",
            ],
            "soil_moisture": [28.0, 30.0, 29.0],
            "temperature": [25.0, 25.0, 25.0],
            "humidity": [60.0, 60.0, 60.0],
            "light": [100.0, 100.0, 100.0],
            "irrigation_event": [0, 0, 0],
        }
    )
    frame, _ = clean_irrigation_frame(rows)
    assert frame["timestamp"].is_monotonic_increasing


def test_split_is_chronological_and_never_overlaps():
    frame, _ = load_irrigation_csv(IRRIGATION_CSV)
    train, test = chronological_split(frame, 0.2)
    assert len(train) + len(test) == len(frame)
    assert train["timestamp"].max() <= test["timestamp"].min()
    assert len(train) == pytest.approx(len(frame) * 0.8, abs=2)


def test_historical_features_never_look_forward():
    """Truncating the future must not change any past feature value."""
    frame, _ = load_irrigation_csv(IRRIGATION_CSV)
    full = add_historical_features(frame)
    cut = len(frame) // 2
    partial = add_historical_features(frame.iloc[:cut].copy())
    assert np.array_equal(
        full[HISTORICAL_FEATURES].iloc[:cut].round(9).to_numpy(),
        partial[HISTORICAL_FEATURES].round(9).to_numpy(),
    )


def test_soil_moisture_change_is_the_difference_to_the_previous_reading():
    rows = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=3, freq="10min"),
            "device_id": ["ESP32_01"] * 3,
            "plot_id": ["PLOT_A"] * 3,
            "soil_moisture": [30.0, 28.0, 27.5],
            "temperature": [25.0] * 3,
            "humidity": [60.0] * 3,
            "light": [100.0] * 3,
            "water_flow": [0.0, 0.0, 0.0],
            "pump_status": [0, 0, 0],
            "irrigation_event": [0, 0, 0],
        }
    )
    out = add_historical_features(rows)
    assert out["soil_moisture_change"].tolist() == [0.0, -2.0, -0.5]


def test_moving_average_uses_only_the_last_thirty_minutes():
    rows = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=6, freq="10min"),
            "device_id": ["ESP32_01"] * 6,
            "plot_id": ["PLOT_A"] * 6,
            "soil_moisture": [30.0, 30.0, 30.0, 30.0, 10.0, 10.0],
            "temperature": [25.0] * 6,
            "humidity": [60.0] * 6,
            "light": [100.0] * 6,
            "water_flow": [0.0] * 6,
            "pump_status": [0, 0, 0, 0, 0, 0],
            "irrigation_event": [0] * 6,
        }
    )
    out = add_historical_features(rows)
    # index 4 is t=+40min; the 30-minute window (t-30min, t] holds 30, 30, 10
    assert out["soil_moisture_avg_30min"].iloc[4] == pytest.approx(
        (30.0 + 30.0 + 10.0) / 3
    )
    # index 3 is t=+30min; the window holds 30, 30, 30
    assert out["soil_moisture_avg_30min"].iloc[3] == pytest.approx(30.0)
    # index 0 always falls back to the reading itself
    assert out["soil_moisture_avg_30min"].iloc[0] == pytest.approx(30.0)


def test_time_since_last_irrigation_uses_pump_status_not_the_label():
    rows = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="10min"),
            "device_id": ["ESP32_01"] * 4,
            "plot_id": ["PLOT_A"] * 4,
            "soil_moisture": [30.0] * 4,
            "temperature": [25.0] * 4,
            "humidity": [60.0] * 4,
            "light": [100.0] * 4,
            "water_flow": [0.0, 4.0, 4.0, 0.0],
            "pump_status": [0, 1, 1, 0],
            "irrigation_event": [0, 1, 0, 1],  # label must be ignored
        }
    )
    out = add_historical_features(rows)
    assert out["time_since_last_irrigation_min"].tolist() == [
        MAX_MINUTES_SINCE_IRRIGATION,
        0.0,
        0.0,
        10.0,
    ]


def test_features_are_computed_per_device():
    rows = pd.DataFrame(
        {
            "timestamp": list(pd.date_range("2026-01-01", periods=4, freq="10min")) * 2,
            "device_id": ["ESP32_01"] * 4 + ["ESP32_02"] * 4,
            "plot_id": ["PLOT_A"] * 4 + ["PLOT_B"] * 4,
            "soil_moisture": [30.0, 28.0, 26.0, 24.0] * 2,
            "temperature": [25.0] * 8,
            "humidity": [60.0] * 8,
            "light": [100.0] * 8,
            "water_flow": [0.0] * 8,
            "pump_status": [0] * 8,
            "irrigation_event": [0] * 8,
        }
    )
    out = add_historical_features(rows)
    device_one = out[out["device_id"] == "ESP32_01"]
    assert device_one["soil_moisture_change"].iloc[0] == 0.0
    assert device_one["soil_moisture_change"].iloc[1] == -2.0


def test_scaler_is_fitted_on_the_training_statistics():
    x = np.array([[1.0, 10.0], [3.0, 30.0]])
    means = x.mean(axis=0)
    stds = x.std(axis=0)
    scaled = scale_features(x, means, stds)
    assert np.allclose(scaled.mean(axis=0), 0.0, atol=1e-9)


def test_feature_matrix_has_the_declared_order():
    frame, _ = load_irrigation_csv(IRRIGATION_CSV)
    matrix = build_feature_matrix(frame.head(3))
    assert matrix.shape == (3, len(FEATURE_COLUMNS))
    assert FEATURE_COLUMNS[0] == "soil_moisture"
    assert FEATURE_COLUMNS[1 : 1 + len(HISTORICAL_FEATURES)] == HISTORICAL_FEATURES
    assert FEATURE_COLUMNS[-len(CROP_FEATURES) :] == CROP_FEATURES
    # The brief requires the four current sensor readings to be model inputs,
    # so all three weather readings must be in the feature vector, in the same
    # order as WEATHER_FEATURES.
    assert FEATURE_COLUMNS[
        len(NUMERIC_FEATURE_COLUMNS) - len(WEATHER_FEATURES) : len(NUMERIC_FEATURE_COLUMNS)
    ] == WEATHER_FEATURES
    for name in ("soil_moisture", "temperature", "humidity", "light"):
        assert name in FEATURE_COLUMNS, f"{name} must be a model input"


# ------------------------------------------- the weather inputs, and their risk
def test_the_four_current_sensor_readings_are_all_model_inputs():
    """The brief requires the current readings to reach the model, not just the API.

    They used to be excluded, on a justification that turned out to be false.
    Pinning the requirement here means a future "simplification" has to argue
    with this test instead of quietly deleting a spec item.
    """
    model = IrrigationModel(model_path=IRRIGATION_MODEL_PATH)
    for name in ("soil_moisture", "temperature", "humidity", "light"):
        assert name in model.feature_columns, f"{name} must be a model input"

    with open(IRRIGATION_MODEL_PATH, encoding="utf-8") as handle:
        stored = json.load(handle)
    assert set(stored["features"]) == set(FEATURE_COLUMNS)
    for name in ("soil_moisture", "temperature", "humidity", "light"):
        assert name in stored["features"]


def test_weather_actually_moves_the_score_in_the_expected_direction():
    """Hotter and drier air must raise the irrigation need; that is the point.

    These are the coefficients that make including the weather readings worth
    anything, so they are asserted rather than assumed. `light` is deliberately
    NOT asserted here - see
    `test_bright_light_pulls_the_score_down_and_that_is_a_documented_weakness`.

    Every reading is held inside the range recorded at training time. Outside it
    the model is not consulted at all, so a test of the model's coefficients
    would silently be testing the fallback rule instead.
    """
    model = IrrigationModel(model_path=IRRIGATION_MODEL_PATH)
    base = dict(soil_moisture=22.0, temperature=25.0, humidity=60.0, light=20000.0)

    def score(**overrides):
        result = model.predict(make_req(**{**base, **overrides}))
        assert result["decision_source"] == "model", (
            f"expected the model to answer, got {result['decision_source']}: "
            f"{result['reason']}"
        )
        return result["prediction_score"]

    assert score(temperature=35.0) > score(temperature=15.0), (
        "hotter air must raise the irrigation score"
    )
    assert score(humidity=40.0) > score(humidity=90.0), (
        "drier air must raise the irrigation score"
    )


def test_bright_light_pulls_the_score_down_and_that_is_a_documented_weakness():
    """Known, accepted, and deliberately not hidden: light has the wrong sign.

    In the synthetic data the label marks the instant a threshold scheduler
    fired, so daytime readings are correlated with "already watered". The fitted
    `light` coefficient is therefore negative - brighter means the model thinks
    less irrigation is needed, which is agronomically backwards.

    This test exists to pin the weakness so it cannot be rediscovered as a
    surprise, and so the out-of-range guard below is justified by a measurement
    rather than a hunch. If real field data ever arrives and this flips sign, the
    test fails and docs/limitations.md gets updated at the same time.
    """
    model = IrrigationModel(model_path=IRRIGATION_MODEL_PATH)
    with open(IRRIGATION_MODEL_PATH, encoding="utf-8") as handle:
        stored = json.load(handle)
    index = stored["features"].index("light")
    assert stored["coefs"][index] < 0, (
        "the light coefficient is no longer negative; the documented weakness "
        "in docs/limitations.md and the out-of-range guard need revisiting"
    )

    def score(light):
        result = model.predict(
            make_req(soil_moisture=22.0, temperature=28.0, humidity=55.0, light=light)
        )
        assert result["decision_source"] == "model", result["reason"]
        return result["prediction_score"]

    assert score(84562.0) < score(0.0), "bright light must visibly lower the score"


def test_dry_soil_is_still_recommended_at_the_worst_weather_the_api_accepts():
    """The safety invariant, now that weather can move the score.

    This is the promise that matters for a farmer: soil below the crop's trigger
    is always reported as needing water, at every combination of the four
    readings the API accepts. It is checked across the full accepted range,
    including the readings that are wildly outside anything in the training data
    - the case that motivated the out-of-range guard.
    """
    model = IrrigationModel(model_path=IRRIGATION_MODEL_PATH)
    for crop, dry in (("tomato", 16.0), ("pepper", 16.0), ("rice", 40.0)):
        for temperature in (-10.0, 0.0, 25.0, 40.0, 60.0, 80.0):
            for humidity in (0.0, 20.0, 55.0, 90.0, 100.0):
                for light in (0.0, 30000.0, 84562.0, 200000.0):
                    result = model.predict(
                        make_req(
                            soil_moisture=dry,
                            temperature=temperature,
                            humidity=humidity,
                            light=light,
                            crop_type=crop,
                        )
                    )
                    assert result["irrigation_needed"] is True, (
                        f"{crop} at {dry} % was told not to irrigate at "
                        f"{temperature} C / {humidity} % RH / {light} lux "
                        f"(source {result['decision_source']})"
                    )


def test_a_reading_outside_the_training_range_hands_the_decision_to_the_rule():
    """When the model would extrapolate, the response must say so.

    The measured failure this exists to prevent: a tomato plot at 16 % soil
    moisture - well past that crop's trigger - was told "do not irrigate" at
    -10 C / 0 % RH / 200 000 lux, because the fitted `light` coefficient is
    negative and a linear model extrapolates a trend hard off the end of its
    training range. No synthetic row is anywhere near that weather.

    The response must not paper over this: `decision_source` says the rule
    answered, and `reason` quotes the model's own score next to the threshold so
    the disagreement is visible rather than hidden.
    """
    model = IrrigationModel(model_path=IRRIGATION_MODEL_PATH)

    extrapolated = model.predict(
        make_req(soil_moisture=16.0, temperature=-10.0, humidity=0.0, light=200000.0)
    )
    assert extrapolated["decision_source"] == "rule_fallback_out_of_range"
    assert extrapolated["irrigation_needed"] is True
    assert "OUT OF TRAINING RANGE" in extrapolated["reason"]
    # The offending reading is named, with the bounds it broke, so an operator
    # can tell a bad sensor from a rare-but-real condition.
    assert "temperature=-10" in extrapolated["reason"]
    assert "light=200000" in extrapolated["reason"]
    assert "for diagnosis only" in extrapolated["reason"]

    # The same request with in-range weather is answered by the model, because
    # the model is only distrusted where it actually has to extrapolate.
    in_range = model.predict(
        make_req(soil_moisture=16.0, temperature=24.0, humidity=68.0, light=3653.0)
    )
    assert in_range["decision_source"] == "model", in_range["reason"]

    # A wet plot out of range must not be talked into being irrigated either.
    wet = model.predict(
        make_req(soil_moisture=70.0, temperature=-10.0, humidity=0.0, light=200000.0)
    )
    assert wet["decision_source"] == "rule_fallback_out_of_range"
    assert wet["irrigation_needed"] is False


def test_in_range_readings_still_reach_the_model_so_history_still_matters():
    """The guard must not quietly disable the model.

    This is the failure mode of a safety override applied too eagerly: a veto on
    every request below a moisture number makes the boundary a constant and
    throws away the history sensitivity that makes the model worth having. The
    per-feature range check is the narrower guard, so an ordinary reading with
    ordinary history must still be decided by the model - and must still change
    its mind when the history changes.
    """
    model = IrrigationModel(model_path=IRRIGATION_MODEL_PATH)

    def predict_with_history(moisture, previous, irrigated_at):
        return model.predict(
            make_req(
                soil_moisture=moisture,
                temperature=24.0,
                humidity=68.0,
                light=3653.0,
                timestamp="2026-09-27T18:30:00Z",
                history={
                    "recent_soil_moisture": [
                        {
                            "timestamp": "2026-09-27T18:20:00Z",
                            "soil_moisture": previous,
                        },
                    ],
                    "last_irrigation_at": irrigated_at,
                },
            )
        )

    # The two history shapes are chosen so every derived feature lands inside the
    # recorded training ranges: a 7-point drop is within the observed -8.61..9.88
    # spread, while the 9-point drop in the obvious-looking version of this pair
    # is not, and would hand the request to the fallback rule.
    watered = predict_with_history(26.0, 33.0, "2026-09-27T18:00:00Z")
    drying = predict_with_history(26.0, 30.0, "2026-09-25T18:00:00Z")

    assert watered["decision_source"] == "model", watered["reason"]
    assert drying["decision_source"] == "model", drying["reason"]
    assert watered["irrigation_needed"] is False, (
        "recently watered soil at 26 % should not be irrigated"
    )
    assert drying["irrigation_needed"] is True, (
        "soil drying fast at 26 % should be irrigated; if the guard is "
        "over-triggering, the history sensitivity is being flattened"
    )


def test_out_of_range_ignores_the_placeholders_the_service_substituted():
    """Only real readings may trip the guard.

    Without history the service fills `soil_moisture_avg_30min` with the current
    reading. If that substituted value were range-checked, a dry-but-valid
    reading would be reported as out of range on the strength of a number the
    client never sent, and the model would be discarded for the wrong reason.
    """
    model = IrrigationModel(model_path=IRRIGATION_MODEL_PATH)

    # 16 % is inside the recorded soil_moisture range (13.76..77.31) but below
    # the recorded avg_30min range (16.88..75.01). The placeholder therefore
    # differs from the real reading's range membership, which is the whole test.
    low_moisture = model.predict(
        make_req(soil_moisture=16.0, temperature=24.0, humidity=68.0, light=3653.0)
    )
    assert low_moisture["decision_source"] == "model", low_moisture["reason"]

    # The real soil_moisture reading itself being out of range still counts.
    genuinely_out = model.predict(
        make_req(soil_moisture=5.0, temperature=24.0, humidity=68.0, light=3653.0)
    )
    assert genuinely_out["decision_source"] == "rule_fallback_out_of_range"
    assert "soil_moisture=5" in genuinely_out["reason"]


def test_an_unknown_crop_is_flagged_even_when_the_model_answers():
    """A crop the service does not recognise must not reach the model.

    The all-zero crop block is what the trained crops encode, so scoring an
    unrecognised name would read the model as saying something confident about
    a crop it was never fitted on. The request is answered by the rule instead,
    and both the reason and `decision_source` say the model was not consulted.
    """
    model = IrrigationModel(model_path=IRRIGATION_MODEL_PATH)
    result = model.predict(
        make_req(soil_moisture=30.0, crop_type="cassava")
    )
    assert "is not a crop this service recognises" in result["reason"]
    assert result["decision_source"] == "rule_fallback", result["reason"]
    assert "the model was not consulted" in result["reason"]


def test_guards_and_fallback_report_their_own_decision_source():
    """Every response names what produced it, so a caller never has to guess."""
    model = IrrigationModel(model_path=IRRIGATION_MODEL_PATH)

    assert model.predict(
        make_req(soil_moisture=8.0, pump_status=1)
    )["decision_source"] == "pump_guard"

    assert model.predict(
        make_req(soil_moisture=8.0, water_flow=1.5)
    )["decision_source"] == "water_guard"

    fallback = rule_based_recommendation(make_req(soil_moisture=8.0))
    assert fallback["decision_source"] == "rule_fallback"

    out_of_range = model.predict(
        make_req(soil_moisture=20.0, temperature=-10.0, humidity=0.0, light=200000.0)
    )
    assert out_of_range["decision_source"] == "rule_fallback_out_of_range", (
        out_of_range["reason"]
    )

    # the full documented set, so a removed or renamed source shows up here
    for source in (
        "pump_guard", "water_guard", "rule_fallback", "model",
        "rule_fallback_out_of_range",
    ):
        assert isinstance(source, str) and source


def test_soil_below_the_crop_trigger_is_always_recommended():
    """Dry soil must never be waved through, whatever the weather says."""
    model = IrrigationModel(model_path=IRRIGATION_MODEL_PATH)
    for crop, dry in (("tomato", 16.0), ("pepper", 16.0), ("rice", 40.0)):
        for light in (0.0, 68000.0):
            result = model.predict(
                make_req(
                    soil_moisture=dry,
                    temperature=33.0,
                    humidity=40.0,
                    light=light,
                    crop_type=crop,
                )
            )
            assert result["irrigation_needed"] is True, (
                f"{crop} at {dry} % with light {light} was waved through"
            )


def test_no_history_falls_back_to_neutral_values_not_the_training_median():
    """A history-less request must not borrow an unconditional average.

    Filling soil_moisture_avg_30min with the training median (~33 %) made a dry
    reading look like it had recently been watered, which pushed dry soil
    towards "do not irrigate". The neutral fallback is the current reading.
    """
    features, notes = build_features_from_request(
        make_req(soil_moisture=20.0, crop_type="tomato")
    )
    index = FEATURE_COLUMNS.index("soil_moisture_avg_30min")
    assert features[index] == pytest.approx(20.0)
    assert not np.isnan(features).any(), "no-history features must not contain NaN"
    assert any("placeholder" in note for note in notes)

    change_index = FEATURE_COLUMNS.index("soil_moisture_change")
    assert features[change_index] == pytest.approx(0.0)
    since_index = FEATURE_COLUMNS.index("time_since_last_irrigation_min")
    assert features[since_index] == pytest.approx(
        MAX_MINUTES_SINCE_IRRIGATION
    )


# ------------------------------------------------------- target-leak guard rails
def test_irrigation_event_rows_do_not_leak_through_the_pump_state():
    """The label row must not be the row where the pump is already running.

    The first version of the simulator reported pump_status=1 / water_flow>0 on
    the same row as irrigation_event=1. That made
    time_since_last_irrigation_min == 0 a perfect giveaway for the positive
    class and inflated the measured scores without any real skill.
    """
    frame, _ = load_irrigation_csv(IRRIGATION_CSV)
    positives = frame[frame["irrigation_event"] == 1]
    assert len(positives) > 0
    assert (positives["pump_status"] == 1).sum() == 0
    assert (positives["water_flow"] > 0).sum() == 0
    assert (positives["time_since_last_irrigation_min"] == 0).sum() == 0


def test_time_since_irrigation_alone_cannot_identify_the_label():
    """The feature must overlap between the classes, not separate them."""
    frame, _ = load_irrigation_csv(IRRIGATION_CSV)
    at_zero = {
        int(label): float((block["time_since_last_irrigation_min"] == 0).mean())
        for label, block in frame.groupby("irrigation_event")
    }
    # A leaked feature would give 1.0 for the positive class and ~0.0 for the
    # negative class; a genuine one leaves both classes with a mix.
    assert at_zero[1] < 0.5
    assert at_zero[0] > 0.0


# ------------------------------------------------------------- crop-context fix
def test_crop_indicators_are_one_hot_over_the_trained_crops():
    assert crop_indicators("tomato") == [1.0, 0.0, 0.0]
    assert crop_indicators("rice") == [0.0, 1.0, 0.0]
    assert crop_indicators("pepper") == [0.0, 0.0, 1.0]
    assert crop_indicators("Tomato") == [1.0, 0.0, 0.0]  # case-insensitive
    # An unknown crop invents nothing instead of guessing a trained crop.
    assert crop_indicators("wheat") == [0.0, 0.0, 0.0]
    assert crop_indicators(None) == [0.0, 0.0, 0.0]


def test_training_and_inference_agree_on_the_crop_block():
    """The request builder must place the crop indicators the same way training does.

    If these two ever disagree the model silently scores a live request with
    the wrong numbers, which is the failure mode the shared feature order is
    meant to prevent.
    """
    frame, _ = load_irrigation_csv(IRRIGATION_CSV)
    row = frame.iloc[0]
    from_frame = build_feature_matrix(frame.head(1))[0]

    request = IrrigationRequest(
        timestamp=str(row["timestamp"]),
        device_id=str(row["device_id"]),
        plot_id=str(row["plot_id"]),
        soil_moisture=float(row["soil_moisture"]),
        temperature=float(row["temperature"]),
        humidity=float(row["humidity"]),
        light=float(row["light"]),
        crop_type=str(row["crop_type"]),
    )
    from_request, _notes = build_features_from_request(request)

    assert len(from_request) == len(FEATURE_COLUMNS)
    start = FEATURE_COLUMNS.index(CROP_FEATURES[0])
    assert list(from_request[start:]) == crop_indicators(row["crop_type"])
    # The crop block is the only thing the request path can differ on here.
    assert np.isclose(from_frame[0], float(row["soil_moisture"]))


def test_rice_is_watered_at_a_much_higher_moisture_than_tomato():
    """Regression test: without crop context the two clusters are inseparable.

    Rice is triggered near 60 % while tomato triggers near 21 %, so a single
    moisture cut-off cannot serve both. The model must rely on the crop
    indicators to tell the two situations apart.
    """
    model = IrrigationModel(model_path=IRRIGATION_MODEL_PATH)
    assert model.is_trained

    common = dict(
        timestamp="2026-09-27T12:00:00Z",
        device_id="ESP32_02",
        plot_id="PLOT_B",
        temperature=30.0,
        humidity=50.0,
        light=50000.0,
    )
    rice = model.predict(
        IrrigationRequest(soil_moisture=45.0, crop_type="rice", **common)
    )
    tomato = model.predict(
        IrrigationRequest(soil_moisture=45.0, crop_type="tomato", **common)
    )
    assert rice["irrigation_needed"] is True
    assert tomato["irrigation_needed"] is False


def test_unseen_crop_is_flagged_as_extrapolation():
    """A known crop with no model column is named as such, not extrapolated.

    `wheat` has a rule-based threshold but no indicator column, so it is a
    legitimate crop that the model was simply not trained on. It takes the rule
    path and the reason says which trained crops the model does cover.
    """
    model = IrrigationModel(model_path=IRRIGATION_MODEL_PATH)
    result = model.predict(
        make_req(
            soil_moisture=30.0,
            crop_type="wheat",
        )
    )
    assert "has no indicator column in the trained model" in result["reason"]
    assert result["decision_source"] == "rule_fallback", result["reason"]


# ------------------------------------------------------------ decision threshold
def test_stored_decision_threshold_is_used_instead_of_a_hardcoded_half():
    model = IrrigationModel(model_path=IRRIGATION_MODEL_PATH)
    stored = json.loads(IRRIGATION_MODEL_PATH.read_text(encoding="utf-8"))
    assert model.decision_threshold == stored["decision_threshold"]
    assert 0.0 < model.decision_threshold < 1.0

    result = model.predict(make_req(soil_moisture=22.0, crop_type="tomato"))
    # Whichever side of the threshold this request lands on, the reason quotes
    # the stored threshold instead of a hardcoded 0.50.
    assert f"{model.decision_threshold:.2f}" in result["reason"]
    assert "0.50;" not in result["reason"]


def test_old_model_file_without_a_threshold_defaults_to_one_half(tmp_path):
    payload = json.loads(IRRIGATION_MODEL_PATH.read_text(encoding="utf-8"))
    payload.pop("decision_threshold", None)
    legacy = tmp_path / "legacy_model.json"
    legacy.write_text(json.dumps(payload), encoding="utf-8")

    model = IrrigationModel(model_path=legacy)
    assert model.is_trained
    assert model.decision_threshold == 0.5


# ------------------------------------------------- case / whitespace invariance
#: Spellings of one crop that must all resolve identically. The threshold table
#: and the one-hot encoder are both case-insensitive, so case cannot change an
#: answer; before _normalise_crop was applied at the rule lookup, "maize" used
#: its 27% reference while "Maize" missed the table and took the 25% default,
#: which inverted the decision across the whole 25-27% band.
CROP_SPELLINGS = ("maize", "Maize", "MAIZE", "  maize  ")


@pytest.mark.parametrize("crop", CROP_SPELLINGS)
def test_rule_fallback_ignores_crop_case_and_whitespace(crop):
    # 26% sits between the 25% default reference and maize's 27% reference,
    # so a lookup that misses the table and one that hits it disagree here.
    result = rule_based_recommendation(make_req(crop_type=crop, soil_moisture=26.0))
    baseline = rule_based_recommendation(
        make_req(crop_type="maize", soil_moisture=26.0)
    )
    assert result["irrigation_needed"] is baseline["irrigation_needed"] is True
    assert result["reason"] == baseline["reason"]
    # The reported reason must name the crop the threshold was actually found
    # under, not the raw spelling, or the message contradicts the threshold.
    assert "27 reference threshold for maize" in result["reason"]
    assert "maize" in result["reason"]


@pytest.mark.parametrize("crop", CROP_SPELLINGS)
def test_model_entry_point_ignores_crop_case_and_whitespace(crop):
    # Same invariance through the public predict() path, where the crop guard
    # decides between the model and the rule.
    model = IrrigationModel(model_path=IRRIGATION_MODEL_PATH)
    spelled = model.predict(make_req(crop_type=crop, soil_moisture=26.0))
    canonical = model.predict(make_req(crop_type="maize", soil_moisture=26.0))
    assert spelled["irrigation_needed"] is canonical["irrigation_needed"]
    assert spelled["decision_source"] == canonical["decision_source"] == "rule_fallback"
    assert spelled["prediction_score"] == canonical["prediction_score"]


@pytest.mark.parametrize("crop", ("wheAt", "WHEAT", " wheat "))
def test_model_agrees_across_case_for_every_rule_only_crop(crop):
    # The same defect would have hit any rule-only crop whose threshold differs
    # from the 25% default, so cover the whole set rather than maize alone.
    model = IrrigationModel(model_path=IRRIGATION_MODEL_PATH)
    threshold = CROP_THRESHOLDS["wheat"]
    # straddle the crop's own threshold
    below = model.predict(make_req(crop_type=crop, soil_moisture=threshold - 0.5))
    canonical_below = model.predict(
        make_req(crop_type="wheat", soil_moisture=threshold - 0.5)
    )
    assert below["irrigation_needed"] is canonical_below["irrigation_needed"] is True
    above = model.predict(make_req(crop_type=crop, soil_moisture=threshold + 0.5))
    canonical_above = model.predict(
        make_req(crop_type="wheat", soil_moisture=threshold + 0.5)
    )
    assert above["irrigation_needed"] is canonical_above["irrigation_needed"] is False


# ------------------------------------------- shipped model's trained crop vocabulary
def test_trained_crops_match_the_shipped_models_features():
    """Pin TRAINED_CROPS to the frozen model actually being served.

    TRAINED_CROPS is derived from preprocess.CROP_FEATURES rather than read from
    the model file, so the two can drift apart: a retrain that changed the crop
    columns would leave the guard routing crops to the model that no longer has
    an indicator for them, or diverting crops the model does handle. The
    backend pins its own copy of the vocabulary to this model in
    Agriguard_Backend/test_crops.py; this is the AI-side half of that check.
    """
    model = IrrigationModel(model_path=IRRIGATION_MODEL_PATH)
    assert model.is_trained, "the shipped irrigation model must load"

    # Read the vocabulary out of the artifact rather than restating it, so the
    # test tracks the model instead of a copy of the model.
    shipped_features = json.loads(IRRIGATION_MODEL_PATH.read_text(encoding="utf-8"))["features"]
    prefix = "crop_is_"
    shipped_crop_features = sorted(
        f[len(prefix):] for f in shipped_features if f.startswith(prefix)
    )

    assert shipped_crop_features == ["pepper", "rice", "tomato"]
    # 1. the model file -> TRAINED_CROPS, the guard's routing vocabulary
    assert sorted(TRAINED_CROPS) == shipped_crop_features
    # 2. the feature definition -> the model file, so a retrain cannot add or
    #    drop a crop column without this failing
    assert sorted(f for f in CROP_FEATURES if f.startswith(prefix)) == sorted(
        f for f in shipped_features if f.startswith(prefix)
    )
    # 3. the preprocessing indicators -> TRAINED_CROPS, so every crop the guard
    #    lets reach the model is actually one-hot encoded to a non-zero block
    assert {c for c in TRAINED_CROPS if any(crop_indicators(c))} == set(TRAINED_CROPS)
    assert not any(crop_indicators("default"))

