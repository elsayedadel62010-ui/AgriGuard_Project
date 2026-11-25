"""Print the moisture boundary the fitted irrigation model actually uses.

The README and docs quote a per-crop "irrigate below this moisture" number.
This script computes it from the shipped model file so the number in the
documentation can be regenerated instead of being trusted.

Two things make the boundary a range rather than a constant, and both are swept
here:

- the three historical features, under three fixed, explicit contexts;
- the current weather readings, which the brief requires the model to consume.

Most importantly, this script asks the *service* what it would answer, via
``IrrigationModel.predict``, rather than re-deriving the decision from the
coefficients. That is deliberate: the served decision can hand a request to the
crop threshold rule when a reading falls outside the recorded training ranges, so
a boundary computed from the raw score crossing alone would quote a number the
API does not actually use.

Each history context is checked against the features the service really computes
before the sweep runs, so the labels below cannot drift away from the requests
that back them.

    python scripts/decision_boundaries.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agriguard.config import IRRIGATION_CSV  # noqa: E402
from agriguard.contracts import IrrigationRequest  # noqa: E402
from agriguard.irrigation import IrrigationModel  # noqa: E402
from agriguard.preprocess import TARGET_COLUMN, load_irrigation_csv  # noqa: E402

#: The current weather reading held fixed while the boundary is swept. These are
#: the training-set medians, so the headline boundary is the one that applies
#: under typical conditions. It moves with temperature/humidity/light, which is
#: exactly what the brief asks for.
WEATHER = (24.0, 68.0, 3653.0)

#: The moment the boundary is evaluated at. Any fixed instant works; it only has to
#: be consistent so `time_since_last_irrigation_min` is reproducible.
NOW = "2026-09-27T18:30:00Z"

#: (label, soil_moisture_avg_30min or None for "no history",
#:  soil_moisture_change, time_since_last_irrigation_min)
CONTEXTS: Dict[str, Tuple[Optional[float], float, float]] = {
    "no history (avg30=current, change=0, tsl=1440)": (None, 0.0, 1440.0),
    "recently watered (avg30=35, change=-2, tsl=60)": (35.0, -2.0, 60.0),
    "drying fast (avg30=30, change=-6, tsl=1440)": (30.0, -6.0, 1440.0),
}

CROPS = ("tomato", "rice", "pepper")


#: The three historical features are not free parameters. The service includes the
#: *current* reading in the 30-minute window and measures the trend from the newest
#: prior reading, so for prior readings ``a`` (older) and ``b`` (newer):
#:
#:     avg30 = (a + b + moisture) / 3
#:     change = moisture - b
#:
#: Solving pins the context while the current moisture is swept underneath it:
#: ``b = moisture - change`` and ``a = 3*avg30 - 2*moisture + change``. Both prior
#: readings are therefore functions of the reading being swept.
def _history_for(moisture: float, avg30: float, change: float) -> list:
    previous = round(moisture - change, 4)
    older = round(3.0 * avg30 - 2.0 * moisture + change, 4)
    return [
        {"timestamp": "2026-09-27T18:20:00Z", "soil_moisture": older},
        {"timestamp": "2026-09-27T18:25:00Z", "soil_moisture": previous},
    ]


def feasible_window(avg30: float, change: float) -> Tuple[float, float]:
    """Moisture range over which the pinned context is physically representable.

    Both synthesised prior readings are derived from the current reading, so
    holding `avg30` fixed means the earlier readings are only sensible near it.
    Sweeping all the way to 100 % would require the history to claim the soil sat
    at -97 % five minutes earlier, which the request contract rejects and which no
    client would send. Rather than report a boundary measured against an absurd
    history, the sweep is confined to where the context holds together and the
    limit is printed.
    """
    previous_lo, previous_hi = change, 100.0 + change
    older_lo = (3.0 * avg30 + change - 100.0) / 2.0
    older_hi = (3.0 * avg30 + change) / 2.0
    lo = max(0.0, previous_lo, older_lo)
    hi = min(100.0, previous_hi, older_hi)
    return lo, hi


def make_asker(model: IrrigationModel, crop: str, avg30: Optional[float],
               change: float, tsl: float, weather=WEATHER):
    """Return ``moisture -> the response the service would give``.

    The context is verified against the features the service actually computes
    before the sweep starts, so a label in ``CONTEXTS`` cannot drift away from the
    request that backs it.
    """
    temperature, humidity, light = weather
    irrigated_at = (
        datetime.fromisoformat(NOW.replace("Z", "+00:00")) - timedelta(minutes=tsl)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")

    def build(moisture: float) -> IrrigationRequest:
        payload = dict(
            soil_moisture=moisture,
            temperature=temperature,
            humidity=humidity,
            light=light,
            pump_status=0,
            crop_type=crop,
        )
        if avg30 is not None:
            payload["timestamp"] = NOW
            payload["history"] = {
                "recent_soil_moisture": _history_for(moisture, avg30, change),
                "last_irrigation_at": irrigated_at,
            }
        return IrrigationRequest(**payload)

    # Prove the label at a probe point on the sweep before trusting it anywhere.
    # This goes through the service's own extractor on purpose: it is the one
    # place that reads `req.history` and turns it into the three historical
    # features. Re-deriving the same thing here would be a second implementation
    # to drift, which is the failure this check exists to catch.
    if avg30 is not None:
        realized, probe_notes, _ = model._request_features(build(avg30))
        for index, (name, target) in enumerate(
            (("change", change), ("avg30", avg30), ("tsl", tsl)), start=1
        ):
            if abs(realized[index] - target) > 0.05:
                raise AssertionError(
                    f"context for {crop!r} asks for {name}={target:g} but the "
                    f"service computed {realized[index]:g} (notes: {probe_notes})"
                )

    def ask(moisture: float) -> dict:
        return model.predict(build(moisture))

    return ask


def boundary(ask, threshold: float, window: Tuple[float, float] = (0.0, 100.0)
             ) -> Tuple[Optional[float], Optional[float]]:
    """Wettest moisture in ``window`` on a 0.1 grid that the service still recommends.

    Returns ``(served_boundary, model_only_boundary)``. The first is what the API
    actually does, the out-of-range guard included. The second is the last point at
    which the model itself still recommends, which is lower whenever the rule took
    over before the model did.

    The window is the range over which the requested history context is
    representable - see `feasible_window`. A boundary of `None` therefore means
    "never recommends anywhere in the window", not "no boundary exists".
    """
    lo, hi = window
    served = None
    model_only = None
    top = int(round(hi * 10))
    for step in range(top, -1, -1):
        moisture = step / 10.0
        if moisture < lo - 1e-9:
            break
        result = ask(moisture)
        if model_only is None and result["decision_source"] == "model" \
                and result["prediction_score"] >= threshold:
            model_only = moisture
        if served is None and result["irrigation_needed"]:
            served = moisture
        if served is not None and model_only is not None:
            break
    return served, model_only


def is_truncated(served: Optional[float], window: Tuple[float, float]) -> bool:
    """True when the sweep ran out of window before the trigger did.

    A boundary equal to the top of the feasible window is not a measurement - it
    is the search giving up. Rice wants to stay wet under a recently-watered
    context, so its trigger sits above what the pinned history can represent.
    Quoting the window ceiling as "rice irrigates up to 51.5 %" would be a number
    derived from the search bounds rather than from the model.
    """
    return served is not None and served >= window[1] - 1e-9


def main() -> int:
    model = IrrigationModel()
    if not model.is_trained:
        print(f"model not loaded: {model.load_error}")
        return 1
    threshold = model.decision_threshold

    print("=" * 78)
    print("IRRIGATION DECISION BOUNDARIES (measured by asking the service)")
    print("=" * 78)
    print(f"model file        : {model.model_path.name} (v{model.model_version})")
    print(f"decision threshold: {threshold}")
    print(f"features          : {model.feature_columns}")
    print()
    print(f"weather held at {WEATHER[0]:g} C / {WEATHER[1]:g} % RH / "
          f"{WEATHER[2]:g} lux unless stated otherwise")
    print()

    for label, (avg30, change, tsl) in CONTEXTS.items():
        window = (0.0, 100.0) if avg30 is None else feasible_window(avg30, change)
        print(f"history context: {label}")
        if avg30 is not None:
            print(f"    (swept over {window[0]:.1f}..{window[1]:.1f} %, the range where "
                  f"that context is representable)")
        for crop in CROPS:
            ask = make_asker(model, crop, avg30, change, tsl)
            served, model_only = boundary(ask, threshold, window)
            if served is None:
                note = "  [never recommends in this window]"
            elif is_truncated(served, window):
                note = f"  [>= {served:.1f} % - window-limited, not a measured boundary]"
            elif served != model_only:
                note = "  [out-of-range rule took over above this]"
            else:
                note = ""
            shown = f"{served:.1f}" if served is not None else "n/a"
            print(f"    {crop:7} irrigate up to {shown} % soil moisture{note}")
        print()

    frame, _ = load_irrigation_csv(IRRIGATION_CSV)
    events = frame[frame[TARGET_COLUMN] == 1]
    print("where events actually happened in the training data (median soil moisture)")
    for crop, group in events.groupby("crop_type"):
        print(f"    {crop:7} n={len(group):<5} median "
              f"{float(group['soil_moisture'].median()):.1f} %")
    print()
    print("The 'recently watered' row is the one to compare against the medians.")

    print()
    print("WEATHER SENSITIVITY (weather is a model input, so it moves the boundary)")
    print("=" * 78)
    print()
    # The first three rows stay inside the recorded training ranges on purpose:
    # outside them the model is not consulted at all, so a row out there would
    # print the fallback rule's threshold and look like model sensitivity.
    # The last row is included precisely to show what out of range looks like.
    header = f"{'condition':40}" + "".join(f"{c:>9}" for c in CROPS)
    print(header)
    conditions = [
        ("cool damp dark  13C 97RH 0lux (in range)", (13.0, 97.0, 0.0)),
        ("mild          24C 68RH 3.6klux (in range)", WEATHER),
        ("hot dry bright  35C 39RH 85klux (in range)", (35.0, 39.0, 84562.0)),
        ("beyond range -10C 0RH 200klux (rule answers)", (-10.0, 0.0, 200000.0)),
    ]
    for label, weather in conditions:
        cells = []
        for crop in CROPS:
            ask = make_asker(model, crop, 35.0, -2.0, 60.0, weather=weather)
            window = feasible_window(35.0, -2.0)
            served, _ = boundary(ask, threshold, window)
            if served is None:
                cells.append("-")
            elif is_truncated(served, window):
                cells.append(f">{served:.0f}")
            else:
                cells.append(f"{served:.1f}")
        print(f"{label:40}" + "".join(f"{c:>9}" for c in cells))
    print()
    print("'>N' means the trigger is above the representable window for the pinned")
    print("history context, so no boundary was measured for that cell.")
    print()
    print("Read this as: the trigger is a range, not a constant. Quote the crop, the")
    print("history context AND the weather, or quote none of them.")
    print()
    print("Per-feature training ranges - the out-of-range guard:")
    for name, bounds in model.feature_ranges.items():
        print(f"    {name:32} {bounds[0]:>12.4g} .. {bounds[1]:>12.4g}")
    print()
    print("A request with any reading outside these bounds does not get the model's")
    print("answer; the crop threshold rule decides and decision_source says so.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
