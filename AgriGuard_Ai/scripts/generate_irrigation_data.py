"""Generate the SYNTHETIC prototype irrigation dataset.

    python scripts/generate_irrigation_data.py --days 30 --period-min 10

    ############################################################################
    # THIS DATASET IS SYNTHETIC PROTOTYPE DATA GENERATED FOR DEVELOPMENT AND  #
    # TESTING. IT IS NOT REAL AGRICULTURAL FIELD DATA.                          #
    # Model performance on it must not be interpreted as real-world field       #
    # accuracy. See docs/limitations.md.                                        #
    ############################################################################

Why it is sequential
--------------------
The previous version drew one independent row at a time, which made the
historical features meaningless. This version simulates plots minute by minute:

* soil moisture dries at a rate driven by temperature, humidity and light
  (evaporative demand) plus an unobservable "wind" term,
* a controller switches the pump on when the *true* moisture falls below the
  crop threshold, and the pump raises moisture while it runs,
* the CSV records the *noisy sensor* reading, not the true value,
* ``irrigation_event`` is 1 when an irrigation action started in that interval.

Because the label depends on the hidden true value and on wind, no model can
separate the classes perfectly from the recorded readings. That residual error
is real and the evaluation reports it honestly.
"""

import argparse
import math
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agriguard.config import IRRIGATION_CSV  # noqa: E402
from agriguard.irrigation import CROP_THRESHOLDS  # noqa: E402

START = datetime(2026, 3, 1, 0, 0, 0)

#: One plot per crop. A plot grows one crop, so the crop is constant per plot.
PLOTS = [
    {"device_id": "ESP32_01", "plot_id": "PLOT_A", "crop_type": "tomato", "start": 34.0},
    {"device_id": "ESP32_02", "plot_id": "PLOT_B", "crop_type": "rice", "start": 72.0},
    {"device_id": "ESP32_03", "plot_id": "PLOT_C", "crop_type": "pepper", "start": 30.0},
]

SENSOR_NOISE_SD = 1.1        # soil-moisture sensor noise, % points
IRRIGATION_GAIN = 6.0        # moisture added per 10-minute step while pumping
MAX_PUMP_STEPS = 3           # a single irrigation event lasts at most 30 minutes
DRYING_BASE = 0.70           # baseline moisture loss per step
DRYING_EVAP = 4.00           # extra loss scaled by evaporative demand


def day_curve(hour, minute):
    """Diurnal temperature / humidity / light for a temperate site."""
    phase = (hour + minute / 60.0) / 24.0 * 2.0 * math.pi
    temperature = 24.0 + 9.0 * math.sin(phase - math.pi / 2)
    humidity = 68.0 - 22.0 * math.sin(phase - math.pi / 2)
    daylight = max(0.0, math.sin(phase - math.pi / 2))
    light = 78000.0 * (daylight ** 1.4)
    return temperature, humidity, light


def simulate_plot(plot, steps, period_min, rng):
    threshold = CROP_THRESHOLDS[plot["crop_type"]]
    rows = []
    moisture = plot["start"]
    wind = rng.uniform(0.6, 1.0)          # hidden drying factor
    pump_steps_left = 0
    pump_start_pending = False
    npk_base = [rng.uniform(15.0, 95.0), rng.uniform(8.0, 60.0), rng.uniform(18.0, 88.0)]

    for step in range(steps):
        stamp = START + timedelta(minutes=step * period_min)
        temperature, humidity, light = day_curve(stamp.hour, stamp.minute)

        # ---- hardware state observed during THIS interval --------------------
        # The pump was started by the controller on some *earlier* step, so this
        # is what the ESP32 can actually have reported at this timestamp.
        pump_active_before = pump_steps_left > 0

        # ---- controller decides on the hidden true value --------------------
        if pump_steps_left == 0 and moisture <= threshold:
            pump_steps_left = MAX_PUMP_STEPS
            pump_start_pending = True

        irrigated_now = pump_start_pending
        pump_start_pending = False

        # ---- physics -------------------------------------------------------
        demand = (
            max(0.05, (temperature - 8.0) / 26.0)
            * max(0.25, (95.0 - humidity) / 65.0)
            * max(0.15, light / 52000.0)
        )
        drying = DRYING_BASE + DRYING_EVAP * demand * wind
        moisture -= drying

        if pump_active_before:
            moisture += IRRIGATION_GAIN
            pump_steps_left -= 1
        moisture = min(100.0, max(2.0, moisture))

        # ---- what the ESP32 actually reports -------------------------------
        # NOTE: ``pump_status`` / ``water_flow`` describe what the hardware did
        # during this interval, while ``irrigation_event`` records the decision
        # that starts the run. Reporting the pump on the same row as the event
        # would make ``time_since_last_irrigation_min == 0`` a perfect giveaway
        # for the label - blatant target leakage - and the trained model would
        # learn nothing transferable.
        recorded = moisture + rng.gauss(0.0, SENSOR_NOISE_SD)
        recorded = round(min(100.0, max(0.0, recorded)), 2)
        water_flow = round(rng.uniform(2.4, 8.5), 2) if pump_active_before else 0.0

        rows.append(
            {
                "timestamp": stamp.isoformat() + "Z",
                "device_id": plot["device_id"],
                "plot_id": plot["plot_id"],
                "soil_moisture": recorded,
                "temperature": round(temperature + rng.gauss(0.0, 0.8), 2),
                "humidity": round(min(100.0, max(5.0, humidity + rng.gauss(0.0, 2.4))), 2),
                "light": round(max(0.0, light + rng.gauss(0.0, 2500.0)), 1),
                # NPK is optional hardware: a third of the plots have no sensor.
                "npk_n": None,
                "npk_p": None,
                "npk_k": None,
                "water_flow": water_flow,
                "pump_status": 1 if pump_active_before else 0,
                "irrigation_event": 1 if irrigated_now else 0,
                "crop_type": plot["crop_type"],
            }
        )

    if plot["device_id"] == "ESP32_01":  # only this plot carries an NPK probe
        for row in rows:
            row["npk_n"] = round(max(0.0, npk_base[0] + rng.gauss(0.0, 3.0)), 1)
            row["npk_p"] = round(max(0.0, npk_base[1] + rng.gauss(0.0, 2.0)), 1)
            row["npk_k"] = round(max(0.0, npk_base[2] + rng.gauss(0.0, 2.5)), 1)

    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--period-min", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=IRRIGATION_CSV)
    args = parser.parse_args()

    steps = int(args.days * 24 * 60 / args.period_min)
    rng = random.Random(args.seed)

    rows = []
    for plot in PLOTS:
        rows.extend(simulate_plot(plot, steps, args.period_min, rng))

    frame = pd.DataFrame(rows)
    frame = frame.sort_values(["timestamp", "device_id"]).reset_index(drop=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)

    print("SYNTHETIC prototype data - not real field measurements.")
    print(f"wrote {len(frame)} rows -> {args.out}")
    print(f"plots: {len(PLOTS)}  days: {args.days}  period: {args.period_min} min")
    print(f"irrigation_event positives: {int(frame['irrigation_event'].sum())} "
          f"({frame['irrigation_event'].mean():.1%})")
    print(f"rows with NPK present: {int(frame['npk_n'].notna().sum())}")
    print(frame.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
