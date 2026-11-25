"""Generate examples/irrigation_prediction.json and examples/crop_health_prediction.json.

    python scripts/make_examples.py

Every value in the output is a real response from the real models, captured by
calling the same code paths the HTTP API uses. Nothing is typed by hand and no
accuracy number is invented.
"""

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agriguard.config import CROP_HEALTH_DIR, EXAMPLES_DIR  # noqa: E402
from agriguard.contracts import IrrigationRequest  # noqa: E402
from agriguard.crophealth import crop_health_model  # noqa: E402
from agriguard.irrigation import irrigation_model  # noqa: E402
from agriguard.version import __version__  # noqa: E402

IRRIGATION_EXAMPLES = [
    {
        "name": "dry_tomato_plot_afternoon",
        "description": (
            "Mid-afternoon, hot and sunny, soil moisture well below the reference "
            "threshold. Expect a recommendation. All four readings sit inside the "
            "range the model was trained on."
        ),
        "request": {
            "timestamp": "2026-09-27T14:20:00Z",
            "device_id": "ESP32_01",
            "plot_id": "PLOT_A",
            "soil_moisture": 20.0,
            "temperature": 33.0,
            "humidity": 42.0,
            "light": 68000.0,
            "npk_n": 42.0,
            "npk_p": 27.0,
            "npk_k": 51.0,
            "water_flow": 0.0,
            "pump_status": 0,
            "crop_type": "tomato",
        },
    },
    {
        "name": "dry_rice_plot_afternoon",
        "description": (
            "Rice is kept wetter than tomato, so 45 % soil moisture means "
            "'irrigate' here while the same reading on the tomato plot above "
            "means 'do not'. This is why the model takes crop_type as an input."
        ),
        "request": {
            "timestamp": "2026-09-27T14:20:00Z",
            "device_id": "ESP32_02",
            "plot_id": "PLOT_B",
            "soil_moisture": 45.0,
            "temperature": 33.0,
            "humidity": 42.0,
            "light": 68000.0,
            "npk_n": None,
            "npk_p": None,
            "npk_k": None,
            "water_flow": 0.0,
            "pump_status": 0,
            "crop_type": "rice",
        },
    },
    {
        "name": "wet_plot_after_rain",
        "description": "Soil is wet and it is cool and overcast. Expect no action.",
        "request": {
            "timestamp": "2026-09-27T09:05:00Z",
            "device_id": "ESP32_01",
            "plot_id": "PLOT_A",
            "soil_moisture": 68.0,
            "temperature": 19.0,
            "humidity": 88.0,
            "light": 12000.0,
            "npk_n": None,
            "npk_p": None,
            "npk_k": None,
            "water_flow": 0.0,
            "pump_status": 0,
            "crop_type": "tomato",
        },
    },
    {
        "name": "pump_already_running",
        "description": (
            "The soil is dry but the pump is on. The model must not ask for a "
            "second irrigation."
        ),
        "request": {
            "timestamp": "2026-09-27T14:20:00Z",
            "device_id": "ESP32_02",
            "plot_id": "PLOT_B",
            "soil_moisture": 20.0,
            "temperature": 33.0,
            "humidity": 42.0,
            "light": 68000.0,
            "npk_n": None,
            "npk_p": None,
            "npk_k": None,
            "water_flow": 5.8,
            "pump_status": 1,
            "crop_type": "rice",
        },
    },
    {
        "name": "with_sequential_history",
        "description": (
            "Same reading as the first example but the backend also sent the last "
            "30 minutes of readings, so the three historical features are real "
            "instead of placeholders. The score comes out LOWER than the "
            "history-less case (0.72 against 0.92) and both still recommend "
            "irrigation: the 30-minute average is 22.65 %, wetter than the 20 % "
            "now, and the model reads a wetter average as 'recently watered'. "
            "History is not a one-way lever - compare this with the boundary "
            "table in docs/model_results_and_tests.md."
        ),
        "request": {
            "timestamp": "2026-09-27T14:20:00Z",
            "device_id": "ESP32_01",
            "plot_id": "PLOT_A",
            "soil_moisture": 20.0,
            "temperature": 33.0,
            "humidity": 42.0,
            "light": 68000.0,
            "npk_n": 42.0,
            "npk_p": 27.0,
            "npk_k": 51.0,
            "water_flow": 0.0,
            "pump_status": 0,
            "crop_type": "tomato",
            "history": {
                "recent_soil_moisture": [
                    {"timestamp": "2026-09-27T14:00:00Z", "soil_moisture": 23.4},
                    {"timestamp": "2026-09-27T14:10:00Z", "soil_moisture": 21.9},
                ],
                "last_irrigation_at": "2026-09-27T06:40:00Z",
            },
        },
    },
    {
        "name": "weather_outside_the_training_range",
        "description": (
            "A reading the model was never fitted on. The fitted light "
            "coefficient is negative, so a linear model extrapolates a bright, "
            "cold, dry reading into a confident 'do not irrigate' - even at 16 % "
            "soil moisture, well past the tomato trigger. Any reading outside the "
            "recorded training range therefore hands the decision to the crop "
            "threshold rule, and the response says so instead of quoting a score "
            "nobody should act on."
        ),
        "request": {
            "timestamp": "2026-09-27T14:20:00Z",
            "device_id": "ESP32_03",
            "plot_id": "PLOT_C",
            "soil_moisture": 16.0,
            "temperature": -10.0,
            "humidity": 0.0,
            "light": 200000.0,
            "npk_n": None,
            "npk_p": None,
            "npk_k": None,
            "water_flow": 0.0,
            "pump_status": 0,
            "crop_type": "tomato",
        },
    },
]


def read_manifest():
    manifest = CROP_HEALTH_DIR / "splits.csv"
    if not manifest.exists():
        return []
    import csv

    with open(manifest, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_irrigation_examples():
    cases = []
    for example in IRRIGATION_EXAMPLES:
        request = IrrigationRequest(**example["request"])
        prediction = irrigation_model.predict(request)
        cases.append(
            {
                "name": example["name"],
                "description": example["description"],
                "endpoint": "POST /v1/predict/irrigation",
                "request": example["request"],
                "response": {
                    "model": "irrigation",
                    "version": __version__,
                    "prediction": prediction,
                },
            }
        )
    return {
        "generated_by": "python scripts/make_examples.py",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "irrigation_model_trained": irrigation_model.is_trained,
        "irrigation_model_version": irrigation_model.model_version,
        "notes": [
            "These are real responses from the model files in models/.",
            "prediction_score is a logistic-regression score, not a calibrated "
            "probability of being correct, and it saturates near 0 and 1.",
            "history_used is false when the three historical features had to be "
            "filled with placeholders, which is the case in every example except "
            "'with_sequential_history'. The score is not trustworthy when it is "
            "false. It is null when no model was consulted at all, i.e. for a "
            "safety guard or the rule-based fallback.",
            "The 'water now / do not water' decision is score >= "
            f"{irrigation_model.decision_threshold}, a threshold tuned on held-out "
            "training data. Do not read 0.50 as the cut-off.",
            "decision_source names what produced the advice: 'model', the two "
            "safety guards 'pump_guard'/'water_guard', 'rule_fallback' when no "
            "model file is present, and 'rule_fallback_out_of_range' when a "
            "client-supplied reading lies outside the range the model was fitted "
            "on. In the last case prediction_score is the rule's score on the "
            "model's scale, and the model's own extrapolated score appears in "
            "reason for diagnosis only. The invariant "
            "irrigation_needed == (prediction_score >= decision_threshold) holds "
            "on every path, so a backend never needs to special-case the source.",
            "crop_type is a real model input. The trigger level is crop specific "
            "(rice is kept wetter than tomato), and only the trained crops "
            "tomato/rice/pepper are validated.",
            "irrigation_duration_min is always null: AgriGuard has no "
            "defensible irrigation-duration model.",
            "advisory_only is always true - the backend/IoT layer decides what "
            "to do with a recommendation.",
        ],
        "examples": cases,
    }


def build_crop_health_examples():
    if not crop_health_model.is_trained:
        return {
            "generated_by": "python scripts/make_examples.py",
            "crop_health_model_trained": False,
            "load_error": crop_health_model.load_error,
            "note": "Run python scripts/train_crophealth.py to produce real examples.",
            "examples": [],
        }

    manifest = read_manifest()
    chosen = []
    # a seeded shuffle rather than "the first N sorted", so the examples are not
    # always drawn from the same one or two original classes
    for category, count in (("healthy", 2), ("possible_disease", 2)):
        rows = sorted(
            (r for r in manifest if r["category"] == category and r["split"] == "test"),
            key=lambda r: r["key"],
        )
        random.Random(0).shuffle(rows)
        chosen.extend(rows[:count])

    cases = []
    for row in chosen:
        path = CROP_HEALTH_DIR / row["key"]
        prediction = crop_health_model.predict(path)
        cases.append(
            {
                "name": f"unseen_test_image_{row['split']}_{row['category']}",
                "description": (
                    f"Real PlantVillage image from the {row['split']} split "
                    f"(original class: {row['original_class']}). The CNN never "
                    f"saw this image during training."
                ),
                "endpoint": "POST /v1/predict/crop-health",
                "request": {
                    "content_type": "multipart/form-data",
                    "field": "image",
                    "filename": path.name,
                    "bytes": path.stat().st_size,
                },
                "ground_truth": row["category"],
                "correct": prediction["class"] == row["category"],
                "response": {
                    "model": "crop-health",
                    "version": __version__,
                    "prediction": prediction,
                },
            }
        )
    return {
        "generated_by": "python scripts/make_examples.py",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "crop_health_model_trained": True,
        "crop_health_model_version": crop_health_model.model_version,
        "notes": [
            "Real responses from the trained MobileNetV3-Small in models/.",
            "confidence is an uncalibrated softmax score, not a guarantee.",
            "The model separates Healthy from Possible Disease only. It does "
            "not name a disease.",
        ],
        "examples": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=EXAMPLES_DIR)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for name, payload in (
        ("irrigation_prediction.json", build_irrigation_examples()),
        ("crop_health_prediction.json", build_crop_health_examples()),
    ):
        path = args.out / name
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"wrote {path}  ({len(payload['examples'])} examples)")


if __name__ == "__main__":
    main()
