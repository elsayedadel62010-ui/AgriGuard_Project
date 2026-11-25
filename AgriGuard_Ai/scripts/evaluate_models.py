"""Recompute every reported metric from the saved models and the data on disk.

    python scripts/evaluate_models.py

This is the script that produces the numbers in
``docs/crop_health_results.md`` and ``docs/model_results_and_tests.md``. It
re-runs the evaluation rather than re-typing numbers, so the documentation
cannot drift away from reality.

Nothing here trains anything: it loads the saved artefacts and measures them.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agriguard.config import (  # noqa: E402
    CROP_HEALTH_DIR,
    CROP_HEALTH_META_PATH,
    IRRIGATION_CSV,
    IRRIGATION_MODEL_PATH,
    MODELS_DIR,
)
from agriguard.crophealth import CropHealthModel, heuristic_predict  # noqa: E402
from agriguard.preprocess import (  # noqa: E402
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_feature_matrix,
    chronological_split,
    load_irrigation_csv,
    scale_features,
)


def evaluate_irrigation():
    print("=" * 72)
    print("IRRIGATION MODEL")
    print("=" * 72)
    with open(IRRIGATION_MODEL_PATH, encoding="utf-8") as handle:
        model = json.load(handle)

    frame, report = load_irrigation_csv(IRRIGATION_CSV)
    train, test = chronological_split(frame, test_fraction=model["split"]["test_fraction"])
    x_train = build_feature_matrix(train)
    x_test = build_feature_matrix(test)
    y_test = test[TARGET_COLUMN].to_numpy(dtype=int)

    means = np.asarray(model["means"])
    stds = np.asarray(model["stds"])
    coefs = np.asarray(model["coefs"])
    intercept = float(model["intercept"])

    def score(matrix):
        z = scale_features(matrix, means, stds)
        return 1.0 / (1.0 + np.exp(-(z @ coefs + intercept)))

    # Evaluate at the threshold the model actually ships, not a hard-coded 0.5.
    # The stored value is the F1-optimal cut-off chosen on a held-out slice of
    # the training timeline; default to 0.5 only for a legacy file that has none.
    threshold = float(model.get("decision_threshold", 0.5))
    y_pred = (score(x_test) >= threshold).astype(int)
    y_pred_half = (score(x_test) >= 0.5).astype(int)
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        precision_score,
        recall_score,
    )

    result = {
        "test_samples": int(len(y_test)),
        "test_positives": int(y_test.sum()),
        "decision_threshold": threshold,
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(
            float(
                2
                * precision_score(y_test, y_pred, zero_division=0)
                * recall_score(y_test, y_pred, zero_division=0)
                / max(
                    1e-9,
                    precision_score(y_test, y_pred, zero_division=0)
                    + recall_score(y_test, y_pred, zero_division=0),
                )
            ),
            4,
        ),
        "majority_class_baseline_accuracy": round(
            float(max(np.mean(y_test == 0), np.mean(y_test == 1))), 4
        ),
    }
    # the same model at a plain 0.50 cut-off, so the effect of the chosen
    # operating point is visible instead of hidden
    result["at_threshold_0_50"] = {
        "accuracy": round(float(accuracy_score(y_test, y_pred_half)), 4),
        "balanced_accuracy": round(
            float(balanced_accuracy_score(y_test, y_pred_half)), 4
        ),
        "precision": round(
            float(precision_score(y_test, y_pred_half, zero_division=0)), 4
        ),
        "recall": round(float(recall_score(y_test, y_pred_half, zero_division=0)), 4),
    }
    matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
    result["confusion_matrix"] = {
        "labels": [0, 1],
        "values": matrix.tolist(),
        "tn": int(matrix[0, 0]),
        "fp": int(matrix[0, 1]),
        "fn": int(matrix[1, 0]),
        "tp": int(matrix[1, 1]),
    }

    print(f"model file        : {IRRIGATION_MODEL_PATH.name}")
    print(f"model version     : {model['model_version']}")
    print(f"features          : {model['features']}")
    print(f"training dataset  : {model['training_dataset']} (SYNTHETIC prototype data)")
    print(f"split             : {model['split']['method']}, "
          f"{model['split']['train_rows']} train / {model['split']['test_rows']} test")
    print(f"rows in / out     : {report['rows_in']} -> {report['rows_out']}")
    print()
    print("Test samples      :", result["test_samples"])
    print("Test positives    :", result["test_positives"])
    print("Decision threshold:", result["decision_threshold"], "(as shipped in the model file)")
    print("Accuracy          :", result["accuracy"])
    print("Balanced accuracy :", result["balanced_accuracy"])
    print("Precision (class1):", result["precision"])
    print("Recall    (class1):", result["recall"])
    print("F1        (class1):", result["f1"])
    print("Majority baseline :", result["majority_class_baseline_accuracy"])
    print("Same model at 0.50: accuracy %s balanced %s precision %s recall %s" % (
        result["at_threshold_0_50"]["accuracy"],
        result["at_threshold_0_50"]["balanced_accuracy"],
        result["at_threshold_0_50"]["precision"],
        result["at_threshold_0_50"]["recall"]))
    print("Confusion matrix  : TN=%d FP=%d FN=%d TP=%d" % (
        result["confusion_matrix"]["tn"], result["confusion_matrix"]["fp"],
        result["confusion_matrix"]["fn"], result["confusion_matrix"]["tp"]))
    print()
    print(classification_report(y_test, y_pred, digits=3, zero_division=0))
    print("coefficients (standardised units):")
    for name, value in sorted(zip(model["features"], model["coefs"]), key=lambda kv: -abs(kv[1])):
        print(f"  {name:<32} {value:+.4f}")
    return result


def evaluate_crop_health():
    print()
    print("=" * 72)
    print("CROP-HEALTH MODEL")
    print("=" * 72)
    if not CROP_HEALTH_META_PATH.exists() or not (MODELS_DIR / "crop_health_model.pth").exists():
        print("crop-health weights not found - run scripts/train_crophealth.py first")
        return None

    from scripts.train_crophealth import (  # noqa: PLC0415
        compute_metrics,
        make_loader,
        predict_all,
        read_manifest,
    )
    from agriguard.imageutils import build_transforms  # noqa: PLC0415
    from scripts.train_crophealth import CropHealthDataset  # noqa: PLC0415
    import torch  # noqa: PLC0415

    with open(CROP_HEALTH_META_PATH, encoding="utf-8") as handle:
        meta = json.load(handle)

    model = CropHealthModel()
    if not model.is_trained:
        print(f"could not load weights: {model.load_error}")
        return None

    records = read_manifest(CROP_HEALTH_DIR)
    test_rows = [r for r in records if r["split"] == "test"]
    dataset = CropHealthDataset(
        CROP_HEALTH_DIR, test_rows, build_transforms(train=False)
    )
    loader = make_loader(dataset, meta.get("batch_size", 32), False)
    probabilities, y_true = predict_all(model.net, loader, torch.device("cpu"))
    y_pred = probabilities.argmax(axis=1)
    metrics = compute_metrics(y_true, y_pred, probabilities)

    # documented non-ML baseline on the same test images, for comparison only
    baseline_correct = 0
    for record in test_rows:
        result = heuristic_predict(CROP_HEALTH_DIR / record["key"])
        baseline_correct += int(result["class"] == record["category"])
    baseline_accuracy = round(baseline_correct / len(test_rows), 4)

    print(f"model             : {meta['method']} ({meta['model_version']})")
    print(f"training dataset  : {meta['training_dataset']}")
    print(f"classes           : {meta['classes']} -> {list(meta['class_labels'].values())}")
    print(f"epochs / batch    : {meta['epochs']} / {meta['batch_size']}  image {meta['image_size']}px")
    print(f"parameters        : {meta['parameters_total']:,}")
    print(f"dataset counts    : {meta['dataset_counts']}")
    print()
    print("Test samples      :", metrics["test_samples"])
    print("Test accuracy     :", metrics["accuracy"])
    print("Balanced accuracy :", metrics["balanced_accuracy"])
    for label in metrics["precision_per_class"]:
        print(
            f"  {label:<18} precision {metrics['precision_per_class'][label]:.4f} "
            f"recall {metrics['recall_per_class'][label]:.4f} "
            f"F1 {metrics['f1_per_class'][label]:.4f} "
            f"support {metrics['support_per_class'][label]}"
        )
    print("Macro F1          :", metrics["macro_f1"])
    matrix = metrics["confusion_matrix"]
    print("Confusion matrix (rows = true, cols = predicted):")
    for label, row in zip(matrix["labels"], matrix["values"]):
        print(f"  {label:<18} {row}")
    print()
    print("Colour-heuristic baseline accuracy on the same test images:", baseline_accuracy)
    metrics["color_heuristic_baseline_accuracy"] = baseline_accuracy
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=MODELS_DIR / "evaluation_summary.json"
    )
    args = parser.parse_args()

    summary = {
        "note": (
            "All numbers are measured by scripts/evaluate_models.py on the data in "
            "this repository. The irrigation dataset is SYNTHETIC; the crop-health "
            "images come from PlantVillage, a laboratory dataset. Neither number "
            "is a real-world field accuracy."
        ),
        "irrigation": evaluate_irrigation(),
        "crop_health": evaluate_crop_health(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
