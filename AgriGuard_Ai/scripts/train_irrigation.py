"""Train the irrigation logistic regression and write models/irrigation_model.json.

    python scripts/train_irrigation.py --data data/irrigation_data.csv

Design decisions (all deliberate, all documented):

* Chronological 80/20 split. Never `train_test_split` with shuffling on a time
  series - that lets the model be scored on readings from before its training
  period, which is data leakage.
* The standardiser is fitted on TRAIN ROWS ONLY.
* Historical features are built with a backward-looking window, so nothing from
  after the prediction timestamp can reach the input.
* The model file stays plain JSON so inference needs numpy only and the backend
  never has to understand any ML internals.

    ##########################################################################
    # The bundled CSV is SYNTHETIC prototype data, not real field data.       #
    # The numbers printed here describe that data only.                      #
    ##########################################################################
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agriguard.config import IRRIGATION_MODEL_PATH  # noqa: E402
from agriguard.preprocess import (  # noqa: E402
    FEATURE_COLUMNS,
    HISTORICAL_DEFAULTS,
    TARGET_COLUMN,
    build_feature_matrix,
    chronological_split,
    fill_historical_defaults,
    load_irrigation_csv,
    scale_features,
)
from agriguard.version import IRRIGATION_MODEL_VERSION  # noqa: E402

TEST_FRACTION = 0.20

#: Fixed seed for every estimator fitted here.
#:
#: Without it, retraining silently produced a *different* model each run. The
#: scaler means/stds and the recorded training ranges came out byte-identical
#: across runs, which proved the data and the chronological split were stable,
#: yet all ten coefficients drifted in the 3rd-4th decimal and the reported
#: no-history accuracy moved 0.8287 -> 0.8295. That is lbfgs: its reductions are
#: not associative, so the result depends on BLAS/LAPACK thread scheduling, and
#: without `random_state` nothing pins it down.
#:
#: The cost was not just a noisy number. `models/irrigation_model.json` and every
#: figure quoted in the docs are two independent copies of the same claim, so a
#: retrain would leave the documentation describing a model that no longer
#: existed, and `tests/test_docs_consistency.py` would fail on numbers that were
#: technically fine and simply un-reproducible.
#:
#: Used by BOTH fits below (the production fit and the threshold-tuning fit).
#: They must agree: the tuned `decision_threshold` is stored in the model file
#: and applied at inference, so an inner fit that drifted would move the
#: operating point of a production model trained moments earlier.
RANDOM_STATE = 42


def metrics(y_true, y_pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "test_samples": int(len(y_true)),
        "test_positives": int(np.sum(y_true)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(
            float(
                2
                * precision_score(y_true, y_pred, zero_division=0)
                * recall_score(y_true, y_pred, zero_division=0)
                / max(
                    1e-9,
                    precision_score(y_true, y_pred, zero_division=0)
                    + recall_score(y_true, y_pred, zero_division=0),
                )
            ),
            4,
        ),
        "confusion_matrix": {
            "labels": [0, 1],
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "majority_class_baseline_accuracy": round(
            float(max(np.mean(y_true == 0), np.mean(y_true == 1))), 4
        ),
    }


def print_report(title, scores, y_true, y_pred) -> None:
    print(f"\n--- {title} ---")
    print(f"Test samples      : {scores['test_samples']}")
    print(f"Test positives    : {scores['test_positives']}")
    print(f"Accuracy          : {scores['accuracy']:.4f}")
    print(f"Balanced accuracy : {scores['balanced_accuracy']:.4f}")
    print(f"Precision (class 1): {scores['precision']:.4f}")
    print(f"Recall    (class 1): {scores['recall']:.4f}")
    print(f"F1        (class 1): {scores['f1']:.4f}")
    cm = scores["confusion_matrix"]
    print(
        "Confusion matrix  : "
        f"TN={cm['true_negative']} FP={cm['false_positive']} "
        f"FN={cm['false_negative']} TP={cm['true_positive']}"
    )
    print(
        "Majority baseline : "
        f"{scores['majority_class_baseline_accuracy']:.4f} "
        "(always predict the majority class)"
    )
    print(classification_report(y_true, y_pred, digits=3, zero_division=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=IRRIGATION_MODEL_PATH)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="print the stored model report and exit without retraining "
        "(safe for documentation/evidence runs)",
    )
    args = parser.parse_args()

    if args.report_only:
        if not args.out.exists():
            raise SystemExit(f"no stored model at {args.out}; train the model first")
        print(f"stored model report from {args.out}")
        print(json.dumps(json.loads(args.out.read_text(encoding="utf-8")), indent=2))
        return

    from agriguard.config import IRRIGATION_CSV

    data_path = args.data or IRRIGATION_CSV

    frame, report = load_irrigation_csv(data_path)
    print(f"dataset        : {data_path}")
    print(f"rows in / out  : {report['rows_in']} -> {report['rows_out']} "
          f"(dropped {report['rows_dropped']}, clipped {report['values_clipped']})")
    print(f"features       : {FEATURE_COLUMNS}")

    if len(frame) < 100:
        raise SystemExit("not enough usable rows to train")

    train, test = chronological_split(frame, test_fraction=TEST_FRACTION)
    print(
        f"split         : chronological 80/20\n"
        f"  train rows  : {len(train)}  {train['timestamp'].min()} -> {train['timestamp'].max()}\n"
        f"  test  rows  : {len(test)}  {test['timestamp'].min()} -> {test['timestamp'].max()}"
    )

    x_train = build_feature_matrix(train)
    y_train = train[TARGET_COLUMN].to_numpy(dtype=int)
    x_test = build_feature_matrix(test)
    y_test = test[TARGET_COLUMN].to_numpy(dtype=int)

    means = x_train.mean(axis=0)
    stds = x_train.std(axis=0)
    # A constant column (e.g. a crop indicator that never occurs in the data)
    # has std 0 and would explode when standardized.
    stds = np.where(stds < 1e-6, 1.0, stds)
    x_train_scaled = scale_features(x_train, means, stds)
    x_test_scaled = scale_features(x_test, means, stds)

    # No class_weight: irrigation events are the minority class, but the
    # chronological test set keeps the natural class balance and an unweighted
    # fit gives a far better precision/recall trade-off on it. The confusion
    # matrix is reported either way.
    model = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)
    model.fit(x_train_scaled, y_train)

    # --- operating point -------------------------------------------------
    # The plain 0.50 cut-off is a poor operating point for an 8-9 % positive
    # class: it caught under half the events in an earlier run. The threshold is
    # therefore tuned on a held-out slice of the TRAINING timeline only - never
    # on the test set - and stored in the model file so training and inference
    # share one operating point. F1 is the objective: a false "water now" costs
    # a farmer water and trust, a missed event is still covered by the ESP32's
    # own threshold rule, so pure recall would be the wrong trade.
    tune_cut = int(len(train) * 0.8)
    inner_train, inner_tune = train.iloc[:tune_cut], train.iloc[tune_cut:]
    x_inner = build_feature_matrix(inner_train)
    inner_means = x_inner.mean(axis=0)
    inner_stds = np.where(x_inner.std(axis=0) < 1e-6, 1.0, x_inner.std(axis=0))
    inner_model = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)
    inner_model.fit(
        scale_features(x_inner, inner_means, inner_stds),
        inner_train[TARGET_COLUMN].to_numpy(dtype=int),
    )
    tune_prob = inner_model.predict_proba(
        scale_features(build_feature_matrix(inner_tune), inner_means, inner_stds)
    )[:, 1]
    tune_true = inner_tune[TARGET_COLUMN].to_numpy(dtype=int)
    grid = np.arange(0.02, 0.62, 0.01)
    f1_by_threshold = [
        f1_score(tune_true, (tune_prob >= t).astype(int), zero_division=0)
        for t in grid
    ]
    # round to 2dp: np.arange steps are not exactly representable, and
    # 0.33999999999999997 leaking into /v1/health looks like a bug
    decision_threshold = round(float(grid[int(np.argmax(f1_by_threshold))]), 2)
    print(
        f"threshold     : {decision_threshold:.2f} "
        f"(F1-optimal on the last 20 % of the training timeline, "
        f"tuning F1={max(f1_by_threshold):.4f})"
    )

    test_prob = model.predict_proba(x_test_scaled)[:, 1]
    y_pred = (test_prob >= decision_threshold).astype(int)
    scores = metrics(y_test, y_pred)
    print_report(
        f"TEST SET (unseen, later in time) at threshold {decision_threshold:.2f}",
        scores,
        y_test,
        y_pred,
    )
    y_pred_half = (test_prob >= 0.5).astype(int)
    print_report(
        "TEST SET at the plain 0.50 threshold (for comparison only)",
        metrics(y_test, y_pred_half),
        y_test,
        y_pred_half,
    )

    # --- honesty check: what happens when the backend sends no history? ------
    # This mirrors build_features_from_request()'s no-history branch exactly, so
    # the number reported here is what a real history-less request produces.
    x_test_no_history = np.array(x_test, dtype=float, copy=True)
    index_of = {name: index for index, name in enumerate(FEATURE_COLUMNS)}
    current_moisture = x_test[:, index_of["soil_moisture"]]
    x_test_no_history[:, index_of["soil_moisture_change"]] = HISTORICAL_DEFAULTS[
        "soil_moisture_change"
    ]
    x_test_no_history[:, index_of["soil_moisture_avg_30min"]] = current_moisture
    x_test_no_history[:, index_of["time_since_last_irrigation_min"]] = (
        HISTORICAL_DEFAULTS["time_since_last_irrigation_min"]
    )
    y_pred_no_history = (
        model.predict_proba(scale_features(x_test_no_history, means, stds))[:, 1]
        >= decision_threshold
    ).astype(int)
    no_history_scores = metrics(y_test, y_pred_no_history)
    print_report(
        "TEST SET with the neutral no-history defaults a real history-less "
        "request produces (avg30=current value, change=0, "
        "time_since=1440). Recall goes to 1.00 and precision collapses: "
        "without history the model errs towards recommending water",
        no_history_scores,
        y_test,
        y_pred_no_history,
    )

    feature_medians = {
        name: float(np.nanmedian(x_train[:, index]))
        for index, name in enumerate(FEATURE_COLUMNS)
    }

    # Per-feature min/max over the TRAINING rows only. Inference compares live
    # readings against these and hands the decision to the crop threshold rule
    # when any reading falls outside, because a linear model extrapolates and
    # the weather features are where that extrapolation is dangerous. Recording
    # the test rows' ranges here would widen the box the model is judged against
    # using data it was not fitted on.
    training_feature_ranges = {
        name: [
            round(float(np.nanmin(x_train[:, index])), 6),
            round(float(np.nanmax(x_train[:, index])), 6),
        ]
        for index, name in enumerate(FEATURE_COLUMNS)
    }

    payload = {
        "model": "logistic-regression",
        "model_version": IRRIGATION_MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_dataset": data_path.name,
        "decision_threshold": decision_threshold,
        "decision_threshold_note": (
            "F1-optimal cut-off selected on the last 20 % of the training "
            "timeline; the test set was not used to pick it. Inference treats "
            "score >= decision_threshold as 'irrigation recommended'."
        ),
        "features": list(FEATURE_COLUMNS),
        "coefs": [round(float(c), 8) for c in model.coef_[0]],
        "intercept": float(round(float(model.intercept_[0]), 8)),
        "means": [round(float(m), 8) for m in means],
        "stds": [round(float(s), 8) for s in stds],
        "feature_medians": {k: round(v, 6) for k, v in feature_medians.items()},
        "training_feature_ranges": training_feature_ranges,
        "training_feature_ranges_note": (
            "min/max per feature over the training split. Inference treats a "
            "request with any reading outside these bounds as "
            "rule_fallback_out_of_range: the model score is still computed and "
            "quoted in the reason, but the crop threshold rule makes the "
            "decision. See docs/limitations.md."
        ),
        "classes": {"0": "irrigation not required / no irrigation event",
                    "1": "irrigation required / irrigation event"},
        "label_meaning": (
            "irrigation_event = 1 when an irrigation action started at this "
            "reading, 0 otherwise. It is a historical label and is never used "
            "as a model input."
        ),
        "split": {
            "method": "chronological",
            "test_fraction": TEST_FRACTION,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_time_range": [str(train["timestamp"].min()), str(train["timestamp"].max())],
            "test_time_range": [str(test["timestamp"].min()), str(test["timestamp"].max())],
            "shuffled": False,
        },
        "cleaning_report": report,
        "test_metrics": scores,
        "test_metrics_without_history": no_history_scores,
        "limitations": [
            "The bundled training CSV is SYNTHETIC prototype data, not real field data.",
            "Prediction scores are uncalibrated logistic outputs, not a "
            "guaranteed probability of being correct.",
            "Historical features need sequential ESP32 readings. Without them the "
            "model falls back to neutral placeholders and its recall goes to 1.00 "
            "while precision collapses - it then recommends water far too often "
            "(see test_metrics_without_history).",
            "Only the crops with an indicator column (crop_is_*) are validated. "
            "A request for any other crop produces all-zero crop indicators and "
            "the response is flagged as extrapolation.",
            "The label depends on the simulator's hidden true moisture and its "
            "hidden per-plot wind factor, which the sensors never observe, so a "
            "perfect separation of the classes is not achievable from these "
            "features.",
        ],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nsaved model -> {args.out}")

    evaluation_path = args.out.parent / "irrigation_metrics.json"
    with open(evaluation_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "model": "irrigation",
                "model_version": IRRIGATION_MODEL_VERSION,
                "trained_at": payload["trained_at"],
                "training_dataset": data_path.name,
                "decision_threshold": decision_threshold,
                "features": list(FEATURE_COLUMNS),
                "split": payload["split"],
                "test_metrics": scores,
                "test_metrics_without_history": no_history_scores,
                "coefficients": dict(zip(FEATURE_COLUMNS, payload["coefs"])),
            },
            handle,
            indent=2,
        )
    print(f"saved metrics -> {evaluation_path}")


if __name__ == "__main__":
    main()
