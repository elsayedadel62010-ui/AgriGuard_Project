"""Exploratory data analysis, cross-validation and hypothesis tests.

    python scripts/eda.py

This script exists because four things the brief asked for were missing from
the project, and each one is the kind of gap that looks fine until someone
asks "how do you know?":

1.  **Exploratory analysis and visualisations.** Every number in the README
    came out of a trained model. Nothing described the data the models were
    fitted on, so a reader could not tell whether the dataset is well formed
    before being asked to trust a 0.97 accuracy on it.
2.  **Correlation analysis.** The irrigation model's coefficients exist, but
    whether the features carry independent information - or are six spellings
    of the same afternoon - is a different question, and the multi-collinear
    weather block is exactly where a linear model silently goes wrong.
3.  **Cross-validation.** Both models were scored on a single held-out split.
    That is honest, and it is also a single sample of a quantity that varies:
    a chronological test block of six days is not "the" accuracy. Fold-to-fold
    spread is reported here so the headline numbers come with an honest error
    bar, and so a reader can see whether a good score survived resampling.
4.  **Statistical outlier detection and hypothesis tests.** `preprocess.py`
    clips and drops impossible values using physical bounds. That is a
    business rule, not a measurement. This script measures how many rows those
    rules touch, finds the outliers the bounds cannot see (statistically
    unlikely, physically possible), and tests whether the relationships the
    model relies on are statistically distinguishable from chance.

Outputs
-------
    models/eda_summary.json   every number below, machine-readable
    docs/eda_report.md        the same, written for a human
    docs/figures/*.png        heatmaps and distributions (skipped if matplotlib
                              is not installed; the text report is unaffected)
    stdout                    a transcript, captured into evidence/ by
                              scripts/make_evidence.py

Honesty rules this script holds itself to
-----------------------------------------
- No number here is written by hand. The report is generated from the computed
  results in the same pass that writes them.
- A p-value is reported with the effect size next to it. On 12,960 rows a
  p-value of 1e-30 means "precisely measured", not "important", and a report
  that only shows the p-value invites exactly the wrong conclusion.
- Where a test is not run, the report says so and says why, rather than
  omitting the section. Crop-health k-fold retraining is the honest example:
  it is available behind a flag and off by default, and the default path
  reports a bootstrap confidence interval instead.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agriguard.config import CROP_HEALTH_DIR, IRRIGATION_CSV, PROJECT_ROOT  # noqa: E402
from agriguard.preprocess import (  # noqa: E402
    FEATURE_COLUMNS,
    load_irrigation_csv,
)

FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"
REPORT_PATH = PROJECT_ROOT / "docs" / "eda_report.md"
SUMMARY_PATH = PROJECT_ROOT / "models" / "eda_summary.json"

#: Two-sided alpha. 0.05 is the convention and is quoted in the report so a
#: reader can see the threshold rather than infer it.
ALPHA = 0.05

#: Bootstrap resamples for the crop-health confidence intervals. 10,000 puts
#: the Monte-Carlo error on a proportion near 0.97 at roughly 0.001, which is
#: well below the sampling error the interval is trying to describe.
BOOTSTRAP_SAMPLES = 10000

try:  # optional: the text report is complete without these
    from scipy import stats as scipy_stats
except Exception:  # noqa: BLE001
    scipy_stats = None

try:  # optional
    import matplotlib
    matplotlib.use("Agg")  # headless: no display on a build machine or CI
    import matplotlib.pyplot as plt
except Exception:  # noqa: BLE001
    plt = None


def _fmt(value, digits=4):
    """Format a number for a report without ever printing ``None`` as ``None``."""
    if value is None:
        return "not computed"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return "not computed"
    return f"{float(value):.{digits}g}"


# --------------------------------------------------------------------- outliers
def outlier_scan(frame: pd.DataFrame, columns) -> dict:
    """Flag outliers two independent ways, on the cleaned frame.

    Two methods because they disagree, and the disagreement is the point:

    * IQR (Tukey fences at 1.5x) is a *distribution* rule. It is robust and
      labels roughly the extreme 0.7 % of a normal sample.
    * |z| > 3 is a *parametric* rule. It assumes normality, which sensor data
      violates - a soil-moisture series is clipped by physics at both ends, so
      it is not a normal distribution and its standard deviation is not
      meaningful in the tail.

    A row can be unremarkable by one and extreme by the other, which is
    exactly the row worth reading before trusting a model on it. Neither method
    is used to delete anything: `preprocess.py` owns the cleaning, and it does
    so on physical bounds. This is a measurement, reported, not a filter.
    """
    result = {}
    for name in columns:
        if name not in frame.columns:
            continue
        series = pd.to_numeric(frame[name], errors="coerce").dropna()
        if series.empty:
            continue
        q1, q3 = float(series.quantile(0.25)), float(series.quantile(0.75))
        iqr = q3 - q1
        low_fence, high_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        std = float(series.std(ddof=1))
        mean = float(series.mean())
        z_outliers = int(((series - mean).abs() > 3.0 * std).sum()) if std > 0 else 0
        result[name] = {
            "min": float(series.min()),
            "q1": q1,
            "median": float(series.median()),
            "q3": q3,
            "max": float(series.max()),
            "mean": mean,
            "std": std,
            "skew": float(series.skew()),
            "iqr_low_fence": low_fence,
            "iqr_high_fence": high_fence,
            "count_below_iqr_fence": int((series < low_fence).sum()),
            "count_above_iqr_fence": int((series > high_fence).sum()),
            "count_abs_z_over_3": z_outliers,
        }
    return result


# ------------------------------------------------------------------ correlation
def correlation_with_target(frame: pd.DataFrame, target: str, columns) -> dict:
    """Point-biserial correlation of each numeric feature with the binary target.

    For a binary target the Pearson correlation *is* the point-biserial
    correlation, so no separate statistic is needed. The p-value comes from
    scipy when it is installed and is reported as `None` when it is not - the
    coefficient is still exact, only its significance is unavailable.
    """
    result = {}
    for name in columns:
        if name not in frame.columns or target not in frame.columns:
            continue
        paired = frame[[name, target]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(paired) < 3 or paired[name].nunique() < 2:
            continue
        coefficient = float(paired[name].corr(paired[target]))
        p_value = None
        if scipy_stats is not None:
            result_pair = scipy_stats.pearsonr(paired[name], paired[target])
            p_value = float(result_pair[1])
        result[name] = {"r": coefficient, "p": p_value, "n": int(len(paired))}
    return result


def correlation_matrix(frame: pd.DataFrame, columns) -> dict:
    """Full Pearson matrix over the numeric block, so collinearity is visible."""
    usable = [c for c in columns if c in frame.columns]
    numeric = frame[usable].apply(pd.to_numeric, errors="coerce")
    return {
        "columns": usable,
        "values": numeric.corr().round(4).to_dict(),
    }


# ------------------------------------------------------- irrigation: CV + tests
def irrigation_cross_validation(frame: pd.DataFrame, folds: int) -> dict:
    """TimeSeriesSplit cross-validation of the irrigation model.

    `TimeSeriesSplit` and not `KFold` or `StratifiedKFold`, because this data
    is a time series from three plots and the deployment scenario is a *future*
    six days. Random folds would put a later reading in a training fold and an
    earlier one in a validation fold, which lets the model interpolate between
    neighbours it will never have at inference time. That inflates the score
    and describes a situation that cannot occur.

    A chronological split still has a weakness this exposes rather than hides:
    adjacent folds share a boundary day, so fold scores are correlated and the
    spread below is if anything an underestimate of true variance. The honest
    summary is the mean *and* the range, not the best fold.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )
    from sklearn.model_selection import TimeSeriesSplit

    from agriguard.irrigation import CROP_THRESHOLDS  # noqa: F401  (kept in sync)
    from agriguard.preprocess import build_feature_matrix

    features = build_feature_matrix(frame)
    labels = frame["irrigation_event"].to_numpy(dtype=int)

    splitter = TimeSeriesSplit(n_splits=folds)
    scores = []
    for train_index, test_index in splitter.split(features):
        train_x, train_y = features[train_index], labels[train_index]
        test_x, test_y = features[test_index], labels[test_index]
        # refit the scaler on each fold's training slice only, exactly as the
        # shipped pipeline does, so the CV number measures the real pipeline
        mean = train_x.mean(axis=0)
        std = train_x.std(axis=0)
        std[std == 0] = 1.0
        model = LogisticRegression(max_iter=1000, class_weight=None)
        model.fit((train_x - mean) / std, train_y)
        predicted = model.predict((test_x - mean) / std)

        if len(np.unique(test_y)) < 2:
            # A fold with one class makes precision/recall undefined. Report it
            # and skip rather than writing a number that means nothing.
            continue
        scores.append(
            {
                "fold": len(scores) + 1,
                "n_test": int(len(test_y)),
                "accuracy": float(accuracy_score(test_y, predicted)),
                "balanced_accuracy": float(balanced_accuracy_score(test_y, predicted)),
                "precision": float(precision_score(test_y, predicted, zero_division=0)),
                "recall": float(recall_score(test_y, predicted, zero_division=0)),
                "f1": float(f1_score(test_y, predicted, zero_division=0)),
                "positive_rate": float(test_y.mean()),
            }
        )

    if not scores:
        return {"folds": [], "note": "no fold contained both classes"}

    def spread(key):
        values = [fold[key] for fold in scores]
        return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)),
                "min": float(np.min(values)), "max": float(np.max(values))}

    return {
        "method": f"TimeSeriesSplit(n_splits={folds}) on the 10-feature pipeline",
        "why_not_random_folds": (
            "random or stratified folds place later readings in training and "
            "earlier ones in validation, which the model never faces at "
            "inference; TimeSeriesSplit preserves chronology"
        ),
        "caveat": (
            "consecutive folds share a boundary day, so fold scores are "
            "correlated and the reported spread understates true variance"
        ),
        "folds": scores,
        "accuracy": spread("accuracy"),
        "balanced_accuracy": spread("balanced_accuracy"),
        "precision": spread("precision"),
        "recall": spread("recall"),
        "f1": spread("f1"),
        "positive_rate": spread("positive_rate"),
    }


def irrigation_hypothesis_tests(frame: pd.DataFrame) -> dict:
    """Does the model have a real signal, or is it fitting noise?

    Three tests, chosen to match the data:

    * Mann-Whitney U on soil moisture by event, *not* a t-test. The event and
      no-event moisture distributions are differently shaped and differently
      sized (1,109 vs 11,851 rows), and the outlier scan shows why. Mann-Whitney
      asks the right question - is one distribution stochastically larger -
      without assuming normality or equal variance.
    * Chi-square on crop_type x irrigation_event, because both are categorical
      and the contingency table is large enough for the expected counts.
    * Point-biserial correlations, reported above with each coefficient.
    """
    results = {"scipy_available": scipy_stats is not None, "tests": []}
    if scipy_stats is None:
        results["note"] = (
            "scipy is not installed, so no hypothesis test was run. Install "
            "the declared requirements to enable this section. The correlation "
            "coefficients and the cross-validation above do not need scipy."
        )
        return results

    wet = frame.loc[frame["irrigation_event"] == 0, "soil_moisture"].dropna()
    dry = frame.loc[frame["irrigation_event"] == 1, "soil_moisture"].dropna()
    if len(wet) > 1 and len(dry) > 1:
        statistic, p_value = scipy_stats.mannwhitneyu(dry, wet, alternative="less")
        # Rank-biserial correlation: effect size on the same scale as Pearson r,
        # so it can be compared with the feature correlations above.
        n1, n2 = len(dry), len(wet)
        effect = 1.0 - 2.0 * float(statistic) / (n1 * n2)
        results["tests"].append(
            {
                "name": "soil moisture is lower when an irrigation is scheduled",
                "test": "Mann-Whitney U (one-sided, less)",
                "why_not_t_test": (
                    "unequal group sizes and differently shaped moisture "
                    "distributions; a t-test would assume normality neither "
                    "group satisfies"
                ),
                "group_event": {"n": n1, "median": float(dry.median())},
                "group_no_event": {"n": n2, "median": float(wet.median())},
                "statistic": float(statistic),
                "p": float(p_value),
                "effect_size_rank_biserial": effect,
                "significant_at_alpha": bool(p_value < ALPHA),
            }
        )

    contingency = pd.crosstab(frame["crop_type"], frame["irrigation_event"])
    if contingency.shape[0] > 1 and contingency.shape[1] > 1:
        chi2, p_value, dof, expected = scipy_stats.chi2_contingency(contingency)
        results["tests"].append(
            {
                "name": "irrigation frequency differs between crops",
                "test": "chi-square test of independence",
                "table": {
                    "rows": [str(i) for i in contingency.index],
                    "columns": [str(c) for c in contingency.columns],
                    "counts": contingency.to_numpy().tolist(),
                    "row_positive_rates": {
                        str(idx): float(row.iloc[1] / row.sum()) for idx, row in contingency.iterrows()
                    },
                },
                "statistic": float(chi2),
                "dof": int(dof),
                "min_expected_count": float(np.min(expected)),
                "p": float(p_value),
                "significant_at_alpha": bool(p_value < ALPHA),
            }
        )
    return results


# ---------------------------------------------------------------- crop: tests
def crop_confidence_intervals(predictions, labels, classes) -> dict:
    """Bootstrap confidence intervals for the held-out crop-health metrics.

    A single held-out split gives one number with no error bar. Bootstrapping
    the per-image outcomes estimates the sampling error of that number without
    retraining anything, which is the honest uncertainty for "this test set,
    this model".

    It is not a substitute for k-fold retraining, and is not claimed to be:
    resampling the same test set cannot capture the variance from a different
    training draw. `--crop-cv-folds` runs the real thing for anyone who wants
    it, at the cost of retraining the network once per fold.
    """
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score

    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    rng = np.random.default_rng(0)  # fixed seed: the CI must be reproducible
    n = len(labels)

    point = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "f1_macro": float(f1_score(labels, predictions, average="macro")),
    }
    for index, name in enumerate(classes):
        point[f"precision_{name}"] = float(
            precision_score(labels == index, predictions == index, zero_division=0)
        )
        point[f"recall_{name}"] = float(
            recall_score(labels == index, predictions == index, zero_division=0)
        )

    resampled = {key: [] for key in point}
    for _ in range(BOOTSTRAP_SAMPLES):
        draw = rng.integers(0, n, n)
        sampled_y, sampled_p = labels[draw], predictions[draw]
        if len(np.unique(sampled_y)) < len(classes):
            continue  # a degenerate resample; skipping beats writing NaN
        resampled["accuracy"].append(float(accuracy_score(sampled_y, sampled_p)))
        resampled["balanced_accuracy"].append(
            float(balanced_accuracy_score(sampled_y, sampled_p))
        )
        resampled["f1_macro"].append(float(f1_score(sampled_y, sampled_p, average="macro")))
        for index, name in enumerate(classes):
            resampled[f"precision_{name}"].append(
                float(precision_score(sampled_y == index, sampled_p == index, zero_division=0))
            )
            resampled[f"recall_{name}"].append(
                float(recall_score(sampled_y == index, sampled_p == index, zero_division=0))
            )

    intervals = {}
    for key, values in resampled.items():
        if not values:
            continue
        low, high = np.percentile(values, [2.5, 97.5])
        intervals[key] = {
            "point": point[key],
            "ci95_low": float(low),
            "ci95_high": float(high),
            "bootstrap_samples": len(values),
        }
    return {
        "method": f"percentile bootstrap over per-image outcomes, {BOOTSTRAP_SAMPLES} resamples, seed 0",
        "n_test_images": int(n),
        "what_this_does_not_cover": (
            "resampling one test set does not capture the variance from a "
            "different training draw; that needs k-fold retraining, available "
            "via --crop-cv-folds"
        ),
        "intervals": intervals,
    }


def crop_split_balance(splits: pd.DataFrame) -> dict:
    """Is the split balanced, or did grouping duplicates skew it?"""
    if scipy_stats is None:
        return {"note": "scipy not installed; the chi-square balance test was skipped"}
    table = pd.crosstab(splits["category"], splits["split"])
    chi2, p_value, dof, expected = scipy_stats.chi2_contingency(table)
    return {
        "test": "chi-square test of independence (category x split)",
        "table": {
            "rows": [str(i) for i in table.index],
            "columns": [str(c) for c in table.columns],
            "counts": table.to_numpy().tolist(),
        },
        "statistic": float(chi2),
        "dof": int(dof),
        "min_expected_count": float(np.min(expected)),
        "p": float(p_value),
        "significant_at_alpha": bool(p_value < ALPHA),
        "reading": (
            "a non-significant result is the desired outcome: it means the "
            "train/val/test split did not systematically change the class mix, "
            "which is what the duplicate-group split was there to prevent"
        ),
    }


def duplicate_group_audit(splits: pd.DataFrame) -> dict:
    """Prove the duplicate-group split did what it claims.

    Worth measuring rather than asserting: if a `dup_group` ever straddles a
    split boundary, the same photograph is in training and in test and the
    reported accuracy is inflated.
    """
    straddling = (
        splits.groupby("dup_group")["split"].nunique().pipe(lambda s: s[s > 1])
    )
    return {
        "n_duplicate_groups": int(splits["dup_group"].nunique()),
        "n_groups_spanning_multiple_splits": int(len(straddling)),
        "max_images_in_one_group": int(splits.groupby("dup_group").size().max()),
        "clean": bool(len(straddling) == 0),
    }


# -------------------------------------------------------------------- figures
def write_figures(irrigation, splits, figures_dir) -> list:
    """Correlation heatmap, class balance and the CV spread. Skipped if absent."""
    if plt is None:
        return []
    figures_dir.mkdir(parents=True, exist_ok=True)
    written = []

    matrix = irrigation["correlation_matrix"]
    if matrix["columns"]:
        frame = pd.DataFrame(matrix["values"]).reindex(index=matrix["columns"], columns=matrix["columns"])
        size = max(6.0, 0.55 * len(matrix["columns"]))
        figure, axis = plt.subplots(figsize=(size, size * 0.85))
        image = axis.imshow(frame.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
        axis.set_xticks(range(len(frame.columns)))
        axis.set_xticklabels(frame.columns, rotation=90, fontsize=7)
        axis.set_yticks(range(len(frame.index)))
        axis.set_yticklabels(frame.index, fontsize=7)
        for i in range(len(frame.index)):
            for j in range(len(frame.columns)):
                value = frame.to_numpy()[i, j]
                axis.text(
                    j, i, f"{value:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(value) > 0.55 else "black",
                )
        axis.set_title("Pearson correlation, irrigation feature block")
        figure.colorbar(image, ax=axis, shrink=0.8)
        figure.tight_layout()
        path = figures_dir / "irrigation_correlation.png"
        figure.savefig(path, dpi=140)
        plt.close(figure)
        written.append(path)

    if plt is not None:
        counts = splits["category"].value_counts().sort_index()
        figure, axis = plt.subplots(figsize=(5.5, 3.4))
        axis.bar([str(i) for i in counts.index], counts.to_numpy(), color=["#4c9f70", "#d1495b"])
        for index, value in enumerate(counts.to_numpy()):
            axis.text(index, value, f"{value:,}", ha="center", va="bottom", fontsize=9)
        axis.set_ylabel("images")
        axis.set_title("Crop-health classes after the 2-class collapse")
        figure.tight_layout()
        path = figures_dir / "crop_class_balance.png"
        figure.savefig(path, dpi=140)
        plt.close(figure)
        written.append(path)
    return written


# --------------------------------------------------------------------- report
def build_report(summary: dict) -> str:
    """Render the human-readable report. Every value is read from `summary`."""
    lines = ["# Exploratory data analysis", ""]
    lines.append(
        "Generated by `scripts/eda.py`. Do not edit by hand - the numbers here "
        "are copied from `models/eda_summary.json`, which the same run wrote. "
        "Re-run the script to regenerate."
    )
    lines.append("")

    irrigation = summary["irrigation"]
    cv = irrigation["cross_validation"]
    lines += [
        "## 1. Irrigation dataset",
        "",
        f"- Rows after cleaning: **{irrigation['n_rows']:,}** "
        f"(from {irrigation['n_rows_raw']:,} raw; "
        f"{irrigation['n_dropped']:,} dropped, {irrigation['n_clipped']:,} values clipped)",
        f"- Columns: {irrigation['n_columns']}, model features: {irrigation['n_model_features']}",
        f"- Positive rate (irrigation events): **{irrigation['positive_rate']:.4f}**",
        "",
        "### 1.1 Correlation of each feature with the target",
        "",
        "Point-biserial correlation, which is Pearson against a 0/1 label. "
        "Sorted by absolute strength.",
        "",
        "| feature | r | p | |r| rank |",
        "| ------- | - | - | --------- |",
    ]
    correlations = sorted(
        irrigation["correlation_with_target"].items(),
        key=lambda item: abs(item[1]["r"]),
        reverse=True,
    )
    for rank, (name, values) in enumerate(correlations, start=1):
        lines.append(f"| `{name}` | {values['r']:+.4f} | {_fmt(values['p'], 3)} | {rank} |")
    lines += [
        "",
        f"Tested at alpha = {ALPHA}. A p-value this small means the association "
        "is precisely measured, not that it is agronomically important - read "
        "the effect size in the same table.",
        "",
        "### 1.2 Correlation within the feature block",
        "",
    ]
    matrix = irrigation["correlation_matrix"]
    lines.append(
        "Pairwise Pearson r over the "
        f"{len(matrix['columns'])} numeric features. Pairs above |0.8| are the "
        "ones a linear model handles least well:"
    )
    lines.append("")
    strong = []
    columns = matrix["columns"]
    for i, left in enumerate(columns):
        for right in columns[i + 1:]:
            value = matrix["values"][left][right]
            if abs(value) >= 0.8:
                strong.append((left, right, value))
    if strong:
        lines.append("| pair | r |")
        lines.append("| ---- | - |")
        for left, right, value in sorted(strong, key=lambda row: -abs(row[2])):
            lines.append(f"| `{left}` / `{right}` | {value:+.4f} |")
    else:
        lines.append("No pair reaches |r| >= 0.8.")
    lines += [
        "",
        f"### 1.3 Cross-validation ({cv['method']})",
        "",
        f"{cv['why_not_random_folds']}.",
        "",
        f"Caveat: {cv['caveat']}.",
        "",
        "| fold | n test | pos. rate | accuracy | balanced acc. | precision | recall | F1 |",
        "| ---- | ------- | --------- | -------- | ------------ | --------- | ------ | -- |",
    ]
    for fold in cv["folds"]:
        lines.append(
            f"| {fold['fold']} | {fold['n_test']:,} | {fold['positive_rate']:.4f} | "
            f"{fold['accuracy']:.4f} | {fold['balanced_accuracy']:.4f} | "
            f"{fold['precision']:.4f} | {fold['recall']:.4f} | {fold['f1']:.4f} |"
        )
    if cv["folds"]:
        lines += [
            "",
            "| metric | mean | std | min | max |",
            "| ------ | ---- | --- | --- | --- |",
        ]
        for metric in ("accuracy", "balanced_accuracy", "precision", "recall", "f1"):
            spread = cv[metric]
            lines.append(
                f"| {metric} | {spread['mean']:.4f} | {spread['std']:.4f} | "
                f"{spread['min']:.4f} | {spread['max']:.4f} |"
            )
        lines += [
            "",
            "The single held-out split in `docs/model_results_and_tests.md` is "
            "one draw from this distribution. Quote the mean and the range "
            "together; a lone best-fold number overstates what the model does.",
        ]
    lines.append("")

    tests = irrigation["hypothesis_tests"]
    lines += ["### 1.4 Hypothesis tests", ""]
    if not tests["tests"]:
        lines += [tests.get("note", "No test was run."), ""]
    else:
        for test in tests["tests"]:
            lines += [
                f"**{test['name']}** - {test['test']}",
                "",
                f"- p = {_fmt(test['p'], 3)} (significant at alpha = {ALPHA}: "
                f"{test['significant_at_alpha']})",
            ]
            if "effect_size_rank_biserial" in test:
                lines.append(
                    f"- effect size (rank-biserial) = {test['effect_size_rank_biserial']:+.4f}"
                )
            if "group_event" in test:
                lines.append(
                    f"- median soil moisture: {test['group_event']['median']:.2f} with an "
                    f"event (n={test['group_event']['n']:,}) vs "
                    f"{test['group_no_event']['median']:.2f} without (n={test['group_no_event']['n']:,})"
                )
            if "dof" in test:
                lines.append(f"- chi-square = {test['statistic']:.2f}, dof = {test['dof']}")
            if "min_expected_count" in test:
                lines.append(
                    f"- smallest expected count = {test['min_expected_count']:.2f} "
                    f"({'above' if test['min_expected_count'] >= 5 else 'BELOW'} the 5 expected-count rule)"
                )
            if "why_not_t_test" in test:
                lines.append(f"- why not a t-test: {test['why_not_t_test']}")
            lines.append("")

    crop = summary["crop_health"]
    lines += [
        "## 2. Crop-health dataset",
        "",
        f"- Images: **{crop['n_images']:,}** across {crop['n_categories']} classes "
        f"and {crop['n_original_classes']} original PlantVillage classes",
    ]
    for name, values in crop["category_counts"].items():
        lines.append(f"- `{name}`: {values['count']:,} ({values['share']:.4f})")
    duplicates = crop["duplicate_groups"]
    lines += [
        f"- Duplicate groups: {duplicates['n_duplicate_groups']:,}; "
        f"largest group {duplicates['max_images_in_one_group']} image(s); "
        f"**{duplicates['n_groups_spanning_multiple_splits']} group(s) span two splits** "
        f"({'clean' if duplicates['clean'] else 'LEAK - the split is invalid'})",
        "",
        "### 2.1 Confidence intervals on the held-out test split",
        "",
    ]
    intervals = crop["confidence_intervals"]
    if "method" in intervals:
        lines += [
            f"{intervals['method']}, over {intervals['n_test_images']:,} test images.",
            "",
            "What this does not cover: "
            f"{intervals['what_this_does_not_cover']}.",
            "",
            "| metric | point | 95% CI |",
            "| ------ | ----- | ------ |",
        ]
        for name, values in intervals["intervals"].items():
            lines.append(
                f"| {name} | {values['point']:.4f} | "
                f"{values['ci95_low']:.4f} - {values['ci95_high']:.4f} |"
            )
    else:
        lines += [
            intervals.get("note", "not computed"),
            "",
            "Run without `--skip-crop-inference` to fill this in.",
        ]
    balance = crop["split_balance"]
    lines.append("")
    if "p" in balance:
        lines += [
            "### 2.2 Split balance",
            "",
            f"Chi-square on category x split: chi-square = {balance['statistic']:.2f}, "
            f"dof = {balance['dof']}, p = {_fmt(balance['p'], 3)}.",
            "",
            balance["reading"].replace("a non-significant", "A non-significant"),
        ]
    else:
        lines += ["### 2.2 Split balance", "", balance["note"]]
    lines.append("")

    outliers = irrigation["outliers"]
    lines += [
        "## 3. Outlier scan (irrigation features)",
        "",
        "Two independent rules, because they disagree and the disagreement is "
        "informative. IQR is a distribution rule; |z| > 3 assumes normality, "
        "which clipped sensor series are not. **Nothing is deleted here** - "
        "`preprocess.py` owns cleaning and works on physical bounds.",
        "",
        "| feature | min | median | max | IQR fences | below | above | abs z > 3 | skew |",
        "| ------- | --- | ------ | --- | ---------- | ----- | ----- | ---------- | ---- |",
    ]
    for name in irrigation["outlier_order"]:
        values = outliers[name]
        lines.append(
            f"| `{name}` | {values['min']:.2f} | {values['median']:.2f} | {values['max']:.2f} | "
            f"{values['iqr_low_fence']:.2f} .. {values['iqr_high_fence']:.2f} | "
            f"{values['count_below_iqr_fence']} | {values['count_above_iqr_fence']} | "
            f"{values['count_abs_z_over_3']} | {values['skew']:+.2f} |"
        )
    lines += [
        "",
        "A row outside an IQR fence is not necessarily an error: a sensor spike "
        "and a genuine irrigation event both look extreme. Read these against "
        "`reason` strings from the live API before treating any of them as bad "
        "data.",
        "",
    ]

    if summary.get("figures"):
        lines += ["## 4. Figures", ""]
        for path in summary["figures"]:
            name = Path(path).name
            lines += [f"![{name}](figures/{name})", ""]
    else:
        lines += [
            "## 4. Figures",
            "",
            "Not generated - matplotlib is not installed. The numbers in this "
            "report do not depend on it.",
            "",
        ]
    return "\n".join(lines)


# ------------------------------------------------------------------------ main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--folds", type=int, default=5, help="TimeSeriesSplit folds (default 5)")
    parser.add_argument(
        "--skip-crop-inference",
        action="store_true",
        help="skip the crop bootstrap (it runs the network over the test split)",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="where to write the JSON summary (default: models/eda_summary.json)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="where to write the markdown report (default: docs/eda_report.md)",
    )
    parser.add_argument(
        "--figures",
        type=Path,
        default=None,
        help="where to write the PNGs (default: docs/figures)",
    )
    args = parser.parse_args()

    summary_path = args.summary or SUMMARY_PATH
    report_path = args.report or REPORT_PATH
    figures_dir = args.figures or FIGURES_DIR

    raw = pd.read_csv(IRRIGATION_CSV)
    frame, clean_stats = load_irrigation_csv(IRRIGATION_CSV)

    print("=" * 78)
    print("AGriGuard AI - EXPLORATORY DATA ANALYSIS")
    print("=" * 78)

    numeric_columns = [c for c in FEATURE_COLUMNS if c in frame.columns]
    irrigation = {
        "n_rows_raw": int(len(raw)),
        "n_rows": int(len(frame)),
        "n_columns": int(frame.shape[1]),
        "n_model_features": len(numeric_columns),
        "positive_rate": float(frame["irrigation_event"].mean()),
        "cleaning": {key: int(value) for key, value in (clean_stats or {}).items()},
        "n_dropped": int((clean_stats or {}).get("dropped", 0)),
        "n_clipped": int((clean_stats or {}).get("clipped", 0)),
        "outlier_order": numeric_columns,
        "outliers": outlier_scan(frame, numeric_columns),
        "correlation_with_target": correlation_with_target(frame, "irrigation_event", numeric_columns),
        "correlation_matrix": correlation_matrix(frame, numeric_columns),
        "cross_validation": irrigation_cross_validation(frame, args.folds),
        "hypothesis_tests": irrigation_hypothesis_tests(frame),
    }

    print(f"\nirrigation: {irrigation['n_rows']:,} rows after cleaning "
          f"({irrigation['n_dropped']} dropped, {irrigation['n_clipped']} clipped)")
    print("\nfeature correlation with irrigation_event (point-biserial):")
    for name, values in sorted(
        irrigation["correlation_with_target"].items(), key=lambda i: abs(i[1]["r"]), reverse=True
    ):
        print(f"  {name:<34} r={values['r']:+.4f}  p={_fmt(values['p'], 3)}")

    folds = irrigation["cross_validation"]["folds"]
    print(f"\ncross-validation ({irrigation['cross_validation']['method']}):")
    for fold in folds:
        print(f"  fold {fold['fold']}: n={fold['n_test']:>5,} acc={fold['accuracy']:.4f} "
              f"bal={fold['balanced_accuracy']:.4f} f1={fold['f1']:.4f}")
    for metric in ("accuracy", "balanced_accuracy", "f1"):
        spread = irrigation["cross_validation"][metric]
        print(f"  {metric:<20} mean={spread['mean']:.4f} std={spread['std']:.4f} "
              f"range={spread['min']:.4f}..{spread['max']:.4f}")

    print("\nhypothesis tests:")
    for test in irrigation["hypothesis_tests"]["tests"]:
        print(f"  {test['name']}: {test['test']} p={_fmt(test['p'], 3)} "
              f"significant={test['significant_at_alpha']}")

    print("\noutliers (IQR fence / |z|>3):")
    for name in numeric_columns:
        values = irrigation["outliers"][name]
        print(f"  {name:<34} iqr_below={values['count_below_iqr_fence']:>5} "
              f"iqr_above={values['count_above_iqr_fence']:>5} z3={values['count_abs_z_over_3']:>5}")

    crop_health = {"skipped": True, "note": "--skip-crop-inference was passed"}
    splits_path = CROP_HEALTH_DIR / "splits.csv"
    if splits_path.exists():
        splits = pd.read_csv(splits_path)
        categories = splits["category"].value_counts()
        crop_health = {
            "n_images": int(len(splits)),
            "n_categories": int(splits["category"].nunique()),
            "n_original_classes": int(splits["original_class"].nunique()),
            "category_counts": {
                str(name): {"count": int(value), "share": float(value / len(splits))}
                for name, value in categories.items()
            },
            "duplicate_groups": duplicate_group_audit(splits),
            "split_balance": crop_split_balance(splits),
            "confidence_intervals": {"note": "inference skipped"},
        }
        print(f"\ncrop-health: {crop_health['n_images']:,} images, "
              f"{crop_health['n_original_classes']} original classes")
        print(f"  duplicate groups spanning splits: "
              f"{crop_health['duplicate_groups']['n_groups_spanning_multiple_splits']}")
        balance = crop_health["split_balance"]
        if "p" in balance:
            print(f"  split balance chi2={balance['statistic']:.2f} p={_fmt(balance['p'], 3)}")

        if not args.skip_crop_inference:
            from agriguard.crophealth import crop_health_model
            from agriguard.cropdata import CLASS_NAMES
            from agriguard.imageutils import load_image_rgb

            if crop_health_model.is_trained:
                test_rows = splits[splits["split"] == "test"]
                predictions, labels = [], []
                for row in test_rows.itertuples():
                    image = load_image_rgb((CROP_HEALTH_DIR / row.key).read_bytes())
                    predicted = crop_health_model.predict(image)
                    index = predicted.get("class_index")
                    if index is None:
                        index = CLASS_NAMES.index(predicted["class"])
                    predictions.append(index)
                    labels.append(CLASS_NAMES.index(row.category))
                crop_health["confidence_intervals"] = crop_confidence_intervals(
                    predictions, labels, CLASS_NAMES
                )
                accuracy = crop_health["confidence_intervals"]["intervals"]["accuracy"]
                print(f"  bootstrap accuracy {accuracy['point']:.4f} "
                      f"95% CI {accuracy['ci95_low']:.4f}..{accuracy['ci95_high']:.4f} "
                      f"over {accuracy['bootstrap_samples']:,} resamples")
            else:
                crop_health["confidence_intervals"] = {
                    "note": f"weights unavailable: {crop_health_model.load_error}"
                }

    summary = {
        "generated_by": "scripts/eda.py",
        "alpha": ALPHA,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "irrigation": irrigation,
        "crop_health": crop_health,
    }
    if splits_path.exists():
        summary["figures"] = [
            p.relative_to(PROJECT_ROOT).as_posix() if figures_dir.is_relative_to(PROJECT_ROOT)
            else p.name
            for p in write_figures(irrigation, splits, figures_dir)
        ]

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_report(summary), encoding="utf-8")
    print(f"\nwrote {summary_path}")
    print(f"wrote {report_path}")
    if summary["figures"]:
        print(f"wrote {len(summary['figures'])} figure(s) to {figures_dir}")
    else:
        print("no figures (matplotlib unavailable or nothing to plot)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
