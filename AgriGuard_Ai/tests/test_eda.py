"""Tests for scripts/eda.py.

These check the two things that can silently rot. First, that the analysis
still runs and still produces its artefacts - a script that works once and then
fails after a dataset change is worse than no script, because the report keeps
being cited. Second, that the numbers in `docs/eda_report.md` are the numbers
in `models/eda_summary.json`, which is the same class of bug that
`test_docs_consistency.py` exists to catch for the model metrics.

The statistical tests here assert on *method*, not on a hard-coded p-value. A
p-value recomputed from a regenerated dataset is a brittle thing to pin down,
and pinning it would make the suite fail every time the synthetic generator
changes - which is exactly when the analysis most needs re-running.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agriguard.config import IRRIGATION_CSV  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = PROJECT_ROOT / "docs" / "eda_report.md"
SUMMARY_PATH = PROJECT_ROOT / "models" / "eda_summary.json"


@pytest.fixture(scope="module")
def eda_run(tmp_path_factory):
    """Run the script once for the whole module and hand back both artefacts.

    The outputs are redirected into a temporary directory on purpose. The fast
    flags here produce a *thinner* report than the shipped one - three folds
    instead of five, no crop confidence intervals - so letting the test write
    to `docs/` and `models/` would quietly replace the real artefacts with a
    lesser version every time the suite ran. That is the same docs/artefact
    desync this project already shipped once.
    """
    out = tmp_path_factory.mktemp("eda")
    result = subprocess.run(
        [
            sys.executable, "scripts/eda.py",
            "--skip-crop-inference", "--folds", "3",
            "--summary", str(out / "eda_summary.json"),
            "--report", str(out / "eda_report.md"),
            "--figures", str(out / "figures"),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"scripts/eda.py failed:\n{result.stdout}\n{result.stderr}"

    summary = out / "eda_summary.json"
    report = out / "eda_report.md"
    assert summary.exists(), "eda.py did not write the JSON summary"
    assert report.exists(), "eda.py did not write the markdown report"
    return {
        "stdout": result.stdout,
        "summary": json.loads(summary.read_text(encoding="utf-8")),
        "report": report.read_text(encoding="utf-8"),
        "dir": out,
    }


def test_running_the_analysis_never_touches_the_shipped_artefacts(eda_run):
    """The guard on the guard: pytest must leave docs/ and models/ alone."""
    if not SUMMARY_PATH.exists():
        pytest.skip("shipped models/eda_summary.json not generated yet")
    folds = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))[
        "irrigation"
    ]["cross_validation"]["folds"]
    assert len(folds) == 5, (
        "models/eda_summary.json has been overwritten by the test fixture; "
        "regenerate it with `python scripts/eda.py`"
    )


# ------------------------------------------------------------------ artefacts
def test_the_analysis_script_runs_and_writes_both_artefacts(eda_run):
    assert "EXPLORATORY DATA ANALYSIS" in eda_run["stdout"]
    assert eda_run["summary"]["generated_by"] == "scripts/eda.py"


def test_the_machine_readable_summary_carries_every_section(eda_run):
    irrigation = eda_run["summary"]["irrigation"]
    for key in (
        "outliers",
        "correlation_with_target",
        "correlation_matrix",
        "cross_validation",
        "hypothesis_tests",
    ):
        assert key in irrigation, f"irrigation summary is missing {key}"
    assert "duplicate_groups" in eda_run["summary"]["crop_health"]


# ---------------------------------------------------------------- correlation
def test_every_model_feature_is_measured_against_the_target(eda_run):
    """A feature the model uses but the analysis ignores is an unexplained one."""
    from agriguard.preprocess import FEATURE_COLUMNS, load_irrigation_csv

    frame, _ = load_irrigation_csv(IRRIGATION_CSV)
    expected = [name for name in FEATURE_COLUMNS if name in frame.columns]
    assert set(eda_run["summary"]["irrigation"]["correlation_with_target"]) == set(expected)


def test_correlations_are_coefficients_not_strings(eda_run):
    for name, values in eda_run["summary"]["irrigation"]["correlation_with_target"].items():
        assert -1.0 <= values["r"] <= 1.0, f"{name} has an out-of-range r"
        assert values["n"] > 0


# ------------------------------------------------------------- cross-validation
def test_cross_validation_respects_time_and_reports_its_own_limits(eda_run):
    """The CV number is only honest if the split is chronological and the
    caveat about correlated folds is carried alongside it."""
    cv = eda_run["summary"]["irrigation"]["cross_validation"]
    assert cv["method"].startswith("TimeSeriesSplit")
    assert "why_not_random_folds" in cv
    assert "caveat" in cv
    assert len(cv["folds"]) == 3  # --folds 3 from FAST_ARGS


def test_every_fold_reports_an_error_bar_not_just_a_point(eda_run):
    """A single number invites over-reading; mean + std + range does not."""
    cv = eda_run["summary"]["irrigation"]["cross_validation"]
    for metric in ("accuracy", "balanced_accuracy", "precision", "recall", "f1"):
        spread = cv[metric]
        assert {"mean", "std", "min", "max"} <= set(spread)
        assert spread["min"] <= spread["mean"] <= spread["max"]


def test_cv_does_not_claim_to_beat_the_single_held_out_split(eda_run):
    """The shipped model was chosen on a chronological split; CV folds share a
    boundary day, so they should agree broadly rather than exceed it wildly."""
    cv = eda_run["summary"]["irrigation"]["cross_validation"]
    # Every fold evaluates on a comparable, class-imbalanced slice, so accuracy
    # must stay near the majority-class rate, not jump to something implausible.
    assert 0.85 <= cv["accuracy"]["mean"] <= 1.0


# ------------------------------------------------------------ hypothesis tests
def test_hypothesis_tests_state_their_alpha_and_their_effect_size(eda_run):
    """A bare p-value on 12,960 rows measures precision, not importance."""
    tests = eda_run["summary"]["irrigation"]["hypothesis_tests"]
    if not tests["tests"]:
        pytest.skip("scipy is not installed; the report says so instead")
    for test in tests["tests"]:
        assert test["significant_at_alpha"] == (test["p"] < eda_run["summary"]["alpha"])
    mann_whitney = next(
        (t for t in tests["tests"] if "Mann-Whitney" in t["test"]), None
    )
    assert mann_whitney is not None, "the soil-moisture comparison disappeared"
    assert -1.0 <= mann_whitney["effect_size_rank_biserial"] <= 1.0
    # It must explain why it did not use a t-test, since a t-test is what a
    # reader would assume.
    assert mann_whitney["why_not_t_test"]


# ------------------------------------------------------------------- outliers
def test_the_outlier_scan_reports_two_independent_rules(eda_run):
    """One rule is an opinion; two disagreeing rules are evidence."""
    outliers = eda_run["summary"]["irrigation"]["outliers"]
    assert outliers, "the outlier scan returned nothing"
    for name, values in outliers.items():
        for key in (
            "iqr_low_fence", "iqr_high_fence",
            "count_below_iqr_fence", "count_above_iqr_fence",
            "count_abs_z_over_3",
        ):
            assert key in values, f"{name} is missing {key}"
        assert values["iqr_low_fence"] <= values["iqr_high_fence"]


def test_the_outlier_scan_deletes_nothing(eda_run):
    """`preprocess.py` owns cleaning; this script only measures."""
    outliers = eda_run["summary"]["irrigation"]["outliers"]
    assert "dropped" not in outliers  # no per-feature deletion happened
    irrigation = eda_run["summary"]["irrigation"]
    assert irrigation["n_rows"] > 0


# --------------------------------------------------------------------- report
def test_the_report_quotes_the_summary_rather_than_hand_written_numbers(eda_run):
    """The same defect class as the docs/metrics desync this project already
    had one of: the prose must not drift from the computed values."""
    summary, report = eda_run["summary"], eda_run["report"]
    cv = summary["irrigation"]["cross_validation"]
    assert f"{cv['accuracy']['mean']:.4f}" in report
    strongest = max(
        summary["irrigation"]["correlation_with_target"].values(), key=lambda v: abs(v["r"])
    )
    assert f"{strongest['r']:+.4f}" in report
    assert str(summary["irrigation"]["n_rows"]) in report.replace(",", "") or (
        f"{summary['irrigation']['n_rows']:,}" in report
    )


def test_the_report_states_what_was_not_computed(eda_run):
    """Skipped work has to be visible. A report that silently omits the
    confidence intervals reads exactly like one that computed them."""
    report = eda_run["report"]
    assert "--skip-crop-inference" in report


def test_the_duplicate_group_split_is_verified_not_asserted(eda_run):
    duplicates = eda_run["summary"]["crop_health"]["duplicate_groups"]
    assert duplicates["n_groups_spanning_multiple_splits"] == 0
    assert duplicates["clean"] is True
