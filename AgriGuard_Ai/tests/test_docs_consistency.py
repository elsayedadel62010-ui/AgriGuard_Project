"""Consistency checks between the documentation and the code.

The README, `docs/` and `evidence/` are generated from the same repository as the
models and the tests, so they can drift. Two things went wrong before and are
pinned here:

- a doc cited `test_the_decision_is_independent_of_weather`, a test that was
  renamed, so the reader was sent looking for something that does not exist;
- a doc quoted `soil_moisture: 14.5`, a reading below the training range, and a
  hand-typed `prediction_score` that the real API did not return.

These tests are cheap and read-only.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from agriguard.config import (
    CROP_HEALTH_META_PATH,
    IRRIGATION_MODEL_PATH,
    PROJECT_ROOT,
)
from agriguard.irrigation import IrrigationModel

DOCS = sorted(list(PROJECT_ROOT.glob("*.md")) + list((PROJECT_ROOT / "docs").glob("*.md")))

#: names that look like test functions but are keys in the model JSON files
JSON_KEY_NAMES = {"test_metrics", "test_metrics_without_history"}


def _doc_text() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in DOCS)


def _defined_test_names() -> set:
    names = set()
    for path in (PROJECT_ROOT / "tests").glob("test_*.py"):
        names |= set(re.findall(r"^def (test_\w+)", path.read_text(encoding="utf-8"), re.M))
    return names


def test_there_are_docs_to_check():
    assert len(DOCS) >= 8, f"only found {len(DOCS)} markdown files"


def test_every_test_name_cited_in_the_docs_exists():
    """A doc must never send a reader to a test that was renamed or removed."""
    defined = _defined_test_names()
    cited = set(re.findall(r"`(test_\w+)`", _doc_text())) - JSON_KEY_NAMES
    missing = sorted(cited - defined)
    assert not missing, (
        "the docs name tests that do not exist: "
        + ", ".join(missing)
        + ". Either restore the test or fix the reference."
    )


def test_json_keys_cited_in_the_docs_are_real_keys():
    with open(IRRIGATION_MODEL_PATH, encoding="utf-8") as handle:
        irrigation = json.load(handle)
    for key in JSON_KEY_NAMES:
        if f"`{key}`" in _doc_text():
            assert key in irrigation, f"docs reference {key}, not present in the model file"


def test_the_quoted_test_count_matches_the_real_suite():
    """The docs quote a total; keep it equal to what pytest actually collects.

    This number has already drifted three times (129 -> 131 -> 133) because
    every new test silently invalidates the sentence above it. Collecting the
    suite from inside a test is cheap next to finding a stale claim in a
    submission.
    """
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    match = re.search(r"(\d+) tests? collected", result.stdout)
    assert match, f"could not read the collected test count from pytest:\n{result.stdout[-500:]}"
    actual = int(match.group(1))

    quoted = (PROJECT_ROOT / "docs" / "model_results_and_tests.md").read_text(encoding="utf-8")
    claims = re.findall(r"(\d+)\s+passed", quoted)
    assert claims, "the test report no longer states a passed count"
    for claim in claims:
        assert int(claim) == actual, (
            f"docs/model_results_and_tests.md claims {claim} passed, "
            f"but pytest collects {actual}. Update the number."
        )


def test_retracted_weather_claims_stay_retracted():
    """Numbers that were measured and found wrong must not reappear.

    The project originally justified dropping temperature/humidity/light by
    claiming it "costs about 0.05 balanced accuracy (0.829 vs 0.779)". Re-measured
    on the same chronological split, the weather variant is the stronger model
    (0.8144 vs 0.7788 balanced accuracy), so that justification was false in the
    direction that mattered. The model was then switched to the weather features
    and these strings are banned so the wrong claim cannot be pasted back in as if
    it were still valid.
    """
    retracted = (
        "costs about 0.05 balanced accuracy",
        "Keeping them cost about 0.05",
        "0.829 vs 0.779",
        "0.05 balanced accuracy on the synthetic test split",
    )
    offenders = []
    for doc in DOCS:
        text = doc.read_text(encoding="utf-8")
        for phrase in retracted:
            if phrase in text:
                offenders.append(f"{doc.name}: {phrase!r}")
    assert not offenders, "retracted weather claims are back in the docs: " + "; ".join(offenders)


def test_no_doc_may_claim_the_weather_readings_are_not_model_inputs():
    """The four current readings are all model inputs. Say so, or say nothing.

    The docs went through a full cycle on this: first they claimed the readings
    were deliberately excluded, then they retracted that on the grounds that the
    weather variant actually scores *better*, and the model was retrained to
    include them. A stale sentence asserting exclusion is now simply false, and a
    reader would act on it - it is the sentence that tells an integrator not to
    expect weather to change a recommendation.
    """
    false_claims = (
        "not model inputs",
        "are not model inputs",
        "cannot move any of these numbers",
        "Weather independence",
        "weather independence",
        "Only four values reach the model",
        "Seven features",
        "seven features",
    )
    offenders = []
    for doc in DOCS:
        text = doc.read_text(encoding="utf-8")
        for phrase in false_claims:
            if phrase in text:
                offenders.append(f"{doc.name}: {phrase!r}")
    assert not offenders, (
        "a doc claims the weather readings are not model inputs, which stopped "
        "being true when the model was retrained: " + "; ".join(offenders)
    )


def test_the_extrapolation_risk_and_its_guard_are_documented():
    """Shipping the weather features has a known failure mode; write it down.

    The fitted `light` coefficient is negative, so the model extrapolates
    confidently in the wrong direction on weather outside the training range - a
    tomato plot at 20 % soil moisture scoring 0.004 and being told not to
    irrigate. The service guards this, and the docs have to say both that the risk
    exists and that `decision_source` reports when the rule answered instead. A
    doc that mentions only one half is the worse option: it either sounds like an
    unqualified endorsement of the model or like an unexplained quirk.
    """
    limitations = (PROJECT_ROOT / "docs" / "limitations.md").read_text(encoding="utf-8")
    assert "0.004" in limitations, (
        "the measured extrapolation failure is not in limitations.md; a guard "
        "with no stated reason is one nobody can review"
    )
    assert "training_feature_ranges" in limitations, (
        "limitations.md must name where the bounds are recorded"
    )
    # the honest coverage gaps, not just the guard
    assert "not range-checked" in limitations, (
        "limitations.md must say which features the guard skips and why"
    )

    contract = (PROJECT_ROOT / "docs" / "api_contract.md").read_text(encoding="utf-8")
    assert "rule_fallback_out_of_range" in contract, (
        "api_contract.md must document the out-of-range decision_source value"
    )
    assert "decision_source" in contract


def test_the_readme_honesty_table_matches_the_shipped_model():
    """The short honesty checklist must not drift from the real numbers.

    The README carries a condensed copy of the "what you may claim" table from
    docs/limitations.md. A condensed copy drifts: it kept quoting the retracted
    `0.93` balanced accuracy long after the leak that produced it was fixed, and
    it still said the weather readings were not model inputs. A reviewer who reads
    only the README would have been misled on both counts.

    This checks the accuracy claim against the artifact, so the copy cannot go
    stale on its own.
    """
    with open(IRRIGATION_MODEL_PATH, encoding="utf-8") as handle:
        model = json.load(handle)
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    balanced = model["test_metrics"]["balanced_accuracy"]
    assert f"{balanced:.4f}" in readme, (
        f"the README honesty table does not quote the shipped balanced accuracy "
        f"({balanced:.4f})"
    )

    # Scope this to the checklist table. Elsewhere the README deliberately
    # narrates the retracted 0.93 as a leak that was found and fixed, and that
    # sentence is worth keeping - it is the "why the numbers are low" answer.
    # What must not survive is 0.93 sitting in a table of claims you may make.
    # next `##` heading, whatever it turns out to be
    table = re.search(
        r"^## Honesty checklist\s*\n(.*?)(?=^## )", readme, re.S | re.M
    )
    assert table, "the README honesty checklist section has moved; update this test"
    # The one legitimate mention is the row that refuses the claim. A blanket
    # substring ban would forbid saying "no, do not claim 0.93", which is the
    # opposite of the point.
    forbidden = [
        line
        for line in table.group(1).splitlines()
        if "0.93" in line and not re.search(r"\|\s*\**no", line, re.I)
    ]
    assert not forbidden, (
        "the README honesty table allows a 0.93 balanced-accuracy claim; that "
        "number came from a target leak that has since been fixed: "
        + " | ".join(forbidden)
    )


def test_the_shipped_irrigation_metrics_are_quoted_in_the_docs():
    """The headline numbers must be the ones in models/irrigation_metrics.json."""
    with open(IRRIGATION_MODEL_PATH, encoding="utf-8") as handle:
        model = json.load(handle)
    text = _doc_text()
    for block in ("test_metrics", "test_metrics_without_history"):
        for key in ("accuracy", "balanced_accuracy", "precision", "recall", "f1"):
            value = model[block][key]
            assert f"{value:.4f}" in text, (
                f"{block}.{key} = {value:.4f} is not quoted anywhere in the docs"
            )


def test_the_documented_irrigation_boundaries_match_the_model():
    """The per-crop triggers quoted in the docs are re-measured, not trusted.

    The numbers are measured with the same helper `scripts/decision_boundaries.py`
    prints, so a retrain that moves them fails here instead of leaving a stale
    trigger in the README. Hard-coding the expected values in this file was the
    previous approach and it needed editing on every retrain, which is exactly the
    kind of bookkeeping that gets skipped.
    """
    from scripts import decision_boundaries

    model = IrrigationModel(model_path=IRRIGATION_MODEL_PATH)
    text = _doc_text()
    window = decision_boundaries.feasible_window(35.0, -2.0)
    checked = 0
    for crop in ("tomato", "rice", "pepper"):
        ask = decision_boundaries.make_asker(model, crop, 35.0, -2.0, 60.0)
        served, _ = decision_boundaries.boundary(ask, model.decision_threshold, window)
        if served is None or decision_boundaries.is_truncated(served, window):
            # Rice's trigger sits above the range where the pinned history context
            # is representable, so there is no boundary to quote for that cell.
            # Requiring a number there is what produced the old bogus "51.5 %".
            continue
        checked += 1
        assert f"{served:.1f} %" in text, (
            f"the measured {crop} boundary ({served:.1f} %) is not quoted in the "
            "docs; re-run scripts/decision_boundaries.py and update them"
        )
    assert checked >= 2, (
        "the boundary check skipped every crop, so it is not really checking "
        "anything; the window or the contexts probably moved"
    )


def test_a_history_context_is_verified_before_its_boundary_is_trusted():
    """The boundary script must prove each label against the real feature builder.

    `soil_moisture_change` and `soil_moisture_avg_30min` are derived by the
    service, not taken from the request. A script that labels a context
    `change=-2` while building a history the service reads as `change=0` prints a
    confident number describing a situation that was never tested. The script
    raises instead; this checks the raising still happens when the arithmetic is
    wrong, which is the only way to know it is not dead code.
    """
    from scripts import decision_boundaries

    model = IrrigationModel(model_path=IRRIGATION_MODEL_PATH)

    # a context the service cannot produce, so make_asker must refuse it
    with pytest.raises(AssertionError):
        decision_boundaries.make_asker(model, "tomato", 35.0, -2.0, 9999.0)

    # the real context must build cleanly
    ask = decision_boundaries.make_asker(model, "tomato", 35.0, -2.0, 60.0)
    result = ask(20.8)
    assert result["history_used"] is True, (
        "the verified context reported placeholders, so the verification is "
        "checking a different request than the one being swept"
    )


def test_feasible_window_excludes_contexts_that_cannot_exist():
    """The sweep must not fabricate a history to reach a number.

    Holding the derived features fixed forces both prior readings to track the
    current one, so outside a window around the context the synthesised history
    would claim impossible soil moisture. `feasible_window` bounds the sweep; this
    checks the bound is real, since a too-wide window is how a plausible number
    gets attached to an absurd history.
    """
    from scripts import decision_boundaries

    lo, hi = decision_boundaries.feasible_window(35.0, -2.0)
    assert 0.0 <= lo < hi <= 100.0
    # every point inside the window yields a history the contract accepts
    for moisture in (lo, (lo + hi) / 2, hi):
        history = decision_boundaries._history_for(moisture, 35.0, -2.0)
        for reading in history:
            assert 0.0 <= reading["soil_moisture"] <= 100.0, (
                f"moisture {moisture:g} would need an impossible prior reading "
                f"{reading['soil_moisture']:g}"
            )
    # and just outside it, the context breaks down
    assert decision_boundaries._history_for(hi + 1.0, 35.0, -2.0)[0]["soil_moisture"] < 0.0
    # no-history contexts are unconstrained
    assert decision_boundaries.feasible_window(35.0, -2.0) != (0.0, 100.0)


def test_crop_metrics_are_the_current_ones():
    """Guard against pasting a previous run's crop numbers back into the docs."""
    with open(CROP_HEALTH_META_PATH, encoding="utf-8") as handle:
        crop = json.load(handle)
    text = _doc_text()

    accuracy = crop["test_metrics"]["accuracy"]
    balanced = crop["test_metrics"]["balanced_accuracy"]
    assert f"{accuracy:.4f}" in text, f"crop accuracy {accuracy} is not documented anywhere"
    assert f"{balanced:.4f}" in text, f"crop balanced accuracy {balanced} is not documented"

    # these two are from a superseded run (the last-epoch checkpoint bug)
    assert "0.9634" not in text
    assert "0.9703" not in text


def test_the_irrigation_operating_point_is_documented_everywhere_it_matters():
    with open(IRRIGATION_MODEL_PATH, encoding="utf-8") as handle:
        irrigation = json.load(handle)
    text = _doc_text()
    accuracy = irrigation["test_metrics"]["accuracy"]
    balanced = irrigation["test_metrics"]["balanced_accuracy"]
    threshold = irrigation["decision_threshold"]

    assert f"{accuracy:.4f}" in text
    assert f"{balanced:.4f}" in text
    assert f"{threshold:.2f}" in text
    # no doc may still tell the backend to re-threshold at 0.50
    assert ">= 0.50" not in text
    assert ">= 0.5;" not in text


@pytest.mark.parametrize("value,what", [
    ("0.6879", "crop majority-class baseline"),
])
def test_derived_numbers_quoted_in_the_docs_are_present(value, what):
    """Numbers the docs derive themselves must still be quoted.

    The per-crop irrigation triggers used to sit in this list as hard-coded
    strings. They no longer do: `test_the_documented_irrigation_boundaries_match_the_model`
    measures them from the shipped model instead, so a retrain cannot leave a
    stale trigger behind and this file does not need editing every time.
    """
    assert value in _doc_text(), f"the {what} ({value}) is not documented anywhere"


def test_the_documented_moisture_example_is_inside_the_training_range():
    """The canonical doc payload must not demonstrate extrapolation.

    An earlier revision used soil_moisture 14.5 for a tomato plot, which is below
    where events occur, so the example showed the model extrapolating rather than
    behaving.
    """
    from agriguard.config import IRRIGATION_CSV
    from agriguard.preprocess import load_irrigation_csv

    frame, _ = load_irrigation_csv(IRRIGATION_CSV)
    events = frame[frame["irrigation_event"] == 1]
    event_min = float(events["soil_moisture"].min())
    event_max = float(events["soil_moisture"].max())

    # the value used in README.md, docs/api_contract.md, docs/integration_guide.md
    documented = 20.0
    assert event_min <= documented <= event_max, (
        f"the documented example {documented} is outside the event range "
        f"[{event_min}, {event_max}]"
    )
