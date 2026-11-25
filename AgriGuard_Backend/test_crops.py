"""The backend's crop vocabulary must match the AI service's.

`crops.py` duplicates the trained/rule-only split so the backend does not have
to import the AI package. That duplication is only safe if something checks it,
so this test reads the shipped model and asserts the two agree. A retrain that
changes the trained crops fails here instead of letting the backend route a
crop to the model that the model cannot actually answer for.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import crops  # noqa: E402

AI_ROOT = Path(__file__).resolve().parents[1] / "AgriGuard_Ai"
MODEL_PATH = AI_ROOT / "models" / "irrigation_model.json"

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="AI service model file not present in this checkout",
)


def _model_crops():
    with open(MODEL_PATH, encoding="utf-8") as handle:
        features = json.load(handle)["features"]
    return {
        column[len("crop_is_"):]
        for column in features
        if column.startswith("crop_is_")
    }


def test_trained_crops_match_the_shipped_model():
    assert set(crops.TRAINED_CROPS) == _model_crops()


def test_the_two_crop_sets_do_not_overlap():
    assert not set(crops.TRAINED_CROPS) & set(crops.RULE_ONLY_CROPS)


def test_every_known_crop_is_known():
    assert set(crops.KNOWN_CROPS) == set(crops.TRAINED_CROPS) | set(
        crops.RULE_ONLY_CROPS
    )


def test_known_crops_match_the_ai_rule_threshold_table():
    """Every rule-only crop must have a threshold in the AI's own table."""
    sys.path.insert(0, str(AI_ROOT))
    from agriguard.irrigation import CROP_THRESHOLDS  # noqa: PLC0415

    for crop in crops.RULE_ONLY_CROPS:
        assert crop in CROP_THRESHOLDS, crop


def test_normalise_crop_handles_case_and_whitespace():
    assert crops.normalise_crop("  Tomato ") == "tomato"
    assert crops.normalise_crop("RICE") == "rice"
    assert crops.normalise_crop("   ") is None
    assert crops.normalise_crop(None) is None
    assert crops.is_known_crop("Tomato") is True
    assert crops.is_known_crop("cassava") is False
