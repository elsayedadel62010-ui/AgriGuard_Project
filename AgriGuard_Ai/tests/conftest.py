"""Shared pytest fixtures for AgriGuard AI.

The crop-health tests need real images. They are taken from the *test split* of
data/crop_health - images the CNN never saw during training. If the dataset has
not been prepared yet, those tests are skipped with an explicit reason instead
of silently passing.
"""

import csv
import random
from pathlib import Path

import pytest

from agriguard.config import CROP_HEALTH_DIR, CROP_HEALTH_WEIGHTS_PATH
from agriguard.crophealth import crop_health_model
from agriguard.irrigation import irrigation_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_manifest():
    manifest = CROP_HEALTH_DIR / "splits.csv"
    if not manifest.exists():
        return []
    with open(manifest, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


MANIFEST = _load_manifest()


def _pick(category, split="test", count=4):
    rows = [
        r
        for r in MANIFEST
        if r["category"] == category and r["split"] == split
    ]
    rows.sort(key=lambda r: r["key"])
    random.Random(0).shuffle(rows)
    return [CROP_HEALTH_DIR / r["key"] for r in rows[:count]]


@pytest.fixture(scope="session")
def manifest():
    return MANIFEST


@pytest.fixture(scope="session")
def real_healthy_images():
    paths = _pick("healthy")
    if not paths:
        pytest.skip("crop-health dataset not prepared; run prepare_crop_dataset.py")
    return paths


@pytest.fixture(scope="session")
def real_disease_images():
    paths = _pick("possible_disease")
    if not paths:
        pytest.skip("crop-health dataset not prepared; run prepare_crop_dataset.py")
    return paths


@pytest.fixture(scope="session")
def irrigation_trained():
    return irrigation_model.is_trained


@pytest.fixture(scope="session")
def crop_health_trained():
    if not CROP_HEALTH_WEIGHTS_PATH.exists():
        pytest.skip("crop-health weights not trained; run train_crophealth.py")
    if not crop_health_model.is_trained:
        pytest.skip(f"crop-health weights failed to load: {crop_health_model.load_error}")
    return crop_health_model
