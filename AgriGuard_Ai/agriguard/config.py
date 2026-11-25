"""Shared filesystem locations for AgriGuard AI.

One place so training scripts, the API and the tests always agree on where the
CSV dataset and the serialized models live.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
IRRIGATION_CSV = DATA_DIR / "irrigation_data.csv"
CROP_HEALTH_DIR = DATA_DIR / "crop_health"

MODELS_DIR = PROJECT_ROOT / "models"
IRRIGATION_MODEL_PATH = MODELS_DIR / "irrigation_model.json"
CROP_HEALTH_WEIGHTS_PATH = MODELS_DIR / "crop_health_model.pth"
CROP_HEALTH_META_PATH = MODELS_DIR / "crop_health_model.json"

EVIDENCE_DIR = PROJECT_ROOT / "evidence"
EXAMPLES_DIR = PROJECT_ROOT / "examples"
