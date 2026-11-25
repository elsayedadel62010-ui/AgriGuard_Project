"""Crop-health classification: Healthy vs Possible Disease.

WHAT THIS MODEL IS
------------------
A MobileNetV3-Small classifier fine-tuned on public PlantVillage leaf images,
reduced to two categories:

    Healthy
    Possible Disease

The correct description of what it does:

    The model classifies images into Healthy or Possible Disease within the
    scope of the training dataset. It is a prototype screening/classification
    model and is not a universal disease diagnostic system. It never names a
    specific disease and it must never be used to spray or to skip a field
    inspection.

The colour heuristic in `heuristic_predict()` is kept ONLY as a documented
non-ML baseline for comparison. The API calls the CNN; if the CNN weights are
missing the endpoint answers 503 instead of silently degrading to the baseline.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Union

from .config import CROP_HEALTH_META_PATH, CROP_HEALTH_WEIGHTS_PATH
from .cropdata import API_LABELS, CATEGORY_DISEASE, CATEGORY_HEALTHY, CLASS_NAMES
from .imageutils import InvalidImageError, load_image_rgb, preprocess_for_inference
from .version import CROP_HEALTH_MODEL_VERSION

CLASS_HEALTHY = CATEGORY_HEALTHY
CLASS_DISEASE = CATEGORY_DISEASE

HEURISTIC_METHOD = "color-heuristic-baseline"


class ModelNotAvailableError(RuntimeError):
    """Raised when the trained crop-health weights cannot be loaded."""


class CropHealthModel:
    """Loads the fine-tuned MobileNetV3-Small and classifies leaf images."""

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        meta_path: Optional[Union[str, Path]] = None,
    ):
        self.weights_path = Path(weights_path or CROP_HEALTH_WEIGHTS_PATH)
        self.meta_path = Path(meta_path or CROP_HEALTH_META_PATH)
        self.net = None
        self.metadata: Dict[str, object] = {}
        self.load_error: Optional[str] = None
        self.method = "MobileNetV3-Small"
        self._loaded = False
        if self.weights_path.exists() and self.meta_path.exists():
            self._load()

    # -- loading ---------------------------------------------------------
    def _load(self) -> None:
        try:
            import torch
            from torch import nn
            from torchvision import models
        except ImportError as exc:
            self.load_error = (
                "PyTorch/torchvision are not installed; run "
                "`pip install -r requirements.txt`"
            )
            return

        try:
            with open(self.meta_path, "r", encoding="utf-8") as handle:
                meta = json.load(handle)
            self.metadata = meta
            self.method = meta.get("method", "MobileNetV3-Small")
            classes = meta.get("classes", list(CLASS_NAMES))
            if tuple(classes) != tuple(CLASS_NAMES):
                raise ValueError(f"unexpected class order in metadata: {classes}")

            net = models.mobilenet_v3_small(weights=None)
            in_features = net.classifier[3].in_features
            net.classifier[3] = nn.Linear(in_features, len(CLASS_NAMES))
            state = torch.load(self.weights_path, map_location="cpu", weights_only=True)
            net.load_state_dict(state)
            net.eval()
            self.net = net
            self._loaded = True
            self.load_error = None
        except Exception as exc:  # noqa: BLE001 - surfaced through /v1/health
            self.net = None
            self.load_error = f"could not load {self.weights_path.name}: {exc}"

    @property
    def is_trained(self) -> bool:
        return self._loaded

    @property
    def model_version(self) -> str:
        return str(self.metadata.get("model_version", CROP_HEALTH_MODEL_VERSION))

    def describe(self) -> Dict[str, object]:
        return {
            "trained": self._loaded,
            "method": self.method,
            "model_version": self.model_version,
            "classes": list(CLASS_NAMES),
            "trained_at": self.metadata.get("trained_at"),
            "training_dataset": self.metadata.get("training_dataset"),
            "test_metrics": self.metadata.get("test_metrics"),
            "load_error": self.load_error,
        }

    # -- inference -------------------------------------------------------
    def predict(self, source) -> Dict[str, object]:
        """Classify an image path, bytes or PIL image.

        Output keys match the API contract:
        class, label, confidence, method, model_version
        """
        if not self._loaded:
            raise ModelNotAvailableError(
                self.load_error
                or f"crop-health weights not found at {self.weights_path.name}"
            )
        import torch

        image = load_image_rgb(source)
        tensor = preprocess_for_inference(image)
        with torch.no_grad():
            logits = self.net(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]

        index = int(torch.argmax(probabilities).item())
        category = CLASS_NAMES[index]
        score = float(probabilities[index].item())
        return {
            "class": category,
            "label": API_LABELS[category],
            "confidence": round(score, 4),
            "method": self.method,
            "model_version": self.model_version,
        }


# ---------------------------------------------------------------------------
# documented non-ML baseline, kept for comparison only
# ---------------------------------------------------------------------------
def heuristic_predict(source) -> Dict[str, object]:
    """Colour-fraction baseline: green vs yellowing/necrotic pixels.

    This is NOT the production model. It exists so the project can show a
    before/after comparison (see docs/crop_health_results.md). It is never
    called by the API.
    """
    import numpy as np

    image = load_image_rgb(source).resize((320, 320))
    arr = np.asarray(image).astype(np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    total = arr.shape[0] * arr.shape[1]

    healthy_green = (g > r + 0.06) & (g > b + 0.06) & (g > 0.28)
    yellowing = (r > g + 0.03) & (g > b + 0.04) & (r > 0.35)
    dark_brown = (r > g * 1.45) & (g * 1.45 >= b) & (r < 0.5)
    background = (np.max(arr, axis=-1) < 0.12) | (
        (np.abs(r - g) < 0.06) & (np.abs(g - b) < 0.06) & (r < 0.35)
    )

    n_healthy = float(np.sum(healthy_green))
    n_damaged = float(
        np.sum(yellowing) + np.sum(dark_brown) - float(np.sum(background & healthy_green))
    )
    healthy_fraction = n_healthy / total
    damaged_fraction = float(np.clip(n_damaged / total, 0.0, None))
    score = healthy_fraction / (healthy_fraction + damaged_fraction + 1e-6)

    category = CLASS_HEALTHY if score >= 0.6 else CLASS_DISEASE
    return {
        "class": category,
        "label": API_LABELS[category],
        "confidence": round(float(min(0.99, 0.5 + 0.5 * abs(score - 0.5))), 3),
        "method": HEURISTIC_METHOD,
        "metrics": {
            "healthy_fraction": round(healthy_fraction, 3),
            "damaged_fraction": round(damaged_fraction, 3),
        },
    }


#: Module-level singleton used by the API.
crop_health_model = CropHealthModel()


def predict_image_bytes(data: bytes) -> Dict[str, object]:
    """Convenience wrapper used by the API. Raises InvalidImageError / ModelNotAvailableError."""
    if not data:
        raise InvalidImageError("empty upload")
    return crop_health_model.predict(data)


__all__ = [
    "CLASS_DISEASE",
    "CLASS_HEALTHY",
    "CropHealthModel",
    "InvalidImageError",
    "ModelNotAvailableError",
    "crop_health_model",
    "heuristic_predict",
    "predict_image_bytes",
]
