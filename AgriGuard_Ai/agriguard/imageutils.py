"""Shared image preprocessing for the crop-health model.

Training, validation, testing and live inference ALL import the constants and
functions in this file, so the network can never be fed a differently resized or
differently normalised image at inference time (TASK 8 of the brief).

The pipeline is the standard torchvision MobileNetV3 transfer-learning recipe:

    RGB  ->  resize to IMAGE_SIZE x IMAGE_SIZE  ->  scale to 0..1
         ->  normalise with IMAGENET_MEAN / IMAGENET_STD

Augmentation (random crop / flip / rotation / colour jitter) is applied for
training only and is never used for validation, test or inference.
"""

from io import BytesIO
from typing import Optional, Tuple

from PIL import Image, ImageFile, UnidentifiedImageError

#: Be forgiving with slightly truncated JPEGs - an ESP32-CAM stream is not
#: always a perfectly finished file. Genuinely broken files still raise.
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SUPPORTED_FORMATS = ("JPEG", "PNG", "BMP", "WEBP", "TIFF")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


class InvalidImageError(ValueError):
    """Raised when an upload is not a decodable image."""


def _require_torch():
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "PyTorch is required for crop-health inference. Install it with:\n"
            "  pip install -r requirements.txt"
        ) from exc


def build_transforms(train: bool = False):
    """Return the torchvision transform for `train` or for evaluation/inference."""
    _require_torch()
    from torchvision import transforms

    if train:
        return transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomResizedCrop(
                    IMAGE_SIZE, scale=(0.75, 1.0), ratio=(0.85, 1.15)
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(degrees=25),
                transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return build_eval_transform()


def build_eval_transform():
    """Deterministic preprocessing shared by validation, test and inference."""
    _require_torch()
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_image_rgb(source) -> Image.Image:
    """Open a path / file object / bytes and return a validated RGB image."""
    if isinstance(source, (bytes, bytearray)):
        if not source:
            raise InvalidImageError("empty image payload")
        if len(source) > MAX_UPLOAD_BYTES:
            raise InvalidImageError("image larger than 10 MB")
        try:
            img = Image.open(BytesIO(bytes(source)))
            img.load()
        except (UnidentifiedImageError, OSError) as exc:
            raise InvalidImageError("file is not a decodable image") from exc
    elif isinstance(source, Image.Image):
        img = source.copy()
    else:
        try:
            img = Image.open(source)
            img.load()
        except (UnidentifiedImageError, OSError) as exc:
            raise InvalidImageError("file is not a decodable image") from exc

    if img.format is not None and img.format not in SUPPORTED_FORMATS:
        raise InvalidImageError(f"unsupported image format: {img.format}")
    return img.convert("RGB")


def preprocess_for_inference(img: Image.Image):
    """Apply the exact evaluation preprocessing and return a batched tensor."""
    _require_torch()
    tensor = build_eval_transform()(img)
    return tensor.unsqueeze(0)


def image_bytes_for(path) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()
