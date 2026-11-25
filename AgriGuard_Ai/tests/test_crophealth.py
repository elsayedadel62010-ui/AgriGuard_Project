"""Tests for the crop-health classifier.

IMPORTANT: the model-evaluation tests here use REAL PlantVillage images from the
*test split* - images the network never saw during training. Artificial
solid-colour images are used only where the point of the test is the API
plumbing (accepting bytes, rejecting junk), and those are labelled as such.
"""

import io
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

from agriguard.config import (
    CROP_HEALTH_DIR,
    CROP_HEALTH_META_PATH,
    CROP_HEALTH_WEIGHTS_PATH,
    PROJECT_ROOT,
)
from agriguard.cropdata import API_LABELS, CLASS_NAMES
from agriguard.crophealth import (
    CropHealthModel,
    ModelNotAvailableError,
    crop_health_model,
    heuristic_predict,
)
from agriguard.imageutils import (
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    InvalidImageError,
    build_eval_transform,
    build_transforms,
    load_image_rgb,
    preprocess_for_inference,
)


def solid_png_bytes(colour=(40, 180, 60), size=(64, 64)):
    """Synthetic solid-colour image - API plumbing only, NOT a model evaluation."""
    buffer = io.BytesIO()
    Image.new("RGB", size, colour).save(buffer, format="PNG")
    return buffer.getvalue()


# ------------------------------------------------------------- class contract
def test_two_classes_only():
    assert CLASS_NAMES == ("healthy", "possible_disease")
    assert API_LABELS["healthy"] == "Healthy"
    assert API_LABELS["possible_disease"] == "Possible Disease"


def test_api_labels_use_the_agreed_spelling():
    assert set(API_LABELS.values()) == {"Healthy", "Possible Disease"}


# ------------------------------------------------------------- preprocessing
def test_training_and_inference_preprocessing_are_identical():
    """A common bug is a different resize or normalisation at inference time.

    The two transform objects are built separately, so identity comparison is
    meaningless. Compare what they actually produce on the same image instead -
    that is the property that matters.
    """
    image = load_image_rgb(solid_png_bytes(size=(97, 61)))
    during_eval = build_transforms(train=False)(image)
    at_inference = build_eval_transform()(image)
    assert torch.allclose(during_eval, at_inference)
    assert during_eval.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert IMAGE_SIZE == 224


def test_eval_transform_has_no_random_augmentation():
    names = [type(step).__name__ for step in build_transforms(train=False).transforms]
    assert "RandomResizedCrop" not in names
    assert "RandomHorizontalFlip" not in names
    assert "RandomVerticalFlip" not in names
    assert "RandomRotation" not in names
    assert "ColorJitter" not in names


def test_inference_tensor_shape_and_normalisation():
    image = load_image_rgb(solid_png_bytes())
    tensor = preprocess_for_inference(image)
    assert tensor.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)
    # values sit around 0 after ImageNet normalisation, not inside [0, 1]
    assert float(tensor.max()) > 1.0 or float(tensor.min()) < -1.0
    assert IMAGENET_MEAN and IMAGENET_STD


def test_augmentation_is_only_used_for_training():
    train_names = [type(t).__name__ for t in build_transforms(train=True).transforms]
    eval_names = [type(t).__name__ for t in build_transforms(train=False).transforms]
    assert "RandomResizedCrop" in train_names
    assert "RandomResizedCrop" not in eval_names
    assert "RandomHorizontalFlip" in train_names
    assert "RandomHorizontalFlip" not in eval_names


# ------------------------------------------------------------- bad input
def test_empty_payload_is_rejected():
    with pytest.raises(InvalidImageError):
        load_image_rgb(b"")


def test_corrupted_image_is_rejected():
    with pytest.raises(InvalidImageError):
        load_image_rgb(b"\x89PNG\r\n\x1a\n" + b"garbage" * 50)


def test_text_file_is_not_an_image():
    with pytest.raises(InvalidImageError):
        load_image_bytes_if_available(b"this is definitely not an image")


def load_image_bytes_if_available(data):
    return load_image_rgb(data)


# ------------------------------------------------------------- model loading
def test_weights_file_exists(crop_health_trained):
    assert CROP_HEALTH_WEIGHTS_PATH.exists()
    assert CROP_HEALTH_META_PATH.exists()


def test_metadata_records_versioning(crop_health_trained):
    with open(CROP_HEALTH_META_PATH, encoding="utf-8") as handle:
        meta = json.load(handle)
    for key in (
        "model_version",
        "trained_at",
        "training_dataset",
        "classes",
        "features",
        "image_size",
        "batch_size",
        "epochs",
        "test_metrics",
        "limitations",
    ):
        assert key in meta
    assert meta["classes"] == ["healthy", "possible_disease"]
    assert meta["method"] == "MobileNetV3-Small"


def test_recorded_input_features_match_the_real_transform(crop_health_trained):
    """The `features` block must describe the transform, not a remembered one.

    The brief requires the model file to document its features. For a CNN that
    is the input contract, so it is checked against agriguard.imageutils
    directly: a stale size or normalisation here would tell an integrator to
    feed the network something it was never trained on.
    """
    from agriguard.imageutils import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD

    with open(CROP_HEALTH_META_PATH, encoding="utf-8") as handle:
        features = json.load(handle)["features"]

    assert features["input_type"] == "image"
    assert features["image_size"] == [IMAGE_SIZE, IMAGE_SIZE]
    assert features["normalisation_mean"] == list(IMAGENET_MEAN)
    assert features["normalisation_std"] == list(IMAGENET_STD)
    assert features["scale"] == "0..1"


def test_dataset_path_is_defined_once_and_is_an_absolute_anchor():
    """One canonical CROP_HEALTH_DIR, and it is a Path.

    agriguard.cropdata used to define its own CROP_HEALTH_DIR = "data/crop_health"
    string, which shadowed the real one in agriguard.config. Two scripts imported
    the string, so they broke on a path join and silently dropped their crop
    results. Pin the invariant so the duplicate cannot come back.
    """
    from agriguard import cropdata
    from agriguard.config import CROP_HEALTH_DIR as canonical

    assert isinstance(canonical, Path)
    assert not hasattr(cropdata, "CROP_HEALTH_DIR"), (
        "agriguard.cropdata must not define CROP_HEALTH_DIR; "
        "import it from agriguard.config instead"
    )
    assert canonical.is_absolute()
    assert canonical.name == "crop_health"
    assert (canonical / "splits.csv").exists()


def test_evaluate_models_can_actually_run(crop_health_trained, tmp_path):
    """scripts/evaluate_models.py must not crash on the crop branch.

    It used to die with a TypeError on a str/Path mismatch, which made the
    evidence bundle quietly lose the crop confusion matrix. Run it for real into
    a temp file; it is the artifact the documentation is generated from.
    """
    out = tmp_path / "evaluation_summary.json"
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "evaluate_models.py"),
         "--out", str(out)],
        cwd=str(tmp_path),           # deliberately NOT the project root
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    summary = json.loads(out.read_text(encoding="utf-8"))
    assert summary["crop_health"]["test_samples"] == 846
    assert math.isclose(summary["crop_health"]["balanced_accuracy"], 0.9668, abs_tol=5e-4)

    # the irrigation branch must evaluate at the threshold the model ships, and
    # at the metrics that threshold produces. The model was retrained to consume
    # temperature/humidity/light as well as soil moisture, so these numbers are
    # not the pre-switch ones.
    irrigation = summary["irrigation"]
    assert irrigation["decision_threshold"] == 0.35
    assert math.isclose(irrigation["accuracy"], 0.9406, abs_tol=5e-4)
    assert math.isclose(irrigation["balanced_accuracy"], 0.8144, abs_tol=5e-4)
    assert irrigation["accuracy"] > 0.9


def test_the_saved_weights_come_from_the_best_validation_epoch(crop_health_trained):
    """The checkpoint must be the best epoch, not the last one.

    Regression test: an indentation bug meant the stage-2 loop only compared the
    final epoch against the best, so the last epoch's weights were saved and
    reported as the best. Here the validation peak (epoch 3, 0.9823) sits in the
    middle of the run, and epochs 4-6 all score lower, so a "last epoch wins"
    script would save the wrong weights and the stored
    best_validation_accuracy would disagree with the history.
    """
    with open(CROP_HEALTH_META_PATH, encoding="utf-8") as handle:
        meta = json.load(handle)

    history = meta["history"]
    assert len(history) == meta["epochs"]
    val = [epoch["val_accuracy"] for epoch in history]
    best = meta["best_validation_accuracy"]

    assert math.isclose(best, max(val)), (
        f"stored best {best} is not the best epoch in the history {val}; "
        "the checkpoint selection is wrong"
    )
    # the peak must not be the final epoch, or this test proves nothing
    assert val.index(max(val)) != len(val) - 1
    assert val.index(max(val)) + 1 == 3

    # the stored test metrics must be a balanced, per-class breakdown
    test = meta["test_metrics"]
    matrix = test["confusion_matrix"]
    assert matrix["values"] == [[256, 8], [21, 561]]
    assert (matrix["tn"], matrix["fp"], matrix["fn"], matrix["tp"]) == (256, 8, 21, 561)
    assert math.isclose(test["accuracy"], 0.9657, abs_tol=5e-4)
    assert math.isclose(test["balanced_accuracy"], 0.9668, abs_tol=5e-4)
    # balanced accuracy has to be the mean of the two recalls, not the accuracy
    assert math.isclose(
        test["balanced_accuracy"],
        (test["recall_per_class"]["Healthy"] + test["recall_per_class"]["Possible Disease"]) / 2,
        abs_tol=1e-3,
    )


def test_model_reports_itself_as_mobilenet(crop_health_trained):
    assert crop_health_model.method == "MobileNetV3-Small"
    assert crop_health_model.is_trained is True
    assert crop_health_model.model_version == "0.2.0"


def test_missing_weights_raise_instead_of_silently_degrading(tmp_path):
    model = CropHealthModel(
        weights_path=tmp_path / "nope.pth", meta_path=tmp_path / "nope.json"
    )
    assert model.is_trained is False
    with pytest.raises(ModelNotAvailableError):
        model.predict(solid_png_bytes())


def test_corrupt_weights_file_is_reported(tmp_path):
    weights = tmp_path / "crop_health_model.pth"
    weights.write_bytes(b"not a torch checkpoint")
    meta = tmp_path / "crop_health_model.json"
    meta.write_text(json.dumps({"classes": list(CLASS_NAMES)}), encoding="utf-8")
    model = CropHealthModel(weights_path=weights, meta_path=meta)
    assert model.is_trained is False
    assert model.load_error


# ------------------------------------------------------------- real inference
def test_real_healthy_test_images_are_classified(crop_health_trained, real_healthy_images):
    results = [crop_health_model.predict(path) for path in real_healthy_images]
    for result in results:
        assert result["class"] in CLASS_NAMES
        assert result["label"] in {"Healthy", "Possible Disease"}
        assert 0.0 <= result["confidence"] <= 1.0
        assert result["method"] == "MobileNetV3-Small"
        assert result["model_version"] == "0.2.0"
    labels = [r["label"] for r in results]
    assert labels.count("Healthy") >= 3, f"expected mostly Healthy, got {labels}"


def test_real_disease_test_images_are_classified(crop_health_trained, real_disease_images):
    results = [crop_health_model.predict(path) for path in real_disease_images]
    for result in results:
        assert result["class"] in CLASS_NAMES
        assert 0.0 <= result["confidence"] <= 1.0
    labels = [r["label"] for r in results]
    assert labels.count("Possible Disease") >= 3, f"expected mostly disease, got {labels}"


def test_real_image_accuracy_on_a_handful_of_test_images(
    crop_health_trained, real_healthy_images, real_disease_images
):
    """A quick smoke check, not the headline number.

    The authoritative metrics live in models/crop_health_model.json and
    docs/crop_health_results.md, computed over the whole unseen test split.
    """
    correct = 0
    total = 0
    for paths, expected in (
        (real_healthy_images, "healthy"),
        (real_disease_images, "possible_disease"),
    ):
        for path in paths:
            correct += int(crop_health_model.predict(path)["class"] == expected)
            total += 1
    assert correct / total >= 0.75


def test_bytes_and_path_paths_agree(crop_health_trained, real_healthy_images):
    path = real_healthy_images[0]
    from_path = crop_health_model.predict(path)
    from_bytes = crop_health_model.predict(path.read_bytes())
    assert from_path["class"] == from_bytes["class"]
    assert from_path["confidence"] == pytest.approx(from_bytes["confidence"], abs=1e-4)


def test_heuristic_baseline_is_not_the_api_model(real_healthy_images):
    """The colour heuristic stays available as a documented baseline only."""
    baseline = heuristic_predict(real_healthy_images[0])
    assert baseline["method"] == "color-heuristic-baseline"
    assert crop_health_model.describe()["method"] != "color-heuristic-baseline"


# ------------------------------------------------------------- dataset
def test_dataset_uses_the_required_folder_layout(manifest):
    assert manifest, "run scripts/prepare_crop_dataset.py first"
    for record in manifest:
        assert record["category"] in CLASS_NAMES
        assert (CROP_HEALTH_DIR / record["key"]).exists()


def test_split_is_reproducible_and_disjoint(manifest):
    train = {r["dup_group"] for r in manifest if r["split"] == "train"}
    validation = {r["dup_group"] for r in manifest if r["split"] == "validation"}
    test = {r["dup_group"] for r in manifest if r["split"] == "test"}
    assert not (train & validation)
    assert not (train & test)
    assert not (validation & test)


def test_every_image_belongs_to_exactly_one_split(manifest):
    splits = {r["split"] for r in manifest}
    assert splits == {"train", "validation", "test"}


def test_test_split_is_a_realistic_share(manifest):
    total = len(manifest)
    test = sum(1 for r in manifest if r["split"] == "test")
    assert 0.10 <= test / total <= 0.20
