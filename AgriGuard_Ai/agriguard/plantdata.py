"""Provenance and class definitions for the plant detector (stage 1 of 2).

Split out from `cropdata.py` on purpose. The crop-health model and the plant
detector answer different questions, come from different sources, and have
different failure modes. Keeping them in one module is how "is it a plant?" and
"is it healthy?" quietly grow into one tangled three-class head.

Why a gate instead of a third class
------------------------------------
The obvious alternative is one model with three outputs: Healthy,
Possible Disease, Not A Plant. It was tried first and rejected, for a reason
that is worth writing down because it is easy to argue the other way:

A third class shares the final softmax across all three, so adding it *dilutes*
the two decisions the dashboard actually renders. The healthy-vs-disease
boundary that currently stands at 0.9657 test accuracy would move, on images
that contain no non-plant ambiguity at all, purely because some of the softmax
mass is now reserved for a category the plant image could never belong to. A
regression in a number the project reports is a bad trade for a gate that can
be measured on its own.

The gate also lets each stage be evaluated honestly, which is the point of
building it separately: plant-vs-non-plant detection and healthy-vs-disease
classification have different class balances, different error costs and
different thresholds, and folding them into one accuracy number hides both.

The cost is honest and worth stating: two forward passes per uploaded image,
and a detector that can be wrong in a new way. The threshold and the abstain
state exist because of that second cost.

Sources
-------
Positives (plant leaves) are the same PlantVillage images the crop-health model
already uses, on the same train/validation/test split, so the two stages are
measured on identical plant photographs and a stage-1 error is attributable to
the detector rather than to a difference in data.

Negatives (not a plant) are real photographs from a public mirror of 100
ImageNet-1k classes. Its labels are read only to choose which of its classes are
non-plants. Nothing here is a AgriGuard class.
"""

from pathlib import Path

#: Stage-1 label order. Fixed and load-bearing: the output column order of the
#: saved detector weights.
CLASS_PLANT = "plant"
CLASS_NONPLANT = "nonplant"
DETECTOR_CLASS_NAMES = (CLASS_PLANT, CLASS_NONPLANT)

#: Index of "is it a plant" in the detector's softmax. 0 = plant, 1 = non-plant.
#: The API reports the *plant* probability, so this is the complement.
NONPLANT_INDEX = 1

#: Public, ungated mirror of 100 ImageNet-1k classes on the Hugging Face Hub.
#: Ungated matters: a gated dataset would put a token requirement in front of
#: rebuilding the project from a clean machine.
IMAGENET100_URL = (
    "https://huggingface.co/datasets/clane9/imagenet-100/resolve/main/"
    "data/validation-00000-of-00001.parquet"
)
IMAGENET100_REPO = "https://huggingface.co/datasets/clane9/imagenet-100"
IMAGENET100_CITATION = (
    "ImageNet-1k (Deng et al., 2009), 100-class subset redistributed at "
    f"{IMAGENET100_REPO}"
)

PLANTVILLAGE_REPO = "https://huggingface.co/datasets/GVJahnavi/PlantVillage_dataset"

#: ImageNet-100 classes deliberately excluded from the negatives, each with the
#: reason. Every one of these is a real plant, plant part or food item, so
#: including one would teach the detector that edible things are not plants -
#: true, and useless for this project.
EXCLUDED_SOURCE_CLASSES = {
    "gyromitra": "fungus, and a food item",
    "stinkhorn, carrion fungus": "fungus",
    "head cabbage": "plant",
    "cauliflower": "plant",
    "pineapple, ananas": "plant part and food",
    "carbonara": "food",
    "chocolate sauce, chocolate syrup": "food",
}

#: Fraction of the non-plant SOURCE CLASSES reserved for each split.
#:
#: The split is over source classes, not over images. This is the single most
#: important decision in the whole feature: it means the test set contains
#: categories of non-plant the detector has never seen a single photograph of,
#: which is the only way to answer "does this generalise to non-plants it was
#: not trained on" rather than the much weaker "does it generalise to new
#: photographs of the same non-plants it was trained on".
#:
#: The cost is a smaller test set and a pessimistic detector, both of which are
#: the correct direction: a robustness claim that survives unseen categories is
#: worth more than a higher number from seen ones.
SOURCE_CLASS_SPLIT_FRACTIONS = {"train": 0.60, "validation": 0.20, "test": 0.20}
SOURCE_CLASS_SPLIT_SEED = 20260930


def detector_dataset_dir(project_root: Path) -> Path:
    """Where stage-1 images live.

    Separate from `data/crop_health/` on purpose. The two stages are separate
    models with separate metrics, and mixing their images in one directory is
    how a test split quietly ends up training on the other stage's data.
    """
    return project_root / "data" / "plant_detector"


def non_plant_source_class_names() -> tuple:
    """Sort key helper kept next to the exclusion list it must stay in sync with."""
    return tuple(sorted(EXCLUDED_SOURCE_CLASSES))
