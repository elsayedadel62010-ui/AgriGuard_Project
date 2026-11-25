"""Crop-health dataset provenance and class mapping.

Kept separate from agriguard.crophealth so the training code, the inference
code and the dataset-preparation script all agree on the same two categories
and the same public source.

The dataset *location* is not defined here. It lives in agriguard.config as a
Path derived from PROJECT_ROOT, so that nothing depends on the current working
directory. This module used to carry its own CROP_HEALTH_DIR = "data/crop_health"
string, which shadowed the real one in two scripts and made them break.
"""

#: Hugging Face parquet mirror of the PlantVillage *color* image set
#: (Mohanty et al., 2016 - "PlantVillage: A dataset for plant disease detection").
#: Only the specific parquet shards we need are fetched, and only the row groups
#: that contain the images we keep (HTTP range requests).
PLANTVILLAGE_URL = "https://huggingface.co/datasets/GVJahnavi/PlantVillage_dataset"

PLANTVILLAGE_FILES = (
    f"{PLANTVILLAGE_URL}/resolve/main/data/test-00000-of-00001.parquet",
    f"{PLANTVILLAGE_URL}/resolve/main/data/train-00000-of-00002.parquet",
    f"{PLANTVILLAGE_URL}/resolve/main/data/train-00001-of-00002.parquet",
)

PLANTVILLAGE_REPO = "https://huggingface.co/datasets/GVJahnavi/PlantVillage_dataset"
PLANTVILLAGE_PAPER = "https://arxiv.org/abs/1511.08060"
PLANTVILLAGE_ORIGINAL = "https://github.com/spMohanty/PlantVillage-Dataset"

CATEGORY_HEALTHY = "healthy"
CATEGORY_DISEASE = "possible_disease"

#: Directory name -> label string that the API is allowed to return.
API_LABELS = {
    CATEGORY_HEALTHY: "Healthy",
    CATEGORY_DISEASE: "Possible Disease",
}

#: Integer order used everywhere (folder name, class index, metadata json).
#:
#: Deliberately TWO classes. "Is this a plant at all?" is a separate question
#: with a separate answer, and it is answered by the plant detector in front of
#: this model rather than by adding a third output here. See
#: agriguard/plantdetect.py for why a gate beats a third class.
#:
#: Order is load-bearing: `model.get(...)["class_index"]` is an index into this
#: tuple and the saved `.pth` carries weights whose output columns are in this
#: order. Reordering invalidates an existing weights file, and crophealth.py
#: rejects a metadata file whose `classes` list disagrees with this tuple
#: rather than silently mis-scoring. Never reorder; retrain instead.
CLASS_NAMES = (CATEGORY_HEALTHY, CATEGORY_DISEASE)
