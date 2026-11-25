# Crop-health image dataset

This document records exactly what data the crop-health model was trained on,
where it came from, how the two AgriGuard classes were formed, and what the
dataset cannot tell you. Nothing here is estimated - every count comes from
`data/crop_health/dataset_stats.json`, written by
`scripts/prepare_crop_dataset.py`.

## Required fields

| Field | Value |
| ----- | ----- |
| Dataset name | PlantVillage (color image set) |
| Dataset source | Public dataset, mirrored on Hugging Face as a parquet dataset |
| Source URL | https://huggingface.co/datasets/GVJahnavi/PlantVillage_dataset |
| Original dataset | https://github.com/spMohanty/PlantVillage-Dataset |
| Original paper | https://arxiv.org/abs/1511.08060 (Mohanty, Hughes & Salathe, 2016) |
| Crop/species | 14 crop species (see the class table below) |
| Classes | `Healthy`, `Possible Disease` |
| Healthy image count | 1,739 (30.8 %) |
| Possible Disease image count | 3,900 (69.2 %) |
| Total images | 5,639 |
| Original classes collapsed into the 2 categories | 38 (12 healthy + 26 disease) |
| Training count | 3,947 (70.0 %) - 1,220 healthy / 2,727 Possible Disease |
| Validation count | 846 (15.0 %) - 255 healthy / 591 Possible Disease |
| Test count | 846 (15.0 %) - 264 healthy / 582 Possible Disease |
| Near-duplicate images grouped | 10 of 5,639 (0.18 %), hamming distance <= 3 on a 64-bit dHash |
| Image preprocessing | resize to 224x224, ImageNet normalisation, JPEG re-encode at max side 256 |
| Known limitations | see the bottom of this file |
| License/usage notes | PlantVillage is distributed for research and educational use; check the upstream repository before any commercial use |

Everything above is reproducible with:

```bash
python scripts/prepare_crop_dataset.py --per-class 150
```

## Why this dataset

The project needs real photographs of real leaves, labelled into
`Healthy` / `Possible Disease`. Downloading random web images would have made
the labels unverifiable. PlantVillage is the standard public leaf-disease image
set, its labels are class-level and well documented, and the mapping into two
categories is completely mechanical - which is exactly what a prototype needs.

`prepare_crop_dataset.py` reads the Hugging Face parquet mirror with HTTP range
requests and downloads only the parquet row groups it actually needs, so the
full 54 000-image source is never transferred. The row groups it fetches are
cached under `.cache/plantvillage/` so a second run needs no network at all.

## Class mapping - which original classes became which category

The rule is exact and auditable:

```text
original class name ends with "healthy"  ->  healthy            (API label "Healthy")
anything else (i.e. a named disease)      ->  possible_disease   (API label "Possible Disease")
```

The authoritative machine-readable version of this table is
`data/crop_health/class_mapping.json`. It maps all 38 original classes.

### Mapped into `healthy` -> API label `Healthy` (12 original classes)

| # | Original PlantVillage class |
| - | --------------------------- |
| 1 | `Apple___healthy` |
| 2 | `Blueberry___healthy` |
| 3 | `Cherry_(including_sour)___healthy` |
| 4 | `Corn_(maize)___healthy` |
| 5 | `Grape___healthy` |
| 6 | `Peach___healthy` |
| 7 | `Pepper,_bell___healthy` |
| 8 | `Potato___healthy` |
| 9 | `Raspberry___healthy` |
| 10 | `Soybean___healthy` |
| 11 | `Strawberry___healthy` |
| 12 | `Tomato___healthy` |

### Mapped into `possible_disease` -> API label `Possible Disease` (26 original classes)

| # | Original PlantVillage class |
| - | --------------------------- |
| 1 | `Apple__Apple_scab` |
| 2 | `Apple__Black_rot` |
| 3 | `Apple__Cedar_apple_rust` |
| 4 | `Cherry_(including_sour)___Powdery_mildew` |
| 5 | `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` |
| 6 | `Corn_(maize)___Common_rust_` |
| 7 | `Corn_(maize)___Northern_Leaf_Blight` |
| 8 | `Grape___Black_rot` |
| 9 | `Grape___Esca_(Black_Measles)` |
| 10 | `Grape___Leaf_blight_(Isariopsis_Leaf_Spot)` |
| 11 | `Orange___Haunglongbing_(Citrus_greening)` |
| 12 | `Peach___Bacterial_spot` |
| 13 | `Pepper,_bell___Bacterial_spot` |
| 14 | `Potato___Early_blight` |
| 15 | `Potato___Late_blight` |
| 16 | `Squash___Powdery_mildew` |
| 17 | `Strawberry___Leaf_scorch` |
| 18 | `Tomato___Bacterial_spot` |
| 19 | `Tomato___Early_blight` |
| 20 | `Tomato___Late_blight` |
| 21 | `Tomato___Leaf_Mold` |
| 22 | `Tomato___Septoria_leaf_spot` |
| 23 | `Tomato___Spider_mites Two-spotted_spider_mite` |
| 24 | `Tomato___Target_Spot` |
| 25 | `Tomato___Tomato_Yellow_Leaf_Curl_Virus` |
| 26 | `Tomato___Tomato_mosaic_virus` |

26 different diseases share one label. The model is therefore learning
"is there something wrong with this leaf", not "which disease is it".

## Per-class cap and why the dataset is not the full 54 000 images

PlantVillage is dominated by a few classes: `Orange___Haunglongbing` alone has
more than 5 500 images while `Blueberry___healthy` has fewer than 100. Training
on the raw class distribution would let tomato and orange dominate the loss.

`--per-class 150` keeps at most 150 images of each original class, so every crop
and every disease contributes. Classes with fewer than 150 available images keep
all of them: only `Blueberry___healthy` is affected, with 89 images. The cap
also keeps the download small - 5,639 images instead of 54,381, and only 85 of
the ~750 parquet row groups have to be fetched.

## Counts

Authoritative source: `data/crop_health/dataset_stats.json`. These numbers were
measured by the run that produced the current dataset, and the same counts are
stored in `models/crop_health_model.json` under `dataset_counts`.

| Quantity | Where it comes from |
| -------- | ------------------- |
| Total images (5,639) | `dataset_stats.json` -> `total_images` |
| Healthy 1,739 / Possible Disease 3,900 | `dataset_stats.json` -> `counts` |
| Train 3,947 / validation 846 / test 846 | `dataset_stats.json` -> `counts` |
| Per-original-class counts | `dataset_stats.json` -> `per_original_class` |

Only one class is smaller than the 150-image cap: `Blueberry___healthy` has
only 89 images in PlantVillage, so all 89 are used. The other 37 classes
contribute exactly 150 each: 37 x 150 + 89 = 5,639.

Re-check them at any time with:

```bash
python -c "import json; print(json.load(open('data/crop_health/dataset_stats.json'))['counts'])"
```

Note the class imbalance this creates: 30.8 % healthy against 69.2 % possible
disease. Always read balanced accuracy, not raw accuracy.

## Folder layout

```text
data/crop_health/
    healthy/              -> API label "Healthy"
    possible_disease/     -> API label "Possible Disease"
    splits.csv            -> one row per image: key, category, original_class,
                             split, dup_group, dhash, source provenance
    class_mapping.json    -> every original class -> AgriGuard category
    dataset_stats.json    -> counts, seed, split fractions, per-class counts
```

Directory names are lowercase on disk because that is the convention
`torchvision.datasets.ImageFolder` expects; the API always returns the label
strings `Healthy` and `Possible Disease`.

## Split

- 70 % train / 15 % validation / 15 % test, fixed seed: `42`.
- The split is written to `splits.csv` once and then reused by every training run
  and by the tests, so it is genuinely reproducible rather than re-rolled.
- **Near-duplicate protection:** every image gets a 64-bit dHash. Images whose
  hashes are within a Hamming distance of 3 are put in the same `dup_group`.
  The split is computed over *whole groups*, so a group can never be cut in half
  - 10 images are grouped in the current dataset.
- The split is deliberately computed globally rather than per category. A
  near-duplicate cluster can contain a healthy image and a diseased image a few
  pixels apart, and a per-category split would put the healthy twin in train and
  the diseased twin in test. Clusters are rare (0.18 % of images), so the
  per-category proportions stay at 70/15/15 - asserted in
  `tests/test_dataset_prep.py` and re-checked on the real manifest by
  `tests/test_crophealth.py::test_split_is_reproducible_and_disjoint`.
- The test split is used exactly once, at the end of training, for reporting. It
  never influences training, the epoch count or model selection. The saved
  weights are the ones with the best **validation** accuracy, chosen without
  looking at the test split.

## Preprocessing

Shared by training, validation, testing and live inference, defined once in
`agriguard/imageutils.py`:

```text
RGB
 -> resize to 224 x 224
 -> scale to 0..1
 -> normalise with ImageNet mean (0.485, 0.456, 0.406) and std (0.229, 0.224, 0.225)
```

Training additionally uses: `RandomResizedCrop(224, scale 0.75-1.0)`,
`RandomHorizontalFlip`, `RandomVerticalFlip`, `RandomRotation(25 degrees)` and
`ColorJitter(brightness 0.25, contrast 0.25, saturation 0.2)`.

No augmentation is ever applied to validation, test or inference, and the
inference path imports the very same transform object, so the network can never
be fed a differently sized or differently normalised image at serving time.

## Known limitations

- **Domain gap.** PlantVillage images are single detached leaves on a plain grey
  or black laboratory background. An ESP32-CAM photograph in a real plot has
  soil, shadows, overlapping leaves, motion blur and mixed colour temperature.
  Expect a clearly lower score in the field. This is the single biggest caveat
  in the whole project.
- **Not a diagnostic system.** Two classes only. The model says "Possible
  Disease" for 26 different diseases and cannot name one.
- **Class imbalance.** `Possible Disease` outnumbers `Healthy`, so the reported
  `confidence` is not a calibrated probability. Balanced accuracy and the
  confusion matrix are the numbers to read.
- **Species coverage.** 14 crop species, none of them verified against
  AgriGuard's actual plots. A crop that is not represented here is out of
  distribution.
- **No pest, nutrient or water-stress classes.** PlantVillage covers fungal,
  bacterial and viral leaf diseases only. Herbicide damage, nitrogen deficiency
  and simple drought stress look like disease to a human and are absent here.
- **Label noise.** PlantVillage labels are class-level, not specialist-verified
  per image.
- **Background shortcut.** A CNN can learn "grey background = healthy-looking
  laboratory photo" as much as leaf symptoms. Random-crop augmentation plus the
  flips above reduce this but do not remove it.
- **Licence.** Re-check the upstream repository before commercial or public
  redistribution.
