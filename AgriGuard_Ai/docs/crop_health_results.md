# Crop-health model: measured results

Every number in this file comes from `models/crop_health_model.json`, which is
written by `scripts/train_crophealth.py` and independently re-measured by
`scripts/evaluate_models.py`. The raw command transcripts are in
`evidence/06_crop_training.txt`, `evidence/07_crop_evaluation.txt` and
`evidence/08_confusion_matrix.txt`.

## The short version

| | |
| --- | --- |
| Method | MobileNetV3-Small, transfer-learned, ImageNet initialisation |
| Version | 0.2.0 |
| Task | Binary: `healthy` vs `possible_disease` |
| Training data | PlantVillage colour images, 5,639 images |
| Test accuracy | **0.9657** |
| Test balanced accuracy | **0.9668** |
| Macro F1 | 0.9606 |
| Majority-class baseline | 0.6879 |
| Documented baseline (colour heuristic) | 0.4208 |

Read the balanced accuracy, not the accuracy. The test split is 31 % Healthy /
69 % Possible Disease, so a model that predicted "Possible Disease" every time
would score 0.6879 accuracy. The CNN is well above that, and 0.4208 shows the
non-ML colour heuristic is *worse* than that trivial baseline.

## Per-class results on the unseen test split (846 images)

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| Healthy | 0.9242 | 0.9697 | 0.9464 | 264 |
| Possible Disease | 0.9859 | 0.9639 | 0.9748 | 582 |

Confusion matrix, rows are the true class:

| true \ predicted | Healthy | Possible Disease |
| --- | --- | --- |
| **Healthy** | 256 | 8 |
| **Possible Disease** | 21 | 561 |

The model misses 21 diseased leaves and falsely alarms on 8 healthy ones. Both
error directions are small, which is why the two recalls are nearly equal and
balanced accuracy is only 0.0011 above plain accuracy.

## How it was trained

| Setting | Value |
| --- | --- |
| Seed | 42 |
| Epochs | 6 (2 frozen-backbone, 4 with the last two blocks unfrozen) |
| Image size | 224 x 224 |
| Batch size | 32 |
| Stage 2 learning rate | 3e-4 |
| Parameters | 1,519,906 |
| Best validation accuracy | 0.9823 (epoch 3) |

Validation accuracy by epoch: 0.7317, 0.9610, **0.9823**, 0.9704, 0.9775, 0.9775.

The peak is in the middle, not at the end. The script restores the weights from
the best epoch rather than the last, and
`test_the_saved_weights_come_from_the_best_validation_epoch` asserts that the
stored `best_validation_accuracy` equals the maximum in the history. An earlier
version of the script had an indentation bug that compared only the final epoch,
so it silently saved epoch 6 and reported it as the best.

## What the model actually is

38 PlantVillage leaf classes across 14 species were collapsed into two
categories:

- `healthy` - the 12 classes that have a `___healthy` folder (Apple, Blueberry,
  Cherry, Corn, Grape, Peach, Bell pepper, Potato, Raspberry, Soybean, Strawberry,
  Tomato)
- `possible_disease` - the other 26 classes, covering 14 species (the 12 above plus
  Orange and Squash, which have no healthy images at all)

So "Possible Disease" is a bucket, not a diagnosis. The model cannot tell you
that a leaf has early blight, and it is not a diagnostic tool. See
`docs/crop_health_dataset.md` for the collapse table and the group-safe split,
and `docs/limitations.md` for the full risk list.

Note the asymmetry that falls out of this: Orange and Squash have **no** healthy
training images. A healthy orange leaf is therefore out of distribution for the
`healthy` class, and the model has no way to represent it.

## What these numbers are not

This is a laboratory number and it is not a field number. PlantVillage images
are a single leaf on a plain, evenly lit background. An ESP32-CAM photograph of a
real plant contains soil, other leaves, shadows, mixed lighting, motion blur and
a much wider colour range. Expect materially lower accuracy on real hardware
photos. Treat this model as a working demonstration of the pipeline, and
re-validate it on field imagery before trusting it.

`confidence` in the response is an uncalibrated softmax score. At 69 %
Possible Disease in training it is not a probability and must not be shown as
"96.6 % sure the plant is sick".
