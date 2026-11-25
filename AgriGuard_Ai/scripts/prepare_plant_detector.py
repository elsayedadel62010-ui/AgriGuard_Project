"""Build the stage-1 plant-detector dataset.

    python scripts/prepare_plant_detector.py
    python scripts/prepare_plant_detector.py --per-class 20

Stage 1 answers "does this image contain a plant?". It is trained on real plant
photographs and real photographs of things that are not plants.

Positives come from the PlantVillage images already prepared by
`scripts/prepare_crop_dataset.py`, reusing that script's existing
train/validation/test assignment verbatim. Reusing the split rather than
drawing a new one means stage 1 and stage 2 are measured on exactly the same
plant photographs, so a disagreement between the two stages is attributable to
the models and not to a reshuffle of the data.

Negatives come from an ungated public mirror of 100 ImageNet-1k classes, and
this is the part that decides whether the feature is worth anything:

**The negative split is over source classes, not over images.** Roughly 90 of
the 100 classes are non-plant (dogs, birds, insects, vehicles, furniture,
tools, textiles, food). They are divided 60/20/20 into train / validation / test
*by class*, then images are drawn from each side. The detector therefore never
sees a single photograph of the non-plant categories reserved for the test set.
That is what makes the test number a statement about unseen non-plants rather
than a restatement of the training distribution - which is the whole point, and
the reason the number comes out lower than a random image-level split would
report.

The alternative - splitting images randomly - would have produced a much better
looking result that measured almost nothing.
"""

import argparse
import csv
import io
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agriguard.config import CROP_HEALTH_DIR, PROJECT_ROOT  # noqa: E402
from agriguard.plantdata import (  # noqa: E402
    CLASS_NONPLANT,
    CLASS_PLANT,
    EXCLUDED_SOURCE_CLASSES,
    IMAGENET100_CITATION,
    IMAGENET100_URL,
    SOURCE_CLASS_SPLIT_FRACTIONS,
    SOURCE_CLASS_SPLIT_SEED,
    detector_dataset_dir,
)
from scripts.prepare_crop_dataset import (  # noqa: E402
    JPEG_QUALITY,
    MAX_SIDE,
    assign_duplicate_groups,
    dhash,
    open_parquet,
    slug,
)


def source_class_names(handle) -> list:
    metadata = handle.schema_arrow.metadata or {}
    raw = metadata.get(b"huggingface", b"{}").decode("utf-8", "replace")
    return json.loads(raw)["info"]["features"]["label"]["names"]


def split_source_classes(names, log):
    """Divide the non-plant source classes into train/validation/test BY CLASS.

    Uses a seeded shuffle so the split is reproducible, and refuses to run if a
    side would end up with too few classes to say anything about - a single test
    category would make the headline number an anecdote.
    """
    excluded_indices = {
        index for index, name in enumerate(names) if name in EXCLUDED_SOURCE_CLASSES
    }
    for index in sorted(excluded_indices):
        log(f"  excluding source class {index:>3} {names[index]} "
            f"- {EXCLUDED_SOURCE_CLASSES[names[index]]}")

    usable = [index for index in range(len(names)) if index not in excluded_indices]
    rng = random.Random(SOURCE_CLASS_SPLIT_SEED)
    rng.shuffle(usable)

    total = len(usable)
    n_train = int(total * SOURCE_CLASS_SPLIT_FRACTIONS["train"])
    n_val = int(total * SOURCE_CLASS_SPLIT_FRACTIONS["validation"])
    sides = {
        "train": usable[:n_train],
        "validation": usable[n_train:n_train + n_val],
        "test": usable[n_train + n_val:],
    }
    for side, indices in sides.items():
        if len(indices) < 3:
            raise SystemExit(
                f"the {side} side would hold only {len(indices)} non-plant "
                "source classes, which is too few to measure. Lower --per-class "
                "and re-check the class count, or widen the source dataset."
            )
        log(f"  {side:<11} {len(indices):>3} source classes")
    return sides, excluded_indices


def fetch_negatives(per_class, out_dir, log):
    """Download non-plant images, tagged with which split their CLASS belongs to."""
    handle = open_parquet(IMAGENET100_URL)
    names = source_class_names(handle)
    log(f"{IMAGENET100_URL.rsplit('/', 1)[-1]}: {handle.metadata.num_rows} rows, "
        f"{len(names)} source classes")

    sides, excluded = split_source_classes(names, log)
    side_of_class = {index: side for side, indices in sides.items() for index in indices}

    remaining = {index: per_class for index in side_of_class}
    images_dir = out_dir / CLASS_NONPLANT
    images_dir.mkdir(parents=True, exist_ok=True)
    records = []
    seen = set()

    for group in range(handle.metadata.num_row_groups):
        if not any(count > 0 for count in remaining.values()):
            break
        table = handle.read_row_group(group, columns=["label", "image"])
        labels = table.column("label").to_pylist()
        payloads = table.column("image").to_pylist()
        for row_index, label in enumerate(labels):
            if remaining.get(label, 0) <= 0 or label in excluded:
                continue
            entry = payloads[row_index]
            blob = entry["bytes"] if isinstance(entry, dict) else entry
            if not blob:
                continue
            try:
                image = Image.open(io.BytesIO(blob)).convert("RGB")
            except Exception:  # noqa: BLE001 - a corrupt source row is skipped
                continue
            if max(image.size) > MAX_SIDE:
                scale = MAX_SIDE / max(image.size)
                image = image.resize(
                    (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
                    Image.LANCZOS,
                )
            base = f"{slug(names[label])}__{group:03d}_{row_index:03d}"
            if base in seen:
                continue
            seen.add(base)
            name = f"{base}.jpg"
            image.save(images_dir / name, format="JPEG", quality=JPEG_QUALITY)
            records.append(
                {
                    "key": f"{CLASS_NONPLANT}/{name}",
                    "label": CLASS_NONPLANT,
                    "source_class": names[label],
                    "source_class_split": side_of_class[label],
                    "origin": "imagenet-100",
                    "dhash": dhash(image),
                }
            )
            remaining[label] -= 1
    handle.close()
    return records, sides, names


def add_plant_positives(out_dir, log):
    """Point at the PlantVillage images, inheriting its split assignment.

    The plant images are *not* copied. They are referenced by relative path from
    the detector directory, so there is one copy of 5,639 images in the project
    instead of two, and the two stages cannot drift apart on disk.
    """
    splits = CROP_HEALTH_DIR / "splits.csv"
    if not splits.exists():
        raise SystemExit(
            f"{splits} not found. Run scripts/prepare_crop_dataset.py first."
        )
    records = []
    with open(splits, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            records.append(
                {
                    "key": f"{CROP_HEALTH_DIR.name}/{row['key']}",
                    "label": CLASS_PLANT,
                    "source_class": f"plantvillage__{row['original_class']}",
                    "source_class_split": row["split"],
                    "origin": "plantvillage",
                    "dhash": row["dhash"],
                }
            )
    log(f"referencing {len(records)} PlantVillage images (no copy on disk)")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-class", type=int, default=20,
                        help="negative images per non-plant source class (default 20)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    def log(message):
        print(message, flush=True)

    out_dir = args.out or detector_dataset_dir(PROJECT_ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"stage-1 dataset -> {out_dir}")

    log("non-plant negatives:")
    negatives, sides, names = fetch_negatives(args.per_class, out_dir, log)
    if not negatives:
        raise SystemExit("no negatives were fetched; refusing to write a detector dataset")

    positives = add_plant_positives(out_dir, log)
    records = positives + negatives
    records = assign_duplicate_groups(records)

    per_side = {}
    for record in records:
        per_side.setdefault(record["source_class_split"], {}).setdefault(record["label"], 0)
        per_side[record["source_class_split"]][record["label"]] += 1
    log("\nimages per split and label:")
    for side in ("train", "validation", "test"):
        counts = per_side.get(side, {})
        log(f"  {side:<11} plant {counts.get(CLASS_PLANT, 0):>5}   "
            f"non-plant {counts.get(CLASS_NONPLANT, 0):>5}")

    splits_path = out_dir / "splits.csv"
    fieldnames = ["key", "label", "split", "source_class", "source_class_split",
                  "origin", "dup_group", "dhash"]
    with open(splits_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in sorted(records, key=lambda r: r["key"]):
            writer.writerow(
                {
                    "key": record["key"],
                    "label": record["label"],
                    "split": record["source_class_split"],
                    "source_class": record["source_class"],
                    "source_class_split": record["source_class_split"],
                    "origin": record["origin"],
                    "dup_group": record["dup_group"],
                    "dhash": record["dhash"],
                }
            )

    stats = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "positive_source": "PlantVillage (referenced, not copied)",
        "negative_source": IMAGENET100_CITATION,
        "negative_source_url": IMAGENET100_URL,
        "per_class_cap": args.per_class,
        "max_side_px": MAX_SIDE,
        "jpeg_quality": JPEG_QUALITY,
        "source_class_split_fractions": SOURCE_CLASS_SPLIT_FRACTIONS,
        "source_class_split_seed": SOURCE_CLASS_SPLIT_SEED,
        "excluded_source_classes": EXCLUDED_SOURCE_CLASSES,
        "non_plant_source_classes_per_side": {
            side: sorted(names[index] for index in indices)
            for side, indices in sides.items()
        },
        "counts_per_split": per_side,
        "total_images": len(records),
        "note": (
            "non-plant images are split BY SOURCE CLASS, so the test and "
            "validation sides contain categories the detector never saw"
        ),
    }
    (out_dir / "dataset_stats.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8"
    )

    log(f"\nwrote {splits_path.name} ({len(records)} rows) and dataset_stats.json")
    log("negatives are split by source CLASS, so test holds out whole categories")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
