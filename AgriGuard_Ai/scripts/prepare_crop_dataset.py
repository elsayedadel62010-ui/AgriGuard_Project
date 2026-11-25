"""Build the AgriGuard crop-health image dataset from PlantVillage.

WHAT THIS DOES
--------------
The project needs real leaf images labelled into two categories:

    Healthy
    Possible Disease

The public source is PlantVillage (Mohanty et al., 2016) - 38 crop/disease
classes of single leaves photographed on a plain background. Every class name
looks like ``Tomato___Early_blight`` or ``Tomato___healthy`` so the mapping to
our two categories is mechanical and auditable:

    original class ends with "healthy"          -> healthy
    any other original class (a named disease)  -> possible_disease

The exact mapping table is written to
``data/crop_health/class_mapping.json`` and reproduced in
``docs/crop_health_dataset.md``.

To keep the download small we read the Hugging Face parquet mirror with HTTP
range requests and only fetch the row groups that actually contain the images we
keep. We cap the number of images per original class so that no single crop
dominates the dataset.

    python scripts/prepare_crop_dataset.py --per-class 150

Outputs
-------
    data/crop_health/healthy/*.jpg
    data/crop_health/possible_disease/*.jpg
    data/crop_health/splits.csv          filename,category,original_class,split,...
    data/crop_health/class_mapping.json   original class -> AgriGuard category
    data/crop_health/dataset_stats.json   counts + sha256 of the source files
"""

import argparse
import io
import json
import pickle
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agriguard.config import CROP_HEALTH_DIR  # noqa: E402
from agriguard.cropdata import PLANTVILLAGE_FILES, PLANTVILLAGE_URL  # noqa: E402

CATEGORY_HEALTHY = "healthy"
CATEGORY_DISEASE = "possible_disease"

API_LABEL = {
    CATEGORY_HEALTHY: "Healthy",
    CATEGORY_DISEASE: "Possible Disease",
}

SPLIT_SEED = 42
SPLIT_FRACTIONS = {"train": 0.70, "validation": 0.15, "test": 0.15}
MAX_SIDE = 256
JPEG_QUALITY = 90


# --------------------------------------------------------------------------
# minimal HTTP range reader (so we can read a remote parquet selectively)
# --------------------------------------------------------------------------
class HttpRangeFile(io.RawIOBase):
    """A read-only, seekable file object backed by HTTP Range requests."""

    def __init__(self, url: str, timeout: int = 180):
        self.url = url
        self.pos = 0
        self.timeout = timeout
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as resp:
            self.size = int(resp.headers["Content-Length"])

    def readable(self):  # noqa: D102
        return True

    def seekable(self):  # noqa: D102
        return True

    def tell(self):  # noqa: D102
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):  # noqa: D102
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        else:
            self.pos = self.size + offset
        return self.pos

    def readinto(self, buf):  # noqa: D102
        want = len(buf)
        start = self.pos
        if start >= self.size:
            return 0
        end = min(start + want - 1, self.size - 1)
        req = urllib.request.Request(
            self.url, headers={"Range": f"bytes={start}-{end}"}
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = resp.read()
        buf[: len(data)] = data
        self.pos = start + len(data)
        return len(data)


def open_parquet(url):
    return pq.ParquetFile(pa.PythonFile(HttpRangeFile(url), "rb"))


def class_names(parquet_file):
    meta = parquet_file.schema_arrow.metadata or {}
    info = json.loads(meta[b"huggingface"].decode())
    return info["info"]["features"]["label"]["names"]


def to_category(original_class: str) -> str:
    """healthy -> healthy, every named disease -> possible_disease."""
    return CATEGORY_HEALTHY if original_class.strip().lower().endswith("healthy") else CATEGORY_DISEASE


def slug(text: str) -> str:
    keep = []
    for ch in text.lower():
        keep.append(ch if (ch.isalnum() or ch in "._-") else "_")
    return "".join(keep)[:80].strip("_") or "class"


# --------------------------------------------------------------------------
# step 1: cheap label-only scan of every shard
# --------------------------------------------------------------------------
def scan_labels(per_file_cap, log, cache_dir=None):
    """Return names plus, per class, the (file, row_group) plan that needs downloading."""
    cache_path = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / "label_scan.pkl"
        if cache_path.exists():
            log("using cached label scan")
            with open(cache_path, "rb") as handle:
                return pickle.load(handle)

    names = None
    # class_id -> list of (file_index, row_group, rows_in_group, offset_in_group)
    occurrences = {}

    for file_index, url in enumerate(PLANTVILLAGE_FILES):
        log(f"scanning labels: {url.rsplit('/', 1)[-1]}")
        pf = open_parquet(url)
        if names is None:
            names = class_names(pf)
        for rg in range(pf.metadata.num_row_groups):
            table = pf.read_row_group(rg, columns=["label"])
            for offset, label in enumerate(table.column("label").to_pylist()):
                occurrences.setdefault(label, []).append((file_index, rg, offset))
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as handle:
            pickle.dump((names, occurrences), handle, protocol=pickle.HIGHEST_PROTOCOL)
    return names, occurrences


def choose_rows(occurrences, per_class_cap, log):
    """Pick up to `per_class_cap` rows per class.

    For every class we try each shard and every window of consecutive row groups
    and keep the pick that yields the most images, so a class that happens to be
    split across two shards is still filled up to the cap. Windows keep the
    number of parquet row groups we have to download small.
    """
    chosen = {}
    window_groups = 14
    for class_id, spots in sorted(occurrences.items()):
        best = []
        for file_index in range(len(PLANTVILLAGE_FILES)):
            in_file = [s for s in spots if s[0] == file_index]
            if not in_file:
                continue
            groups = sorted({s[1] for s in in_file})
            for start in range(len(groups)):
                window = set(groups[start : start + window_groups])
                picked = sorted(
                    (s for s in in_file if s[1] in window), key=lambda s: (s[1], s[2])
                )[:per_class_cap]
                if len(picked) > len(best):
                    best = picked
                if len(best) >= per_class_cap:
                    break
            if len(best) >= per_class_cap:
                break
        chosen[class_id] = best
        flag = "" if len(best) >= per_class_cap else "  (all available)"
        log(f"  class {class_id:2d}: {len(best):4d} images selected{flag}")
    return chosen


def fetch_images(plan, log, cache_dir=None, jobs=4):
    """plan: {(file_index, row_group): {row_offset: class_id}} -> bytes per row.

    Results are cached on disk so a second run does not hit the network again.
    """
    entries = sorted(plan.items())
    log(f"downloading {len(entries)} parquet row groups")

    def cache_path(file_index, rg):
        if cache_dir is None:
            return None
        name = f"{Path(PLANTVILLAGE_FILES[file_index]).stem}__{rg:05d}.pkl"
        return Path(cache_dir) / name

    def work(entry):
        (file_index, rg), wanted = entry
        path = cache_path(file_index, rg)
        if path is not None and path.exists():
            with open(path, "rb") as handle:
                return (file_index, rg), pickle.load(handle)

        pf = open_parquet(PLANTVILLAGE_FILES[file_index])
        table = pf.read_row_group(rg)
        out = {}
        for row_offset, class_id in wanted.items():
            record = table.slice(row_offset, 1).to_pylist()[0]
            out[row_offset] = (class_id, record["image"]["bytes"], record["image"]["path"])
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as handle:
                pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return (file_index, rg), out

    results = {}
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for done, (key, value) in enumerate(pool.map(work, entries), start=1):
            results[key] = value
            log(f"  row group {done}/{len(entries)} ready")
    return results


# --------------------------------------------------------------------------
# step 2: near-duplicate detection so train and test cannot share a photo
# --------------------------------------------------------------------------
def dhash(image: Image.Image, size: int = 8) -> int:
    """64-bit difference hash - simple, dependency-free, good enough for leakage checks."""
    small = image.convert("L").resize((size + 1, size), Image.BILINEAR)
    pixels = list(small.getdata())
    bits = 0
    for row in range(size):
        base = row * (size + 1)
        for col in range(size):
            bits = (bits << 1) | int(pixels[base + col] > pixels[base + col + 1])
    return bits


def assign_duplicate_groups(records, threshold=3):
    """records: list of dicts with 'dhash'. Groups near-identical hashes together.

    PlantVillage contains a few genuinely repeated shots. Keeping every member
    of a near-duplicate cluster inside the same split means the test set can
    never contain a re-encoded twin of a training photo.
    """
    group_of = {}  # representative dhash -> group id
    counter = 0
    for rec in records:
        value = rec["dhash"]
        match = None
        for other_hash, group_id in group_of.items():
            if (value ^ other_hash).bit_count() <= threshold:
                match = group_id
                break
        if match is None:
            group_id = f"dup_{counter:05d}"
            counter += 1
            group_of[value] = group_id
            rec["dup_group"] = group_id
        else:
            rec["dup_group"] = match
    return records


def stratified_group_split(records, seed=SPLIT_SEED):
    """70/15/15 keeping every near-duplicate group inside a single split.

    The split is driven by *global* duplicate groups rather than by category.
    A near-duplicate cluster can contain both a healthy and a diseased image (the
    two differ by a few pixels), and splitting per category would then put the
    healthy member in train and the diseased twin in test - which is exactly the
    leakage this function exists to prevent. Because clusters are rare (a handful
    of images out of thousands) the per-category proportions stay at 70/15/15
    anyway, and tests/test_dataset_prep.py asserts that.
    """
    import random

    rng = random.Random(seed)
    groups = {}
    for rec in records:
        groups.setdefault(rec["dup_group"], []).append(rec)

    keys = sorted(groups)
    rng.shuffle(keys)

    total = len(records)
    cut_train = int(round(total * SPLIT_FRACTIONS["train"]))
    cut_val = cut_train + int(round(total * SPLIT_FRACTIONS["validation"]))

    assignment = {}
    running = 0
    for key in keys:
        size = len(groups[key])
        if running + size <= cut_train:
            split = "train"
        elif running + size <= cut_val:
            split = "validation"
        else:
            split = "test"
        running += size
        for rec in groups[key]:
            assignment[rec["key"]] = split
    return assignment


# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-class", type=int, default=150)
    parser.add_argument("--out", type=Path, default=CROP_HEALTH_DIR)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(".cache") / "plantvillage",
        help="directory for cached parquet row groups so re-runs skip the download",
    )
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="keep already-downloaded images instead of clearing the folder first",
    )
    args = parser.parse_args()

    def log(message):
        print(message, flush=True)

    names, occurrences = scan_labels(args.per_class, log, cache_dir=args.cache)
    log(f"found {len(names)} original PlantVillage classes")

    chosen = choose_rows(occurrences, args.per_class, log)

    plan = {}
    for class_id, spots in chosen.items():
        for file_index, rg, offset in spots:
            plan.setdefault((file_index, rg), {})[offset] = class_id

    fetched = fetch_images(plan, log, cache_dir=args.cache, jobs=args.jobs)

    out_dir = args.out
    if out_dir.exists() and not args.reuse:
        for folder in out_dir.iterdir():
            if folder.is_dir():
                for image_file in folder.iterdir():
                    image_file.unlink()
    for category in (CATEGORY_HEALTHY, CATEGORY_DISEASE):
        (out_dir / category).mkdir(parents=True, exist_ok=True)

    records = []
    written = set()
    sequence = 0
    for (file_index, rg), rows in sorted(fetched.items()):
        for offset in sorted(rows):
            class_id, blob, source_name = rows[offset]
            original = names[class_id]
            category = to_category(original)
            sequence += 1
            filename = f"{slug(original)}__{sequence:05d}.jpg"
            key = f"{category}/{filename}"
            if key in written:
                continue
            written.add(key)
            with Image.open(io.BytesIO(blob)) as img:
                img = img.convert("RGB")
                hash_value = dhash(img)
                img.thumbnail((MAX_SIDE, MAX_SIDE), Image.BICUBIC)
                img.save(out_dir / category / filename, "JPEG", quality=JPEG_QUALITY)
            records.append(
                {
                    "key": key,
                    "category": category,
                    "original_class": original,
                    "dhash": hash_value,
                    "source_file": PLANTVILLAGE_FILES[file_index].rsplit("/", 1)[-1],
                    "source_row_group": rg,
                    "source_row": offset,
                    "source_image_name": source_name,
                }
            )

    before = len(records)
    records = assign_duplicate_groups(records)
    clustered = before - len({r["dup_group"] for r in records})
    log(
        f"wrote {len(records)} images; {clustered} of them sit in a near-duplicate "
        f"cluster (hamming distance <= 3 on a 64-bit dHash)"
    )

    splits = stratified_group_split(records)
    for rec in records:
        rec["split"] = splits[rec["key"]]

    import csv

    with open(out_dir / "splits.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "key",
                "category",
                "original_class",
                "split",
                "dup_group",
                "dhash",
                "source_file",
                "source_row_group",
                "source_row",
                "source_image_name",
            ],
        )
        writer.writeheader()
        for rec in sorted(records, key=lambda r: r["key"]):
            writer.writerow({k: rec[k] for k in writer.fieldnames})

    mapping = {}
    for class_id, original in enumerate(names):
        mapping[original] = {
            "agriguard_category": to_category(original),
            "api_label": API_LABEL[to_category(original)],
        }
    with open(out_dir / "class_mapping.json", "w", encoding="utf-8") as fh:
        json.dump(mapping, fh, indent=2, sort_keys=True)

    counts = {}
    for rec in records:
        for bucket in (rec["category"], rec["split"]):
            counts[bucket] = counts.get(bucket, 0) + 1
        counts[f"{rec['split']}_{rec['category']}"] = (
            counts.get(f"{rec['split']}_{rec['category']}", 0) + 1
        )
    stats = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": "PlantVillage (Hugging Face parquet mirror)",
        "source_urls": list(PLANTVILLAGE_FILES),
        "per_class_cap": args.per_class,
        "max_side_px": MAX_SIDE,
        "jpeg_quality": JPEG_QUALITY,
        "split_seed": SPLIT_SEED,
        "split_fractions": SPLIT_FRACTIONS,
        "total_images": len(records),
        "near_duplicate_groups": len({r["dup_group"] for r in records}),
        "counts": counts,
        "per_original_class": {},
    }
    for rec in records:
        stats["per_original_class"][rec["original_class"]] = (
            stats["per_original_class"].get(rec["original_class"], 0) + 1
        )
    with open(out_dir / "dataset_stats.json", "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, sort_keys=True)

    log(json.dumps(stats["counts"], indent=2))
    log(f"dataset ready -> {out_dir}")


if __name__ == "__main__":
    main()
