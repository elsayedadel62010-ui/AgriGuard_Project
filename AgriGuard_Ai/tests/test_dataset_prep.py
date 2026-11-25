"""Tests for scripts/prepare_crop_dataset.py helpers.

These are the pure/offline parts of the dataset builder. They are tested because
the download step is slow and must never be able to crash on a plan/cache bug:
the whole PlantVillage download is expensive, so the planning, caching,
duplicate-grouping and splitting logic has to be verifiable without a network.
"""

import json
import random
import sys
import tempfile
from pathlib import Path

import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import prepare_crop_dataset as prep  # noqa: E402


def _fake_parquet(monkeypatch, calls):
    """Replace the HTTP parquet reader with an in-memory fake."""

    def fake_open_parquet(url):
        calls.append(url)

        class FakeTable:
            def __init__(self, row_group):
                self.row_group = row_group

            def slice(self, offset, length):
                payload = f"bytes-rg{self.row_group}-off{offset}".encode()
                return _row_table(payload, f"name-rg{self.row_group}-off{offset}.jpg")

        class FakeParquet:
            def read_row_group(self, rg, columns=None):
                if columns is not None:
                    return _label_table(rg)
                return FakeTable(rg)

        return FakeParquet()

    monkeypatch.setattr(prep, "open_parquet", fake_open_parquet)


def _row_table(payload, name):
    import pyarrow as pa

    return pa.table({"image": [{"bytes": payload, "path": name}]})


def _label_table(row_group):
    import pyarrow as pa

    labels = [0, 1, 0][: row_group % 3 + 1]
    return pa.table({"label": labels})


# --------------------------------------------------------------------------
# the download plan: {(file_index, row_group): {row_offset: class_id}}
# --------------------------------------------------------------------------
def test_fetch_images_returns_bytes_for_every_requested_row(monkeypatch):
    calls = []
    _fake_parquet(monkeypatch, calls)
    plan = {(0, 3): {5: 0, 7: 1}, (1, 11): {2: 0}}

    with tempfile.TemporaryDirectory() as cache:
        fetched = prep.fetch_images(plan, lambda m: None, cache_dir=cache, jobs=2)

    assert sorted(fetched) == [(0, 3), (1, 11)]
    for key, rows in fetched.items():
        for offset, (class_id, blob, name) in rows.items():
            assert isinstance(class_id, int)
            assert isinstance(blob, bytes) and blob
            assert isinstance(name, str) and name


def test_fetch_images_uses_the_row_group_and_file_from_the_plan_key(monkeypatch):
    """The cache filename and the shard URL must both come from the plan key.

    A previous version unpacked the whole ``plan.items()`` entry, so
    ``PLANTVILLAGE_FILES[file_index]`` was indexed with a tuple and the run
    crashed with ``TypeError: tuple indices must be integers`` on the very first
    row group. This test fails if that regresses.
    """
    calls = []
    _fake_parquet(monkeypatch, calls)
    plan = {(1, 11): {2: 0}}

    with tempfile.TemporaryDirectory() as cache:
        prep.fetch_images(plan, lambda m: None, cache_dir=cache, jobs=1)
        cached = sorted(p.name for p in Path(cache).iterdir())

    assert calls == [prep.PLANTVILLAGE_FILES[1]]
    assert cached == ["train-00000-of-00002__00011.pkl"]


def test_second_fetch_is_served_from_the_cache_without_network(monkeypatch):
    calls = []
    _fake_parquet(monkeypatch, calls)
    plan = {(0, 3): {5: 0, 7: 1}, (1, 11): {2: 0}}

    with tempfile.TemporaryDirectory() as cache:
        first = prep.fetch_images(plan, lambda m: None, cache_dir=cache, jobs=2)
        assert len(calls) == 2
        second = prep.fetch_images(plan, lambda m: None, cache_dir=cache, jobs=2)

    assert len(calls) == 2, "a cached row group must not be re-downloaded"
    assert first == second


def test_cached_rows_survive_a_restart(monkeypatch):
    calls = []
    _fake_parquet(monkeypatch, calls)
    plan = {(0, 0): {0: 0}}

    with tempfile.TemporaryDirectory() as cache:
        first = prep.fetch_images(plan, lambda m: None, cache_dir=cache, jobs=1)
        monkeypatch.setattr(prep, "open_parquet", lambda url: pytest.fail("network used"))
        second = prep.fetch_images(plan, lambda m: None, cache_dir=cache, jobs=1)

    assert first == second
    for key in first:
        assert key in second


def test_label_scan_is_cached(monkeypatch):
    calls = []

    class FakeParquet:
        class schema_arrow:
            metadata = {
                b"huggingface": json.dumps(
                    {"info": {"features": {"label": {"names": ["a", "b"]}}}}
                ).encode()
            }

        class metadata:
            num_row_groups = 2

        def read_row_group(self, rg, columns=None):
            calls.append(rg)
            return _label_table(rg)

    monkeypatch.setattr(prep, "open_parquet", lambda url: FakeParquet())

    with tempfile.TemporaryDirectory() as cache:
        first_names, first_occurrences = prep.scan_labels(150, lambda m: None, cache_dir=cache)
        assert calls, "the first scan must read the shards"

        calls.clear()
        second_names, second_occurrences = prep.scan_labels(150, lambda m: None, cache_dir=cache)

    assert calls == [], "a cached label scan must not read the shards again"
    assert first_names == second_names == ["a", "b"]
    assert first_occurrences == second_occurrences
    assert first_occurrences, "the fake shards contain labels"


# --------------------------------------------------------------------------
# near-duplicate detection
# --------------------------------------------------------------------------
def _flat_records(count, prefix=0):
    return [
        {"key": f"healthy/img_{i:03d}.jpg", "dhash": (i * 37 + prefix) & 0xFFFFFFFFFFFF}
        for i in range(count)
    ]


def test_identical_images_land_in_the_same_group():
    records = [
        {"key": "a.jpg", "dhash": 0b1010101010101010},
        {"key": "b.jpg", "dhash": 0b1010101010101010},  # exact twin
        {"key": "c.jpg", "dhash": 0xFFFFFFFFFFFF0000},
    ]
    prep.assign_duplicate_groups(records)
    assert records[0]["dup_group"] == records[1]["dup_group"]
    assert records[2]["dup_group"] != records[0]["dup_group"]


def test_slightly_different_images_land_in_the_same_group():
    base = 0b1010101010101010
    near = base ^ 0b11  # two bits differ -> hamming distance 2 <= 3
    records = [
        {"key": "a.jpg", "dhash": base},
        {"key": "b.jpg", "dhash": near},
    ]
    prep.assign_duplicate_groups(records, threshold=3)
    assert records[0]["dup_group"] == records[1]["dup_group"]


def test_different_images_get_different_groups():
    records = [
        {"key": "a.jpg", "dhash": 0x0000000000000000},
        {"key": "b.jpg", "dhash": 0xFFFFFFFFFFFFFFFF},
    ]
    prep.assign_duplicate_groups(records, threshold=3)
    assert records[0]["dup_group"] != records[1]["dup_group"]


def test_dhash_is_deterministic_and_image_sensitive():
    # a flat colour or a smooth gradient has no horizontal structure, so build
    # two deterministic noise images instead
    def noise(seed):
        rng = random.Random(seed)
        return Image.frombytes(
            "L", (64, 64), bytes(rng.randrange(256) for _ in range(64 * 64))
        )

    first, second = noise(1), noise(2)
    assert prep.dhash(first) == prep.dhash(first.copy())
    assert prep.dhash(first) != prep.dhash(second)


# --------------------------------------------------------------------------
# the split
# --------------------------------------------------------------------------
def test_split_keeps_every_duplicate_group_in_one_split():
    records = []
    for i in range(400):
        records.append(
            {
                "key": f"healthy/img_{i:03d}.jpg",
                "category": "healthy",
                # every 10th image is an exact twin of the one before it
                "dhash": (i // 10) * 41,
            }
        )
    prep.assign_duplicate_groups(records)
    assignment = prep.stratified_group_split(records, seed=42)

    by_group = {}
    for rec in records:
        by_group.setdefault(rec["dup_group"], set()).add(assignment[rec["key"]])
    for group, splits in by_group.items():
        assert len(splits) == 1, f"group {group} leaked across {splits}"


def test_split_is_reproducible_for_a_fixed_seed():
    records = _flat_records(200)
    for rec in records:
        rec["category"] = "healthy"
    prep.assign_duplicate_groups(records)
    first = prep.stratified_group_split(records, seed=42)
    second = prep.stratified_group_split(records, seed=42)
    assert first == second


def test_split_uses_every_split_name_and_respects_the_fractions():
    records = _flat_records(1000)
    for rec in records:
        rec["category"] = "healthy"
    prep.assign_duplicate_groups(records)
    assignment = prep.stratified_group_split(records, seed=42)

    counts = {name: 0 for name in prep.SPLIT_FRACTIONS}
    for split in assignment.values():
        counts[split] += 1
    assert sum(counts.values()) == 1000
    assert counts["train"] / 1000 == pytest.approx(0.70, abs=0.05)
    assert counts["validation"] / 1000 == pytest.approx(0.15, abs=0.05)
    assert counts["test"] / 1000 == pytest.approx(0.15, abs=0.05)


def test_duplicate_group_spanning_both_categories_stays_in_one_split():
    """The real bug this guards: a healthy/diseased near-twin pair.

    Two images that differ by a few pixels can carry opposite labels. If the
    split were computed per category, the healthy twin would land in train and
    the diseased twin in test - a genuine leak of a test photo into training.
    """
    records = [
        {
            "key": "healthy/a.jpg",
            "category": "healthy",
            # 3 bits away from the diseased twin below -> same cluster
            "dhash": 0b1010101010101010,
        },
        {
            "key": "possible_disease/b.jpg",
            "category": "possible_disease",
            "dhash": 0b1010101010101010 ^ 0b111,
        },
    ]
    for index in range(600):
        records.append(
            {
                "key": f"healthy/f{index}.jpg",
                "category": "healthy",
                "dhash": (index + 50) * 2654435761 & 0xFFFFFFFFFF,
            }
        )
        records.append(
            {
                "key": f"possible_disease/g{index}.jpg",
                "category": "possible_disease",
                "dhash": (index + 900) * 40503 & 0xFFFFFFFFFF,
            }
        )

    prep.assign_duplicate_groups(records, threshold=3)
    assignment = prep.stratified_group_split(records, seed=42)

    by_group = {}
    for rec in records:
        by_group.setdefault(rec["dup_group"], set()).add(assignment[rec["key"]])
    straddling = {g: s for g, s in by_group.items() if len(s) > 1}
    assert not straddling, f"groups leaked across splits: {straddling}"


def test_per_category_proportions_survive_a_global_group_split():
    records = []
    for index in range(700):
        records.append(
            {
                "key": f"healthy/h{index}.jpg",
                "category": "healthy",
                "dhash": (index + 3) * 999331 & 0xFFFFFFFFFF,
            }
        )
        records.append(
            {
                "key": f"possible_disease/d{index}.jpg",
                "category": "possible_disease",
                "dhash": (index + 88) * 1237 & 0xFFFFFFFFFF,
            }
        )
    prep.assign_duplicate_groups(records)
    assignment = prep.stratified_group_split(records, seed=42)

    for category in ("healthy", "possible_disease"):
        rows = [r for r in records if r["category"] == category]
        for split, fraction in prep.SPLIT_FRACTIONS.items():
            share = sum(1 for r in rows if assignment[r["key"]] == split) / len(rows)
            assert share == pytest.approx(fraction, abs=0.05), (category, split, share)
