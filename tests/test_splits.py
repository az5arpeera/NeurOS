"""The locked test set and development folds (architecture §8.1, D-026)."""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from neuros.data.registry import get_spec, locked_test_dataset_keys
from neuros.errors import NeurOSConfigError, NeurOSSplitError
from neuros.eval.splits import (
    LOCKED_TEST_SEED,
    _allocate,
    load_locked_test_set,
    make_dev_folds,
    make_locked_test_set,
    save_locked_test_set,
)

from .conftest import REPO_ROOT

COMMITTED = REPO_ROOT / "configs" / "splits" / "locked_test_v1.yaml"
EXPECTED_COUNTS = {
    "PhysionetMI": 22,
    "Dreyer2023": 17,
    "Stieger2021": 12,
    "Lee2019_MI": 11,
    "Cho2017": 10,
    "EEGMAT": 7,
}


def test_counts_are_twenty_percent_per_dataset() -> None:
    split = make_locked_test_set()
    assert {k: len(v) for k, v in split.test.items()} == EXPECTED_COUNTS
    assert sum(EXPECTED_COUNTS.values()) == 79  # 72 motor-imagery + 7 EEGMAT people
    assert set(split.test) == set(locked_test_dataset_keys())


def test_dreyer_is_stratified_by_subgroup() -> None:
    dreyer = get_spec("Dreyer2023")
    chosen = make_locked_test_set().test["Dreyer2023"]
    groups = [dreyer.subgroup_of(s) for s in chosen]
    assert (groups.count("A"), groups.count("B"), groups.count("C")) == (12, 4, 1)


def test_deterministic_and_seed_sensitive() -> None:
    a, b = make_locked_test_set(created="x"), make_locked_test_set(created="y")
    assert dict(a.test) == dict(b.test)
    assert a.digest == b.digest  # the creation date isn't part of the digest
    other = make_locked_test_set(seed=LOCKED_TEST_SEED + 1)
    assert dict(other.test) != dict(a.test)


def test_datasets_are_drawn_independently() -> None:
    full = make_locked_test_set()
    partial = make_locked_test_set(["Cho2017", "EEGMAT"])
    assert partial.test["Cho2017"] == full.test["Cho2017"]
    assert partial.test["EEGMAT"] == full.test["EEGMAT"]


def test_committed_file_matches_recomputation() -> None:
    stored = load_locked_test_set(COMMITTED)
    recomputed = make_locked_test_set()
    assert dict(stored.test) == dict(recomputed.test)
    assert stored.digest == recomputed.digest
    assert stored.seed == LOCKED_TEST_SEED


def test_save_load_roundtrip_and_no_overwrite(tmp_path: Path) -> None:
    split = make_locked_test_set(["Cho2017"])
    path = tmp_path / "split.yaml"
    save_locked_test_set(split, path)
    assert load_locked_test_set(path).test == split.test
    with pytest.raises(NeurOSSplitError, match="never regenerated"):
        save_locked_test_set(split, path)


def test_tampering_is_detected(tmp_path: Path) -> None:
    path = tmp_path / "split.yaml"
    save_locked_test_set(make_locked_test_set(["Cho2017"]), path)
    first = make_locked_test_set(["Cho2017"]).test["Cho2017"][0]
    text = path.read_text(encoding="utf-8").replace(f"- {first}\n", "- 999\n", 1)
    path.write_text(text, encoding="utf-8")
    with pytest.raises(NeurOSSplitError, match="sha256 mismatch"):
        load_locked_test_set(path)


def test_malformed_and_missing_files(tmp_path: Path) -> None:
    with pytest.raises(NeurOSConfigError, match="not found"):
        load_locked_test_set(tmp_path / "missing.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("seed: 1\n", encoding="utf-8")
    with pytest.raises(NeurOSConfigError, match="malformed"):
        load_locked_test_set(bad)


def test_assert_disjoint() -> None:
    split = make_locked_test_set()
    locked = sorted(split.people())[0]
    with pytest.raises(NeurOSSplitError, match="training"):
        split.assert_disjoint([locked, "Cho2017:999"], purpose="training")
    dev = [f"Cho2017:{s}" for s in split.dev_subjects("Cho2017")]
    split.assert_disjoint(dev, purpose="training")  # no error
    assert len(dev) == 52 - 10


def test_exclusions_shrink_dev_pool_not_the_locked_set() -> None:
    split = make_locked_test_set()
    # D-027 excludes PhysionetMI 88, 92, 100; none was drawn into the locked test set.
    assert {88, 92, 100}.isdisjoint(split.test["PhysionetMI"])
    assert split.usable_test_subjects("PhysionetMI") == split.test["PhysionetMI"]
    dev = split.dev_subjects("PhysionetMI")
    assert len(dev) == 109 - 22 - 3
    assert {88, 92, 100}.isdisjoint(dev)


def test_dev_folds_partition_people() -> None:
    split = make_locked_test_set()
    people = {key: split.dev_subjects(key) for key in split.test}
    folds = make_dev_folds(people, n_folds=5)
    union = set().union(*folds)
    assert sum(len(f) for f in folds) == len(union)  # disjoint
    assert union == {f"{k}:{s}" for k, subjects in people.items() for s in subjects}
    for key in people:
        sizes = [sum(1 for p in fold if p.startswith(f"{key}:")) for fold in folds]
        assert max(sizes) - min(sizes) <= 1, (key, sizes)
    assert folds == make_dev_folds(people, n_folds=5)  # deterministic
    split.assert_disjoint(union, purpose="development")


def test_dev_folds_errors() -> None:
    with pytest.raises(NeurOSConfigError, match="n_folds"):
        make_dev_folds({"Cho2017": [1, 2, 3]}, n_folds=1)
    with pytest.raises(NeurOSConfigError, match="folds"):
        make_dev_folds({"Cho2017": [1, 2, 3]}, n_folds=5)


@given(
    st.dictionaries(
        st.text("ABC", min_size=1, max_size=3), st.integers(0, 200), min_size=1, max_size=5
    )
)
def test_allocation_property(sizes: dict[str, int]) -> None:
    counts = _allocate(sizes, 0.2)
    assert sum(counts.values()) == round(0.2 * sum(sizes.values()))
    for name, size in sizes.items():
        assert int(0.2 * size) <= counts[name] <= size
