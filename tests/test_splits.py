"""The locked test set and development folds (architecture §8.1, D-026)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from neuros.data.registry import get_spec, locked_test_dataset_keys
from neuros.errors import NeurOSConfigError, NeurOSSplitError
from neuros.eval.splits import (
    LOCKED_TEST_DATASETS,
    LOCKED_TEST_SEED,
    LOCKED_TEST_VERSION,
    Fold,
    _allocate,
    extends_check,
    load_locked_test_set,
    loso_folds,
    make_dev_folds,
    make_locked_test_set,
    save_locked_test_set,
)

from .conftest import REPO_ROOT

SPLITS = REPO_ROOT / "configs" / "splits"
COMMITTED_V1 = SPLITS / "locked_test_v1.yaml"
COMMITTED_V2 = SPLITS / "locked_test_v2.yaml"
COMMITTED = SPLITS / "locked_test_v3.yaml"
V1_COUNTS = {
    "PhysionetMI": 22,
    "Dreyer2023": 17,
    "Stieger2021": 12,
    "Lee2019_MI": 11,
    "Cho2017": 10,
    "EEGMAT": 7,
}
# D-033 (4 Oct): round(20%) of 51, 29 and 31 people.
V2_COUNTS = {**V1_COUNTS, "Yang2025": 10, "Shin2017A": 6, "GuttmannFlury2025": 6}
# D-038 (7 Oct): round(20%) of Liu2024's 50 people.
EXPECTED_COUNTS = {**V2_COUNTS, "Liu2024": 10}


def test_counts_are_twenty_percent_per_dataset() -> None:
    split = make_locked_test_set()
    assert split.version == LOCKED_TEST_VERSION == 3
    assert {k: len(v) for k, v in split.test.items()} == EXPECTED_COUNTS
    assert sum(EXPECTED_COUNTS.values()) == 111  # v2's 101 + 10 from Liu2024 (D-038)
    v2 = make_locked_test_set(version=2)
    assert {k: len(v) for k, v in v2.test.items()} == V2_COUNTS
    assert sum(V2_COUNTS.values()) == 101  # v1's 79 + 22 from D-033's datasets
    v1 = make_locked_test_set(version=1)
    assert {k: len(v) for k, v in v1.test.items()} == V1_COUNTS
    assert sum(V1_COUNTS.values()) == 79  # 72 motor-imagery + 7 EEGMAT people


def test_current_version_covers_every_contributing_dataset() -> None:
    # A newly registered dataset with a contributing role forces a new version (D-033).
    assert set(LOCKED_TEST_DATASETS[LOCKED_TEST_VERSION]) == set(locked_test_dataset_keys())
    with pytest.raises(NeurOSConfigError, match="Unknown locked test set version"):
        make_locked_test_set(version=99)


def test_v2_extends_v1_without_changing_it() -> None:
    v1, v2 = make_locked_test_set(version=1), make_locked_test_set(version=2)
    extends_check(v2, v1)  # no error
    assert {k: v2.test[k] for k in v1.test} == dict(v1.test)
    moved = {**v2.test, "Cho2017": (*v2.test["Cho2017"][1:], 1)}
    with pytest.raises(NeurOSSplitError, match="changes Cho2017"):
        extends_check(replace(v2, test=moved), v1)
    dropped = {k: v for k, v in v2.test.items() if k != "EEGMAT"}
    with pytest.raises(NeurOSSplitError, match="drops EEGMAT"):
        extends_check(replace(v2, test=dropped), v1)
    reseeded = make_locked_test_set(version=2, seed=LOCKED_TEST_SEED + 1)
    with pytest.raises(NeurOSSplitError, match="seed/fraction"):
        extends_check(reseeded, v1)


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


def test_committed_files_match_recomputation() -> None:
    for version, path in ((1, COMMITTED_V1), (2, COMMITTED_V2), (3, COMMITTED)):
        stored = load_locked_test_set(path)
        recomputed = make_locked_test_set(version=version)
        assert stored.version == version
        assert dict(stored.test) == dict(recomputed.test)
        assert stored.digest == recomputed.digest
        assert stored.seed == LOCKED_TEST_SEED
    v2 = load_locked_test_set(COMMITTED_V2)
    assert v2.extends == ("locked_test_v1.yaml", load_locked_test_set(COMMITTED_V1).digest)
    v3 = load_locked_test_set(COMMITTED)
    assert v3.extends == ("locked_test_v2.yaml", v2.digest)


def test_v2_detects_an_edited_v1(tmp_path: Path) -> None:
    # An edited v1 whose own digest was "fixed up" passes on its own, but v2 recorded the
    # original digest, so loading v2 next to it fails.
    from neuros.eval.splits import _digest

    v1 = load_locked_test_set(COMMITTED_V1)
    test = {**v1.test, "Cho2017": (1, 2, 3)}
    forged = replace(v1, test=test, digest=_digest(test, v1.seed, v1.fraction))
    save_locked_test_set(forged, tmp_path / "locked_test_v1.yaml")
    load_locked_test_set(tmp_path / "locked_test_v1.yaml")  # self-consistent
    copy = tmp_path / "locked_test_v2.yaml"
    copy.write_text(COMMITTED_V2.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(NeurOSSplitError, match=r"extends locked_test_v1\.yaml"):
        load_locked_test_set(copy)


def test_writing_a_later_version_needs_the_earlier_one(tmp_path: Path) -> None:
    v2 = make_locked_test_set(version=2)
    with pytest.raises(NeurOSConfigError, match="pass `extends`"):
        save_locked_test_set(v2, tmp_path / "v2.yaml")
    with pytest.raises(NeurOSConfigError, match="pass `extends`"):
        save_locked_test_set(
            make_locked_test_set(version=1), tmp_path / "v1.yaml", extends=COMMITTED_V1
        )
    save_locked_test_set(v2, tmp_path / "v2.yaml", extends=COMMITTED_V1)
    assert load_locked_test_set(tmp_path / "v2.yaml").extends is not None


def test_save_load_roundtrip_and_no_overwrite(tmp_path: Path) -> None:
    split = make_locked_test_set(["Cho2017"], version=1)
    path = tmp_path / "split.yaml"
    save_locked_test_set(split, path)
    assert load_locked_test_set(path).test == split.test
    with pytest.raises(NeurOSSplitError, match="never regenerated"):
        save_locked_test_set(split, path)


def test_tampering_is_detected(tmp_path: Path) -> None:
    path = tmp_path / "split.yaml"
    save_locked_test_set(make_locked_test_set(["Cho2017"], version=1), path)
    first = make_locked_test_set(["Cho2017"], version=1).test["Cho2017"][0]
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


# ---------------------------------------------------------------------- leave-one-person-out
PEOPLE = [f"BNCI2014_001:{i}" for i in range(1, 10)]


def test_loso_tests_everyone_once() -> None:
    folds = loso_folds(PEOPLE)
    assert [f.test for f in folds] == [(p,) for p in PEOPLE]
    for fold in folds:
        assert fold.validation == ()
        assert set(fold.train) == set(PEOPLE) - set(fold.test)


def test_loso_validation_people_come_from_training_people() -> None:
    folds = loso_folds(PEOPLE, n_validation=2, seed=7)
    for fold in folds:
        assert len(fold.validation) == 2
        assert len(fold.train) == 6
        assert set(fold.validation) | set(fold.train) == set(PEOPLE) - set(fold.test)
    # Deterministic, and each fold's draw doesn't depend on the order of the people.
    assert folds == loso_folds(PEOPLE, n_validation=2, seed=7)
    reordered = {f.test: f.validation for f in loso_folds(PEOPLE[::-1], n_validation=2, seed=7)}
    assert all(reordered[f.test] == f.validation for f in folds)
    assert folds != loso_folds(PEOPLE, n_validation=2, seed=8)


def test_loso_errors() -> None:
    with pytest.raises(NeurOSConfigError, match="unique"):
        loso_folds(["a", "a", "b"])
    with pytest.raises(NeurOSConfigError, match="at least 2"):
        loso_folds(["a"])
    with pytest.raises(NeurOSConfigError, match="n_validation"):
        loso_folds(["a", "b", "c"], n_validation=2)  # nobody left to train on
    assert len(loso_folds(["a", "b", "c"], n_validation=1)[0].train) == 1


def test_fold_refuses_overlaps() -> None:
    with pytest.raises(NeurOSSplitError, match="both test and train"):
        Fold(index=0, test=("a",), validation=(), train=("a", "b"))
    with pytest.raises(NeurOSSplitError, match="both validation and train"):
        Fold(index=0, test=("a",), validation=("b",), train=("b", "c"))
    with pytest.raises(NeurOSSplitError, match="non-empty"):
        Fold(index=0, test=("a",), validation=(), train=())
    assert Fold(0, ("a",), ("b",), ("c",)).to_dict()["validation"] == ["b"]
