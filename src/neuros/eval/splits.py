"""Person-level splits: the locked test set and development folds (architecture §8.1).

Principle 2 (person-level isolation): no data from a test person touches training,
normalisation fitting, model selection or early stopping. Splits are by *person*, never by
trial, and identified by ``dataset:subject`` strings.

**Locked test set (D-026).** 20% of the people in each dataset that contributes (the five core
motor-imagery datasets + EEGMAT), drawn with seed ``20261003`` on 2026-10-03, before any data
was examined, and stratified by dataset (and by subgroup where the registry defines one, e.g.
Dreyer2023 A/B/C). It's written once to ``configs/splits/locked_test_v1.yaml`` with a SHA-256
digest, and stays untouched until the freeze on Sat 24 Oct.

Determinism: each dataset draws from its own RNG, seeded by hashing ``seed:dataset``. Adding a
dataset therefore never changes another dataset's test people.

Per-stratum counts use largest-remainder rounding, so each dataset's total is exactly
``round(fraction × n)`` while every stratum gets its fair share.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from neuros.data.recording import person_id
from neuros.data.registry import DatasetSpec, get_spec, locked_test_dataset_keys
from neuros.errors import NeurOSConfigError, NeurOSSplitError

LOCKED_TEST_SEED = 20261003
LOCKED_TEST_FRACTION = 0.2
LOCKED_TEST_VERSION = 1
SPLIT_METHOD = "per-dataset sha256(seed:dataset) RNG; stratified by subgroup; largest-remainder"


def _dataset_rng(seed: int, dataset: str) -> np.random.Generator:
    """An RNG that depends only on (seed, dataset)."""
    digest = hashlib.sha256(f"{seed}:{dataset}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def _allocate(stratum_sizes: Mapping[str, int], fraction: float) -> dict[str, int]:
    """Largest-remainder allocation of round(fraction × total) across strata."""
    total = sum(stratum_sizes.values())
    target = round(fraction * total)
    exact = {name: fraction * size for name, size in stratum_sizes.items()}
    counts = {name: math.floor(value) for name, value in exact.items()}
    remainder = target - sum(counts.values())
    # Ties broken by stratum name so the allocation is deterministic.
    order = sorted(exact, key=lambda name: (-(exact[name] - counts[name]), name))
    for name in order[: max(remainder, 0)]:
        counts[name] += 1
    return counts


def draw_test_people(spec: DatasetSpec, fraction: float, seed: int) -> list[int]:
    """Draw one dataset's test subjects, stratified by the registry's subgroups."""
    if not 0 < fraction < 1:
        raise NeurOSConfigError(f"fraction must be in (0, 1), got {fraction}")
    strata: dict[str, list[int]] = {}
    for subject in spec.subjects:
        strata.setdefault(spec.subgroup_of(subject), []).append(subject)
    counts = _allocate({name: len(members) for name, members in strata.items()}, fraction)
    rng = _dataset_rng(seed, spec.key)
    chosen: list[int] = []
    for name in sorted(strata):
        members = sorted(strata[name])
        order = rng.permutation(len(members))
        chosen.extend(members[i] for i in order[: counts[name]])
    return sorted(chosen)


def _digest(test: Mapping[str, Sequence[int]], seed: int, fraction: float) -> str:
    canonical = json.dumps(
        {
            "fraction": fraction,
            "seed": seed,
            "test": {k: sorted(v) for k, v in sorted(test.items())},
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass(frozen=True)
class LockedTestSet:
    """The locked test people, loaded and integrity-checked."""

    version: int
    seed: int
    fraction: float
    created: str
    test: Mapping[str, tuple[int, ...]]
    digest: str

    def people(self) -> frozenset[str]:
        """All locked test people as ``dataset:subject`` ids."""
        return frozenset(person_id(ds, s) for ds, subjects in self.test.items() for s in subjects)

    def contains(self, dataset: str, subject: int) -> bool:
        return subject in self.test.get(dataset, ())

    def dev_subjects(self, dataset: str) -> tuple[int, ...]:
        """Subjects of ``dataset`` available for development: usable and not locked.

        The locked set was drawn from *declared* subjects before any data was examined, so a
        later data-quality exclusion (registry ``excluded_subjects``) never changes it; it only
        shrinks the development pool (and, if it ever hit a test person, the test N).
        """
        spec = get_spec(dataset)
        locked = set(self.test.get(dataset, ()))
        return tuple(s for s in spec.usable_subjects if s not in locked)

    def usable_test_subjects(self, dataset: str) -> tuple[int, ...]:
        """Locked test subjects of ``dataset`` that aren't excluded for data quality."""
        excluded = get_spec(dataset).excluded_subjects
        return tuple(s for s in self.test.get(dataset, ()) if s not in excluded)

    def assert_disjoint(self, people: Iterable[str], purpose: str) -> None:
        """Raise if any of ``people`` (``dataset:subject`` ids) is a locked test person.

        Call this wherever data enters training, normalisation fitting, model selection or
        threshold tuning.

        Raises:
            NeurOSSplitError: listing the offending people. Never catch and ignore it.
        """
        leaked = sorted(set(people) & self.people())
        if leaked:
            raise NeurOSSplitError(
                f"{len(leaked)} locked test people would be used for {purpose}: {leaked[:10]}"
                f"{' ...' if len(leaked) > 10 else ''}"
            )


def make_locked_test_set(
    dataset_keys: Sequence[str] | None = None,
    *,
    fraction: float = LOCKED_TEST_FRACTION,
    seed: int = LOCKED_TEST_SEED,
    created: str | None = None,
) -> LockedTestSet:
    """Compute the locked test set from the registry (pure: doesn't touch the disk)."""
    keys = tuple(dataset_keys) if dataset_keys is not None else locked_test_dataset_keys()
    test = {key: tuple(draw_test_people(get_spec(key), fraction, seed)) for key in keys}
    return LockedTestSet(
        version=LOCKED_TEST_VERSION,
        seed=seed,
        fraction=fraction,
        created=created or dt.date.today().isoformat(),
        test=test,
        digest=_digest(test, seed, fraction),
    )


def save_locked_test_set(split: LockedTestSet, path: str | Path) -> None:
    """Write the split once. Refuses to overwrite: the test set is locked (D-026).

    Raises:
        NeurOSSplitError: if the file already exists.
    """
    file = Path(path)
    if file.exists():
        raise NeurOSSplitError(f"{file} exists; the locked test set is never regenerated in place")
    file.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "version": split.version,
        "created": split.created,
        "seed": split.seed,
        "fraction": split.fraction,
        "method": SPLIT_METHOD,
        "decision": "D-026",
        "n_people": {key: len(subjects) for key, subjects in split.test.items()},
        "test": {key: list(subjects) for key, subjects in split.test.items()},
        "sha256": split.digest,
    }
    header = (
        "# LOCKED TEST SET (architecture §8.1, D-026). Do not edit by hand.\n"
        "# These people are untouched until the freeze on Sat 24 Oct 2026.\n"
        "# Integrity: neuros.eval.splits.load_locked_test_set() recomputes sha256.\n"
    )
    # newline="\n": identical bytes on every OS (Windows would otherwise write CRLF).
    file.write_text(header + yaml.safe_dump(body, sort_keys=False), encoding="utf-8", newline="\n")


def load_locked_test_set(path: str | Path) -> LockedTestSet:
    """Load and integrity-check the locked test set.

    Raises:
        NeurOSConfigError: if the file is missing or malformed.
        NeurOSSplitError: if the digest doesn't match the contents (the file was edited).
    """
    file = Path(path)
    if not file.is_file():
        raise NeurOSConfigError(f"Locked test set not found: {file} (run `neuros split --write`)")
    try:
        raw = yaml.safe_load(file.read_text(encoding="utf-8"))
        test = {str(k): tuple(int(s) for s in v) for k, v in raw["test"].items()}
        seed, fraction, digest = int(raw["seed"]), float(raw["fraction"]), str(raw["sha256"])
        version, created = int(raw["version"]), str(raw["created"])
    except (yaml.YAMLError, KeyError, TypeError, ValueError, AttributeError) as exc:
        raise NeurOSConfigError(f"{file}: malformed locked test set: {exc}") from exc
    if _digest(test, seed, fraction) != digest:
        raise NeurOSSplitError(f"{file}: sha256 mismatch; the locked test set was modified")
    return LockedTestSet(version, seed, fraction, created, test, digest)


@dataclass(frozen=True)
class Fold:
    """One person-level split: who trains, who validates (early stopping), who is tested.

    Raises:
        NeurOSSplitError: if the three groups overlap, or the train or test group is empty.
    """

    index: int
    test: tuple[str, ...]
    validation: tuple[str, ...]
    train: tuple[str, ...]

    def __post_init__(self) -> None:
        groups = {
            "test": set(self.test),
            "validation": set(self.validation),
            "train": set(self.train),
        }
        if not self.test or not self.train:
            raise NeurOSSplitError(f"fold {self.index}: train and test must both be non-empty")
        for a, b in (("test", "train"), ("test", "validation"), ("validation", "train")):
            shared = groups[a] & groups[b]
            if shared:
                raise NeurOSSplitError(f"fold {self.index}: {sorted(shared)} in both {a} and {b}")

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "test": list(self.test),
            "validation": list(self.validation),
            "train": list(self.train),
        }


def loso_folds(
    people: Sequence[str], *, n_validation: int = 0, seed: int = LOCKED_TEST_SEED
) -> list[Fold]:
    """Leave-one-person-out folds, in the order of ``people``.

    Each fold tests one person. If ``n_validation`` > 0, that many validation people (for early
    stopping and monitoring) are drawn from the fold's *training* people, with an RNG that
    depends only on (seed, test person): reordering ``people`` changes no fold's draw.

    Raises:
        NeurOSConfigError: on duplicate people, fewer than 2 people, or so many validation
            people that no one would be left to train on.
    """
    order = list(people)
    if len(set(order)) != len(order):
        raise NeurOSConfigError("people must be unique")
    if len(order) < 2:
        raise NeurOSConfigError("leave-one-person-out needs at least 2 people")
    if n_validation < 0 or n_validation > len(order) - 2:
        raise NeurOSConfigError(
            f"n_validation must be in [0, {len(order) - 2}] for {len(order)} people "
            f"(at least one training person per fold), got {n_validation}"
        )
    folds: list[Fold] = []
    for index, test in enumerate(order):
        rest = sorted(p for p in order if p != test)
        validation: tuple[str, ...] = ()
        if n_validation:
            rng = _dataset_rng(seed, f"loso-validation:{test}")
            picks = rng.choice(len(rest), size=n_validation, replace=False)
            validation = tuple(sorted(rest[i] for i in picks))
        train = tuple(p for p in rest if p not in validation)
        folds.append(Fold(index=index, test=(test,), validation=validation, train=train))
    return folds


def make_dev_folds(
    people: Mapping[str, Sequence[int]], n_folds: int = 5, seed: int = LOCKED_TEST_SEED
) -> list[frozenset[str]]:
    """Subject-wise folds for development, stratified by dataset (architecture §8.1).

    ``people`` maps dataset → its *development* subjects (locked test people already removed).
    Within each dataset, subjects are shuffled and dealt round-robin, starting at a
    dataset-dependent fold so small datasets don't all pile into fold 0.

    Raises:
        NeurOSConfigError: if ``n_folds`` < 2 or a dataset has fewer subjects than folds.
    """
    if n_folds < 2:
        raise NeurOSConfigError("n_folds must be >= 2")
    folds: list[set[str]] = [set() for _ in range(n_folds)]
    for dataset in sorted(people):
        subjects = sorted(people[dataset])
        if len(subjects) < n_folds:
            raise NeurOSConfigError(f"{dataset}: {len(subjects)} subjects < {n_folds} folds")
        rng = _dataset_rng(seed + 1, dataset)
        offset = int(rng.integers(n_folds))
        for position, index in enumerate(rng.permutation(len(subjects))):
            folds[(position + offset) % n_folds].add(person_id(dataset, subjects[index]))
    return [frozenset(fold) for fold in folds]
