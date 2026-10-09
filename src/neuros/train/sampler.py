"""Subject-balanced batches: K people × M windows, class-balanced within each person (§6.4).

**Why.** Pooled training data is lopsided: PhysionetMI alone has 106 people, and people differ
in how many trials they recorded. A plain shuffle lets the biggest datasets, and the people
with the most trials, dominate every gradient. Sampling *people* uniformly, then windows
class-balanced within each person, gives every person and every class equal weight. That's
also what the evaluation counts (statistics over people, §8.2). The K × M layout is what
domain-specific batch norm needs later (per-person statistics within a batch, §6.1, E3b).

**Mechanics**, all driven by one seeded numpy generator, so runs are reproducible:

- each batch draws K distinct people uniformly at random (K is capped at the number of people);
- each chosen person contributes M windows, split as evenly as possible across the classes
  that person has; any remainder goes to randomly chosen classes;
- windows come from a shuffled queue per (person, class) that is reshuffled when it runs out,
  so a person's windows are all used once before any is used twice;
- an epoch is ``ceil(n_windows / (K × M))`` batches: about one pass over the data.

With every person balanced to begin with (BNCI2014-001: 144 + 144 windows each), this behaves
like an ordinary shuffle; it starts to matter when datasets are pooled (E1).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from neuros.errors import NeurOSConfigError, NeurOSDataError

log = logging.getLogger(__name__)


@dataclass
class _Queue:
    """A shuffled, self-refilling queue over one (person, class) pool of window indices."""

    pool: NDArray[np.int64]
    order: NDArray[np.int64]
    cursor: int = 0


class SubjectBalancedSampler:
    """Draws batches of window indices: K people × M windows, class-balanced per person.

    Args:
        people: the person id of every window (``dataset:subject``).
        labels: the class of every window (any hashable values; class indices in practice).
        people_per_batch: K. Capped at the number of distinct people (logged).
        windows_per_person: M.
        seed: seeds the generator for people, class remainders and queue shuffles.
        batches_per_epoch: default ``ceil(epoch_windows / (K × M))``.
        epoch_windows: how many windows an epoch covers; default all of them. With D-039's
            training crops it is one per trial, so crops don't lengthen epochs.

    Raises:
        NeurOSDataError: on empty or mismatched inputs.
        NeurOSConfigError: on non-positive K, M or batches per epoch, or ``epoch_windows``
            outside [1, n_windows].
    """

    def __init__(
        self,
        people: Sequence[str] | NDArray[np.str_],
        labels: Sequence[Hashable] | NDArray[np.generic],
        *,
        people_per_batch: int,
        windows_per_person: int,
        seed: int,
        batches_per_epoch: int | None = None,
        epoch_windows: int | None = None,
    ) -> None:
        person_of = np.asarray(people)
        label_of = np.asarray(labels)
        if person_of.ndim != 1 or label_of.ndim != 1:
            raise NeurOSDataError("people and labels must be 1-D (one entry per window)")
        if person_of.shape != label_of.shape:
            raise NeurOSDataError(
                f"{person_of.size} person ids but {label_of.size} labels: one each per window"
            )
        if person_of.size == 0:
            raise NeurOSDataError("no windows to sample from")
        if people_per_batch < 1 or windows_per_person < 1:
            raise NeurOSConfigError(
                f"people_per_batch and windows_per_person must be >= 1, "
                f"got {people_per_batch} and {windows_per_person}"
            )
        if batches_per_epoch is not None and batches_per_epoch < 1:
            raise NeurOSConfigError(f"batches_per_epoch must be >= 1, got {batches_per_epoch}")
        if epoch_windows is not None and not 1 <= epoch_windows <= person_of.size:
            raise NeurOSConfigError(
                f"epoch_windows must be in [1, {person_of.size}], got {epoch_windows}"
            )

        self._rng = np.random.default_rng(seed)
        # Person order = first appearance; class order = sorted. Both deterministic.
        self.people: tuple[str, ...] = tuple(dict.fromkeys(person_of.tolist()))
        self._classes: list[list[Hashable]] = []
        self._queues: dict[tuple[int, Hashable], _Queue] = {}
        for p, person in enumerate(self.people):
            mine = person_of == person
            classes = sorted(set(label_of[mine].tolist()))
            self._classes.append(classes)
            for label in classes:
                pool = np.flatnonzero(mine & (label_of == label)).astype(np.int64)
                self._queues[(p, label)] = _Queue(pool=pool, order=self._rng.permutation(pool))

        self.requested_people_per_batch = people_per_batch
        self.people_per_batch = min(people_per_batch, len(self.people))
        if self.people_per_batch < people_per_batch:
            log.info(
                "people_per_batch %d capped at the %d people available",
                people_per_batch,
                len(self.people),
            )
        self.windows_per_person = windows_per_person
        self.batch_size = self.people_per_batch * windows_per_person
        self.n_windows = int(person_of.size)
        self.epoch_windows = epoch_windows or self.n_windows
        self.batches_per_epoch = batches_per_epoch or max(
            1, math.ceil(self.epoch_windows / self.batch_size)
        )

    def _take(self, key: tuple[int, Hashable], count: int) -> NDArray[np.int64]:
        """``count`` indices from one queue, reshuffling whenever it runs out.

        If ``count`` exceeds the pool (a person with few windows of a class), the pool is
        reshuffled mid-batch and some windows repeat within the batch; that's the only way to
        keep the person's weight equal to everyone else's.
        """
        queue = self._queues[key]
        parts: list[NDArray[np.int64]] = []
        while count > 0:
            if queue.cursor == queue.order.size:
                queue.order = self._rng.permutation(queue.pool)
                queue.cursor = 0
            n = min(count, queue.order.size - queue.cursor)
            parts.append(queue.order[queue.cursor : queue.cursor + n])
            queue.cursor += n
            count -= n
        return np.concatenate(parts)

    def batch(self) -> NDArray[np.int64]:
        """One batch of ``K × M`` window indices (grouped by person)."""
        chosen = self._rng.choice(len(self.people), size=self.people_per_batch, replace=False)
        parts: list[NDArray[np.int64]] = []
        for p in chosen.tolist():
            classes = self._classes[p]
            base, extra = divmod(self.windows_per_person, len(classes))
            counts = np.full(len(classes), base, dtype=np.int64)
            if extra:
                counts[self._rng.choice(len(classes), size=extra, replace=False)] += 1
            for label, count in zip(classes, counts.tolist(), strict=True):
                if count:
                    parts.append(self._take((p, label), count))
        return np.concatenate(parts)

    def epoch(self) -> list[NDArray[np.int64]]:
        """``batches_per_epoch`` batches (about one pass over the data)."""
        return [self.batch() for _ in range(self.batches_per_epoch)]
