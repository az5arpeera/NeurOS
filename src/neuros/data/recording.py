"""The canonical in-memory recording: data plus everything needed to trust it.

Architecture §3, principle 3: every array carries its metadata (units, sampling rate, channel
names, dataset/subject/session/run). A ``Recording`` is the only way continuous EEG moves between
NeurOS components offline.

Division of labour:

- **Construction** (here) enforces *structural* invariants: shapes, dtype, canonical unique
  channel names, events inside the recording. A structurally broken object can't exist.
- **Validation** (``neuros.data.validate``) checks *content*: units, NaN/Inf, flat channels,
  amplitudes, sampling rate vs the registry. It reports problems with full context instead of
  failing on the first one, so a 380-person run produces a complete diagnosis.

Arrays are made read-only on construction: windows and caches are derived from recordings, and
an in-place edit would silently desynchronise them.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from neuros.data.channels import require_canonical
from neuros.errors import DataContext, NeurOSDataError


@dataclass(frozen=True, eq=False)
class EventTable:
    """Labelled events of one recording.

    ``onsets`` are sample indices of the **cue onset** in the recording (dataset-specific marker
    offsets are already applied by harmonisation), sorted ascending. ``labels`` are canonical
    labels (``neuros.data.vocabulary``).

    Equality is by value (the generated dataclass ``==`` would compare numpy arrays
    element-wise and fail on truth-testing).
    """

    onsets: NDArray[np.int64]
    labels: tuple[str, ...]

    def __post_init__(self) -> None:
        onsets = np.asarray(self.onsets)
        if onsets.ndim != 1:
            raise NeurOSDataError(f"Event onsets must be 1-D, got shape {onsets.shape}")
        if onsets.size and onsets.dtype.kind not in "iu":
            raise NeurOSDataError(f"Event onsets must be integers, got dtype {onsets.dtype}")
        if len(self.labels) != onsets.size:
            raise NeurOSDataError(f"{onsets.size} event onsets but {len(self.labels)} labels")
        if np.any(onsets < 0):
            raise NeurOSDataError("Event onsets must be >= 0")
        if onsets.size > 1 and np.any(np.diff(onsets) < 0):
            raise NeurOSDataError("Event onsets must be sorted ascending")
        frozen = onsets.astype(np.int64, copy=True)
        frozen.flags.writeable = False
        object.__setattr__(self, "onsets", frozen)
        object.__setattr__(self, "labels", tuple(self.labels))

    def __len__(self) -> int:
        return len(self.labels)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EventTable):
            return NotImplemented
        return self.labels == other.labels and np.array_equal(self.onsets, other.onsets)

    def __hash__(self) -> int:
        return hash((self.onsets.tobytes(), self.labels))

    @classmethod
    def empty(cls) -> EventTable:
        return cls(onsets=np.zeros(0, dtype=np.int64), labels=())

    @classmethod
    def from_pairs(cls, pairs: Sequence[tuple[int, str]]) -> EventTable:
        """Build from (onset, label) pairs in any order (sorted stably by onset)."""
        ordered = sorted(pairs, key=lambda pair: pair[0])
        return cls(
            onsets=np.array([onset for onset, _ in ordered], dtype=np.int64),
            labels=tuple(label for _, label in ordered),
        )

    def counts(self) -> dict[str, int]:
        """Number of events per label."""
        return dict(Counter(self.labels))


@dataclass(frozen=True, eq=False)
class Recording:
    """Continuous EEG in the canonical contract: µV, float32, ``[channels, samples]``.

    Compared by identity: value equality over large arrays is never what a caller wants
    implicitly; compare ``data`` explicitly with ``np.array_equal`` when needed.
    """

    data: NDArray[np.float32]
    sfreq: float
    ch_names: tuple[str, ...]
    events: EventTable
    dataset: str
    subject: int
    session: str
    run: str
    provenance: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        context = DataContext(self.dataset, self.subject, self.session, self.run)
        if not isinstance(self.data, np.ndarray):
            raise NeurOSDataError(
                f"data must be a numpy array, got {type(self.data).__name__}", context
            )
        if self.data.dtype != np.float32:
            raise NeurOSDataError(f"data must be float32 µV, got {self.data.dtype}", context)
        if self.data.ndim != 2:
            raise NeurOSDataError(
                f"data must be [channels, samples], got shape {self.data.shape}", context
            )
        n_channels, n_samples = self.data.shape
        if n_channels == 0 or n_samples == 0:
            raise NeurOSDataError(f"data is empty: shape {self.data.shape}", context)
        if not (
            isinstance(self.sfreq, (int, float)) and math.isfinite(self.sfreq) and self.sfreq > 0
        ):
            raise NeurOSDataError(
                f"sfreq must be a positive finite number, got {self.sfreq!r}", context
            )
        names = require_canonical(self.ch_names, context)
        if len(names) != n_channels:
            raise NeurOSDataError(f"{n_channels} data rows but {len(names)} channel names", context)
        if not isinstance(self.subject, int) or self.subject < 0:
            raise NeurOSDataError(
                f"subject must be a non-negative int, got {self.subject!r}", context
            )
        if len(self.events) and int(self.events.onsets[-1]) >= n_samples:
            raise NeurOSDataError(
                f"event at sample {int(self.events.onsets[-1])} is beyond the recording "
                f"({n_samples} samples)",
                context,
            )
        data = np.ascontiguousarray(self.data)
        data.flags.writeable = False
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "ch_names", names)
        object.__setattr__(self, "sfreq", float(self.sfreq))
        object.__setattr__(self, "provenance", dict(self.provenance))

    # ------------------------------------------------------------------ convenience
    @property
    def n_channels(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_samples(self) -> int:
        return int(self.data.shape[1])

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.sfreq

    @property
    def context(self) -> DataContext:
        return DataContext(self.dataset, self.subject, self.session, self.run)

    @property
    def person_id(self) -> str:
        """``dataset:subject``: the unit of every split (architecture §3, principle 2)."""
        return person_id(self.dataset, self.subject)


def person_id(dataset: str, subject: int) -> str:
    """Canonical person identifier used by splits and reports."""
    return f"{dataset}:{subject}"
