"""Model windows: offline epochs, sliding windows and the streaming window buffer (§4.2).

A window is ``[channels, 256]`` samples (2.0 s at 128 Hz) of the preprocessed signal:

- **Offline task epochs** start at ``cue + 0.5 s`` (skipping the visual response to the cue)
  and end at ``cue + 2.5 s``: ``extract_epochs``.
- **Online**, the engine decides every hop (0.125 s = 16 samples) on the latest 256 samples:
  ``WindowStream``. ``sliding_windows`` is the offline equivalent; tests assert both yield the
  same windows from the same signal.

``WindowSet`` is the unit of training data: windows plus, *per window*, everything needed for
person-level splits and honest reporting (dataset, subject, session, run, label, start sample,
quality flags). Flagged windows are kept and counted, never silently dropped (§4.5); each
consumer decides what to do with them and reports it.
"""

from __future__ import annotations

import enum
import math
from collections import Counter
from collections.abc import Collection, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from neuros.config.schemas import SignalContract, ValidationThresholds
from neuros.data.recording import Recording, person_id
from neuros.errors import DataContext, NeurOSConfigError, NeurOSDataError
from neuros.signal._checks import check_chunk
from neuros.signal.filters import TRANSIENT_S

_SFREQ_RTOL = 1e-9


class WindowFlag(enum.IntFlag):
    """Quality flags per window (a bit mask; 0 = clean)."""

    NONE = 0
    # Some channel's peak-to-peak exceeds ``ValidationThresholds.ptp_flag_uv`` (in-band).
    EXTREME_AMPLITUDE = 1
    # Starts within ``filters.TRANSIENT_S`` of the filter (re)start.
    FILTER_TRANSIENT = 2


@dataclass(frozen=True, eq=False)
class WindowSet:
    """Windows ``[n, channels, samples]`` (µV, float32) with per-window metadata.

    All windows share the sampling rate, channel list and length; they may come from many
    people and datasets. Arrays are read-only. Compared by identity.
    """

    data: NDArray[np.float32]
    sfreq: float
    ch_names: tuple[str, ...]
    labels: tuple[str, ...]
    datasets: tuple[str, ...]
    subjects: NDArray[np.int64]
    sessions: tuple[str, ...]
    runs: tuple[str, ...]
    starts: NDArray[np.int64]
    flags: NDArray[np.uint8]

    def __post_init__(self) -> None:
        data = self.data
        if not isinstance(data, np.ndarray) or data.dtype != np.float32 or data.ndim != 3:
            shape = getattr(data, "shape", None)
            dtype = getattr(data, "dtype", None)
            raise NeurOSDataError(
                f"window data must be float32 [n, channels, samples], got {dtype} {shape}"
            )
        n, n_channels, n_samples = data.shape
        if n_channels == 0 or n_samples == 0:
            raise NeurOSDataError(f"windows must have channels and samples, got {data.shape}")
        if not (math.isfinite(self.sfreq) and self.sfreq > 0):
            raise NeurOSDataError(f"sfreq must be positive and finite, got {self.sfreq!r}")
        if len(self.ch_names) != n_channels or len(set(self.ch_names)) != n_channels:
            raise NeurOSDataError(
                f"{n_channels} channels but channel names {self.ch_names} (must be unique)"
            )
        for name, values in (
            ("labels", self.labels),
            ("datasets", self.datasets),
            ("sessions", self.sessions),
            ("runs", self.runs),
        ):
            if len(values) != n:
                raise NeurOSDataError(f"{n} windows but {len(values)} {name}")
        arrays: dict[str, NDArray[np.generic]] = {}
        for name, raw, dtype in (
            ("subjects", self.subjects, np.int64),
            ("starts", self.starts, np.int64),
            ("flags", self.flags, np.uint8),
        ):
            array = np.asarray(raw)
            if array.shape != (n,):
                raise NeurOSDataError(f"{n} windows but {name} has shape {array.shape}")
            if array.size and array.dtype.kind not in "iu":
                raise NeurOSDataError(f"{name} must be integers, got {array.dtype}")
            frozen = array.astype(dtype, copy=True)
            frozen.flags.writeable = False
            arrays[name] = frozen
        if np.any(np.less(arrays["subjects"], 0)) or np.any(np.less(arrays["starts"], 0)):
            raise NeurOSDataError("subjects and window starts must be >= 0")
        if not np.isfinite(data).all():
            raise NeurOSDataError("windows contain NaN/Inf samples")
        frozen_data = np.ascontiguousarray(data)
        frozen_data.flags.writeable = False
        object.__setattr__(self, "data", frozen_data)
        object.__setattr__(self, "sfreq", float(self.sfreq))
        for name in ("ch_names", "labels", "datasets", "sessions", "runs"):
            object.__setattr__(self, name, tuple(str(v) for v in getattr(self, name)))
        for name, frozen in arrays.items():
            object.__setattr__(self, name, frozen)

    # ------------------------------------------------------------------ views
    def __len__(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_samples(self) -> int:
        return int(self.data.shape[2])

    @property
    def person_ids(self) -> tuple[str, ...]:
        """``dataset:subject`` per window (the unit of every split)."""
        return tuple(
            person_id(d, int(s)) for d, s in zip(self.datasets, self.subjects, strict=True)
        )

    @property
    def people(self) -> tuple[str, ...]:
        """Distinct person ids, in order of first appearance."""
        return tuple(dict.fromkeys(self.person_ids))

    def label_counts(self) -> dict[str, int]:
        return dict(Counter(self.labels))

    def n_flagged(self, flag: WindowFlag | None = None) -> int:
        """Windows carrying ``flag`` (any flag if None)."""
        mask = self.flags != 0 if flag is None else (self.flags & int(flag)) != 0
        return int(np.count_nonzero(mask))

    # ------------------------------------------------------------------ selection
    def select(self, index: NDArray[np.bool_] | NDArray[np.integer] | Sequence[int]) -> WindowSet:
        """A new set with the windows at ``index`` (boolean mask or integer indices)."""
        idx = np.asarray(index)
        if idx.dtype == np.bool_:
            if idx.shape != (len(self),):
                raise NeurOSDataError(f"mask has shape {idx.shape}, expected ({len(self)},)")
            idx = np.flatnonzero(idx)
        elif idx.size and idx.dtype.kind not in "iu":
            raise NeurOSDataError(f"index must be a boolean mask or integers, got {idx.dtype}")
        idx = idx.astype(np.int64, copy=False)
        if idx.size and (idx.min() < -len(self) or idx.max() >= len(self)):
            raise NeurOSDataError(f"index out of range for {len(self)} windows")

        def pick(values: tuple[str, ...]) -> tuple[str, ...]:
            return tuple(values[i] for i in idx)

        return WindowSet(
            data=self.data[idx],
            sfreq=self.sfreq,
            ch_names=self.ch_names,
            labels=pick(self.labels),
            datasets=pick(self.datasets),
            subjects=self.subjects[idx],
            sessions=pick(self.sessions),
            runs=pick(self.runs),
            starts=self.starts[idx],
            flags=self.flags[idx],
        )

    def where_label(self, labels: Collection[str]) -> WindowSet:
        """Windows whose label is in ``labels``."""
        wanted = set(labels)
        return self.select(np.array([label in wanted for label in self.labels], dtype=bool))

    def for_people(self, people: Collection[str]) -> WindowSet:
        """Windows of the given ``dataset:subject`` people."""
        wanted = set(people)
        return self.select(np.array([p in wanted for p in self.person_ids], dtype=bool))

    def pick_channels(self, names: Sequence[str]) -> WindowSet:
        """Keep (and reorder to) ``names``.

        Raises:
            NeurOSDataError: if a requested channel is missing or repeated.
        """
        if len(set(names)) != len(names):
            raise NeurOSDataError(f"channel list has duplicates: {list(names)}")
        missing = [name for name in names if name not in self.ch_names]
        if missing:
            raise NeurOSDataError(f"channels {missing} are not in this window set")
        rows = [self.ch_names.index(name) for name in names]
        return WindowSet(
            data=self.data[:, rows, :],
            sfreq=self.sfreq,
            ch_names=tuple(names),
            labels=self.labels,
            datasets=self.datasets,
            subjects=self.subjects,
            sessions=self.sessions,
            runs=self.runs,
            starts=self.starts,
            flags=self.flags,
        )

    def with_data(self, data: NDArray[np.float32]) -> WindowSet:
        """Same metadata, new data of the same shape (e.g. after alignment)."""
        if not isinstance(data, np.ndarray) or data.shape != self.data.shape:
            raise NeurOSDataError(
                f"replacement data must have shape {self.data.shape}, "
                f"got {getattr(data, 'shape', None)}"
            )
        return WindowSet(
            data=data,
            sfreq=self.sfreq,
            ch_names=self.ch_names,
            labels=self.labels,
            datasets=self.datasets,
            subjects=self.subjects,
            sessions=self.sessions,
            runs=self.runs,
            starts=self.starts,
            flags=self.flags,
        )

    @classmethod
    def concat(cls, sets: Sequence[WindowSet]) -> WindowSet:
        """Concatenate window sets with the same rate, channels and window length.

        Raises:
            NeurOSDataError: on an empty list or mismatched sets.
        """
        if not sets:
            raise NeurOSDataError("concat needs at least one window set")
        first = sets[0]
        for other in sets[1:]:
            if not math.isclose(other.sfreq, first.sfreq, rel_tol=_SFREQ_RTOL):
                raise NeurOSDataError(f"can't concat {other.sfreq} Hz with {first.sfreq} Hz")
            if other.ch_names != first.ch_names:
                raise NeurOSDataError(
                    f"can't concat channels {other.ch_names} with {first.ch_names}"
                )
            if other.n_samples != first.n_samples:
                raise NeurOSDataError(
                    f"can't concat {other.n_samples}-sample windows with {first.n_samples}"
                )
        return cls(
            data=np.concatenate([s.data for s in sets], axis=0),
            sfreq=first.sfreq,
            ch_names=first.ch_names,
            labels=tuple(v for s in sets for v in s.labels),
            datasets=tuple(v for s in sets for v in s.datasets),
            subjects=np.concatenate([s.subjects for s in sets]),
            sessions=tuple(v for s in sets for v in s.sessions),
            runs=tuple(v for s in sets for v in s.runs),
            starts=np.concatenate([s.starts for s in sets]),
            flags=np.concatenate([s.flags for s in sets]),
        )


# ---------------------------------------------------------------------- offline windows
def window_starts(n_samples: int, window: int, hop: int) -> NDArray[np.int64]:
    """Start samples of every complete window: 0, hop, 2·hop, … (online decision times)."""
    if window < 1 or hop < 1:
        raise NeurOSConfigError(f"window and hop must be >= 1, got {window}, {hop}")
    if n_samples < window:
        return np.zeros(0, dtype=np.int64)
    return np.arange(0, n_samples - window + 1, hop, dtype=np.int64)


def sliding_windows(data: NDArray[np.floating], window: int, hop: int) -> NDArray[np.floating]:
    """All complete windows of ``data`` ``[channels, samples]`` as ``[w, channels, window]``.

    Windows start at 0, hop, 2·hop, …, exactly where ``WindowStream`` emits them. Returns a
    read-only *view* (no copy); use ``np.array(...)`` for a writable copy.
    """
    if data.ndim != 2:
        raise NeurOSDataError(f"data must be [channels, samples], got shape {data.shape}")
    starts = window_starts(data.shape[1], window, hop)
    if starts.size == 0:
        return np.zeros((0, data.shape[0], window), dtype=data.dtype)
    view = np.lib.stride_tricks.sliding_window_view(data, window, axis=1)  # [ch, n-w+1, w]
    return view[:, ::hop, :].transpose(1, 0, 2)


def _flags_for(
    windows: NDArray[np.floating], starts: NDArray[np.int64], sfreq: float, ptp_flag_uv: float
) -> NDArray[np.uint8]:
    flags = np.zeros(len(starts), dtype=np.uint8)
    if len(starts):
        ptp = (windows.max(axis=2) - windows.min(axis=2)).max(axis=1)
        flags[ptp > ptp_flag_uv] |= np.uint8(WindowFlag.EXTREME_AMPLITUDE)
        flags[starts < round(TRANSIENT_S * sfreq)] |= np.uint8(WindowFlag.FILTER_TRANSIENT)
    return flags


def _require_contract_rate(recording: Recording, contract: SignalContract) -> None:
    if not math.isclose(recording.sfreq, contract.sfreq, rel_tol=_SFREQ_RTOL):
        raise NeurOSDataError(
            f"windows come from the preprocessed {contract.sfreq:g} Hz signal, but this "
            f"recording is at {recording.sfreq:g} Hz (run preprocess_recording first)",
            recording.context,
        )


def extract_epochs(
    recording: Recording,
    contract: SignalContract | None = None,
    thresholds: ValidationThresholds | None = None,
    *,
    labels: Collection[str] | None = None,
) -> tuple[WindowSet, int]:
    """Cut one window per task event at ``[cue + epoch_s[0], cue + epoch_s[1]]``.

    Args:
        recording: a *preprocessed* recording (contract rate; events at the cue).
        contract: the signal contract (epoch timing, window length).
        thresholds: ``ptp_flag_uv`` for the extreme-amplitude flag.
        labels: keep only events with these labels (default: all).

    Returns:
        ``(windows, skipped)``: ``skipped`` counts selected events whose epoch runs outside
        the recording. They're reported by the caller, never silently lost.

    Raises:
        NeurOSDataError: if the recording isn't at the contract rate.
    """
    contract = contract or SignalContract()
    thresholds = thresholds or ValidationThresholds()
    _require_contract_rate(recording, contract)
    window = contract.window_samples
    offset = round(contract.epoch_s[0] * recording.sfreq)
    wanted = None if labels is None else set(labels)

    starts: list[int] = []
    chosen: list[str] = []
    skipped = 0
    for onset, label in zip(recording.events.onsets, recording.events.labels, strict=True):
        if wanted is not None and label not in wanted:
            continue
        start = int(onset) + offset
        if start < 0 or start + window > recording.n_samples:
            skipped += 1
            continue
        starts.append(start)
        chosen.append(label)
    start_array = np.array(starts, dtype=np.int64)
    if starts:
        data = np.stack([recording.data[:, s : s + window] for s in starts]).astype(np.float32)
    else:
        data = np.zeros((0, recording.n_channels, window), dtype=np.float32)
    return (
        _window_set(recording, data, start_array, tuple(chosen), thresholds),
        skipped,
    )


def continuous_windows(
    recording: Recording,
    contract: SignalContract | None = None,
    thresholds: ValidationThresholds | None = None,
    *,
    label: str | None = None,
) -> WindowSet:
    """Every window the online engine would decide on (start 0, hop, 2·hop, …).

    Used for replay-style evaluation and for unlabeled data (e.g. alignment references). Each
    window gets ``label`` (default: ``"unlabeled"``).

    Raises:
        NeurOSDataError: if the recording isn't at the contract rate.
    """
    contract = contract or SignalContract()
    thresholds = thresholds or ValidationThresholds()
    _require_contract_rate(recording, contract)
    data = np.array(
        sliding_windows(recording.data, contract.window_samples, contract.hop_samples),
        dtype=np.float32,
    )
    starts = window_starts(recording.n_samples, contract.window_samples, contract.hop_samples)
    names = (label or "unlabeled",) * len(starts)
    return _window_set(recording, data, starts, names, thresholds)


def _window_set(
    recording: Recording,
    data: NDArray[np.float32],
    starts: NDArray[np.int64],
    labels: tuple[str, ...],
    thresholds: ValidationThresholds,
) -> WindowSet:
    n = len(starts)
    return WindowSet(
        data=data,
        sfreq=recording.sfreq,
        ch_names=recording.ch_names,
        labels=labels,
        datasets=(recording.dataset,) * n,
        subjects=np.full(n, recording.subject, dtype=np.int64),
        sessions=(recording.session,) * n,
        runs=(recording.run,) * n,
        starts=starts,
        flags=_flags_for(data, starts, recording.sfreq, thresholds.ptp_flag_uv),
    )


# ---------------------------------------------------------------------- streaming windows
class WindowStream:
    """Emits the latest ``window`` samples every ``hop`` samples (the engine's decision clock).

    Windows end at sample counts ``window``, ``window + hop``, ``window + 2·hop``, … since the
    last reset, i.e. they start at 0, hop, 2·hop, … like ``sliding_windows``. Keeps only the
    last ``window - 1`` samples between pushes.
    """

    def __init__(
        self, n_channels: int, window: int, hop: int, context: DataContext | None = None
    ) -> None:
        if not isinstance(n_channels, int) or n_channels < 1:
            raise NeurOSConfigError(f"n_channels must be a positive int, got {n_channels!r}")
        if window < 1 or hop < 1:
            raise NeurOSConfigError(f"window and hop must be >= 1, got {window}, {hop}")
        self.n_channels = n_channels
        self.window = window
        self.hop = hop
        self.context = context or DataContext()
        self.reset()

    def reset(self) -> None:
        """Forget buffered samples; the next window needs ``window`` new samples."""
        self._buffer = np.zeros((self.n_channels, 0), dtype=np.float32)
        self.samples_seen = 0

    def push(self, chunk: NDArray[np.floating]) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
        """Add samples; return ``(windows [w, channels, window], start samples [w])``.

        Samples are stored as float32 (the contract dtype), exactly as offline windows are.

        Raises:
            NeurOSSignalError: on a wrong shape or NaN/Inf samples (state unchanged).
        """
        x = check_chunk(chunk, self.n_channels, self.context).astype(np.float32)
        combined = np.concatenate([self._buffer, x], axis=1)
        first_global = self.samples_seen - self._buffer.shape[1]  # global index of combined[0]
        before, after = self.samples_seen, self.samples_seen + x.shape[1]
        # Window starts s = j·hop whose end s + window falls in (before, after].
        lowest = max(0, -(-(before + 1 - self.window) // self.hop))  # ceil division
        highest = (after - self.window) // self.hop if after >= self.window else -1
        starts = np.arange(lowest, highest + 1, dtype=np.int64) * self.hop
        if starts.size:
            local = starts - first_global
            windows = np.stack([combined[:, s : s + self.window] for s in local])
        else:
            windows = np.zeros((0, self.n_channels, self.window), dtype=np.float32)
        keep = min(self.window - 1, combined.shape[1])
        self._buffer = combined[:, combined.shape[1] - keep :].copy()
        self.samples_seen = after
        return windows, starts
