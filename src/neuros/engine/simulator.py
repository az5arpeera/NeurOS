"""The simulator, a "virtual headset" (architecture §7.4).

Replays a continuous recording as the timestamped chunks a headset stream delivers, cut to a
chosen set of channels (a headset profile), optionally with added noise, plus the ground-truth
markers (each cue's time and class) a second stream would carry.

What it's for: end-to-end tests, demo rehearsal before the headset arrives, and replay
evaluation through the exact engine users run (§7.4). Transport-free: ``chunks()`` yields what
an LSL inlet's ``pull_chunk`` would return; publishing it as a real LSL stream wraps this.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from neuros.data.recording import Recording
from neuros.engine.events import Event
from neuros.engine.pipeline import Engine
from neuros.errors import NeurOSConfigError, NeurOSDataError


@dataclass(frozen=True)
class Marker:
    """Ground truth: a cue's stream time and class (never shown to the engine)."""

    t: float
    label: str


class Simulator:
    """Replays one recording (see module docs)."""

    def __init__(
        self,
        recording: Recording,
        *,
        channels: Sequence[str] | None = None,
        chunk_s: float = 1.0 / 16,
        t0: float = 0.0,
        noise_uv: float = 0.0,
        seed: int = 0,
    ) -> None:
        """
        Args:
            recording: a continuous recording at its native rate (µV).
            channels: the headset's channels, in stream order (default: all of them).
            chunk_s: samples per chunk, in seconds (LSL inlets typically pull 1/16-1/10 s).
            t0: stream time of the first sample.
            noise_uv: standard deviation of added white noise (µV), 0 for none.
            seed: the noise generator's seed (replays are reproducible).

        Raises:
            NeurOSConfigError: on bad settings or channels the recording lacks.
        """
        names = tuple(channels) if channels is not None else tuple(recording.ch_names)
        missing = [ch for ch in names if ch not in recording.ch_names]
        if missing:
            raise NeurOSConfigError(f"the recording lacks channels {missing}")
        if len(set(names)) != len(names) or not names:
            raise NeurOSConfigError(f"channels must be distinct and non-empty: {list(names)}")
        if not (math.isfinite(chunk_s) and chunk_s > 0):
            raise NeurOSConfigError(f"chunk_s must be > 0, got {chunk_s}")
        if not (math.isfinite(noise_uv) and noise_uv >= 0):
            raise NeurOSConfigError(f"noise_uv must be >= 0, got {noise_uv}")
        if not math.isfinite(t0):
            raise NeurOSConfigError(f"t0 must be finite, got {t0}")
        rows = [recording.ch_names.index(ch) for ch in names]
        data = np.asarray(recording.data[rows], dtype=np.float64)
        if noise_uv > 0:
            data = data + np.random.default_rng(seed).normal(0.0, noise_uv, data.shape)
        if not np.isfinite(data).all():
            raise NeurOSDataError("the recording contains NaN/Inf samples")
        self.channels = names
        self.sfreq = float(recording.sfreq)
        self.data = data
        self.timestamps = t0 + np.arange(data.shape[1], dtype=np.float64) / self.sfreq
        self.chunk = max(1, round(chunk_s * self.sfreq))
        self.markers = tuple(
            Marker(float(t0 + onset / self.sfreq), label)
            for onset, label in zip(
                recording.events.onsets.tolist(), recording.events.labels, strict=True
            )
        )

    @property
    def duration_s(self) -> float:
        return float(self.data.shape[1]) / self.sfreq

    def chunks(self) -> Iterator[tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """``(samples [channels, n], timestamps [n])`` in order, ``chunk`` samples at a time."""
        for start in range(0, self.data.shape[1], self.chunk):
            stop = start + self.chunk
            yield self.data[:, start:stop], self.timestamps[start:stop]

    def run(self, engine: Engine, *, speed: float | None = None) -> list[Event]:
        """Feed every chunk to ``engine`` and collect its events.

        Args:
            speed: None replays as fast as possible; 1.0 paces chunks in real time (demo
                rehearsal), 2.0 at twice real time, and so on.

        Raises:
            NeurOSConfigError: on a non-positive speed.
        """
        if speed is not None and not (math.isfinite(speed) and speed > 0):
            raise NeurOSConfigError(f"speed must be None or > 0, got {speed}")
        events: list[Event] = []
        started = time.perf_counter()
        for samples, stamps in self.chunks():
            if speed is not None:  # wait until this chunk's last sample "exists"
                due = (stamps[-1] - self.timestamps[0]) / speed
                delay = due - (time.perf_counter() - started)
                if delay > 0:
                    time.sleep(delay)
            events.extend(engine.push(samples, stamps))
        events.extend(engine.close())
        return events
