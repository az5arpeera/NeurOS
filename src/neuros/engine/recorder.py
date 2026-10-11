"""Saving a live session to disk: the founders' headset tests leave evidence (D-040).

``neuros run --record DIR`` writes, as the session runs:

- ``samples.f32``: the headset's samples exactly as the engine received them (µV, after the
  source's unit conversion), float32 little-endian, one row of ``n_channels`` per sample;
- ``timestamps.f64``: each sample's stream-clock time, float64 little-endian;
- ``events.jsonl``: every event the engine emitted (intents, status, calibration steps), one
  ``neuros.intent.v0`` JSON object per line, plus the app's markers
  (``{"type": "marker", "t": ..., "label": "LEFT", "source": "app"}``: e.g. the door game's
  prompts, so its score can be recomputed offline);
- ``meta.json``: channels, rate, source, bundle provenance, start and end (written at start and
  rewritten with the sample count at the end).

Appending raw bytes keeps writing cheap and a crash loses at most the last chunk; the loader
trims a partial row. **Local files only** (architecture §10): sessions hold a person's brain
data; ``sessions/`` is git-ignored and nothing uploads them.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any

import numpy as np
from numpy.typing import NDArray

from neuros.engine.events import Event
from neuros.errors import NeurOSConfigError, NeurOSDataError

SESSION_VERSION = 1
SAMPLES_FILE = "samples.f32"
TIMES_FILE = "timestamps.f64"
EVENTS_FILE = "events.jsonl"
META_FILE = "meta.json"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class SessionRecorder:
    """Writes one live session to a new folder (see the module docs). Use as a context manager
    or call ``close``; not thread-safe (one per engine loop)."""

    def __init__(
        self,
        folder: Path,
        *,
        channels: Sequence[str],
        sfreq: float,
        source: str,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        """
        Raises:
            NeurOSConfigError: if ``folder`` already exists (a session is never overwritten),
                or on no channels or a non-positive rate.
        """
        if folder.exists():
            raise NeurOSConfigError(f"{folder} already exists: sessions are never overwritten")
        if not channels or len(set(channels)) != len(channels):
            raise NeurOSConfigError(f"channels must be distinct and non-empty: {list(channels)}")
        if not (math.isfinite(sfreq) and sfreq > 0):
            raise NeurOSConfigError(f"sfreq must be positive, got {sfreq}")
        folder.mkdir(parents=True)
        self.folder = folder
        self.channels = tuple(channels)
        self.n_samples = 0
        self.n_events = 0
        self._meta: dict[str, Any] = {
            "session_version": SESSION_VERSION,
            "channels": list(self.channels),
            "sfreq": float(sfreq),
            "units": "uV",
            "source": source,
            "started_utc": _now(),
            "ended_utc": None,
            "n_samples": None,
        } | dict(details or {})
        self._write_meta()
        self._samples: IO[bytes] = (folder / SAMPLES_FILE).open("ab")
        self._times: IO[bytes] = (folder / TIMES_FILE).open("ab")
        self._events: IO[str] = (folder / EVENTS_FILE).open("a", encoding="utf-8", newline="\n")
        self.closed = False

    def write_chunk(self, chunk: NDArray[np.floating], timestamps: NDArray[np.floating]) -> None:
        """Append ``[channels, n]`` µV samples and their ``[n]`` timestamps.

        Raises:
            NeurOSDataError: on a shape that doesn't match the channels or the timestamps.
        """
        x = np.asarray(chunk)
        stamps = np.asarray(timestamps, dtype=np.float64)
        if x.ndim != 2 or x.shape[0] != len(self.channels) or stamps.shape != (x.shape[1],):
            raise NeurOSDataError(
                f"chunk {x.shape} / timestamps {stamps.shape} don't match {len(self.channels)} "
                "channels"
            )
        if x.shape[1] == 0:
            return
        self._samples.write(np.ascontiguousarray(x.T, dtype="<f4").tobytes())
        self._times.write(np.ascontiguousarray(stamps, dtype="<f8").tobytes())
        self.n_samples += x.shape[1]

    def write_event(self, event: Event) -> None:
        self._events.write(event.to_json() + "\n")
        self.n_events += 1

    def write_marker(self, t: float, label: str, *, source: str = "app") -> None:
        """An app's marker (e.g. the door the page asked for) at stream time ``t``."""
        if not math.isfinite(t):
            raise NeurOSDataError(f"marker time must be finite, got {t}")
        line = {"type": "marker", "t": t, "label": str(label), "source": source}
        self._events.write(json.dumps(line, allow_nan=False, separators=(",", ":")) + "\n")
        self.n_events += 1

    def flush(self) -> None:
        for handle in (self._samples, self._times, self._events):
            handle.flush()

    def close(self) -> None:
        """Flush, close and finish ``meta.json`` (idempotent)."""
        if self.closed:
            return
        for handle in (self._samples, self._times, self._events):
            handle.close()
        self._meta["ended_utc"] = _now()
        self._meta["n_samples"] = self.n_samples
        self._write_meta()
        self.closed = True

    def __enter__(self) -> SessionRecorder:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _write_meta(self) -> None:
        text = json.dumps(self._meta, indent=1, allow_nan=False)
        (self.folder / META_FILE).write_text(text, encoding="utf-8", newline="\n")


@dataclass(frozen=True)
class RecordedSession:
    """A saved session, loaded."""

    data: NDArray[np.float32]  # [channels, n] µV
    timestamps: NDArray[np.float64]  # [n]
    channels: tuple[str, ...]
    sfreq: float
    events: tuple[dict[str, Any], ...]
    meta: dict[str, Any]

    def markers(self) -> list[tuple[float, str]]:
        """The app's markers, ``(t, label)`` in time order."""
        found = [(float(e["t"]), str(e["label"])) for e in self.events if e.get("type") == "marker"]
        return sorted(found)


def load_session(folder: Path) -> RecordedSession:
    """Read a session folder (a partial last row, from a crash, is dropped).

    Raises:
        NeurOSDataError: on missing files, another version, or samples and timestamps that
            disagree.
    """
    meta_path = folder / META_FILE
    if not meta_path.is_file():
        raise NeurOSDataError(f"{folder}: no {META_FILE}: not a NeurOS session")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("session_version") != SESSION_VERSION:
        raise NeurOSDataError(f"{folder}: session version {meta.get('session_version')}")
    channels = tuple(meta["channels"])
    raw = np.fromfile(folder / SAMPLES_FILE, dtype="<f4")
    stamps = np.fromfile(folder / TIMES_FILE, dtype="<f8")
    n = min(raw.size // len(channels), stamps.size)
    if meta.get("n_samples") is not None and int(meta["n_samples"]) != n:
        raise NeurOSDataError(f"{folder}: meta says {meta['n_samples']} samples, files hold {n}")
    data = raw[: n * len(channels)].reshape(n, len(channels)).T.copy()
    events = []
    with (folder / EVENTS_FILE).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                try:
                    events.append(json.loads(line))
                except ValueError:  # a line cut by a crash: everything before it stands
                    break
    return RecordedSession(
        data=data,
        timestamps=stamps[:n].copy(),
        channels=channels,
        sfreq=float(meta["sfreq"]),
        events=tuple(events),
        meta=meta,
    )


# ---------------------------------------------------------------------- scoring a session
# The demo page's door game (clients/js/demo.html): a door is cued for 1 s and the first
# committed command whose window began after the cue (plus the policy's smoothing) counts.
DOOR_SCORE_FROM_S = 3.5
DOOR_SCORE_UNTIL_S = 7.0


@dataclass(frozen=True)
class DoorScore:
    """The door game's outcome for one calibration level ("zero" or "quick")."""

    level: str
    right: int
    wrong: int
    held: int

    @property
    def doors(self) -> int:
        return self.right + self.wrong + self.held

    @property
    def accuracy(self) -> float:
        """Right among the doors answered (committed), NaN with none."""
        answered = self.right + self.wrong
        return self.right / answered if answered else math.nan

    @property
    def answered_fraction(self) -> float:
        return (self.right + self.wrong) / self.doors if self.doors else math.nan

    def p_value(self) -> float:
        """One-sided binomial test of the answered doors against chance (50%)."""
        from scipy.stats import binomtest

        answered = self.right + self.wrong
        if answered == 0:
            return math.nan
        return float(binomtest(self.right, answered, 0.5, alternative="greater").pvalue)


def door_scores(
    session: RecordedSession,
    *,
    score_from_s: float = DOOR_SCORE_FROM_S,
    score_until_s: float = DOOR_SCORE_UNTIL_S,
) -> dict[str, DoorScore]:
    """Recompute the door game from a recorded session: per calibration level in force at
    each door (``zero`` or ``quick``), right / wrong / held.

    Raises:
        NeurOSConfigError: on an empty or inverted scoring span.
    """
    if not 0 <= score_from_s < score_until_s:
        raise NeurOSConfigError(f"bad scoring span [{score_from_s}, {score_until_s}] s")
    intents = sorted(
        (e for e in session.events if e.get("type") == "intent"), key=lambda e: float(e["t"])
    )
    times = np.array([float(e["t"]) for e in intents])
    counts: dict[str, list[int]] = {}
    for t, label in session.markers():
        lo, hi = np.searchsorted(times, [t + score_from_s, t + score_until_s], side="left")
        span = intents[lo:hi]
        at_cue = int(np.searchsorted(times, t, side="right")) - 1
        level = (intents[at_cue]["calibration"].get("level") if at_cue >= 0 else None) or "zero"
        row = counts.setdefault(level, [0, 0, 0])
        command = next((e["command"] for e in span if e["command"] != "HOLD"), None)
        if command is None:
            row[2] += 1
        elif command == label:
            row[0] += 1
        else:
            row[1] += 1
    return {level: DoorScore(level, *row) for level, row in counts.items()}
