"""Live sources for the engine (§7.1 stage 1): a headset's samples as timestamped µV chunks.

- ``LslSource``: any Lab Streaming Layer EEG stream (most research headsets, OpenBCI GUI,
  BrainFlow's streamer, our simulator). Channel names and units come from the stream's
  description; timestamps are corrected to the local LSL clock and de-jittered.
- ``BrainFlowSource``: a board through BrainFlow directly (OpenBCI, Muse, g.tec Unicorn, ...;
  ``BoardIds.SYNTHETIC_BOARD`` for tests). BrainFlow delivers EEG in µV with UNIX timestamps.
- ``publish_lsl``: replays simulators as an LSL stream (plus a marker stream), so ``neuros run``
  can be rehearsed end to end exactly as with a headset.

**Never guess units (principle 4).** A stream whose channel units aren't declared is refused
unless the caller states them (``units=``): a volt stream read as µV would be a million times
too small and every window would look flat.
"""

from __future__ import annotations

import importlib
import math
import time
from collections.abc import Iterator, Sequence
from types import ModuleType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from neuros.engine.simulator import Simulator
from neuros.errors import NeurOSConfigError

# Unit names (lower-case) → factor to µV.
UNIT_FACTORS: dict[str, float] = {
    "microvolts": 1.0,
    "microvolt": 1.0,
    "uv": 1.0,
    "µv": 1.0,
    "μv": 1.0,  # Greek mu as well as the micro sign
    "millivolts": 1e3,
    "millivolt": 1e3,
    "mv": 1e3,
    "volts": 1e6,
    "volt": 1e6,
    "v": 1e6,
}
# How long a pull waits for data before yielding control (seconds).
PULL_TIMEOUT_S = 0.05


def unit_factor(unit: str) -> float:
    """The factor that converts ``unit`` to µV.

    Raises:
        NeurOSConfigError: on an unknown unit.
    """
    factor = UNIT_FACTORS.get(unit.strip().lower())
    if factor is None:
        raise NeurOSConfigError(f"unknown EEG unit {unit!r}; known: {sorted(set(UNIT_FACTORS))}")
    return factor


class LslSource:
    """Pull chunks from an LSL EEG stream (see the module docs)."""

    def __init__(
        self,
        name: str | None = None,
        *,
        stream_type: str = "EEG",
        units: str | None = None,
        resolve_timeout_s: float = 10.0,
    ) -> None:
        """
        Args:
            name: the stream's name (None: the first stream of ``stream_type``).
            units: the samples' unit when the stream doesn't declare one.

        Raises:
            NeurOSConfigError: if no stream is found, its rate is irregular, channel names
                are missing or repeat, or units are neither declared nor given.
        """
        import pylsl

        predicate = f"name='{name}'" if name else f"type='{stream_type}'"
        found = pylsl.resolve_bypred(predicate, 1, resolve_timeout_s)
        if not found:
            raise NeurOSConfigError(
                f"no LSL stream with {predicate} within {resolve_timeout_s:g} s"
            )
        info = found[0]
        if info.nominal_srate() <= 0:
            raise NeurOSConfigError(f"LSL stream {info.name()!r} has an irregular rate")
        self.inlet = pylsl.StreamInlet(
            info,
            max_buflen=10,  # §7.1: a 10-s ring buffer
            processing_flags=pylsl.proc_clocksync | pylsl.proc_dejitter,
        )
        full = self.inlet.info()
        names: list[str] = []
        declared: list[str] = []
        channel = full.desc().child("channels").child("channel")
        for _ in range(full.channel_count()):
            names.append(channel.child_value("label"))
            declared.append(channel.child_value("unit"))
            channel = channel.next_sibling()
        if any(not n for n in names) or len(set(names)) != len(names):
            raise NeurOSConfigError(
                f"LSL stream {full.name()!r} must label every channel uniquely, got {names}"
            )
        stated = {u for u in declared if u}
        if len(stated) > 1:
            raise NeurOSConfigError(f"LSL stream {full.name()!r} mixes units {sorted(stated)}")
        if stated and units is not None and unit_factor(next(iter(stated))) != unit_factor(units):
            raise NeurOSConfigError(
                f"the stream declares {next(iter(stated))!r} but units={units!r} was given"
            )
        unit = next(iter(stated)) if stated else units
        if unit is None:
            raise NeurOSConfigError(
                f"LSL stream {full.name()!r} doesn't declare its units: pass units= (e.g. 'uV')"
            )
        self.factor = unit_factor(unit)
        self.ch_names = tuple(names)
        self.sfreq = float(full.nominal_srate())
        self.name = full.name()

    def chunks(self) -> Iterator[tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """``(µV [channels, n], timestamps [n])`` forever. A quiet pull yields an empty chunk
        (n = 0), so the caller regains control at least every ``PULL_TIMEOUT_S``."""
        empty = (np.zeros((len(self.ch_names), 0)), np.zeros(0))
        while True:
            samples, stamps = self.inlet.pull_chunk(timeout=PULL_TIMEOUT_S)
            if not stamps:
                yield empty
                continue
            data = np.asarray(samples, dtype=np.float64).T * self.factor
            yield data, np.asarray(stamps, dtype=np.float64)

    def close(self) -> None:
        self.inlet.close_stream()


def brainflow_api() -> ModuleType:
    """BrainFlow's ``board_shim`` module, or a clear error saying how to get it.

    BrainFlow is in the optional engine extra, on Python 3.12+ only: BrainFlow 5.23 loads its
    native library through ``pkg_resources`` on Python 3.11, and setuptools removed that module
    in version 82 (Feb 2026), so the import fails there.

    Raises:
        NeurOSConfigError: if BrainFlow isn't installed or can't load.
    """
    try:
        return importlib.import_module("brainflow.board_shim")
    except ImportError as error:  # not installed, or 3.11's missing pkg_resources
        raise NeurOSConfigError(
            "BrainFlow sources need the engine extra on Python 3.12 or newer "
            f"(`uv sync --extra engine`); here: {type(error).__name__}: {error}"
        ) from error


class BrainFlowSource:
    """Pull chunks from a BrainFlow board (see the module docs)."""

    def __init__(
        self,
        board_id: int,
        *,
        serial_port: str = "",
        ch_names: Sequence[str] | None = None,
        poll_s: float = 0.05,
    ) -> None:
        """
        Args:
            board_id: a ``brainflow.board_shim.BoardIds`` value.
            ch_names: the EEG channels' 10-20 names in BrainFlow's EEG-channel order (default:
                the board's own, when BrainFlow knows them).

        Raises:
            NeurOSConfigError: if BrainFlow can't load (``brainflow_api``), or the board's
                channel names are unknown and not given, or don't match its channel count.
        """
        api = brainflow_api()
        BoardShim, BrainFlowInputParams = api.BoardShim, api.BrainFlowInputParams

        if not (math.isfinite(poll_s) and poll_s > 0):
            raise NeurOSConfigError(f"poll_s must be > 0, got {poll_s}")
        params = BrainFlowInputParams()
        params.serial_port = serial_port
        self.board = BoardShim(board_id, params)
        self.eeg_rows = list(BoardShim.get_eeg_channels(board_id))
        self.time_row = int(BoardShim.get_timestamp_channel(board_id))
        if ch_names is None:
            try:
                ch_names = list(BoardShim.get_eeg_names(board_id))
            except Exception as error:  # BrainFlow raises when a board has no names
                raise NeurOSConfigError(
                    f"BrainFlow doesn't name board {board_id}'s channels: pass ch_names"
                ) from error
        if len(ch_names) != len(self.eeg_rows) or len(set(ch_names)) != len(ch_names):
            raise NeurOSConfigError(
                f"{len(self.eeg_rows)} EEG channels but names {list(ch_names)} (must be unique)"
            )
        self.ch_names = tuple(ch_names)
        self.sfreq = float(BoardShim.get_sampling_rate(board_id))
        self.poll_s = poll_s
        self._started = False

    def start(self) -> None:
        self.board.prepare_session()
        self.board.start_stream()
        self._started = True

    def chunks(self) -> Iterator[tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """``(µV [channels, n], UNIX timestamps [n])`` forever (starts the stream if needed)."""
        if not self._started:
            self.start()
        empty = (np.zeros((len(self.ch_names), 0)), np.zeros(0))
        while True:
            time.sleep(self.poll_s)
            data = self.board.get_board_data()
            if data.shape[1] == 0:
                yield empty  # the caller regains control even when the board is quiet
                continue
            yield (
                np.asarray(data[self.eeg_rows], dtype=np.float64),
                np.asarray(data[self.time_row], dtype=np.float64),
            )

    def close(self) -> None:
        if self._started:
            self.board.stop_stream()
            self.board.release_session()
            self._started = False


def publish_lsl(
    simulators: Sequence[Simulator],
    *,
    name: str = "NeurOS-Simulator",
    speed: float = 1.0,
    markers: bool = True,
) -> None:
    """Replay ``simulators`` as an LSL EEG stream (µV, channel labels declared) in real time
    (``speed`` × real time), with a string marker stream of the ground truth if ``markers``.

    Raises:
        NeurOSConfigError: on no simulators, a non-positive speed, or runs that differ in
            channels or rate (one stream has one layout).
    """
    import pylsl

    if not simulators:
        raise NeurOSConfigError("nothing to publish")
    if not (math.isfinite(speed) and speed > 0):
        raise NeurOSConfigError(f"speed must be > 0, got {speed}")
    first = simulators[0]
    if any(s.channels != first.channels or s.sfreq != first.sfreq for s in simulators):
        raise NeurOSConfigError("every run must have the same channels and rate")
    info = pylsl.StreamInfo(name, "EEG", len(first.channels), first.sfreq, "float32", name)
    channels = info.desc().append_child("channels")
    for label in first.channels:
        entry = channels.append_child("channel")
        entry.append_child_value("label", label)
        entry.append_child_value("unit", "microvolts")
        entry.append_child_value("type", "EEG")
    outlet = pylsl.StreamOutlet(info, chunk_size=first.chunk)
    marker_outlet = (
        pylsl.StreamOutlet(pylsl.StreamInfo(f"{name}-Markers", "Markers", 1, 0, "string", name))
        if markers
        else None
    )
    start_wall = pylsl.local_clock()
    first_t = float(first.timestamps[0])
    pending = sorted((m for s in simulators for m in s.markers), key=lambda m: m.t)
    for simulator in simulators:
        for samples, stamps in simulator.chunks():
            due = start_wall + (float(stamps[-1]) - first_t) / speed
            delay = due - pylsl.local_clock()
            if delay > 0:
                time.sleep(delay)
            now = pylsl.local_clock()
            # Sample times on the local LSL clock, spaced at the (sped-up) rate.
            local = now - (float(stamps[-1]) - stamps) / speed
            outlet.push_chunk(np.ascontiguousarray(samples.T, dtype=np.float32), local.tolist())
            while marker_outlet is not None and pending and pending[0].t <= float(stamps[-1]):
                marker = pending.pop(0)
                marker_outlet.push_sample(
                    [marker.label], now - (float(stamps[-1]) - marker.t) / speed
                )


def describe_source(source: Any) -> str:
    """One line about a source, for the console."""
    return f"{type(source).__name__}: {len(source.ch_names)} channels at {source.sfreq:g} Hz"
