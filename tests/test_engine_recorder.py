"""Recorded sessions (``neuros run --record``): round trip, crash safety, and markers from an app
saved on the engine's clock during a live loop."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from neuros.data.synthetic import SyntheticConfig, make_recording
from neuros.engine.bundle import ModelBundle
from neuros.engine.events import IntentEvent, StatusEvent
from neuros.engine.pipeline import Engine
from neuros.engine.recorder import SessionRecorder, door_scores, load_session
from neuros.engine.runner import run_live
from neuros.engine.server import EventHub
from neuros.engine.simulator import Simulator
from neuros.errors import NeurOSConfigError, NeurOSDataError

from .test_bundle import bundle  # noqa: F401

CHANNELS = ("C3", "Cz", "C4")


def _status(seq: int, t: float) -> StatusEvent:
    return StatusEvent(seq=seq, t=t, state="settling", previous="disconnected", reason="test")


def test_round_trip_and_never_overwrite(tmp_path: Path) -> None:
    folder = tmp_path / "s1"
    rng = np.random.default_rng(0)
    chunks = [rng.normal(0, 10, (3, n)).astype(np.float64) for n in (16, 0, 9)]
    stamps = [100 + np.arange(16) / 250, np.zeros(0), 100.064 + np.arange(9) / 250]
    with SessionRecorder(
        folder, channels=CHANNELS, sfreq=250.0, source="test", details={"who": "x"}
    ) as rec:
        for x, t in zip(chunks, stamps, strict=True):
            rec.write_chunk(x, t)
        rec.write_event(_status(0, 100.0))
        rec.write_marker(100.05, "LEFT")
        with pytest.raises(NeurOSDataError):
            rec.write_chunk(np.zeros((2, 4)), np.zeros(4))  # wrong channel count
    session = load_session(folder)
    assert session.channels == CHANNELS
    assert session.sfreq == 250.0
    np.testing.assert_allclose(session.data, np.concatenate(chunks, axis=1).astype(np.float32))
    np.testing.assert_array_equal(session.timestamps, np.concatenate(stamps))
    assert session.events[0]["type"] == "status"
    assert session.markers() == [(100.05, "LEFT")]
    assert session.meta["n_samples"] == 25
    assert session.meta["who"] == "x"
    assert session.meta["ended_utc"] is not None
    with pytest.raises(NeurOSConfigError, match="never overwritten"):
        SessionRecorder(folder, channels=CHANNELS, sfreq=250.0, source="test")


def test_a_crashed_session_still_loads(tmp_path: Path) -> None:
    folder = tmp_path / "crash"
    rec = SessionRecorder(folder, channels=CHANNELS, sfreq=250.0, source="test")
    rec.write_chunk(np.ones((3, 10)), np.arange(10.0))
    rec.write_marker(1.0, "RIGHT")
    rec.flush()  # never closed: the process "died"
    with (folder / "samples.f32").open("ab") as handle:
        handle.write(b"\x00\x00\x80")  # a partial row
    with (folder / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"type": "mar')  # a cut line
    session = load_session(folder)
    assert session.data.shape == (3, 10)
    assert session.markers() == [(1.0, "RIGHT")]
    assert session.meta["n_samples"] is None


class _ReplaySource:
    """A live-source stand-in: a simulator's chunks, then an endless quiet stream."""

    def __init__(self, simulator: Simulator) -> None:
        self.simulator = simulator
        self.sfreq = simulator.sfreq
        self.closed = False

    def chunks(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        yield from self.simulator.chunks()
        while True:
            yield np.zeros((len(self.simulator.channels), 0)), np.zeros(0)

    def close(self) -> None:
        self.closed = True


def test_live_loop_records_chunks_events_and_app_markers(
    bundle: ModelBundle,  # noqa: F811
    tmp_path: Path,
) -> None:
    raw = make_recording(SyntheticConfig(n_trials=4, seed=21), dataset="SynC", subject=1)
    simulator = Simulator(raw)
    engine = Engine(bundle.decoder(), bundle.engine_config(raw.ch_names, raw.sfreq))
    hub = EventHub(["LEFT", "RIGHT"], requests=["mark"])
    assert hub.receive(json.dumps({"type": "request", "request": "mark", "label": "LEFT"}))
    assert not hub.receive(json.dumps({"type": "request", "request": "mark", "label": "FEET"}))
    assert not hub.receive(json.dumps({"type": "request", "request": "mark"}))
    source = _ReplaySource(simulator)
    folder = tmp_path / "live"
    recorder = SessionRecorder(folder, channels=raw.ch_names, sfreq=raw.sfreq, source="replay")
    events = asyncio.run(run_live(source, engine, hub=hub, seconds=3.0, recorder=recorder))
    assert source.closed
    assert recorder.closed
    session = load_session(folder)
    n = session.data.shape[1]
    assert n == simulator.data.shape[1]  # every sample, as the engine received it
    np.testing.assert_allclose(session.data, simulator.data[:, :n].astype(np.float32))
    np.testing.assert_array_equal(session.timestamps, simulator.timestamps[:n])
    intents = [e for e in session.events if e.get("type") == "intent"]
    assert len(intents) == sum(isinstance(e, IntentEvent) for e in events)
    markers = session.markers()
    assert len(markers) == 1
    assert markers[0][1] == "LEFT"
    assert simulator.timestamps[0] <= markers[0][0] <= simulator.timestamps[-1]  # engine clock


def _intent(seq: int, t: float, command: str, level: str | None) -> IntentEvent:
    hold = command == "HOLD"
    return IntentEvent(
        seq=seq,
        t=t,
        command=command,
        confidence=0.7,
        probs={"LEFT": 0.7, "RIGHT": 0.3},
        hold_reason="low_confidence" if hold else None,
        state="steady",
        seconds_of_data=60.0,
        quality_ok=True,
        calibration_level=level,  # type: ignore[arg-type]
    )


def test_door_scores_recompute_the_game_per_level(tmp_path: Path) -> None:
    folder = tmp_path / "doors"
    with SessionRecorder(folder, channels=CHANNELS, sfreq=250.0, source="test") as rec:
        seq = 0

        def say(t: float, command: str, level: str | None) -> None:
            nonlocal seq
            rec.write_event(_intent(seq, t, command, level))
            seq += 1

        # Zero calibration: door at 10 s answered right at 14 s (inside 3.5-7 s); door at 20 s
        # answered only at 22 s (too early: the cue was still in the window) -> held; door at
        # 30 s answered wrong at 34 s.
        for t, label in ((10.0, "LEFT"), (20.0, "RIGHT"), (30.0, "LEFT")):
            say(t, "HOLD", None)
            rec.write_marker(t, label)
        say(14.0, "LEFT", None)
        say(22.0, "RIGHT", None)
        say(34.0, "RIGHT", None)
        # Quick calibration: door at 50 s answered right at 54 s, then a later command ignored.
        say(50.0, "HOLD", "quick")
        rec.write_marker(50.0, "RIGHT")
        say(54.0, "RIGHT", "quick")
        say(55.0, "LEFT", "quick")
    scores = door_scores(load_session(folder))
    zero, quick = scores["zero"], scores["quick"]
    assert (zero.right, zero.wrong, zero.held) == (1, 1, 1)
    assert zero.accuracy == pytest.approx(0.5)
    assert zero.answered_fraction == pytest.approx(2 / 3)
    assert (quick.right, quick.wrong, quick.held) == (1, 0, 0)
    assert 0.0 < quick.p_value() <= 0.5
    with pytest.raises(NeurOSConfigError):
        door_scores(load_session(folder), score_from_s=5.0, score_until_s=4.0)
