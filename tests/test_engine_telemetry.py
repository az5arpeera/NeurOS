"""Telemetry for the stage view: the event, the observer, the runner and the hub's hello."""

from __future__ import annotations

import asyncio
import json
import math

import numpy as np
import pytest

from neuros.engine.events import IntentEvent, TelemetryEvent
from neuros.engine.pipeline import Engine
from neuros.engine.runner import replay
from neuros.engine.server import STATIC_FILES, EventHub
from neuros.engine.simulator import Simulator
from neuros.engine.telemetry import Telemetry, stage_info
from neuros.errors import NeurOSConfigError, NeurOSDataError
from tests.test_engine import _config, _recording, trained  # noqa: F401 (pytest fixture)


def _event(**change: object) -> TelemetryEvent:
    fields: dict[str, object] = {
        "seq": 4,
        "t": 10.0,
        "channels": ("C3", "C4"),
        "sfreq": 64.0,
        "samples": ((1.0, 2.0), (3.0, 4.0)),
        "rms_uv": (5.0, 6.0),
        "bad_channels": (),
        "in_transient": False,
        "state": "adapting",
        "seconds_of_data": 3.0,
    }
    fields.update(change)
    return TelemetryEvent(**fields)  # type: ignore[arg-type]


def test_telemetry_event_serialises() -> None:
    wire = json.loads(_event().to_json())
    assert wire["type"] == "telemetry"
    assert wire["schema"] == "neuros.intent.v0"
    assert wire["samples"] == [[1.0, 2.0], [3.0, 4.0]]
    assert wire["channels"] == ["C3", "C4"]


@pytest.mark.parametrize(
    "change",
    [
        {"samples": ((1.0,), (2.0, 3.0))},  # ragged
        {"samples": ((1.0, 2.0),)},  # a channel missing
        {"rms_uv": (5.0, -1.0)},
        {"samples": ((math.nan, 2.0), (3.0, 4.0))},
        {"bad_channels": ("Pz",)},
        {"channels": ("C3", "C3")},
        {"state": "sleepy"},
        {"sfreq": 0.0},
    ],
)
def test_telemetry_event_refuses_bad_content(change: dict[str, object]) -> None:
    with pytest.raises(NeurOSDataError):
        _event(**change)


def test_telemetry_follows_every_decision(trained: dict[str, object]) -> None:  # noqa: F811
    recording = _recording(subject=5, seed=15, n_trials=6)
    engine = Engine(trained["decoder"], _config(recording, trained))  # type: ignore[arg-type]
    telemetry = Telemetry(engine)
    simulator = Simulator(recording, t0=0.0)
    events = asyncio.run(replay([simulator], engine, telemetry=telemetry))
    shown = [e for e in events if isinstance(e, TelemetryEvent)]
    decisions = [e for e in events if isinstance(e, IntentEvent)]
    assert len(shown) == len(decisions) > 0
    seqs = [e.seq for e in events]
    assert seqs == sorted(seqs)  # one counter, in order
    assert len(set(seqs)) == len(seqs)
    first = shown[-1]
    assert first.channels == engine.profile
    assert first.sfreq == engine.config.contract.sfreq / 2
    hop = round(engine.hop_s * engine.config.contract.sfreq)
    assert all(len(row) == hop // 2 for row in first.samples)
    assert all(r > 0 for r in first.rms_uv)
    # A telemetry event comes just before its window's decision, on the same window clock.
    index = events.index(first)
    assert isinstance(events[index + 1], IntentEvent)
    assert events[index + 1].t == first.t


def test_replay_without_telemetry_is_unchanged(trained: dict[str, object]) -> None:  # noqa: F811
    recording = _recording(subject=5, seed=15, n_trials=4)
    engine = Engine(trained["decoder"], _config(recording, trained))  # type: ignore[arg-type]
    events = asyncio.run(replay([Simulator(recording, t0=0.0)], engine))
    assert not any(isinstance(e, TelemetryEvent) for e in events)


def test_telemetry_refuses_a_decimation_that_splits_the_hop(
    trained: dict[str, object],  # noqa: F811
) -> None:
    recording = _recording(subject=5, seed=15, n_trials=2)
    engine = Engine(trained["decoder"], _config(recording, trained))  # type: ignore[arg-type]
    with pytest.raises(NeurOSConfigError, match="divisor"):
        Telemetry(engine, decimate=3)


def test_hello_carries_stage_info(trained: dict[str, object]) -> None:  # noqa: F811
    recording = _recording(subject=5, seed=15, n_trials=2)
    engine = Engine(trained["decoder"], _config(recording, trained))  # type: ignore[arg-type]
    info = stage_info(engine, source="test", model="mvp-v0", train_people=339)
    hub = EventHub(["LEFT", "RIGHT"], info=info)
    hello = json.loads(hub.hello())
    assert hello["info"]["channels"] == list(engine.profile)
    assert hello["info"]["threshold"] == engine.config.policy.threshold
    assert hello["info"]["telemetry"] is True
    assert "info" not in json.loads(EventHub(["LEFT"]).hello())  # unchanged without telemetry
    with pytest.raises(NeurOSConfigError, match="JSON"):
        EventHub(["LEFT"], info={"bad": np.float64("nan")})


def test_stage_page_is_served() -> None:
    assert STATIC_FILES["/stage"][0] == "stage.html"
    assert STATIC_FILES["/stage.html"][0] == "stage.html"
