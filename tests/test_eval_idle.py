"""E7, quiet when idle (neuros.eval.idle): replay spans, scoring, pooling and the bar."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pytest
from numpy.typing import NDArray

from neuros.config.schemas import SignalContract
from neuros.data.recording import Recording
from neuros.data.synthetic import SyntheticConfig, make_recording
from neuros.engine.events import HOLD, IntentEvent, StatusEvent
from neuros.engine.pipeline import Engine, EngineConfig
from neuros.engine.policy import PolicySettings
from neuros.errors import NeurOSConfigError, NeurOSDataError
from neuros.eval import idle

LAYOUT = ("FC3", "FC4", "C3", "Cz", "C4", "CP3", "CP4")
CONTRACT = SignalContract(band_hz=(8.0, 30.0))


def _probs(command: str) -> dict[str, float]:
    """Valid smoothed probabilities for a decision (they sum to 1)."""
    if command == HOLD:
        return {"LEFT": 0.5, "RIGHT": 0.5}
    others = [c for c in ("LEFT", "RIGHT") if c != command]
    return {command: 0.9} | {c: 0.1 / len(others) for c in others}


def _decision(t: float, command: str = HOLD, seq: int = 0) -> IntentEvent:
    held = command == HOLD
    return IntentEvent(
        seq=seq,
        t=t,
        command=command,
        confidence=0.55 if held else 0.9,
        probs=_probs(command),
        hold_reason="low_confidence" if held else None,
        state="adapting",
        seconds_of_data=10.0,
        quality_ok=True,
    )


def test_replay_spans() -> None:
    assert idle.replay_spans([180.0, 62.0]) == [(0.0, 180.0), (185.0, 247.0)]
    assert idle.replay_spans([10.0], gap_s=0.0) == [(0.0, 10.0)]
    with pytest.raises(NeurOSConfigError):
        idle.replay_spans([])
    with pytest.raises(NeurOSConfigError):
        idle.replay_spans([10.0, 0.0])
    with pytest.raises(NeurOSConfigError):
        idle.replay_spans([10.0], gap_s=-1.0)


def test_policy_for_enables_every_class() -> None:
    policy = idle.policy_for(("left_hand", "right_hand", "feet"), 0.7)
    assert dict(policy.commands) == {"left_hand": "LEFT", "right_hand": "RIGHT", "feet": "FEET"}
    assert policy.threshold == 0.7
    for bad in (0.4, 1.1, math.nan):
        with pytest.raises(NeurOSConfigError):
            idle.policy_for(("left_hand",), bad)


def test_count_segments_scores_by_window_and_settling() -> None:
    spans = [(0.0, 60.0), (65.0, 100.0)]
    events = [
        StatusEvent(seq=0, t=0.0, previous="disconnected", state="settling", reason="first"),
        _decision(20.0, "LEFT"),  # inside settling: not scored, but sets "previous"
        _decision(33.0, "LEFT"),  # scored at rest; previous was LEFT, so not an onset
        _decision(34.0, HOLD),
        _decision(35.0, "RIGHT"),  # onset
        _decision(36.0, "LEFT"),  # onset (the other command)
        _decision(66.0, "RIGHT"),  # window [64, 66] straddles the pause: neither segment
        _decision(70.0, HOLD),
        _decision(71.0, "RIGHT"),  # arithmetic, onset
    ]
    counts = idle.count_segments(
        events, spans, window_s=2.0, settle_s=30.0, commands=["LEFT", "RIGHT"]
    )
    rest, task = counts["rest"], counts["arithmetic"]
    assert (rest.decisions, rest.commits, rest.onsets) == (4, 3, 2)
    assert rest.by_command == {"LEFT": 2, "RIGHT": 1}
    assert rest.holds == {"low_confidence": 1}
    assert (task.decisions, task.commits, task.onsets) == (2, 1, 1)
    assert rest.commit_rate == pytest.approx(0.75)
    assert rest.onsets_per_minute(0.125) == pytest.approx(2 / (4 * 0.125 / 60))
    assert math.isnan(idle.SegmentCount().commit_rate)
    with pytest.raises(NeurOSDataError, match="emitted"):
        idle.count_segments(
            [_decision(40.0, "FEET")], spans, window_s=2.0, settle_s=30.0, commands=["LEFT"]
        )
    with pytest.raises(NeurOSConfigError):
        idle.count_segments(events, spans[:1], window_s=2.0, settle_s=30.0, commands=["LEFT"])


def test_pooled_and_verdict() -> None:
    a = idle.SegmentCount(decisions=80, commits=4, onsets=2, by_command={"LEFT": 4})
    b = idle.SegmentCount(decisions=20, commits=6, onsets=3, by_command={"RIGHT": 6})
    summary = idle.pooled([a, b], hop_s=0.125)
    assert summary["commit_rate"] == pytest.approx(0.10)
    assert summary["by_command"] == {"LEFT": 4, "RIGHT": 6}
    assert summary["people_with_any_commit"] == 2
    with pytest.raises(NeurOSDataError):
        idle.pooled([idle.SegmentCount()], hop_s=0.125)
    assert idle.verdict(0.10, 0.05)["outcome"] == "PASS"  # the bar is "at most 10%"
    assert idle.verdict(0.11, 0.05)["outcome"] == "FAIL"
    assert idle.verdict(0.05, 0.2)["checks"] == {"rest": True, "arithmetic": False}
    with pytest.raises(NeurOSDataError):
        idle.verdict(math.nan, 0.1)


class FixedDecoder:
    """Always the same probabilities: a model that leans one way whatever the EEG."""

    def __init__(self, p_left: float) -> None:
        self.layout = LAYOUT
        self.classes = ("left_hand", "right_hand")
        self.p_left = p_left

    def predict_proba(self, x: NDArray[np.float32]) -> NDArray[np.float64]:
        return np.tile([self.p_left, 1.0 - self.p_left], (x.shape[0], 1))


def _recording(run: str, seed: int) -> Recording:
    config = SyntheticConfig(n_trials=12, seed=seed)
    return make_recording(config, dataset="SynIdle", subject=1, session="0", run=run)


@pytest.mark.parametrize(("p_left", "expect_commits"), [(0.95, True), (0.5, False)])
def test_replay_through_the_engine(p_left: float, expect_commits: bool) -> None:
    recordings = [_recording("rest", 1), _recording("task", 2)]

    def new_engine(channels: Sequence[str], sfreq: float, policy: PolicySettings) -> Engine:
        profile = [ch for ch in LAYOUT if ch in channels]
        config = EngineConfig(
            stream_channels=tuple(channels),
            native_sfreq=sfreq,
            prior=np.eye(len(profile)) * 25.0,
            input_scale=1.0,
            contract=CONTRACT,
            policy=policy,
        )
        return Engine(FixedDecoder(p_left), config)

    policy = idle.policy_for(("left_hand", "right_hand"), 0.6)
    events, simulators, engine = idle.replay_recordings(
        new_engine, recordings, policy, channels=idle.IDLE_CHANNELS
    )
    assert engine.profile == idle.IDLE_CHANNELS
    spans = idle.replay_spans([s.duration_s for s in simulators])
    counts = idle.count_segments(
        events,
        spans,
        window_s=CONTRACT.window_s,
        settle_s=engine.config.settling_s,
        commands=list(policy.commands.values()),
    )
    rest, task = counts["rest"], counts["arithmetic"]
    assert rest.decisions > 0
    assert task.decisions > 0
    if expect_commits:  # a model that always leans left: false LEFTs, never RIGHT
        assert rest.commit_rate > 0.5
        assert task.commit_rate > 0.5
        assert set(rest.by_command) == {"LEFT"}
    else:  # a model that never leans: the engine holds throughout
        assert rest.commits == task.commits == 0
    # Nothing in the settling period is scored.
    intents = [e for e in events if isinstance(e, IntentEvent)]
    early = sum(1 for e in intents if e.t - CONTRACT.window_s < engine.config.settling_s)
    assert early > 0
    assert rest.decisions + task.decisions <= len(intents) - early
