"""The online engine (architecture §7): components, train-serve parity, and end to end.

The end-to-end test is the §9 one: simulator -> engine -> events, on synthetic people whose
mu rhythm desynchronises over the hemisphere opposite the imagined hand. A small decoder is
trained offline on other synthetic people, exactly as training places windows, then a new
person streams through the engine with no calibration.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from numpy.typing import NDArray

from neuros.align.covariance import window_covariances
from neuros.align.ea import euclidean_reference, reference_from_mean
from neuros.config.schemas import SignalContract
from neuros.data.recording import Recording
from neuros.data.synthetic import SyntheticConfig, make_recording
from neuros.engine.alignment import OnlineReference
from neuros.engine.events import HOLD, OTHER, IntentEvent, StatusEvent
from neuros.engine.pipeline import Engine, EngineConfig, EngineWindow, place_window
from neuros.engine.policy import Policy, PolicySettings
from neuros.engine.quality import QualityGate, QualityThresholds
from neuros.engine.simulator import Simulator
from neuros.errors import NeurOSConfigError, NeurOSDataError
from neuros.eval.pooled import to_layout
from neuros.signal.preprocess import preprocess_recording
from neuros.signal.reference import common_average_reference
from neuros.signal.windows import extract_epochs, sliding_windows

# The model's layout: a motor strip with one channel the synthetic headset lacks (CP5, zero
# filled) and without one it has (Pz, ignored).
LAYOUT = ("FC3", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP3", "CP4", "CP5")
CLASSES = ("left_hand", "right_hand")
CONTRACT = SignalContract(band_hz=(8.0, 30.0))


def _recording(subject: int, seed: int, n_trials: int = 40) -> Recording:
    return make_recording(SyntheticConfig(n_trials=n_trials, seed=seed), subject=subject)


def _profile(recording: Recording) -> tuple[str, ...]:
    return tuple(ch for ch in LAYOUT if ch in recording.ch_names)


class LogVarDecoder:
    """Log-variance of each present layout row + logistic regression (a test decoder)."""

    def __init__(self, rows: list[int]) -> None:
        from sklearn.linear_model import LogisticRegression

        self.layout = LAYOUT
        self.classes = CLASSES
        self.rows = rows
        self.model = LogisticRegression(max_iter=1000)

    def _features(self, x: NDArray[np.float32]) -> NDArray[np.float64]:
        return np.log(np.var(x[:, self.rows, :].astype(np.float64), axis=-1))

    def fit(self, x: NDArray[np.float32], labels: list[str]) -> LogVarDecoder:
        self.model.fit(self._features(x), [CLASSES.index(label) for label in labels])
        return self

    def predict_proba(self, x: NDArray[np.float32]) -> NDArray[np.float64]:
        return np.asarray(self.model.predict_proba(self._features(x)), dtype=np.float64)


@pytest.fixture(scope="module")
def trained() -> dict[str, object]:
    """A decoder trained on two synthetic people, placed exactly as E1 places windows."""
    xs, labels, priors = [], [], []
    for subject, seed in ((1, 11), (2, 12)):
        prepared = preprocess_recording(_recording(subject, seed), CONTRACT)
        windows, _ = extract_epochs(prepared, epoch_s=(0.5, 2.5), contract=CONTRACT)
        x, _, _ = to_layout(windows, LAYOUT, reference="car", alignment="ea", unit="session")
        xs.append(x)
        labels.extend(windows.labels)
        profile_windows = windows.pick_channels(list(_profile(prepared)))
        priors.append(window_covariances(common_average_reference(profile_windows.data)).mean(0))
    x = np.concatenate(xs)
    scale = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    rows = [LAYOUT.index(ch) for ch in LAYOUT if ch != "CP5"]
    decoder = LogVarDecoder(rows).fit((x / scale).astype(np.float32), labels)
    return {"decoder": decoder, "scale": scale, "prior": np.mean(priors, axis=0)}


def _config(recording: Recording, trained: dict[str, object], **extra: object) -> EngineConfig:
    settings: dict[str, object] = {
        "stream_channels": tuple(recording.ch_names),
        "native_sfreq": recording.sfreq,
        "prior": trained["prior"],
        "input_scale": trained["scale"],
        "contract": CONTRACT,
    }
    return EngineConfig(**(settings | extra))  # type: ignore[arg-type]


# ---------------------------------------------------------------------- events
def test_events_validate_and_serialise() -> None:
    event = IntentEvent(
        seq=3,
        t=12.5,
        command="LEFT",
        confidence=0.8,
        probs={"LEFT": 0.8, "RIGHT": 0.2, OTHER: 0.0},
        hold_reason=None,
        state="adapting",
        seconds_of_data=4.0,
        quality_ok=True,
        latency_ms=1.5,
    )
    wire = json.loads(event.to_json())
    assert wire["schema"] == "neuros.intent.v0"
    assert wire["type"] == "intent"
    assert wire["calibration"] == {"state": "adapting", "seconds_of_data": 4.0, "level": None}
    status = json.loads(
        StatusEvent(seq=0, t=0.0, state="settling", previous="disconnected").to_json()
    )
    assert status["type"] == "status"
    bad = {
        "hold without reason": {"command": HOLD, "hold_reason": None},
        "reason without hold": {"command": "LEFT", "hold_reason": "warmup"},
        "unknown reason": {"command": HOLD, "hold_reason": "bored"},
        "confidence above 1": {"confidence": 1.5},
        "probs above 1 in sum": {"probs": {"LEFT": 0.8, "RIGHT": 0.8}},
        "command not in probs": {"command": "UP"},
        "unknown state": {"state": "sleepy"},
        "nan time": {"t": math.nan},
        "negative seq": {"seq": -1},
    }
    for name, change in bad.items():
        fields = {**event.__dict__, **change}
        with pytest.raises(NeurOSDataError):  # each case must be rejected: {name}
            IntentEvent(**fields)
        assert name  # (the case's name shows in the traceback when it isn't rejected)


# ---------------------------------------------------------------------- quality gate
def test_quality_gate_reasons_and_blame() -> None:
    names = ("C3", "Cz", "C4")
    gate = QualityGate(names)
    rng = np.random.default_rng(0)
    clean = rng.normal(0, 10, (3, 256))
    assert gate.check(clean).ok
    flat = clean.copy()
    flat[1] = 0.0
    verdict = gate.check(flat)
    assert (verdict.ok, verdict.reason, verdict.bad_channels) == (False, "flat", ("Cz",))
    saturated = clean.copy()
    saturated[2, 100] = 5000.0
    assert gate.check(saturated).reason == "saturated"
    moving = clean.copy()
    moving[0, :128] += 900.0  # a 900-µV step: 600 µV after the common average of 3 channels
    assert gate.check(moving).reason == "motion"
    with pytest.raises(NeurOSDataError):
        gate.check(np.zeros((2, 256)))
    with pytest.raises(NeurOSDataError):
        gate.check(np.full((3, 256), np.nan))
    with pytest.raises(NeurOSConfigError):
        QualityThresholds(flat_uv=0.0)
    with pytest.raises(NeurOSConfigError):
        QualityThresholds(flat_uv=600.0, ptp_uv=500.0)
    with pytest.raises(NeurOSConfigError):
        QualityGate(("C3",))


# ---------------------------------------------------------------------- online reference
def test_online_reference_is_e2s_blend_refreshed_once_per_update() -> None:
    rng = np.random.default_rng(1)
    prior = np.diag([1.0, 2.0, 3.0])
    online = OnlineReference(prior, prior_windows=4.0, update_every=3)
    np.testing.assert_allclose(online.current.inv_sqrt, reference_from_mean(prior, 0).inv_sqrt)
    covs = [window_covariances(rng.normal(0, 1, (3, 256))) for _ in range(5)]
    for c in covs[:2]:
        online.add(c)
    # Two windows in: not refreshed yet (once per update_every), still the prior.
    np.testing.assert_allclose(online.current.inv_sqrt, reference_from_mean(prior, 0).inv_sqrt)
    online.add(covs[2])
    expected = (4.0 * prior + 3 * np.mean(covs[:3], axis=0)) / (4.0 + 3)
    np.testing.assert_allclose(online.blended_mean(), expected, rtol=1e-12)
    np.testing.assert_allclose(
        online.current.inv_sqrt, reference_from_mean(expected, 3).inv_sqrt, rtol=1e-10
    )
    # EMA mode forgets: after its memory, new windows weigh 1/ema each.
    ema = OnlineReference(prior, prior_windows=1.0, update_every=1, ema_windows=2)
    for c in covs:
        ema.add(c)
    assert ema.n_windows == 5
    online.reset()
    assert online.n_windows == 0
    with pytest.raises(NeurOSDataError):
        online.add(np.full((3, 3), np.nan))
    with pytest.raises(NeurOSDataError):
        online.add(np.eye(2))
    with pytest.raises(NeurOSConfigError):
        OnlineReference(prior, prior_windows=0.0, update_every=1)
    with pytest.raises(NeurOSConfigError):
        OnlineReference(prior, prior_windows=1.0, update_every=0)
    with pytest.raises(NeurOSDataError):
        OnlineReference(np.eye(3)[:2], prior_windows=1.0, update_every=1)


# ---------------------------------------------------------------------- policy
def test_policy_smooths_thresholds_debounces_and_flags_decoys() -> None:
    settings = PolicySettings(
        commands={"left_hand": "LEFT", "right_hand": "RIGHT"}, threshold=0.6, debounce=2
    )
    policy = Policy(("left_hand", "right_hand", "feet"), settings, hop_s=0.125)
    assert policy.alpha == pytest.approx(1 - math.exp(-0.125 / 0.5))
    confidence, probs = policy.snapshot()
    assert confidence == pytest.approx(1 / 3)
    assert set(probs) == {"LEFT", "RIGHT", OTHER}
    left = np.array([0.9, 0.05, 0.05])
    first = policy.decide(left)
    assert (first.command, first.hold_reason) == (HOLD, "low_confidence")  # debounce: 1 of 2
    second = policy.decide(left)
    assert second.command == "LEFT"
    assert second.probs[OTHER] == pytest.approx(0.05)
    # A decoy class winning is "unrecognized", never a command.
    policy.reset()
    feet = policy.decide(np.array([0.1, 0.1, 0.8]))
    assert (feet.command, feet.hold_reason) == (HOLD, "unrecognized")
    # Unsure: below the threshold.
    policy.reset()
    unsure = policy.decide(np.array([0.5, 0.4, 0.1]))
    assert unsure.hold_reason == "low_confidence"
    # Smoothing: one contrary window doesn't flip a settled decision.
    policy.reset()
    for _ in range(10):
        policy.decide(left)
    blip = policy.decide(np.array([0.05, 0.9, 0.05]))
    assert blip.command == "LEFT"
    for bad in ([0.5, 0.5], [0.7, 0.7, -0.4], [np.nan, 0.5, 0.5], [0.2, 0.2, 0.2]):
        with pytest.raises(NeurOSDataError):
            policy.decide(np.array(bad))
    with pytest.raises(NeurOSConfigError):
        Policy(("left_hand", "right_hand"), PolicySettings(commands={"feet": "DOWN"}), 0.125)
    with pytest.raises(NeurOSConfigError):
        PolicySettings(commands={"left_hand": HOLD})
    with pytest.raises(NeurOSConfigError):
        PolicySettings(threshold=0.4)


# ---------------------------------------------------------------------- parity
def test_engine_windows_equal_offline_windows_bit_for_bit(trained: dict[str, object]) -> None:
    """Principle 1: the engine's windows are the training pipeline's, whatever the chunking."""
    recording = _recording(3, 13, n_trials=6)
    seen: list[EngineWindow] = []
    engine = Engine(trained["decoder"], _config(recording, trained), observer=seen.append)  # type: ignore[arg-type]
    rng = np.random.default_rng(5)
    start = 0
    stamps = np.arange(recording.n_samples) / recording.sfreq
    while start < recording.n_samples:  # irregular chunk sizes, as a real inlet delivers
        size = int(rng.integers(1, 90))
        engine.push(recording.data[:, start : start + size], stamps[start : start + size])
        start += size

    prepared = preprocess_recording(recording, CONTRACT)
    profile = list(_profile(recording))
    rows = [prepared.ch_names.index(ch) for ch in profile]
    offline = np.asarray(
        sliding_windows(prepared.data[rows], CONTRACT.window_samples, CONTRACT.hop_samples),
        dtype=np.float32,
    )
    offline = np.asarray(common_average_reference(offline), dtype=np.float32)
    assert len(seen) == offline.shape[0]
    live = np.stack([w.referenced for w in seen])
    assert np.array_equal(live, offline)
    # Each window's time is its last sample's on the 128-Hz grid.
    ends = (np.arange(len(seen)) * CONTRACT.hop_samples + CONTRACT.window_samples - 1) / 128.0
    np.testing.assert_allclose([w.t for w in seen], ends, atol=1e-9)


def test_live_placement_equals_offline_placement(trained: dict[str, object]) -> None:
    """With the same reference, a window placed live equals the window placed for training."""
    prepared = preprocess_recording(_recording(4, 14, n_trials=10), CONTRACT)
    windows, _ = extract_epochs(prepared, epoch_s=(0.5, 2.5), contract=CONTRACT)
    offline, present, _ = to_layout(
        windows, LAYOUT, reference="car", alignment="ea", unit="session"
    )
    profile = list(_profile(prepared))
    car = np.asarray(common_average_reference(windows.pick_channels(profile).data), np.float32)
    reference = euclidean_reference(car)
    rows = [LAYOUT.index(ch) for ch in profile]
    live = np.stack([place_window(w, reference, rows, len(LAYOUT), 2.0) for w in car])
    np.testing.assert_allclose(live, offline / np.float32(2.0), rtol=1e-6, atol=1e-6)
    assert not present[:, LAYOUT.index("CP5")].any()
    assert not live[:, LAYOUT.index("CP5")].any()  # the missing channel stays zero
    with pytest.raises(NeurOSDataError):
        place_window(car[0][:3], reference, rows, len(LAYOUT), 1.0)


# ---------------------------------------------------------------------- end to end
def _task_accuracy(events: list[object], simulator: Simulator) -> tuple[float, int]:
    """Share of commands whose window lies inside a trial's task that name its class."""
    commands = {"left_hand": "LEFT", "right_hand": "RIGHT"}
    right = total = 0
    for event in events:
        if not isinstance(event, IntentEvent) or event.command == HOLD:
            continue
        for marker in simulator.markers:
            # The 2-s window ending at t lies in the 4-s task when cue + 2 <= t <= cue + 4.
            if marker.t + 2.0 <= event.t <= marker.t + 4.0:
                total += 1
                right += event.command == commands[marker.label]
    return (right / total if total else 0.0), total


def test_simulator_to_engine_decodes_a_new_person(trained: dict[str, object]) -> None:
    recording = _recording(5, 15, n_trials=40)
    simulator = Simulator(recording)
    engine = Engine(
        trained["decoder"],  # type: ignore[arg-type]
        _config(recording, trained, settling_s=10.0, steady_s=60.0),
    )
    events = simulator.run(engine)
    statuses = [e for e in events if isinstance(e, StatusEvent)]
    intents = [e for e in events if isinstance(e, IntentEvent)]
    assert [s.state for s in statuses] == ["settling", "adapting", "steady", "disconnected"]
    # Every decision before the filters settle is a warmup HOLD; decisions flow from 2.5 s.
    first_command = next(e for e in intents if e.hold_reason != "warmup")
    assert first_command.t == pytest.approx(TRANSIENT_FIRST_END, abs=1e-6)
    assert all(e.hold_reason == "warmup" for e in intents if e.t < TRANSIENT_FIRST_END - 1e-9)
    accuracy, n = _task_accuracy(intents, simulator)
    assert n > 100
    assert accuracy >= 0.8, accuracy
    assert all(e.latency_ms >= 0 for e in intents)
    assert [e.seq for e in events] == list(range(len(events)))  # one counter, no gaps
    for event in events[:50]:
        json.loads(event.to_json())


# The first window starting after the filter start-up (0.5 s = 64 samples) ends at
# (64 + 256 - 1) / 128 s.
TRANSIENT_FIRST_END = (64 + 256 - 1) / 128.0


def test_gap_restarts_the_stream_but_keeps_the_reference(trained: dict[str, object]) -> None:
    recording = _recording(6, 16, n_trials=6)
    engine = Engine(trained["decoder"], _config(recording, trained))  # type: ignore[arg-type]
    half = recording.n_samples // 2
    stamps = np.arange(recording.n_samples) / recording.sfreq
    stamps[half:] += 5.0  # the headset dropped out for 5 s
    events = engine.push(recording.data[:, :half], stamps[:half])
    before = engine.reference.n_windows
    assert before > 0
    events = engine.push(recording.data[:, half:], stamps[half:])
    gap = [e for e in events if isinstance(e, IntentEvent) and e.hold_reason == "signal_gap"]
    assert len(gap) == 1
    assert engine.reference.n_windows >= before  # the reference survived the restart
    after_gap = [e for e in events if isinstance(e, IntentEvent) and e.seq > gap[0].seq]
    assert after_gap[0].hold_reason == "warmup"  # filters restarted


def test_bad_signal_degrades_then_recovers(trained: dict[str, object]) -> None:
    recording = _recording(7, 17, n_trials=8)
    data = recording.data.copy()
    sfreq = recording.sfreq
    c3 = recording.ch_names.index("C3")
    data[c3, int(20 * sfreq) : int(26 * sfreq)] = 0.0  # C3 disconnected for 6 s
    engine = Engine(trained["decoder"], _config(recording, trained))  # type: ignore[arg-type]
    events = engine.push(data, np.arange(data.shape[1]) / sfreq)
    flat = [e for e in events if isinstance(e, IntentEvent) and e.hold_reason == "flat"]
    assert flat
    assert all("C3" in e.bad_channels and not e.quality_ok for e in flat)
    states = [e.state for e in events if isinstance(e, StatusEvent)]
    assert "degraded" in states
    assert states[states.index("degraded") + 1] == "settling"  # recovered (still < 30 s in)


def test_engine_configuration_errors(trained: dict[str, object]) -> None:
    recording = _recording(8, 18, n_trials=4)
    decoder = trained["decoder"]
    with pytest.raises(NeurOSConfigError, match="at least 2"):
        Engine(decoder, _config(recording, trained, stream_channels=("Pz", "C3")))  # type: ignore[arg-type]
    with pytest.raises(NeurOSConfigError, match="prior"):
        Engine(decoder, _config(recording, trained, prior=np.eye(3)))  # type: ignore[arg-type]
    engine = Engine(decoder, _config(recording, trained))  # type: ignore[arg-type]
    with pytest.raises(NeurOSConfigError, match="channels"):
        engine.push(np.zeros((3, 10)), np.arange(10) / 250.0)
    assert engine.push(np.zeros((12, 0)), np.zeros(0)) == []
    for bad in ({"input_scale": 0.0}, {"settling_s": -1.0}, {"ema_s": 0.0}):
        with pytest.raises(NeurOSConfigError):
            _config(recording, trained, **bad)


def test_simulator_replays_profiles_markers_and_noise() -> None:
    recording = _recording(9, 19, n_trials=4)
    simulator = Simulator(recording, channels=("C4", "C3"), chunk_s=0.1, t0=100.0)
    chunks = list(simulator.chunks())
    assert chunks[0][0].shape == (2, 25)
    assert chunks[0][1][0] == 100.0
    np.testing.assert_array_equal(
        np.concatenate([c for c, _ in chunks], axis=1),
        recording.data[[recording.ch_names.index("C4"), recording.ch_names.index("C3")]],
    )
    assert [m.label for m in simulator.markers] == list(recording.events.labels)
    assert simulator.markers[0].t == pytest.approx(100.0 + recording.events.onsets[0] / 250.0)
    noisy_a = Simulator(recording, noise_uv=5.0, seed=1).data
    noisy_b = Simulator(recording, noise_uv=5.0, seed=1).data
    assert np.array_equal(noisy_a, noisy_b)
    with pytest.raises(NeurOSConfigError):
        Simulator(recording, channels=("C3", "XX"))
    with pytest.raises(NeurOSConfigError):
        Simulator(recording, chunk_s=0.0)
    with pytest.raises(NeurOSConfigError):
        Simulator(recording).run(None, speed=0.0)  # type: ignore[arg-type]
