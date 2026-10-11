"""The optional guided calibration (D-040): protocol, cue sources, fine-tune and switch, failure,
cancel and reset, and app requests through the hub."""

from __future__ import annotations

import asyncio
import itertools
import json

import numpy as np
import pytest
import torch

from neuros.config.schemas import SignalContract
from neuros.data.synthetic import SyntheticConfig, make_recording
from neuros.engine.bundle import ModelBundle
from neuros.engine.calibration import (
    CalibrationProtocol,
    CalibrationRecipe,
    Calibrator,
    recorded_trials,
    scheduled_trials,
)
from neuros.engine.events import CalibrationEvent, IntentEvent
from neuros.engine.pipeline import Engine
from neuros.engine.runner import replay
from neuros.engine.server import EventHub
from neuros.engine.simulator import Marker, Simulator
from neuros.errors import NeurOSConfigError, NeurOSDataError
from neuros.eval.finetune import FinetuneSettings

from .synthetic_people import FAST_TRAIN
from .test_bundle import bundle  # noqa: F401

COMMANDS = {"left_hand": "LEFT", "right_hand": "RIGHT"}
CLASSES = ("left_hand", "right_hand")
RECIPE = CalibrationRecipe(
    FAST_TRAIN, FinetuneSettings(sizes=(6,), epochs=15, lr=5e-3, min_lr=5e-4)
)
# Synthetic recordings: cue at the marker, imagery for 4 s; E1's synthetic window starts 0.5 s in.
SYNTHETIC_OFFSET_S = 0.5
SYNTHETIC_TASK_S = 4.0


def _engine_and_simulator(bundle: ModelBundle, seed: int) -> tuple[Engine, Simulator]:  # noqa: F811
    raw = make_recording(SyntheticConfig(n_trials=40, seed=seed), dataset="SynC", subject=1)
    simulator = Simulator(raw)
    engine = Engine(bundle.decoder(), bundle.engine_config(raw.ch_names, raw.sfreq))
    return engine, simulator


def _recorded(
    engine: Engine,
    simulator: Simulator,
    bundle: ModelBundle,  # noqa: F811
    **protocol: object,
) -> Calibrator:
    return Calibrator(
        engine,
        bundle.model,
        recipe=RECIPE,
        commands=COMMANDS,
        protocol=CalibrationProtocol(**protocol),  # type: ignore[arg-type]
        markers=simulator.markers,
        recorded_window_offset_s=SYNTHETIC_OFFSET_S,
        recorded_task_s=SYNTHETIC_TASK_S,
    )


def test_protocol_orders_times_and_validates() -> None:
    protocol = CalibrationProtocol(seed=4)
    order = protocol.order(CLASSES)
    assert sorted(order) == sorted(CLASSES * 10)
    runs = [len(list(group)) for group in _runs(order)]
    assert max(runs) <= 2
    assert protocol.order(CLASSES) == order  # seeded
    assert CalibrationProtocol(seed=5).order(CLASSES) != order
    contract = SignalContract()
    assert protocol.window_offset_s(contract) == pytest.approx(1.0 + contract.cue_margin_s)
    assert 150 < protocol.expected_s(2) < 200  # "about 3 minutes"
    assert protocol.min_good() == 8
    with pytest.raises(NeurOSConfigError, match="doesn't fit"):
        CalibrationProtocol(imagine_s=1.5).window_offset_s(contract)
    for bad in (
        {"per_class": 0},
        {"cue_s": 0.0},
        {"rest_s": (3.0, 2.0)},
        {"min_good_fraction": 0.0},
    ):
        with pytest.raises(NeurOSConfigError):
            CalibrationProtocol(**bad)  # type: ignore[arg-type]


def _runs(order: tuple[str, ...]) -> list[list[str]]:
    groups: list[list[str]] = []
    for label in order:
        if groups and groups[-1][0] == label:
            groups[-1].append(label)
        else:
            groups.append([label])
    return groups


def test_scheduled_and_recorded_trials() -> None:
    contract = SignalContract()
    protocol = CalibrationProtocol(per_class=3, seed=1)
    trials = scheduled_trials(protocol, CLASSES, 100.0, contract)
    assert [t.number for t in trials] == list(range(1, 7))
    first = trials[0]
    assert first.cue_t == pytest.approx(100.0 + protocol.lead_s + protocol.ready_s)
    assert first.window_end_t == pytest.approx(first.cue_t + 1.25 + contract.window_s)
    assert [phase for phase, _, _ in first.phases] == ["ready", "cue", "imagine", "rest"]
    for previous, current in itertools.pairwise(trials):
        assert current.phases[0][1] >= previous.phases[-1][1] + protocol.rest_s[0] - 1e-9

    markers = [
        Marker(t, label)
        for t, label in [
            (5, "left_hand"),
            (15, "right_hand"),
            (25, "left_hand"),
            (35, "feet"),
            (45, "right_hand"),
            (55, "left_hand"),
        ]
    ]
    picked = recorded_trials(
        markers, CLASSES, 2, 10.0, window_offset_s=1.5, window_s=2.0, task_s=4.0
    )
    assert [(t.cue_t, t.label) for t in picked] == [
        (15, "right_hand"),
        (25, "left_hand"),
        (45, "right_hand"),
        (55, "left_hand"),
    ]
    assert picked[0].window_end_t == pytest.approx(15 + 1.5 + 2.0)
    with pytest.raises(NeurOSDataError, match="too few trials"):
        recorded_trials(markers, CLASSES, 3, 10.0, window_offset_s=1.5, window_s=2.0, task_s=4.0)


def test_recorded_calibration_fine_tunes_and_switches(bundle: ModelBundle) -> None:  # noqa: F811
    engine, simulator = _engine_and_simulator(bundle, seed=314)
    calibrator = _recorded(engine, simulator, bundle, per_class=6)
    before = {k: v.clone() for k, v in bundle.model.state_dict().items()}
    events = asyncio.run(
        replay([simulator], engine, calibrator=calibrator, calibrate_at_start=True)
    )
    phases = [e.phase for e in events if isinstance(e, CalibrationEvent)]
    assert phases[0] == "started"
    assert phases.count("cue") == 12
    assert phases[-2:] == ["training", "done"]
    done = next(e for e in events if isinstance(e, CalibrationEvent) and e.phase == "done")
    assert done.details["usable"] == {"LEFT": 6, "RIGHT": 6}
    assert 0.0 <= done.details["fit_on_trials"] <= 1.0
    assert calibrator.result is not None
    assert calibrator.result.ok
    assert engine.calibration_level == "quick"
    assert engine.decoder is not calibrator.base_decoder
    # Intents after the switch say so (and serialise it); those before don't.
    intents = [e for e in events if isinstance(e, IntentEvent)]
    assert intents[0].calibration_level is None
    assert intents[-1].calibration_level == "quick"
    assert json.loads(intents[-1].to_json())["calibration"]["level"] == "quick"
    seqs = [e.seq for e in events]
    assert seqs == sorted(seqs)  # one shared counter, never reused
    assert len(set(seqs)) == len(seqs)
    # The bundle's model is untouched; reset returns to it.
    for name, value in bundle.model.state_dict().items():
        assert torch.equal(value, before[name]), name
    reset = calibrator.handle("reset_calibration", 1e6)
    assert [e.phase for e in reset] == ["reset"]
    assert engine.calibration_level is None
    assert engine.decoder is calibrator.base_decoder


def test_bad_signal_fails_and_zero_calibration_stays(bundle: ModelBundle) -> None:  # noqa: F811
    engine, simulator = _engine_and_simulator(bundle, seed=315)
    simulator.data[:] = 0.0  # every electrode off: the gate flags every window "flat"
    calibrator = _recorded(engine, simulator, bundle, per_class=4)
    events = asyncio.run(
        replay([simulator], engine, calibrator=calibrator, calibrate_at_start=True)
    )
    failed = [e for e in events if isinstance(e, CalibrationEvent) and e.phase == "failed"]
    assert len(failed) == 1
    assert "usable" in failed[0].message
    assert "Zero calibration stays" in failed[0].message
    assert failed[0].details["usable"] == {"LEFT": 0, "RIGHT": 0}
    assert engine.calibration_level is None
    assert calibrator.result is not None
    assert not calibrator.result.ok


def test_too_few_recorded_trials_fails_at_start(bundle: ModelBundle) -> None:  # noqa: F811
    engine, simulator = _engine_and_simulator(bundle, seed=316)
    calibrator = _recorded(engine, simulator, bundle, per_class=30)  # the recording has 20 each
    events = calibrator.start(0.0)
    assert [e.phase for e in events] == ["failed"]
    assert "too few trials" in events[0].message
    assert not calibrator.active


def test_scheduled_calibration_through_app_requests(bundle: ModelBundle) -> None:  # noqa: F811
    """A live-style calibration: an app's request starts it; the engine schedules the trials
    and emits each phase on its clock; a second request is ignored while it runs."""
    engine, simulator = _engine_and_simulator(bundle, seed=317)
    calibrator = Calibrator(
        engine,
        bundle.model,
        recipe=RECIPE,
        commands=COMMANDS,
        protocol=CalibrationProtocol(per_class=2, min_good_fraction=0.5, seed=2),
    )
    hub = EventHub(["LEFT", "RIGHT"], requests=["calibrate", "cancel_calibration"])
    assert hub.receive(json.dumps({"type": "request", "request": "calibrate"}))
    assert hub.receive(json.dumps({"type": "request", "request": "calibrate"}))  # ignored later
    events = asyncio.run(replay([simulator], engine, hub=hub, calibrator=calibrator))
    calibration = [e for e in events if isinstance(e, CalibrationEvent)]
    phases = [e.phase for e in calibration]
    assert phases.count("started") == 1
    assert phases.count("ready") == phases.count("cue") == phases.count("imagine") == 4
    assert phases[-1] in ("done", "failed")  # synthetic imagery doesn't follow our cues
    cues = [e for e in calibration if e.phase == "cue"]
    assert sorted(e.command for e in cues) == ["LEFT", "LEFT", "RIGHT", "RIGHT"]
    assert all(e.duration_s == pytest.approx(1.0) for e in cues)
    times = [e.t for e in calibration if e.phase in ("ready", "cue", "imagine", "rest")]
    assert times == sorted(times)


def test_cancel_stops_a_running_calibration(bundle: ModelBundle) -> None:  # noqa: F811
    engine, simulator = _engine_and_simulator(bundle, seed=318)
    calibrator = _recorded(engine, simulator, bundle, per_class=4)
    assert [e.phase for e in calibrator.start(0.0)] == ["started"]
    assert calibrator.start(1.0) == []  # already running
    assert [e.phase for e in calibrator.handle("cancel_calibration", 2.0)] == ["cancelled"]
    assert not calibrator.active
    assert engine.calibration_level is None
    assert calibrator.handle("cancel_calibration", 3.0) == []
    with pytest.raises(NeurOSConfigError, match="unknown request"):
        calibrator.handle("self_destruct", 4.0)
    with pytest.raises(NeurOSConfigError, match="no command name"):
        Calibrator(engine, bundle.model, recipe=RECIPE, commands={"left_hand": "LEFT"})
    with pytest.raises(NeurOSConfigError, match="window offset"):
        Calibrator(
            engine, bundle.model, recipe=RECIPE, commands=COMMANDS, markers=simulator.markers
        )


def test_engine_refuses_a_decoder_for_another_layout(bundle: ModelBundle) -> None:  # noqa: F811
    engine, _ = _engine_and_simulator(bundle, seed=319)

    class Other:
        layout = ("C3", "C4")
        classes = CLASSES

        def predict_proba(self, x: np.ndarray) -> np.ndarray:
            return np.full((x.shape[0], 2), 0.5)

    with pytest.raises(NeurOSConfigError, match="layout"):
        engine.set_decoder(Other(), level="quick")  # type: ignore[arg-type]


def test_recipe_comes_from_the_q30_experiment_file() -> None:
    recipe = CalibrationRecipe.from_experiment("configs/e2_finetune.yaml")
    assert recipe.finetune.epochs == 50
    assert recipe.finetune.lr == pytest.approx(5e-4)
    config = recipe.finetune.train_config(recipe.train, 20)
    assert (config.windows_per_person, config.patience, config.swa_fraction) == (20, None, 0.0)
    with pytest.raises(NeurOSConfigError, match="no finetune section"):
        CalibrationRecipe.from_experiment("configs/e2.yaml")
