"""Live sources: BrainFlow's synthetic board through the engine, units, headset mapping, LSL.

The LSL test opens network sockets (LSL discovers streams over UDP), which can raise a
firewall prompt on a developer machine, so it runs only with NEUROS_LSL_TESTS=1.
"""

from __future__ import annotations

import asyncio
import os
import threading

import pytest

pytest.importorskip("brainflow")  # the engine extra (`uv sync --extra engine`)
pytest.importorskip("pylsl")

from neuros.cli import _source_channels
from neuros.engine.bundle import ModelBundle
from neuros.engine.events import IntentEvent, StatusEvent
from neuros.engine.pipeline import Engine
from neuros.engine.runner import run_live
from neuros.engine.sources import BrainFlowSource, LslSource, publish_lsl, unit_factor
from neuros.errors import NeurOSConfigError

from .test_bundle import bundle  # noqa: F401  (a module-scoped fixture)


def test_units_convert_to_microvolts() -> None:
    for unit in ("uV", "µV", "μV", "microvolts", " Microvolts "):
        assert unit_factor(unit) == 1.0
    assert unit_factor("V") == 1e6
    assert unit_factor("mV") == 1e3
    with pytest.raises(NeurOSConfigError, match="unknown EEG unit"):
        unit_factor("counts")


def test_headset_profiles_name_the_stream_channels() -> None:
    assert _source_channels(["1", "2", "3", "4"], "ganglion") == ("C3", "Cz", "C4", "Pz")
    assert _source_channels(["C3", "C4"], None) == ("C3", "C4")
    with pytest.raises(NeurOSConfigError, match="aren't devices"):
        _source_channels(["EXG1", "2"], "ganglion")


def test_brainflow_board_streams_through_the_engine(bundle: ModelBundle) -> None:  # noqa: F811
    from brainflow.board_shim import BoardIds

    source = BrainFlowSource(BoardIds.SYNTHETIC_BOARD.value)
    assert {"C3", "Cz", "C4"} <= set(source.ch_names)
    engine = Engine(bundle.decoder(), bundle.engine_config(source.ch_names, source.sfreq))
    assert engine.profile == ("C3", "Cz", "C4")
    events = asyncio.run(run_live(source, engine, seconds=4.0))
    states = [e.state for e in events if isinstance(e, StatusEvent)]
    assert states[0] == "settling"
    assert states[-1] == "disconnected"
    intents = [e for e in events if isinstance(e, IntentEvent)]
    assert intents  # decisions flow from 2.5 s (2-s windows after the filters' 0.5-s start-up)
    assert not source._started  # closed whatever stopped the loop
    with pytest.raises(NeurOSConfigError, match="seconds"):
        asyncio.run(run_live(source, engine, seconds=0.0))


def test_boards_without_channel_names_need_them() -> None:
    from brainflow.board_shim import BoardIds

    with pytest.raises(NeurOSConfigError, match="pass ch_names"):
        BrainFlowSource(BoardIds.GANGLION_BOARD.value)
    with pytest.raises(NeurOSConfigError, match="EEG channels but names"):
        BrainFlowSource(BoardIds.GANGLION_BOARD.value, ch_names=["C3", "C4"])


@pytest.mark.skipif(os.environ.get("NEUROS_LSL_TESTS") != "1", reason="opens network sockets")
def test_lsl_virtual_headset_end_to_end(bundle: ModelBundle) -> None:  # noqa: F811
    from neuros.data.synthetic import SyntheticConfig, make_recording
    from neuros.engine.simulator import Simulator

    raw = make_recording(SyntheticConfig(n_trials=3, seed=5), dataset="SynL", subject=1)
    publisher = threading.Thread(
        target=publish_lsl, args=([Simulator(raw)],), kwargs={"name": "NeurOS-Test"}, daemon=True
    )
    publisher.start()
    source = LslSource("NeurOS-Test", resolve_timeout_s=10.0)
    assert source.ch_names == tuple(raw.ch_names)
    engine = Engine(bundle.decoder(), bundle.engine_config(source.ch_names, source.sfreq))
    events = asyncio.run(run_live(source, engine, seconds=5.0))
    assert any(isinstance(e, IntentEvent) for e in events)
