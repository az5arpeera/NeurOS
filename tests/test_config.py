"""Configuration schemas: the committed config must equal the signed-off contract."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from neuros.config.schemas import DataConfig, SignalContract, ValidationThresholds, load_data_config
from neuros.errors import NeurOSConfigError

from .conftest import REPO_ROOT


def test_contract_defaults_match_architecture() -> None:
    contract = SignalContract()
    assert contract.sfreq == 128.0
    assert contract.band_hz == (4.0, 30.0)
    assert contract.window_samples == 256
    assert contract.hop_samples == 16
    assert contract.epoch_s == (0.5, 2.5)
    assert contract.reference == "car"


def test_committed_config_is_the_contract() -> None:
    config = load_data_config(REPO_ROOT / "configs" / "data.yaml")
    assert config.signal == SignalContract()
    assert config.validation == ValidationThresholds()
    assert config.data_dir == (REPO_ROOT / "data").resolve()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"band_hz": (4.0, 70.0)},  # above Nyquist at 128 Hz
        {"band_hz": (30.0, 4.0)},
        {"hop_s": 3.0},  # hop longer than the window
        {"window_s": 2.003},  # not a whole number of samples
        {"epoch_s": (0.5, 3.0)},  # epoch ≠ one window
        {"notch_hz": (-50.0,)},
    ],
)
def test_invalid_contracts(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        SignalContract(**kwargs)  # type: ignore[arg-type]


def test_invalid_thresholds() -> None:
    with pytest.raises(ValidationError):
        ValidationThresholds(amplitude_mad_uv=(200.0, 0.5))


def test_load_errors(tmp_path: Path) -> None:
    with pytest.raises(NeurOSConfigError, match="not found"):
        load_data_config(tmp_path / "missing.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("signal: {sfreq: -1}\n", encoding="utf-8")
    with pytest.raises(NeurOSConfigError, match="invalid data config"):
        load_data_config(bad)
    not_yaml = tmp_path / "x.yaml"
    not_yaml.write_text("a: [1,\n", encoding="utf-8")
    with pytest.raises(NeurOSConfigError, match="invalid YAML"):
        load_data_config(not_yaml)


def test_environment_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEUROS_DATA_DIR", str(tmp_path / "elsewhere"))
    config = DataConfig().resolved(tmp_path)
    assert config.data_dir == tmp_path / "elsewhere"
    assert config.cache_dir == (tmp_path / "cache").resolve()
