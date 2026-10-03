"""C-core and headset profiles (architecture §4.3)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from neuros.data.channels import C_CORE_ROWS, row_of, sort_spatially
from neuros.data.montage import (
    PACKAGED_PROFILES,
    compute_c_core,
    load_c_core,
    load_profile,
)
from neuros.data.registry import DATASETS, keys_with_role
from neuros.errors import NeurOSChannelError, NeurOSConfigError

from .conftest import make_spec

EXPECTED_C_CORE = (
    "F3", "Fz", "F4",
    "FC5", "FC3", "FC1", "FC2", "FC4", "FC6",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
    "P3", "Pz", "P4",
)  # fmt: skip


def test_c_core_is_recomputable_and_committed() -> None:
    computed = compute_c_core([DATASETS[k] for k in keys_with_role("core")])
    assert computed == load_c_core() == EXPECTED_C_CORE


def test_c_core_properties() -> None:
    core = load_c_core()
    assert len(core) == 26
    assert list(core) == sort_spatially(core)
    assert all(row_of(ch) in C_CORE_ROWS for ch in core)
    assert {"C3", "Cz", "C4"} <= set(core)
    # FCz is absent because Lee2019_MI doesn't record it (it's that cap's reference site).
    assert "FCz" not in core


def test_compute_c_core_errors() -> None:
    with pytest.raises(NeurOSConfigError, match="at least one"):
        compute_c_core([])
    no_motor = make_spec(eeg_channels=("Fz", "Pz"))
    with pytest.raises(NeurOSConfigError, match="motor-essential"):
        compute_c_core([no_motor])


@pytest.mark.parametrize("name", PACKAGED_PROFILES)
def test_packaged_profiles_load_and_cover_motor_cortex(name: str) -> None:
    profile = load_profile(name)
    assert profile.name == name
    profile.require_motor_coverage(load_c_core())


def test_ganglion_profile_matches_d025() -> None:
    profile = load_profile("ganglion")
    present, missing = profile.coverage(load_c_core())
    assert present == ("C3", "Cz", "C4", "Pz")
    assert len(missing) == 22
    assert profile.native_sfreq == 200.0
    assert profile.electrodes == "wet"


def test_profile_without_c4_is_rejected(tmp_path: Path) -> None:
    raw = load_profile("bcic2b").model_dump(mode="json")
    raw["channels"] = [{"device": "C3", "canonical": "C3"}, {"device": "Cz", "canonical": "Cz"}]
    path = tmp_path / "half.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(NeurOSChannelError, match="C4"):
        load_profile(path).require_motor_coverage(load_c_core())


@pytest.mark.parametrize(
    "channels",
    [
        [{"device": "1", "canonical": "c3"}],  # not canonical
        [
            {"device": "1", "canonical": "C3"},
            {"device": "1", "canonical": "C4"},
        ],  # duplicate device
        [
            {"device": "1", "canonical": "C3"},
            {"device": "2", "canonical": "C3"},
        ],  # duplicate position
    ],
)
def test_invalid_profiles(tmp_path: Path, channels: list[dict[str, str]]) -> None:
    raw = load_profile("bcic2b").model_dump(mode="json")
    raw["channels"] = channels
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(NeurOSConfigError, match="invalid headset profile"):
        load_profile(path)


def test_unknown_profile() -> None:
    with pytest.raises(NeurOSConfigError, match="Unknown headset"):
        load_profile("muse")
