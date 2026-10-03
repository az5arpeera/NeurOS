"""Channel-name normalisation and spatial ordering (architecture §4.3)."""

from __future__ import annotations

import random

import pytest
from hypothesis import given
from hypothesis import strategies as st

from neuros.data.channels import (
    C_CORE_ROWS,
    canonical_names,
    in_rows,
    is_scalp_electrode,
    lateral_position,
    normalize_channel_name,
    require_canonical,
    row_of,
    sort_spatially,
)
from neuros.data.montage import load_c_core
from neuros.errors import NeurOSChannelError


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("C3..", "C3"),  # PhysioNet EEGMMIDB padding
        ("Fc5.", "FC5"),
        ("Cpz.", "CPz"),
        ("Iz..", "Iz"),
        ("fcz", "FCz"),
        ("CPZ", "CPz"),
        ("FP1", "Fp1"),
        ("Fpz", "Fpz"),
        ("poz", "POz"),
        ("afz", "AFz"),
        (" Cz ", "Cz"),
        ("EEG Fp1", "Fp1"),  # EDF type prefix
        ("EEG C3-REF", "C3"),  # reference suffix
        ("C3-A1", "C3"),
        ("T3", "T7"),  # old 10-20 names
        ("T4", "T8"),
        ("T5", "P7"),
        ("t6", "P8"),
        ("EEG T3", "T7"),
        ("FTT9h", "FTT9h"),  # 10-05 half position (Lee2019)
    ],
)
def test_normalize_known_spellings(raw: str, expected: str) -> None:
    assert normalize_channel_name(raw) == expected


@pytest.mark.parametrize(
    "raw", ["EOG1", "EMG1", "EMGd", "stim", "STI 014", "Fp1-F7", "", "   ", "C33", "foo"]
)
def test_normalize_rejects_non_electrodes(raw: str) -> None:
    assert normalize_channel_name(raw) is None


def test_reference_sites_are_not_scalp_inputs() -> None:
    assert normalize_channel_name("A2-A1") == "A2"
    assert not is_scalp_electrode("A2")
    assert not is_scalp_electrode("M1")
    assert is_scalp_electrode("C3")


def test_normalize_rejects_non_strings() -> None:
    with pytest.raises(TypeError):
        normalize_channel_name(3)  # type: ignore[arg-type]


def test_old_names_are_not_canonical() -> None:
    names = canonical_names()
    assert {"T3", "T4", "T5", "T6"}.isdisjoint(names)
    assert {"T7", "T8", "P7", "P8", "FCz", "CPz", "Fpz", "FTT9h"} <= names


@given(st.sampled_from(sorted(canonical_names())))
def test_normalization_is_idempotent_and_spelling_insensitive(name: str) -> None:
    assert normalize_channel_name(name) == name
    assert normalize_channel_name(name.upper()) == name
    assert normalize_channel_name(name.lower()) == name
    assert normalize_channel_name(name + "..") == name
    assert normalize_channel_name("EEG " + name) == name


def test_require_canonical() -> None:
    assert require_canonical(["C3", "Cz"]) == ("C3", "Cz")
    with pytest.raises(NeurOSChannelError, match="Non-canonical"):
        require_canonical(["c3"])
    with pytest.raises(NeurOSChannelError, match="Duplicate"):
        require_canonical(["C3", "C3"])


@pytest.mark.parametrize(
    ("name", "lateral"),
    [
        ("C3", -2.0),
        ("Cz", 0.0),
        ("C4", 2.0),
        ("C5", -3.0),
        ("FC6", 3.0),
        ("T7", -4.0),
        ("FCC5h", -2.5),
        ("FCC6h", 2.5),
    ],
)
def test_lateral_position(name: str, lateral: float) -> None:
    assert lateral_position(name) == lateral


def test_row_of() -> None:
    assert row_of("FC3") == "FC"
    assert row_of("Fpz") == "Fp"
    assert row_of("CPz") == "CP"
    assert row_of("FTT9h") == "FTT"
    assert row_of("LPA") is None


def test_sort_spatially_recovers_c_core_order() -> None:
    core = list(load_c_core())
    shuffled = core[:]
    random.Random(0).shuffle(shuffled)
    assert sort_spatially(shuffled) == core


def test_in_rows_keeps_only_sensorimotor_rows() -> None:
    names = ["F3", "FC3", "C3", "CP3", "P3", "FT7", "T7", "TP7", "AF3", "PO3", "Fp1", "O1"]
    assert in_rows(names) == ["F3", "FC3", "C3", "CP3", "P3"]
    assert C_CORE_ROWS == ("F", "FC", "C", "CP", "P")
