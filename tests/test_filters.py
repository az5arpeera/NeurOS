"""Causal filters (architecture §4.2): the response we claim, and chunking that changes nothing."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError
from scipy.signal import sosfilt

from neuros.config.schemas import SignalContract
from neuros.data.montage import PACKAGED_PROFILES, load_profile
from neuros.data.registry import DATASETS
from neuros.errors import NeurOSConfigError, NeurOSSignalError
from neuros.signal.filters import (
    TRANSIENT_S,
    CausalFilter,
    design_filter,
    frequency_response,
    group_delay_s,
    settle_time_s,
)

# Every native rate in scope: the registry's datasets and the headset profiles.
NATIVE_RATES = sorted(
    {spec.native_sfreq for spec in DATASETS.values()}
    | {load_profile(name).native_sfreq for name in PACKAGED_PROFILES}
)


def _db(value: complex) -> float:
    return float(20 * np.log10(max(abs(value), 1e-15)))


@pytest.mark.parametrize("rate", NATIVE_RATES)
def test_design_at_every_native_rate_in_scope(rate: float) -> None:
    design = design_filter(rate)
    assert design.notch_hz == (50.0, 60.0)
    assert design.skipped_notch_hz == ()
    gains = frequency_response(design, [1.0, 10.0, 20.0, 50.0, 60.0, 64.0])
    assert abs(_db(gains[1])) < 0.5  # mu band passes
    assert abs(_db(gains[2])) < 0.5  # beta band passes
    assert _db(gains[0]) < -20  # slow drift removed
    assert _db(gains[3]) < -40
    assert _db(gains[4]) < -40  # line noise removed
    assert _db(gains[5]) < -30  # 128 Hz Nyquist: the band-pass is the anti-alias filter


@pytest.mark.parametrize("rate", NATIVE_RATES)
def test_settling_time_is_within_the_flagged_transient(rate: float) -> None:
    # TRANSIENT_S flags windows near a (re)start; it must cover the filter's real settling.
    assert settle_time_s(design_filter(rate)) < TRANSIENT_S / 2


def test_group_delay_matches_the_documented_value() -> None:
    delay = group_delay_s(design_filter(250.0), 10.0)
    assert 0.025 < delay < 0.045  # documented: ≈ 35 ms at 10 Hz


def test_notch_at_or_above_nyquist_is_skipped_and_recorded() -> None:
    design = design_filter(110.0)  # Nyquist 55 Hz: 50 Hz notch fits, 60 Hz doesn't
    assert design.notch_hz == (50.0,)
    assert design.skipped_notch_hz == (60.0,)
    assert design.describe()["filter_notch_skipped_hz"] == "60"


@pytest.mark.parametrize("rate", [50.0, 0.0, -250.0, float("nan"), float("inf")])
def test_impossible_rates_are_rejected(rate: float) -> None:
    with pytest.raises(NeurOSConfigError):
        design_filter(rate)


def test_contract_rejects_bad_filter_settings() -> None:
    with pytest.raises(ValidationError):
        SignalContract(filter_order=0)
    with pytest.raises(ValidationError):
        SignalContract(filter_order=12)
    with pytest.raises(ValidationError):
        SignalContract(notch_q=0.0)


def test_design_is_immutable() -> None:
    design = design_filter(250.0)
    with pytest.raises(ValueError, match="read-only"):
        design.sos[0, 0] = 1.0


def _noise(n_channels: int, n_samples: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_channels, n_samples)) * 20 + rng.normal(0, 3000, (n_channels, 1))


@given(cuts=st.lists(st.integers(min_value=1, max_value=1999), max_size=12, unique=True))
def test_chunked_filtering_is_bit_identical(cuts: list[int]) -> None:
    x = _noise(3, 2000)
    design = design_filter(250.0)
    whole = CausalFilter(design, 3).process(x)
    streaming = CausalFilter(design, 3)
    bounds = [0, *sorted(cuts), x.shape[1]]
    pieces = [streaming.process(x[:, a:b]) for a, b in pairwise(bounds)]
    assert np.array_equal(np.concatenate(pieces, axis=1), whole)


def test_dc_offset_does_not_cause_a_start_transient() -> None:
    # A DC-coupled amplifier with a 5 mV offset: a zero-state start would ring for ~1000 µV.
    rng = np.random.default_rng(1)
    x = 5000.0 + rng.standard_normal((2, 1000))
    design = design_filter(250.0)
    y = CausalFilter(design, 2).process(x)
    naive = sosfilt(np.array(design.sos), x, axis=-1)
    assert np.abs(y).max() < 10.0
    assert np.abs(naive).max() > 1000.0


def test_reset_restarts_like_a_fresh_filter() -> None:
    x = _noise(2, 500)
    design = design_filter(250.0)
    filt = CausalFilter(design, 2)
    filt.process(_noise(2, 300, seed=5))
    filt.reset()
    assert not filt.started
    assert np.array_equal(filt.process(x), CausalFilter(design, 2).process(x))


@pytest.mark.parametrize(
    "bad",
    [
        np.full((2, 10), np.nan),
        np.zeros((3, 10)),
        np.zeros(10),
        np.array([["a", "b"], ["c", "d"]]),
        [[0.0, 1.0], [2.0, 3.0]],
    ],
)
def test_bad_chunks_are_rejected_without_touching_state(bad: object) -> None:
    design = design_filter(250.0)
    filt = CausalFilter(design, 2)
    first = _noise(2, 100)
    filt.process(first)
    with pytest.raises(NeurOSSignalError):
        filt.process(bad)  # type: ignore[arg-type]
    rest = _noise(2, 100, seed=2)
    reference = CausalFilter(design, 2)
    reference.process(first)
    assert np.array_equal(filt.process(rest), reference.process(rest))


def test_empty_chunk_is_a_no_op() -> None:
    filt = CausalFilter(design_filter(250.0), 2)
    out = filt.process(np.zeros((2, 0)))
    assert out.shape == (2, 0)
    assert not filt.started


def test_invalid_channel_count_is_a_config_error() -> None:
    with pytest.raises(NeurOSConfigError):
        CausalFilter(design_filter(250.0), 0)
