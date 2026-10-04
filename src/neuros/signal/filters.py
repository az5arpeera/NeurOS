"""Causal IIR filters: the contract band-pass and line-noise notches (architecture §4.2).

Design, at the signal's **native** rate (the 128 Hz grid comes after, in ``resample``):

- **Band-pass** 4-30 Hz, Butterworth, second-order sections (SOS). SOS keeps an 8-pole IIR
  numerically stable at 1 kHz, where a single transfer function's coefficients would lose
  precision. The 30 Hz edge also serves as the resampler's anti-alias filter: content that could
  alias into 4-30 Hz at 128 Hz lies above 98 Hz, where the band-pass is down > 80 dB.
- **Notches** at 50 and 60 Hz (both: datasets come from Europe, Asia and the US). The band-pass
  alone leaves line noise only ~18 dB down at 50 Hz, and raw line noise can be 100× the EEG.
  Notches at or above the native Nyquist frequency are skipped and recorded.

Runtime: ``CausalFilter`` carries the filter state across chunks, so streaming a recording in
chunks of any size is **bit-identical** to filtering it in one call (tested). That is what lets
offline training data and the live engine share one implementation.

**Start-up:** the state is initialised to the filter's steady state for a constant input equal to
the first sample. DC-coupled amplifiers carry offsets of several mV; starting from a zero state
would turn that offset into a large step transient. What remains is a short transient from the
signal itself (99 % of the impulse-response energy within ~0.22 s at every native rate in scope),
so windows that start within ``TRANSIENT_S`` of a (re)start are flagged, never silently used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, freqz_sos, iirnotch, sosfilt, sosfilt_zi, tf2sos

from neuros.config.schemas import SignalContract
from neuros.errors import DataContext, NeurOSConfigError
from neuros.signal._checks import check_chunk

# Windows starting earlier than this after a filter (re)start are flagged as transient.
# ≥ 2× the measured 99 %-energy settling time (0.22 s); a test re-measures it for every native
# rate in the registry and the headset profiles.
TRANSIENT_S = 0.5

# Poles must sit strictly inside the unit circle by this margin, or the design is rejected.
_POLE_MARGIN = 1e-9


@dataclass(frozen=True, eq=False)
class FilterDesign:
    """A causal filter for one native sampling rate (immutable; share it across channels).

    Attributes:
        sfreq: native sampling rate (Hz) the coefficients were designed for.
        sos: second-order sections ``[n_sections, 6]`` (band-pass first, then notches).
        band_hz, order: the band-pass edges and Butterworth order per edge.
        notch_hz: notch frequencies actually applied; ``skipped_notch_hz`` were at or above
            the Nyquist frequency.
        notch_q: notch quality factor.
    """

    sfreq: float
    sos: NDArray[np.float64]
    band_hz: tuple[float, float]
    order: int
    notch_hz: tuple[float, ...]
    skipped_notch_hz: tuple[float, ...]
    notch_q: float
    # Steady-state initial conditions for a unit constant input, [n_sections, 2] (cached).
    zi_unit: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        sos = np.array(self.sos, dtype=np.float64)  # private copy
        if sos.ndim != 2 or sos.shape[1] != 6 or sos.shape[0] == 0:
            raise NeurOSConfigError(f"SOS must be [n_sections, 6], got shape {sos.shape}")
        if not np.isfinite(sos).all():
            raise NeurOSConfigError("SOS coefficients contain NaN/Inf")
        sos.flags.writeable = False
        zi = np.asarray(sosfilt_zi(sos), dtype=np.float64)
        zi.flags.writeable = False
        object.__setattr__(self, "sos", sos)
        object.__setattr__(self, "zi_unit", zi)

    @property
    def n_sections(self) -> int:
        return int(self.sos.shape[0])

    def describe(self) -> dict[str, str]:
        """Provenance strings (recorded with every preprocessed recording)."""
        return {
            "filter": "butterworth-sos-causal",
            "filter_sfreq": f"{self.sfreq:g}",
            "filter_band_hz": f"{self.band_hz[0]:g}-{self.band_hz[1]:g}",
            "filter_order": str(self.order),
            "filter_notch_hz": ",".join(f"{f:g}" for f in self.notch_hz) or "none",
            "filter_notch_skipped_hz": ",".join(f"{f:g}" for f in self.skipped_notch_hz) or "none",
            "filter_notch_q": f"{self.notch_q:g}",
        }


def design_filter(sfreq: float, contract: SignalContract | None = None) -> FilterDesign:
    """Design the contract's causal band-pass + notches for a native sampling rate.

    Raises:
        NeurOSConfigError: if the rate is not a positive finite number, the band doesn't fit
            below its Nyquist frequency, or the resulting filter is unstable.
    """
    contract = contract or SignalContract()
    if not (isinstance(sfreq, (int, float)) and math.isfinite(sfreq) and sfreq > 0):
        raise NeurOSConfigError(f"sampling rate must be a positive finite number, got {sfreq!r}")
    sfreq = float(sfreq)
    nyquist = sfreq / 2
    low, high = contract.band_hz
    if not 0 < low < high < nyquist:
        raise NeurOSConfigError(
            f"band {contract.band_hz} Hz doesn't fit below the Nyquist frequency of a "
            f"{sfreq:g} Hz signal ({nyquist:g} Hz)"
        )
    sections = [
        butter(contract.filter_order, [low, high], btype="bandpass", fs=sfreq, output="sos")
    ]
    applied: list[float] = []
    skipped: list[float] = []
    for f0 in contract.notch_hz:
        # A notch at (or numerically at) Nyquist can't be built; the band-pass already removes
        # anything that high, so skipping it is safe, and it's recorded.
        if f0 >= nyquist * (1 - 1e-6):
            skipped.append(float(f0))
            continue
        b, a = iirnotch(f0, contract.notch_q, fs=sfreq)
        sections.append(tf2sos(b, a))
        applied.append(float(f0))
    sos = np.vstack(sections).astype(np.float64)
    _check_stable(sos, sfreq)
    return FilterDesign(
        sfreq=sfreq,
        sos=sos,
        band_hz=(float(low), float(high)),
        order=contract.filter_order,
        notch_hz=tuple(applied),
        skipped_notch_hz=tuple(skipped),
        notch_q=float(contract.notch_q),
    )


def _check_stable(sos: NDArray[np.float64], sfreq: float) -> None:
    """Reject designs with a pole on or outside the unit circle (would blow up over time)."""
    for index, section in enumerate(sos):
        poles = np.roots(section[3:])
        if poles.size and float(np.max(np.abs(poles))) >= 1 - _POLE_MARGIN:
            raise NeurOSConfigError(
                f"filter section {index} at {sfreq:g} Hz is unstable (pole magnitude "
                f"{float(np.max(np.abs(poles))):.12f}); lower filter_order or check band_hz"
            )


def frequency_response(
    design: FilterDesign, freqs_hz: NDArray[np.float64] | list[float]
) -> NDArray[np.complex128]:
    """Complex frequency response of the whole chain at the given frequencies (Hz)."""
    freqs = np.asarray(freqs_hz, dtype=np.float64)
    if freqs.ndim != 1 or np.any(freqs < 0) or np.any(freqs > design.sfreq / 2):
        raise NeurOSConfigError(f"frequencies must be a 1-D array within [0, {design.sfreq / 2}]")
    _, response = freqz_sos(design.sos, worN=freqs, fs=design.sfreq)
    return np.asarray(response, dtype=np.complex128)


def group_delay_s(design: FilterDesign, freq_hz: float) -> float:
    """Group delay (seconds) at one frequency: how late the filter shows that frequency.

    Computed as -dφ/dω by a central difference on the unwrapped phase.
    """
    nyquist = design.sfreq / 2
    step = min(0.01, freq_hz / 2, (nyquist - freq_hz) / 2)
    if step <= 0:
        raise NeurOSConfigError(f"group delay needs 0 < freq < Nyquist ({nyquist:g} Hz)")
    freqs = np.array([freq_hz - step, freq_hz, freq_hz + step])
    phase = np.unwrap(np.angle(frequency_response(design, freqs)))
    return float(-(phase[2] - phase[0]) / (2 * np.pi * 2 * step))


def settle_time_s(design: FilterDesign, energy: float = 0.99, horizon_s: float = 10.0) -> float:
    """Time until ``energy`` of the impulse response's total energy has passed (seconds)."""
    if not 0 < energy < 1:
        raise NeurOSConfigError("energy must be in (0, 1)")
    n = max(2, round(horizon_s * design.sfreq))
    impulse = np.zeros(n)
    impulse[0] = 1.0
    # scipy's Cython kernel wants a writable buffer; the design's copy is read-only.
    response = sosfilt(np.array(design.sos), impulse)
    cumulative = np.cumsum(np.square(response))
    if cumulative[-1] <= 0:
        return 0.0
    index = int(np.searchsorted(cumulative / cumulative[-1], energy))
    return index / design.sfreq


class CausalFilter:
    """Streaming application of a ``FilterDesign`` to ``[channels, samples]`` chunks.

    Not thread-safe: one instance per stream. Feed chunks in time order; call ``reset()`` after
    a gap or a reconnect (the next chunk re-initialises the state from its first sample).
    """

    def __init__(
        self, design: FilterDesign, n_channels: int, context: DataContext | None = None
    ) -> None:
        if not isinstance(n_channels, int) or n_channels < 1:
            raise NeurOSConfigError(f"n_channels must be a positive int, got {n_channels!r}")
        self.design = design
        # Private writable copy: scipy's Cython kernel rejects the design's read-only array.
        self._sos = np.array(design.sos)
        self.n_channels = n_channels
        self.context = context or DataContext()
        self._zi: NDArray[np.float64] | None = None
        self.samples_since_reset = 0

    @property
    def started(self) -> bool:
        """True once a chunk has initialised the filter state."""
        return self._zi is not None

    def reset(self) -> None:
        """Forget the filter state (after a gap or reconnect)."""
        self._zi = None
        self.samples_since_reset = 0

    def process(self, chunk: NDArray[np.floating]) -> NDArray[np.float64]:
        """Filter one chunk ``[channels, samples]``; returns float64 of the same shape.

        Raises:
            NeurOSSignalError: on a wrong shape or channel count, or NaN/Inf samples (the state
                is left untouched, so the caller can skip the chunk or reset).
        """
        x = check_chunk(chunk, self.n_channels, self.context)
        if x.shape[1] == 0:
            return x
        # First chunk since a (re)start: the steady state for a constant input equal to each
        # channel's first sample, [sections, 2] × [channels] → [sections, channels, 2].
        zi = self.design.zi_unit[:, None, :] * x[None, :, 0, None] if self._zi is None else self._zi
        y, zi_next = sosfilt(self._sos, x, axis=-1, zi=zi)
        self._zi = np.asarray(zi_next, dtype=np.float64)
        self.samples_since_reset += x.shape[1]
        return np.asarray(y, dtype=np.float64)
