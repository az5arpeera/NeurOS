"""Load-time validation of recordings against the data contract (architecture §4.5).

Checks, each with an issue code:

==================== ======== ============================================================
code                 severity meaning
==================== ======== ============================================================
sampling_rate        error    rate differs from the registry's declared native rate
unknown_channel      error    a channel the registry doesn't declare for this dataset
non_finite           error    NaN or Inf samples (count per channel)
flat_channel         error    window-detrended std below ``flat_std_uv`` (dead electrode)
units                error    typical amplitude outside ``amplitude_mad_uv`` (volts vs µV bug)
too_short            error    shorter than one validation window
unknown_label        error    an event label missing from the vocabulary
no_events            error    a motor-imagery recording without task events
extreme_amplitude    flag     windows whose peak-to-peak *in the contract band* exceeds
                              ``ptp_flag_uv`` (counted)
epoch_outside        flag     events whose offline epoch runs past the recording (counted)
==================== ======== ============================================================

Errors make ``ValidationReport.ok`` false and ``raise_if_errors()`` raise
``NeurOSDataError``. Flags are counted and reported, never silently dropped (§4.5).

Two measurement aids, neither of which ever feeds the model (that's the causal filter chain in
``neuros.signal``):

- **Flatness** uses *window-detrended* broadband data: each window's linear trend is removed
  first. DC-coupled amplifiers (e.g. Biosemi in Cho2017) carry millivolt offsets and slow drift;
  without detrending every channel of such a file would fail although the EEG is fine.
- **Extreme amplitude and typical amplitude (units)** are measured in the contract band (4-30 Hz,
  zero-phase Butterworth), i.e. on what the model will actually see. Day-1 finding: on raw
  PhysionetMI (unfiltered: line noise and drift included) 39% of windows exceeded 500 µV
  peak-to-peak, which said nothing about artifacts the decoder would face.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, detrend, sosfiltfilt

from neuros.config.schemas import SignalContract, ValidationThresholds
from neuros.data.recording import Recording
from neuros.data.registry import DatasetSpec, task_epoch
from neuros.data.vocabulary import Vocabulary
from neuros.errors import DataContext, NeurOSDataError

Severity = Literal["error", "flag"]

# Windows processed per chunk: bounds memory on long, high-rate recordings (Stieger: 1 kHz,
# 60 channels → 64 windows ≈ 60 MB per float64 block).
_CHUNK_WINDOWS = 64
# Relative tolerance when comparing sampling rates.
_SFREQ_RTOL = 1e-6
# Butterworth order of the measurement band-pass (zero-phase, so the effective order doubles).
_MEASURE_ORDER = 4


@dataclass(frozen=True)
class Issue:
    """One finding of a validation run."""

    code: str
    severity: Severity
    message: str


@dataclass
class ValidationReport:
    """Everything validation found for one recording."""

    context: DataContext
    sfreq: float
    n_channels: int
    n_samples: int
    issues: list[Issue] = field(default_factory=list)
    label_counts: dict[str, int] = field(default_factory=dict)
    windows_total: int = 0
    windows_flagged: int = 0
    events_epoch_outside: int = 0
    amplitude_mad_uv: float = float("nan")
    flat_channels: list[str] = field(default_factory=list)
    non_finite: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when there are no error-level issues (flags are allowed)."""
        return not any(issue.severity == "error" for issue in self.issues)

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.sfreq if self.sfreq > 0 else float("nan")

    def add(self, code: str, severity: Severity, message: str) -> None:
        self.issues.append(Issue(code, severity, message))

    def errors(self) -> list[Issue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    def raise_if_errors(self) -> None:
        """Raise ``NeurOSDataError`` listing every error-level issue.

        Raises:
            NeurOSDataError: if any error-level issue was found.
        """
        errors = self.errors()
        if errors:
            details = "; ".join(f"{e.code}: {e.message}" for e in errors)
            raise NeurOSDataError(
                f"Validation failed ({len(errors)} errors): {details}", self.context
            )

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable summary (for result files and the CLI)."""
        return {
            "dataset": self.context.dataset,
            "subject": self.context.subject,
            "session": self.context.session,
            "run": self.context.run,
            "ok": self.ok,
            "sfreq": self.sfreq,
            "n_channels": self.n_channels,
            "duration_s": round(self.duration_s, 3),
            "label_counts": self.label_counts,
            "windows_total": self.windows_total,
            "windows_flagged": self.windows_flagged,
            "events_epoch_outside": self.events_epoch_outside,
            "amplitude_mad_uv": None
            if math.isnan(self.amplitude_mad_uv)
            else round(self.amplitude_mad_uv, 3),
            "flat_channels": self.flat_channels,
            "non_finite": self.non_finite,
            "issues": [issue.__dict__ for issue in self.issues],
        }


def _windows(data: NDArray[np.float32], window: int) -> Iterator[NDArray[np.float64]]:
    """Yield chunks of complete non-overlapping windows as float64 ``[windows, ch, samples]``."""
    n_channels, n_samples = data.shape
    n_windows = n_samples // window
    for start in range(0, n_windows, _CHUNK_WINDOWS):
        stop = min(start + _CHUNK_WINDOWS, n_windows)
        block = data[:, start * window : stop * window].astype(np.float64)
        # [ch, w*window] → [w, ch, window]
        yield block.reshape(n_channels, stop - start, window).transpose(1, 0, 2)


def validate_recording(
    recording: Recording,
    spec: DatasetSpec,
    vocabulary: Vocabulary,
    contract: SignalContract | None = None,
    thresholds: ValidationThresholds | None = None,
) -> ValidationReport:
    """Check one recording against the contract and the dataset's declarations.

    Never raises for data problems: it reports them (call ``raise_if_errors`` to enforce).

    Raises:
        NeurOSDataError: only if ``recording`` belongs to a different dataset than ``spec``.
    """
    contract = contract or SignalContract()
    thresholds = thresholds or ValidationThresholds()
    if recording.dataset != spec.key:
        raise NeurOSDataError(
            f"Recording of {recording.dataset!r} validated against spec {spec.key!r}",
            recording.context,
        )
    report = ValidationReport(
        context=recording.context,
        sfreq=recording.sfreq,
        n_channels=recording.n_channels,
        n_samples=recording.n_samples,
        label_counts=recording.events.counts(),
    )

    # 1. Sampling rate vs the registry.
    if not math.isclose(recording.sfreq, spec.native_sfreq, rel_tol=_SFREQ_RTOL):
        report.add(
            "sampling_rate",
            "error",
            f"{recording.sfreq} Hz, but the registry declares {spec.native_sfreq} Hz",
        )

    # 2. Channels vs the registry.
    undeclared = [ch for ch in recording.ch_names if ch not in spec.eeg_channels]
    if undeclared:
        report.add(
            "unknown_channel", "error", f"channels not declared for {spec.key}: {undeclared}"
        )

    # 3. Non-finite samples. Later statistics treat them as zeros so one bad sample doesn't hide
    #    every other finding; the error above already blocks the recording.
    finite = np.isfinite(recording.data)
    if not finite.all():
        bad_counts = (~finite).sum(axis=1)
        report.non_finite = {
            recording.ch_names[i]: int(count) for i, count in enumerate(bad_counts) if count
        }
        report.add("non_finite", "error", f"NaN/Inf samples per channel: {report.non_finite}")
    data = np.where(finite, recording.data, np.float32(0.0)) if not finite.all() else recording.data

    # 4. Window statistics: flatness, units, extreme amplitudes.
    _window_checks(data, recording, contract, thresholds, report)

    # 5. Labels and events.
    unknown_labels = sorted(set(recording.events.labels) - set(vocabulary.labels))
    if unknown_labels:
        report.add("unknown_label", "error", f"labels not in the vocabulary: {unknown_labels}")
    if spec.paradigm == "motor_imagery" and len(recording.events) == 0:
        report.add("no_events", "error", "motor-imagery recording without task events")
    _epoch_checks(recording, spec, contract, report)
    return report


def _band_stats(
    data: NDArray[np.float32], sfreq: float, window: int, band: tuple[float, float]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Band-limited window statistics after a zero-phase band-pass (measurement only).

    Returns:
        ``(worst_ptp, mads)``: per window, the largest peak-to-peak over channels
        (``[windows]``), and each channel's median absolute deviation (``[windows, channels]``).

    Filters one channel at a time so memory stays at one channel's length (Stieger: 1 kHz).
    """
    n_windows = data.shape[1] // window
    low, high = band[0], min(band[1], 0.45 * sfreq)  # stay below Nyquist for any native rate
    sos = butter(_MEASURE_ORDER, [low, high], btype="bandpass", fs=sfreq, output="sos")
    worst = np.zeros(n_windows, dtype=np.float64)
    mads = np.zeros((n_windows, data.shape[0]), dtype=np.float64)
    for index, channel in enumerate(data):
        filtered = sosfiltfilt(sos, channel.astype(np.float64))
        windows = filtered[: n_windows * window].reshape(n_windows, window)
        np.maximum(worst, windows.max(axis=1) - windows.min(axis=1), out=worst)
        centred = windows - np.median(windows, axis=1, keepdims=True)
        mads[:, index] = np.median(np.abs(centred), axis=1)
    return worst, mads


def _window_checks(
    data: NDArray[np.float32],
    recording: Recording,
    contract: SignalContract,
    thresholds: ValidationThresholds,
    report: ValidationReport,
) -> None:
    window = round(thresholds.window_s * recording.sfreq)
    if window < 2 or recording.n_samples < window:
        report.add(
            "too_short",
            "error",
            f"recording ({recording.n_samples} samples) is shorter than one "
            f"{thresholds.window_s} s validation window",
        )
        return
    # Flatness: broadband, window-detrended ("no signal at all" must not hide behind a band).
    sum_sq = np.zeros(recording.n_channels, dtype=np.float64)
    n_values = 0
    total = 0
    for block in _windows(data, window):
        # Remove each window's linear trend (DC offset and slow drift); see module docstring.
        flat = detrend(block, axis=-1, type="linear")
        sum_sq += np.square(flat).sum(axis=(0, 2))
        n_values += flat.shape[0] * flat.shape[2]
        total += flat.shape[0]

    # Extreme amplitude and typical amplitude: measured in the contract band, i.e. on what the
    # model will see, so line noise and drift can't fake an artifact or a unit error.
    worst, mads = _band_stats(data, recording.sfreq, window, contract.band_hz)
    flagged = int(np.sum(worst > thresholds.ptp_flag_uv))
    report.windows_total = total
    report.windows_flagged = flagged
    if flagged:
        report.add(
            "extreme_amplitude",
            "flag",
            f"{flagged}/{total} windows exceed {thresholds.ptp_flag_uv} µV peak-to-peak "
            f"in the {contract.band_hz[0]:g}-{contract.band_hz[1]:g} Hz band",
        )

    std = np.sqrt(sum_sq / max(n_values, 1))
    flat_mask = std < thresholds.flat_std_uv
    if flat_mask.all():
        # Every channel "flat" is a unit problem (data still in volts), not 30 dead electrodes:
        # report the root cause once instead of a misleading list of flat channels.
        report.amplitude_mad_uv = float(np.median(mads))
        report.add(
            "units",
            "error",
            f"every channel's std is below {thresholds.flat_std_uv} µV (typical amplitude "
            f"{report.amplitude_mad_uv:.3g} µV): the data is probably still in volts",
        )
        return
    flat_channels = [recording.ch_names[i] for i in np.flatnonzero(flat_mask)]
    if flat_channels:
        report.flat_channels = flat_channels
        report.add("flat_channel", "error", f"std < {thresholds.flat_std_uv} µV on {flat_channels}")

    # Typical amplitude: median over windows and *live* channels of the window MAD. Flat
    # channels are excluded so one dead electrode can't fake a unit error.
    report.amplitude_mad_uv = float(np.median(mads[:, ~flat_mask]))
    low, high = thresholds.amplitude_mad_uv
    if not low <= report.amplitude_mad_uv <= high:
        report.add(
            "units",
            "error",
            f"typical amplitude {report.amplitude_mad_uv:.4g} µV is outside [{low}, {high}] µV; "
            "check the volts → µV conversion",
        )


def _epoch_checks(
    recording: Recording, spec: DatasetSpec, contract: SignalContract, report: ValidationReport
) -> None:
    """Count events whose offline epoch [cue + start, cue + end] runs past the recording.

    The epoch is the dataset's own (D-030, ``task_epoch``); paradigms without class cues
    (EEGMAT) have none to check.
    """
    if not len(recording.events) or spec.class_on_screen_s is None:
        return
    epoch = task_epoch(spec, contract)
    start = round(epoch.start_s * recording.sfreq)
    end = round(epoch.end_s * recording.sfreq)
    onsets = recording.events.onsets
    outside = int(np.sum((onsets + start < 0) | (onsets + end > recording.n_samples)))
    report.events_epoch_outside = outside
    if outside:
        report.add(
            "epoch_outside",
            "flag",
            f"{outside}/{len(onsets)} events: offline epoch [{epoch.start_s:g}, "
            f"{epoch.end_s:g}] s runs past the recording",
        )
