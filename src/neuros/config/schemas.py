"""Configuration schemas: the canonical signal contract and data-layer settings.

The defaults *are* the signed-off contract (architecture §4.2, D-022). Configs may only override
them for named experiments (e.g. E0b tries other bands), and every result records the config
hash (principle 5), so an override can never go unnoticed.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from neuros.errors import NeurOSConfigError


class SignalContract(BaseModel):
    """What every window looks like after preprocessing (architecture §4.2)."""

    # No inf/NaN: an infinite window would otherwise crash the sample check below with an
    # OverflowError instead of a clean validation error.
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    units: Literal["uV"] = "uV"
    sfreq: float = Field(default=128.0, gt=0)
    # Motor-imagery information sits in mu (8-12 Hz) and beta (13-30 Hz); 4 Hz keeps theta for
    # the brain-state channel and drops slow drift. The 30 Hz top excludes most muscle activity.
    band_hz: tuple[float, float] = (4.0, 30.0)
    # Butterworth order per band edge (scipy doubles it for a band-pass: 4 → 8 poles). Measured
    # 2026-10-03 at every native rate in scope (160-1000 Hz): group delay ≈ 35 ms at 10 Hz,
    # 99 % of the impulse-response energy within 0.22 s, ≥ 30 dB down at 64 Hz (the 128 Hz
    # Nyquist, so the band-pass doubles as the resampler's anti-alias filter). Order 2 halves the
    # delay but leaves only 15-29 dB at 64 Hz.
    filter_order: int = Field(default=4, ge=1, le=8)
    notch_hz: tuple[float, ...] = (50.0, 60.0)
    # Quality factor of each notch: -3 dB width = f0 / Q (1.7 Hz at 50 Hz).
    notch_q: float = Field(default=30.0, gt=0)
    window_s: float = Field(default=2.0, gt=0)
    hop_s: float = Field(default=0.125, gt=0)
    # Offline motor-imagery epoch relative to the cue: skips the visual response to the cue's
    # onset. Not the whole cue: BNCI2014-001's arrow stays on screen until 1.25 s, and
    # zero-calibration models decode it (E0 diagnostics, 4 Oct; proposal Q25).
    epoch_s: tuple[float, float] = (0.5, 2.5)
    reference: Literal["car", "as_recorded"] = "car"

    @model_validator(mode="after")
    def _check(self) -> SignalContract:
        nyquist = self.sfreq / 2
        low, high = self.band_hz
        if not 0 < low < high < nyquist:
            raise ValueError(
                f"band {self.band_hz} must satisfy 0 < low < high < Nyquist ({nyquist})"
            )
        if any(f <= 0 for f in self.notch_hz):
            raise ValueError(f"notch frequencies must be positive: {self.notch_hz}")
        if self.hop_s > self.window_s:
            raise ValueError(f"hop ({self.hop_s} s) must not exceed the window ({self.window_s} s)")
        for name, seconds in (("window_s", self.window_s), ("hop_s", self.hop_s)):
            samples = seconds * self.sfreq
            if not math.isclose(samples, round(samples), abs_tol=1e-9):
                raise ValueError(
                    f"{name}={seconds} s is not a whole number of samples at {self.sfreq} Hz"
                )
        start, end = self.epoch_s
        if not start < end:
            raise ValueError(f"epoch {self.epoch_s} must have start < end")
        if not math.isclose(end - start, self.window_s):
            # One offline epoch must be exactly one model window, or train and serve differ.
            raise ValueError(f"epoch length {end - start} s must equal window_s {self.window_s} s")
        return self

    @property
    def window_samples(self) -> int:
        """Samples per model window (256 at 128 Hz and 2 s)."""
        return round(self.window_s * self.sfreq)

    @property
    def hop_samples(self) -> int:
        """Samples between consecutive online decisions (16 at 128 Hz and 0.125 s)."""
        return round(self.hop_s * self.sfreq)


class ValidationThresholds(BaseModel):
    """Boundary checks applied when a recording is loaded (architecture §4.5)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # A channel whose (window-detrended) standard deviation is below this is dead or unplugged.
    flat_std_uv: float = Field(default=0.1, gt=0)
    # Any channel's peak-to-peak above this flags the window (movement, electrode pop, saturation).
    ptp_flag_uv: float = Field(default=500.0, gt=0)
    # Plausible range for the typical amplitude (median absolute deviation in the contract band,
    # µV). Data still in volts gives ~1e-5 and doubly scaled data ~1e7, so both unit bugs fail
    # loudly; measuring in-band keeps line noise from faking one.
    amplitude_mad_uv: tuple[float, float] = (0.5, 200.0)
    # Window length for the window-level flags; defaults to the model window.
    window_s: float = Field(default=2.0, gt=0)

    @model_validator(mode="after")
    def _check(self) -> ValidationThresholds:
        low, high = self.amplitude_mad_uv
        if not 0 < low < high:
            raise ValueError(
                f"amplitude_mad_uv {self.amplitude_mad_uv} must satisfy 0 < low < high"
            )
        return self


class DataConfig(BaseModel):
    """Where data lives and which contract applies."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    data_dir: Path = Path("data")
    cache_dir: Path = Path("cache")
    results_dir: Path = Path("results")
    signal: SignalContract = SignalContract()
    validation: ValidationThresholds = ValidationThresholds()

    def resolved(self, base: Path) -> DataConfig:
        """Return a copy with relative directories resolved against ``base``.

        Environment variables ``NEUROS_DATA_DIR``, ``NEUROS_CACHE_DIR`` and
        ``NEUROS_RESULTS_DIR`` override the file (useful on the cluster).
        """

        def pick(env: str, value: Path) -> Path:
            override = os.environ.get(env)
            path = Path(override) if override else value
            return path if path.is_absolute() else (base / path).resolve()

        return self.model_copy(
            update={
                "data_dir": pick("NEUROS_DATA_DIR", self.data_dir),
                "cache_dir": pick("NEUROS_CACHE_DIR", self.cache_dir),
                "results_dir": pick("NEUROS_RESULTS_DIR", self.results_dir),
            }
        )


def load_data_config(path: str | Path | None = None) -> DataConfig:
    """Load a data config. Relative paths resolve against the config file's directory.

    With ``path=None`` the defaults apply and relative paths resolve against the current
    working directory (run the CLI from the repository root).

    Raises:
        NeurOSConfigError: if the file is missing, isn't YAML, or fails validation.
    """
    if path is None:
        return DataConfig().resolved(Path.cwd())
    file = Path(path)
    if not file.is_file():
        raise NeurOSConfigError(f"Config file not found: {file}")
    try:
        raw = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise NeurOSConfigError(f"{file}: invalid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise NeurOSConfigError(f"{file}: expected a mapping at the top level")
    try:
        config = DataConfig.model_validate(raw)
    except ValidationError as exc:
        raise NeurOSConfigError(f"{file}: invalid data config:\n{exc}") from exc
    return config.resolved(file.parent.resolve())
