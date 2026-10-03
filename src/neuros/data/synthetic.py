"""Synthetic motor-imagery EEG with known labels (architecture §9, "test data").

Fast, deterministic data for unit tests and, later, end-to-end engine tests, with the one
property that matters for motor imagery built in: **event-related desynchronisation (ERD)**.
Imagining a hand movement suppresses the mu rhythm (8-12 Hz) over the *opposite* hemisphere:

- ``left_hand``  → mu drops around C4 (right motor cortex);
- ``right_hand`` → mu drops around C3 (left motor cortex);
- ``both_hands`` → both; ``feet`` → around Cz; ``rest`` → no change.

Signal model per channel: pink (1/f) background noise + three mu sources (left, midline and
right motor cortex) spread over the scalp with Gaussian weights on the 10-10 grid. It's a
caricature of real EEG, good enough to test plumbing, alignment and decoders; it's never used
to support a claim.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from neuros.data.channels import lateral_position, row_of, spatial_sort_key
from neuros.data.recording import EventTable, Recording
from neuros.errors import NeurOSConfigError

# Source positions on the (anterior-posterior level, left-right) grid of channels.spatial_sort_key.
_SOURCES: dict[str, tuple[float, float]] = {
    "left_motor": (5.0, -2.0),  # C3
    "midline_motor": (5.0, 0.0),  # Cz
    "right_motor": (5.0, 2.0),  # C4
}
# Which sources each label desynchronises.
_ERD_SOURCES: dict[str, tuple[str, ...]] = {
    "left_hand": ("right_motor",),
    "right_hand": ("left_motor",),
    "both_hands": ("left_motor", "right_motor"),
    "feet": ("midline_motor",),
    "rest": (),
    "tongue": (),
}
_SOURCE_SPREAD = 1.5  # Gaussian width in grid units: C3's mu also reaches C1, C5, FC3, CP3
_ERD_RAMP_S = 0.3  # ERD develops and recovers over ~300 ms


@dataclass(frozen=True)
class SyntheticConfig:
    """Parameters of a synthetic recording (defaults: 40 trials, 2 classes, 250 Hz)."""

    ch_names: tuple[str, ...] = (
        "FC3",
        "FC4",
        "C5",
        "C3",
        "C1",
        "Cz",
        "C2",
        "C4",
        "C6",
        "CP3",
        "CP4",
        "Pz",
    )
    sfreq: float = 250.0
    n_trials: int = 40
    labels: tuple[str, ...] = ("left_hand", "right_hand")
    pre_cue_s: float = 2.0
    task_s: float = 4.0
    iti_s: tuple[float, float] = (1.5, 2.5)
    background_uv: float = 10.0
    mu_uv: float = 8.0
    mu_hz: float = 10.0
    erd_depth: float = 0.6
    seed: int = 0

    def __post_init__(self) -> None:
        unknown = [label for label in self.labels if label not in _ERD_SOURCES]
        if unknown:
            raise NeurOSConfigError(f"Synthetic data can't simulate labels {unknown}")
        if not 0 <= self.erd_depth < 1:
            raise NeurOSConfigError("erd_depth must be in [0, 1)")
        if self.n_trials < 1 or self.sfreq <= 2 * (self.mu_hz + 2):
            raise NeurOSConfigError("need >= 1 trial and a sampling rate well above the mu band")
        low, high = self.iti_s
        if not 0 <= low <= high:
            raise NeurOSConfigError("iti_s must satisfy 0 <= low <= high")


def _pink_noise(rng: np.random.Generator, n_channels: int, n_samples: int) -> NDArray[np.float64]:
    """Unit-variance 1/f noise per channel (spectrum shaped in the Fourier domain)."""
    white = rng.standard_normal((n_channels, n_samples))
    spectrum = np.fft.rfft(white, axis=-1)
    freqs = np.arange(spectrum.shape[-1], dtype=np.float64)
    freqs[0] = 1.0  # avoid dividing by zero at DC
    spectrum /= np.sqrt(freqs)
    pink = np.fft.irfft(spectrum, n=n_samples, axis=-1)
    pink -= pink.mean(axis=-1, keepdims=True)
    # Explicit float64 array: numpy 2.4's stubs type this division as Any (CI on Python 3.11).
    return np.asarray(pink / pink.std(axis=-1, keepdims=True), dtype=np.float64)


def _narrowband(
    rng: np.random.Generator, n_samples: int, sfreq: float, centre: float, width: float = 2.0
) -> NDArray[np.float64]:
    """Unit-variance noise band-limited to ``centre ± width/2`` Hz: a waxing-waning rhythm."""
    spectrum = np.fft.rfft(rng.standard_normal(n_samples))
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sfreq)
    spectrum[np.abs(freqs - centre) > width / 2] = 0.0
    band = np.fft.irfft(spectrum, n=n_samples)
    std = band.std()
    return band / std if std > 0 else band


def _erd_envelope(
    n_samples: int, sfreq: float, starts: list[int], stop_after: int
) -> NDArray[np.float64]:
    """1 → 0 → 1 envelope marking imagery periods, with smooth ramps."""
    envelope = np.zeros(n_samples)
    ramp = max(1, round(_ERD_RAMP_S * sfreq))
    shape = np.concatenate(
        [np.linspace(0, 1, ramp), np.ones(max(stop_after - 2 * ramp, 0)), np.linspace(1, 0, ramp)]
    )
    for start in starts:
        stop = min(start + len(shape), n_samples)
        envelope[start:stop] = np.maximum(envelope[start:stop], shape[: stop - start])
    return envelope


def make_recording(
    config: SyntheticConfig | None = None,
    *,
    dataset: str = "Synthetic",
    subject: int = 1,
    session: str = "0",
    run: str = "0",
) -> Recording:
    """Generate a deterministic synthetic recording (same config and seed → same data)."""
    cfg = config or SyntheticConfig()
    rng = np.random.default_rng(cfg.seed)
    sfreq = cfg.sfreq

    # Balanced, shuffled label sequence.
    reps = -(-cfg.n_trials // len(cfg.labels))
    sequence = list(
        np.array(cfg.labels * reps)[rng.permutation(len(cfg.labels) * reps)][: cfg.n_trials]
    )

    # Trial timing: [pre-cue fixation | task | inter-trial interval].
    cues: list[int] = []
    cursor = 0.0
    for _ in sequence:
        cursor += cfg.pre_cue_s
        cues.append(round(cursor * sfreq))
        cursor += cfg.task_s + rng.uniform(*cfg.iti_s)
    n_samples = round((cursor + 1.0) * sfreq)

    n_channels = len(cfg.ch_names)
    signal = cfg.background_uv * _pink_noise(rng, n_channels, n_samples)

    # Each source: a mu rhythm, modulated by the ERD envelope of the labels that suppress it.
    task_samples = round(cfg.task_s * sfreq)
    for source, (level, lateral) in _SOURCES.items():
        rhythm = cfg.mu_uv * _narrowband(rng, n_samples, sfreq, cfg.mu_hz)
        starts = [
            cue for cue, label in zip(cues, sequence, strict=True) if source in _ERD_SOURCES[label]
        ]
        rhythm *= 1.0 - cfg.erd_depth * _erd_envelope(n_samples, sfreq, starts, task_samples)
        for ch_index, name in enumerate(cfg.ch_names):
            ch_level, ch_lateral, _ = spatial_sort_key(name)
            if row_of(name) is None or lateral_position(name) is None:
                continue
            distance_sq = (ch_level - level) ** 2 + (ch_lateral - lateral) ** 2
            signal[ch_index] += np.exp(-distance_sq / (2 * _SOURCE_SPREAD**2)) * rhythm

    events = EventTable(
        onsets=np.array(cues, dtype=np.int64), labels=tuple(str(x) for x in sequence)
    )
    return Recording(
        data=signal.astype(np.float32),
        sfreq=sfreq,
        ch_names=cfg.ch_names,
        events=events,
        dataset=dataset,
        subject=subject,
        session=session,
        run=run,
        provenance={"loader": "synthetic", "seed": str(cfg.seed)},
    )
