"""Tiny synthetic window sets and model settings for the deep-model tests.

Each person gets a 10 Hz rhythm on a class-specific channel (left_hand → channel 0,
right_hand → channel 1) plus white noise, all multiplied by that person's own per-channel
gains and overall scale: the cross-person shift that Euclidean Alignment removes. Windows are
short (64 samples) and the network tiny, so a whole training run takes well under a second.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from neuros.config.experiment import EEGNetConfig, TrainConfig
from neuros.signal.windows import WindowSet

CHANNELS = ("C3", "C4", "Cz", "Pz")
SFREQ = 128.0
N_SAMPLES = 64

# EEGNet shrunk to ~300 parameters; same layer types and order as the real one.
TINY_MODEL = EEGNetConfig(
    f1=4,
    depth=1,
    f2=4,
    kernel_samples=16,
    separable_kernel_samples=4,
    pool1=4,
    pool2=4,
    embedding_dim=8,
    dropout=0.0,
)

# A short, fast recipe for tests (the real one is 300 epochs at lr 1e-3).
FAST_TRAIN = TrainConfig(
    max_epochs=12,
    people_per_batch=2,
    windows_per_person=8,
    lr=1e-2,
    weight_decay=0.0,
    swa_fraction=0.25,
    log_every=0,
)


def people_windows(
    scales: Sequence[float] = (0.2, 1.0, 5.0),
    *,
    per_class: int = 12,
    sessions: Sequence[str] = ("0", "1"),
    amplitude: float = 3.0,
    seed: int = 0,
) -> WindowSet:
    """Synthetic people (subjects 1, 2, ...) with one overall ``scale`` each.

    Every person has ``per_class`` windows per class in each session, in the order
    person → session → class.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(N_SAMPLES) / SFREQ
    data, labels, subjects, session_of = [], [], [], []
    for subject, scale in enumerate(scales, start=1):
        gains = np.exp(rng.uniform(np.log(0.5), np.log(2.0), len(CHANNELS))) * scale
        for session in sessions:
            for label, channel in (("left_hand", 0), ("right_hand", 1)):
                for _ in range(per_class):
                    window = rng.normal(0.0, 1.0, (len(CHANNELS), N_SAMPLES))
                    phase = rng.uniform(0, 2 * np.pi)
                    window[channel] += amplitude * np.sin(2 * np.pi * 10.0 * t + phase)
                    data.append(window * gains[:, None])
                    labels.append(label)
                    subjects.append(subject)
                    session_of.append(session)
    n = len(data)
    return WindowSet(
        data=np.asarray(data, dtype=np.float32),
        sfreq=SFREQ,
        ch_names=CHANNELS,
        labels=tuple(labels),
        datasets=("Synthetic",) * n,
        subjects=np.asarray(subjects, dtype=np.int64),
        sessions=tuple(session_of),
        runs=("0",) * n,
        starts=np.zeros(n, dtype=np.int64),
        flags=np.zeros(n, dtype=np.uint8),
    )
