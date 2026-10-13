"""Build a model from its configuration: EEGNet from scratch, or an A6 pretrained encoder.

One function serves the pooled evaluation (``eval.pooled.build_classifier``), the bundle
trainer and the bundle loader (``engine.bundle``). So the model a bundle ships is built exactly
as the experiment that scored it built it, and a bundle can be reloaded without the published
weights file: its own ``model.pt`` holds every fine-tuned weight.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from neuros.config.experiment import BackboneConfig, EEGNetConfig
from neuros.errors import NeurOSConfigError

# What a shipped copy of each pretrained encoder must carry (read 12 Oct; not legal advice).
# Bundles record this so the SDK packaging step can't lose it.
BACKBONE_LICENCES: dict[str, str] = {
    "mirepnet": (
        "MIT (code github.com/staraink/MIRepNet at b35d113; weights huggingface.co/starself/"
        "MIRepNet at 9bac043): ship the MIT licence text and copyright notice with the model"
    ),
    "reve-base": (
        "REVE Responsible Use License v1.0 (huggingface.co/brain-bzh/reve-base at dc2a075): "
        "ship the licence, its notices and a 'modified from REVE' label; no identifying people "
        "from EEG"
    ),
}

# REVE's head dropout when the configuration leaves it unset (the authors' recipe: 0.3-0.5).
REVE_DEFAULT_DROPOUT = 0.3


def build_model(
    model_config: EEGNetConfig,
    backbone: BackboneConfig | None,
    layout: Sequence[str],
    n_samples: int,
    classes: Sequence[str],
    sfreq: float,
    *,
    load_weights: bool = True,
) -> Any:
    """The classifier for ``layout`` windows of ``n_samples`` at ``sfreq`` Hz.

    ``backbone`` ``None`` gives EEGNet (``model_config``); otherwise the named pretrained
    encoder with a new head for ``classes``. ``load_weights`` loads the encoder's published
    weights (strictly) when the configuration names a file; a bundle loader passes ``False``
    because the bundle's own state replaces every weight.

    Call it after seeding: new heads (and EEGNet) draw their initial weights from the seed.

    Raises:
        NeurOSConfigError: on an empty layout or class list, a non-positive size or rate, a
            missing weights file, or a layout channel the encoder can't place.
        NeurOSDataError: if the published weights don't match the implementation.
    """
    if not layout or not classes:
        raise NeurOSConfigError("a model needs at least one layout channel and one class")
    if n_samples <= 0 or not sfreq > 0:
        raise NeurOSConfigError(f"bad window: {n_samples} samples at {sfreq} Hz")
    if backbone is None:
        from neuros.models.eegnet import EEGNetClassifier

        return EEGNetClassifier(len(layout), n_samples, tuple(classes), model_config)

    from neuros.models.pretrained import MIRepNetClassifier, ReveClassifier

    model: ReveClassifier | MIRepNetClassifier
    if backbone.name == "reve-base":
        dropout = REVE_DEFAULT_DROPOUT if backbone.dropout is None else backbone.dropout
        model = ReveClassifier(layout, n_samples, classes, sfreq=sfreq, dropout=dropout)
    elif backbone.name == "mirepnet":
        model = MIRepNetClassifier(
            layout, n_samples, classes, sfreq=sfreq, dropout=backbone.dropout
        )
    else:  # pragma: no cover - BackboneConfig's Literal rules this out
        raise NeurOSConfigError(f"unknown backbone {backbone.name!r}")
    if load_weights and backbone.weights is not None:  # None: the random-weights control
        path = Path(backbone.weights)
        if not path.is_file():
            raise NeurOSConfigError(f"{backbone.name}'s published weights aren't at {path}")
        model.load_pretrained(path)
    return model
