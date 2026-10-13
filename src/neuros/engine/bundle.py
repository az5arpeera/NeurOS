"""The model bundle (architecture §6.5): everything the engine needs to decode, in one folder.

A bundle holds:

- the trained model (``model.pt``: weights only, loaded with ``weights_only=True``, so a
  bundle can't run code) and its configuration: EEGNet, or (version 2) an A6 pretrained
  encoder fine-tuned with a new head, with the licence a shipped copy must carry;
- the input layout, the scored classes and the input scale (the RMS of the training windows);
- the signal contract it was trained with (band, rate, window, hop, reference);
- **per training person, their mean covariance of the raw layout channels they recorded**.
  The population prior R_pop for *any* headset is built from these when the engine starts:
  for a headset's profile P (the layout channels it has), R_pop = mean over the training
  people who recorded all of P of H·C_P·H, H the common-average projection on P. Covariance is
  linear, so this equals E2's prior (the mean of each person's common-averaged window
  covariances on P) exactly, without storing any windows;
- provenance: datasets, people, git state, experiment hash.

``bundle.json`` records the SHA-256 of ``model.pt``; loading checks it.

Versions: 2 adds ``backbone`` (``null`` for EEGNet), its licence and ``model_sfreq``.
Version 1 bundles (``mvp-v0``) are EEGNet and still load.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from neuros.config.experiment import BackboneConfig, EEGNetConfig
from neuros.config.schemas import SignalContract
from neuros.engine.pipeline import EngineConfig
from neuros.errors import NeurOSConfigError, NeurOSDataError
from neuros.signal.windows import WindowSet

BUNDLE_VERSION = 2
READABLE_VERSIONS = (1, 2)
MODEL_FILE = "model.pt"
META_FILE = "bundle.json"


@dataclass(frozen=True)
class PersonCovariance:
    """One training person's mean covariance of their raw (not re-referenced) layout channels."""

    person: str
    channels: tuple[str, ...]
    matrix: NDArray[np.float64]

    def __post_init__(self) -> None:
        n = len(self.channels)
        if n < 2 or len(set(self.channels)) != n:
            raise NeurOSDataError(f"{self.person}: need >= 2 distinct channels")
        m = np.asarray(self.matrix, dtype=np.float64)
        if m.shape != (n, n) or not np.isfinite(m).all():
            raise NeurOSDataError(f"{self.person}: covariance must be finite [{n}, {n}]")
        if not np.allclose(m, m.T, rtol=1e-6, atol=1e-9):
            raise NeurOSDataError(f"{self.person}: covariance isn't symmetric")
        object.__setattr__(self, "matrix", m)


def person_covariances(windows: WindowSet, layout: Sequence[str]) -> list[PersonCovariance]:
    """Each person's mean raw covariance on the layout channels ``windows`` has (layout order).

    Raises:
        NeurOSDataError: if the windows share fewer than 2 channels with the layout.
    """
    from neuros.align.covariance import window_covariances

    have = [ch for ch in layout if ch in windows.ch_names]
    if len(have) < 2:
        raise NeurOSDataError(f"only {have} of the layout's channels")
    picked = windows.pick_channels(have)
    covs = window_covariances(picked.data)
    people = np.asarray(picked.person_ids)
    return [
        PersonCovariance(person, tuple(have), covs[people == person].mean(axis=0))
        for person in dict.fromkeys(people.tolist())
    ]


class TorchDecoder:
    """The engine's ``Decoder`` for a trained classifier with ``EEGNetClassifier``'s interface
    (EEGNet or an A6 pretrained encoder; CPU, evaluation mode).

    With extra trained classes (D-039), only the scored classes' logits are read, as in
    evaluation: a left/right app is never answered "feet".
    """

    def __init__(self, model: Any, layout: Sequence[str], classes: Sequence[str]) -> None:
        if tuple(model.classes[: len(classes)]) != tuple(classes):
            raise NeurOSConfigError(
                f"the model's first classes {list(model.classes)} aren't {list(classes)}"
            )
        if model.n_channels != len(layout):
            raise NeurOSConfigError(
                f"the model takes {model.n_channels} channels; the layout has {len(layout)}"
            )
        self.model = model
        self.layout = tuple(layout)
        self.classes = tuple(classes)

    def predict_proba(self, x: NDArray[np.float32]) -> NDArray[np.float64]:
        import torch

        from neuros.train.trainer import predict_logits

        logits = predict_logits(self.model, torch.from_numpy(np.ascontiguousarray(x)), 256)
        scored = logits[:, : len(self.classes)].double()
        return np.asarray(torch.softmax(scored, dim=1).numpy(), dtype=np.float64)


@dataclass
class ModelBundle:
    """A trained model and what the engine needs to run it (see the module docs)."""

    model: Any  # EEGNetClassifier or an A6 pretrained classifier (torch imported lazily)
    model_config: EEGNetConfig  # EEGNet's settings (unused when ``backbone`` is set)
    layout: tuple[str, ...]
    classes: tuple[str, ...]
    input_scale: float
    contract: SignalContract
    people_covariances: tuple[PersonCovariance, ...]
    provenance: dict[str, Any] = field(default_factory=dict)
    backbone: BackboneConfig | None = None  # A6: the pretrained encoder; None = EEGNet

    def __post_init__(self) -> None:
        if not (math.isfinite(self.input_scale) and self.input_scale > 0):
            raise NeurOSDataError(f"input_scale must be positive, got {self.input_scale}")
        if not self.people_covariances:
            raise NeurOSDataError("a bundle needs training people's covariances for its prior")
        unknown = {ch for p in self.people_covariances for ch in p.channels} - set(self.layout)
        if unknown:
            raise NeurOSDataError(f"covariances on channels outside the layout: {sorted(unknown)}")

    # ------------------------------------------------------------------ use
    def profile(self, stream_channels: Sequence[str]) -> tuple[str, ...]:
        """The layout channels a headset with ``stream_channels`` has (layout order)."""
        return tuple(ch for ch in self.layout if ch in stream_channels)

    def prior_for(self, profile: Sequence[str]) -> NDArray[np.float64]:
        """R_pop on ``profile``: the mean of H·C_P·H over the training people who recorded all
        of it (H: the common-average projection, when the contract re-references to CAR).

        Raises:
            NeurOSConfigError: if no training person recorded every profile channel.
        """
        chosen = [p for p in self.people_covariances if set(profile) <= set(p.channels)]
        if not chosen:
            raise NeurOSConfigError(
                f"no training person recorded all of {list(profile)}: no prior for this headset"
            )
        n = len(profile)
        h = np.eye(n) - np.full((n, n), 1.0 / n) if self.contract.reference == "car" else np.eye(n)
        mats = []
        for person in chosen:
            idx = [person.channels.index(ch) for ch in profile]
            sub = person.matrix[np.ix_(idx, idx)]
            mats.append(h @ sub @ h)
        return np.asarray(np.mean(mats, axis=0), dtype=np.float64)

    def decoder(self) -> TorchDecoder:
        return TorchDecoder(self.model, self.layout, self.classes)

    def engine_config(
        self, stream_channels: Sequence[str], native_sfreq: float, **settings: Any
    ) -> EngineConfig:
        """An ``EngineConfig`` for a headset: its prior, this bundle's scale and contract."""
        profile = self.profile(stream_channels)
        if len(profile) < 2:
            raise NeurOSConfigError(
                f"the headset shares {list(profile)} with the model's layout; at least 2 needed"
            )
        return EngineConfig(
            stream_channels=tuple(stream_channels),
            native_sfreq=native_sfreq,
            prior=self.prior_for(profile),
            input_scale=self.input_scale,
            contract=self.contract,
            **settings,
        )

    # ------------------------------------------------------------------ disk
    def save(self, folder: Path) -> None:
        """Write ``model.pt`` and ``bundle.json``.

        Raises:
            NeurOSDataError: if ``folder`` already holds a bundle (never overwritten).
        """
        import torch

        meta_path = folder / META_FILE
        if meta_path.exists():
            raise NeurOSDataError(f"{folder} already holds a bundle")
        folder.mkdir(parents=True, exist_ok=True)
        model_path = folder / MODEL_FILE
        from neuros.models.factory import BACKBONE_LICENCES

        torch.save(self.model.state_dict(), model_path)
        meta = {
            "bundle_version": BUNDLE_VERSION,
            "model_sha256": _sha256(model_path),
            "model_config": self.model_config.model_dump(mode="json"),
            "backbone": None if self.backbone is None else self.backbone.model_dump(mode="json"),
            "backbone_licence": (
                None if self.backbone is None else BACKBONE_LICENCES[self.backbone.name]
            ),
            # The rate the model's windows arrive at (a pretrained encoder resamples inside).
            "model_sfreq": float(getattr(self.model, "sfreq", self.contract.sfreq)),
            "n_samples": int(self.model.n_samples),
            "model_classes": list(self.model.classes),
            "layout": list(self.layout),
            "classes": list(self.classes),
            "input_scale": self.input_scale,
            "contract": self.contract.model_dump(mode="json"),
            "people_covariances": [
                {"person": p.person, "channels": list(p.channels), "matrix": p.matrix.tolist()}
                for p in self.people_covariances
            ],
            "provenance": self.provenance,
        }
        meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8", newline="\n")

    @classmethod
    def load(cls, folder: Path) -> ModelBundle:
        """Read a bundle, checking its version and the model file's SHA-256.

        Raises:
            NeurOSDataError: on a missing file, another version, or a checksum mismatch.
        """
        import torch

        from neuros.models.factory import build_model

        meta_path = folder / META_FILE
        model_path = folder / MODEL_FILE
        if not meta_path.is_file() or not model_path.is_file():
            raise NeurOSDataError(f"{folder}: no bundle ({META_FILE} and {MODEL_FILE} needed)")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("bundle_version") not in READABLE_VERSIONS:
            raise NeurOSDataError(
                f"{folder}: bundle version {meta.get('bundle_version')}, this code reads "
                f"{list(READABLE_VERSIONS)}"
            )
        if _sha256(model_path) != meta["model_sha256"]:
            raise NeurOSDataError(f"{model_path}: SHA-256 doesn't match {META_FILE}")
        config = EEGNetConfig.model_validate(meta["model_config"])
        raw_backbone = meta.get("backbone")  # absent in version 1: EEGNet
        backbone = None if raw_backbone is None else BackboneConfig.model_validate(raw_backbone)
        contract = SignalContract.model_validate(meta["contract"])
        layout = tuple(meta["layout"])
        # The bundle's own state replaces every weight, so the published file isn't needed.
        model = build_model(
            config,
            backbone,
            layout,
            int(meta["n_samples"]),
            meta["model_classes"],
            float(meta.get("model_sfreq", contract.sfreq)),
            load_weights=False,
        )
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        try:
            model.load_state_dict(state)  # strict: every name and shape must match
        except RuntimeError as error:
            raise NeurOSDataError(f"{model_path} doesn't fit the bundled model: {error}") from error
        model.eval()
        return cls(
            model=model,
            model_config=config,
            layout=layout,
            classes=tuple(meta["classes"]),
            input_scale=float(meta["input_scale"]),
            contract=contract,
            people_covariances=tuple(
                PersonCovariance(p["person"], tuple(p["channels"]), np.asarray(p["matrix"]))
                for p in meta["people_covariances"]
            ),
            provenance=dict(meta.get("provenance", {})),
            backbone=backbone,
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_people(people: Sequence[str]) -> Mapping[str, int]:
    """People per dataset, for provenance."""
    counts: dict[str, int] = {}
    for person in people:
        dataset = person.split(":", 1)[0]
        counts[dataset] = counts.get(dataset, 0) + 1
    return counts


def train_bundle(
    task_windows: Mapping[str, WindowSet],
    layout: Sequence[str],
    *,
    classes: Sequence[str],
    model_config: EEGNetConfig,
    train_config: Any,
    contract: SignalContract,
    reference: Any,
    alignment: Any,
    unit: Any,
    derived: Mapping[str, Mapping[str, Sequence[str]]] | None,
    validation_fraction: float,
    seed: int,
    device: str,
    provenance: Mapping[str, Any],
    progress: Any = None,
    backbone: BackboneConfig | None = None,
) -> ModelBundle:
    """Train the deployable model on every development person of ``task_windows`` (E1's
    recipe and placement), with ``validation_fraction`` of them for early stopping, and bundle it.

    There are no test people: the numbers that describe this model's accuracy are E1/E2's
    cross-validated ones (same recipe), and the locked test set at the freeze.

    ``backbone`` (A6) fine-tunes that pretrained encoder, starting from its published weights,
    instead of training EEGNet from scratch. The model is built exactly as the pooled
    evaluation builds it (``models.factory.build_model``).

    Raises:
        NeurOSDataError: on windows that can't be placed, or labels outside ``classes``.
    """
    import torch

    from neuros.eval.pooled import _gather, _rms, assemble, split_validation
    from neuros.models.factory import build_model
    from neuros.train.data import TensorData
    from neuros.train.device import configure_determinism, resolve_device
    from neuros.train.trainer import train_classifier

    pooled, _ = assemble(
        task_windows, layout, reference=reference, alignment=alignment, unit=unit, derived=derived
    )
    people = list(dict.fromkeys(pooled.people.tolist()))
    train_people, validation_people = split_validation(
        people, validation_fraction, seed=seed, tag="bundle"
    )
    index = {name: i for i, name in enumerate(classes)}
    unknown = sorted(set(pooled.labels.tolist()) - set(index))
    if unknown:
        raise NeurOSDataError(f"windows carry labels {unknown} outside {list(classes)}")
    rows_train = pooled.rows_of(train_people)
    scale = _rms(pooled.x, rows_train)

    def tensors(rows: NDArray[np.int64]) -> TensorData:
        x = _gather(pooled.x, rows)
        x /= np.float32(scale)
        return TensorData(
            x=torch.from_numpy(x),
            y=torch.tensor([index[label] for label in pooled.labels[rows].tolist()]),
            people=tuple(pooled.people[rows].tolist()),
            classes=tuple(classes),
        )

    target = resolve_device(device)
    configure_determinism(target, True)
    torch.manual_seed(seed)  # weight initialisation (EEGNet, or a pretrained encoder's new head)
    model = build_model(model_config, backbone, layout, pooled.n_samples, classes, pooled.sfreq)
    result = train_classifier(
        model,
        tensors(rows_train),
        tensors(pooled.rows_of(validation_people)) if validation_people else None,
        train_config,
        seed=seed,
        progress=progress,
        device=target,
    )
    model.to("cpu")
    model.eval()
    covariances = tuple(
        pc for windows in task_windows.values() for pc in person_covariances(windows, layout)
    )
    record = result.to_dict()
    record.pop("history", None)  # epoch by epoch is in the run log; the bundle keeps the outcome
    started_from: dict[str, Any] = {}
    if backbone is not None and backbone.weights is not None:
        # Which published weights the fine-tuning started from (provenance and licence).
        started_from = {"pretrained_weights_sha256": _sha256(Path(backbone.weights))}
    return ModelBundle(
        model=model,
        model_config=model_config,
        layout=tuple(layout),
        classes=tuple(classes),
        input_scale=scale,
        contract=contract,
        people_covariances=covariances,
        provenance=dict(provenance)
        | {
            "people_per_dataset": dict(summarize_people(people)),
            "train_people": list(train_people),
            "validation_people": list(validation_people),
            "seed": seed,
            "training": record,
        }
        | started_from,
        backbone=backbone,
    )
