"""Experiment configs: the deep model, its training recipe, leave-one-person-out runs (E0), and
pooled runs across datasets (E1).

One YAML per experiment ID (architecture §9): ``configs/e0.yaml`` is E0. The schemas live here,
free of PyTorch, so a config is validated (and hashed into provenance, principle 5) before any
data is loaded or any training starts: a typo fails in milliseconds, not after an hour.

The defaults are the architecture's recipe (§6.1 encoder, §6.4 training). E0's file pins the
values of the paper it reproduces (Sagayaraju 2026: AdamW, lr 1e-3, weight decay 0.05,
dropout 0.5, 300 epochs, SWA), and every deviation from the defaults is visible in that file.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from neuros.config.schemas import SignalContract
from neuros.errors import NeurOSConfigError

Alignment = Literal["none", "ea"]
AlignmentUnit = Literal["person", "session"]
# "motor": the motor strip (FC, FCC, C, CCP, CP rows; neuros.data.montage.SCALP_REGIONS),
# which D-031 makes the intent head's input region.
ChannelSet = Literal["all", "c-core", "motor"]
Reference = Literal["as_recorded", "car"]


class EEGNetConfig(BaseModel):
    """EEGNet encoder + linear head (architecture §6.1; Lawhern et al. 2018)."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    # F1 temporal filters, D spatial filters per temporal filter, F2 separable filters.
    # §6.1 sets 16/2/32: twice the width of the classic EEGNet-8,2, still ~20k parameters.
    f1: int = Field(default=16, ge=1, le=256)
    depth: int = Field(default=2, ge=1, le=16)
    f2: int = Field(default=32, ge=1, le=512)
    # Temporal kernel: 64 samples = 0.5 s at 128 Hz, long enough to resolve rhythms to ~2 Hz.
    kernel_samples: int = Field(default=64, ge=2, le=1024)
    # The separable convolution's temporal kernel (16 samples at 32 Hz after the first pool).
    separable_kernel_samples: int = Field(default=16, ge=1, le=256)
    pool1: int = Field(default=4, ge=1, le=32)
    pool2: int = Field(default=8, ge=1, le=64)
    # §6.1 allows 0.25-0.5. EEGNet's authors suggest 0.25 across people; E0's paper used 0.5.
    dropout: float = Field(default=0.5, ge=0.0, lt=1.0)
    # The embedding z that every head reads (§6.2).
    embedding_dim: int = Field(default=64, ge=2, le=1024)
    # EEGNet's max-norm constraints: each spatial filter's weight vector has L2 norm <= 1, and
    # each class's head weights <= 0.25. ``null`` disables one.
    spatial_max_norm: float | None = Field(default=1.0, gt=0)
    head_max_norm: float | None = Field(default=0.25, gt=0)


class BackboneConfig(BaseModel):
    """A6: start from a published pretrained encoder instead of training EEGNet from scratch
    (``neuros.models.pretrained``; ``knowledge-base/plan/a6-pretrained-backbone.md``).

    The encoder is fine-tuned with the rest of the recipe (``train``); the trainer gives it its
    own learning-rate scale and can hold it frozen for the first epochs (``TrainConfig``).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    name: Literal["reve-base", "mirepnet"]
    # The published weights (downloaded with the founders' permission; never committed).
    # ``null`` = the same architecture with random weights: the control that shows whether the
    # pretraining (rather than the architecture) does the work.
    weights: str | None = Field(default=None, min_length=1)
    # Dropout before the new linear head (REVE's recipe uses 0.3-0.5). For MIRepNet, ``null``
    # keeps its published 0.5 throughout the encoder.
    dropout: float | None = Field(default=None, ge=0.0, lt=1.0)


class TrainConfig(BaseModel):
    """The training recipe (architecture §6.4). The seed is per run, not part of the recipe."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    max_epochs: int = Field(default=300, ge=1, le=10_000)
    # Batches are K people × M windows, class-balanced within each person (§6.4). K is capped
    # at the number of training people.
    people_per_batch: int = Field(default=8, ge=1, le=1024)
    windows_per_person: int = Field(default=8, ge=1, le=4096)
    # AdamW (decoupled weight decay), learning rate on a cosine from ``lr`` to ``min_lr``.
    lr: float = Field(default=1e-3, gt=0, le=1.0)
    min_lr: float = Field(default=0.0, ge=0)
    weight_decay: float = Field(default=0.05, ge=0, le=1.0)
    # §6.3 recommends 0.1 for the pooled recipe; E0 uses 0 like the paper it reproduces.
    label_smoothing: float = Field(default=0.0, ge=0, lt=1.0)
    # SWA: the final weights are the average over the last ``swa_fraction`` of the epochs of
    # the run (see neuros.train.trainer for how this combines with early stopping). 0 = off.
    swa_fraction: float = Field(default=0.25, ge=0, le=1.0)
    # Early stopping on validation people (never test people): the run ends at the epoch with
    # the lowest validation loss; training stops after ``patience`` epochs without an
    # improvement larger than ``min_delta``. ``null`` = train all ``max_epochs``.
    patience: int | None = Field(default=None, ge=1)
    min_delta: float = Field(default=1e-4, ge=0)
    # Windows per forward pass when evaluating (memory only; doesn't change any number).
    eval_batch_size: int = Field(default=512, ge=1)
    # A6, pretrained backbones only (models with ``parameter_groups``; EEGNet ignores both):
    # the backbone's learning rate is ``lr`` times this, and it stays frozen for the first
    # ``frozen_backbone_epochs`` epochs while the new head learns (a linear probe first, as
    # REVE's authors fine-tune).
    backbone_lr_scale: float = Field(default=1.0, gt=0, le=1.0)
    frozen_backbone_epochs: int = Field(default=0, ge=0)
    # Print one progress line every this many epochs (0 = only the summary).
    log_every: int = Field(default=25, ge=0)

    @model_validator(mode="after")
    def _check(self) -> TrainConfig:
        if self.min_lr > self.lr:
            raise ValueError(f"min_lr ({self.min_lr}) must not exceed lr ({self.lr})")
        return self


class LosoExperimentConfig(BaseModel):
    """A leave-one-person-out experiment on one dataset (E0 and its variants)."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    # Results go to ``results/<name>/``; lowercase so file names are portable.
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$", max_length=64)
    description: str = ""
    dataset: str = Field(min_length=1)
    # Class order = logit order. Each must be a vocabulary label (checked when data loads).
    labels: tuple[str, ...] = ("left_hand", "right_hand")
    channels: ChannelSet = "all"
    reference: Reference = "as_recorded"
    # Each alignment is a separate LOSO run on the same folds, so they can be compared
    # person by person (paired Wilcoxon).
    alignments: tuple[Alignment, ...] = ("none", "ea")
    # One EA reference per recording session (the paper's protocol, and the offline twin of
    # the online engine, which starts a new reference each time the headset goes on), or one
    # per person with all sessions pooled.
    alignment_unit: AlignmentUnit = "session"
    # One LOSO run per seed (weight initialisation, dropout and batch order). The seed spread
    # is reported next to the person spread.
    seeds: tuple[int, ...] = (20261005,)
    # Inner validation people per fold, drawn from the training people only (for early
    # stopping and monitoring). 0 = train on every non-test person, like the paper.
    validation_people: int = Field(default=0, ge=0)
    # Overrides of the data config's signal contract for this experiment only (e.g. the
    # literature's 4-s window). Applied with ``apply_signal_overrides``, which re-validates the
    # whole contract; the overridden contract's hash is in every result's provenance.
    signal: dict[str, Any] = Field(default_factory=dict)
    model: EEGNetConfig = EEGNetConfig()
    train: TrainConfig = TrainConfig()

    @model_validator(mode="after")
    def _check(self) -> LosoExperimentConfig:
        for field, values in (
            ("labels", self.labels),
            ("alignments", self.alignments),
            ("seeds", self.seeds),
        ):
            if not values:
                raise ValueError(f"{field} must not be empty")
            if len(set(values)) != len(values):
                raise ValueError(f"{field} has duplicates: {list(values)}")
        if len(self.labels) < 2:
            raise ValueError(f"a classifier needs at least 2 labels, got {list(self.labels)}")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError(f"seeds must be >= 0, got {list(self.seeds)}")
        return self


def apply_signal_overrides(base: SignalContract, overrides: dict[str, Any]) -> SignalContract:
    """The contract ``base`` with ``overrides`` applied, validated as a whole.

    Raises:
        NeurOSConfigError: on an unknown field or an inconsistent result (e.g. an epoch whose
            length differs from the window).
    """
    if not overrides:
        return base
    try:
        return SignalContract.model_validate({**base.model_dump(), **overrides})
    except ValidationError as exc:
        raise NeurOSConfigError(f"invalid signal overrides {overrides}:\n{exc}") from exc


def _read_yaml_mapping(path: str | Path) -> dict[str, Any]:
    """The top-level mapping of an experiment YAML.

    Raises:
        NeurOSConfigError: if the file is missing, isn't YAML, or isn't a mapping.
    """
    file = Path(path)
    if not file.is_file():
        raise NeurOSConfigError(f"Experiment config not found: {file}")
    try:
        raw = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise NeurOSConfigError(f"{file}: invalid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise NeurOSConfigError(f"{file}: expected a mapping at the top level")
    return raw


def load_experiment_config(path: str | Path) -> LosoExperimentConfig:
    """Load and validate an experiment YAML.

    Raises:
        NeurOSConfigError: if the file is missing, isn't YAML, or fails validation.
    """
    raw = _read_yaml_mapping(path)
    try:
        return LosoExperimentConfig.model_validate(raw)
    except ValidationError as exc:
        raise NeurOSConfigError(f"{path}: invalid experiment config:\n{exc}") from exc


# E1's model conditions (neuros.eval.pooled).
PooledCondition = Literal["pooled", "single", "lodo", "cue_only"]
POOLED_CONDITIONS: tuple[PooledCondition, ...] = ("pooled", "single", "lodo", "cue_only")


class WarmupConfig(BaseModel):
    """E2, the warm-up curve (architecture §5 online, §8.5): how many seconds of a new user's
    unlabeled EEG the online alignment needs before the pooled model reaches its full-session
    accuracy (``neuros.eval.warmup``)."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    # Datasets whose test people get the curve (cue-free ones: the headline is cue-controlled).
    datasets: tuple[str, ...] = Field(min_length=1)
    # Seconds of the session's unlabeled data the reference may use; 0 = population prior only.
    budgets_s: tuple[float, ...] = (0.0, 5.0, 10.0, 20.0, 30.0, 45.0, 60.0, 120.0)
    # n₀: the population prior weighs as much as this many seconds of the user's data (§5).
    prior_s: float = Field(default=10.0, gt=0.0, le=600.0)
    # The reference's windows: the decoder's window length, sliding by the contract's hop.
    stream_window_s: float = Field(default=2.0, gt=0.0, le=10.0)
    stream_hop_s: float = Field(default=0.125, gt=0.0, le=10.0)
    # H1: seed from the relaxed pre-cue fixation only, as well as from the whole stream.
    rest_seeded: bool = True
    # E5 (virtual headsets): name -> model channels a headset class has. Each set is scored on
    # every warm-up person whose dataset has all its channels: the windows cut to those
    # channels (common average over them, zeros elsewhere, as the live engine places a
    # headset), aligned with the stream reference after ``channel_set_budget_s``.
    channel_sets: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    channel_set_budget_s: float = 120.0

    @model_validator(mode="after")
    def _check(self) -> WarmupConfig:
        if len(set(self.datasets)) != len(self.datasets):
            raise ValueError(f"warmup datasets have duplicates: {list(self.datasets)}")
        budgets = list(self.budgets_s)
        if not budgets or budgets != sorted(set(budgets)) or budgets[0] < 0:
            raise ValueError(f"budgets_s must be distinct, ascending and >= 0, got {budgets}")
        if budgets[-1] > 1800:
            raise ValueError(f"budgets_s above 1800 s leave nothing to score, got {budgets[-1]}")
        if self.stream_hop_s > self.stream_window_s:
            raise ValueError("stream_hop_s must not exceed stream_window_s (gaps in the stream)")
        for name, channels in self.channel_sets.items():
            if not re.fullmatch(r"[a-z0-9_]+", name):
                raise ValueError(f"channel set names are lower-case words, got {name!r}")
            if len(channels) < 2 or len(set(channels)) != len(channels):
                raise ValueError(f"channel set {name!r} needs >= 2 distinct channels")
        if self.channel_sets and self.channel_set_budget_s not in budgets:
            raise ValueError(
                f"channel_set_budget_s {self.channel_set_budget_s} must be one of {budgets} "
                "(its research-cap comparison is that budget's stream set)"
            )
        return self


class TrainingWindowsConfig(BaseModel):
    """D-039, "use more of what we have": train on more windows than the test uses.

    Validation and test windows never change: the D-030 window, scored on ``labels`` only.
    Without this section (or with both options off) training is exactly E1's.

    - **Crops** (``crop_stride_s``): extra training windows every ``crop_stride_s`` seconds,
      from the test window to the end of the task, never earlier (towards the cue). Only
      datasets whose class leaves the screen early gain any (Yang2025, BNCI2014-001); see
      ``neuros.data.registry.training_crop_starts``. An epoch still counts one window per
      trial, so crops vary what each step sees without adding steps.
    - **More classes** (``extra_labels``): the head also learns these classes, from the
      training datasets that have them (BNCI2014-001's feet and tongue, PhysionetMI's and
      Stieger2021's both hands and rest). Each window's softmax runs over its own dataset's
      classes only, and validation and test still score ``labels`` only.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    # Seconds between training crops; null = one window per trial (the test window).
    crop_stride_s: float | None = Field(default=None, gt=0.0, le=10.0)
    # Vocabulary labels the head learns besides the experiment's labels (checked at load).
    extra_labels: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> TrainingWindowsConfig:
        if len(set(self.extra_labels)) != len(self.extra_labels):
            raise ValueError(f"extra_labels has duplicates: {list(self.extra_labels)}")
        if self.crop_stride_s is None and not self.extra_labels:
            raise ValueError(
                "training_windows changes nothing: set crop_stride_s and/or extra_labels, or "
                "leave the section out"
            )
        return self


class FinetuneConfig(BaseModel):
    """Q30, "calibration saved": each test person's pooled model fine-tuned on their first few
    labelled trials, scored on the calibration curve's test trials (``neuros.eval.finetune``).

    The recipe is fixed here **before** the run and never tuned on test people (10 Oct): every
    layer, AdamW with the training weight decay and label smoothing, a cosine learning rate
    from ``lr`` to ``min_lr`` over ``epochs`` full batches (all the person's calibration
    windows, balanced by class), no early stopping (there is no one to validate on), no SWA,
    batch-norm statistics recomputed on the calibration windows (the trainer always does).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    # Datasets whose test people are fine-tuned (each must also be a warm-up dataset: the
    # stream settings and the population prior are E2's).
    datasets: tuple[str, ...] = Field(min_length=1)
    # Labelled trials per class to calibrate on (the curve's x-axis).
    sizes: tuple[int, ...] = (5, 10, 20)
    # Calibrate and test within each session (default) or within each person (sessions in
    # order; Shin2017A's sessions hold 10 trials per class).
    blocks: dict[str, Literal["session", "person"]] = Field(default_factory=dict)
    # Calibration time starts this long before the first calibration window (the trial's cue
    # and fixation), so the reference only uses data recorded during the calibration.
    lead_s: float = Field(default=4.0, ge=0.0, le=30.0)
    epochs: int = Field(default=50, ge=1, le=1000)
    lr: float = Field(default=5e-4, gt=0.0, le=1.0)
    min_lr: float = Field(default=5e-5, ge=0.0)

    @model_validator(mode="after")
    def _check(self) -> FinetuneConfig:
        if len(set(self.datasets)) != len(self.datasets):
            raise ValueError(f"finetune datasets have duplicates: {list(self.datasets)}")
        sizes = list(self.sizes)
        if not sizes or sizes != sorted(set(sizes)) or sizes[0] < 1:
            raise ValueError(f"sizes must be distinct, ascending and >= 1, got {sizes}")
        unknown = sorted(set(self.blocks) - set(self.datasets))
        if unknown:
            raise ValueError(f"blocks name datasets that aren't fine-tuned: {unknown}")
        if self.min_lr > self.lr:
            raise ValueError(f"min_lr ({self.min_lr}) must not exceed lr ({self.lr})")
        return self


class PooledExperimentConfig(BaseModel):
    """E1: pooled training across datasets, against single-dataset and leave-one-dataset-out
    models, with the cue-only control (``neuros.eval.pooled``)."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$", max_length=64)
    description: str = ""
    # Datasets whose development people train (and are tested). Each must be allowed to train
    # a model we ship (registry train_restriction; checked when the experiment runs).
    train_datasets: tuple[str, ...] = Field(min_length=1)
    # Datasets that are only ever tested (D-033: Shin2017A, GuttmannFlury2025).
    test_only_datasets: tuple[str, ...] = ()
    labels: tuple[str, ...] = ("left_hand", "right_hand")
    # The model's input layout: C-core with zero-fill (§4.3) or the motor strip (D-031).
    channels: Literal["c-core", "motor"] = "c-core"
    reference: Reference = "car"
    alignment: Alignment = "ea"
    alignment_unit: AlignmentUnit = "session"
    conditions: tuple[PooledCondition, ...] = POOLED_CONDITIONS
    # Development folds (people stratified by dataset).
    folds: int = Field(default=5, ge=2, le=20)
    # Share of each model's training people (per dataset) held out for early stopping.
    validation_fraction: float = Field(default=0.1, ge=0.0, lt=0.5)
    # The cue-only control's window, in seconds from the cue (D-030).
    cue_only_epoch_s: tuple[float, float] = (0.0, 0.5)
    seeds: tuple[int, ...] = (20261005,)
    signal: dict[str, Any] = Field(default_factory=dict)
    model: EEGNetConfig = EEGNetConfig()
    # A6: a pretrained encoder replaces EEGNet (``model`` is then unused). None = EEGNet.
    backbone: BackboneConfig | None = None
    train: TrainConfig = TrainConfig()
    # E2: score the pooled models' test people under online-style warm-up alignment too.
    warmup: WarmupConfig | None = None
    # D-039: training crops and extra classes (validation and test windows are unchanged).
    training_windows: TrainingWindowsConfig | None = None
    # Q30: the pooled models fine-tuned on each test person's first few labelled trials.
    finetune: FinetuneConfig | None = None

    @model_validator(mode="after")
    def _check(self) -> PooledExperimentConfig:
        for field, values in (
            ("train_datasets", self.train_datasets),
            ("test_only_datasets", self.test_only_datasets),
            ("labels", self.labels),
            ("conditions", self.conditions),
            ("seeds", self.seeds),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"{field} has duplicates: {list(values)}")
        for field, values in (
            ("labels", self.labels),
            ("conditions", self.conditions),
            ("seeds", self.seeds),
        ):
            if not values:
                raise ValueError(f"{field} must not be empty")
        both = sorted(set(self.train_datasets) & set(self.test_only_datasets))
        if both:
            raise ValueError(f"datasets both trained on and test-only: {both}")
        if len(self.labels) < 2:
            raise ValueError(f"a classifier needs at least 2 labels, got {list(self.labels)}")
        if "lodo" in self.conditions and len(self.train_datasets) < 2:
            raise ValueError("leave-one-dataset-out needs at least 2 train_datasets")
        start, end = self.cue_only_epoch_s
        if not 0 <= start < end:
            raise ValueError(f"cue_only_epoch_s must satisfy 0 <= start < end, got {start}, {end}")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError(f"seeds must be >= 0, got {list(self.seeds)}")
        if self.warmup is not None:
            unknown = sorted(set(self.warmup.datasets) - set(self.datasets))
            if unknown:
                raise ValueError(f"warmup datasets {unknown} aren't in this experiment")
            if "pooled" not in self.conditions:
                raise ValueError("warmup scores the pooled models: conditions must include pooled")
            if self.alignment != "ea" or self.alignment_unit != "session":
                raise ValueError("warmup replaces per-session EA: needs alignment ea, unit session")
            if len(self.seeds) != 1:
                raise ValueError("warmup builds its sets for one seed's folds: give one seed")
        if self.training_windows is not None:
            overlap = sorted(set(self.training_windows.extra_labels) & set(self.labels))
            if overlap:
                raise ValueError(f"extra_labels repeat the experiment's labels: {overlap}")
        if self.finetune is not None:
            if self.warmup is None:
                raise ValueError(
                    "finetune uses the warm-up's stream settings and prior: add warmup"
                )
            outside = sorted(set(self.finetune.datasets) - set(self.warmup.datasets))
            if outside:
                raise ValueError(f"finetune datasets {outside} aren't warm-up datasets")
            if self.training_windows is not None and self.training_windows.extra_labels:
                raise ValueError("finetune fine-tunes a head over labels only: no extra_labels")
        return self

    @property
    def datasets(self) -> tuple[str, ...]:
        """Every dataset the experiment loads: trained-on first, then test-only."""
        return (*self.train_datasets, *self.test_only_datasets)

    @property
    def train_labels(self) -> tuple[str, ...]:
        """The classes the model's head learns: ``labels`` first (so their logits keep their
        indices), then any extra classes (D-039)."""
        extra = self.training_windows.extra_labels if self.training_windows else ()
        return (*self.labels, *extra)


def load_pooled_experiment_config(path: str | Path) -> PooledExperimentConfig:
    """Load and validate an E1 experiment YAML (same errors as ``load_experiment_config``)."""
    raw = _read_yaml_mapping(path)
    try:
        return PooledExperimentConfig.model_validate(raw)
    except ValidationError as exc:
        raise NeurOSConfigError(f"{path}: invalid experiment config:\n{exc}") from exc
