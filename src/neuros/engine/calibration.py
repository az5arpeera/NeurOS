"""The optional guided calibration (D-040): about 3 minutes of cued trials, then our model
fine-tuned to the person, live.

**Why** (Q30, ``knowledge-base/results/2026-10-10-q30-finetune-q31-fit-check.md``): our model
plus 10 trials per hand came within 1-3 points of a 10-13-minute conventional calibration,
offline. This module does the same inside the engine, with the same recipe: the experiment
file that produced that result (``configs/e2_finetune.yaml``) is the recipe's source.

**A trial** (``CalibrationProtocol``; about 8.5 s, so 20 trials take about 3 minutes)::

    ready (cross, 1.5 s) -> cue (arrow, 1 s) -> imagine (cross only, 4 s) -> rest (blank, 1.5-2.5 s)

The labelled window is the contract's D-030 window: it starts ``cue_margin_s`` (0.25 s) after
the arrow has left the screen, so 1.25 s after the cue, and lasts ``window_s`` (2 s). One
window per trial, as in Q30. Trials come in a seeded random order, at most 2 of a class in a
row, so a person can't predict the next one.

**The engine owns the timing.** It emits each phase once its data has reached that time and
labels the EEG by its own stream clock, so a slow page can't mislabel a trial. A page shows
the arrow up to about a chunk (~60 ms) late; the window still starts after the arrow is gone.

**Two cue sources:**

- ``scheduled``: the engine schedules the trials (a live headset, ``neuros run``).
- ``recorded``: a replayed recording's own trials (``neuros simulate``). The person in a
  recording followed their own cues, not ours, so the next ``per_class`` trials per class
  after the request are used, with the dataset's own window.

**When the last window is in** (``Calibrator.step``):

1. At least ``min_good_fraction`` (80%) of each class's windows must have passed the quality
   gate outside the filter start-up. Otherwise the calibration fails, and zero calibration
   stays.
2. The windows are aligned with the engine's current reference and placed exactly as live
   windows. That is the reference the engine keeps using afterwards, so training and use see
   the same alignment.
3. A copy of the model is fine-tuned on them with Q30's recipe. That takes about a second on
   a laptop CPU. The bundle's model is never modified.
4. A copy that predicts one class for every calibration window is refused (``failed``).
5. The engine switches to the copy (``Engine.set_decoder``, level ``"quick"``).

``reset`` returns to the bundle's model. Not covered yet (D-040): whether a calibration still
holds on another day, so nothing is saved to disk.
"""

from __future__ import annotations

import copy
import itertools
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from neuros.config.experiment import TrainConfig
from neuros.config.schemas import SignalContract
from neuros.engine.bundle import TorchDecoder
from neuros.engine.events import CALIBRATION_REQUESTS, CalibrationEvent, CalibrationPhase
from neuros.engine.pipeline import Engine, EngineWindow, place_window
from neuros.engine.simulator import Marker
from neuros.errors import NeurOSConfigError, NeurOSDataError
from neuros.eval.finetune import FinetuneSettings

log = logging.getLogger(__name__)

CueSource = Literal["scheduled", "recorded"]
REQUESTS = CALIBRATION_REQUESTS
DEFAULT_RECIPE = Path("configs/e2_finetune.yaml")
DEFAULT_PER_CLASS = 10  # Q30: 10 trials per hand, about 3 minutes
TRIAL_PHASES = ("ready", "cue", "imagine", "rest")
_EPS_S = 1e-6
# Attempts at a random trial order with no run longer than ``max_run`` before giving up.
_ORDER_ATTEMPTS = 10_000


# ---------------------------------------------------------------------- the recipe
@dataclass(frozen=True)
class CalibrationRecipe:
    """How to fine-tune: the base training recipe and Q30's fine-tuning settings."""

    train: TrainConfig
    finetune: FinetuneSettings
    source: str = ""

    @classmethod
    def from_experiment(cls, path: str | Path) -> CalibrationRecipe:
        """The recipe of an experiment YAML with a ``finetune`` section (the Q30 run's).

        Raises:
            NeurOSConfigError: if the file has no ``finetune`` section.
        """
        from neuros.config.experiment import load_pooled_experiment_config

        config = load_pooled_experiment_config(path)
        if config.finetune is None:
            raise NeurOSConfigError(f"{path} has no finetune section: no calibration recipe")
        return cls(config.train, FinetuneSettings.from_config(config.finetune), str(path))


def fine_tune(
    model: Any,
    x: NDArray[np.float32],
    labels: Sequence[str],
    *,
    recipe: CalibrationRecipe,
    seed: int,
) -> Any:
    """A copy of ``model`` fine-tuned on placed windows ``x`` (Q30's ``finetune_and_score``
    step: every layer, every batch = all the windows, no early stopping, no SWA), on the CPU,
    in evaluation mode. ``model`` itself is never modified.

    Raises:
        NeurOSDataError: on labels outside the model's classes, or windows that don't match
            the labels or the model's input.
    """
    import torch

    from neuros.train.data import TensorData
    from neuros.train.trainer import train_classifier

    head = tuple(model.classes)
    index = {name: i for i, name in enumerate(head)}
    unknown = sorted(set(labels) - set(index))
    if unknown:
        raise NeurOSDataError(f"calibration labels {unknown} aren't the model's classes {head}")
    data = np.ascontiguousarray(x, dtype=np.float32)
    if data.ndim != 3 or data.shape[0] != len(labels) or data.shape[0] == 0:
        raise NeurOSDataError(f"{data.shape} windows for {len(labels)} labels")
    tuned = copy.deepcopy(model)
    train_classifier(
        tuned,
        TensorData(
            x=torch.from_numpy(data),
            y=torch.tensor([index[label] for label in labels], dtype=torch.int64),
            people=("calibration",) * data.shape[0],
            classes=head,
        ),
        None,
        recipe.finetune.train_config(recipe.train, data.shape[0]),
        seed=seed,
    )
    tuned.to("cpu")
    tuned.eval()
    return tuned


# ---------------------------------------------------------------------- the protocol
@dataclass(frozen=True)
class CalibrationProtocol:
    """The guided trials (see the module docs). Times in seconds."""

    per_class: int = DEFAULT_PER_CLASS
    ready_s: float = 1.5
    cue_s: float = 1.0
    imagine_s: float = 4.0
    rest_s: tuple[float, float] = (1.5, 2.5)
    lead_s: float = 2.0
    min_good_fraction: float = 0.8
    max_run: int = 2
    seed: int = 0

    def __post_init__(self) -> None:
        if not 1 <= self.per_class <= 100:
            raise NeurOSConfigError(f"per_class must be in 1..100, got {self.per_class}")
        for name in ("ready_s", "cue_s", "imagine_s", "lead_s"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value >= 0):
                raise NeurOSConfigError(f"{name} must be >= 0, got {value}")
        if self.cue_s <= 0 or self.imagine_s <= 0:
            raise NeurOSConfigError("cue_s and imagine_s must be > 0")
        low, high = self.rest_s
        if not (math.isfinite(low) and math.isfinite(high) and 0 <= low <= high):
            raise NeurOSConfigError(
                f"rest_s must be (low, high) with 0 <= low <= high: {self.rest_s}"
            )
        if not 0 < self.min_good_fraction <= 1:
            raise NeurOSConfigError(
                f"min_good_fraction must be in (0, 1]: {self.min_good_fraction}"
            )
        if self.max_run < 1:
            raise NeurOSConfigError(f"max_run must be >= 1, got {self.max_run}")

    @property
    def task_s(self) -> float:
        """From the cue's onset to the end of the imagery."""
        return self.cue_s + self.imagine_s

    def window_offset_s(self, contract: SignalContract) -> float:
        """The labelled window's start after the cue (D-030: the arrow's end + the margin).

        Raises:
            NeurOSConfigError: if that window doesn't fit in the imagery.
        """
        start = self.cue_s + contract.cue_margin_s
        if start + contract.window_s > self.task_s + _EPS_S:
            raise NeurOSConfigError(
                f"a {contract.window_s}-s window starting {start:g} s after the cue doesn't fit "
                f"in {self.task_s:g} s of cue + imagery"
            )
        return start

    def min_good(self) -> int:
        """Usable windows needed per class."""
        return math.ceil(self.min_good_fraction * self.per_class - 1e-9)

    def expected_s(self, n_classes: int) -> float:
        """About how long a scheduled calibration takes."""
        trial = self.ready_s + self.task_s + sum(self.rest_s) / 2
        return self.lead_s + n_classes * self.per_class * trial

    def order(self, classes: Sequence[str]) -> tuple[str, ...]:
        """A seeded random order with ``per_class`` of each class, no run above ``max_run``.

        Raises:
            NeurOSConfigError: if no such order turns up (only with extreme settings).
        """
        labels = np.repeat(np.asarray(classes, dtype=object), self.per_class)
        rng = np.random.default_rng(self.seed)
        for _ in range(_ORDER_ATTEMPTS):
            candidate = labels[rng.permutation(labels.size)]
            run = longest = 1
            for previous, current in itertools.pairwise(candidate):
                run = run + 1 if current == previous else 1
                longest = max(longest, run)
            if longest <= self.max_run:
                return tuple(str(label) for label in candidate)
        raise NeurOSConfigError(
            f"no order of {self.per_class} x {list(classes)} without runs above {self.max_run}"
        )


@dataclass(frozen=True)
class CalibrationTrial:
    """One trial on the stream clock."""

    number: int  # 1-based
    label: str
    cue_t: float
    window_end_t: float  # the labelled window's last sample
    phases: tuple[tuple[CalibrationPhase, float, float | None], ...]  # (phase, start, duration)


def scheduled_trials(
    protocol: CalibrationProtocol,
    classes: Sequence[str],
    start_t: float,
    contract: SignalContract,
) -> list[CalibrationTrial]:
    """The engine's own trials, starting ``protocol.lead_s`` after ``start_t``."""
    offset = protocol.window_offset_s(contract)
    rng = np.random.default_rng(protocol.seed + 1)  # rest jitter; the order uses ``seed``
    trials = []
    t = start_t + protocol.lead_s
    for number, label in enumerate(protocol.order(classes), start=1):
        cue = t + protocol.ready_s
        rest = float(rng.uniform(*protocol.rest_s))
        phases: tuple[tuple[CalibrationPhase, float, float | None], ...] = (
            ("ready", t, protocol.ready_s),
            ("cue", cue, protocol.cue_s),
            ("imagine", cue + protocol.cue_s, protocol.imagine_s),
            ("rest", cue + protocol.task_s, rest),
        )
        trials.append(
            CalibrationTrial(number, label, cue, cue + offset + contract.window_s, phases)
        )
        t = cue + protocol.task_s + rest
    return trials


def recorded_trials(
    markers: Sequence[Marker],
    classes: Sequence[str],
    per_class: int,
    after_t: float,
    *,
    window_offset_s: float,
    window_s: float,
    task_s: float,
) -> list[CalibrationTrial]:
    """A recording's next ``per_class`` trials per class whose cue is at or after ``after_t``,
    in time order, labelled with the dataset's window (``window_offset_s`` after the cue).

    Raises:
        NeurOSDataError: if the recording has too few trials of a class left.
    """
    taken: dict[str, list[Marker]] = {label: [] for label in classes}
    for marker in sorted(markers, key=lambda m: m.t):
        if marker.t >= after_t and marker.label in taken and len(taken[marker.label]) < per_class:
            taken[marker.label].append(marker)
    short = {label: len(found) for label, found in taken.items() if len(found) < per_class}
    if short:
        raise NeurOSDataError(
            f"the recording has too few trials left for a calibration: {short} (need {per_class})"
        )
    chosen = sorted((m for found in taken.values() for m in found), key=lambda m: m.t)
    return [
        CalibrationTrial(
            number,
            m.label,
            m.t,
            m.t + window_offset_s + window_s,
            (("cue", m.t, task_s), ("rest", m.t + task_s, None)),
        )
        for number, m in enumerate(chosen, start=1)
    ]


# ---------------------------------------------------------------------- the calibrator
@dataclass
class _Slot:
    """A trial's labelled window, once the engine has processed it."""

    trial: CalibrationTrial
    referenced: NDArray[np.float32] | None = None
    good: bool = False


@dataclass
class CalibrationResult:
    """How the last calibration went (for reports and replay scores)."""

    ok: bool
    started_t: float
    ended_t: float
    source: CueSource
    usable: dict[str, int] = field(default_factory=dict)
    fit_on_trials: float | None = None
    message: str = ""


class Calibrator:
    """Runs guided calibrations on one engine (see the module docs).

    Feed it the stream clock after every ``Engine.push`` (``step``) and the apps' requests
    (``handle``); emit the events it returns. Not thread-safe (like the engine).
    """

    def __init__(
        self,
        engine: Engine,
        model: Any,
        *,
        recipe: CalibrationRecipe,
        commands: Mapping[str, str],
        protocol: CalibrationProtocol | None = None,
        markers: Sequence[Marker] | None = None,
        recorded_window_offset_s: float | None = None,
        recorded_task_s: float | None = None,
        seed: int = 0,
    ) -> None:
        """
        Args:
            model: the bundle's model (``TorchDecoder.model``); fine-tuned copies come from it.
            commands: class -> the app's command name (what the cue shows).
            markers: a replay's markers: calibrations then use the recording's own trials
                (``recorded``), with ``recorded_window_offset_s`` (the dataset's task epoch
                start) and ``recorded_task_s``.

        Raises:
            NeurOSConfigError: on commands missing a class, or recorded markers without the
                dataset's window offset and task length.
        """
        self.engine = engine
        self.model = model
        self.recipe = recipe
        self.protocol = protocol or CalibrationProtocol()
        self.classes = tuple(engine.decoder.classes)
        missing = [c for c in self.classes if c not in commands]
        if missing:
            raise NeurOSConfigError(f"no command name for classes {missing}")
        self.commands = dict(commands)
        if markers is not None and (recorded_window_offset_s is None or recorded_task_s is None):
            raise NeurOSConfigError("recorded calibration needs the window offset and task length")
        self.markers = tuple(markers) if markers is not None else None
        self.source: CueSource = "recorded" if markers is not None else "scheduled"
        self._recorded_offset = recorded_window_offset_s
        self._recorded_task = recorded_task_s
        self.seed = seed
        self.base_decoder = engine.decoder
        self.state: Literal["idle", "collecting", "training"] = "idle"
        self.result: CalibrationResult | None = None
        self._slots: list[_Slot] = []
        self._pending: list[tuple[float, CalibrationTrial, CalibrationPhase, float | None]] = []
        self._started_t = 0.0
        engine.add_observer(self._observe)

    # ------------------------------------------------------------------ requests
    @property
    def active(self) -> bool:
        return self.state != "idle"

    @property
    def n_trials(self) -> int:
        return len(self._slots)

    def handle(self, request: str, t: float) -> list[CalibrationEvent]:
        """An app's request (one of ``REQUESTS``) at stream time ``t``.

        Raises:
            NeurOSConfigError: on an unknown request.
        """
        if request == "calibrate":
            return self.start(t)
        if request == "cancel_calibration":
            return self.cancel(t)
        if request == "reset_calibration":
            return self.reset(t)
        raise NeurOSConfigError(f"unknown request {request!r}; expected one of {REQUESTS}")

    def start(self, t: float) -> list[CalibrationEvent]:
        """Begin a calibration at stream time ``t`` (ignored while one is running)."""
        if self.active:
            log.info("a calibration is already running; request ignored")
            return []
        contract = self.engine.config.contract
        try:
            if self.markers is None:
                trials = scheduled_trials(self.protocol, self.classes, t, contract)
            else:
                if self._recorded_offset is None or self._recorded_task is None:  # __init__ checks
                    raise NeurOSConfigError("recorded calibration needs the dataset's window")
                trials = recorded_trials(
                    self.markers,
                    self.classes,
                    self.protocol.per_class,
                    t,
                    window_offset_s=self._recorded_offset,
                    window_s=contract.window_s,
                    task_s=self._recorded_task,
                )
        except NeurOSDataError as error:
            self.result = CalibrationResult(False, t, t, self.source, message=str(error))
            return [self._event(t, "failed", n_trials=0, message=str(error))]
        self._slots = [_Slot(trial) for trial in trials]
        self._pending = sorted(
            (
                (start, trial, phase, duration)
                for trial in trials
                for phase, start, duration in trial.phases
            ),
            key=lambda item: (item[0], item[1].number),
        )
        self._started_t = t
        self.state = "collecting"
        expected = trials[-1].window_end_t - t
        return [
            self._event(
                t,
                "started",
                message=f"{len(trials)} trials, about {expected / 60:.1f} minutes",
                details={
                    "source": self.source,
                    "per_class": self.protocol.per_class,
                    "expected_s": round(expected, 1),
                },
            )
        ]

    def cancel(self, t: float) -> list[CalibrationEvent]:
        """Stop a running calibration; the model in use stays."""
        if not self.active:
            return []
        self._clear()
        return [self._event(t, "cancelled", message="calibration cancelled")]

    def reset(self, t: float) -> list[CalibrationEvent]:
        """Back to zero calibration (the bundle's model); cancels a running calibration."""
        self._clear()
        self.engine.set_decoder(self.base_decoder, level=None)
        return [self._event(t, "reset", message="back to zero calibration")]

    # ------------------------------------------------------------------ progress
    def step(self, t: float | None) -> list[CalibrationEvent]:
        """Phase events due by stream time ``t``; then, once every labelled window is in, the
        ``training`` event, and on the next step the fine-tune and ``done`` / ``failed``."""
        if t is None or self.state == "idle":
            return []
        if self.state == "training":
            return self._finish(t)
        events = []
        while self._pending and self._pending[0][0] <= t + _EPS_S:
            start, trial, phase, duration = self._pending.pop(0)
            command = self.commands[trial.label] if phase in ("cue", "imagine") else None
            events.append(
                self._event(
                    start,
                    phase,
                    trial=trial.number,
                    label=trial.label if command else None,
                    command=command,
                    duration_s=duration,
                )
            )
        last_window = max(slot.trial.window_end_t for slot in self._slots)
        if t >= last_window + self.engine.hop_s:
            self._pending.clear()  # the last rest isn't needed
            self.state = "training"
            events.append(self._event(t, "training", message="fine-tuning to you"))
        return events

    def _observe(self, window: EngineWindow) -> None:
        """Keep the window nearest each trial's labelled window (within half a hop)."""
        if self.state != "collecting":
            return
        tolerance = self.engine.hop_s / 2 + _EPS_S
        for slot in self._slots:
            if slot.referenced is None and abs(window.t - slot.trial.window_end_t) <= tolerance:
                slot.referenced = np.array(window.referenced, dtype=np.float32, copy=True)
                slot.good = bool(window.quality.ok and not window.in_transient)
                break

    def _finish(self, t: float) -> list[CalibrationEvent]:
        self.state = "idle"
        usable = {
            self.commands[c]: sum(s.good and s.trial.label == c for s in self._slots)
            for c in self.classes
        }
        need = self.protocol.min_good()
        short = [command for command, k in usable.items() if k < need]
        if short:
            message = (
                f"Only {', '.join(f'{usable[c]} {c}' for c in short)} trials of "
                f"{self.protocol.per_class} were usable (signal quality). Zero calibration "
                "stays; check the electrodes and try again."
            )
            return self._fail(t, message, usable)
        good = [s for s in self._slots if s.good and s.referenced is not None]
        reference = self.engine.reference.current
        rows = [self.engine.layout.index(ch) for ch in self.engine.profile]
        x = np.stack(
            [
                place_window(
                    s.referenced,  # type: ignore[arg-type]
                    reference,
                    rows,
                    len(self.engine.layout),
                    self.engine.config.input_scale,
                )
                for s in good
            ]
        )
        labels = [s.trial.label for s in good]
        tuned = fine_tune(self.model, x, labels, recipe=self.recipe, seed=self.seed)
        decoder = TorchDecoder(tuned, self.engine.layout, self.classes)
        predicted = np.asarray(decoder.predict_proba(x)).argmax(axis=1)
        if np.unique(predicted).size < 2:
            return self._fail(
                t,
                "The calibration didn't separate the commands. Zero calibration stays; try "
                "again, imagining each movement clearly.",
                usable,
            )
        fit = float(
            np.mean([self.classes[i] == label for i, label in zip(predicted, labels, strict=True)])
        )
        self.engine.set_decoder(decoder, level="quick")
        self.result = CalibrationResult(
            True, self._started_t, t, self.source, usable, fit, "calibrated"
        )
        self._slots = []
        return [
            self._event(
                t,
                "done",
                n_trials=len(labels),
                message="calibrated: your model is live",
                details={
                    "usable": usable,
                    "seconds": round(t - self._started_t, 1),
                    # How well the copy fits its own calibration trials: a sanity check, not
                    # an accuracy estimate (20 trials can't give one).
                    "fit_on_trials": round(fit, 3),
                },
            )
        ]

    # ------------------------------------------------------------------ helpers
    def _fail(self, t: float, message: str, usable: dict[str, int]) -> list[CalibrationEvent]:
        self.result = CalibrationResult(
            False, self._started_t, t, self.source, usable, None, message
        )
        n = len(self._slots)
        self._slots = []
        return [self._event(t, "failed", n_trials=n, message=message, details={"usable": usable})]

    def _clear(self) -> None:
        self.state = "idle"
        self._slots = []
        self._pending = []

    def _event(
        self,
        t: float,
        phase: CalibrationPhase,
        *,
        n_trials: int | None = None,
        trial: int | None = None,
        label: str | None = None,
        command: str | None = None,
        duration_s: float | None = None,
        message: str = "",
        details: Mapping[str, Any] | None = None,
    ) -> CalibrationEvent:
        return CalibrationEvent(
            seq=self.engine.next_seq(),
            t=t,
            phase=phase,
            n_trials=self.n_trials if n_trials is None else n_trials,
            trial=trial,
            label=label,
            command=command,
            duration_s=duration_s,
            message=message,
            details=dict(details or {}),
        )
