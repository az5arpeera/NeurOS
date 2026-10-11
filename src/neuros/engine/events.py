"""The engine's output: schema ``neuros.intent.v0`` (architecture §7.3).

Four event types, all JSON objects with ``"schema": "neuros.intent.v0"``:

- **intent**, one per decision (every hop, 8 per second at the contract's 0.125 s): the
  command, or ``"HOLD"`` with a reason; the smoothed probabilities of the enabled commands
  (the rest summed under ``"other"``); the alignment state; the window's quality; latency.
  ``calibration.level`` is null with zero calibration and ``"quick"`` once the optional
  guided calibration (D-040, ``neuros.engine.calibration``) has fine-tuned the model.
- **status**, on every engine state change (§7.2).
- **calibration**, the guided calibration's progress: each trial's phases (``ready``, ``cue``,
  ``imagine``, ``rest``) for the app to draw, then ``training`` and ``done`` or ``failed``.
- **telemetry**, opt-in (``neuros run --telemetry``), for displays such as the demo's stage
  view: with each decision, the newest samples of every channel the model reads (cleaned:
  band-passed and common-averaged, decimated for drawing) and each channel's RMS. Apps that
  only act on commands never need it.

Validation happens at construction, so an event that reaches an app is always well-formed
(principle 4: the engine never emits garbage).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

from neuros.errors import NeurOSDataError

SCHEMA = "neuros.intent.v0"
HOLD = "HOLD"
# Probabilities of commands the app didn't enable are summed under this key.
OTHER = "other"

HoldReason = Literal[
    "low_confidence",
    "unrecognized",
    "motion",
    "muscle",
    "flat",
    "saturated",
    "signal_gap",
    "warmup",
    "adaptation_reset",
]
HOLD_REASONS: tuple[str, ...] = get_args(HoldReason)

EngineState = Literal["disconnected", "settling", "adapting", "steady", "degraded"]
ENGINE_STATES: tuple[str, ...] = get_args(EngineState)

# The levels of ``calibration.level``: None (zero calibration) or one of these.
CalibrationLevel = Literal["quick"]
CALIBRATION_LEVELS: tuple[str, ...] = get_args(CalibrationLevel)

CalibrationPhase = Literal[
    "started",  # a calibration began (details: trials, expected seconds)
    "ready",  # fixation cross: get ready, don't move
    "cue",  # the arrow for ``command`` is on screen
    "imagine",  # cross only: imagine the movement
    "rest",  # blank: relax
    "training",  # all trials collected; fine-tuning (about a second)
    "done",  # the fine-tuned model is live (level "quick")
    "failed",  # not enough usable trials (or another reason): zero calibration stays
    "cancelled",  # the app cancelled it: the model in use stays
    "reset",  # back to zero calibration
]
CALIBRATION_PHASES: tuple[str, ...] = get_args(CalibrationPhase)
# What an app may ask an engine that offers the guided calibration (``EventHub`` requests).
CALIBRATION_REQUESTS = ("calibrate", "cancel_calibration", "reset_calibration")

# Slack for probabilities that should sum to 1 after float32 arithmetic.
_PROB_TOL = 1e-4


def _finite(name: str, value: float) -> None:
    if not (isinstance(value, int | float) and math.isfinite(value)):
        raise NeurOSDataError(f"{name} must be a finite number, got {value!r}")


@dataclass(frozen=True)
class IntentEvent:
    """One decision.

    Attributes:
        seq: per-engine counter shared with status events (gaps mean lost events).
        t: the stream clock (LSL seconds) at the decided window's last sample.
        command: an enabled command, or ``HOLD``.
        confidence: the smoothed probability of ``command`` (of the top class for a HOLD).
        probs: smoothed probability per enabled command, plus ``OTHER``.
        hold_reason: why the engine held; None exactly when ``command`` isn't ``HOLD``.
        state: the engine state (§7.2) when the decision was made.
        seconds_of_data: good (quality-passed) data in the alignment reference so far.
        quality_ok: whether this window passed the quality gate.
        bad_channels: channels the gate blamed, in stream order.
        latency_ms: compute time from the window's arrival to this event.
        calibration_level: None (zero calibration) or the guided calibration's level.
    """

    seq: int
    t: float
    command: str
    confidence: float
    probs: Mapping[str, float]
    hold_reason: HoldReason | None
    state: EngineState
    seconds_of_data: float
    quality_ok: bool
    bad_channels: tuple[str, ...] = ()
    latency_ms: float = 0.0
    calibration_level: CalibrationLevel | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.seq, int) or self.seq < 0:
            raise NeurOSDataError(f"seq must be a non-negative int, got {self.seq!r}")
        _finite("t", self.t)
        if self.calibration_level is not None and self.calibration_level not in CALIBRATION_LEVELS:
            raise NeurOSDataError(f"unknown calibration level {self.calibration_level!r}")
        if not self.command:
            raise NeurOSDataError("command must be a non-empty string")
        if (self.command == HOLD) != (self.hold_reason is not None):
            raise NeurOSDataError(
                f"hold_reason must be set exactly when command is {HOLD!r} "
                f"(command {self.command!r}, reason {self.hold_reason!r})"
            )
        if self.hold_reason is not None and self.hold_reason not in HOLD_REASONS:
            raise NeurOSDataError(f"unknown hold reason {self.hold_reason!r} {HOLD_REASONS}")
        if self.state not in ENGINE_STATES:
            raise NeurOSDataError(f"unknown engine state {self.state!r} {ENGINE_STATES}")
        _finite("confidence", self.confidence)
        if not 0.0 <= self.confidence <= 1.0:
            raise NeurOSDataError(f"confidence must be in [0, 1], got {self.confidence}")
        if not self.probs:
            raise NeurOSDataError("probs must name at least one command")
        for name, p in self.probs.items():
            _finite(f"probs[{name!r}]", p)
            if not 0.0 <= p <= 1.0 + _PROB_TOL:
                raise NeurOSDataError(f"probs[{name!r}] must be in [0, 1], got {p}")
        if sum(self.probs.values()) > 1.0 + _PROB_TOL:
            raise NeurOSDataError(f"probs sum to {sum(self.probs.values())}, above 1")
        if self.command != HOLD and self.command not in self.probs:
            raise NeurOSDataError(f"command {self.command!r} is missing from probs")
        _finite("seconds_of_data", self.seconds_of_data)
        _finite("latency_ms", self.latency_ms)
        if self.seconds_of_data < 0 or self.latency_ms < 0:
            raise NeurOSDataError("seconds_of_data and latency_ms must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        """The wire format (§7.3)."""
        return {
            "type": "intent",
            "schema": SCHEMA,
            "seq": self.seq,
            "t": self.t,
            "command": self.command,
            "confidence": round(self.confidence, 6),
            "probs": {name: round(float(p), 6) for name, p in self.probs.items()},
            "hold_reason": self.hold_reason,
            "calibration": {
                "state": self.state,
                "seconds_of_data": round(self.seconds_of_data, 3),
                "level": self.calibration_level,
            },
            "quality": {"ok": self.quality_ok, "bad_channels": list(self.bad_channels)},
            "latency_ms": round(self.latency_ms, 3),
        }

    def to_json(self) -> str:
        """One line of JSON (no NaN/Infinity: they aren't JSON)."""
        return json.dumps(self.to_dict(), allow_nan=False, separators=(",", ":"))


@dataclass(frozen=True)
class StatusEvent:
    """An engine state change (§7.2), with the reason in plain words."""

    seq: int
    t: float
    state: EngineState
    previous: EngineState
    reason: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.seq, int) or self.seq < 0:
            raise NeurOSDataError(f"seq must be a non-negative int, got {self.seq!r}")
        _finite("t", self.t)
        for name in ("state", "previous"):
            if getattr(self, name) not in ENGINE_STATES:
                raise NeurOSDataError(f"unknown engine state {getattr(self, name)!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "status",
            "schema": SCHEMA,
            "seq": self.seq,
            "t": self.t,
            "state": self.state,
            "previous": self.previous,
            "reason": self.reason,
            "details": dict(self.details),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), allow_nan=False, separators=(",", ":"))


@dataclass(frozen=True)
class CalibrationEvent:
    """A step of the guided calibration (``neuros.engine.calibration``).

    Attributes:
        seq: the engine's shared counter.
        t: the stream clock at which the phase starts (the engine emits it once its data has
            reached that time, so an app receives it within about a chunk).
        phase: see ``CalibrationPhase``.
        trial: 1-based trial number for the trial phases, else None.
        n_trials: trials in this calibration.
        label: the class of this trial (``cue``, ``imagine``), else None.
        command: the app's command for ``label`` (e.g. ``"LEFT"``): what the cue shows.
        duration_s: how long the phase lasts (for countdowns), if known.
        message: plain words for a person (``failed``: why).
        details: numbers (``done``: usable trials per class, seconds, the fit on the trials).
    """

    seq: int
    t: float
    phase: CalibrationPhase
    n_trials: int
    trial: int | None = None
    label: str | None = None
    command: str | None = None
    duration_s: float | None = None
    message: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.seq, int) or self.seq < 0:
            raise NeurOSDataError(f"seq must be a non-negative int, got {self.seq!r}")
        _finite("t", self.t)
        if self.phase not in CALIBRATION_PHASES:
            raise NeurOSDataError(f"unknown calibration phase {self.phase!r}")
        if not isinstance(self.n_trials, int) or self.n_trials < 0:
            raise NeurOSDataError(f"n_trials must be a non-negative int, got {self.n_trials!r}")
        if self.trial is not None and not 1 <= self.trial <= self.n_trials:
            raise NeurOSDataError(f"trial {self.trial} is outside 1..{self.n_trials}")
        if self.phase in ("cue", "imagine") and (self.label is None or self.command is None):
            raise NeurOSDataError(f"a {self.phase!r} event needs its label and command")
        if self.duration_s is not None:
            _finite("duration_s", self.duration_s)
            if self.duration_s < 0:
                raise NeurOSDataError(f"duration_s must be >= 0, got {self.duration_s}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "calibration",
            "schema": SCHEMA,
            "seq": self.seq,
            "t": self.t,
            "phase": self.phase,
            "trial": self.trial,
            "n_trials": self.n_trials,
            "label": self.label,
            "command": self.command,
            "duration_s": None if self.duration_s is None else round(self.duration_s, 3),
            "message": self.message,
            "details": dict(self.details),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), allow_nan=False, separators=(",", ":"))


@dataclass(frozen=True)
class TelemetryEvent:
    """What the engine just saw, for drawing (opt-in; never needed to act on commands).

    Attributes:
        seq: the engine's shared counter.
        t: the stream clock at the window's last sample.
        channels: the channels the model reads on this headset (the profile), in order.
        sfreq: the rate of ``samples`` (Hz), after decimation.
        samples: per channel, the newest samples since the previous decision (µV, cleaned:
            band-passed and common-averaged), oldest first.
        rms_uv: per channel, the RMS of the whole decision window (µV).
        bad_channels: channels the quality gate blamed for this window.
        in_transient: the filters are still settling (the engine holds with "warmup").
        state: the engine state (§7.2).
        seconds_of_data: good data in the alignment reference so far (the warm-up's progress).
    """

    seq: int
    t: float
    channels: tuple[str, ...]
    sfreq: float
    samples: tuple[tuple[float, ...], ...]
    rms_uv: tuple[float, ...]
    bad_channels: tuple[str, ...]
    in_transient: bool
    state: EngineState
    seconds_of_data: float

    def __post_init__(self) -> None:
        if not isinstance(self.seq, int) or self.seq < 0:
            raise NeurOSDataError(f"seq must be a non-negative int, got {self.seq!r}")
        for name in ("t", "sfreq", "seconds_of_data"):
            _finite(name, getattr(self, name))
        if self.sfreq <= 0 or self.seconds_of_data < 0:
            raise NeurOSDataError("sfreq must be > 0 and seconds_of_data >= 0")
        if not self.channels or len(set(self.channels)) != len(self.channels):
            raise NeurOSDataError(f"channels must be non-empty and unique: {self.channels}")
        if len(self.samples) != len(self.channels) or len(self.rms_uv) != len(self.channels):
            raise NeurOSDataError("samples and rms_uv need one entry per channel")
        if len({len(row) for row in self.samples}) > 1:
            raise NeurOSDataError("every channel needs the same number of samples")
        for row in self.samples:
            for value in row:
                _finite("sample", value)
        for value in self.rms_uv:
            _finite("rms_uv", value)
            if value < 0:
                raise NeurOSDataError(f"rms_uv must be >= 0, got {value}")
        unknown = [c for c in self.bad_channels if c not in self.channels]
        if unknown:
            raise NeurOSDataError(f"bad channels {unknown} aren't among {list(self.channels)}")
        if self.state not in ENGINE_STATES:
            raise NeurOSDataError(f"unknown engine state {self.state!r} {ENGINE_STATES}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "telemetry",
            "schema": SCHEMA,
            "seq": self.seq,
            "t": self.t,
            "channels": list(self.channels),
            "sfreq": self.sfreq,
            "samples": [[round(v, 2) for v in row] for row in self.samples],
            "rms_uv": [round(v, 2) for v in self.rms_uv],
            "bad_channels": list(self.bad_channels),
            "in_transient": self.in_transient,
            "state": self.state,
            "seconds_of_data": round(self.seconds_of_data, 3),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), allow_nan=False, separators=(",", ":"))


Event = IntentEvent | StatusEvent | CalibrationEvent | TelemetryEvent
