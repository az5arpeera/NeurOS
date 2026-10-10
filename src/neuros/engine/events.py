"""The engine's output: schema ``neuros.intent.v0`` (architecture §7.3).

Two event types, both JSON objects with ``"schema": "neuros.intent.v0"``:

- **intent**, one per decision (every hop, 8 per second at the contract's 0.125 s): the
  command, or ``"HOLD"`` with a reason; the smoothed probabilities of the enabled commands
  (the rest summed under ``"other"``); the alignment state; the window's quality; latency.
- **status**, on every engine state change (§7.2).

Validation happens at construction, so an event that reaches an app is always well-formed
(principle 4: the engine never emits garbage). ``calibration.level`` from the §7.3 sketch is
reserved and sent as null until calibration levels are defined.
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

    def __post_init__(self) -> None:
        if not isinstance(self.seq, int) or self.seq < 0:
            raise NeurOSDataError(f"seq must be a non-negative int, got {self.seq!r}")
        _finite("t", self.t)
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
                "level": None,
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


Event = IntentEvent | StatusEvent
