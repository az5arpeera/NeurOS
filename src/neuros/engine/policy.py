"""The decision policy (architecture §7.1, stage 8): probabilities in, a command or HOLD out.

Per decision (one per hop):

1. **Smooth** the model's probabilities over all its classes with an exponential moving
   average of time constant ``smoothing_s`` (§7.1: ~0.5 s). One window is noisy; a command
   that flickers between LEFT and RIGHT 8 times a second is useless to an app.
2. **Unrecognized** (D-024): if the most likely class isn't one the app enabled (a decoy, for
   example "feet" when the app only listens for LEFT/RIGHT), hold with ``unrecognized``.
3. **Threshold**: if its smoothed probability is below ``threshold`` (θ), hold with
   ``low_confidence``.
4. **Debounce**: emit the command only after ``debounce`` consecutive decisions agree; until
   then hold with ``low_confidence``.

§7.1 also holds when a window is far from every text anchor; that needs the text-anchored head
(E4) and isn't part of the linear head's policy.

**Provisional θ.** §7.1 tunes θ on validation people to a target precision; until that runs,
the default (0.6 on two classes) is a placeholder that apps can override.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from neuros.engine.events import HOLD, OTHER, HoldReason
from neuros.errors import NeurOSConfigError, NeurOSDataError

# Probabilities from a decoder must sum to 1 within this (softmax in float32).
_SUM_TOL = 1e-3


@dataclass(frozen=True)
class PolicySettings:
    """What the app enabled, and how cautious to be.

    Attributes:
        commands: model class -> command name the app receives, e.g.
            ``{"left_hand": "LEFT", "right_hand": "RIGHT"}``. Classes not listed are decoys.
        threshold: θ, the smoothed probability a command needs (0.5-1).
        smoothing_s: the probability EMA's time constant in seconds (0 = no smoothing).
        debounce: consecutive agreeing decisions before a command is emitted (>= 1).
    """

    commands: Mapping[str, str] = field(
        default_factory=lambda: {"left_hand": "LEFT", "right_hand": "RIGHT"}
    )
    threshold: float = 0.6
    smoothing_s: float = 0.5
    debounce: int = 2

    def __post_init__(self) -> None:
        if not self.commands:
            raise NeurOSConfigError("enable at least one command")
        names = list(self.commands.values())
        if len(set(names)) != len(names) or any(not n for n in names):
            raise NeurOSConfigError(f"command names must be distinct and non-empty: {names}")
        if HOLD in names or OTHER in names:
            raise NeurOSConfigError(f"{HOLD!r} and {OTHER!r} are reserved command names")
        if not (math.isfinite(self.threshold) and 0.5 <= self.threshold <= 1.0):
            raise NeurOSConfigError(f"threshold must be in [0.5, 1], got {self.threshold}")
        if not (math.isfinite(self.smoothing_s) and self.smoothing_s >= 0):
            raise NeurOSConfigError(f"smoothing_s must be >= 0, got {self.smoothing_s}")
        if not isinstance(self.debounce, int) or self.debounce < 1:
            raise NeurOSConfigError(f"debounce must be an int >= 1, got {self.debounce!r}")


@dataclass(frozen=True)
class Decision:
    """The policy's output for one window."""

    command: str  # a command name, or HOLD
    confidence: float  # smoothed probability of the top class
    probs: dict[str, float]  # per enabled command, plus OTHER
    hold_reason: HoldReason | None


class Policy:
    """Smoothing, threshold and debounce over a decoder's classes (see module docs)."""

    def __init__(self, classes: Sequence[str], settings: PolicySettings, hop_s: float) -> None:
        """
        Raises:
            NeurOSConfigError: if a command isn't one of the decoder's classes, or hop_s <= 0.
        """
        if len(classes) < 2 or len(set(classes)) != len(classes):
            raise NeurOSConfigError(f"need at least 2 distinct classes, got {list(classes)}")
        unknown = sorted(set(settings.commands) - set(classes))
        if unknown:
            raise NeurOSConfigError(f"commands {unknown} aren't classes of the model {classes}")
        if not (math.isfinite(hop_s) and hop_s > 0):
            raise NeurOSConfigError(f"hop_s must be > 0, got {hop_s}")
        self.classes = tuple(classes)
        self.settings = settings
        # EMA weight of the newest window: 1 - exp(-hop / tau); 1 means no smoothing.
        tau = settings.smoothing_s
        self.alpha = 1.0 if tau == 0 else 1.0 - math.exp(-hop_s / tau)
        self.reset()

    def reset(self) -> None:
        """Forget the smoothed state (after a gap or bad signal: old evidence is stale)."""
        self._smoothed: NDArray[np.float64] | None = None
        self._candidate: str | None = None
        self._streak = 0

    def interrupt(self) -> None:
        """A window the policy didn't see (bad signal): break the debounce streak only."""
        self._candidate = None
        self._streak = 0

    def snapshot(self) -> tuple[float, dict[str, float]]:
        """``(confidence, probs per command + OTHER)`` of the current smoothed state, for HOLD
        events the policy didn't decide (uniform over the classes before any decision)."""
        if self._smoothed is None:
            smoothed = np.full(len(self.classes), 1.0 / len(self.classes))
        else:
            smoothed = self._smoothed
        return float(smoothed.max()), self._by_command(smoothed)

    def _by_command(self, smoothed: NDArray[np.float64]) -> dict[str, float]:
        commands = self.settings.commands
        by_command = {
            commands[c]: float(smoothed[i]) for i, c in enumerate(self.classes) if c in commands
        }
        by_command[OTHER] = float(
            sum(smoothed[i] for i, c in enumerate(self.classes) if c not in commands)
        )
        return by_command

    def decide(self, probs: NDArray[np.floating]) -> Decision:
        """Update with one window's probabilities ``[classes]`` and decide.

        Raises:
            NeurOSDataError: on probabilities of the wrong shape, non-finite, negative, or not
                summing to 1 (a broken decoder must not silently drive commands).
        """
        p = np.asarray(probs, dtype=np.float64)
        if p.shape != (len(self.classes),):
            raise NeurOSDataError(f"probs must be [{len(self.classes)}], got {p.shape}")
        if not np.isfinite(p).all() or (p < 0).any() or abs(p.sum() - 1.0) > _SUM_TOL:
            raise NeurOSDataError(f"probs must be finite, >= 0 and sum to 1, got {p.tolist()}")
        if self._smoothed is None:
            self._smoothed = p.copy()
        else:
            self._smoothed = (1.0 - self.alpha) * self._smoothed + self.alpha * p
        smoothed = self._smoothed
        top = int(np.argmax(smoothed))
        top_class = self.classes[top]
        confidence = float(smoothed[top])
        commands = self.settings.commands
        by_command = self._by_command(smoothed)

        reason: HoldReason | None = None
        if top_class not in commands:
            reason = "unrecognized"
        elif confidence < self.settings.threshold:
            reason = "low_confidence"
        if reason is not None:
            self.interrupt()
            return Decision(HOLD, confidence, by_command, reason)
        if top_class == self._candidate:
            self._streak += 1
        else:
            self._candidate, self._streak = top_class, 1
        if self._streak < self.settings.debounce:
            return Decision(HOLD, confidence, by_command, "low_confidence")
        return Decision(commands[top_class], confidence, by_command, None)
