"""E7: does the engine stay quiet when nobody is trying to give a command?

The plan, with its bar set before any code: ``knowledge-base/plan/e7-quiet-when-idle.md``.

**The test.** EEGMAT's development people (never seen by the shipping bundle) are replayed
through the real engine as a 3-channel headset (C3, Cz, C4: the Ganglion's class): their rest
recording, a 5-s pause, then their mental-arithmetic recording. Nobody is imagining a hand
movement, so every command the engine emits is a false one. Decisions are scored once the
engine's 30-s settling period (§7.2) is over, by the recording their whole window lies in; a
window that straddles the pause belongs to neither.

**The measures.**

- *Commit rate* (primary): the share of scored decisions that are a command rather than HOLD,
  pooled over people. It's the same measure as the 10 Oct replay's 20% on BNCI2014-004.
- *False-command onsets per minute*: a command decision whose previous decision (scored or not)
  emitted something else. Median over people.

**Pass** (θ = 0.6): a pooled commit rate ≤ 10% at rest **and** ≤ 10% during arithmetic.

Also reported, not part of the bar: the left/right split of false commands, and θ = 0.7, 0.8 and
0.9 next to the same engine's commit rate and committed accuracy during real imagery
(BNCI2014-004's 9 never-seen people, first session, as the 10 Oct replay ran).
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from neuros.data.recording import Recording
from neuros.engine.events import HOLD, Event, IntentEvent
from neuros.engine.pipeline import Engine
from neuros.engine.policy import PolicySettings
from neuros.engine.runner import replay
from neuros.engine.simulator import Simulator
from neuros.errors import NeurOSConfigError, NeurOSDataError

# The 3-channel class this tests (the Ganglion's, BCI IV-2b's): all EEGMAT has of the layout.
IDLE_CHANNELS: tuple[str, ...] = ("C3", "Cz", "C4")
# EEGMAT's two recordings, in replay order.
SEGMENTS: tuple[str, ...] = ("rest", "arithmetic")
# The pause between recordings, as `neuros simulate` leaves between a session's runs.
GAP_S = 5.0
# θ values: the bar is judged at the shipping threshold; the others are reported.
THRESHOLDS: tuple[float, ...] = (0.6, 0.7, 0.8, 0.9)
BAR_THRESHOLD = 0.6
BAR_COMMIT_RATE = 0.10
# The model's classes -> the commands an app receives (as `neuros simulate` and `neuros run`).
COMMANDS: Mapping[str, str] = {"left_hand": "LEFT", "right_hand": "RIGHT"}

EngineFactory = Callable[[Sequence[str], float, PolicySettings], Engine]


def policy_for(classes: Sequence[str], threshold: float) -> PolicySettings:
    """The engine's policy at ``threshold`` (other settings at their defaults: 0.5-s smoothing,
    debounce 2), with every model class enabled, so any committed class counts as a command."""
    if not (math.isfinite(threshold) and 0.5 <= threshold <= 1.0):
        raise NeurOSConfigError(f"threshold must be in [0.5, 1], got {threshold}")
    return PolicySettings(
        commands={c: COMMANDS.get(c, c.upper()) for c in classes}, threshold=threshold
    )


def replay_spans(durations: Sequence[float], gap_s: float = GAP_S) -> list[tuple[float, float]]:
    """Each recording's ``[start, end]`` stream time when replayed back to back from t = 0,
    with ``gap_s`` of silence between them (``end`` is one sample past the last).

    Raises:
        NeurOSConfigError: on no recordings, a non-positive duration or a negative gap.
    """
    if not durations:
        raise NeurOSConfigError("no recordings to replay")
    if not (math.isfinite(gap_s) and gap_s >= 0):
        raise NeurOSConfigError(f"gap_s must be >= 0, got {gap_s}")
    spans: list[tuple[float, float]] = []
    start = 0.0
    for duration in durations:
        if not (math.isfinite(duration) and duration > 0):
            raise NeurOSConfigError(f"recording durations must be > 0, got {duration}")
        spans.append((start, start + duration))
        start += duration + gap_s
    return spans


def replay_recordings(
    new_engine: EngineFactory,
    recordings: Sequence[Recording],
    policy: PolicySettings,
    *,
    channels: Sequence[str] | None = None,
    gap_s: float = GAP_S,
    seed: int = 0,
) -> tuple[list[Event], list[Simulator], Engine]:
    """Replay ``recordings`` back to back (``gap_s`` apart) through a fresh engine, as fast as
    possible, streaming only ``channels`` (default: all of the first recording's).

    Returns:
        The engine's events, the simulators (their markers are the recordings' cues) and the
        engine (its config holds the settling time).

    Raises:
        NeurOSConfigError: on no recordings, or recordings that disagree on rate or channels.
    """
    if not recordings:
        raise NeurOSConfigError("no recordings to replay")
    first = recordings[0]
    stream = tuple(channels) if channels is not None else tuple(first.ch_names)
    for recording in recordings[1:]:
        if recording.sfreq != first.sfreq:
            raise NeurOSConfigError(
                f"{recording.person_id}: recordings at {first.sfreq:g} and "
                f"{recording.sfreq:g} Hz can't share one stream"
            )
    engine = new_engine(stream, first.sfreq, policy)
    simulators = []
    t0 = 0.0
    for recording in recordings:  # Simulator rejects a recording that lacks a stream channel
        simulator = Simulator(recording, channels=stream, t0=t0, seed=seed)
        simulators.append(simulator)
        t0 += simulator.duration_s + gap_s
    events = asyncio.run(replay(simulators, engine))
    return events, simulators, engine


@dataclass
class SegmentCount:
    """Scored decisions in one recording: how many, how many committed, and why the rest held."""

    decisions: int = 0
    commits: int = 0
    onsets: int = 0
    by_command: dict[str, int] = field(default_factory=dict)
    holds: dict[str, int] = field(default_factory=dict)

    @property
    def commit_rate(self) -> float:
        """Committed share of the scored decisions (NaN when nothing was scored)."""
        return self.commits / self.decisions if self.decisions else math.nan

    def onsets_per_minute(self, hop_s: float) -> float:
        """False-command onsets per minute of scored decisions (one decision per hop)."""
        minutes = self.decisions * hop_s / 60.0
        return self.onsets / minutes if minutes > 0 else math.nan

    def to_dict(self, hop_s: float) -> dict[str, Any]:
        return {
            "decisions": self.decisions,
            "commits": self.commits,
            "commit_rate": self.commit_rate,
            "onsets": self.onsets,
            "onsets_per_minute": self.onsets_per_minute(hop_s),
            "by_command": dict(sorted(self.by_command.items())),
            "holds": dict(sorted(self.holds.items())),
        }


def count_segments(
    events: Sequence[Event],
    spans: Sequence[tuple[float, float]],
    *,
    window_s: float,
    settle_s: float,
    commands: Sequence[str],
) -> dict[str, SegmentCount]:
    """Count each recording's scored decisions (see the module docs).

    A decision counts for a recording when its window ``[t - window_s, t]`` lies inside the
    recording's span and starts at or after ``settle_s``. An onset is a command whose previous
    decision, scored or not, emitted something else (HOLD or the other command).

    Raises:
        NeurOSConfigError: on spans that don't match ``SEGMENTS`` or a bad window or settle time.
    """
    if len(spans) != len(SEGMENTS):
        raise NeurOSConfigError(f"expected {len(SEGMENTS)} spans {SEGMENTS}, got {len(spans)}")
    if not (math.isfinite(window_s) and window_s > 0):
        raise NeurOSConfigError(f"window_s must be > 0, got {window_s}")
    if not (math.isfinite(settle_s) and settle_s >= 0):
        raise NeurOSConfigError(f"settle_s must be >= 0, got {settle_s}")
    enabled = set(commands)
    counts = {name: SegmentCount() for name in SEGMENTS}
    slack = 1e-9  # decision times are sample-quantised floats
    previous = HOLD
    for event in events:
        if not isinstance(event, IntentEvent):
            continue
        committed = event.command != HOLD
        if committed and event.command not in enabled:
            raise NeurOSDataError(f"the engine emitted {event.command!r}, not one of {commands}")
        onset = committed and event.command != previous
        previous = event.command
        start = event.t - window_s
        if start < settle_s - slack:
            continue
        segment = next(
            (
                name
                for name, (lo, hi) in zip(SEGMENTS, spans, strict=True)
                if start >= lo - slack and event.t <= hi + slack
            ),
            None,
        )
        if segment is None:
            continue  # straddles the pause
        count = counts[segment]
        count.decisions += 1
        if committed:
            count.commits += 1
            count.onsets += onset
            count.by_command[event.command] = count.by_command.get(event.command, 0) + 1
        else:
            reason = str(event.hold_reason)
            count.holds[reason] = count.holds.get(reason, 0) + 1
    return counts


def pooled(counts: Sequence[SegmentCount], hop_s: float) -> dict[str, Any]:
    """One segment pooled over people: the commit rate (the bar's measure), the median onsets
    per minute over people, and the split of false commands between commands.

    Raises:
        NeurOSDataError: if no decision was scored (nothing to judge).
    """
    decisions = sum(c.decisions for c in counts)
    if decisions == 0:
        raise NeurOSDataError("no decision was scored: check the spans and settling time")
    commits = sum(c.commits for c in counts)
    by_command: dict[str, int] = {}
    for count in counts:
        for command, n in count.by_command.items():
            by_command[command] = by_command.get(command, 0) + n
    rates = [c.onsets_per_minute(hop_s) for c in counts if c.decisions]
    return {
        "people": len(counts),
        "decisions": decisions,
        "commits": commits,
        "commit_rate": commits / decisions,
        "people_with_any_commit": sum(1 for c in counts if c.commits),
        "onsets_per_minute_median": float(np.median(rates)),
        "onsets_per_minute_max": float(np.max(rates)),
        "by_command": dict(sorted(by_command.items())),
    }


def verdict(
    rest_rate: float, arithmetic_rate: float, *, bar: float = BAR_COMMIT_RATE
) -> dict[str, Any]:
    """The bar, judged at θ = 0.6: both pooled commit rates at most ``bar``.

    Raises:
        NeurOSDataError: on a rate outside [0, 1] (NaN included).
    """
    for name, rate in (("rest", rest_rate), ("arithmetic", arithmetic_rate)):
        if not (math.isfinite(rate) and 0.0 <= rate <= 1.0):
            raise NeurOSDataError(f"the {name} commit rate must be in [0, 1], got {rate}")
    checks = {"rest": rest_rate <= bar, "arithmetic": arithmetic_rate <= bar}
    return {
        "bar": (
            f"pooled commit rate <= {bar:.0%} at rest and during arithmetic, theta {BAR_THRESHOLD}"
        ),
        "checks": checks,
        "outcome": "PASS" if all(checks.values()) else "FAIL",
    }
