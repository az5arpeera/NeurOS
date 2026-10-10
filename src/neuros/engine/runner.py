"""Running the engine on a source, optionally serving its events to apps (§7.4-7.5).

``replay`` feeds one or more simulated runs (a session's runs in order, with the gaps between
them a real headset session would have) through an engine, as fast as possible or paced in
real time, and broadcasts every event when a hub is given. ``score_replay`` compares the
decisions with the ground-truth markers the engine never saw.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from neuros.engine.events import HOLD, Event, IntentEvent, StatusEvent
from neuros.engine.pipeline import Engine
from neuros.engine.server import EventHub
from neuros.engine.simulator import Marker, Simulator
from neuros.errors import NeurOSConfigError


async def replay(
    simulators: Sequence[Simulator],
    engine: Engine,
    *,
    hub: EventHub | None = None,
    speed: float | None = None,
    on_event: Callable[[Event], None] | None = None,
) -> list[Event]:
    """Feed every simulator's chunks to ``engine`` in order; return all events.

    Args:
        simulators: runs in time order (each with its own ``t0``; a jump in timestamps between
            runs is a gap the engine handles).
        speed: None = as fast as possible; 1.0 = real time (demo rehearsal).

    Raises:
        NeurOSConfigError: on no simulators or a non-positive speed.
    """
    if not simulators:
        raise NeurOSConfigError("nothing to replay")
    if speed is not None and not (math.isfinite(speed) and speed > 0):
        raise NeurOSConfigError(f"speed must be None or > 0, got {speed}")
    loop = asyncio.get_running_loop()
    started = loop.time()
    first_t = float(simulators[0].timestamps[0])
    events: list[Event] = []

    async def emit(batch: list[Event]) -> None:
        for event in batch:
            events.append(event)
            if on_event is not None:
                on_event(event)
            if hub is not None:
                await hub.broadcast(event)

    for simulator in simulators:
        for samples, stamps in simulator.chunks():
            if speed is not None:
                due = (float(stamps[-1]) - first_t) / speed
                delay = due - (loop.time() - started)
                if delay > 0:
                    await asyncio.sleep(delay)
            await emit(engine.push(samples, stamps))
            if speed is None:
                await asyncio.sleep(0)  # let apps receive while replaying fast
    await emit(engine.close())
    return events


@dataclass(frozen=True)
class ReplayScore:
    """How the decisions compare with the markers (see ``score_replay``)."""

    n_decisions: int
    n_commands: int
    holds: Mapping[str, int]
    task_commands: int  # commands whose window lies in a cue-free task period
    task_correct: int
    per_class: Mapping[str, tuple[int, int]]  # class -> (correct, commands)
    latency_median_ms: float
    latency_p99_ms: float
    states: tuple[tuple[float, str], ...]  # (t, state) at each change

    @property
    def task_accuracy(self) -> float:
        return self.task_correct / self.task_commands if self.task_commands else math.nan

    @property
    def balanced_task_accuracy(self) -> float:
        rates = [c / n for c, n in self.per_class.values() if n]
        return float(np.mean(rates)) if rates else math.nan

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_decisions": self.n_decisions,
            "n_commands": self.n_commands,
            "holds": dict(self.holds),
            "task_commands": self.task_commands,
            "task_accuracy": self.task_accuracy,
            "balanced_task_accuracy": self.balanced_task_accuracy,
            "per_class": {k: list(v) for k, v in self.per_class.items()},
            "latency_median_ms": self.latency_median_ms,
            "latency_p99_ms": self.latency_p99_ms,
            "states": [list(s) for s in self.states],
        }


def score_replay(
    events: Sequence[Event],
    markers: Sequence[Marker],
    commands: Mapping[str, str],
    *,
    window_s: float,
    task_start_s: float,
    task_end_s: float,
) -> ReplayScore:
    """Score commands against the markers.

    A command counts when its decision window ``[t - window_s, t]`` lies within
    ``[cue + task_start_s, cue + task_end_s]`` of some marker: the cue-free part of a trial
    (D-030's start), up to the end of the task. Commands elsewhere (rest, between trials) are
    counted but not scored: there is no ground truth for them.

    Raises:
        NeurOSConfigError: on a task period shorter than the window.
    """
    if not task_end_s - task_start_s >= window_s:
        raise NeurOSConfigError(
            f"the task period [{task_start_s}, {task_end_s}] s is shorter than a "
            f"{window_s}-s window"
        )
    intents = [e for e in events if isinstance(e, IntentEvent)]
    holds: dict[str, int] = {}
    per_class: dict[str, list[int]] = {label: [0, 0] for label in commands}
    task_commands = task_correct = 0
    cues = np.array([m.t for m in markers])
    labels = [m.label for m in markers]
    for event in intents:
        if event.command == HOLD:
            key = str(event.hold_reason)
            holds[key] = holds.get(key, 0) + 1
            continue
        start = event.t - window_s
        inside = np.flatnonzero(
            (start >= cues + task_start_s - 1e-9) & (event.t <= cues + task_end_s + 1e-9)
        )
        if inside.size == 0:
            continue
        label = labels[int(inside[-1])]
        if label not in commands:
            continue  # a class this app didn't enable
        task_commands += 1
        right = event.command == commands[label]
        task_correct += right
        per_class[label][0] += right
        per_class[label][1] += 1
    latencies = np.array([e.latency_ms for e in intents]) if intents else np.zeros(1)
    states = tuple((e.t, e.state) for e in events if isinstance(e, StatusEvent))
    return ReplayScore(
        n_decisions=len(intents),
        n_commands=sum(1 for e in intents if e.command != HOLD),
        holds=holds,
        task_commands=task_commands,
        task_correct=task_correct,
        per_class={k: (v[0], v[1]) for k, v in per_class.items()},
        latency_median_ms=float(np.median(latencies)),
        latency_p99_ms=float(np.percentile(latencies, 99)),
        states=states,
    )
