"""Running the engine on a source, optionally serving its events to apps (§7.4-7.5).

``replay`` feeds one or more simulated runs (a session's runs in order, with the gaps between
them a real headset session would have) through an engine, as fast as possible or paced in
real time, and broadcasts every event when a hub is given. ``score_replay`` compares the
decisions with the ground-truth markers the engine never saw.

With a ``Calibrator`` (the optional guided calibration, D-040), both loops step it after every
chunk and pass it the apps' requests (``EventHub.take_requests``), so its cue events and the
switch to the fine-tuned model happen on the engine's clock, between decisions.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from neuros.engine.events import HOLD, Event, IntentEvent, StatusEvent
from neuros.engine.pipeline import Engine
from neuros.engine.server import EventHub
from neuros.engine.simulator import Marker, Simulator
from neuros.errors import NeurOSConfigError

if TYPE_CHECKING:
    from neuros.engine.calibration import Calibrator

# HOLD reasons whose probabilities still say where the model leans (the window was decoded).
_DECODED_HOLDS = ("low_confidence", "unrecognized")


def _calibration_events(
    calibrator: Calibrator | None, hub: EventHub | None, engine: Engine
) -> list[Event]:
    """The calibrator's due events, after handling any app requests (in arrival order)."""
    if calibrator is None:
        if hub is not None:
            hub.take_requests()  # nothing can act on them (the hub didn't advertise any)
        return []
    t = engine.last_t
    events: list[Event] = []
    if hub is not None and t is not None:
        for request in hub.take_requests():
            events.extend(calibrator.handle(request, t))
    events.extend(calibrator.step(t))
    return events


async def replay(
    simulators: Sequence[Simulator],
    engine: Engine,
    *,
    hub: EventHub | None = None,
    speed: float | None = None,
    on_event: Callable[[Event], None] | None = None,
    calibrator: Calibrator | None = None,
    calibrate_at_start: bool = False,
) -> list[Event]:
    """Feed every simulator's chunks to ``engine`` in order; return all events.

    Args:
        simulators: runs in time order (each with its own ``t0``; a jump in timestamps between
            runs is a gap the engine handles).
        speed: None = as fast as possible; 1.0 = real time (demo rehearsal).
        calibrator: steps after every chunk and takes the hub's requests.
        calibrate_at_start: start a calibration at the first sample (replay evaluation).

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

    if calibrator is not None and calibrate_at_start:
        await emit(list(calibrator.start(first_t)))
    for simulator in simulators:
        for samples, stamps in simulator.chunks():
            if speed is not None:
                due = (float(stamps[-1]) - first_t) / speed
                delay = due - (loop.time() - started)
                if delay > 0:
                    await asyncio.sleep(delay)
            await emit(engine.push(samples, stamps))
            await emit(_calibration_events(calibrator, hub, engine))
            if speed is None:
                await asyncio.sleep(0)  # let apps receive while replaying fast
    await emit(engine.close())
    return events


async def run_live(
    source: Any,
    engine: Engine,
    *,
    hub: EventHub | None = None,
    seconds: float | None = None,
    on_event: Callable[[Event], None] | None = None,
    calibrator: Calibrator | None = None,
) -> list[Event]:
    """Feed a live source (``sources.LslSource`` / ``BrainFlowSource``) to ``engine`` until
    ``seconds`` have passed (None: until cancelled, e.g. Ctrl+C); return the events.

    The source's blocking pulls run in a worker thread, so the WebSocket keeps serving apps.
    The source is closed and the engine told the stream ended, whatever stops the loop.

    Raises:
        NeurOSConfigError: on a non-positive ``seconds``.
    """
    if seconds is not None and not (math.isfinite(seconds) and seconds > 0):
        raise NeurOSConfigError(f"seconds must be None or > 0, got {seconds}")
    loop = asyncio.get_running_loop()
    iterator = source.chunks()
    events: list[Event] = []
    started = loop.time()

    async def emit(batch: list[Event]) -> None:
        for event in batch:
            events.append(event)
            if on_event is not None:
                on_event(event)
            if hub is not None:
                await hub.broadcast(event)

    try:
        while seconds is None or loop.time() - started < seconds:
            samples, stamps = await loop.run_in_executor(None, next, iterator)
            await emit(engine.push(samples, stamps))
            await emit(_calibration_events(calibrator, hub, engine))
    finally:
        await emit(engine.close())
        source.close()
    return events


@dataclass(frozen=True)
class ReplayScore:
    """How the decisions compare with the markers (see ``score_replay``)."""

    n_decisions: int
    n_commands: int
    holds: Mapping[str, int]
    task_decisions: int  # decisions (commands and HOLDs) whose window lies in a task period
    task_commands: int  # commands whose window lies in a task period
    task_correct: int
    per_class: Mapping[str, tuple[int, int]]  # class -> (correct, commands)
    # Where the model leaned (the argmax of the smoothed probabilities) on every decoded
    # task-period decision, committed or held for low confidence: the window-level accuracy
    # the offline evaluation measures, without the policy's threshold.
    argmax_per_class: Mapping[str, tuple[int, int]]  # class -> (correct, decisions)
    latency_median_ms: float
    latency_p99_ms: float
    states: tuple[tuple[float, str], ...]  # (t, state) at each change

    @property
    def commit_rate(self) -> float:
        """The share of task-period decisions that were commands rather than HOLD."""
        return self.task_commands / self.task_decisions if self.task_decisions else math.nan

    @property
    def task_accuracy(self) -> float:
        return self.task_correct / self.task_commands if self.task_commands else math.nan

    @property
    def balanced_task_accuracy(self) -> float:
        rates = [c / n for c, n in self.per_class.values() if n]
        return float(np.mean(rates)) if rates else math.nan

    @property
    def balanced_argmax_accuracy(self) -> float:
        rates = [c / n for c, n in self.argmax_per_class.values() if n]
        return float(np.mean(rates)) if rates else math.nan

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_decisions": self.n_decisions,
            "n_commands": self.n_commands,
            "holds": dict(self.holds),
            "task_decisions": self.task_decisions,
            "commit_rate": self.commit_rate,
            "task_commands": self.task_commands,
            "task_accuracy": self.task_accuracy,
            "balanced_task_accuracy": self.balanced_task_accuracy,
            "per_class": {k: list(v) for k, v in self.per_class.items()},
            "balanced_argmax_accuracy": self.balanced_argmax_accuracy,
            "argmax_per_class": {k: list(v) for k, v in self.argmax_per_class.items()},
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
    tolerance_s: float = 0.0,
) -> ReplayScore:
    """Score commands against the markers.

    A decision is scored when its window ``[t - window_s, t]`` lies within
    ``[cue + task_start_s - tolerance_s, cue + task_end_s + tolerance_s]`` of some marker: the
    task epoch of a trial (D-030's start), up to the end of the task. Decisions elsewhere (rest,
    between trials) are counted but not scored: there is no ground truth for them.

    ``tolerance_s`` should be half the engine's hop. Without it, a task period exactly one window
    long (the latest-window rule, for datasets whose class stays on screen) scores nothing: it
    needs a decision ending at one exact instant, and decisions come every hop at sample-quantized
    times (bug found 10 Oct: all 34 test-only replays scored 0 commands). With half a hop, the
    decision nearest that instant counts, as the offline evaluation's single window does.

    Raises:
        NeurOSConfigError: on a task period shorter than the window, or a tolerance that is
            negative or not shorter than the window.
    """
    if not task_end_s - task_start_s >= window_s:
        raise NeurOSConfigError(
            f"the task period [{task_start_s}, {task_end_s}] s is shorter than a "
            f"{window_s}-s window"
        )
    if not (math.isfinite(tolerance_s) and 0.0 <= tolerance_s < window_s):
        raise NeurOSConfigError(f"tolerance_s must be in [0, {window_s}), got {tolerance_s}")
    intents = [e for e in events if isinstance(e, IntentEvent)]
    holds: dict[str, int] = {}
    per_class: dict[str, list[int]] = {label: [0, 0] for label in commands}
    argmax: dict[str, list[int]] = {label: [0, 0] for label in commands}
    task_decisions = task_commands = task_correct = 0
    cues = np.array([m.t for m in markers])
    labels = [m.label for m in markers]
    slack = tolerance_s + 1e-9
    for event in intents:
        if event.command == HOLD:
            key = str(event.hold_reason)
            holds[key] = holds.get(key, 0) + 1
        start = event.t - window_s
        inside = np.flatnonzero(
            (start >= cues + task_start_s - slack) & (event.t <= cues + task_end_s + slack)
        )
        if inside.size == 0:
            continue
        label = labels[int(inside[-1])]
        if label not in commands:
            continue  # a class this app didn't enable
        task_decisions += 1
        if event.command != HOLD or event.hold_reason in _DECODED_HOLDS:
            leaning = max(commands.values(), key=lambda c: event.probs.get(c, -1.0))
            argmax[label][0] += leaning == commands[label]
            argmax[label][1] += 1
        if event.command == HOLD:
            continue
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
        task_decisions=task_decisions,
        task_commands=task_commands,
        task_correct=task_correct,
        per_class={k: (v[0], v[1]) for k, v in per_class.items()},
        argmax_per_class={k: (v[0], v[1]) for k, v in argmax.items()},
        latency_median_ms=float(np.median(latencies)),
        latency_p99_ms=float(np.percentile(latencies, 99)),
        states=states,
    )
