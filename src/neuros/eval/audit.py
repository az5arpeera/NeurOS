"""Cue-contamination audit (D-030): when, and where on the head, a cued dataset's class is readable.

Every cued motor-imagery dataset shows the class on screen (an arrow, a target, classifier
feedback). Cross-person models learn the brain's response to that display before they learn the
imagery, because the response looks alike in everyone (E0, 4 Oct: 75.0% with the arrow on
screen, 60.6% after it). Before a dataset trains or tests anything, this audit measures how much
of its class can be read from things that aren't motor imagery.

It reuses the zero-calibration Riemannian baseline, TS+LR with EA per recording session under
leave-one-person-out, so its numbers sit in the same tables as E0's.

Windows, in seconds from the cue's onset, all cut from the same preprocessed recordings:

| window | span | a score above chance means |
|---|---|---|
| ``pre_cue`` | [-0.5, 0] | a leak or wrong event timing: nothing about the class exists yet |
| ``cue_onset`` | [0, 0.5] | the brain's response to the cue itself, before imagery develops |
| ``task`` | the dataset's D-030 window | the class is readable where training happens |

Scalp regions (``neuros.data.montage.SCALP_REGIONS``): ``eyes``, ``motor``, ``parietal`` and
``visual``, plus ``all`` channels. A region with fewer than ``MIN_REGION_CHANNELS`` channels in
the dataset's montage is skipped, and the skip is reported.

Reading the findings (``audit_findings``):
- **leak**: the pre-cue window decodes above chance (its 95% CI over people excludes 50%) by at
  least ``LEAK_MARGIN``, in any region. Stop: labels or event timing are wrong.
- **warn**: a pre-cue effect whose CI excludes chance but which is smaller than ``LEAK_MARGIN``.
  With hundreds of trials per person, a 1–2-point effect passes the CI test from chance across
  the ~10 pre-cue cells, or from slow drifts; a timing or label bug shows up at the size of the
  task effect instead. Check timing and trial order when the dataset is used.
- **cue-dominated**: in the task window, the eyes or the visual cortex decode the class within
  ``NON_MOTOR_MARGIN`` of the motor strip, or better. Down-weight the dataset for training and
  never quote it without its controls.
- Volume conduction spreads every source across the scalp, so a region's score is evidence of
  where the class comes from, not proof.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from neuros.align.ea import AlignmentUnit
from neuros.config.experiment import apply_signal_overrides
from neuros.config.schemas import SignalContract, ValidationThresholds
from neuros.data.montage import SCALP_REGIONS, region_channels
from neuros.data.recording import Recording
from neuros.data.registry import DatasetSpec, TaskEpoch, task_epoch
from neuros.errors import NeurOSConfigError, NeurOSDataError
from neuros.eval.baselines import Alignment, Method, Reference, build_features, leave_one_person_out
from neuros.eval.metrics import PersonScore, summarize
from neuros.signal.windows import WindowSet, extract_epochs

log = logging.getLogger(__name__)

# The two fixed probe windows, in seconds from the cue's onset; "task" comes from D-030.
PROBE_WINDOWS: Mapping[str, tuple[float, float]] = MappingProxyType(
    {"pre_cue": (-0.5, 0.0), "cue_onset": (0.0, 0.5)}
)
WINDOW_ORDER: tuple[str, ...] = ("pre_cue", "cue_onset", "task")
REGION_ORDER: tuple[str, ...] = ("all", *SCALP_REGIONS)
AUDIT_METHOD: Method = "ts_lr"
AUDIT_ALIGNMENT: Alignment = "ea"
# A covariance needs at least two channels to carry spatial information.
MIN_REGION_CHANNELS = 2
# How close (in balanced accuracy) a non-motor region may come to the motor strip on the task
# window before the dataset counts as cue-dominated. Two points is about one seed's noise in E0.
NON_MOTOR_MARGIN = 0.02
# How far above chance a pre-cue effect must be to count as a leak (see "warn" above). Set after
# BNCI2014-004 (4 Oct): pre-cue 51.4%, CI [50.7, 52.2], 720 trials per person, no trial-order
# structure (classes switch on 50.5% of consecutive trials); its cue-onset window was at 50.6%.
LEAK_MARGIN = 0.03
CHANCE = 0.5


@dataclass(frozen=True)
class AuditWindow:
    """One audit window: its epoch, and a contract whose window length matches it."""

    name: str
    epoch: TaskEpoch
    contract: SignalContract


def audit_windows(spec: DatasetSpec, contract: SignalContract) -> tuple[AuditWindow, ...]:
    """The audit's windows for ``spec``: the two probes, then the dataset's task window.

    Raises:
        NeurOSConfigError: for a dataset without class cues, or a window its task can't host.
    """
    if spec.class_on_screen_s is None:
        raise NeurOSConfigError(f"{spec.key}: the cue audit is for cued motor-imagery datasets")
    windows: list[AuditWindow] = []
    for name, (start, end) in PROBE_WINDOWS.items():
        # Rounded so the window length is exact (0.5 s = 64 samples at 128 Hz).
        probe = apply_signal_overrides(
            contract, {"epoch_s": [start, end], "window_s": round(end - start, 9)}
        )
        windows.append(AuditWindow(name, task_epoch(spec, probe), probe))
    windows.append(AuditWindow("task", task_epoch(spec, contract), contract))
    return tuple(windows)


def cut_audit_windows(
    recordings: Sequence[Recording],
    windows: Sequence[AuditWindow],
    thresholds: ValidationThresholds,
    labels: Sequence[str],
) -> tuple[dict[str, WindowSet], dict[str, int]]:
    """Cut every audit window from the same preprocessed recordings.

    Returns:
        ``(window sets by name, events skipped by name)``. Skipped events are those whose
        window runs outside the recording; they're reported, never silently lost.

    Raises:
        NeurOSDataError: if there are no recordings, or a window keeps no event at all.
    """
    if not recordings:
        raise NeurOSDataError("no recordings to cut audit windows from")
    parts: dict[str, list[WindowSet]] = {w.name: [] for w in windows}
    skipped: dict[str, int] = {w.name: 0 for w in windows}
    for recording in recordings:
        for window in windows:
            cut, n_skipped = extract_epochs(
                recording,
                window.contract,
                thresholds,
                labels=labels,
                epoch_s=(window.epoch.start_s, window.epoch.end_s),
            )
            parts[window.name].append(cut)
            skipped[window.name] += n_skipped
    merged = {name: WindowSet.concat(sets) for name, sets in parts.items()}
    empty = [name for name, ws in merged.items() if len(ws) == 0]
    if empty:
        raise NeurOSDataError(f"audit windows {empty} kept no events", recordings[0].context)
    return merged, skipped


def _cell(
    window: str,
    epoch: TaskEpoch,
    region: str,
    channels: Sequence[str],
    scores: Sequence[PersonScore],
) -> dict[str, Any]:
    return {
        "window": window,
        **epoch.to_dict(),
        "region": region,
        "channels": list(channels),
        "summary_balanced_accuracy": summarize([s.balanced_accuracy for s in scores]),
        "summary_kappa": summarize([s.kappa for s in scores]),
        "per_person": [s.to_dict() for s in scores],
    }


def run_audit(
    window_sets: Mapping[str, WindowSet],
    epochs: Mapping[str, TaskEpoch],
    *,
    reference: Reference = "as_recorded",
    unit: AlignmentUnit = "session",
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Score every (window × region) cell with TS+LR + EA, leave-one-person-out.

    Returns a JSON-serialisable dict: the cells (summaries and per-person scores), the skipped
    cells with their reason, and the plain-language findings.

    Raises:
        NeurOSConfigError: if a window has no epoch, or the window sets disagree on channels.
        NeurOSDataError: from the baseline (e.g. fewer than 2 people).
    """
    say = progress or (lambda message: log.info("%s", message))
    if set(window_sets) - set(epochs):
        raise NeurOSConfigError(f"no epoch for windows {sorted(set(window_sets) - set(epochs))}")
    channel_sets = {ws.ch_names for ws in window_sets.values()}
    if len(channel_sets) != 1:
        raise NeurOSConfigError("audit window sets must share one channel list")
    cells: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    ordered = [name for name in WINDOW_ORDER if name in window_sets]
    ordered += sorted(set(window_sets) - set(WINDOW_ORDER))
    for name in ordered:
        windows = window_sets[name]
        for region in REGION_ORDER:
            channels = (
                windows.ch_names if region == "all" else region_channels(region, windows.ch_names)
            )
            if len(channels) < MIN_REGION_CHANNELS:
                skipped.append(
                    {
                        "window": name,
                        "region": region,
                        "channels": list(channels),
                        "reason": f"{len(channels)} channel(s) in this montage; "
                        f"needs >= {MIN_REGION_CHANNELS}",
                    }
                )
                continue
            subset = windows if region == "all" else windows.pick_channels(channels)
            say(f"{name} / {region} ({len(channels)} channels) ...")
            features = build_features(
                subset, alignment=AUDIT_ALIGNMENT, reference=reference, unit=unit
            )
            scores = leave_one_person_out(AUDIT_METHOD, features)
            cells.append(_cell(name, epochs[name], region, channels, scores))
    return {
        "method": f"{AUDIT_METHOD}+{AUDIT_ALIGNMENT}",
        "protocol": "loso",
        "reference": reference,
        "alignment_unit": unit,
        "min_region_channels": MIN_REGION_CHANNELS,
        "non_motor_margin": NON_MOTOR_MARGIN,
        "cells": cells,
        "skipped_cells": skipped,
        "findings": audit_findings(cells),
    }


def audit_findings(cells: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Plain-language findings from the scored cells (rules in the module docstring).

    Each finding has a ``level`` ("leak", "warn", "cue", "ok" or "info") and a ``message``.
    """

    def mean(cell: Mapping[str, Any]) -> float:
        return float(cell["summary_balanced_accuracy"]["mean"])

    by_key = {(c["window"], c["region"]): c for c in cells}
    findings: list[dict[str, str]] = []

    above_chance = [
        c
        for c in cells
        if c["window"] == "pre_cue" and c["summary_balanced_accuracy"]["ci95_low"] > CHANCE
    ]
    for cell in above_chance:
        low = cell["summary_balanced_accuracy"]["ci95_low"]
        if mean(cell) >= CHANCE + LEAK_MARGIN:
            findings.append(
                {
                    "level": "leak",
                    "message": f"the pre-cue window decodes at {mean(cell):.1%} from "
                    f"{cell['region']} (CI from {low:.1%}): check labels and event timing "
                    "before anything else",
                }
            )
        else:
            findings.append(
                {
                    "level": "warn",
                    "message": f"a small pre-cue effect from {cell['region']}: {mean(cell):.1%} "
                    f"(CI from {low:.1%}), below the {LEAK_MARGIN:.0%} leak threshold; check "
                    "timing and trial order when this dataset is used",
                }
            )
    if not above_chance and any(c["window"] == "pre_cue" for c in cells):
        findings.append({"level": "ok", "message": "the pre-cue window is at chance everywhere"})

    onset = [c for c in cells if c["window"] == "cue_onset" and c["region"] != "all"]
    if onset:
        best = max(onset, key=mean)
        findings.append(
            {
                "level": "info",
                "message": f"the cue's onset decodes at up to {mean(best):.1%} "
                f"(best region: {best['region']})",
            }
        )

    motor = by_key.get(("task", "motor"))
    if motor is not None:
        rivals = [by_key[("task", r)] for r in ("eyes", "visual") if ("task", r) in by_key]
        strong = [c for c in rivals if mean(c) >= mean(motor) - NON_MOTOR_MARGIN]
        for cell in strong:
            findings.append(
                {
                    "level": "cue",
                    "message": f"cue-dominated: {cell['region']} decodes the task window at "
                    f"{mean(cell):.1%}, vs {mean(motor):.1%} for the motor strip",
                }
            )
        if rivals and not strong:
            others = ", ".join(f"{c['region']} {mean(c):.1%}" for c in rivals)
            findings.append(
                {
                    "level": "ok",
                    "message": f"the motor strip leads the task window: {mean(motor):.1%} "
                    f"(vs {others})",
                }
            )
    task = next((c for c in cells if c["window"] == "task"), None)
    if task is not None and not task["cue_free"]:
        findings.append(
            {
                "level": "info",
                "message": "the class is on screen during the task window (D-030 flag): "
                "every result on it needs the cue-only control",
            }
        )
    return findings
