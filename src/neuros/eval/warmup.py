"""E2, the warm-up curve (architecture §5 "online", §8.5 E2; sets S in the §8.6 headline).

**Question.** A new user puts on a headset. The online engine aligns their EEG with a reference
built from what it has seen so far, shrunk toward a population prior (§5):

    R̂(t) = (n₀·R_pop + n_t·R_user(t)) / (n₀ + n_t)

How many seconds t until the pooled model reaches the accuracy it has with the full-session
reference (E1's offline per-session EA, the upper bound)?

**Replay protocol.** Offline, on the cached continuous recordings; no labels are used anywhere
in building a reference.
- **Stream** (what the engine does): R_user(t) is the mean covariance of the session's
  ``stream_window_s`` windows, sliding by ``stream_hop_s``, that end within the first t seconds
  of the session. Runs play in order and only recorded time counts. A quality gate drops windows
  with non-finite samples or any channel's peak-to-peak above ``max_ptp_uv``, and windows that
  start inside a run's filter start-up (``TRANSIENT_S``).
- **Rest (H1):** the same, but only windows wholly inside a pre-cue fixation (the registry's
  ``pre_cue_fixation_s`` before each cue). That is, seeding from relaxed rest instead of
  everything.
- **Prior:** R_pop for a dataset's channel profile is the mean, over the fold's training people
  (every training dataset that holds all those channels), of each person's mean task-window
  covariance on the same channels with the same common average. At t = 0 the reference is the
  prior alone. n₀ is ``prior_s`` worth of stream windows.
- **Scored windows:** each test person's task windows that start after the largest budget. They
  are the same windows for every budget and for the full-session reference, so the curve is
  paired person by person.
- **Placement:** each scored window is placed in the layout exactly as E1 places it (derived
  channels, common average over the dataset's present layout channels, zeros elsewhere), but
  aligned with the budget's reference instead of the offline one.

The windows a person is scored on never enter their own reference at any budget (they start
after the largest budget), and a person's prior never includes them (it comes from other
people's training data).

**E5, virtual headsets** (``knowledge-base/plan/e5-virtual-headsets.md``). With
``channel_sets`` set, the same scored windows are also cut to each headset class's channels:
common average over the set, the stream reference after ``channel_set_budget_s`` built from the
same channels (and a prior on them), zeros elsewhere: exactly how the live engine places a
headset with fewer channels. A set is scored on every warm-up person whose dataset has all of
its channels; ``summarize_channel_sets`` labels each set against the research cap (the dataset's
own full profile at the same budget).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from neuros.align.covariance import window_covariances
from neuros.align.ea import (
    AlignmentReference,
    apply_alignment,
    euclidean_reference,
    reference_from_mean,
)
from neuros.config.experiment import WarmupConfig
from neuros.data.recording import Recording
from neuros.data.registry import DatasetSpec
from neuros.errors import DataContext, NeurOSConfigError, NeurOSDataError
from neuros.eval.pooled import PooledWindows, derive_rows
from neuros.signal.filters import TRANSIENT_S
from neuros.signal.reference import common_average_reference
from neuros.signal.windows import WindowSet

Progress = Callable[[str], None]

# Windows per chunk when sliding over a continuous recording (bounded memory).
_STREAM_CHUNK = 2048
# Floating-point slack when comparing a window's end with a budget, in seconds.
_EPS_S = 1e-9


def kind_name(mode: str, budget_s: float) -> str:
    """The data-kind name of one warm-up set: ``e2_prior``, ``e2_full``, ``e2_stream_30s``,
    ``e2_stream_all`` (an infinite budget: the whole session, the mode's asymptote)."""
    if mode in ("prior", "full"):
        return f"e2_{mode}"
    if mode not in ("stream", "rest"):
        raise NeurOSConfigError(f"unknown warm-up mode {mode!r} (prior, full, stream, rest)")
    if math.isinf(budget_s):
        return f"e2_{mode}_all"
    if float(budget_s).is_integer():
        label = f"{int(budget_s)}"
    else:
        label = f"{budget_s:g}".replace(".", "p")
    return f"e2_{mode}_{label}s"


def channel_set_kind(name: str, budget_s: float) -> str:
    """The data-kind name of one E5 channel set at a stream budget: ``e5_three_120s``."""
    return "e5_" + name + "_" + kind_name("stream", budget_s).removeprefix("e2_stream_")


def channel_set_kinds(settings: WarmupConfig) -> list[tuple[str, str, tuple[str, ...]]]:
    """Every E5 channel set as ``(kind, name, channels)``, in the config's order."""
    budget = settings.channel_set_budget_s
    return [
        (channel_set_kind(name, budget), name, tuple(channels))
        for name, channels in settings.channel_sets.items()
    ]


def warmup_kinds(settings: WarmupConfig) -> list[tuple[str, str, float]]:
    """Every warm-up set as ``(kind, mode, budget_s)``, in report order: prior; the stream
    budgets and the whole-session stream; the same for rest (if H1 runs); then E1's
    full-session reference.

    The "all" sets are each mode's asymptote. E1 (so the trained model) aligns with the mean
    covariance of *task windows*, while the engine averages everything it streams (breaks,
    cues, rest), a different statistic: the gap between ``e2_stream_all`` and ``e2_full``
    is what no amount of warm-up time closes, so it is reported apart from the time curve.
    """
    kinds = [(kind_name("prior", 0.0), "prior", 0.0)]
    modes = ["stream", "rest"] if settings.rest_seeded else ["stream"]
    for mode in modes:
        kinds += [(kind_name(mode, b), mode, b) for b in settings.budgets_s if b > 0]
        kinds.append((kind_name(mode, math.inf), mode, math.inf))
    kinds.append((kind_name("full", math.inf), "full", math.inf))
    return kinds


def profile_channels(layout: Sequence[str], spec: DatasetSpec) -> tuple[str, ...]:
    """The layout rows a dataset fills (its model channels, in layout order).

    Raises:
        NeurOSDataError: if fewer than 2 (no common average or covariance is possible).
    """
    available = set(spec.model_channels)
    have = tuple(ch for ch in layout if ch in available)
    if len(have) < 2:
        raise NeurOSDataError(f"{spec.key}: only {list(have)} of the layout's channels")
    return have


def _run_sort_key(run: str) -> tuple[int, int | str]:
    """Runs in recording order: numeric run names numerically, others alphabetically after."""
    return (0, int(run)) if run.isdigit() else (1, run)


@dataclass(frozen=True)
class SessionStream:
    """One session's reference windows (one profile's channels, common-averaged).

    ``covariances``/``end_s`` are the engine's stream: windows on a fixed hop grid over the
    whole session. ``rest_covariances``/``rest_end_s`` are H1's: windows anchored to each
    pre-cue fixation (ending at the cue, then stepping back by the hop while they still fit),
    since a 2-s fixation fits a 2-s window only when the window lines up with it.

    The prior is weighed against the user's data in *stream windows* (n₀ = ``prior_s`` / hop).
    Stream windows overlap (one per hop), rest windows of a short fixation don't, so rest data
    count by the seconds they cover: ``rest_covered_s`` is the distinct recorded time the rest
    windows cover up to each one, in the same order.
    """

    covariances: NDArray[np.float64]  # [n, ch, ch]
    end_s: NDArray[np.float64]  # session time at each window's end
    rest_covariances: NDArray[np.float64]  # [m, ch, ch]
    rest_end_s: NDArray[np.float64]
    rest_covered_s: NDArray[np.float64]  # cumulative distinct seconds covered, per rest window
    hop_s: float  # the stream's hop: one stream window per hop_s of data
    run_offsets_s: Mapping[str, float]  # session time at each run's first sample
    n_rejected: int  # windows the quality gate dropped (both kinds)

    def __post_init__(self) -> None:
        if self.covariances.shape[0] != len(self.end_s):
            raise NeurOSDataError("stream covariances and end times disagree in length")
        n_rest = len(self.rest_end_s)
        if self.rest_covariances.shape[0] != n_rest or len(self.rest_covered_s) != n_rest:
            raise NeurOSDataError("rest covariances, end times and coverage disagree in length")
        if not self.hop_s > 0:
            raise NeurOSDataError(f"hop_s must be > 0, got {self.hop_s}")


def _covered_seconds(starts: NDArray[np.int64], width: int, sfreq: float) -> NDArray[np.float64]:
    """Cumulative distinct time covered by windows ``[s, s + width)`` taken in order (sorted
    starts), in seconds: the union, so overlapping windows don't count twice."""
    covered = np.zeros(starts.size, dtype=np.float64)
    reach = -1  # one past the last covered sample
    total = 0
    for i, start in enumerate(starts.tolist()):
        total += max(0, start + width - max(start, reach))
        reach = max(reach, start + width)
        covered[i] = total / sfreq
    return covered


def _profile_rows(
    data: NDArray[np.floating],
    ch_names: Sequence[str],
    spec: DatasetSpec,
    profile: Sequence[str],
    context: DataContext,
) -> NDArray[np.float32]:
    """Derived channels, then the profile's rows in profile order, then the common average."""
    derived, names = derive_rows(data, ch_names, spec.derived_channels)
    missing = [ch for ch in profile if ch not in names]
    if missing:
        raise NeurOSDataError(f"profile channels {missing} are missing", context)
    picked = np.take(derived, [names.index(ch) for ch in profile], axis=-2)
    return np.asarray(common_average_reference(picked), dtype=np.float32)


def _gated_covariances(
    x: NDArray[np.float32], starts: NDArray[np.int64], width: int, max_ptp_uv: float
) -> tuple[NDArray[np.float64], NDArray[np.int64], int]:
    """Covariances of the windows ``x[:, s:s+width]`` that pass the quality gate (finite, every
    channel's peak-to-peak within ``max_ptp_uv``): ``(covariances, kept starts, n rejected)``."""
    n_ch = x.shape[0]
    empty = (np.zeros((0, n_ch, n_ch)), np.zeros(0, dtype=np.int64))
    if starts.size == 0:
        return empty[0], empty[1], 0
    view = np.lib.stride_tricks.sliding_window_view(x, width, axis=-1)
    covs: list[NDArray[np.float64]] = []
    kept: list[NDArray[np.int64]] = []
    rejected = 0
    for first in range(0, starts.size, _STREAM_CHUNK):
        chunk = starts[first : first + _STREAM_CHUNK]
        windows = np.ascontiguousarray(np.moveaxis(view[:, chunk, :], 1, 0))
        finite = np.isfinite(windows).all(axis=(1, 2))
        spread = np.ptp(np.nan_to_num(windows), axis=-1).max(axis=1)
        good = finite & (np.where(finite, spread, np.inf) <= max_ptp_uv)
        rejected += int((~good).sum())
        if good.any():
            covs.append(window_covariances(windows[good]))
            kept.append(chunk[good])
    if not covs:
        return empty[0], empty[1], rejected
    return np.concatenate(covs), np.concatenate(kept), rejected


def _rest_starts(
    onsets: Sequence[int], fixation: int, width: int, hop: int, first: int, n_samples: int
) -> NDArray[np.int64]:
    """Starts of the windows wholly inside a pre-cue fixation: each ends at the cue, then steps
    back by ``hop`` while it still fits (and starts at or after ``first``)."""
    starts: set[int] = set()
    for onset in onsets:
        start = onset - width
        while start >= onset - fixation and start >= first:
            if start + width <= n_samples:
                starts.add(start)
            start -= hop
    return np.array(sorted(starts), dtype=np.int64)


def session_streams(
    recordings: Sequence[Recording],
    spec: DatasetSpec,
    profile: Sequence[str],
    settings: WarmupConfig,
    *,
    max_ptp_uv: float,
) -> dict[str, SessionStream]:
    """Each session's reference windows, runs in order, with session times.

    Raises:
        NeurOSDataError: on recordings from different people or rates, or a session with no
            stream window passing the gate.
        NeurOSConfigError: on a non-positive ``max_ptp_uv``.
    """
    if not recordings:
        raise NeurOSDataError(f"{spec.key}: no recordings to stream")
    if not (math.isfinite(max_ptp_uv) and max_ptp_uv > 0):
        raise NeurOSConfigError(f"max_ptp_uv must be positive, got {max_ptp_uv}")
    if len({(r.dataset, r.subject) for r in recordings}) != 1:
        raise NeurOSDataError("session_streams needs one person's recordings")
    rates = {r.sfreq for r in recordings}
    if len(rates) != 1:
        raise NeurOSDataError(f"recordings differ in sampling rate: {sorted(rates)}")
    sfreq = float(next(iter(rates)))
    width = round(settings.stream_window_s * sfreq)
    hop = max(1, round(settings.stream_hop_s * sfreq))
    transient = math.ceil(TRANSIENT_S * sfreq)
    fixation = 0 if spec.pre_cue_fixation_s is None else round(spec.pre_cue_fixation_s * sfreq)
    by_session: dict[str, list[Recording]] = {}
    for recording in recordings:
        by_session.setdefault(recording.session, []).append(recording)

    n_ch = len(profile)
    streams: dict[str, SessionStream] = {}
    for session, runs in sorted(by_session.items()):
        runs = sorted(runs, key=lambda r: _run_sort_key(r.run))
        covs: list[NDArray[np.float64]] = []
        ends: list[NDArray[np.float64]] = []
        rest_covs: list[NDArray[np.float64]] = []
        rest_ends: list[NDArray[np.float64]] = []
        rest_covered: list[NDArray[np.float64]] = []
        covered_so_far = 0.0  # distinct rest seconds in the session's earlier runs
        offsets: dict[str, float] = {}
        offset_s = 0.0
        rejected = 0
        for recording in runs:
            context = DataContext(spec.key, recording.subject, session, recording.run)
            x = _profile_rows(recording.data, recording.ch_names, spec, profile, context)
            n_samples = x.shape[-1]
            offsets[recording.run] = offset_s
            grid = np.arange(transient, n_samples - width + 1, hop, dtype=np.int64)
            c, kept, r = _gated_covariances(x, grid, width, max_ptp_uv)
            covs.append(c)
            ends.append(offset_s + (kept + width) / sfreq)
            rejected += r
            if fixation >= width:
                anchored = _rest_starts(
                    recording.events.onsets.tolist(), fixation, width, hop, transient, n_samples
                )
                c, kept, r = _gated_covariances(x, anchored, width, max_ptp_uv)
                rest_covs.append(c)
                rest_ends.append(offset_s + (kept + width) / sfreq)
                covered = covered_so_far + _covered_seconds(kept, width, sfreq)
                rest_covered.append(covered)
                covered_so_far = float(covered[-1]) if covered.size else covered_so_far
                rejected += r
            offset_s += n_samples / sfreq
        no_rest = np.zeros((0, n_ch, n_ch))
        stream = SessionStream(
            covariances=np.concatenate(covs),
            end_s=np.concatenate(ends),
            rest_covariances=np.concatenate(rest_covs) if rest_covs else no_rest,
            rest_end_s=np.concatenate(rest_ends) if rest_ends else np.zeros(0),
            rest_covered_s=np.concatenate(rest_covered) if rest_covered else np.zeros(0),
            hop_s=hop / sfreq,
            run_offsets_s=offsets,
            n_rejected=rejected,
        )
        if stream.end_s.size == 0:
            raise NeurOSDataError(
                f"{spec.key}: no reference window passed the quality gate",
                DataContext(spec.key, runs[0].subject, session),
            )
        streams[session] = stream
    return streams


def blended_reference(
    stream: SessionStream,
    budget_s: float,
    prior: NDArray[np.float64],
    prior_windows: float,
    *,
    mode: str = "stream",
    context: DataContext | None = None,
) -> AlignmentReference:
    """R̂(t) = (n₀·R_pop + n_t·R_user(t)) / (n₀ + n_t) from the windows ending by ``budget_s``.

    n₀ (``prior_windows``) and n_t are in stream windows: for the stream, n_t counts its
    windows; for rest, n_t is the distinct seconds the rest windows cover divided by the hop,
    so a second of rest weighs what a second of streamed data does.

    Raises:
        NeurOSConfigError: on an unknown mode, a negative budget or a non-positive n₀.
        NeurOSDataError: if the prior's shape doesn't match the stream's channels.
    """
    if mode not in ("stream", "rest"):
        raise NeurOSConfigError(f"unknown seeding mode {mode!r} (stream, rest)")
    if not budget_s >= 0:  # inf = every window of the session
        raise NeurOSConfigError(f"budget_s must be >= 0, got {budget_s}")
    if not prior_windows > 0:
        raise NeurOSConfigError(f"prior_windows must be > 0, got {prior_windows}")
    n_ch = stream.covariances.shape[1]
    if prior.shape != (n_ch, n_ch):
        raise NeurOSDataError(f"prior is {prior.shape}, the stream has {n_ch} channels", context)
    if mode == "rest":
        covariances, end_s = stream.rest_covariances, stream.rest_end_s
    else:
        covariances, end_s = stream.covariances, stream.end_s
    used = end_s <= budget_s + _EPS_S
    n_used = int(used.sum())
    if n_used == 0:
        return reference_from_mean(prior, 0, context=context)
    user = covariances[used].mean(axis=0)
    if mode == "rest":  # windows are in time order, so coverage is the last used one's
        weight = float(stream.rest_covered_s[np.flatnonzero(used)[-1]]) / stream.hop_s
    else:
        weight = float(n_used)
    mean = (prior_windows * prior + weight * user) / (prior_windows + weight)
    return reference_from_mean(mean, n_used, context=context)


def person_mean_covariances(
    windows: WindowSet, spec: DatasetSpec, profile: Sequence[str]
) -> dict[str, NDArray[np.float64]]:
    """Each person's mean task-window covariance on ``profile`` (derived channels applied,
    common average over the profile), or {} if the dataset lacks a profile channel."""
    if not set(profile) <= set(spec.model_channels):
        return {}
    context = DataContext(spec.key)
    x = _profile_rows(windows.data, windows.ch_names, spec, profile, context)
    covs = window_covariances(x)
    people = np.array(windows.person_ids)
    return {
        person: covs[people == person].mean(axis=0) for person in dict.fromkeys(people.tolist())
    }


@dataclass
class _SetParts:
    """One channel set's placed windows and their metadata, collected person by person."""

    x: list[NDArray[np.float32]]
    present: list[NDArray[np.bool_]]
    labels: list[str]
    people: list[str]
    datasets: list[str]
    flags: list[NDArray[np.uint8]]

    @classmethod
    def empty(cls) -> _SetParts:
        return cls([], [], [], [], [], [])

    def stack(self, layout: Sequence[str], sfreq: float) -> PooledWindows:
        return PooledWindows(
            x=np.concatenate(self.x),
            present=np.concatenate(self.present),
            labels=np.array(self.labels, dtype=str),
            people=np.array(self.people, dtype=str),
            datasets=np.array(self.datasets, dtype=str),
            flags=np.concatenate(self.flags).astype(np.uint8),
            layout=tuple(layout),
            sfreq=sfreq,
        )


def build_warmup_sets(
    task_windows: Mapping[str, WindowSet],
    layout: Sequence[str],
    *,
    specs: Mapping[str, DatasetSpec],
    folds: Sequence[frozenset[str]],
    trainable: Sequence[str],
    settings: WarmupConfig,
    load_recordings: Callable[[str, int], Sequence[Recording]],
    max_ptp_uv: float,
    progress: Progress | None = None,
) -> tuple[dict[str, PooledWindows], dict[str, Any]]:
    """Every warm-up set (``warmup_kinds``) for the warm-up datasets' people, plus a report.

    Args:
        task_windows: each dataset's development task windows, unaligned (as ``_e1_windows``).
        layout: the model's input channels.
        specs: the registry spec of every dataset in ``task_windows``.
        folds: the development folds (person ids), as ``plan_tasks`` draws them.
        trainable: the datasets whose people train (and so form priors).
        settings: the experiment's warm-up section.
        load_recordings: ``(dataset, subject) → that person's preprocessed recordings``
            (the cache), the same preprocessing as ``task_windows``.
        max_ptp_uv: the stream's quality gate (the validation contract's ``ptp_flag_uv``).

    Raises:
        NeurOSDataError: on a warm-up dataset without windows, a person in no fold, a missing
            prior, or a person with no window to score.
    """
    say: Progress = progress or (lambda message: print(message, flush=True))
    fold_of = {person: k for k, fold in enumerate(folds) for person in fold}
    max_budget = max(settings.budgets_s)
    kinds = warmup_kinds(settings)
    parts: dict[str, list[NDArray[np.float32]]] = {kind: [] for kind, _, _ in kinds}
    present_rows: list[NDArray[np.bool_]] = []
    meta: dict[str, list[Any]] = {"labels": [], "people": [], "datasets": [], "flags": []}
    report: dict[str, Any] = {"max_budget_s": max_budget, "datasets": {}}
    rates = {w.sfreq for w in task_windows.values()}
    if len(rates) != 1:
        raise NeurOSDataError(f"task windows differ in sampling rate: {sorted(rates)}")
    sfreq = float(next(iter(rates)))
    prior_windows = settings.prior_s / settings.stream_hop_s
    set_kinds = channel_set_kinds(settings)
    for _, name, channels in set_kinds:
        outside_layout = [ch for ch in channels if ch not in layout]
        if outside_layout:
            raise NeurOSConfigError(f"channel set {name!r}: {outside_layout} aren't model inputs")
    set_parts = {kind: _SetParts.empty() for kind, _, _ in set_kinds}

    for dataset in settings.datasets:
        if dataset not in task_windows:
            raise NeurOSDataError(f"warm-up dataset {dataset} has no task windows")
        spec = specs[dataset]
        profile = profile_channels(layout, spec)
        rows = np.array([ch in profile for ch in layout])
        # Every training person's mean covariance on this profile (from their task windows).
        person_means: dict[str, NDArray[np.float64]] = {}
        for source in trainable:
            person_means |= person_mean_covariances(task_windows[source], specs[source], profile)
        # Raw windows: _profile_rows applies the derived channels (so each slice is derived once).
        windows = task_windows[dataset]
        entry: dict[str, Any] = {"profile": list(profile), "people": {}}
        # E5: the channel sets this dataset has in full, each in layout order, with the training
        # people's mean covariances on exactly those channels (for the set's prior).
        set_profiles: dict[str, tuple[str, ...]] = {}
        set_means: dict[str, dict[str, NDArray[np.float64]]] = {}
        entry["channel_sets"] = {}
        for kind, name, channels in set_kinds:
            lacking = [ch for ch in channels if ch not in profile]
            if lacking:
                entry["channel_sets"][name] = f"not scored: the dataset lacks {lacking}"
                continue
            set_profiles[kind] = tuple(ch for ch in layout if ch in channels)
            entry["channel_sets"][name] = list(set_profiles[kind])
            set_means[kind] = {}
            for source in trainable:
                set_means[kind] |= person_mean_covariances(
                    task_windows[source], specs[source], set_profiles[kind]
                )
        for person in windows.people:
            if person not in fold_of:
                raise NeurOSDataError(f"{person} is in no development fold")
            outside = [p for p in person_means if p not in folds[fold_of[person]]]
            if not outside:
                raise NeurOSDataError(f"no training person forms a prior for {dataset}")
            prior = np.mean([person_means[p] for p in outside], axis=0)
            subject = int(person.split(":", 1)[1])
            recordings = load_recordings(dataset, subject)
            streams = session_streams(recordings, spec, profile, settings, max_ptp_uv=max_ptp_uv)
            set_streams = {
                kind: session_streams(recordings, spec, chans, settings, max_ptp_uv=max_ptp_uv)
                for kind, chans in set_profiles.items()
            }
            set_priors: dict[str, NDArray[np.float64]] = {}
            for kind, means in set_means.items():
                others = [p for p in means if p not in folds[fold_of[person]]]
                if not others:
                    raise NeurOSDataError(f"no training person forms a prior for {kind}")
                set_priors[kind] = np.mean([means[p] for p in others], axis=0)
            mine = np.flatnonzero(np.array(windows.person_ids) == person)
            sessions = np.array(windows.sessions)[mine]
            runs = np.array(windows.runs)[mine]
            starts_s = windows.starts[mine] / sfreq
            person_report: dict[str, Any] = {"n_prior_people": len(outside), "sessions": {}}
            scored_total = 0
            for session in dict.fromkeys(sessions.tolist()):
                if session not in streams:
                    raise NeurOSDataError(f"{person} session {session}: no recording")
                stream = streams[session]
                in_session = mine[sessions == session]
                offsets = np.array(
                    [stream.run_offsets_s[r] for r in runs[sessions == session].tolist()]
                )
                session_time = offsets + starts_s[sessions == session]
                scored = in_session[session_time >= max_budget - _EPS_S]
                context = DataContext(dataset, subject, session)
                all_x = _profile_rows(
                    windows.data[in_session], windows.ch_names, spec, profile, context
                )
                x = _profile_rows(windows.data[scored], windows.ch_names, spec, profile, context)
                person_report["sessions"][session] = {
                    "n_task_windows": int(in_session.size),
                    "n_scored": int(scored.size),
                    "n_stream_windows": int(stream.end_s.size),
                    "n_rest_windows": int(stream.rest_end_s.size),
                    "n_rejected": stream.n_rejected,
                }
                if scored.size == 0:
                    continue
                scored_total += int(scored.size)
                references = {}
                for kind, mode, budget in kinds:
                    if mode == "full":  # E1's offline reference: all the session's task windows
                        references[kind] = euclidean_reference(all_x, context=context)
                    elif mode == "prior":
                        references[kind] = reference_from_mean(prior, 0, context=context)
                    else:
                        references[kind] = blended_reference(
                            stream, budget, prior, prior_windows, mode=mode, context=context
                        )
                for kind, reference in references.items():
                    placed = np.zeros((scored.size, len(layout), x.shape[-1]), dtype=np.float32)
                    placed[:, rows, :] = apply_alignment(x, reference)
                    parts[kind].append(placed)
                person_report["sessions"][session]["n_user_windows"] = {
                    kind: references[kind].n_windows
                    for kind, mode, _ in kinds
                    if mode in ("stream", "rest")
                }
                for kind, chans in set_profiles.items():
                    set_rows = np.array([ch in chans for ch in layout])
                    set_x = _profile_rows(
                        windows.data[scored], windows.ch_names, spec, chans, context
                    )
                    set_reference = blended_reference(
                        set_streams[kind][session],
                        settings.channel_set_budget_s,
                        set_priors[kind],
                        prior_windows,
                        mode="stream",
                        context=context,
                    )
                    placed = np.zeros((scored.size, len(layout), x.shape[-1]), dtype=np.float32)
                    placed[:, set_rows, :] = apply_alignment(set_x, set_reference)
                    collected = set_parts[kind]
                    collected.x.append(placed)
                    collected.present.append(np.repeat(set_rows[None, :], scored.size, axis=0))
                    collected.labels += [windows.labels[i] for i in scored]
                    collected.people += [person] * scored.size
                    collected.datasets += [dataset] * scored.size
                    collected.flags.append(windows.flags[scored])
                present_rows.append(np.repeat(rows[None, :], scored.size, axis=0))
                meta["labels"] += [windows.labels[i] for i in scored]
                meta["people"] += [person] * scored.size
                meta["datasets"] += [dataset] * scored.size
                meta["flags"].append(windows.flags[scored])
            if scored_total == 0:
                raise NeurOSDataError(
                    f"{person}: no task window starts after {max_budget} s in any session"
                )
            person_report["n_scored"] = scored_total
            entry["people"][person] = person_report
        report["datasets"][dataset] = entry
        say(
            f"warm-up {dataset}: {len(entry['people'])} people, profile {len(profile)} channels, "
            f"{sum(p['n_scored'] for p in entry['people'].values())} scored windows"
        )

    if not present_rows:
        raise NeurOSDataError("no warm-up windows to score")
    present = np.concatenate(present_rows)
    labels = np.array(meta["labels"], dtype=str)
    people = np.array(meta["people"], dtype=str)
    datasets = np.array(meta["datasets"], dtype=str)
    flags = np.concatenate(meta["flags"]).astype(np.uint8)
    sets: dict[str, PooledWindows] = {}
    for kind, _, _ in kinds:
        # Each set's parts are released as soon as it is stacked: holding every set's parts
        # and every stacked set at once doubled the peak (≈ 20 GB for E2 on 30 GB; 9 Oct).
        x = np.concatenate(parts.pop(kind))
        sets[kind] = PooledWindows(
            x=x,
            present=present,
            labels=labels,
            people=people,
            datasets=datasets,
            flags=flags,
            layout=tuple(layout),
            sfreq=sfreq,
        )
    for kind, collected in set_parts.items():
        if collected.x:  # a set no warm-up dataset has stays unscored (reported per dataset)
            sets[kind] = collected.stack(layout, sfreq)
    report["channel_set_kinds"] = [kind for kind in set_parts if kind in sets]
    return sets, report


def summarize_warmup(
    outcomes: Sequence[Mapping[str, Any]],
    settings: WarmupConfig,
    *,
    within_points: float = 2.0,
) -> dict[str, Any]:
    """Per warm-up dataset: each set's mean balanced accuracy, its paired gap to the full-session
    reference, and S, the smallest stream budget within ``within_points`` of full-session.

    Raises:
        NeurOSDataError: if a person is scored twice in one set.
    """
    from neuros.eval.metrics import paired_wilcoxon, summarize

    kinds = warmup_kinds(settings)
    table: dict[str, dict[str, float]] = {kind: {} for kind, _, _ in kinds}
    for outcome in outcomes:
        for kind, scores in (outcome.get("extra_scores") or {}).items():
            if kind not in table:
                continue
            for score in scores:
                if score["person"] in table[kind]:
                    raise NeurOSDataError(f"{score['person']} scored twice in {kind}")
                table[kind][score["person"]] = float(score["balanced_accuracy_raw"])
    full = table[kind_name("full", math.inf)]
    summary: dict[str, Any] = {}
    for dataset in settings.datasets:
        people = sorted(p for p in full if p.startswith(f"{dataset}:"))
        if not people:
            continue
        entry: dict[str, Any] = {"n_people": len(people), "sets": {}}
        for kind, mode, budget in kinds:
            values = [table[kind][p] for p in people if p in table[kind]]
            if len(values) != len(people):
                continue
            result: dict[str, Any] = {"mode": mode, "budget_s": budget, **summarize(values)}
            if mode != "full" and len(people) >= 2:
                result["vs_full"] = paired_wilcoxon(values, [full[p] for p in people])
            entry["sets"][kind] = result
        full_mean = entry["sets"][kind_name("full", math.inf)]["mean"]
        reached = [
            budget
            for kind, mode, budget in kinds
            if mode in ("prior", "stream")
            and math.isfinite(budget)
            and kind in entry["sets"]
            and entry["sets"][kind]["mean"] >= full_mean - within_points / 100
        ]
        entry["s_within_points"] = within_points
        entry["s_seconds"] = min(reached) if reached else None
        summary[dataset] = entry
    return summary


# E5's bar (knowledge-base/plan/e5-virtual-headsets.md, set before any run): the research cap
# is compared on these datasets' people only (Shin2017A's full profile *is* its 3-channel set).
E5_PAIRED_DATASETS: tuple[str, ...] = ("Yang2025", "BNCI2014_001")
E5_MAX_DROP_POINTS = 3.0
E5_ALPHA = 0.05


def _scores_by_kind(outcomes: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    """``kind -> person -> balanced accuracy`` from the folds' ``extra_scores``.

    Raises:
        NeurOSDataError: if a person is scored twice in one set (folds overlap).
    """
    table: dict[str, dict[str, float]] = {}
    for outcome in outcomes:
        for kind, scores in (outcome.get("extra_scores") or {}).items():
            column = table.setdefault(kind, {})
            for score in scores:
                if score["person"] in column:
                    raise NeurOSDataError(f"{score['person']} scored twice in {kind}")
                column[score["person"]] = float(score["balanced_accuracy_raw"])
    return table


def channel_set_label(
    above_chance: bool, paired_drop_points: float | None, *, max_drop: float = E5_MAX_DROP_POINTS
) -> str:
    """E5's label: supported (above chance and at most ``max_drop`` points below the research
    cap), experimental (above chance only, or no paired people to compare), not supported."""
    if not above_chance:
        return "not supported"
    if paired_drop_points is not None and paired_drop_points <= max_drop:
        return "supported"
    return "experimental"


def summarize_channel_sets(
    outcomes: Sequence[Mapping[str, Any]], settings: WarmupConfig
) -> dict[str, Any]:
    """E5: each channel set's accuracy, its paired drop from the research cap and its label.

    - **Above chance:** every person scored on the set, two-sided Wilcoxon against 50%,
      p < ``E5_ALPHA``, and a mean above 50%.
    - **Drop:** research cap minus set, paired over the set's people in ``E5_PAIRED_DATASETS``
      (in points; positive = the set is worse). The cap is the same budget's stream set on the
      dataset's own full profile.
    """
    from neuros.eval.metrics import paired_wilcoxon, summarize

    table = _scores_by_kind(outcomes)
    cap_kind = kind_name("stream", settings.channel_set_budget_s)
    cap = table.get(cap_kind, {})
    result: dict[str, Any] = {
        "budget_s": settings.channel_set_budget_s,
        "research_cap_kind": cap_kind,
        "paired_datasets": list(E5_PAIRED_DATASETS),
        "max_drop_points": E5_MAX_DROP_POINTS,
        "alpha": E5_ALPHA,
        "sets": {},
    }
    for kind, name, channels in channel_set_kinds(settings):
        scores = table.get(kind, {})
        entry: dict[str, Any] = {"kind": kind, "channels": list(channels)}
        if not scores:
            entry["label"] = "not scored (no warm-up dataset has these channels)"
            result["sets"][name] = entry
            continue
        people = sorted(scores)
        values = [scores[p] for p in people]
        chance = paired_wilcoxon(values, [0.5] * len(values))
        above = bool(chance["p_value"] < E5_ALPHA and np.mean(values) > 0.5)
        per_dataset = {}
        for dataset in dict.fromkeys(p.split(":", 1)[0] for p in people):
            mine = [scores[p] for p in people if p.startswith(f"{dataset}:")]
            per_dataset[dataset] = summarize(mine)
        paired_people = [p for p in people if p.split(":", 1)[0] in E5_PAIRED_DATASETS and p in cap]
        paired: dict[str, Any] | None = None
        drop: float | None = None
        if paired_people:
            paired = paired_wilcoxon(
                [scores[p] for p in paired_people], [cap[p] for p in paired_people]
            )
            drop = -100.0 * float(paired["mean_diff"])
            paired["research_cap_mean"] = float(np.mean([cap[p] for p in paired_people]))
            paired["set_mean"] = float(np.mean([scores[p] for p in paired_people]))
        entry |= {
            "all_people": summarize(values),
            "per_dataset": per_dataset,
            "vs_chance": chance,
            "above_chance": above,
            "vs_research_cap": paired,
            "drop_points": drop,
            "label": channel_set_label(above, drop),
        }
        result["sets"][name] = entry
    return result
