"""E2's warm-up replay (neuros.eval.warmup): streams, blended references, the paired sets."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from pydantic import ValidationError

from neuros.config.experiment import PooledExperimentConfig, WarmupConfig
from neuros.config.schemas import DataConfig
from neuros.data.recording import Recording
from neuros.data.synthetic import SyntheticConfig, make_recording
from neuros.errors import NeurOSConfigError, NeurOSDataError
from neuros.eval.pooled import assemble
from neuros.eval.splits import make_dev_folds
from neuros.eval.warmup import (
    blended_reference,
    build_warmup_sets,
    kind_name,
    session_streams,
    summarize_warmup,
    warmup_kinds,
)
from neuros.signal.preprocess import preprocess_recording
from neuros.signal.windows import WindowSet, extract_epochs

from .conftest import make_spec

LAYOUT = ("C3", "Cz", "C4", "CP3", "CP4")
LABELS = ("left_hand", "right_hand")
SETTINGS = WarmupConfig(datasets=("SynB",), budgets_s=(0.0, 5.0, 20.0), prior_s=4.0)


def _recordings(dataset: str, subject: int, sessions: int = 1, runs: int = 2) -> list[Recording]:
    """Preprocessed synthetic recordings (128 Hz), one per session and run, distinct seeds."""
    out = []
    for session in range(sessions):
        for run in range(runs):
            seed = subject * 100 + session * 10 + run + (0 if dataset == "SynA" else 5000)
            config = SyntheticConfig(n_trials=12, seed=seed)
            raw = make_recording(
                config, dataset=dataset, subject=subject, session=str(session), run=str(run)
            )
            out.append(preprocess_recording(raw, DataConfig().signal))
    return out


def _windows(recordings: list[Recording]) -> WindowSet:
    sets = [
        extract_epochs(r, DataConfig().signal, labels=LABELS, epoch_s=(0.5, 2.5))[0]
        for r in recordings
    ]
    return WindowSet.concat(sets)


@pytest.fixture(scope="module")
def world() -> dict[str, object]:
    specs = {
        "SynA": make_spec(key="SynA", pre_cue_fixation_s=2.0),
        "SynB": make_spec(key="SynB", pre_cue_fixation_s=2.0),
    }
    recordings = {(d, s): _recordings(d, s) for d in ("SynA", "SynB") for s in range(1, 6)}
    task = {
        d: WindowSet.concat([_windows(recordings[(d, s)]) for s in range(1, 6)])
        for d in ("SynA", "SynB")
    }
    people = {d: list(range(1, 6)) for d in ("SynA", "SynB")}
    folds = make_dev_folds(people, n_folds=5, seed=7)
    return {"specs": specs, "recordings": recordings, "task": task, "folds": folds}


def test_kinds_are_named_and_ordered() -> None:
    kinds = warmup_kinds(SETTINGS)
    assert [k for k, _, _ in kinds] == [
        "e2_prior",
        "e2_stream_5s",
        "e2_stream_20s",
        "e2_stream_all",
        "e2_rest_5s",
        "e2_rest_20s",
        "e2_rest_all",
        "e2_full",
    ]
    assert kind_name("stream", 2.5) == "e2_stream_2p5s"
    no_rest = SETTINGS.model_copy(update={"rest_seeded": False})
    assert "e2_rest_5s" not in [k for k, _, _ in warmup_kinds(no_rest)]
    with pytest.raises(NeurOSConfigError, match="unknown warm-up mode"):
        kind_name("later", 1.0)


def test_config_rules() -> None:
    with pytest.raises(ValidationError, match="ascending"):
        WarmupConfig(datasets=("X",), budgets_s=(5.0, 0.0))
    with pytest.raises(ValidationError, match="hop"):
        WarmupConfig(datasets=("X",), stream_window_s=1.0, stream_hop_s=2.0)
    base = {"name": "e2", "train_datasets": ["BNCI2014_001", "PhysionetMI"]}
    ok = PooledExperimentConfig.model_validate(
        base | {"conditions": ["pooled"], "warmup": {"datasets": ["BNCI2014_001"]}}
    )
    assert ok.warmup is not None
    with pytest.raises(ValidationError, match="aren't in this experiment"):
        PooledExperimentConfig.model_validate(base | {"warmup": {"datasets": ["Yang2025"]}})
    with pytest.raises(ValidationError, match="must include pooled"):
        PooledExperimentConfig.model_validate(
            base | {"conditions": ["single"], "warmup": {"datasets": ["BNCI2014_001"]}}
        )
    with pytest.raises(ValidationError, match="one seed"):
        PooledExperimentConfig.model_validate(
            base | {"seeds": [1, 2], "warmup": {"datasets": ["BNCI2014_001"]}}
        )


def test_session_streams_time_rest_and_gate(world: dict[str, object]) -> None:
    spec = world["specs"]["SynB"]  # type: ignore[index]
    recordings = world["recordings"][("SynB", 1)]  # type: ignore[index]
    profile = LAYOUT
    streams = session_streams(recordings, spec, profile, SETTINGS, max_ptp_uv=1e6)
    stream = streams["0"]
    run0 = recordings[0].data.shape[1] / recordings[0].sfreq
    assert stream.run_offsets_s == {"0": 0.0, "1": pytest.approx(run0)}
    assert np.all(np.diff(stream.end_s) > 0)  # session time only moves forward
    assert stream.end_s[0] >= 0.5 + SETTINGS.stream_window_s  # the filter start-up is skipped
    assert stream.covariances.shape[1:] == (len(profile), len(profile))
    # H1's rest windows are anchored to the fixations: with a 2-s fixation and 2-s windows,
    # exactly one per cue, ending at the cue (a fixed hop grid would hit 1 cue in 16).
    sfreq = recordings[0].sfreq
    cues_s = recordings[0].events.onsets / sfreq
    rest_ends = stream.rest_end_s[stream.rest_end_s < run0]
    assert 0 < rest_ends.size <= cues_s.size
    assert all(np.min(np.abs(cues_s - end)) < 1.5 / sfreq for end in rest_ends)
    assert stream.rest_covariances.shape == (stream.rest_end_s.size, len(profile), len(profile))
    # The quality gate: a tiny peak-to-peak limit rejects everything.
    with pytest.raises(NeurOSDataError, match="quality gate"):
        session_streams(recordings, spec, profile, SETTINGS, max_ptp_uv=1e-6)
    with pytest.raises(NeurOSDataError, match="one person"):
        session_streams(
            recordings + world["recordings"][("SynB", 2)],  # type: ignore[index]
            spec,
            profile,
            SETTINGS,
            max_ptp_uv=1e6,
        )


def test_blended_reference_moves_from_prior_to_user(world: dict[str, object]) -> None:
    spec = world["specs"]["SynB"]  # type: ignore[index]
    recordings = world["recordings"][("SynB", 1)]  # type: ignore[index]
    stream = session_streams(recordings, spec, LAYOUT, SETTINGS, max_ptp_uv=1e6)["0"]
    prior = 3.0 * np.eye(len(LAYOUT))
    at_zero = blended_reference(stream, 0.0, prior, 32.0)
    assert at_zero.n_windows == 0
    np.testing.assert_allclose(at_zero.covariance.matrix, prior, rtol=1e-2)
    user = stream.covariances.mean(axis=0)
    gaps = [
        np.linalg.norm(blended_reference(stream, b, prior, 32.0).covariance.matrix - user)
        for b in (5.0, 20.0, 1e4)
    ]
    assert gaps[0] > gaps[1] > gaps[2]  # more of the user's data, closer to their own mean
    rest = blended_reference(stream, 1e4, prior, 32.0, mode="rest")
    assert rest.n_windows == stream.rest_end_s.size
    with pytest.raises(NeurOSConfigError, match="seeding mode"):
        blended_reference(stream, 5.0, prior, 32.0, mode="task")
    with pytest.raises(NeurOSDataError, match="prior is"):
        blended_reference(stream, 5.0, np.eye(2), 32.0)


def _build(world: dict[str, object]) -> tuple[dict, dict]:
    recordings = world["recordings"]
    return build_warmup_sets(
        world["task"],  # type: ignore[arg-type]
        LAYOUT,
        specs=world["specs"],  # type: ignore[arg-type]
        folds=world["folds"],  # type: ignore[arg-type]
        trainable=("SynA", "SynB"),
        settings=SETTINGS,
        load_recordings=lambda d, s: recordings[(d, s)],  # type: ignore[index]
        max_ptp_uv=1e6,
        progress=lambda _: None,
    )


def test_sets_are_paired_and_full_matches_e1(world: dict[str, object]) -> None:
    sets, report = _build(world)
    kinds = [k for k, _, _ in warmup_kinds(SETTINGS)]
    assert list(sets) == kinds
    first = sets["e2_full"]
    for kind in kinds:  # the same scored windows, in the same order, for every set
        assert np.array_equal(sets[kind].people, first.people)
        assert np.array_equal(sets[kind].labels, first.labels)
    assert set(first.datasets.tolist()) == {"SynB"}
    # Only windows starting after the largest budget are scored.
    entry = report["datasets"]["SynB"]["people"]["SynB:1"]
    assert entry["n_scored"] < sum(s["n_task_windows"] for s in entry["sessions"].values())
    # e2_full is exactly E1's placement of those windows (per-session EA over all of them).
    pooled, _ = assemble(
        {"SynB": world["task"]["SynB"]},  # type: ignore[index]
        LAYOUT,
        reference="car",
        alignment="ea",
        unit="session",
    )
    task = world["task"]["SynB"]  # type: ignore[index]
    sfreq = task.sfreq
    for person in ("SynB:1", "SynB:3"):
        mine = np.flatnonzero(np.array(task.person_ids) == person)
        recs = world["recordings"][("SynB", int(person.split(":")[1]))]  # type: ignore[index]
        offsets = {r.run: 0.0 for r in recs}
        offsets["1"] = recs[0].data.shape[1] / sfreq
        times = np.array([offsets[task.runs[i]] for i in mine]) + task.starts[mine] / sfreq
        expected = pooled.x[mine[times >= max(SETTINGS.budgets_s)]]
        got = first.x[first.people == person]
        np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)

    # More of the user's stream moves the alignment from the prior toward the stream's own
    # asymptote (e2_stream_all). The asymptote need not equal e2_full: the engine streams
    # everything, E1 averages task windows only (warmup_kinds explains).
    def distance(kind: str, to: str) -> float:
        return float(np.linalg.norm(sets[kind].x - sets[to].x))

    assert distance("e2_stream_20s", "e2_stream_all") < distance("e2_stream_5s", "e2_stream_all")
    assert distance("e2_stream_5s", "e2_stream_all") < distance("e2_prior", "e2_stream_all")
    assert distance("e2_prior", "e2_full") > 0


def test_summary_finds_s(world: dict[str, object]) -> None:
    sets, _ = _build(world)
    people = sorted(set(sets["e2_full"].people.tolist()))
    accuracy = {"e2_prior": 0.55, "e2_stream_5s": 0.62, "e2_stream_20s": 0.69, "e2_full": 0.70}
    outcomes = [
        {
            "extra_scores": {
                kind: [
                    {"person": p, "balanced_accuracy_raw": accuracy.get(kind, 0.6) + i * 1e-3}
                    for i, p in enumerate(people)
                ]
                for kind in sets
            }
        }
    ]
    summary = summarize_warmup(outcomes, SETTINGS)
    entry = summary["SynB"]
    assert entry["n_people"] == len(people)
    assert entry["s_seconds"] == 20.0  # 69% is within 2 points of 70%; 62% isn't
    assert entry["sets"]["e2_prior"]["vs_full"]["mean_diff"] == pytest.approx(-0.15)
    twice = outcomes + outcomes
    with pytest.raises(NeurOSDataError, match="twice"):
        summarize_warmup(twice, SETTINGS)
    never = summarize_warmup(
        [
            {
                "extra_scores": {
                    k: [
                        {"person": p, "balanced_accuracy_raw": 0.5 if k != "e2_full" else 0.8}
                        for p in people
                    ]
                    for k in sets
                }
            }
        ],
        SETTINGS,
    )
    assert never["SynB"]["s_seconds"] is None
    assert math.isfinite(entry["sets"]["e2_full"]["mean"])


def test_derived_profiles_stream_the_same_way(world: dict[str, object]) -> None:
    # A cap without Cz: derive it from C1 + C2, as Shin2017A derives C3/C4 (D-037).
    spec = dataclasses.replace(
        world["specs"]["SynB"],  # type: ignore[index]
        derived_channels={"FCz": ("C1", "C2")},
        train_restriction="test-only",
    )
    recordings = world["recordings"][("SynB", 1)]  # type: ignore[index]
    streams = session_streams(recordings, spec, ("C3", "FCz", "C4"), SETTINGS, max_ptp_uv=1e6)
    assert streams["0"].covariances.shape[1:] == (3, 3)


def test_rest_counts_by_covered_seconds(world: dict[str, object]) -> None:
    from neuros.eval.warmup import _covered_seconds

    # Overlapping windows count their union once; disjoint ones add up.
    starts = np.array([0, 16, 512], dtype=np.int64)
    np.testing.assert_allclose(_covered_seconds(starts, 256, 128.0), [2.0, 2.125, 4.125])
    spec = world["specs"]["SynB"]  # type: ignore[index]
    stream = session_streams(
        world["recordings"][("SynB", 1)],  # type: ignore[index]
        spec,
        LAYOUT,
        SETTINGS,
        max_ptp_uv=1e6,
    )["0"]
    assert np.all(np.diff(stream.rest_covered_s) > 0)
    # 2-s fixations, 2-s windows: each rest window adds 2 distinct seconds.
    np.testing.assert_allclose(np.diff(stream.rest_covered_s), 2.0)
    # Weighted by seconds, rest pulls away from the prior as fast as the same seconds of stream.
    prior = 5.0 * np.eye(len(LAYOUT))
    one_rest = blended_reference(stream, float(stream.rest_end_s[0]), prior, 16.0, mode="rest")
    weight = 2.0 / stream.hop_s  # 2 s of data = 16 stream windows
    expected = (16.0 * prior + weight * stream.rest_covariances[0]) / (16.0 + weight)
    np.testing.assert_allclose(one_rest.covariance.matrix, expected, rtol=2e-3, atol=1e-6)
