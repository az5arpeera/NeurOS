"""Q30's fine-tuning: our pooled model plus a person's first few labelled trials.

Synthetic people as in the warm-up tests: one session of two runs, 12 trials per class.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from neuros.config.experiment import FinetuneConfig, PooledExperimentConfig
from neuros.eval.calibration import split_block
from neuros.eval.finetune import FinetuneSettings, build_finetune_sets, finetune_kinds
from neuros.eval.pooled import TaskSettings, assemble, plan_tasks, run_tasks
from neuros.eval.warmup import session_streams

from .synthetic_people import FAST_TRAIN, TINY_MODEL
from .test_warmup import LABELS, LAYOUT, SETTINGS, world  # noqa: F401  (world: a fixture)

FINETUNE = FinetuneConfig(datasets=("SynB",), sizes=(2, 4), epochs=5, lr=1e-3, min_lr=1e-4)


def _build(world: dict[str, object]) -> tuple[dict, dict]:  # noqa: F811
    recordings = world["recordings"]
    return build_finetune_sets(
        world["task"],  # type: ignore[arg-type]
        LAYOUT,
        specs=world["specs"],  # type: ignore[arg-type]
        folds=world["folds"],  # type: ignore[arg-type]
        trainable=("SynA", "SynB"),
        labels=LABELS,
        warmup=SETTINGS,
        finetune=FINETUNE,
        load_recordings=lambda d, s: recordings[(d, s)],  # type: ignore[index]
        max_ptp_uv=1e6,
        progress=lambda _: None,
    )


def test_sets_follow_the_calibration_split_and_calibration_time(
    world: dict[str, object],  # noqa: F811
) -> None:
    sets, report = _build(world)
    kinds = finetune_kinds(FINETUNE.sizes)
    assert sorted(sets) == sorted(k for _, cal, test in kinds for k in (cal, test))
    windows = world["task"]["SynB"]  # type: ignore[index]
    ids = np.array(windows.person_ids)
    for person in windows.people:
        rows = np.flatnonzero(ids == person).astype(np.int64)
        pool, test = split_block(windows, rows, LABELS)  # the calibration curve's split
        for n, cal_kind, test_kind in kinds:
            cal_set, test_set = sets[cal_kind], sets[test_kind]
            assert cal_set.rows_of([person]).size == 2 * n
            # The calibration trials are the pool's first n per class.
            assert all(pool[label].size >= n for label in LABELS)
            cal_labels = cal_set.labels[cal_set.rows_of([person])].tolist()
            assert sorted(cal_labels) == sorted([label for label in LABELS for _ in range(n)])
            assert test_set.rows_of([person]).size == test.size
            np.testing.assert_array_equal(
                np.sort(test_set.labels[test_set.rows_of([person])]),
                np.sort(np.asarray(windows.labels)[test]),
            )
            entry = report["datasets"]["SynB"]["people"][person][str(n)][0]
            assert entry["n_calibration"] == 2 * n
            assert entry["n_test"] == test.size
            assert entry["calibration_s"] > 0
        # More calibration trials: a longer calibration, more stream windows in the reference.
        small, large = (report["datasets"]["SynB"]["people"][person][str(n)][0] for n in (2, 4))
        assert large["calibration_s"] > small["calibration_s"]
        assert large["n_stream_windows"] > small["n_stream_windows"] > 0
        recordings = world["recordings"][("SynB", int(person.split(":")[1]))]  # type: ignore[index]
        total = session_streams(
            recordings, world["specs"]["SynB"], LAYOUT, SETTINGS, max_ptp_uv=1e6
        )  # type: ignore[index]
        assert large["n_stream_windows"] < total["0"].end_s.size  # only calibration time
    # Each size aligns with its own reference: the same test trials, different inputs.
    a, b = sets["ft2_test"], sets["ft4_test"]
    np.testing.assert_array_equal(a.labels, b.labels)
    assert not np.array_equal(a.x, b.x)


def test_finetuning_runs_in_the_workers_and_leaves_the_pooled_model_alone(
    world: dict[str, object],  # noqa: F811
    tmp_path: Path,
) -> None:
    task = world["task"]
    pooled, _ = assemble(task, LAYOUT, reference="car", alignment="ea", unit="session")  # type: ignore[arg-type]
    pooled.save(tmp_path / "task")
    sets, _ = _build(world)
    folders = {"task": tmp_path / "task"}
    for kind, windows in sets.items():
        windows.save(tmp_path / kind)
        folders[kind] = tmp_path / kind
    people = pooled.people_by_dataset()
    tasks = plan_tasks(
        people, ["SynA", "SynB"], conditions=("pooled",), folds=5, validation_fraction=0.0, seed=7
    )
    base = {"classes": LABELS, "model": TINY_MODEL, "train": FAST_TRAIN, "device": "cpu",
            "deterministic": True, "threads": 1}  # fmt: skip
    with_ft = TaskSettings(**base, finetune=FinetuneSettings.from_config(FINETUNE))  # type: ignore[arg-type]
    without = TaskSettings(**base)  # type: ignore[arg-type]
    tuned = run_tasks(tasks, folders, with_ft, workers=1, progress=lambda _: None)
    plain = run_tasks(
        tasks, {"task": tmp_path / "task"}, without, workers=1, progress=lambda _: None
    )
    for outcome, reference in zip(tuned, plain, strict=True):
        assert outcome["scores"] == reference["scores"]  # fine-tuning works on copies
        synb = [s["person"] for s in outcome["scores"] if s["person"].startswith("SynB")]
        for n in ("2", "4"):
            entry = outcome["finetune"][n]
            assert [s["person"] for s in entry["zero_calibration"]] == synb
            assert [s["person"] for s in entry["finetuned"]] == synb
            for score in entry["zero_calibration"] + entry["finetuned"]:
                assert 0.0 <= score["balanced_accuracy_raw"] <= 1.0
        assert "finetune" not in reference


def test_config_rules() -> None:
    with pytest.raises(ValidationError, match="ascending"):
        FinetuneConfig(datasets=("X",), sizes=(10, 5))
    with pytest.raises(ValidationError, match="blocks"):
        FinetuneConfig(datasets=("X",), blocks={"Y": "person"})
    with pytest.raises(ValidationError, match="min_lr"):
        FinetuneConfig(datasets=("X",), lr=1e-4, min_lr=1e-3)
    base = {
        "name": "ft",
        "train_datasets": ["BNCI2014_001", "PhysionetMI"],
        "conditions": ["pooled"],
    }
    with pytest.raises(ValidationError, match="add warmup"):
        PooledExperimentConfig.model_validate(base | {"finetune": {"datasets": ["BNCI2014_001"]}})
    with pytest.raises(ValidationError, match="aren't warm-up datasets"):
        PooledExperimentConfig.model_validate(
            base
            | {"warmup": {"datasets": ["BNCI2014_001"]}, "finetune": {"datasets": ["PhysionetMI"]}}
        )
    ok = PooledExperimentConfig.model_validate(
        base
        | {"warmup": {"datasets": ["BNCI2014_001"]}, "finetune": {"datasets": ["BNCI2014_001"]}}
    )
    settings = FinetuneSettings.from_config(ok.finetune)  # type: ignore[arg-type]
    recipe = settings.train_config(ok.train, 20)
    assert (recipe.windows_per_person, recipe.people_per_batch, recipe.patience) == (20, 1, None)
    assert (recipe.max_epochs, recipe.swa_fraction) == (50, 0.0)
