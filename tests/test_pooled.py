"""E1's pooled runner: layout mapping, the task plan, isolation, saving, a tiny end-to-end run."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from neuros.errors import NeurOSConfigError, NeurOSDataError, NeurOSSplitError
from neuros.eval.pooled import (
    ModelTask,
    PooledWindows,
    TaskSettings,
    assemble,
    plan_tasks,
    report,
    run_tasks,
    split_validation,
    to_layout,
)
from neuros.signal.windows import WindowSet

from .synthetic_people import FAST_TRAIN, TINY_MODEL, people_windows

LAYOUT = ("C3", "Cz", "C4", "Pz")


def _dataset(name: str, channels: tuple[str, ...], n_people: int, seed: int) -> WindowSet:
    """``n_people`` synthetic people renamed to dataset ``name``, keeping ``channels``."""
    scales = tuple(float(s) for s in np.geomspace(0.3, 3.0, n_people))
    windows = people_windows(scales, per_class=6, seed=seed).pick_channels(list(channels))
    return dataclasses.replace(windows, datasets=(name,) * len(windows))


@pytest.fixture(scope="module")
def three_datasets() -> dict[str, WindowSet]:
    return {
        "SynA": _dataset("SynA", ("C3", "C4", "Cz", "Pz"), 6, seed=1),
        "SynB": _dataset("SynB", ("C3", "C4", "Cz"), 6, seed=2),
        "SynC": _dataset("SynC", ("C3", "C4", "Pz"), 4, seed=3),  # test-only below
    }


def test_to_layout_zero_fills_and_aligns_present_channels_only(
    three_datasets: dict[str, WindowSet],
) -> None:
    windows = three_datasets["SynB"]  # no Pz
    x, present, refs = to_layout(windows, LAYOUT, reference="car", alignment="ea", unit="session")
    assert x.shape == (len(windows), 4, windows.n_samples)
    assert present.all(axis=0).tolist() == [True, True, True, False]
    assert not x[:, 3].any()  # Pz is zero-filled
    # CAR over the 3 present channels: they sum to zero in every sample.
    np.testing.assert_allclose(x[:, :3].sum(axis=1), 0.0, atol=1e-3)
    assert len(refs) == len(windows.people) * 2  # one EA reference per person and session
    with pytest.raises(NeurOSDataError, match="at least 2"):
        to_layout(
            windows.pick_channels(["C3"]), LAYOUT, reference="car", alignment="none", unit="session"
        )
    with pytest.raises(NeurOSConfigError, match="unknown reference"):
        to_layout(windows, LAYOUT, reference="bipolar", alignment="none", unit="session")  # type: ignore[arg-type]


def test_assemble_stacks_datasets_with_masks(three_datasets: dict[str, WindowSet]) -> None:
    pooled, reports = assemble(
        three_datasets, LAYOUT, reference="car", alignment="ea", unit="session"
    )
    assert len(pooled) == sum(len(w) for w in three_datasets.values())
    assert list(pooled.people_by_dataset()) == ["SynA", "SynB", "SynC"]
    assert reports["SynB"]["channels_missing"] == ["Pz"]
    assert reports["SynC"]["channels_missing"] == ["Cz"]
    rows = pooled.rows_of(pooled.people_by_dataset()["SynC"])
    assert not pooled.present[rows][:, 1].any()  # SynC has no Cz
    wrong = {"SynA": three_datasets["SynB"]}
    with pytest.raises(NeurOSDataError, match="hold"):
        assemble(wrong, LAYOUT, reference="car", alignment="none", unit="session")


def test_save_and_map_read_only(tmp_path: Path, three_datasets: dict[str, WindowSet]) -> None:
    pooled, _ = assemble(three_datasets, LAYOUT, reference="car", alignment="ea", unit="session")
    pooled.save(tmp_path / "task")
    loaded = PooledWindows.load(tmp_path / "task", verify=True)
    assert isinstance(loaded.x, np.memmap)
    assert np.array_equal(np.asarray(loaded.x), pooled.x)
    assert loaded.layout == LAYOUT
    with pytest.raises(NeurOSDataError, match="already holds"):
        pooled.save(tmp_path / "task")  # never overwritten
    # Release the memory maps first: Windows refuses to write a file that is mapped.
    del loaded
    import gc

    gc.collect()
    labels = tmp_path / "task" / "labels.npy"
    np.save(labels, np.array(["x"] * len(pooled)))
    with pytest.raises(NeurOSDataError, match="sha256 mismatch"):
        PooledWindows.load(tmp_path / "task", verify=True)


def test_split_validation_is_stratified_and_stable() -> None:
    people = [f"A:{i}" for i in range(10)] + [f"B:{i}" for i in range(3)] + ["C:1"]
    train, validation = split_validation(people, 0.2, seed=1, tag="t")
    assert sorted(p.split(":")[0] for p in validation) == ["A", "A", "B"]  # round(0.2 x 3) = 1
    assert "C:1" in train  # a dataset keeps at least one training person
    assert set(train).isdisjoint(validation)
    assert split_validation(people, 0.2, seed=1, tag="t") == (train, validation)
    assert split_validation(people, 0.2, seed=1, tag="other") != (train, validation)
    assert split_validation(people, 0.0, seed=1, tag="t") == (tuple(people), ())
    with pytest.raises(NeurOSConfigError, match="validation fraction"):
        split_validation(people, 1.0, seed=1, tag="t")


def test_plan_tests_everyone_once_and_never_trains_test_only(
    three_datasets: dict[str, WindowSet],
) -> None:
    people = {name: windows.people for name, windows in three_datasets.items()}
    tasks = plan_tasks(
        people,
        ["SynA", "SynB"],
        conditions=("pooled", "single", "lodo", "cue_only"),
        folds=3,
        validation_fraction=0.2,
        seed=7,
    )
    names = [t.name for t in tasks]
    assert sum(n.startswith("pooled/") for n in names) == 3
    assert sum(n.startswith("cue_only/") for n in names) == 3
    assert sum(n.startswith("single/") for n in names) == 6
    assert sorted(n for n in names if n.startswith("lodo/")) == ["lodo/SynA", "lodo/SynB"]
    everyone = {p for members in people.values() for p in members}
    for condition, expected in (
        ("pooled", everyone),
        ("cue_only", everyone),
        ("single", set(people["SynA"]) | set(people["SynB"])),
        ("lodo", set(people["SynA"]) | set(people["SynB"])),
    ):
        tested = [p for t in tasks if t.condition == condition for p in t.test]
        assert len(tested) == len(set(tested)), condition  # each person once
        assert set(tested) == expected, condition
    for task in tasks:
        assert not any(p.startswith("SynC:") for p in (*task.train, *task.validation))
        assert set(task.train).isdisjoint(task.test)
        if task.condition == "lodo":
            held_out = task.name.split("/")[1]
            assert not any(p.startswith(f"{held_out}:") for p in task.train)
    with pytest.raises(NeurOSSplitError):
        ModelTask("bad", "pooled", "task", ("SynA:1",), (), ("SynA:1",), 0)
    with pytest.raises(NeurOSConfigError, match="at least 2 trainable"):
        plan_tasks(people, ["SynA"], conditions=("lodo",), folds=3, validation_fraction=0, seed=1)


def test_tiny_end_to_end_run(tmp_path: Path, three_datasets: dict[str, WindowSet]) -> None:
    pooled, _ = assemble(three_datasets, LAYOUT, reference="car", alignment="ea", unit="session")
    pooled.save(tmp_path / "task")
    pooled.save(tmp_path / "cue")  # stands in for the cue-only windows
    people = pooled.people_by_dataset()
    tasks = plan_tasks(
        people,
        ["SynA", "SynB"],
        conditions=("pooled", "single", "lodo", "cue_only"),
        folds=3,
        validation_fraction=0.0,
        seed=3,
    )
    settings = TaskSettings(
        classes=("left_hand", "right_hand"),
        model=TINY_MODEL,
        train=FAST_TRAIN,
        device="cpu",
        deterministic=True,
        threads=1,
    )
    seen: list[str] = []
    outcomes = run_tasks(
        tasks,
        {"task": tmp_path / "task", "cue_only": tmp_path / "cue"},
        settings,
        workers=1,
        progress=lambda line: None,
        on_task=lambda outcome: seen.append(outcome["task"]),
    )
    assert seen == [t.name for t in tasks]
    assert [o["task"] for o in outcomes] == [t.name for t in tasks]
    summary = report(outcomes, people, {"SynA": True, "SynB": False, "SynC": True})
    syn_a = summary["datasets"]["SynA"]
    for condition in ("pooled", "single", "lodo", "cue_only"):
        assert syn_a[condition]["n_people"] == len(people["SynA"])
    assert "pooled_vs_single" in syn_a
    assert "single" not in summary["datasets"]["SynC"]  # test-only: never a single model
    # The synthetic class signal is strong: pooled models should beat chance clearly.
    assert syn_a["pooled"]["mean"] > 0.7
    with pytest.raises(NeurOSConfigError, match="weren't assembled"):
        run_tasks(tasks, {"task": tmp_path / "task"}, settings)
