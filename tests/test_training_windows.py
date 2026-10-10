"""D-039, "use more of what we have": training crops and extra classes.

Validation and test windows never change; these tests pin that down along with each piece:
the crop rule (never towards the cue), the extraction, alignment by the test windows'
references, the masked loss, the epoch length, and a tiny end-to-end run.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from pydantic import ValidationError

from neuros.align.ea import align_groups, align_with_references
from neuros.config.experiment import PooledExperimentConfig, TrainingWindowsConfig
from neuros.config.schemas import DataConfig, SignalContract
from neuros.data import prepare as prepare_module
from neuros.data.recording import Recording
from neuros.data.registry import MAX_CROPS_PER_TRIAL, task_epoch, training_crop_starts
from neuros.data.synthetic import SyntheticConfig, make_recording
from neuros.data.training_windows import subject_training_windows
from neuros.data.vocabulary import Vocabulary
from neuros.errors import NeurOSConfigError, NeurOSDataError
from neuros.eval.pooled import (
    ModelTask,
    PooledWindows,
    TaskSettings,
    assemble,
    plan_tasks,
    run_tasks,
    to_layout,
)
from neuros.signal.preprocess import preprocess_recording
from neuros.signal.windows import WindowSet
from neuros.train.data import TensorData
from neuros.train.sampler import SubjectBalancedSampler
from neuros.train.trainer import masked_cross_entropy

from .conftest import make_spec
from .synthetic_people import FAST_TRAIN, TINY_MODEL, people_windows

LAYOUT = ("C3", "Cz", "C4", "Pz")
LR = ("left_hand", "right_hand")


# ---------------------------------------------------------------------- the crop rule
def test_crops_start_at_the_test_window_and_only_move_later() -> None:
    contract = SignalContract()  # 2-s windows, the D-030 rule
    # Cue-free, like BNCI2014-001: the class leaves at 1.25 s, the task ends at 4 s.
    spec = make_spec(task_s=4.0, class_on_screen_s=1.25)
    assert task_epoch(spec, contract).start_s == 1.5
    assert training_crop_starts(spec, contract, 0.25) == (1.5, 1.75, 2.0)
    assert training_crop_starts(spec, contract, 0.5) == (1.5, 2.0)
    assert training_crop_starts(spec, contract, 5.0) == (1.5,)  # a stride past the span
    # Like Yang2025: imagery 1.5-5.5 s.
    yang = make_spec(task_s=5.5, class_on_screen_s=1.5, task_from_s=1.5)
    assert training_crop_starts(yang, contract, 0.5) == (1.75, 2.25, 2.75, 3.25)
    # The class stays up all task: the test window is the latest, so it's the only crop.
    cued = make_spec(task_s=4.0, class_on_screen_s=4.0)
    assert training_crop_starts(cued, contract, 0.25) == (2.0,)
    for stride in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(NeurOSConfigError, match="stride"):
            training_crop_starts(spec, contract, stride)
    with pytest.raises(NeurOSConfigError, match="at most"):
        training_crop_starts(make_spec(task_s=600.0), contract, 0.01)
    assert MAX_CROPS_PER_TRIAL >= 16
    fixed = SignalContract(epoch_s=(0.5, 2.5))
    with pytest.raises(NeurOSConfigError, match="fixed"):
        training_crop_starts(spec, fixed, 0.5)


# ---------------------------------------------------------------------- the config
def test_training_windows_config() -> None:
    both = TrainingWindowsConfig(crop_stride_s=0.5, extra_labels=("feet", "tongue"))
    assert both.crop_stride_s == 0.5
    with pytest.raises(ValidationError, match="changes nothing"):
        TrainingWindowsConfig()
    with pytest.raises(ValidationError, match="duplicates"):
        TrainingWindowsConfig(extra_labels=("feet", "feet"))
    with pytest.raises(ValidationError):
        TrainingWindowsConfig(crop_stride_s=0.0)
    base = {"name": "x", "train_datasets": ["A", "B"]}
    plain = PooledExperimentConfig.model_validate(base)
    assert plain.train_labels == LR
    extra = PooledExperimentConfig.model_validate(
        {**base, "training_windows": {"extra_labels": ["feet", "rest"]}}
    )
    assert extra.train_labels == (*LR, "feet", "rest")  # the scored labels keep indices 0, 1
    with pytest.raises(ValidationError, match="repeat"):
        PooledExperimentConfig.model_validate(
            {**base, "training_windows": {"extra_labels": ["left_hand"]}}
        )


# ---------------------------------------------------------------------- extraction
def _recordings(config: DataConfig, subject: int, labels: tuple[str, ...]) -> list[Recording]:
    return [
        preprocess_recording(
            make_recording(
                SyntheticConfig(seed=subject * 10 + run, labels=labels, n_trials=24),
                subject=subject,
                run=str(run),
            ),
            config.signal,
        )
        for run in range(2)
    ]


def test_subject_training_windows_first_crop_is_the_test_window(
    vocab: Vocabulary, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, config = make_spec(), DataConfig()
    labels3 = ("left_hand", "right_hand", "feet")
    maps = {**vocab.event_maps, "Synthetic": {name: name for name in labels3}}
    vocab = vocab.model_copy(update={"event_maps": maps})
    monkeypatch.setattr(
        prepare_module,
        "prepare_subject",
        lambda spec_arg, subject, *a, **k: _recordings(config, subject, labels3),
    )
    test, _ = prepare_module.subject_epochs(spec, 1, vocab, config, LR)
    starts = training_crop_starts(spec, config.signal, 0.25)  # (1.5, 1.75, 2.0)
    windows, primary, summary = subject_training_windows(spec, 1, vocab, config, labels3, starts)
    assert summary["n_trials"] == int(primary.sum())
    assert summary["windows_per_crop"][0] == summary["n_trials"]
    assert len(windows) == sum(summary["windows_per_crop"])
    assert set(windows.labels) == set(labels3)
    # The primary left/right windows are exactly the test windows (same cut, same order).
    first_lr = np.flatnonzero(primary & np.isin(np.array(windows.labels), LR))
    assert np.array_equal(windows.data[first_lr], test.data)
    assert tuple(windows.labels[i] for i in first_lr) == test.labels
    # Every later crop starts 0.25 or 0.5 s (32 or 64 samples at 128 Hz) after a primary
    # window of the same run: the same trial, later, never earlier.
    crops = np.flatnonzero(~primary)
    assert crops.size == summary["windows_per_crop"][1] + summary["windows_per_crop"][2] > 0
    runs = np.array(windows.runs)
    for i in crops:
        firsts = set(windows.starts[primary & (runs == runs[i])].tolist())
        assert {int(windows.starts[i]) - 32, int(windows.starts[i]) - 64} & firsts
    with pytest.raises(NeurOSConfigError, match="test window"):
        subject_training_windows(spec, 1, vocab, config, labels3, (1.75, 2.0))
    with pytest.raises(NeurOSConfigError, match="increase"):
        subject_training_windows(spec, 1, vocab, config, labels3, (1.5, 1.5))
    with pytest.raises(NeurOSConfigError, match="past the task"):
        subject_training_windows(spec, 1, vocab, config, labels3, (1.5, 2.5))
    with pytest.raises(NeurOSConfigError, match="vocabulary"):
        subject_training_windows(spec, 1, vocab, config, ("jump",), starts)


# ---------------------------------------------------------------------- alignment
def _two_sessions(seed: int) -> WindowSet:
    windows = people_windows((0.5, 2.0), per_class=6, seed=seed)
    return dataclasses.replace(windows, datasets=("SynA",) * len(windows))


def test_training_windows_take_the_test_windows_references() -> None:
    test = _two_sessions(seed=1)
    crops = _two_sessions(seed=2)  # other windows of the same people and sessions
    _, references = align_groups(test, unit="session")
    aligned = align_with_references(crops, references, unit="session")
    by_hand = np.concatenate(
        [
            (references[f"SynA:{s}/{sess}"].inv_sqrt @ crops.data[i].astype(np.float64))[None]
            for i, (s, sess) in enumerate(zip(crops.subjects, crops.sessions, strict=True))
        ]
    ).astype(np.float32)
    np.testing.assert_allclose(aligned.data, by_hand, rtol=1e-5, atol=1e-5)
    # Aligned with their own references they would differ: the reference really is shared.
    own, _ = align_groups(crops, unit="session")
    assert not np.allclose(own.data, aligned.data)
    stranger = dataclasses.replace(crops, subjects=crops.subjects + 10)
    with pytest.raises(NeurOSDataError, match="no alignment reference"):
        align_with_references(stranger, references, unit="session")
    # The same through the layout mapping, with CAR first on both.
    x, _, _ = to_layout(crops, LAYOUT, reference="car", alignment="ea", unit="session",
                        references_from=test)  # fmt: skip
    x_self, _, _ = to_layout(test, LAYOUT, reference="car", alignment="ea", unit="session")
    x_test_by_test, _, _ = to_layout(
        test, LAYOUT, reference="car", alignment="ea", unit="session", references_from=test
    )
    assert np.array_equal(x_self, x_test_by_test)  # unchanged when they are the same windows
    assert not np.allclose(x, x_self)
    with pytest.raises(NeurOSDataError, match="same channels"):
        to_layout(crops, LAYOUT, reference="car", alignment="ea", unit="session",
                  references_from=test.pick_channels(["C3", "C4", "Cz"]))  # fmt: skip


def test_assemble_records_primary_windows_and_saves_them(tmp_path: Path) -> None:
    test = _two_sessions(seed=1)
    crops = _two_sessions(seed=2)
    primary = np.arange(len(crops)) % 3 == 0
    pooled, reports = assemble(
        {"SynA": crops},
        LAYOUT,
        reference="car",
        alignment="ea",
        unit="session",
        references_from={"SynA": test},
        primary={"SynA": primary},
    )
    assert np.array_equal(pooled.primary, primary)
    assert reports["SynA"]["n_trials"] == int(primary.sum())
    pooled.save(tmp_path / "train")
    loaded = PooledWindows.load(tmp_path / "train", verify=True)
    assert loaded.primary is not None
    assert np.array_equal(np.asarray(loaded.primary), primary)
    plain, _ = assemble({"SynA": test}, LAYOUT, reference="car", alignment="ea", unit="session")
    assert plain.primary is None
    plain.save(tmp_path / "task")
    assert PooledWindows.load(tmp_path / "task").primary is None
    with pytest.raises(NeurOSDataError, match="cover exactly"):
        assemble({"SynA": crops}, LAYOUT, reference="car", alignment="ea", unit="session",
                 references_from={})  # fmt: skip
    with pytest.raises(NeurOSDataError, match="one flag per window"):
        assemble({"SynA": crops}, LAYOUT, reference="car", alignment="ea", unit="session",
                 primary={"SynA": primary[:-1]})  # fmt: skip
    with pytest.raises(NeurOSDataError, match="primary"):
        dataclasses.replace(pooled, primary=np.zeros(len(pooled), dtype=bool))


# ---------------------------------------------------------------------- loss, data, epochs
def test_masked_cross_entropy_matches_pytorch_when_nothing_is_masked() -> None:
    gen = torch.Generator().manual_seed(0)
    logits = torch.randn(32, 5, generator=gen)
    target = torch.randint(0, 5, (32,), generator=gen)
    everything = torch.ones(32, 5, dtype=torch.bool)
    for smoothing in (0.0, 0.1, 0.3):
        ours = masked_cross_entropy(logits, target, everything, label_smoothing=smoothing)
        theirs = F.cross_entropy(logits, target, label_smoothing=smoothing)
        assert torch.allclose(ours, theirs, atol=1e-6), smoothing


def test_masked_classes_neither_compete_nor_take_gradient() -> None:
    logits = torch.tensor([[2.0, 1.0, 50.0, -3.0]], requires_grad=True)
    target = torch.tensor([0])
    allowed = torch.tensor([[True, True, False, False]])
    loss = masked_cross_entropy(logits, target, allowed, label_smoothing=0.1)
    # Exactly the two-class problem on the allowed logits.
    two = F.cross_entropy(logits[:, :2].detach(), target, label_smoothing=0.1)
    assert torch.allclose(loss.detach(), two, atol=1e-6)
    loss.backward()  # type: ignore[no-untyped-call]
    assert logits.grad is not None
    assert torch.all(logits.grad[0, 2:] == 0)
    assert torch.isfinite(logits.grad).all()
    with pytest.raises(NeurOSDataError, match="match"):
        masked_cross_entropy(logits, target, allowed[:, :3])


def test_tensor_data_checks_its_masks() -> None:
    x = torch.zeros(4, 2, 8)
    y = torch.tensor([0, 1, 2, 0])
    classes = ("a", "b", "c")
    good = torch.tensor([[1, 1, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]], dtype=torch.bool)
    data = TensorData(
        x, y, ("p",) * 4, classes, allowed=good, primary=torch.tensor([1, 0, 1, 1]).bool()
    )
    assert data.n_primary == 3
    assert TensorData(x, y, ("p",) * 4, classes).n_primary == 4
    own_masked = good.clone()
    own_masked[2, 2] = False
    with pytest.raises(NeurOSDataError, match="own class"):
        TensorData(x, y, ("p",) * 4, classes, allowed=own_masked)
    lonely = torch.tensor([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 1, 0]], dtype=torch.bool)
    with pytest.raises(NeurOSDataError, match="at least 2"):
        TensorData(x, y, ("p",) * 4, classes, allowed=lonely)
    with pytest.raises(NeurOSDataError, match="no primary"):
        TensorData(x, y, ("p",) * 4, classes, primary=torch.zeros(4, dtype=torch.bool))
    with pytest.raises(NeurOSDataError, match="allowed must be"):
        TensorData(x, y, ("p",) * 4, classes, allowed=good[:, :2])


def test_an_epoch_counts_trials_not_crops() -> None:
    people = ["a"] * 40 + ["b"] * 40
    labels = [0, 1] * 40
    plain = SubjectBalancedSampler(people, labels, people_per_batch=2, windows_per_person=4, seed=0)
    trials = SubjectBalancedSampler(
        people, labels, people_per_batch=2, windows_per_person=4, seed=0, epoch_windows=20
    )
    assert plain.batches_per_epoch == 10
    assert trials.batches_per_epoch == 3  # ceil(20 / 8)
    # Same draws, only fewer of them per epoch.
    assert all(np.array_equal(a, b) for a, b in zip(trials.epoch(), plain.epoch(), strict=False))
    for bad in (0, 81):
        with pytest.raises(NeurOSConfigError, match="epoch_windows"):
            SubjectBalancedSampler(
                people, labels, people_per_batch=2, windows_per_person=4, seed=0, epoch_windows=bad
            )


# ---------------------------------------------------------------------- end to end
def _dataset(name: str, n_people: int, seed: int, labels: tuple[str, ...] = LR) -> WindowSet:
    scales = tuple(float(s) for s in np.geomspace(0.3, 3.0, n_people))
    windows = people_windows(scales, per_class=6, seed=seed)
    windows = dataclasses.replace(windows, datasets=(name,) * len(windows))
    if labels == LR:
        return windows
    # Relabel every third window as an extra class (the synthetic signal is left/right only;
    # the plumbing, not the learning, is under test here).
    relabelled = tuple(labels[2] if i % 3 == 2 else label for i, label in enumerate(windows.labels))
    return dataclasses.replace(windows, labels=relabelled)


def _settings(**extra: object) -> TaskSettings:
    return TaskSettings(
        classes=LR,
        model=TINY_MODEL,
        train=FAST_TRAIN,
        device="cpu",
        deterministic=True,
        threads=1,
        **extra,  # type: ignore[arg-type]
    )


def test_training_on_a_copy_of_the_test_windows_changes_nothing(tmp_path: Path) -> None:
    # The D-039 plumbing with nothing added (all primary, no extra classes) reproduces the
    # plain run bit for bit: what D-039 runs change is the data, not the machinery.
    data = {"SynA": _dataset("SynA", 6, seed=1), "SynB": _dataset("SynB", 6, seed=2)}
    pooled, _ = assemble(data, LAYOUT, reference="car", alignment="ea", unit="session")
    pooled.save(tmp_path / "task")
    copy, _ = assemble(
        data,
        LAYOUT,
        reference="car",
        alignment="ea",
        unit="session",
        references_from=data,
        primary={key: np.ones(len(w), dtype=bool) for key, w in data.items()},
    )
    copy.save(tmp_path / "train")
    people = pooled.people_by_dataset()
    plan = {"conditions": ("pooled", "single"), "folds": 3, "validation_fraction": 0.2, "seed": 3}
    plain_tasks = plan_tasks(people, ["SynA", "SynB"], **plan)  # type: ignore[arg-type]
    via_train = plan_tasks(people, ["SynA", "SynB"], **plan, train_data="train")  # type: ignore[arg-type]
    folders = {"task": tmp_path / "task", "train": tmp_path / "train"}
    a = run_tasks(plain_tasks, folders, _settings(), workers=1, progress=lambda _: None)
    b = run_tasks(
        via_train, folders, _settings(train_classes=LR), workers=1, progress=lambda _: None
    )
    for x, y in zip(a, b, strict=True):
        assert y["train_data"] == "train"
        assert x["scores"] == y["scores"]
        untimed = [
            [{k: v for k, v in e.items() if k != "seconds"} for e in o["train"]["history"]]
            for o in (x, y)
        ]
        assert untimed[0] == untimed[1]  # identical losses, epoch by epoch


def test_extra_classes_train_but_only_left_right_is_scored(tmp_path: Path) -> None:
    task = {"SynA": _dataset("SynA", 6, seed=1), "SynB": _dataset("SynB", 6, seed=2)}
    train = {
        "SynA": _dataset("SynA", 6, seed=1, labels=(*LR, "feet")),  # SynA also has feet
        "SynB": _dataset("SynB", 6, seed=2),
    }
    pooled, _ = assemble(task, LAYOUT, reference="car", alignment="ea", unit="session")
    pooled.save(tmp_path / "task")
    primary = {k: np.arange(len(w)) % 2 == 0 for k, w in train.items()}  # half are "crops"
    train_pooled, _ = assemble(
        train, LAYOUT, reference="car", alignment="ea", unit="session",
        references_from=task, primary=primary,
    )  # fmt: skip
    train_pooled.save(tmp_path / "train")
    people = pooled.people_by_dataset()
    tasks = plan_tasks(
        people,
        ["SynA", "SynB"],
        conditions=("pooled", "cue_only"),
        folds=3,
        validation_fraction=0.2,
        seed=5,
        train_data="train",
    )
    assert {t.train_data for t in tasks if t.condition == "pooled"} == {"train"}
    assert {t.train_data for t in tasks if t.condition == "cue_only"} == {None}
    settings = _settings(train_classes=(*LR, "feet"))
    folders = {
        "task": tmp_path / "task",
        "cue_only": tmp_path / "task",
        "train": tmp_path / "train",
    }
    outcomes = run_tasks(tasks, folders, settings, workers=1, progress=lambda _: None)
    for outcome in outcomes:
        heads = outcome["model"]["classes"]
        if outcome["condition"] == "pooled":
            assert heads == [*LR, "feet"]
            assert outcome["n_train_trials"] < outcome["n_train_windows"]  # crops don't count
        else:
            assert heads == list(LR)  # the cue-only control keeps E1's recipe
        for score in outcome["scores"]:
            assert 0.0 <= score["balanced_accuracy_raw"] <= 1.0
    with pytest.raises(NeurOSConfigError, match="weren't assembled"):
        run_tasks(tasks, {"task": tmp_path / "task", "cue_only": tmp_path / "task"}, settings)
    with pytest.raises(NeurOSConfigError, match="start with"):
        _settings(train_classes=("feet", *LR))
    with pytest.raises(NeurOSConfigError, match="repeats"):
        ModelTask("t", "pooled", "task", ("SynA:1",), (), ("SynA:2",), 0, "task")


def test_assembling_person_by_person_equals_assembling_at_once() -> None:
    # The CLI places training windows one person at a time (memory, 9 Oct): alignment groups
    # never cross people, so the result must be exactly the all-at-once placement.
    test = _two_sessions(seed=1)
    crops = _two_sessions(seed=2)
    primary = np.arange(len(crops)) % 3 == 0
    placement = {"reference": "car", "alignment": "ea", "unit": "session"}
    whole, _ = assemble(
        {"SynA": crops},
        LAYOUT,
        **placement,  # type: ignore[arg-type]
        references_from={"SynA": test},
        primary={"SynA": primary},
    )
    person_of = np.array(crops.person_ids)
    parts = []
    for person in crops.people:
        part, _ = assemble(
            {"SynA": crops.for_people({person})},
            LAYOUT,
            **placement,  # type: ignore[arg-type]
            references_from={"SynA": test.for_people({person})},
            primary={"SynA": primary[person_of == person]},
        )
        parts.append(part)
    joined = PooledWindows.concat(parts)
    for name in ("x", "present", "labels", "people", "datasets", "flags", "primary"):
        assert np.array_equal(getattr(joined, name), getattr(whole, name)), name
    plain, _ = assemble({"SynA": test}, LAYOUT, **placement)  # type: ignore[arg-type]
    mixed = PooledWindows.concat([plain, parts[0]])  # a part without primary counts all
    assert mixed.primary is not None
    assert mixed.primary[: len(plain)].all()
    with pytest.raises(NeurOSDataError, match="no pooled"):
        PooledWindows.concat([])
    other = dataclasses.replace(plain, x=plain.x[:, :3], present=plain.present[:, :3],
                                layout=LAYOUT[:3])  # fmt: skip
    with pytest.raises(NeurOSDataError, match="different layouts"):
        PooledWindows.concat([plain, other])
