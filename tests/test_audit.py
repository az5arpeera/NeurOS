"""Cue-contamination audit (D-030): windows, scalp regions, findings and the CLI's guards."""

from __future__ import annotations

import numpy as np
import pytest

from neuros.cli import main
from neuros.config.schemas import SignalContract
from neuros.data.channels import require_canonical
from neuros.data.montage import SCALP_REGIONS, region_channels
from neuros.data.registry import TaskEpoch, get_spec
from neuros.errors import NeurOSConfigError
from neuros.eval.audit import audit_findings, audit_windows, run_audit
from neuros.signal.windows import WindowSet

from .conftest import REPO_ROOT

# Two channels per region, so every region can be scored.
CHANNELS = ("Fp1", "Fp2", "C3", "Cz", "C4", "P3", "P4", "O1", "O2")
SPLIT = str(REPO_ROOT / "configs" / "splits" / "locked_test_v2.yaml")


def _people(
    signal_on: tuple[str, str] | None, *, n_people: int = 5, per_class: int = 24, seed: int = 0
) -> WindowSet:
    """Synthetic people whose class is a 10 Hz burst on one channel per class, or pure noise.

    ``signal_on = (left_hand channel, right_hand channel)``. Every person has their own
    per-channel gains: the cross-person shift that EA removes.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(64) / 128.0
    data, labels, subjects = [], [], []
    for subject in range(1, n_people + 1):
        gains = np.exp(rng.uniform(np.log(0.5), np.log(2.0), len(CHANNELS)))
        for index, label in enumerate(("left_hand", "right_hand")):
            for _ in range(per_class):
                window = rng.normal(0.0, 1.0, (len(CHANNELS), 64))
                if signal_on is not None:
                    row = CHANNELS.index(signal_on[index])
                    window[row] += 3.0 * np.sin(2 * np.pi * 10.0 * t + rng.uniform(0, 2 * np.pi))
                data.append(window * gains[:, None])
                labels.append(label)
                subjects.append(subject)
    n = len(data)
    return WindowSet(
        data=np.asarray(data, dtype=np.float32),
        sfreq=128.0,
        ch_names=CHANNELS,
        labels=tuple(labels),
        datasets=("Synthetic",) * n,
        subjects=np.asarray(subjects, dtype=np.int64),
        sessions=("0",) * n,
        runs=("0",) * n,
        starts=np.zeros(n, dtype=np.int64),
        flags=np.zeros(n, dtype=np.uint8),
    )


def _epochs(key: str) -> dict[str, TaskEpoch]:
    return {w.name: w.epoch for w in audit_windows(get_spec(key), SignalContract())}


def _means(result: dict[str, object]) -> dict[tuple[str, str], float]:
    cells = result["cells"]
    assert isinstance(cells, list)
    return {(c["window"], c["region"]): c["summary_balanced_accuracy"]["mean"] for c in cells}


# The explicit 10-10 lists the 4 Oct audits used. The row rules must reproduce them exactly on
# 10-10 names (plus P9/P10, which now count as parietal), so those results stay comparable.
AUDIT_4_OCT_REGIONS = {
    "eyes": ("Fp1", "Fpz", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8", "F9", "F7", "F8", "F10"),
    "motor": (
        "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "C5", "C3", "C1", "Cz", "C2", "C4",
        "C6", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
    ),
    "parietal": ("P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8"),
    "visual": (
        "PO9", "PO7", "PO5", "PO3", "POz", "PO4", "PO6", "PO8", "PO10", "O1", "Oz", "O2", "Iz",
    ),
}  # fmt: skip


def test_scalp_regions_are_canonical_and_disjoint() -> None:
    from neuros.data.channels import canonical_names

    every = sorted(canonical_names())
    members = {region: set(region_channels(region, every)) for region in SCALP_REGIONS}
    for region, names in members.items():
        require_canonical(names)
        for other, other_names in members.items():
            assert other == region or not names & other_names, (region, other)
    assert {"C3", "Cz", "C4", "FCC3h", "CCP4h"} <= members["motor"]  # 10-5 half positions too
    assert {"AFp1", "AFp2"} <= members["eyes"]
    assert {"PPO1h", "P9"} <= members["parietal"]
    assert "POO1" in members["visual"]
    assert region_channels("motor", ("O1", "C4", "Fz", "C3")) == ("C4", "C3")  # caller's order
    with pytest.raises(NeurOSConfigError, match="unknown scalp region"):
        region_channels("temporal", CHANNELS)


def test_scalp_regions_match_the_4_oct_audits_on_10_10_names() -> None:
    ten_ten = set().union(*(set(get_spec(k).eeg_channels) for k in ("PhysionetMI", "Lee2019_MI")))
    ten_ten |= set().union(*AUDIT_4_OCT_REGIONS.values())
    names = sorted(ten_ten - {"P9", "P10"})
    for region, listed in AUDIT_4_OCT_REGIONS.items():
        assert set(region_channels(region, names)) == set(listed) & set(names), region


def test_scalp_regions_cover_a_10_5_cap() -> None:
    # Shin2017A samples the motor strip only at half positions: the row rules still find it.
    shin = get_spec("Shin2017A").eeg_channels
    assert region_channels("motor", shin) == (
        "FCC3h", "FCC4h", "FCC5h", "FCC6h", "Cz", "CCP3h", "CCP4h", "CCP5h", "CCP6h",
    )  # fmt: skip
    assert region_channels("eyes", shin) == ("AFp1", "AFp2", "F7", "F8")
    assert region_channels("visual", shin) == ("POO1", "POO2")


def test_audit_windows_follow_d030() -> None:
    windows = audit_windows(get_spec("BNCI2014_001"), SignalContract())
    assert [(w.name, w.epoch.start_s, w.epoch.end_s, w.epoch.cue_free) for w in windows] == [
        ("pre_cue", -0.5, 0.0, True),
        ("cue_onset", 0.0, 0.5, False),
        ("task", 1.5, 3.5, True),
    ]
    assert windows[0].contract.window_samples == 64  # each probe gets its own window length
    task = audit_windows(get_spec("PhysionetMI"), SignalContract())[-1].epoch
    assert (task.start_s, task.end_s, task.cue_free) == (1.0, 3.0, False)
    with pytest.raises(NeurOSConfigError, match="cued motor-imagery"):
        audit_windows(get_spec("EEGMAT"), SignalContract())


def test_audit_traces_the_class_to_its_region() -> None:
    sets = {
        "pre_cue": _people(None, seed=1),  # nothing to decode yet
        "cue_onset": _people(("O2", "O1"), seed=2),  # the cue, seen by the visual cortex
        "task": _people(("C4", "C3"), seed=3),  # imagery, on the motor strip
    }
    result = run_audit(sets, _epochs("BNCI2014_001"))
    means = _means(result)
    assert means[("task", "motor")] > 0.85
    assert means[("task", "visual")] < 0.7
    assert means[("cue_onset", "visual")] > 0.85
    assert abs(means[("pre_cue", "all")] - 0.5) < 0.15
    assert result["skipped_cells"] == []
    levels = [f["level"] for f in result["findings"]]
    assert "leak" not in levels
    assert "cue" not in levels
    messages = " ".join(f["message"] for f in result["findings"])
    assert "motor strip leads" in messages
    assert "on screen" not in messages  # BNCI2014-001's task window is cue-free


def test_audit_flags_a_leak_and_a_cue_dominated_task_window() -> None:
    sets = {
        "pre_cue": _people(("C4", "C3"), seed=4),  # the class before the cue: a leak
        "task": _people(("Fp2", "Fp1"), seed=5),  # the eyes carry the class
    }
    result = run_audit(sets, _epochs("PhysionetMI"))
    levels = [f["level"] for f in result["findings"]]
    assert "leak" in levels
    assert "cue" in levels
    # PhysionetMI's task window keeps the target on screen: the D-030 flag is reported.
    assert any("on screen" in f["message"] for f in result["findings"])


def test_small_pre_cue_effects_warn_and_large_ones_leak() -> None:
    def pre_cue(mean: float, low: float) -> dict[str, object]:
        return {
            "window": "pre_cue",
            "region": "all",
            "cue_free": True,
            "summary_balanced_accuracy": {"mean": mean, "ci95_low": low},
        }

    def levels(*cells: dict[str, object]) -> list[str]:
        return [f["level"] for f in audit_findings(cells)]

    # BNCI2014-004 (4 Oct): significant, but too small for a timing or label bug.
    assert levels(pre_cue(0.514, 0.507)) == ["warn"]
    assert levels(pre_cue(0.56, 0.53)) == ["leak"]
    assert levels(pre_cue(0.52, 0.48)) == ["ok"]


def test_regions_with_too_few_channels_are_skipped_and_reported() -> None:
    task = _people(("C4", "C3"), seed=6).pick_channels(("Fp1", "C3", "Cz", "C4", "O1"))
    result = run_audit({"task": task}, _epochs("BNCI2014_001"))
    skipped = {(s["window"], s["region"]) for s in result["skipped_cells"]}
    assert skipped == {("task", "eyes"), ("task", "parietal"), ("task", "visual")}
    assert set(_means(result)) == {("task", "all"), ("task", "motor")}


def test_audit_reports_progress_after_every_cell() -> None:
    # An interrupted audit keeps the cells it finished: the CLI saves them on each call.
    calls: list[tuple[int, int]] = []
    run_audit(
        {"task": _people(("C4", "C3"), seed=7)},
        _epochs("BNCI2014_001"),
        on_cell=lambda cells, skipped: calls.append((len(cells), len(skipped))),
    )
    assert calls == [(1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]  # all, eyes, motor, parietal, visual


def test_run_audit_rejects_inconsistent_inputs() -> None:
    with pytest.raises(NeurOSConfigError, match="no epoch"):
        run_audit({"surprise": _people(None)}, _epochs("BNCI2014_001"))
    other = _people(None).pick_channels(("C3", "Cz", "C4"))
    with pytest.raises(NeurOSConfigError, match="one channel list"):
        run_audit({"pre_cue": _people(None), "task": other}, _epochs("BNCI2014_001"))


def test_cli_audit_guards(capsys: pytest.CaptureFixture[str]) -> None:
    # All refused before any data loads.
    assert main(["audit", "--dataset", "EEGMAT", "--split", SPLIT]) == 2
    assert "cued motor-imagery" in capsys.readouterr().err
    assert main(["audit", "--dataset", "BNCI2014_001", "--workers", "0"]) == 2
    assert "--workers" in capsys.readouterr().err
    assert main(["baseline", "--dataset", "BNCI2014_001", "--workers", "0"]) == 2
    assert "--workers" in capsys.readouterr().err
    assert main(["audit", "--dataset", "BNCI2014_001", "--labels", "left_hand"]) == 2
    assert "--labels" in capsys.readouterr().err
    # PhysionetMI 4 is in the locked test set (D-026).
    assert main(["audit", "--dataset", "PhysionetMI", "--subjects", "4", "--split", SPLIT]) == 2
    assert "locked test people" in capsys.readouterr().err
