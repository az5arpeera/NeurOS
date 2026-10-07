"""CLI smoke tests (no network) and provenance helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from neuros.cli import _baseline_output_name, _window_overrides, build_parser, main
from neuros.config.experiment import apply_signal_overrides
from neuros.config.schemas import SignalContract
from neuros.data.registry import get_spec, task_epoch
from neuros.provenance import config_hash, file_sha256, stamp

from .conftest import REPO_ROOT


def test_cli_datasets(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["datasets"]) == 0
    assert "PhysionetMI" in capsys.readouterr().out


def test_cli_datasets_check(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["datasets", "--check"]) == 0
    assert "registry check: OK" in capsys.readouterr().out


def test_cli_montage(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["montage"]) == 0
    out = capsys.readouterr().out
    assert "C-core (26 channels)" in out
    assert "headset:ganglion" in out


def test_cli_split_check_and_write(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    splits = REPO_ROOT / "configs" / "splits"
    v1, v3 = splits / "locked_test_v1.yaml", splits / "locked_test_v3.yaml"
    assert main(["split", "--path", str(v3)]) == 0  # the current version (D-038)
    out = capsys.readouterr().out
    assert "locked test set v3 (D-038)" in out
    assert "integrity OK; matches" in out
    assert "total: 111 people" in out
    assert main(["split", "--split-version", "1", "--path", str(v1)]) == 0
    assert "total: 79 people" in capsys.readouterr().out
    assert main(["split", "--path", str(v1)]) == 1  # v1's file isn't the current version
    assert "DIFFERS FROM" in capsys.readouterr().out
    target = tmp_path / "split.yaml"
    args = ["split", "--path", str(target), "--write", "--extends", str(v1)]
    assert main(args) == 0
    assert target.is_file()
    assert main(args) == 2  # refuses to overwrite
    assert main(["split", "--split-version", "7"]) == 2


def test_cli_filters(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["filters"]) == 0
    out = capsys.readouterr().out
    for rate in ("160", "200", "250", "1000"):
        assert any(line.startswith(rate + " ") for line in out.splitlines()), rate
    assert "headset:ganglion" in out
    assert "PROBLEM" not in out


def test_cli_baseline_refuses_locked_test_people(capsys: pytest.CaptureFixture[str]) -> None:
    # PhysionetMI 4 is in the locked test set (D-026): no development run may touch it.
    split = REPO_ROOT / "configs" / "splits" / "locked_test_v1.yaml"
    args = ["baseline", "--dataset", "PhysionetMI", "--subjects", "4", "--split", str(split)]
    assert main(args) == 2
    assert "locked test people" in capsys.readouterr().err


def test_cli_baseline_rejects_bad_windows(capsys: pytest.CaptureFixture[str]) -> None:
    # Checked before any data loads: end before start, shorter than one hop, not finite.
    for window in (["2.5", "0.5"], ["0.5", "0.51"], ["0.5", "inf"], ["nan", "1.0"]):
        assert main(["baseline", "--dataset", "BNCI2014_001", "--window", *window]) == 2, window
        assert "invalid signal overrides" in capsys.readouterr().err


def test_baseline_output_names_never_clobber() -> None:
    spec = get_spec("BNCI2014_001")

    def name(*extra: str) -> str:
        args = build_parser().parse_args(["baseline", "--dataset", spec.key, *extra])
        contract = SignalContract()
        if args.window is not None:
            contract = apply_signal_overrides(contract, _window_overrides(args.window))
        return _baseline_output_name(spec.key, args, task_epoch(spec, contract))

    stem = "BNCI2014_001_left_hand-right_hand_all_as_recorded"
    # Every name carries its epoch, so no run can overwrite a pre-D-030 file (which has none).
    assert name() == f"{stem}_aftercue.json"
    assert name("--alignment-unit", "session") == f"{stem}_session_aftercue.json"
    assert (
        name("--alignment-unit", "session", "--window", "1.5", "3.5")
        == f"{stem}_session_window1.5-3.5s.json"
    )
    assert name("--window", "-0.5", "0") == f"{stem}_window-0.5-0s.json"
    assert name("--subjects", "1", "2") == f"{stem}_aftercue_subj1-2.json"
    # E0b: a non-default band, and the motor strip (D-031), get their own files.
    assert name("--band", "8", "30") == f"{stem}_aftercue_band8-30.json"
    motor = "BNCI2014_001_left_hand-right_hand_motor_car_aftercue.json"
    assert name("--channels", "motor", "--reference", "car") == motor


def test_cli_baseline_rejects_bad_bands(capsys: pytest.CaptureFixture[str]) -> None:
    # Checked against the whole contract before any data loads.
    for band in (["30", "8"], ["0", "30"], ["8", "80"]):
        assert main(["baseline", "--dataset", "BNCI2014_001", "--band", *band]) == 2, band
        assert "invalid signal overrides" in capsys.readouterr().err


def test_cli_loso_refuses_locked_test_people(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("torch")
    split = REPO_ROOT / "configs" / "splits" / "locked_test_v1.yaml"
    experiment = tmp_path / "e.yaml"
    experiment.write_text("name: e\ndataset: PhysionetMI\n", encoding="utf-8")
    args = ["loso", "--experiment", str(experiment), "--subjects", "4", "--split", str(split)]
    assert main(args) == 2
    assert "locked test people" in capsys.readouterr().err


def test_cli_e1_refuses_restricted_or_missing_data(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("torch")
    from neuros.data import registry

    # Safety net: if a guard ever failed, MOABB must not start a real download from a test.
    def no_moabb(*args: object, **kwargs: object) -> None:
        raise AssertionError("E1 reached MOABB: a guard failed")

    monkeypatch.setattr(registry, "moabb_dataset", no_moabb)
    restricted = tmp_path / "e1.yaml"
    restricted.write_text("name: e1\ntrain_datasets: [PhysionetMI, Shin2017A]\n", encoding="utf-8")
    assert main(["e1", "--experiment", str(restricted)]) == 2  # D-033: GPL, test-only
    assert "may not train" in capsys.readouterr().err
    # Cho2017 and Dreyer2023 aren't downloaded here: E1 refuses rather than let MOABB fetch them.
    missing = tmp_path / "e1_missing.yaml"
    missing.write_text(
        "name: e1_missing\ntrain_datasets: [Cho2017, Dreyer2023]\nconditions: [pooled]\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    args = ["e1", "--experiment", str(missing), "--max-epochs", "1"]
    monkeypatch.setenv("NEUROS_DATA_DIR", str(out / "data"))
    monkeypatch.setenv("NEUROS_CACHE_DIR", str(out / "cache"))
    monkeypatch.setenv("NEUROS_RESULTS_DIR", str(out / "results"))
    assert main(args) == 2
    assert "isn't downloaded" in capsys.readouterr().err
    assert main(["e1", "--experiment", str(missing), "--max-subjects", "2"]) == 2  # < 5 folds
    assert "--max-subjects" in capsys.readouterr().err


def test_cli_e1_refuses_a_dataset_the_layout_cant_hold_before_loading(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # 6 Oct: E1 stopped at assembly, after all its data preparation, because Shin2017A's 10-5
    # cap shared only Cz with the motor strip (Q27). The registry says so up front. Since D-037
    # Shin2017A derives C3/C4, so the test removes that profile to recreate the misfit.
    pytest.importorskip("torch")
    import dataclasses
    from types import MappingProxyType

    from neuros.data import registry

    def no_moabb(*args: object, **kwargs: object) -> None:
        raise AssertionError("E1 reached MOABB: the layout check came too late")

    monkeypatch.setattr(registry, "moabb_dataset", no_moabb)
    without_profile = dataclasses.replace(registry.DATASETS["Shin2017A"], derived_channels={})
    monkeypatch.setattr(
        registry, "DATASETS", MappingProxyType({**registry.DATASETS, "Shin2017A": without_profile})
    )
    experiment = tmp_path / "e1_shin.yaml"
    experiment.write_text(
        "name: e1_shin\ntrain_datasets: [BNCI2014_001, PhysionetMI]\n"
        "test_only_datasets: [Shin2017A]\nchannels: motor\nconditions: [pooled]\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("NEUROS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("NEUROS_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("NEUROS_RESULTS_DIR", str(tmp_path / "results"))
    assert main(["e1", "--experiment", str(experiment)]) == 2
    err = capsys.readouterr().err
    assert "Shin2017A has ['Cz']" in err
    for fits in ("BNCI2014_001", "PhysionetMI"):  # they fit the motor strip
        assert fits not in err
    assert not (tmp_path / "cache").exists()  # refused before anything was prepared


def test_cli_loso_validates_the_experiment(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    e0 = str(REPO_ROOT / "configs" / "e0.yaml")
    assert main(["loso", "--experiment", str(tmp_path / "missing.yaml")]) == 2
    assert "not found" in capsys.readouterr().err
    assert main(["loso", "--experiment", e0, "--seeds", "-1"]) == 2
    assert "override" in capsys.readouterr().err
    assert main(["loso", "--experiment", e0, "--threads", "0"]) == 2
    assert ">= 1" in capsys.readouterr().err


def test_cli_unknown_dataset(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["download", "--dataset", "Nope", "--verify"]) == 2
    assert "Unknown dataset" in capsys.readouterr().err


def test_config_hash_ignores_key_order() -> None:
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
    assert config_hash(SignalContract()) != config_hash(SignalContract(reference="as_recorded"))


def test_file_sha256(tmp_path: Path) -> None:
    file = tmp_path / "x.bin"
    file.write_bytes(b"neuros" * 1000)
    assert file_sha256(file) == hashlib.sha256(b"neuros" * 1000).hexdigest()


def test_stamp_has_reproducibility_fields() -> None:
    block = stamp(SignalContract(), seed="7")
    expected = {"created_utc", "git_sha", "git_dirty", "git_dirty_code", "versions"}
    assert expected | {"config_sha256", "seed"} <= set(block)
    assert block["git_dirty_code"] in {"true", "false", "unknown"}
    assert block["versions"]["numpy"] != "absent"
