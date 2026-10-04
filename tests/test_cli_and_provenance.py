"""CLI smoke tests (no network) and provenance helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from neuros.cli import main
from neuros.config.schemas import SignalContract
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
    committed = REPO_ROOT / "configs" / "splits" / "locked_test_v1.yaml"
    assert main(["split", "--path", str(committed)]) == 0
    assert "integrity OK; matches" in capsys.readouterr().out
    target = tmp_path / "split.yaml"
    assert main(["split", "--path", str(target), "--write"]) == 0
    assert target.is_file()
    assert main(["split", "--path", str(target), "--write"]) == 2  # refuses to overwrite


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
