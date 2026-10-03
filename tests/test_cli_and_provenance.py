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
    assert {"created_utc", "git_sha", "git_dirty", "versions", "config_sha256", "seed"} <= set(
        block
    )
    assert block["versions"]["numpy"] != "absent"
