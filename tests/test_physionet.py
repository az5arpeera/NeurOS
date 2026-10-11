"""PhysioNet mirror downloads: checksum parsing, verification, retries, MOABB paths (no network)."""

from __future__ import annotations

import hashlib
import io
import urllib.error
from pathlib import Path
from typing import Any

import pytest

from neuros.data import physionet
from neuros.data.physionet import (
    PHYSIONET_PROJECTS,
    PhysioNetProject,
    fetch_subjects,
    fetch_verified,
    moabb_destination,
    parse_sha256sums,
)
from neuros.errors import NeurOSDataError

PAYLOAD = b"EDF payload " * 1000
SHA = hashlib.sha256(PAYLOAD).hexdigest()


class _Response(io.BytesIO):
    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(physionet.time, "sleep", lambda _s: None)


def test_parse_sha256sums() -> None:
    text = f"{SHA} S001/S001R01.edf\n{'0' * 64} *S001/S001R02.edf\n\n"
    assert parse_sha256sums(text) == {"S001/S001R01.edf": SHA, "S001/S001R02.edf": "0" * 64}


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("abc S001.edf\n", "malformed"),
        (f"{'g' * 64} S001.edf\n", "hex"),
        (f"{SHA} a.edf\n{SHA} a.edf\n", "duplicate"),
        ("\n\n", "no checksums"),
    ],
)
def test_parse_sha256sums_rejects_bad_input(text: str, message: str) -> None:
    with pytest.raises(NeurOSDataError, match=message):
        parse_sha256sums(text)


def test_physionet_mi_file_list_matches_moabb_runs() -> None:
    files = PHYSIONET_PROJECTS["PhysionetMI"].files_for_subject(7)
    assert files == [f"S007/S007R{r:02d}.edf" for r in (1, 2, 4, 8, 12, 6, 10, 14)]


def test_moabb_destination_matches_moabb_layout(tmp_path: Path) -> None:
    url = PHYSIONET_PROJECTS["PhysionetMI"].upstream_url("S001/S001R01.edf")
    expected = (
        tmp_path / "MNE-eegbci-data" / "files" / "eegmmidb" / "1.0.0" / "S001" / "S001R01.edf"
    )
    assert moabb_destination(tmp_path, "EEGBCI", url) == expected


def test_fetch_verified_downloads_and_skips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def fake_open(url: str) -> _Response:
        calls.append(url)
        return _Response(PAYLOAD)

    monkeypatch.setattr(physionet, "_open", fake_open)
    dest = tmp_path / "a" / "S001R01.edf"
    assert fetch_verified("https://mirror/x", dest, SHA) == len(PAYLOAD)
    assert dest.read_bytes() == PAYLOAD
    assert not dest.with_name(dest.name + ".part").exists()
    assert fetch_verified("https://mirror/x", dest, SHA) == 0  # already present and correct
    assert len(calls) == 1


def test_fetch_verified_rejects_checksum_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(physionet, "_open", lambda url: _Response(b"tampered"))
    dest = tmp_path / "S001R01.edf"
    with pytest.raises(NeurOSDataError, match="Checksum mismatch"):
        fetch_verified("https://mirror/x", dest, SHA)
    assert not dest.exists()
    assert not dest.with_name(dest.name + ".part").exists()


def test_fetch_verified_retries_transient_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_sleep: None
) -> None:
    attempts = {"n": 0}

    def flaky(url: str) -> _Response:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise urllib.error.URLError("connection reset")
        return _Response(PAYLOAD)

    monkeypatch.setattr(physionet, "_open", flaky)
    assert fetch_verified("https://mirror/x", tmp_path / "f.edf", SHA) == len(PAYLOAD)
    assert attempts["n"] == 3


def test_fetch_verified_gives_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_sleep: None
) -> None:
    def down(url: str) -> Any:
        raise urllib.error.URLError("down")

    monkeypatch.setattr(physionet, "_open", down)
    with pytest.raises(NeurOSDataError, match="Giving up"):
        fetch_verified("https://mirror/x", tmp_path / "f.edf", SHA)


def _project() -> PhysioNetProject:
    return PhysioNetProject(
        "PhysionetMI", "eegmmidb", "1.0.0", "EEGBCI", lambda s: [f"S{s:03d}/a.edf"]
    )


def test_fetch_subjects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(physionet, "_open", lambda url: _Response(PAYLOAD))
    sums = {"S001/a.edf": SHA, "S002/a.edf": SHA}
    verified = fetch_subjects(_project(), [1, 2], tmp_path, sums=sums, workers=2)
    assert verified == {
        "MNE-eegbci-data/files/eegmmidb/1.0.0/S001/a.edf": SHA,
        "MNE-eegbci-data/files/eegmmidb/1.0.0/S002/a.edf": SHA,
    }


def test_fetch_subjects_requires_published_checksums(tmp_path: Path) -> None:
    with pytest.raises(NeurOSDataError, match="no published checksum"):
        fetch_subjects(_project(), [3], tmp_path, sums={"S001/a.edf": SHA})


def test_fetch_subjects_reports_all_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(physionet, "_open", lambda url: _Response(b"bad"))
    sums = {"S001/a.edf": SHA, "S002/a.edf": SHA}
    with pytest.raises(NeurOSDataError, match="2 of 2 downloads failed"):
        fetch_subjects(_project(), [1, 2], tmp_path, sums=sums)


def test_eegmat_files_and_local_destination(tmp_path: Path) -> None:
    project = PHYSIONET_PROJECTS["EEGMAT"]
    assert project.moabb_sign is None
    assert project.files_for_subject(7) == ["Subject07_1.edf", "Subject07_2.edf"]
    destination = physionet.local_destination(tmp_path, project, "Subject07_1.edf")
    assert destination == tmp_path / "physionet-eegmat-1.0.0" / "Subject07_1.edf"


@pytest.mark.parametrize("relpath", ["../escape.edf", "/abs.edf"])
def test_local_destination_refuses_unsafe_paths(tmp_path: Path, relpath: str) -> None:
    with pytest.raises(NeurOSDataError, match="unsafe"):
        physionet.local_destination(tmp_path, PHYSIONET_PROJECTS["EEGMAT"], relpath)


def test_fetch_subjects_without_moabb_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(physionet, "_open", lambda url: _Response(PAYLOAD))
    sums = {"Subject00_1.edf": SHA, "Subject00_2.edf": SHA}
    verified = fetch_subjects(PHYSIONET_PROJECTS["EEGMAT"], [0], tmp_path, sums=sums)
    assert verified == {
        "physionet-eegmat-1.0.0/Subject00_1.edf": SHA,
        "physionet-eegmat-1.0.0/Subject00_2.edf": SHA,
    }
