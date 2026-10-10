"""Downloads: subject file selection, publisher archive checksums, licence acceptance (no network).

The MOABB side is replaced by small fakes: these tests check our integrity layer, not MOABB.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from neuros.data import download as download_module
from neuros.data.download import (
    download,
    file_md5,
    publisher_checksums,
    subject_files,
    verify_archives,
    verify_manifest,
)
from neuros.data.registry import get_spec
from neuros.errors import NeurOSConfigError, NeurOSDataError

from .conftest import make_spec


@pytest.fixture(autouse=True)
def _restore_mne_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """``download`` points MNE at the test's folder through environment variables; undo it."""
    import os

    names = (
        "MNE_DATA",
        "MOABB_DOWNLOAD_PROVIDER",
        *(f"MNE_DATASETS_{sign}_PATH" for sign in download_module._KNOWN_SIGNS),
    )
    for name in names:
        if name in os.environ:
            monkeypatch.setenv(name, os.environ[name])  # restored at teardown
        else:
            monkeypatch.delenv(name, raising=False)  # removed again at teardown


def _write(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _md5(content: bytes) -> str:
    return hashlib.md5(content, usedforsecurity=False).hexdigest()


def test_publisher_checksums_are_packaged_and_well_formed() -> None:
    zenodo = publisher_checksums("GuttmannFlury2025")
    assert len(zenodo["files"]) == 31
    assert sum(entry["bytes"] for entry in zenodo["files"].values()) == 26_198_687_829
    assert "zenodo/18970793/S01.zip" in zenodo["files"]
    figshare = publisher_checksums("Yang2025")
    assert figshare["files"]["WBCIC_SHU_Motor_Imagery_dataset.zip"]["bytes"] == 65_630_556_299
    # D-038: MOABB saves Liu2024's three Figshare files under their file ids, not their names.
    liu = publisher_checksums("Liu2024")["files"]
    assert set(liu) == {f"MNE-liu2024-data/files/{i}" for i in (38516654, 38516078, 38516084)}
    assert liu["MNE-liu2024-data/files/38516654"]["bytes"] == 462_843_633  # edffile.zip
    assert publisher_checksums("BNCI2014_001") == {}  # no published archive checksums


def test_subject_files_selects_one_person_from_a_dataset_folder(tmp_path: Path) -> None:
    spec = get_spec("Yang2025")
    root = tmp_path / "MNE-yang2025-data"
    sourcedata = root / "WBCIC_SHU Motor Imagery dataset" / "sourcedata"
    mine = [
        _write(sourcedata / "2C dataset" / "sub-001" / "ses-01" / "eeg" / name, b"x")
        for name in ("data.bdf", "evt.bdf")
    ]
    _write(sourcedata / "2C dataset" / "sub-002" / "ses-01" / "eeg" / "data.bdf", b"y")
    _write(sourcedata / "3C dataset" / "sub-001" / "x.bdf", b"z")  # the other paradigm's 1
    assert subject_files(spec, 1, str(root)) == sorted(mine)
    with pytest.raises(NeurOSDataError, match="No files for the subject"):
        subject_files(spec, 7, str(root))


def test_subject_files_rejects_unpatterned_folders_and_missing_paths(tmp_path: Path) -> None:
    spec = get_spec("BNCI2014_001")
    file = _write(tmp_path / "A01T.mat", b"x")
    assert subject_files(spec, 1, [str(file)]) == [file]
    with pytest.raises(NeurOSDataError, match="no subject file pattern"):
        subject_files(spec, 1, str(tmp_path))
    with pytest.raises(NeurOSDataError, match="doesn't exist"):
        subject_files(spec, 1, [str(tmp_path / "missing.mat")])


def _fake_published(monkeypatch: pytest.MonkeyPatch, key: str, files: dict[str, Any]) -> None:
    table = {"source": "test", "files": files}
    monkeypatch.setattr(download_module, "publisher_checksums", lambda k: table if k == key else {})


def test_verify_archives_matches_size_and_md5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = get_spec("GuttmannFlury2025")
    content = b"archive bytes"
    archive = _write(tmp_path / "MNE-x-data" / "zenodo" / "18970793" / "S01.zip", content)
    assert file_md5(archive) == _md5(content)
    _fake_published(
        monkeypatch,
        spec.key,
        {
            "zenodo/18970793/S01.zip": {"bytes": len(content), "md5": _md5(content)},
            "zenodo/18970793/S02.zip": {"bytes": 5, "md5": "0" * 32},  # not downloaded: skipped
        },
    )
    found = verify_archives(spec, tmp_path)
    key = "MNE-x-data/zenodo/18970793/S01.zip"
    assert list(found) == [key]
    assert found[key]["md5"] == _md5(content)
    # Already verified at this size: not hashed again (65.6 GB takes minutes).
    monkeypatch.setattr(download_module, "file_md5", lambda path: pytest.fail("re-hashed"))
    assert verify_archives(spec, tmp_path, found) == found


def test_verify_archives_rejects_bad_size_md5_and_ambiguity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = get_spec("GuttmannFlury2025")
    content = b"archive bytes"
    suffix = "zenodo/18970793/S01.zip"
    _write(tmp_path / "a" / suffix, content)
    _fake_published(monkeypatch, spec.key, {suffix: {"bytes": 99, "md5": _md5(content)}})
    with pytest.raises(NeurOSDataError, match="incomplete or re-released"):
        verify_archives(spec, tmp_path)
    _fake_published(monkeypatch, spec.key, {suffix: {"bytes": len(content), "md5": "f" * 32}})
    with pytest.raises(NeurOSDataError, match="differs from the publisher"):
        verify_archives(spec, tmp_path)
    _write(tmp_path / "b" / suffix, content)
    with pytest.raises(NeurOSDataError, match="matches 2 files"):
        verify_archives(spec, tmp_path)


class _FakeDataset:
    """Stands in for a MOABB dataset: ``data_path`` "downloads" one small file per subject."""

    code = "Fake"

    def __init__(self, root: Path, **kwargs: Any) -> None:
        self.root, self.kwargs = root, kwargs

    def data_path(self, subject: int) -> list[str]:
        return [str(_write(self.root / f"S{subject}" / "cnt.mat", f"s{subject}".encode()))]


def _patch_moabb(monkeypatch: pytest.MonkeyPatch, raw: Path) -> list[_FakeDataset]:
    made: list[_FakeDataset] = []

    def fake_moabb(spec: object, **kwargs: Any) -> _FakeDataset:
        made.append(_FakeDataset(raw, **kwargs))
        return made[-1]

    monkeypatch.setattr(download_module, "moabb_dataset", fake_moabb)
    return made


def test_download_needs_an_explicit_licence_acceptance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = make_spec(key="Licensed", licence_acceptance="GPL-3.0", moabb_class="Shin2017A")
    made = _patch_moabb(monkeypatch, tmp_path / "raw")
    with pytest.raises(NeurOSConfigError, match="pass --accept-licence"):
        download(spec, [1], tmp_path)
    assert made == []  # nothing was instantiated, let alone downloaded
    manifest = download(spec, [1, 2], tmp_path, accept_licence=True)
    assert made[-1].kwargs == {"accept": True}
    assert manifest["licence_accepted"] == "GPL-3.0"
    assert manifest["n_files"] == 2
    assert verify_manifest(spec, tmp_path) == []
    stored = json.loads((tmp_path / "manifests" / "Licensed.json").read_text(encoding="utf-8"))
    assert sorted(stored["files"]) == ["S1/cnt.mat", "S2/cnt.mat"]


def test_require_downloaded_refuses_what_neuros_download_did_not_fetch(tmp_path: Path) -> None:
    from neuros.data.download import require_downloaded

    spec = get_spec("Cho2017")
    with pytest.raises(NeurOSDataError, match="isn't downloaded"):
        require_downloaded(spec, [1, 2], tmp_path)
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    body = {"files": {"a/s01.mat": {"subject": 1, "bytes": 1, "sha256": "x"}}}
    (manifests / "Cho2017.json").write_text(json.dumps(body), encoding="utf-8")
    require_downloaded(spec, [1], tmp_path)  # no error
    with pytest.raises(NeurOSDataError, match="subjects 2 aren't downloaded"):
        require_downloaded(spec, [1, 2], tmp_path)


def test_download_without_a_licence_passes_no_acceptance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = make_spec(key="Open", moabb_class="BNCI2014_001")
    made = _patch_moabb(monkeypatch, tmp_path / "raw")
    manifest = download(spec, [3], tmp_path, accept_licence=True)  # accepted, but not needed
    assert made[-1].kwargs == {}
    assert "licence_accepted" not in manifest


def test_loading_honours_the_licence_acceptance_recorded_at_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from neuros.data.download import loading_dataset, loading_kwargs

    spec = make_spec(key="Licensed", licence_acceptance="GPL-3.0", moabb_class="Shin2017A")
    made = _patch_moabb(monkeypatch, tmp_path / "raw")
    with pytest.raises(NeurOSDataError, match="isn't downloaded"):
        loading_dataset(spec, tmp_path)
    assert made == []  # refused before anything was instantiated
    download(spec, [1], tmp_path, accept_licence=True)
    loading_dataset(spec, tmp_path)
    assert made[-1].kwargs == {"accept": True}  # MOABB re-checks the acceptance on every load

    path = tmp_path / "manifests" / "Licensed.json"
    stored = json.loads(path.read_text(encoding="utf-8"))
    stored["licence_accepted"] = "GPL-2.0"  # accepted, but not the licence declared today
    path.write_text(json.dumps(stored), encoding="utf-8")
    with pytest.raises(NeurOSConfigError, match="accepted at download"):
        loading_kwargs(spec, tmp_path)
    del stored["licence_accepted"]  # never accepted
    path.write_text(json.dumps(stored), encoding="utf-8")
    with pytest.raises(NeurOSConfigError, match="accepted at download"):
        loading_kwargs(spec, tmp_path)


def test_loading_an_ungated_dataset_needs_no_acceptance(tmp_path: Path) -> None:
    from neuros.data.download import loading_kwargs

    assert loading_kwargs(make_spec(key="Open", moabb_class="BNCI2014_001"), tmp_path) == {}


def test_loading_points_mne_at_our_raw_folder_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 10 Oct: a command that skipped prepare_mne_environment let MOABB look in ~/mne_data and
    # start re-downloading Yang2025 (65.6 GB). Loading now sets the location itself.
    from neuros.data.download import loading_dataset

    spec = make_spec(key="Open", moabb_class="BNCI2014_001")
    _patch_moabb(monkeypatch, tmp_path / "raw")
    download(spec, [1], tmp_path)
    monkeypatch.delenv("MNE_DATA", raising=False)
    monkeypatch.setenv("MOABB_DOWNLOAD_PROVIDER", "mirror")
    loading_dataset(spec, tmp_path)
    assert os.environ["MNE_DATA"] == str((tmp_path / "raw").resolve())
    assert os.environ["MOABB_DOWNLOAD_PROVIDER"] == "upstream"
