"""Dataset downloads through MOABB, plus our own integrity manifest (architecture §4.5).

Two things MOABB doesn't give us, added here:

1. **Location control.** MOABB/MNE resolve download folders from ``MNE_DATA`` and per-dataset
   ``MNE_DATASETS_<SIGN>_PATH`` settings, and may write them into ``~/.mne`` on first use. We set
   them as environment variables of *this process only*, pointing at ``<data_dir>/raw``, so
   nothing global changes and the cluster can use a different disk.
2. **Integrity.** MOABB 1.7 downloads with TLS verification disabled and no known hashes on first
   download. We record each raw file's size and SHA-256 in ``<data_dir>/manifests/<key>.json``.
   Later downloads or ``verify_manifest`` compare against it, so a corrupted or silently
   re-released file is caught before it reaches a model.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from neuros.data.registry import DatasetSpec, moabb_dataset
from neuros.errors import DataContext, NeurOSDataError
from neuros.provenance import file_sha256, stamp

log = logging.getLogger(__name__)

# MOABB "signifiers" known to be used by our datasets. Any dataset whose signifier isn't listed
# still lands in MNE_DATA (MNE falls back to it when the per-dataset key is unset).
_KNOWN_SIGNS = ("BNCI", "EEGBCI", "GIGADB")


def prepare_mne_environment(data_dir: Path) -> Path:
    """Point MNE/MOABB downloads at ``<data_dir>/raw`` for this process; return that folder."""
    raw_dir = (data_dir / "raw").resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MNE_DATA"] = str(raw_dir)
    for sign in _KNOWN_SIGNS:
        os.environ[f"MNE_DATASETS_{sign}_PATH"] = str(raw_dir)
    return raw_dir


def _flatten(paths: Any) -> list[Path]:
    """MOABB returns str, lists, or nested lists of paths depending on the dataset."""
    if isinstance(paths, (str, os.PathLike)):
        return [Path(paths)]
    if isinstance(paths, Iterable):
        flat: list[Path] = []
        for item in paths:
            flat.extend(_flatten(item))
        return flat
    raise NeurOSDataError(f"Unexpected path type from MOABB: {type(paths).__name__}")


def manifest_path(data_dir: Path, spec: DatasetSpec) -> Path:
    return data_dir / "manifests" / f"{spec.key}.json"


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise NeurOSDataError(f"Manifest {path} is not valid JSON: {exc}") from exc
    if not isinstance(loaded, dict):
        raise NeurOSDataError(f"Manifest {path} must hold a JSON object")
    return loaded


def download(spec: DatasetSpec, subjects: Sequence[int], data_dir: Path) -> dict[str, Any]:
    """Download ``subjects`` of ``spec`` (if not present) and update the integrity manifest.

    Files already in the manifest must still hash to the recorded value.

    Raises:
        NeurOSDataError: on an unknown subject, a missing file after download, or a hash that
            differs from the manifest.
    """
    unknown = sorted(set(subjects) - set(spec.subjects))
    if unknown:
        raise NeurOSDataError(f"Subjects {unknown} are not in the registry", DataContext(spec.key))
    raw_dir = prepare_mne_environment(data_dir)
    dataset = moabb_dataset(spec)
    path = manifest_path(data_dir, spec)
    manifest = _load_manifest(path)
    files: dict[str, dict[str, Any]] = dict(manifest.get("files", {}))
    for subject in subjects:
        context = DataContext(spec.key, subject)
        log.info("%s subject %s: fetching", spec.key, subject)
        for file in _flatten(dataset.data_path(subject)):
            if not file.is_file():
                raise NeurOSDataError(f"MOABB reported {file} but it doesn't exist", context)
            key = file.resolve().relative_to(raw_dir).as_posix()
            digest = file_sha256(file)
            previous = files.get(key)
            if previous is not None and previous["sha256"] != digest:
                raise NeurOSDataError(
                    f"{key} changed since it was first recorded (sha256 "
                    f"{str(previous['sha256'])[:12]}… → {digest[:12]}…)",
                    context,
                )
            files[key] = {"subject": subject, "bytes": file.stat().st_size, "sha256": digest}
    body = {
        "dataset": spec.key,
        "moabb_code": str(dataset.code),
        "raw_dir_layout": "paths are relative to <data_dir>/raw",
        "n_files": len(files),
        "total_bytes": sum(int(f["bytes"]) for f in files.values()),
        "provenance": manifest.get("provenance") or stamp(),
        "files": dict(sorted(files.items())),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=1), encoding="utf-8", newline="\n")
    return body


def verify_manifest(spec: DatasetSpec, data_dir: Path) -> list[str]:
    """Re-hash every file in the manifest; return a list of problems (empty = intact)."""
    raw_dir = (data_dir / "raw").resolve()
    manifest = _load_manifest(manifest_path(data_dir, spec))
    if not manifest:
        return [f"{spec.key}: no manifest; run `neuros download --dataset {spec.key}` first"]
    problems: list[str] = []
    for key, entry in manifest.get("files", {}).items():
        file = raw_dir / key
        if not file.is_file():
            problems.append(f"{key}: missing")
        elif file_sha256(file) != entry["sha256"]:
            problems.append(f"{key}: sha256 mismatch")
    return problems
