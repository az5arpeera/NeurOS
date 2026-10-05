"""Dataset downloads through MOABB, plus our own integrity manifest (architecture §4.5).

Two things MOABB doesn't give us, added here:

1. **Location and source control.** MOABB/MNE resolve download folders from ``MNE_DATA`` and
   per-dataset ``MNE_DATASETS_<SIGN>_PATH`` settings, and may write them into ``~/.mne`` on first
   use. We set them as environment variables of *this process only*, pointing at
   ``<data_dir>/raw``, so nothing global changes and the cluster can use a different disk.

   MOABB 1.7 also loads, by default, from a NEMAR mirror of the original files (its
   ``sourcedata/`` store), even when the upstream copies are already on disk. We pin
   ``MOABB_DOWNLOAD_PROVIDER=upstream`` (again per process) so that ``get_data`` loads
   *exactly* the files ``data_path`` returned, which are the files our manifest hashes. Found on
   Day 1: without the pin, BNCI2014-001 was fetched twice (upstream + NEMAR ``nm000139``).
2. **Integrity.** MOABB 1.7 downloads with TLS verification disabled and no known hashes on first
   download. We record each raw file's size and SHA-256 in ``<data_dir>/manifests/<key>.json``.
   Later downloads or ``verify_manifest`` compare against it, so a corrupted or silently
   re-released file is caught before it reaches a model.
3. **Publisher checksums** where the publisher posts them: PhysioNet's SHA256SUMS
   (``neuros.data.physionet``), and the MD5s that Zenodo and Figshare list for archives
   (``checksums/<key>.json``, D-033: GuttmannFlury2025's per-person zips, Yang2025's single
   65.6 GB zip). An archive that doesn't match is an error before anything is recorded.
4. **Licences that must be accepted** (Shin2017A: GPL-3.0) are accepted by the person running
   the download, with ``accept_licence=True`` (``--accept-licence``), never by configuration.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
from collections.abc import Iterable, Mapping, Sequence
from importlib import resources
from pathlib import Path
from types import MappingProxyType
from typing import Any

from neuros.data.physionet import PHYSIONET_PROJECTS, fetch_subjects
from neuros.data.registry import DatasetSpec, moabb_dataset
from neuros.errors import DataContext, NeurOSConfigError, NeurOSDataError
from neuros.provenance import file_sha256, stamp

log = logging.getLogger(__name__)

# MOABB "signifiers" known to be used by our datasets. Any dataset whose signifier isn't listed
# still lands in MNE_DATA (MNE falls back to it when the per-dataset key is unset). MOABB
# upper-cases them: Yang2025 → YANG2025, Shin2017A → BBCIFNIRS, GuttmannFlury2025 → its code.
_KNOWN_SIGNS = ("BNCI", "EEGBCI", "GIGADB", "YANG2025", "BBCIFNIRS", "GUTTMANNFLURY2025-MI")

# Datasets whose MOABB ``data_path`` returns a folder instead of the subject's files: a glob,
# relative to that folder and formatted with the subject number, that selects one subject's
# files. Yang2025 returns the whole dataset's folder; GuttmannFlury2025 the subject's own.
_SUBJECT_FILE_GLOBS: Mapping[str, str] = MappingProxyType(
    {
        "Yang2025": "**/sourcedata/2C dataset/sub-{subject:03d}/**/*",
        "GuttmannFlury2025": "**/*",
    }
)

_HASH_CHUNK_BYTES = 8 * 1024 * 1024


def prepare_mne_environment(data_dir: Path) -> Path:
    """Point MNE/MOABB at ``<data_dir>/raw`` and the upstream source, for this process only.

    Returns the raw-data folder.
    """
    raw_dir = (data_dir / "raw").resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MNE_DATA"] = str(raw_dir)
    for sign in _KNOWN_SIGNS:
        os.environ[f"MNE_DATASETS_{sign}_PATH"] = str(raw_dir)
    # Load what we downloaded and hashed, not a mirror copy (see module docstring).
    os.environ["MOABB_DOWNLOAD_PROVIDER"] = "upstream"
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


def subject_files(spec: DatasetSpec, subject: int, paths: Any) -> list[Path]:
    """The files of one subject, from whatever MOABB's ``data_path`` returned.

    Files are taken as they are; a folder is searched with the dataset's glob
    (``_SUBJECT_FILE_GLOBS``), so a dataset-wide folder never records other people's files.

    Raises:
        NeurOSDataError: if a path doesn't exist, a folder comes from a dataset without a glob,
            or the glob finds nothing for the subject.
    """
    context = DataContext(spec.key, subject)
    files: list[Path] = []
    for path in _flatten(paths):
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            pattern = _SUBJECT_FILE_GLOBS.get(spec.key)
            if pattern is None:
                raise NeurOSDataError(
                    f"MOABB returned the folder {path}, and {spec.key} has no subject file "
                    "pattern (_SUBJECT_FILE_GLOBS)",
                    context,
                )
            matched = sorted(p for p in path.glob(pattern.format(subject=subject)) if p.is_file())
            if not matched:
                raise NeurOSDataError(f"No files for the subject under {path}", context)
            files.extend(matched)
        else:
            raise NeurOSDataError(f"MOABB reported {path} but it doesn't exist", context)
    return files


@functools.cache
def publisher_checksums(key: str) -> Mapping[str, Any]:
    """The publisher's archive checksums for dataset ``key`` (``checksums/<key>.json``), or {}.

    Raises:
        NeurOSConfigError: if the packaged file exists but is malformed.
    """
    resource = resources.files("neuros.data").joinpath("checksums", f"{key}.json")
    if not resource.is_file():
        return {}
    try:
        body = json.loads(resource.read_text(encoding="utf-8"))
        files = body["files"]
        if body["algorithm"] != "md5" or not isinstance(files, dict):
            raise ValueError("expected an 'md5' algorithm and a 'files' object")
        for suffix, entry in files.items():
            if not (isinstance(entry["bytes"], int) and entry["bytes"] > 0):
                raise ValueError(f"{suffix}: 'bytes' must be a positive integer")
            if not (isinstance(entry["md5"], str) and len(entry["md5"]) == 32):
                raise ValueError(f"{suffix}: 'md5' must be 32 hex characters")
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise NeurOSConfigError(f"checksums/{key}.json is malformed: {exc}") from exc
    return MappingProxyType({"source": str(body.get("source", "")), "files": files})


def file_md5(path: Path) -> str:
    """MD5 of a file, read in chunks (only to match publishers that post MD5s)."""
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archives(
    spec: DatasetSpec, raw_dir: Path, known: Mapping[str, Mapping[str, Any]] | None = None
) -> dict[str, dict[str, Any]]:
    """Check every downloaded archive of ``spec`` against the publisher's size and MD5.

    Archives are found under ``raw_dir`` by their path suffix (the keys of
    ``checksums/<key>.json``). One that isn't on disk is skipped: only some people may have
    been fetched so far. ``known`` (from the manifest) skips re-hashing an archive already
    verified at the same size; hashing 65.6 GB takes minutes.

    Returns:
        Manifest entries for the archives found, keyed by path relative to ``raw_dir``.

    Raises:
        NeurOSDataError: if an archive's size or MD5 differs from the publisher's, or a suffix
            matches more than one file.
    """
    published = publisher_checksums(spec.key)
    if not published:
        return {}
    known = known or {}
    found: dict[str, dict[str, Any]] = {}
    for suffix, expected in published["files"].items():
        name = Path(suffix).name
        matches = [p for p in raw_dir.rglob(name) if p.as_posix().endswith("/" + suffix)]
        if not matches:
            continue
        if len(matches) > 1:
            raise NeurOSDataError(
                f"{suffix} matches {len(matches)} files under {raw_dir}", DataContext(spec.key)
            )
        file = matches[0]
        key = file.resolve().relative_to(raw_dir.resolve()).as_posix()
        size = file.stat().st_size
        if size != expected["bytes"]:
            raise NeurOSDataError(
                f"{key}: {size} bytes, but the publisher lists {expected['bytes']} (incomplete "
                "or re-released download)",
                DataContext(spec.key),
            )
        previous = known.get(key)
        if previous is not None and previous.get("bytes") == size and previous.get("md5"):
            found[key] = dict(previous)
            continue
        log.info("%s: checking %s against the publisher's MD5", spec.key, key)
        digest = file_md5(file)
        if digest != expected["md5"]:
            raise NeurOSDataError(
                f"{key}: MD5 {digest} differs from the publisher's {expected['md5']}",
                DataContext(spec.key),
            )
        found[key] = {"bytes": size, "md5": digest, "verified_against": published["source"]}
    return found


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


def download(
    spec: DatasetSpec,
    subjects: Sequence[int],
    data_dir: Path,
    *,
    accept_licence: bool = False,
) -> dict[str, Any]:
    """Download ``subjects`` of ``spec`` (if not present) and update the integrity manifest.

    Files already in the manifest must still hash to the recorded value. Archives with
    publisher checksums are verified first (``verify_archives``).

    Args:
        accept_licence: the person running the download accepts ``spec.licence_acceptance``
            (required for such datasets; MOABB then gets ``accept=True``).

    Raises:
        NeurOSConfigError: if the dataset's licence must be accepted and ``accept_licence`` is
            False.
        NeurOSDataError: on an unknown subject, a missing file after download, an archive that
            doesn't match its publisher's checksum, or a hash that differs from the manifest.
    """
    unknown = sorted(set(subjects) - set(spec.subjects))
    if unknown:
        raise NeurOSDataError(f"Subjects {unknown} are not in the registry", DataContext(spec.key))
    extra: dict[str, Any] = {}
    if spec.licence_acceptance is not None:
        if not accept_licence:
            raise NeurOSConfigError(
                f"{spec.key}: downloading means accepting its licence, {spec.licence_acceptance}. "
                "Read it; if you accept it, pass --accept-licence."
            )
        extra["accept"] = True
        log.info("%s: licence accepted by the caller: %s", spec.key, spec.licence_acceptance)
    raw_dir = prepare_mne_environment(data_dir)
    dataset = moabb_dataset(spec, **extra)
    path = manifest_path(data_dir, spec)
    manifest = _load_manifest(path)
    files: dict[str, dict[str, Any]] = dict(manifest.get("files", {}))
    archives: dict[str, dict[str, Any]] = dict(manifest.get("archives", {}))

    # PhysioNet datasets: prefetch from PhysioNet's mirror, verified against the publisher's
    # checksums (neuros.data.physionet). MOABB's data_path below then finds the files locally.
    publisher_sums: dict[str, str] = {}
    project = PHYSIONET_PROJECTS.get(spec.key)
    if project is not None:
        publisher_sums = fetch_subjects(project, subjects, raw_dir)

    for subject in subjects:
        context = DataContext(spec.key, subject)
        log.info("%s subject %s: fetching", spec.key, subject)
        fetched = dataset.data_path(subject)
        # Archives first: nothing is recorded from an archive that fails its checksum.
        archives.update(verify_archives(spec, raw_dir, archives))
        for file in subject_files(spec, subject, fetched):
            key = file.resolve().relative_to(raw_dir).as_posix()
            digest = file_sha256(file)
            previous = files.get(key)
            if previous is not None and previous["sha256"] != digest:
                raise NeurOSDataError(
                    f"{key} changed since it was first recorded (sha256 "
                    f"{str(previous['sha256'])[:12]}… → {digest[:12]}…)",
                    context,
                )
            entry: dict[str, Any] = {
                "subject": subject,
                "bytes": file.stat().st_size,
                "sha256": digest,
            }
            if key in publisher_sums:
                if publisher_sums[key] != digest:
                    raise NeurOSDataError(f"{key} doesn't match the publisher's checksum", context)
                entry["verified_against"] = "publisher SHA256SUMS.txt"
            files[key] = entry
    body: dict[str, Any] = {
        "dataset": spec.key,
        "moabb_code": str(dataset.code),
        "raw_dir_layout": "paths are relative to <data_dir>/raw",
        "n_files": len(files),
        "total_bytes": sum(int(f["bytes"]) for f in files.values()),
        "provenance": manifest.get("provenance") or stamp(),
        "files": dict(sorted(files.items())),
    }
    if spec.licence_acceptance is not None:
        body["licence_accepted"] = spec.licence_acceptance
    if archives:
        body["archives"] = dict(sorted(archives.items()))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=1), encoding="utf-8", newline="\n")
    return body


def require_downloaded(spec: DatasetSpec, subjects: Sequence[int], data_dir: Path) -> None:
    """Raise unless every subject's files were fetched by ``neuros download`` (its manifest).

    MOABB downloads anything missing the moment data is *loaded*, without asking. Every command
    that loads data calls this first, so ``neuros download`` (each download approved by the
    founders) stays the only way data arrives.

    Raises:
        NeurOSDataError: naming the dataset and the subjects that aren't downloaded.
    """
    manifest = _load_manifest(manifest_path(data_dir, spec))
    if not manifest:
        raise NeurOSDataError(
            f"{spec.key} isn't downloaded (no manifest in {data_dir / 'manifests'}). Data "
            f"arrives only through `neuros download --dataset {spec.key}`, after approval.",
            DataContext(spec.key),
        )
    have = {int(entry["subject"]) for entry in manifest.get("files", {}).values()}
    missing = sorted(set(subjects) - have)
    if missing:
        listed = ", ".join(map(str, missing[:10])) + (" ..." if len(missing) > 10 else "")
        raise NeurOSDataError(
            f"{spec.key}: subjects {listed} aren't downloaded; run `neuros download --dataset "
            f"{spec.key} --subjects ...` (after approval) or leave them out",
            DataContext(spec.key),
        )


def loading_kwargs(spec: DatasetSpec, data_dir: Path) -> dict[str, Any]:
    """MOABB constructor arguments for *loading* data that ``neuros download`` already fetched.

    Licence-gated datasets (``spec.licence_acceptance``, e.g. Shin2017A under GPL-3.0): MOABB
    re-checks the acceptance on every load (its ``data_path``), not only when downloading. The
    acceptance is never configuration: a person gives it with ``neuros download --accept-licence``
    and the manifest records it. Loading honours exactly that record, so the licence text the
    registry declares today must be the one accepted at download.

    Raises:
        NeurOSDataError: if the dataset isn't downloaded (no manifest).
        NeurOSConfigError: if its licence must be accepted and the manifest doesn't record it.
    """
    if spec.licence_acceptance is None:
        return {}
    manifest = _load_manifest(manifest_path(data_dir, spec))
    if not manifest:
        raise NeurOSDataError(
            f"{spec.key} isn't downloaded (no manifest in {data_dir / 'manifests'})",
            DataContext(spec.key),
        )
    accepted = manifest.get("licence_accepted")
    if accepted != spec.licence_acceptance:
        raise NeurOSConfigError(
            f"{spec.key}: loading needs its licence, {spec.licence_acceptance}, accepted at "
            f"download; the manifest records {accepted!r}. Re-run `neuros download --dataset "
            f"{spec.key} --accept-licence` if you accept it."
        )
    return {"accept": True}


def loading_dataset(spec: DatasetSpec, data_dir: Path) -> Any:
    """The MOABB dataset object for loading downloaded data (see ``loading_kwargs``)."""
    return moabb_dataset(spec, **loading_kwargs(spec, data_dir))


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
    # Archives may be deleted once extracted (Yang2025's is 65.6 GB): absent is fine, changed
    # is not.
    for key, entry in manifest.get("archives", {}).items():
        file = raw_dir / key
        if file.is_file() and file_md5(file) != entry["md5"]:
            problems.append(f"{key}: md5 mismatch")
    return problems
