"""PhysioNet datasets: fast, publisher-verified downloads (PhysionetMI and EEGMAT).

Day-1 finding: physionet.org served EEGMMIDB at ~100 KB/s (12 s per 1.2 MB file, ~5 h for
PhysionetMI's 872 files). PhysioNet publishes the same files on its AWS Open Data mirror (bucket
``physionet-open``), which served one in 0.4 s.

So we download from the mirror and check every file against ``SHA256SUMS.txt`` fetched from
**physionet.org itself**: the mirror never has to be trusted, and unlike plain MOABB downloads
(TLS checks off, no known hashes) every byte is verified against the publisher's checksums.

Files of MOABB datasets are written exactly where MOABB stores them (computed with MOABB's own
path functions), so MOABB later finds them locally and loads them without touching the network.
Datasets MOABB doesn't serve (EEGMAT) go to ``<raw>/physionet-<project>-<version>/`` and are read
by their own loader (``neuros.data.eegmat``).
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neuros import __version__
from neuros.errors import DataContext, NeurOSDataError

log = logging.getLogger(__name__)

PHYSIONET_FILES = "https://physionet.org/files"
PHYSIONET_MIRROR = "https://physionet-open.s3.amazonaws.com"
_USER_AGENT = f"neuros/{__version__} (research data download)"
_CHUNK = 1 << 20
_RETRIES = 3
_TIMEOUT_S = 120


@dataclass(frozen=True)
class PhysioNetProject:
    """A PhysioNet project and the files one subject needs."""

    dataset: str  # our registry key
    project: str  # PhysioNet slug, e.g. "eegmmidb"
    version: str
    # MOABB signifier: files live under <raw>/MNE-<sign>-data/. None for datasets MOABB doesn't
    # serve: their files live under <raw>/physionet-<project>-<version>/ (``local_destination``).
    moabb_sign: str | None
    files_for_subject: Callable[[int], list[str]]

    def upstream_url(self, relpath: str) -> str:
        return f"{PHYSIONET_FILES}/{self.project}/{self.version}/{relpath}"

    def mirror_url(self, relpath: str) -> str:
        return f"{PHYSIONET_MIRROR}/{self.project}/{self.version}/{relpath}"


def _physionet_mi_files(subject: int) -> list[str]:
    """The runs MOABB's ``PhysionetMI.data_path`` fetches (imagined=True): baseline 1-2,
    left/right fist 4/8/12, both fists/feet 6/10/14 (MOABB 1.7.2, physionet_mi.py)."""
    runs = (1, 2, 4, 8, 12, 6, 10, 14)
    return [f"S{subject:03d}/S{subject:03d}R{run:02d}.edf" for run in runs]


def _eegmat_files(subject: int) -> list[str]:
    """EEGMAT's two recordings per person: ``_1`` background (rest), ``_2`` mental arithmetic.
    Subjects are numbered 00-35 on PhysioNet (registry: 0-35)."""
    return [f"Subject{subject:02d}_1.edf", f"Subject{subject:02d}_2.edf"]


PHYSIONET_PROJECTS: dict[str, PhysioNetProject] = {
    "PhysionetMI": PhysioNetProject(
        "PhysionetMI", "eegmmidb", "1.0.0", "EEGBCI", _physionet_mi_files
    ),
    "EEGMAT": PhysioNetProject("EEGMAT", "eegmat", "1.0.0", None, _eegmat_files),
}


def _open(url: str) -> Any:
    """Open an HTTPS URL. The default SSL context verifies certificates (MOABB's doesn't).

    Returns the urlopen response (typeshed types it as ``Any``): a context manager with read().
    """
    if not url.startswith("https://"):
        raise NeurOSDataError(f"Refusing a non-HTTPS download: {url}")
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    return urllib.request.urlopen(request, timeout=_TIMEOUT_S)


def fetch_sha256sums(project: PhysioNetProject) -> dict[str, str]:
    """Download and parse the publisher's ``SHA256SUMS.txt`` (relative path → hex digest).

    Raises:
        NeurOSDataError: if the file can't be fetched or a line is malformed.
    """
    url = project.upstream_url("SHA256SUMS.txt")
    try:
        with _open(url) as response:
            text = response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise NeurOSDataError(
            f"Could not fetch {url}: {exc}", DataContext(project.dataset)
        ) from exc
    return parse_sha256sums(text, source=url)


def parse_sha256sums(text: str, source: str = "SHA256SUMS.txt") -> dict[str, str]:
    """Parse ``<sha256> <relative path>`` lines (``*`` binary markers allowed).

    Raises:
        NeurOSDataError: on a malformed line or a duplicate path.
    """
    sums: dict[str, str] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise NeurOSDataError(f"{source}:{number}: malformed checksum line {line!r}")
        digest, path = parts[0].lower(), parts[1].strip().lstrip("*")
        if any(c not in "0123456789abcdef" for c in digest):
            raise NeurOSDataError(f"{source}:{number}: not a hex digest {parts[0]!r}")
        if path in sums:
            raise NeurOSDataError(f"{source}:{number}: duplicate entry for {path}")
        sums[path] = digest
    if not sums:
        raise NeurOSDataError(f"{source}: no checksums found")
    return sums


def moabb_destination(raw_dir: Path, sign: str, url: str) -> Path:
    """Where MOABB's ``data_dl`` stores ``url``: computed with MOABB's own helpers.

    Uses private MOABB functions on purpose: re-implementing the layout would risk a silent
    mismatch (MOABB would then re-download from the slow host). Pinned by a unit test.
    """
    from moabb.datasets.download import _normalize_destination, _sanitize_path

    root = raw_dir / f"MNE-{sign.lower()}-data"
    return Path(_sanitize_path(_normalize_destination(url, root)))


def local_destination(raw_dir: Path, project: PhysioNetProject, relpath: str) -> Path:
    """Where a project's file is stored: MOABB's layout for MOABB datasets, else our own."""
    if project.moabb_sign is not None:
        return moabb_destination(raw_dir, project.moabb_sign, project.upstream_url(relpath))
    if relpath.startswith("/") or ".." in Path(relpath).parts:
        raise NeurOSDataError(f"refusing an unsafe relative path {relpath!r}")
    return raw_dir / f"physionet-{project.project}-{project.version}" / relpath


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_verified(url: str, destination: Path, expected_sha256: str) -> int:
    """Download ``url`` to ``destination`` unless an identical file is already there.

    Streams to a ``.part`` file while hashing, compares with ``expected_sha256``, then renames
    atomically, so a crash never leaves a truncated file where MOABB would trust it.

    Returns:
        Bytes downloaded (0 if the file was already present and correct).

    Raises:
        NeurOSDataError: on a checksum mismatch (never retried) or after ``_RETRIES`` failed
            attempts.
    """
    if destination.is_file() and _sha256_file(destination) == expected_sha256:
        return 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    last_error: Exception | None = None
    for attempt in range(1, _RETRIES + 1):
        try:
            digest = hashlib.sha256()
            size = 0
            with _open(url) as response, partial.open("wb") as handle:
                for chunk in iter(lambda: response.read(_CHUNK), b""):
                    digest.update(chunk)
                    handle.write(chunk)
                    size += len(chunk)
            if digest.hexdigest() != expected_sha256:
                partial.unlink(missing_ok=True)
                raise NeurOSDataError(
                    f"Checksum mismatch for {url}: got {digest.hexdigest()[:12]}…, "
                    f"publisher says {expected_sha256[:12]}…"
                )
            os.replace(partial, destination)
            return size
        except NeurOSDataError:
            raise
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            partial.unlink(missing_ok=True)
            log.warning("attempt %d/%d failed for %s: %s", attempt, _RETRIES, url, exc)
            time.sleep(2**attempt)
    raise NeurOSDataError(f"Giving up on {url} after {_RETRIES} attempts: {last_error}")


def fetch_subjects(
    project: PhysioNetProject,
    subjects: Sequence[int],
    raw_dir: Path,
    *,
    workers: int = 8,
    sums: dict[str, str] | None = None,
) -> dict[str, str]:
    """Fetch every file the subjects need from the mirror, verified against the publisher.

    Returns:
        Map from destination path (relative to ``raw_dir``, POSIX) to its verified SHA-256.

    Raises:
        NeurOSDataError: if a needed file has no published checksum, or any download fails
            (all failures are collected and reported together).
    """
    checksums = sums if sums is not None else fetch_sha256sums(project)
    jobs: list[tuple[str, Path, str]] = []
    for subject in subjects:
        for relpath in project.files_for_subject(subject):
            if relpath not in checksums:
                raise NeurOSDataError(
                    f"{relpath} has no published checksum", DataContext(project.dataset, subject)
                )
            destination = local_destination(raw_dir, project, relpath)
            jobs.append((project.mirror_url(relpath), destination, checksums[relpath]))

    verified: dict[str, str] = {}
    failures: list[str] = []
    downloaded = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(fetch_verified, url, dest, sha): (dest, sha) for url, dest, sha in jobs
        }
        for future in as_completed(futures):
            dest, sha = futures[future]
            try:
                downloaded += future.result()
                verified[dest.resolve().relative_to(raw_dir.resolve()).as_posix()] = sha
            except NeurOSDataError as exc:
                failures.append(str(exc))
    log.info(
        "%s: %d files verified (%.1f MB downloaded)",
        project.dataset,
        len(verified),
        downloaded / 1e6,
    )
    if failures:
        raise NeurOSDataError(
            f"{len(failures)} of {len(jobs)} downloads failed; first: {failures[0]}",
            DataContext(project.dataset),
        )
    return verified
