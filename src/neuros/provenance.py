"""Provenance stamps for results and manifests (architecture §3, principle 5).

Every result row records the git SHA, config hash, data hash and seed, so every number in the YC
application can be regenerated and traced. This module produces those stamps; it never raises
because of a missing git checkout (results computed from an installed package say so instead).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
import subprocess
from collections.abc import Iterable, Mapping
from importlib import metadata
from pathlib import Path
from typing import Any

from pydantic import BaseModel

# Packages whose versions can change numbers.
_TRACKED_PACKAGES = ("numpy", "scipy", "mne", "moabb", "pyriemann", "torch")
_HASH_CHUNK = 1 << 20  # 1 MiB


# Paths whose uncommitted changes can change a number. Edits elsewhere (the knowledge base,
# tests) leave ``git_dirty_code`` false, so a result is only marked irreproducible when it is.
CODE_PATHS = ("src", "configs", "pyproject.toml", "uv.lock")


def git_state(repo: Path | None = None) -> dict[str, str]:
    """The checkout's state: ``git_sha``, ``git_dirty`` (any uncommitted change) and
    ``git_dirty_code`` (uncommitted changes under ``CODE_PATHS``), as strings.

    Values are ``"unknown"`` outside a git checkout (e.g. an installed package).
    """
    cwd = str(repo) if repo else None

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=True, timeout=10
        ).stdout

    try:
        sha = git("rev-parse", "HEAD").strip()
        status = git("status", "--porcelain")
        code_status = git("status", "--porcelain", "--", *CODE_PATHS)
    except (OSError, subprocess.SubprocessError):
        return {"git_sha": "unknown", "git_dirty": "unknown", "git_dirty_code": "unknown"}
    return {
        "git_sha": sha,
        "git_dirty": "true" if status.strip() else "false",
        "git_dirty_code": "true" if code_status.strip() else "false",
    }


def config_hash(config: BaseModel | Mapping[str, Any]) -> str:
    """SHA-256 of a config's canonical JSON form (key order and whitespace don't matter)."""
    payload = config.model_dump(mode="json") if isinstance(config, BaseModel) else dict(config)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def package_versions(packages: Iterable[str] = _TRACKED_PACKAGES) -> dict[str, str]:
    """Installed versions of the packages that can change numbers (``absent`` if missing)."""
    versions = {"python": platform.python_version()}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "absent"
    return versions


def stamp(config: BaseModel | Mapping[str, Any] | None = None, **extra: str) -> dict[str, Any]:
    """A provenance block for a result file.

    Take it when a run **starts**, before any work: the git state then describes the code that
    actually ran, even if the checkout changes while a long run is in progress (worker
    processes import modules when they start). Add ``finished_utc`` when writing the result.
    """
    block: dict[str, Any] = {
        "created_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        **git_state(),
        "versions": package_versions(),
    }
    if config is not None:
        block["config_sha256"] = config_hash(config)
    block.update(extra)
    return block


def file_sha256(path: Path) -> str:
    """SHA-256 of a file, streamed in 1 MiB chunks (raw EEG files can be GBs)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()
