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


def git_state(repo: Path | None = None) -> dict[str, str]:
    """Return ``{"git_sha": ..., "git_dirty": "true"/"false"}`` or ``"unknown"`` values."""
    cwd = str(repo) if repo else None
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {"git_sha": "unknown", "git_dirty": "unknown"}
    return {"git_sha": sha, "git_dirty": "true" if status.strip() else "false"}


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
    """A provenance block for a result file."""
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
