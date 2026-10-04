"""Repository hygiene: every text file is clean UTF-8.

Day-2 finding (2026-10-03): ``validate.py`` had been double-encoded on Day 1 (UTF-8 read as
cp1252 and written back), so every "§" and "µ" gained a stray "A-circumflex" in front, including
in error messages users see. Windows tools (PowerShell ``Set-Content``, cp1252 consoles) make
this easy to repeat. This test fails on a byte-order mark or the telltale double-encoding.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from .conftest import REPO_ROOT

_TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".toml", ".txt", ".cfg", ".ini", ".json"}
_SKIP_DIRS = {".git", ".venv", "data", "cache", "results", "bundles", "__pycache__", ".mypy_cache"}
# UTF-8 bytes of a non-ASCII character, misread as cp1252 and re-encoded: a lead character from
# "Â" (U+00C2), "Ã" (U+00C3) or "â" (U+00E2) followed by what was a continuation byte. Written
# with escapes so this file doesn't trip its own check.
_MOJIBAKE = re.compile("[ÂÃ][\u0080-¿]|â[\u0080-¿‘-™€]")


def _text_files() -> list[Path]:
    files = []
    for path in REPO_ROOT.rglob("*"):
        if any(part in _SKIP_DIRS for part in path.relative_to(REPO_ROOT).parts):
            continue
        if path.is_file() and path.suffix in _TEXT_SUFFIXES:
            files.append(path)
    return sorted(files)


def test_there_are_files_to_check() -> None:
    assert len(_text_files()) > 50


@pytest.mark.parametrize("path", _text_files(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_file_is_clean_utf8(path: Path) -> None:
    raw = path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf"), "byte-order mark"
    text = raw.decode("utf-8")  # raises on invalid UTF-8
    match = _MOJIBAKE.search(text)
    if match is not None:
        context = text[max(0, match.start() - 20) : match.end() + 20]
        pytest.fail(f"double-encoded text near {context!r}")
