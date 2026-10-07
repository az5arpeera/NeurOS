"""Disk cache of preprocessed recordings (architecture §4.5).

E1 needs ~380 people, and some datasets arrive at 1000 Hz: harmonising, validating and filtering
them takes far longer than training on them. So each subject's *preprocessed* recordings (the
contract's 128 Hz, µV float32, events at the cue) are saved once and reused. The cache sits
between preprocessing and epoch extraction, so one entry serves every window an experiment
cuts: the task epoch, the cue-only control, the audit's probes.

**What decides an entry (the key).** A SHA-256 over everything that can change a cached array:

- the dataset's declarations that shape the data (channels, exclusions, rates, cue offset,
  MOABB arguments) and its event map;
- the preprocessing half of the signal contract (rate, band, filter order, notches) and the
  validation thresholds (they decide which recordings may pass at all);
- the *source code* of the modules that load, check and filter recordings, so editing any of
  them invalidates the cache without anyone remembering to;
- the versions of MOABB, MNE, NumPy and SciPy.

Window settings (epoch, window length, cue margin) are deliberately *not* in the key.

**Integrity.** Every file's SHA-256 is recorded and checked on load, with shapes, dtypes and the
key. An entry that fails is moved aside (``*.corrupt-<time>``), never deleted, and rebuilt;
nothing corrupt is ever returned. Entries are written to a temporary folder and renamed into
place, so a crash leaves no half-written entry, and two processes building the same subject
can't mix their files.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import importlib.util
import json
import logging
import os
import shutil
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from neuros.config.schemas import DataConfig
from neuros.data.recording import EventTable, Recording
from neuros.data.registry import DatasetSpec
from neuros.data.vocabulary import Vocabulary
from neuros.errors import DataContext, NeurOSDataError
from neuros.provenance import file_sha256

log = logging.getLogger(__name__)

# Bump when the on-disk layout changes.
CACHE_FORMAT = 1

# Modules whose code decides what a cached recording contains (hashed into the key).
KEYED_MODULES: tuple[str, ...] = (
    "neuros.data.channels",
    "neuros.data.harmonize",
    "neuros.data.recording",
    "neuros.data.validate",
    "neuros.data.prepare",
    "neuros.signal._checks",
    "neuros.signal.filters",
    "neuros.signal.resample",
    "neuros.signal.preprocess",
)

# Packages whose version can change loaded or filtered samples.
KEYED_PACKAGES: tuple[str, ...] = ("moabb", "mne", "numpy", "scipy")

# Registry fields that describe a dataset without changing its samples or events.
_DESCRIPTIVE_FIELDS = frozenset(
    {
        "title",
        "role",
        "licence",
        "url",
        "channels_source",
        "unverified",
        "notes",
        "subgroups",
        "excluded_subjects",
        "train_restriction",
        "licence_acceptance",
        "class_on_screen_s",
        "task_s",
        "task_from_s",
        "pre_cue_fixation_s",
        "rest_source",
        "subjects",
        # Applied when windows are placed in a model's layout, after the cache (D-037).
        "derived_channels",
    }
)

Preparer = Callable[..., list[Recording]]


def _module_source_digest(names: Sequence[str]) -> str:
    """SHA-256 over the source files of ``names`` (in order).

    Raises:
        NeurOSDataError: if a module can't be found (the key would silently ignore it).
    """
    digest = hashlib.sha256()
    for name in names:
        found = importlib.util.find_spec(name)
        if found is None or found.origin is None or not Path(found.origin).is_file():
            raise NeurOSDataError(f"cache key: can't find the source of {name}")
        digest.update(name.encode())
        digest.update(Path(found.origin).read_bytes())
    return digest.hexdigest()


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "absent"


def key_payload(spec: DatasetSpec, vocabulary: Vocabulary, config: DataConfig) -> dict[str, Any]:
    """Everything that can change a cached recording of ``spec`` (see the module docstring)."""
    declared = {
        name: value
        for name, value in dataclasses.asdict(spec).items()
        if name not in _DESCRIPTIVE_FIELDS
    }
    signal = config.signal
    return {
        "format": CACHE_FORMAT,
        "dataset": declared,
        "event_map": dict(sorted(vocabulary.event_maps.get(spec.key, {}).items())),
        "preprocessing": {
            "sfreq": signal.sfreq,
            "band_hz": list(signal.band_hz),
            "filter_order": signal.filter_order,
            "notch_hz": list(signal.notch_hz),
            "notch_q": signal.notch_q,
        },
        "validation": config.validation.model_dump(mode="json"),
        "code_sha256": _module_source_digest(KEYED_MODULES),
        "versions": {name: _package_version(name) for name in KEYED_PACKAGES},
    }


def cache_key(payload: dict[str, Any]) -> str:
    """SHA-256 of the key payload's canonical JSON."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


@dataclass(frozen=True)
class RecordingCache:
    """Preprocessed recordings under ``root/recordings/<dataset>/<key[:16]>/sub-<subject>/``."""

    root: Path

    def entry_dir(self, dataset: str, subject: int, key: str) -> Path:
        if len(key) != 64:
            raise NeurOSDataError(f"cache keys are 64-character SHA-256 digests, got {key!r}")
        return self.root / "recordings" / dataset / key[:16] / f"sub-{subject}"

    # ------------------------------------------------------------------ reading
    def load(self, spec: DatasetSpec, subject: int, key: str) -> list[Recording] | None:
        """The cached recordings, or None if absent or unusable (moved aside, logged)."""
        entry = self.entry_dir(spec.key, subject, key)
        index_file = entry / "index.json"
        if not index_file.is_file():
            return None
        try:
            return self._read(entry, index_file, spec, subject, key)
        except (NeurOSDataError, OSError, ValueError, KeyError, TypeError) as exc:
            aside = entry.with_name(f"{entry.name}.corrupt-{_now()}-{uuid.uuid4().hex[:6]}")
            log.warning("%s: unusable cache entry (%s); moved to %s", entry, exc, aside)
            os.replace(entry, aside)
            return None

    def _read(
        self, entry: Path, index_file: Path, spec: DatasetSpec, subject: int, key: str
    ) -> list[Recording]:
        index = json.loads(index_file.read_text(encoding="utf-8"))
        if index["key"] != key or index["dataset"] != spec.key or index["subject"] != subject:
            raise NeurOSDataError("index doesn't match the requested entry")
        recordings: list[Recording] = []
        for item in index["recordings"]:
            file = entry / str(item["file"])
            if file_sha256(file) != item["sha256"]:
                raise NeurOSDataError(f"{file.name}: sha256 mismatch")
            with np.load(file, allow_pickle=False) as arrays:
                data = np.asarray(arrays["data"])
                onsets = np.asarray(arrays["onsets"])
                labels = tuple(str(label) for label in arrays["labels"])
            expected = (len(item["ch_names"]), int(item["n_samples"]))
            if data.dtype != np.float32 or data.shape != expected:
                raise NeurOSDataError(
                    f"{file.name}: {data.dtype} {data.shape}, expected float32 {expected}"
                )
            recordings.append(
                Recording(
                    data=data,
                    sfreq=float(item["sfreq"]),
                    ch_names=tuple(item["ch_names"]),
                    events=EventTable(onsets=onsets.astype(np.int64), labels=labels),
                    dataset=spec.key,
                    subject=subject,
                    session=str(item["session"]),
                    run=str(item["run"]),
                    provenance={str(k): str(v) for k, v in item["provenance"].items()},
                )
            )
        if not recordings:
            raise NeurOSDataError("entry holds no recordings")
        return recordings

    # ------------------------------------------------------------------ writing
    def store(
        self,
        spec: DatasetSpec,
        subject: int,
        key: str,
        payload: dict[str, Any],
        recordings: Sequence[Recording],
    ) -> Path:
        """Write one subject's recordings atomically; return the entry folder.

        If another process finished the same entry first, its copy is kept and ours dropped.

        Raises:
            NeurOSDataError: if a recording belongs to another dataset or subject, or there are
                none.
        """
        if not recordings:
            raise NeurOSDataError("nothing to cache", DataContext(spec.key, subject))
        for recording in recordings:
            if recording.dataset != spec.key or recording.subject != subject:
                raise NeurOSDataError(
                    f"recording of {recording.dataset}:{recording.subject} can't be cached "
                    f"as {spec.key}:{subject}"
                )
        entry = self.entry_dir(spec.key, subject, key)
        entry.parent.mkdir(parents=True, exist_ok=True)
        key_file = entry.parent / "key.json"
        if not key_file.is_file():  # describes every entry under this key, for humans
            _write_json_atomic(key_file, {"key": key, **payload})
        staging = entry.with_name(f"{entry.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
        staging.mkdir()
        try:
            items = []
            for number, recording in enumerate(recordings):
                file = staging / f"rec-{number}.npz"
                with file.open("wb") as handle:
                    np.savez(
                        handle,
                        data=recording.data,
                        onsets=recording.events.onsets,
                        labels=np.array(recording.events.labels, dtype=str),
                    )
                items.append(
                    {
                        "file": file.name,
                        "sha256": file_sha256(file),
                        "session": recording.session,
                        "run": recording.run,
                        "sfreq": recording.sfreq,
                        "ch_names": list(recording.ch_names),
                        "n_samples": recording.n_samples,
                        "n_events": len(recording.events),
                        "provenance": dict(recording.provenance),
                    }
                )
            index = {
                "format": CACHE_FORMAT,
                "dataset": spec.key,
                "subject": subject,
                "key": key,
                "created_utc": _now(),
                "recordings": items,
            }
            _write_json_atomic(staging / "index.json", index)
            try:
                os.replace(staging, entry)
            except OSError:
                if not (entry / "index.json").is_file():
                    raise
                log.info("%s: another process cached it first; keeping theirs", entry)
        finally:
            if staging.exists():
                # Our own, never-published staging folder (the entry itself is never removed).
                shutil.rmtree(staging, ignore_errors=True)
        return entry

    # ------------------------------------------------------------------ both
    def load_or_prepare(
        self,
        spec: DatasetSpec,
        subject: int,
        vocabulary: Vocabulary,
        config: DataConfig,
        *,
        prepare: Preparer,
        dataset: Any | None = None,
    ) -> tuple[list[Recording], bool]:
        """The subject's preprocessed recordings from the cache, or prepared and cached now.

        Returns:
            ``(recordings, hit)``: ``hit`` is True when they came from the cache.
        """
        payload = key_payload(spec, vocabulary, config)
        key = cache_key(payload)
        cached = self.load(spec, subject, key)
        if cached is not None:
            return cached, True
        recordings = prepare(spec, subject, vocabulary, config, dataset=dataset)
        self.store(spec, subject, key, payload, recordings)
        return recordings, False


def _write_json_atomic(path: Path, body: dict[str, Any]) -> None:
    staging = path.with_name(f"{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    staging.write_text(json.dumps(body, indent=1, default=str), encoding="utf-8", newline="\n")
    os.replace(staging, path)
