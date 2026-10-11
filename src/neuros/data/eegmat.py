"""EEGMAT's own loader: PhysioNet's EDF files → canonical ``Recording`` objects.

EEGMAT (rest vs mental arithmetic, 36 people) isn't served by MOABB, so ``neuros download``
fetches it straight from PhysioNet (``neuros.data.physionet``, every file checked against the
publisher's SHA256SUMS) into ``<raw>/physionet-eegmat-1.0.0/``. This module reads those files and
hands them to the same harmonisation as every other dataset (``harmonize_raw``): canonical
channel names, µV float32, declared non-EEG channels dropped, undeclared ones refused.

Each person has two recordings, one per state:

| file | run | label (labels.yaml) |
|---|---|---|
| ``SubjectNN_1.edf`` | ``1_background`` | ``state_rest`` |
| ``SubjectNN_2.edf`` | ``2_task`` | ``state_arithmetic`` |

A whole recording is one state, so each gets a single event at its first sample. There's no
class cue on any screen (the arithmetic problem was given orally).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from neuros.data.download import require_downloaded
from neuros.data.harmonize import harmonize_raw
from neuros.data.physionet import PHYSIONET_PROJECTS, local_destination
from neuros.data.recording import EventTable, Recording
from neuros.data.registry import DatasetSpec
from neuros.data.vocabulary import Vocabulary
from neuros.errors import DataContext, NeurOSConfigError, NeurOSDataError

KEY = "EEGMAT"
# (file index, run name, the dataset's event name in labels.yaml)
RUNS: tuple[tuple[int, str, str], ...] = ((0, "1_background", "background"), (1, "2_task", "task"))


def eegmat_files(subject: int, data_dir: Path) -> tuple[Path, Path]:
    """The person's (rest, task) EDF paths under ``<data_dir>/raw``.

    Raises:
        NeurOSDataError: if either file is missing.
    """
    project = PHYSIONET_PROJECTS[KEY]
    raw_dir = (data_dir / "raw").resolve()
    rest, task = (
        local_destination(raw_dir, project, rel) for rel in project.files_for_subject(subject)
    )
    for path in (rest, task):
        if not path.is_file():
            raise NeurOSDataError(f"missing {path}", DataContext(KEY, subject))
    return rest, task


def harmonize_eegmat_subject(
    spec: DatasetSpec, subject: int, vocabulary: Vocabulary, data_dir: Path
) -> list[Recording]:
    """Load and harmonise one person's two EEGMAT recordings (rest first, then the task).

    Raises:
        NeurOSConfigError: if ``spec`` isn't EEGMAT.
        NeurOSDataError: if the person isn't in the registry or wasn't downloaded, a file is
            missing, or its sampling rate differs from the registry's.
        NeurOSChannelError, NeurOSLabelError: from harmonisation (undeclared channels, unmapped
            labels).
    """
    if spec.key != KEY:
        raise NeurOSConfigError(f"the EEGMAT loader can't load {spec.key}")
    context = DataContext(KEY, subject)
    if subject not in spec.subjects:
        raise NeurOSDataError("subject is not in the registry", context)
    require_downloaded(spec, [subject], data_dir)
    import mne  # lazy: heavy import

    recordings: list[Recording] = []
    for path, (_, run, event) in zip(eegmat_files(subject, data_dir), RUNS, strict=True):
        raw = mne.io.read_raw_edf(path, preload=True, verbose="error")
        sfreq = float(raw.info["sfreq"])
        if sfreq != spec.native_sfreq:
            raise NeurOSDataError(
                f"{path.name}: {sfreq:g} Hz, the registry says {spec.native_sfreq:g}",
                DataContext(KEY, subject, "0", run),
            )
        recording = harmonize_raw(
            raw,
            spec,
            vocabulary,
            subject=subject,
            session="0",
            run=run,
            event_id={},  # no annotations in these files; the label is the whole file's
            provenance={
                "loader": "neuros.data.eegmat",
                "mne_version": str(mne.__version__),
                "source_files": str(path),
            },
        )
        label = vocabulary.map_event(KEY, event, recording.context)
        recordings.append(
            dataclasses.replace(recording, events=EventTable.from_pairs([(0, label)]))
        )
    return recordings
