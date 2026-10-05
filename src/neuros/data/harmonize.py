"""Harmonisation: dataset-native recordings → canonical ``Recording`` objects (architecture §4).

What changes on the way in, and why:

1. **Channels** are renamed to canonical 10-10 names and split into EEG (kept) and declared
   non-EEG (EOG, EMG, stim: dropped). Anything else is an error: an undeclared electrode means
   the registry no longer describes the data.
2. **Units** go from volts (MNE's convention) to µV float32, the contract unit.
3. **Events** are mapped to canonical labels through ``labels.yaml`` and moved to the **cue
   onset** using the dataset's ``cue_offset_s``. Every event code must map; nothing is dropped
   silently.

Signal content is *not* changed here (no filtering, re-referencing or resampling): that's the
causal preprocessing shared with the online engine (architecture §3, principle 1).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from neuros.data.channels import normalize_channel_name
from neuros.data.recording import EventTable, Recording
from neuros.data.registry import DatasetSpec, moabb_dataset
from neuros.data.vocabulary import Vocabulary
from neuros.errors import DataContext, NeurOSChannelError, NeurOSDataError, NeurOSLabelError

VOLTS_TO_MICROVOLTS = 1e6

# Annotation descriptions MNE or vendors add that are not task events.
_IGNORED_ANNOTATION_PREFIXES = ("bad", "edge", "boundary")


def classify_channels(
    raw_names: Sequence[str],
    spec: DatasetSpec,
    context: DataContext,
    channel_types: Sequence[str] | None = None,
) -> tuple[list[int], list[str]]:
    """Return (indices, canonical names) of the EEG channels to keep, in file order.

    Channels whose MNE type is ``stim`` are the event channel and are always dropped (MOABB
    1.7 names it ``STI``; older versions ``stim``). Every other non-EEG channel must be declared
    in the registry, so montage surprises fail loudly. EEG channels the registry excludes
    (publisher-reported faults, ``excluded_channels``) must be present, like every declared
    channel, but are not kept.

    Raises:
        NeurOSChannelError: on an undeclared channel, a duplicate after normalisation, or a
            declared EEG channel missing from the file.
    """
    if channel_types is not None and len(channel_types) != len(raw_names):
        raise NeurOSChannelError(
            f"{len(raw_names)} channel names but {len(channel_types)} channel types", context
        )
    declared_eeg = set(spec.eeg_channels)
    excluded = set(spec.excluded_channels)
    declared_other = {name.strip().lower() for name in spec.non_eeg_channels}
    keep_idx: list[int] = []
    keep_names: list[str] = []
    found: list[str] = []  # every declared EEG channel seen, kept or excluded
    unknown: list[str] = []
    for index, raw_name in enumerate(raw_names):
        if channel_types is not None and channel_types[index] == "stim":
            continue
        canonical = normalize_channel_name(raw_name)
        if canonical is not None and canonical in declared_eeg:
            if canonical in found:
                raise NeurOSChannelError(
                    f"Channel {raw_name!r} duplicates {canonical!r} after normalisation", context
                )
            found.append(canonical)
            if canonical not in excluded:
                keep_idx.append(index)
                keep_names.append(canonical)
        elif raw_name.strip().lower() in declared_other:
            continue
        else:
            unknown.append(raw_name)
    if unknown:
        raise NeurOSChannelError(
            f"Undeclared channels {unknown}: add them to the registry's eeg_channels or "
            "non_eeg_channels for this dataset",
            context,
        )
    missing = [name for name in spec.eeg_channels if name not in found]
    if missing:
        raise NeurOSChannelError(f"Declared EEG channels missing from the file: {missing}", context)
    return keep_idx, keep_names


def extract_events(
    raw: Any,
    event_id: Mapping[str, int],
    spec: DatasetSpec,
    vocabulary: Vocabulary,
    context: DataContext,
) -> EventTable:
    """Read task events from a raw recording and map them to canonical labels at the cue.

    Uses the stim channel if the recording has one (MOABB's convention), otherwise the
    annotations. Onsets are converted to indices into ``raw.get_data()`` (MNE event samples
    include ``raw.first_samp``) and shifted by ``spec.cue_offset_s``.

    Raises:
        NeurOSLabelError: on an event code or annotation the dataset's event map doesn't cover.
        NeurOSDataError: if a shifted onset falls outside the recording.
    """
    import mne  # lazy: heavy import

    code_to_name = {int(code): name for name, code in event_id.items()}
    if len(code_to_name) != len(event_id):
        raise NeurOSDataError(f"event_id has duplicate codes: {dict(event_id)}", context)

    stim_picks = mne.pick_types(raw.info, meg=False, eeg=False, stim=True)
    if len(stim_picks):
        stim_name = raw.ch_names[int(stim_picks[0])]
        events = mne.find_events(
            raw, stim_channel=stim_name, shortest_event=0, initial_event=True, verbose="error"
        )
    else:
        descriptions = [str(d) for d in raw.annotations.description]
        relevant = [
            d for d in descriptions if not d.lower().startswith(_IGNORED_ANNOTATION_PREFIXES)
        ]
        unknown_desc = sorted(set(relevant) - set(event_id))
        if unknown_desc:
            raise NeurOSLabelError(
                f"Annotations {unknown_desc} are not in the dataset's event_id {sorted(event_id)}",
                context,
            )
        if relevant:
            events, _ = mne.events_from_annotations(raw, event_id=dict(event_id), verbose="error")
        else:
            events = np.zeros((0, 3), dtype=np.int64)

    codes = events[:, 2].astype(int) if len(events) else np.zeros(0, dtype=int)
    unmapped = Counter(int(c) for c in codes if int(c) not in code_to_name)
    if unmapped:
        raise NeurOSLabelError(
            f"Event codes {dict(unmapped)} (code: count) are not in event_id {dict(event_id)}",
            context,
        )

    sfreq = float(raw.info["sfreq"])
    shift = round(spec.cue_offset_s * sfreq)
    n_samples = int(raw.n_times)
    pairs: list[tuple[int, str]] = []
    for sample, code in zip(events[:, 0] if len(events) else [], codes, strict=True):
        onset = int(sample) - int(raw.first_samp) + shift
        if not 0 <= onset < n_samples:
            raise NeurOSDataError(
                f"Event {code_to_name[int(code)]!r} at sample {int(sample)} lies outside the "
                f"recording after the {spec.cue_offset_s} s cue offset ({n_samples} samples)",
                context,
            )
        label = vocabulary.map_event(spec.key, code_to_name[int(code)], context)
        pairs.append((onset, label))
    return EventTable.from_pairs(pairs)


def harmonize_raw(
    raw: Any,
    spec: DatasetSpec,
    vocabulary: Vocabulary,
    *,
    subject: int,
    session: str,
    run: str,
    event_id: Mapping[str, int],
    provenance: Mapping[str, str] | None = None,
) -> Recording:
    """Convert one MNE ``Raw`` (volts, dataset naming) into a canonical ``Recording``.

    Raises:
        NeurOSChannelError, NeurOSLabelError, NeurOSDataError: see the helpers above.
    """
    context = DataContext(spec.key, subject, session, run)
    picks, names = classify_channels(
        list(raw.ch_names), spec, context, channel_types=list(raw.get_channel_types())
    )
    events = extract_events(raw, event_id, spec, vocabulary, context)
    volts = raw.get_data(picks=picks)
    data = (np.asarray(volts, dtype=np.float64) * VOLTS_TO_MICROVOLTS).astype(np.float32)
    return Recording(
        data=data,
        sfreq=float(raw.info["sfreq"]),
        ch_names=tuple(names),
        events=events,
        dataset=spec.key,
        subject=subject,
        session=session,
        run=run,
        provenance=dict(provenance or {}),
    )


def harmonize_moabb_subject(
    spec: DatasetSpec,
    subject: int,
    vocabulary: Vocabulary,
    *,
    dataset: Any | None = None,
) -> list[Recording]:
    """Load one subject through MOABB (downloading if needed) and harmonise every run.

    Raises:
        NeurOSDataError: if MOABB returns no data for the subject.
    """
    if subject not in spec.subjects:
        raise NeurOSDataError(
            f"Subject {subject} is not in the registry", DataContext(spec.key, subject)
        )
    import mne
    import moabb

    ds = dataset if dataset is not None else moabb_dataset(spec)
    by_subject = ds.get_data(subjects=[subject])
    sessions = by_subject.get(subject)
    if not sessions:
        raise NeurOSDataError("MOABB returned no sessions", DataContext(spec.key, subject))
    recordings: list[Recording] = []
    for session in sorted(sessions, key=str):
        runs = sessions[session]
        for run in sorted(runs, key=str):
            raw = runs[run]
            provenance = {
                "loader": "moabb",
                "moabb_version": str(moabb.__version__),
                "mne_version": str(mne.__version__),
                "moabb_code": str(ds.code),
                "source_files": ";".join(str(f) for f in (raw.filenames or []) if f),
            }
            recordings.append(
                harmonize_raw(
                    raw,
                    spec,
                    vocabulary,
                    subject=subject,
                    session=str(session),
                    run=str(run),
                    event_id=ds.event_id,
                    provenance=provenance,
                )
            )
    return recordings
