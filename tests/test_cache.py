"""The preprocessed-recording cache (architecture §4.5): round trip, keys, integrity, races."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neuros.config.experiment import apply_signal_overrides
from neuros.config.schemas import DataConfig
from neuros.data import prepare as prepare_module
from neuros.data.cache import RecordingCache, cache_key, key_payload
from neuros.data.recording import Recording
from neuros.data.registry import DatasetSpec
from neuros.data.synthetic import SyntheticConfig, make_recording
from neuros.data.vocabulary import Vocabulary
from neuros.errors import NeurOSDataError
from neuros.signal.preprocess import preprocess_recording

from .conftest import make_spec


def _recordings(config: DataConfig, subject: int = 1) -> list[Recording]:
    """Two preprocessed synthetic runs of one subject, as prepare_subject would return them."""
    return [
        preprocess_recording(
            make_recording(SyntheticConfig(seed=subject * 10 + run), subject=subject, run=str(run)),
            config.signal,
        )
        for run in range(2)
    ]


def _key(spec: DatasetSpec, vocab: Vocabulary, config: DataConfig) -> str:
    return cache_key(key_payload(spec, vocab, config))


def test_round_trip_is_exact(tmp_path: Path, vocab: Vocabulary) -> None:
    spec, config = make_spec(), DataConfig()
    cache = RecordingCache(tmp_path)
    original = _recordings(config)
    key = _key(spec, vocab, config)
    entry = cache.store(spec, 1, key, key_payload(spec, vocab, config), original)
    assert (entry / "index.json").is_file()
    assert (entry.parent / "key.json").is_file()
    loaded = cache.load(spec, 1, key)
    assert loaded is not None
    assert len(loaded) == len(original)
    for a, b in zip(original, loaded, strict=True):
        assert np.array_equal(a.data, b.data)  # bit-identical float32
        assert a.events == b.events
        assert (a.ch_names, a.sfreq, a.session, a.run) == (b.ch_names, b.sfreq, b.session, b.run)
        assert dict(a.provenance) == dict(b.provenance)
    assert cache.load(spec, 2, key) is None  # another subject: a miss


def test_key_covers_preprocessing_not_windows(vocab: Vocabulary) -> None:
    spec, config = make_spec(), DataConfig()
    base = _key(spec, vocab, config)

    def with_signal(**overrides: object) -> DataConfig:
        return config.model_copy(
            update={"signal": apply_signal_overrides(config.signal, dict(overrides))}
        )

    # Window placement doesn't change a cached recording: one entry serves every window.
    assert _key(spec, vocab, with_signal(epoch_s=[0.0, 0.5], window_s=0.5)) == base
    assert _key(spec, vocab, with_signal(cue_margin_s=0.5)) == base
    # Filtering, declarations and event maps do.
    assert _key(spec, vocab, with_signal(band_hz=[8.0, 30.0])) != base
    assert _key(make_spec(cue_offset_s=0.5), vocab, config) != base
    assert _key(make_spec(excluded_channels={"Pz": "faulty"}), vocab, config) != base
    assert _key(make_spec(unit_scale=1 / 32), vocab, config) != base  # D-035: rescales samples
    other_maps = {**vocab.event_maps, "Synthetic": {"left_hand": "left_hand"}}
    assert _key(spec, vocab.model_copy(update={"event_maps": other_maps}), config) != base
    # Descriptive fields don't.
    assert _key(make_spec(notes="anything", licence="other"), vocab, config) == base


def test_corrupt_entries_are_moved_aside_never_returned(tmp_path: Path, vocab: Vocabulary) -> None:
    spec, config = make_spec(), DataConfig()
    cache = RecordingCache(tmp_path)
    key = _key(spec, vocab, config)
    entry = cache.store(spec, 1, key, key_payload(spec, vocab, config), _recordings(config))
    target = entry / "rec-0.npz"
    raw = bytearray(target.read_bytes())
    raw[-10] ^= 0xFF  # flip bits near the end of the file
    target.write_bytes(bytes(raw))
    assert cache.load(spec, 1, key) is None
    assert not entry.exists()
    aside = list(entry.parent.glob(f"{entry.name}.corrupt-*"))
    assert len(aside) == 1
    assert (aside[0] / "rec-0.npz").is_file()  # kept, not deleted


def test_load_or_prepare_prepares_once(tmp_path: Path, vocab: Vocabulary) -> None:
    spec, config = make_spec(), DataConfig()
    cache = RecordingCache(tmp_path)
    calls: list[int] = []

    def prepare(spec_arg: DatasetSpec, subject: int, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        calls.append(subject)
        return _recordings(config, subject)

    first, hit1 = cache.load_or_prepare(spec, 3, vocab, config, prepare=prepare)
    second, hit2 = cache.load_or_prepare(spec, 3, vocab, config, prepare=prepare)
    assert (hit1, hit2, calls) == (False, True, [3])
    assert all(np.array_equal(a.data, b.data) for a, b in zip(first, second, strict=True))
    # A different band is a different entry: prepared again.
    band = config.model_copy(
        update={"signal": apply_signal_overrides(config.signal, {"band_hz": [8.0, 30.0]})}
    )
    _, hit3 = cache.load_or_prepare(spec, 3, vocab, band, prepare=prepare)
    assert (hit3, calls) == (False, [3, 3])


def test_store_checks_its_inputs_and_keeps_the_first_copy(
    tmp_path: Path, vocab: Vocabulary
) -> None:
    spec, config = make_spec(), DataConfig()
    cache = RecordingCache(tmp_path)
    key = _key(spec, vocab, config)
    payload = key_payload(spec, vocab, config)
    with pytest.raises(NeurOSDataError, match="nothing to cache"):
        cache.store(spec, 1, key, payload, [])
    with pytest.raises(NeurOSDataError, match="can't be cached as"):
        cache.store(spec, 2, key, payload, _recordings(config, subject=1))
    with pytest.raises(NeurOSDataError, match="64-character"):
        cache.entry_dir(spec.key, 1, "abc")
    first = _recordings(config, subject=1)
    cache.store(spec, 1, key, payload, first)
    # A second writer (another process) finds the entry already there: theirs is kept.
    cache.store(spec, 1, key, payload, _recordings(config, subject=1))
    entries = [p for p in cache.entry_dir(spec.key, 1, key).parent.iterdir() if p.is_dir()]
    assert [p.name for p in entries] == ["sub-1"]  # no staging folders left behind
    loaded = cache.load(spec, 1, key)
    assert loaded is not None
    assert np.array_equal(loaded[0].data, first[0].data)


def test_subject_epochs_identical_with_and_without_cache(
    tmp_path: Path, vocab: Vocabulary, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, config = make_spec(), DataConfig()
    maps = {**vocab.event_maps, "Synthetic": {"left_hand": "left_hand", "right_hand": "right_hand"}}
    vocab = vocab.model_copy(update={"event_maps": maps})
    monkeypatch.setattr(
        prepare_module,
        "prepare_subject",
        lambda spec_arg, subject, *a, **k: _recordings(config, subject),
    )
    labels = ("left_hand", "right_hand")
    plain, summary = prepare_module.subject_epochs(spec, 1, vocab, config, labels)
    cache = RecordingCache(tmp_path)
    cold, cold_summary = prepare_module.subject_epochs(spec, 1, vocab, config, labels, cache=cache)
    warm, warm_summary = prepare_module.subject_epochs(spec, 1, vocab, config, labels, cache=cache)
    assert (summary.cache, cold_summary.cache, warm_summary.cache) == ("off", "miss", "hit")
    for other in (cold, warm):
        assert np.array_equal(plain.data, other.data)
        assert plain.labels == other.labels
