"""The model bundle (architecture §6.5): train, save, load, build the engine for a headset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from neuros.config.schemas import SignalContract
from neuros.data.synthetic import SyntheticConfig, make_recording
from neuros.engine.bundle import ModelBundle, PersonCovariance, person_covariances, train_bundle
from neuros.engine.events import IntentEvent, StatusEvent
from neuros.engine.pipeline import Engine
from neuros.engine.simulator import Simulator
from neuros.errors import NeurOSConfigError, NeurOSDataError
from neuros.signal.preprocess import preprocess_recording
from neuros.signal.windows import WindowSet, extract_epochs

from .synthetic_people import FAST_TRAIN, TINY_MODEL

LAYOUT = ("FC3", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP3", "CP4", "CP5")
LABELS = ("left_hand", "right_hand")
CONTRACT = SignalContract(band_hz=(8.0, 30.0))


def _dataset(name: str, subjects: range, seed: int) -> WindowSet:
    sets = []
    for subject in subjects:
        raw = make_recording(
            SyntheticConfig(n_trials=24, seed=seed + subject), dataset=name, subject=subject
        )
        prepared = preprocess_recording(raw, CONTRACT)
        sets.append(extract_epochs(prepared, CONTRACT, labels=LABELS, epoch_s=(0.5, 2.5))[0])
    return WindowSet.concat(sets)


@pytest.fixture(scope="module")
def bundle() -> ModelBundle:
    windows = {"SynA": _dataset("SynA", range(1, 5), 10), "SynB": _dataset("SynB", range(1, 5), 50)}
    return train_bundle(
        windows,
        LAYOUT,
        classes=LABELS,
        model_config=TINY_MODEL,
        train_config=FAST_TRAIN.model_copy(update={"patience": 5}),
        contract=CONTRACT,
        reference="car",
        alignment="ea",
        unit="session",
        derived=None,
        validation_fraction=0.25,
        seed=3,
        device="cpu",
        provenance={"test": True},
    )


def test_trained_bundle_records_people_and_scale(bundle: ModelBundle) -> None:
    assert bundle.classes == LABELS
    assert bundle.layout == LAYOUT
    assert bundle.input_scale > 0
    assert len(bundle.people_covariances) == 8
    assert bundle.provenance["people_per_dataset"] == {"SynA": 4, "SynB": 4}
    assert len(bundle.provenance["validation_people"]) == 2  # 25% of 4 per dataset
    assert "history" not in bundle.provenance["training"]


def test_save_load_round_trip_and_tamper_check(bundle: ModelBundle, tmp_path: Path) -> None:
    bundle.save(tmp_path / "b")
    loaded = ModelBundle.load(tmp_path / "b")
    x = np.random.default_rng(0).normal(size=(5, len(LAYOUT), 256)).astype(np.float32)
    np.testing.assert_allclose(
        loaded.decoder().predict_proba(x), bundle.decoder().predict_proba(x), rtol=1e-6
    )
    np.testing.assert_allclose(
        loaded.prior_for(("C3", "Cz", "C4")), bundle.prior_for(("C3", "Cz", "C4"))
    )
    assert loaded.contract == bundle.contract
    with pytest.raises(NeurOSDataError, match="already holds"):
        bundle.save(tmp_path / "b")
    (tmp_path / "b" / "model.pt").write_bytes(b"tampered")
    with pytest.raises(NeurOSDataError, match="SHA-256"):
        ModelBundle.load(tmp_path / "b")
    meta = json.loads((tmp_path / "b" / "bundle.json").read_text(encoding="utf-8"))
    meta["bundle_version"] = 99
    (tmp_path / "b" / "bundle.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(NeurOSDataError, match="version"):
        ModelBundle.load(tmp_path / "b")
    with pytest.raises(NeurOSDataError, match="no bundle"):
        ModelBundle.load(tmp_path / "missing")


def test_prior_equals_the_mean_of_common_averaged_covariances() -> None:
    # Covariance is linear: H·mean(C)·H == mean(H·C·H) == E2's prior from CAR'd windows.
    from neuros.align.covariance import window_covariances
    from neuros.signal.reference import common_average_reference

    windows = _dataset("SynA", range(1, 4), 70)
    covs = person_covariances(windows, LAYOUT)
    profile = ("C3", "Cz", "C4", "CP3")
    expected = []
    for person in windows.people:
        mine = windows.for_people({person}).pick_channels(list(profile))
        expected.append(window_covariances(common_average_reference(mine.data)).mean(axis=0))
    b = ModelBundle.__new__(ModelBundle)  # only the prior's inputs are needed
    b.people_covariances = tuple(covs)
    b.contract = CONTRACT
    # Equal up to float32 rounding (the windows are float32; the algebra here is float64).
    np.testing.assert_allclose(b.prior_for(profile), np.mean(expected, axis=0), rtol=1e-6)
    with pytest.raises(NeurOSConfigError, match="no training person"):
        b.prior_for(("C3", "CP5"))  # nobody recorded CP5
    with pytest.raises(NeurOSDataError):
        PersonCovariance("x", ("C3",), np.eye(1))


def test_a_bundle_drives_the_engine_on_a_new_person(bundle: ModelBundle) -> None:
    raw = make_recording(SyntheticConfig(n_trials=12, seed=999), dataset="SynC", subject=1)
    simulator = Simulator(raw)
    engine = Engine(bundle.decoder(), bundle.engine_config(raw.ch_names, raw.sfreq))
    events = simulator.run(engine)
    intents = [e for e in events if isinstance(e, IntentEvent)]
    assert intents
    assert {e.state for e in events if isinstance(e, StatusEvent)} >= {"settling", "disconnected"}
    assert all(0.0 <= e.confidence <= 1.0 for e in intents)
    with pytest.raises(NeurOSConfigError, match="at least 2"):
        bundle.engine_config(("Pz", "C3"), 250.0)


def test_a_pretrained_encoder_bundle_round_trips_without_the_published_file(
    tmp_path: Path,
) -> None:
    # A6 (mvp-v1): a MIRepNet bundle trains through the same path, records its backbone and
    # licence, and reloads from its own model.pt alone (random weights stand in for the
    # published file here: the bundle never needs it once trained).
    from neuros.config.experiment import BackboneConfig
    from neuros.models.pretrained import MIRepNetClassifier

    windows = {"SynA": _dataset("SynA", range(1, 4), 10), "SynB": _dataset("SynB", range(1, 4), 50)}
    backbone = BackboneConfig(name="mirepnet", weights=None)
    trained = train_bundle(
        windows,
        LAYOUT,
        classes=LABELS,
        model_config=TINY_MODEL,
        train_config=FAST_TRAIN.model_copy(update={"max_epochs": 1, "patience": None}),
        contract=CONTRACT,
        reference="car",
        alignment="ea",
        unit="session",
        derived=None,
        validation_fraction=0.34,
        seed=3,
        device="cpu",
        provenance={"test": True},
        backbone=backbone,
    )
    assert isinstance(trained.model, MIRepNetClassifier)
    assert "pretrained_weights_sha256" not in trained.provenance  # random weights: no file
    trained.save(tmp_path / "m")
    meta = json.loads((tmp_path / "m" / "bundle.json").read_text(encoding="utf-8"))
    assert meta["bundle_version"] == 2
    assert meta["backbone"]["name"] == "mirepnet"
    assert meta["backbone_licence"].startswith("MIT")
    assert meta["model_sfreq"] == CONTRACT.sfreq
    loaded = ModelBundle.load(tmp_path / "m")
    assert isinstance(loaded.model, MIRepNetClassifier)
    assert loaded.backbone == backbone
    x = np.random.default_rng(1).normal(size=(4, len(LAYOUT), 256)).astype(np.float32)
    np.testing.assert_allclose(
        loaded.decoder().predict_proba(x), trained.decoder().predict_proba(x), rtol=1e-5
    )
    # A model.pt that doesn't fit the recorded architecture fails as a data error, not silently.
    meta["backbone"] = None  # claim EEGNet; the MIRepNet weights can't load into it
    (tmp_path / "m" / "bundle.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(NeurOSDataError, match="doesn't fit"):
        ModelBundle.load(tmp_path / "m")


def test_version_1_bundles_still_load_as_eegnet(bundle: ModelBundle, tmp_path: Path) -> None:
    # mvp-v0 was written before version 2 added the backbone fields.
    bundle.save(tmp_path / "v1")
    meta_path = tmp_path / "v1" / "bundle.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for key in ("backbone", "backbone_licence", "model_sfreq"):
        meta.pop(key)
    meta["bundle_version"] = 1
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    loaded = ModelBundle.load(tmp_path / "v1")
    assert loaded.backbone is None
    x = np.random.default_rng(2).normal(size=(3, len(LAYOUT), 256)).astype(np.float32)
    np.testing.assert_allclose(
        loaded.decoder().predict_proba(x), bundle.decoder().predict_proba(x), rtol=1e-6
    )


def test_decoder_reads_only_the_scored_classes(bundle: ModelBundle) -> None:
    probs = bundle.decoder().predict_proba(np.zeros((2, len(LAYOUT), 256), dtype=np.float32))
    assert probs.shape == (2, 2)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)
    with pytest.raises(NeurOSConfigError):
        type(bundle.decoder())(bundle.model, LAYOUT[:-1], LABELS)
