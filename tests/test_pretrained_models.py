"""A6's pretrained encoders (neuros.models.pretrained): safe weight loading, the ports' building
blocks, absent-channel handling, strict loading, and training with a frozen backbone.

These run without the downloaded weights (tiny random models). The last test loads the real
published weights when they are present locally (data/pretrained/, never committed).
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from neuros.align.ea import align_per_person  # noqa: E402
from neuros.config.experiment import BackboneConfig, TrainConfig  # noqa: E402
from neuros.errors import NeurOSConfigError, NeurOSDataError  # noqa: E402
from neuros.models import pretrained as pt  # noqa: E402
from neuros.train.data import windows_to_tensors  # noqa: E402
from neuros.train.trainer import train_classifier  # noqa: E402

from .synthetic_people import CHANNELS, N_SAMPLES, SFREQ, people_windows  # noqa: E402

CLASSES = ("left_hand", "right_hand")
TINY_REVE = pt.ReveDims(
    embed_dim=32, depth=1, heads=2, head_dim=8, mlp_dim_ratio=2.0, freqs=3,
    patch_size=50, patch_overlap=5, sfreq=200.0,
)  # fmt: skip
TINY_MI = pt.MIRepNetDims(emb_size=16, depth=1, heads=2, dropout=0.0)
REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _one_thread() -> None:
    torch.set_num_threads(1)


def write_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    """A minimal safetensors writer (float32 only), for the reader's tests."""
    header: dict[str, object] = {"__metadata__": {"format": "pt"}}
    blobs, offset = [], 0
    for name, tensor in tensors.items():
        data = tensor.detach().to(torch.float32).contiguous().numpy().tobytes()
        header[name] = {
            "dtype": "F32",
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        blobs.append(data)
        offset += len(data)
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"".join(blobs))


# ---------------------------------------------------------------------- loading


def test_safetensors_round_trip_and_corruption(tmp_path: Path) -> None:
    tensors = {"a": torch.randn(3, 4), "b.c": torch.arange(5.0)}
    path = tmp_path / "w.safetensors"
    write_safetensors(path, tensors)
    loaded = pt.load_safetensors(path)
    assert set(loaded) == {"a", "b.c"}
    for k, v in tensors.items():
        assert torch.equal(loaded[k], v)
    raw = path.read_bytes()
    (path.parent / "short.safetensors").write_bytes(raw[:-4])  # truncated body
    with pytest.raises(NeurOSDataError, match="don't match its shape"):
        pt.load_safetensors(path.parent / "short.safetensors")
    (path.parent / "junk.safetensors").write_bytes(struct.pack("<Q", 10**12) + b"{}")
    with pytest.raises(NeurOSDataError, match="header length"):
        pt.load_safetensors(path.parent / "junk.safetensors")
    with pytest.raises(NeurOSConfigError, match="not found"):
        pt.load_safetensors(tmp_path / "missing.safetensors")


def test_tensor_state_refuses_non_tensors(tmp_path: Path) -> None:
    good = tmp_path / "good.pth"
    torch.save({"w": torch.ones(2)}, good)
    assert torch.equal(pt.load_tensor_state(good)["w"], torch.ones(2))
    bad = tmp_path / "bad.pth"
    torch.save({"w": torch.ones(2), "note": "a string, not a tensor"}, bad)
    with pytest.raises(NeurOSDataError, match="names to tensors"):
        pt.load_tensor_state(bad)


def test_electrode_positions_are_mne_1005() -> None:
    pos = pt.electrode_positions(["C3", "Cz", "C4"])
    assert pos.shape == (3, 3)
    assert pos[0, 0] < 0 < pos[2, 0]  # left and right of the midline, in metres
    assert abs(float(pos[1, 0])) < 0.01
    with pytest.raises(NeurOSConfigError, match="no 10-05 position"):
        pt.electrode_positions(["C3", "NOT_AN_ELECTRODE"])


# ---------------------------------------------------------------------- building blocks


def test_fft_resample_is_exact_for_periodic_band_limited_signals() -> None:
    t_in = torch.arange(256) / 128.0
    x = torch.sin(2 * math.pi * 10 * t_in) + 0.5 * torch.cos(2 * math.pi * 21.5 * t_in)
    y = pt.fft_resample(x[None], 400)[0]
    t_out = torch.arange(400) / 200.0
    expected = torch.sin(2 * math.pi * 10 * t_out) + 0.5 * torch.cos(2 * math.pi * 21.5 * t_out)
    assert torch.allclose(y, expected, atol=1e-4)
    assert torch.equal(pt.fft_resample(x[None], 256), x[None])
    with pytest.raises(NeurOSConfigError):
        pt.fft_resample(x[None], 0)


def test_fourier_4d_matches_the_published_loop_order() -> None:
    torch.manual_seed(0)
    pos = torch.rand(2, 5, 4) * 0.1
    freqs, dim = 3, 40
    got = pt.fourier_4d(pos, dim, freqs)
    # Reference: REVE's FourierEmb4D written as explicit loops over (x, y, z, t) frequencies.
    p = pos.clone()
    p[..., 3] *= 0.1
    p = p + 0.4
    w = 2 * math.pi / 1.8
    phases = []
    for i in range(freqs):
        for j in range(freqs):
            for k in range(freqs):
                for m in range(freqs):
                    phase = p[..., 0] * i * w + p[..., 1] * j * w
                    phases.append(phase + p[..., 2] * k * w + p[..., 3] * m * w)
    loc = torch.stack(phases, dim=-1)[..., : dim // 2]
    assert torch.allclose(got, torch.cat([loc.cos(), loc.sin()], dim=-1), atol=1e-6)
    with pytest.raises(NeurOSConfigError):
        pt.fourier_4d(pos, 512, 2)  # 2^4 = 16 phases can't fill 256


def test_present_channels() -> None:
    x = torch.randn(2, 3, 10)
    x[0, 1] = 0.0
    assert pt.present_channels(x).tolist() == [[True, False, True], [True, True, True]]
    with pytest.raises(NeurOSDataError, match="no non-zero channel"):
        pt.present_channels(torch.zeros(1, 3, 10))


# ---------------------------------------------------------------------- REVE


def _reve(layout: tuple[str, ...], seed: int = 0) -> pt.ReveClassifier:
    torch.manual_seed(seed)
    return pt.ReveClassifier(layout, N_SAMPLES, CLASSES, sfreq=SFREQ, dropout=0.0, dims=TINY_REVE)


def test_reve_shapes_and_interface() -> None:
    model = _reve(("C3", "Cz", "C4", "CP3")).eval()
    x = torch.randn(5, 4, N_SAMPLES)
    assert model(x).shape == (5, 2)
    assert model.embed(x).shape == (5, TINY_REVE.embed_dim)
    groups = dict(model.parameter_groups())
    n_backbone = sum(p.numel() for p in groups["backbone"])
    n_head = sum(p.numel() for p in groups["head"])
    assert n_backbone + n_head == model.n_parameters
    assert n_head == TINY_REVE.embed_dim + TINY_REVE.embed_dim * 2 + 2  # norm + linear
    with pytest.raises(NeurOSDataError):
        model(torch.randn(5, 3, N_SAMPLES))  # wrong channel count
    with pytest.raises(NeurOSConfigError, match="whole number"):
        pt.ReveClassifier(("C3", "C4"), 100, CLASSES, sfreq=SFREQ, dims=TINY_REVE)


def test_reve_ignores_absent_channels_entirely() -> None:
    """A zero-filled channel must change nothing: the same as a layout without it."""
    with_cp3 = _reve(("C3", "Cz", "C4", "CP3")).eval()
    without = _reve(("C3", "Cz", "C4")).eval()
    without.load_state_dict(with_cp3.state_dict())
    x = torch.randn(4, 3, N_SAMPLES)
    padded = torch.cat([x, torch.zeros(4, 1, N_SAMPLES)], dim=1)
    with torch.no_grad():
        assert torch.allclose(with_cp3.embed(padded), without.embed(x), atol=1e-5)
        # ...and a present CP3 does change the embedding.
        filled = torch.cat([x, torch.randn(4, 1, N_SAMPLES)], dim=1)
        assert not torch.allclose(with_cp3.embed(filled), without.embed(x), atol=1e-3)


def test_reve_loads_weights_strictly(tmp_path: Path) -> None:
    source = _reve(("C3", "Cz", "C4"), seed=1)
    path = tmp_path / "reve.safetensors"
    write_safetensors(path, source.encoder.state_dict())
    target = _reve(("C3", "Cz", "C4"), seed=2)
    target.load_pretrained(path)
    for k, v in source.encoder.state_dict().items():
        assert torch.equal(target.encoder.state_dict()[k], v)
    state = source.encoder.state_dict()
    state.pop("ln.weight")
    write_safetensors(path, state)
    with pytest.raises(NeurOSDataError, match="missing"):
        target.load_pretrained(path)


# ---------------------------------------------------------------------- MIRepNet


def _mirepnet(layout: tuple[str, ...] = CHANNELS, seed: int = 0) -> pt.MIRepNetClassifier:
    torch.manual_seed(seed)
    return pt.MIRepNetClassifier(layout, N_SAMPLES, CLASSES, sfreq=SFREQ, dims=TINY_MI)


def test_mirepnet_template_mapping() -> None:
    model = _mirepnet(("C3", "Cz", "C4"))
    x = torch.randn(2, 3, N_SAMPLES)
    x[1, 1] = 0.0  # Cz absent in the second window
    present = pt.present_channels(x)
    template = model.to_template(x, present)
    names = [n for n, _, _ in pt.MIREPNET_TEMPLATE]
    assert template.shape == (2, 45, N_SAMPLES)
    assert torch.allclose(template[0, names.index("C3")], x[0, 0])
    assert torch.allclose(template[0, names.index("CZ")], x[0, 1])
    # Absent Cz is interpolated from C3 and C4, equidistant on the grid: their mean.
    assert torch.allclose(template[1, names.index("CZ")], (x[1, 0] + x[1, 2]) / 2, atol=1e-6)
    # A far template channel is a weighted mean too (weights sum to 1).
    grid = {n: (a, b) for n, a, b in pt.MIREPNET_TEMPLATE}
    d = [math.dist(grid["F7"], grid[c]) for c in ("C3", "CZ", "C4")]
    w = torch.tensor([1 / (v + 1e-6) for v in d])
    w = w / w.sum()
    assert torch.allclose(template[0, names.index("F7")], (w[:, None] * x[0]).sum(0), atol=1e-5)
    with pytest.raises(NeurOSConfigError, match="no position"):
        pt.MIRepNetClassifier(("C3", "O9"), N_SAMPLES, CLASSES, sfreq=SFREQ, dims=TINY_MI)


def test_mirepnet_loads_the_encoder_and_drops_pretraining_parts(tmp_path: Path) -> None:
    source = _mirepnet(seed=1)
    state = {f"embedding.{k}": v for k, v in source.embedding.state_dict().items()}
    state |= {f"transformer.{k}": v for k, v in source.transformer.state_dict().items()}
    state |= {
        "mask_token": torch.zeros(1, 1, 16),
        "clshead.weight": torch.zeros(3, 16),
        "decoder.transformer.0.0.fn.0.weight": torch.ones(16),
    }
    path = tmp_path / "mi.pth"
    torch.save(state, path)
    target = _mirepnet(seed=2)
    target.load_pretrained(path)
    for k, v in source.embedding.state_dict().items():
        assert torch.equal(target.embedding.state_dict()[k], v)
    assert target(torch.randn(3, len(CHANNELS), N_SAMPLES)).shape == (3, 2)
    del state["transformer.0.0.fn.1.keys.weight"]
    torch.save(state, path)
    with pytest.raises(NeurOSDataError, match="missing"):
        target.load_pretrained(path)


# ---------------------------------------------------------------------- training


def test_frozen_backbone_epochs_and_backbone_lr_scale() -> None:
    aligned, _ = align_per_person(people_windows())
    train = windows_to_tensors(aligned.for_people({"Synthetic:1", "Synthetic:2"}), CLASSES)
    config = TrainConfig(
        max_epochs=2, people_per_batch=2, windows_per_person=8, lr=1e-3, swa_fraction=0.0,
        backbone_lr_scale=0.1, frozen_backbone_epochs=2, log_every=0,
    )  # fmt: skip
    model = _mirepnet()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    train_classifier(model, train, None, config, seed=0)
    after = model.state_dict()
    backbone = [k for k in before if k.startswith(("embedding.", "transformer."))]
    weights = [k for k in backbone if "running" not in k and "num_batches" not in k]
    assert all(torch.equal(before[k], after[k]) for k in weights)  # frozen throughout
    assert not torch.equal(before["head.weight"], after["head.weight"])  # the head learned
    assert all(p.requires_grad for p in model.parameters())  # trainable again afterwards

    unfrozen = config.model_copy(update={"frozen_backbone_epochs": 0})
    model = _mirepnet()
    train_classifier(model, train, None, unfrozen, seed=0)
    assert not all(torch.equal(before[k], model.state_dict()[k]) for k in weights)


def test_backbone_config() -> None:
    assert BackboneConfig(name="mirepnet").weights is None  # the random-weights control
    with pytest.raises(ValueError, match="name"):
        BackboneConfig(name="labram")  # type: ignore[arg-type]


@pytest.mark.skipif(
    not (REPO / "data/pretrained/reve-base/model.safetensors").is_file(),
    reason="the published weights aren't downloaded here",
)
def test_published_weights_load_strictly() -> None:
    layout = ("FC5", "C3", "Cz", "C4", "CP3")
    reve = pt.ReveClassifier(layout, 256, CLASSES, sfreq=128.0)
    reve.load_pretrained(REPO / "data/pretrained/reve-base/model.safetensors")
    assert sum(p.numel() for p in reve.encoder.parameters()) == 69_189_632
    mirepnet = pt.MIRepNetClassifier(layout, 256, CLASSES, sfreq=128.0)
    mirepnet.load_pretrained(REPO / "data/pretrained/mirepnet/MIRepNet.pth")
    x = torch.randn(2, len(layout), 256)
    with torch.no_grad():
        assert np.isfinite(reve.eval()(x).numpy()).all()
        assert np.isfinite(mirepnet.eval()(x).numpy()).all()
