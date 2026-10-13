"""A6's real-time check: decision latency of each candidate model on this machine's CPU.

The engine decides once per hop (8 times a second) on one 2-s window, so the bar
(``knowledge-base/plan/a6-pretrained-backbone.md``) is p99 < 50 ms for a single window, the
same bar the engine replay applies. The models load their published weights strictly first,
so this also proves the weights fit the re-implementation.

Usage (from the repository root, after the approved downloads into data/pretrained/):
    uv run --no-sync python scripts/analysis/a6_latency.py [--repeats 300] [--threads 4]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from neuros.models.eegnet import EEGNetClassifier
from neuros.models.pretrained import MIRepNetClassifier, ReveClassifier

LAYOUT = (
    "FC5", "FC3", "FC1", "FC2", "FC4", "FC6", "C5", "C3", "C1", "Cz",
    "C2", "C4", "C6", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
)  # fmt: skip
CLASSES = ("left_hand", "right_hand")
SFREQ, N_SAMPLES = 128.0, 256  # the contract: 2-s windows at 128 Hz
BAR_P99_MS = 50.0


def time_model(model: torch.nn.Module, x: torch.Tensor, repeats: int) -> dict[str, float]:
    """Per-call milliseconds for ``model(x)`` in eval mode without gradients (after warm-up)."""
    model.eval()
    times = []
    with torch.inference_mode():
        for _ in range(10):  # warm-up: allocator and kernel choice
            model(x)
        for _ in range(repeats):
            started = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - started) * 1000)
    times.sort()
    return {
        "median_ms": statistics.median(times),
        "p99_ms": times[min(len(times) - 1, int(0.99 * len(times)))],
        "max_ms": times[-1],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeats", type=int, default=300)
    parser.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    parser.add_argument("--pretrained", default="data/pretrained")
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    root = Path(args.pretrained)
    torch.manual_seed(0)

    reve = ReveClassifier(LAYOUT, N_SAMPLES, CLASSES, sfreq=SFREQ)
    reve.load_pretrained(root / "reve-base" / "model.safetensors")
    mirepnet = MIRepNetClassifier(LAYOUT, N_SAMPLES, CLASSES, sfreq=SFREQ)
    mirepnet.load_pretrained(root / "mirepnet" / "MIRepNet.pth")
    eegnet = EEGNetClassifier(len(LAYOUT), N_SAMPLES, CLASSES)

    # A Ganglion-class window too: only C3, Cz and C4 present (the others zero-filled).
    full = torch.randn(1, len(LAYOUT), N_SAMPLES)
    three = torch.zeros_like(full)
    for ch in ("C3", "Cz", "C4"):
        three[0, LAYOUT.index(ch)] = full[0, LAYOUT.index(ch)]

    report: dict[str, object] = {"threads": torch.get_num_threads(), "bar_p99_ms": BAR_P99_MS}
    for name, model in (("eegnet", eegnet), ("mirepnet", mirepnet), ("reve-base", reve)):
        full_layout = time_model(model, full, args.repeats)
        three_channels = time_model(model, three, args.repeats)
        n_parameters = sum(p.numel() for p in model.parameters())
        passes = full_layout["p99_ms"] < BAR_P99_MS
        report[name] = {
            "n_parameters": n_parameters,
            "full_layout": full_layout,
            "three_channels": three_channels,
            "passes": passes,
        }
        print(
            f"{name:10s} {n_parameters:>11,} params  20 ch: median "
            f"{full_layout['median_ms']:.1f} ms, p99 {full_layout['p99_ms']:.1f} ms  |  3 ch: "
            f"p99 {three_channels['p99_ms']:.1f} ms  -> {'PASS' if passes else 'FAIL'} "
            f"(bar p99 < {BAR_P99_MS:g} ms)",
            flush=True,
        )
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
