"""A6's verdict: does a fine-tuned pretrained encoder beat E2's from-scratch EEGNet?

The bar (``knowledge-base/plan/a6-pretrained-backbone.md``, set before any run), per candidate:
1. Gain: mean >= E2 + 3.0 points over E2's 73 cue-free people after 120 s of unlabeled
   streaming (``e2_stream_120s``), paired Wilcoxon p < 0.05 (two-sided).
2. No dataset loses: each dataset's mean >= its E2 mean.
3. Real-time: p99 < 50 ms per decision on the laptop CPU (``a6_latency.py``'s JSON; passed 13 Oct).
4. Shippable licence (both candidates' licences allow a fine-tuned copy in the SDK; plan).

Usage:
    uv run --no-sync python scripts/analysis/a6_bar.py RESULT.json E2_REFERENCE.json \
        [--latency LATENCY.json]

E2_REFERENCE is any run with E2's models and ``e2_stream_120s`` scores: E2's own result file
or E5's (which reproduced E2 exactly).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from neuros.eval.metrics import paired_wilcoxon

KIND = "e2_stream_120s"
DATASETS = ("Yang2025", "BNCI2014_001", "Shin2017A")
MIN_GAIN_POINTS = 3.0
ALPHA = 0.05
LATENCY_BAR_MS = 50.0


def person_scores(path: Path, kind: str = KIND) -> dict[str, float]:
    """``person -> balanced accuracy`` for ``kind`` over every fold of a pooled run.

    Raises:
        SystemExit: if the file isn't a complete run or a person is scored twice.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data.get("complete"):
        raise SystemExit(f"{path}: the run isn't complete")
    scores: dict[str, float] = {}
    for task in data["tasks"]:
        for entry in (task.get("extra_scores") or {}).get(kind, []):
            if entry["person"] in scores:
                raise SystemExit(f"{path}: {entry['person']} scored twice")
            scores[entry["person"]] = float(entry["balanced_accuracy_raw"])
    if not scores:
        raise SystemExit(f"{path}: no {kind} scores")
    return scores


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("result", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--latency", type=Path, default=None, help="a6_latency.py JSON output")
    args = parser.parse_args()

    candidate = person_scores(args.result)
    reference = person_scores(args.reference)
    people = sorted(reference)
    missing = [p for p in people if p not in candidate]
    if missing:
        raise SystemExit(f"the candidate lacks {len(missing)} of E2's people: {missing[:5]}")
    a = np.array([candidate[p] for p in people])
    b = np.array([reference[p] for p in people])
    test = paired_wilcoxon(a, b)
    gain = 100 * float(np.mean(a - b))
    print(f"{args.result.name} vs {args.reference.name}: {len(people)} people, {KIND}")
    print(
        f"  overall: {100 * a.mean():.1f}% vs E2 {100 * b.mean():.1f}%: {gain:+.2f} points, "
        f"p = {test['p_value']:.3g}"
    )
    datasets_ok = True
    for dataset in DATASETS:
        mine = [i for i, p in enumerate(people) if p.startswith(f"{dataset}:")]
        if not mine:
            continue
        da, db = a[mine].mean(), b[mine].mean()
        datasets_ok &= bool(da >= db)
        print(
            f"  {dataset:13s} ({len(mine):2d}): {100 * da:.1f}% vs {100 * db:.1f}% "
            f"({100 * (da - db):+.2f}); at >= 70%: {int((a[mine] >= 0.7).sum())} vs "
            f"{int((b[mine] >= 0.7).sum())}"
        )

    latency_ok: bool | None = None
    if args.latency is not None:
        report = json.loads(args.latency.read_text(encoding="utf-8"))
        name = json.loads(args.result.read_text(encoding="utf-8"))["experiment"]["backbone"]["name"]
        latency_ok = bool(report[name]["full_layout"]["p99_ms"] < LATENCY_BAR_MS)
        print(
            f"  latency: p99 {report[name]['full_layout']['p99_ms']:.1f} ms "
            f"(bar < {LATENCY_BAR_MS:g})"
        )
    checks = {
        "gain >= +3 points": gain >= MIN_GAIN_POINTS,
        "p < 0.05": test["p_value"] < ALPHA,
        "no dataset loses": datasets_ok,
        "p99 latency < 50 ms": latency_ok,
    }
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else ('n/a ' if ok is None else 'FAIL')}  {name}")
    verdict = all(v for v in checks.values() if v is not None)
    print(f"VERDICT: {'PASS' if verdict else 'FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
