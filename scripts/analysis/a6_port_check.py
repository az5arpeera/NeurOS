"""A6's port check: do REVE's published weights carry real knowledge through our port?

The check (``knowledge-base/plan/a6-pretrained-backbone.md``, 'Port check', set before any run):
REVE with its encoder frozen for every epoch (a linear probe on top of a fixed encoder), on a
10-person-per-dataset subset, once with the published weights and once with random weights.
A faithful port of a useful encoder must beat its random-weights twin by at least 5 points of
validation accuracy (the mean over folds of the kept model's balanced accuracy on the fold's
validation people). If it doesn't, the port is suspect and A6 stops until the cause is found.

Usage:
    uv run --no-sync python scripts/analysis/a6_port_check.py PRETRAINED.json RANDOM.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

MIN_GAIN_POINTS = 5.0


def load_run(path: Path) -> dict[str, Any]:
    """A complete pooled run's JSON.

    Raises:
        SystemExit: if the file is missing, unreadable or the run didn't finish.
    """
    try:
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"{path}: can't read the run ({error})") from error
    if not data.get("complete"):
        raise SystemExit(f"{path}: the run isn't complete")
    return data


def fold_validation(data: dict[str, Any], path: Path) -> dict[str, dict[str, float]]:
    """``task -> {validation, test, train_people, validation_people}`` for each pooled fold.

    Raises:
        SystemExit: if a fold has no validation result (the probe can't be judged without one).
    """
    folds: dict[str, dict[str, float]] = {}
    for task in data["tasks"]:
        if task["condition"] != "pooled":
            continue
        final = task["train"].get("final_validation")
        if not final:
            raise SystemExit(f"{path}: {task['task']} has no validation result")
        test = [s["balanced_accuracy_raw"] for s in task["scores"]]
        folds[task["task"]] = {
            "validation": float(final["balanced_accuracy"]),
            "test": float(np.mean(test)) if test else float("nan"),
            "train_people": float(task["n_train_people"]),
            "validation_people": float(task["n_validation_people"]),
        }
    if not folds:
        raise SystemExit(f"{path}: no pooled folds")
    return folds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pretrained", type=Path)
    parser.add_argument("random", type=Path)
    args = parser.parse_args()

    runs = {"pretrained": load_run(args.pretrained), "random": load_run(args.random)}
    # Guard the comparison itself: the two runs must differ only in the weights.
    weights = {k: (v["experiment"].get("backbone") or {}).get("weights") for k, v in runs.items()}
    if not weights["pretrained"] or weights["random"] is not None:
        raise SystemExit(f"expected published weights vs none, got {weights}")
    pre = fold_validation(runs["pretrained"], args.pretrained)
    rnd = fold_validation(runs["random"], args.random)
    if sorted(pre) != sorted(rnd):
        raise SystemExit(f"the runs have different folds: {sorted(pre)} vs {sorted(rnd)}")
    for name in pre:  # same seed and subset, so the same people in every fold
        for key in ("train_people", "validation_people"):
            if pre[name][key] != rnd[name][key]:
                raise SystemExit(f"{name}: {key} differs ({pre[name][key]} vs {rnd[name][key]})")

    print(f"{'fold':16s} {'pretrained':>11s} {'random':>8s} {'gain':>7s}   (held-out people)")
    for name in sorted(pre):
        a, b = pre[name], rnd[name]
        print(
            f"{name:16s} {100 * a['validation']:10.1f}% {100 * b['validation']:7.1f}% "
            f"{100 * (a['validation'] - b['validation']):+6.1f}   "
            f"({100 * a['test']:.1f}% vs {100 * b['test']:.1f}%)"
        )
    a_val = np.mean([pre[n]["validation"] for n in pre])
    b_val = np.mean([rnd[n]["validation"] for n in rnd])
    gain = 100 * float(a_val - b_val)
    print(f"{'mean':16s} {100 * a_val:10.1f}% {100 * b_val:7.1f}% {gain:+6.1f}")
    passed = gain >= MIN_GAIN_POINTS
    print(f"PORT CHECK: {'PASS' if passed else 'FAIL'} (bar: gain >= +{MIN_GAIN_POINTS:g} points)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
