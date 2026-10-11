"""How much training data would zero calibration need to work for (nearly) everyone?

A desk projection from results we already have; it trains nothing and spends no compute.

Usage: uv run --no-sync python scripts/analysis/scaling_projection.py [results/aws]

The three steps:
1. **How fast accuracy grows with training people** (E1, 3 seeds): each dataset's people scored
   by a model trained on their own dataset only (~6-60 people) and by the pooled model (~244
   people). The gain per doubling of training people is beta = gain / log2(N_pooled / N_single).
2. **Where people stand today** (E2, cue-free people, zero calibration after 120 s of use): each
   person's accuracy, with sampling noise removed by empirical-Bayes shrinkage within each
   dataset (a person scored on 60 windows has a noisier score than one scored on 600).
3. **The projection.** If more data lifted everyone by the same amount, how big a lift, so how
   many doublings, so how many training people, would put a given share of people at >= 70%?
   Then the ceiling check: the same share with each person's *own calibrated* accuracy (the
   calibration curve's half-session model), i.e. what a model that transfers perfectly could
   at best reach for that person with this hardware and task.

Assumptions (stated, not tested):
- **Log-linear scaling:** each doubling of training people adds the same beta.
- **A uniform lift:** everyone gains equally. Both are generous to scaling: gains usually shrink
  per doubling, and people with no usable signal gain least.

The helpers above ``main`` (loading, scoring, shrinkage, the cue-free people's two scores) are
shared with ``reliability_time.py``.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

USABLE = 0.70  # the usual two-command "can steer an app" line (headset self-test, E2)
CUE_FREE = ("Yang2025", "BNCI2014_001", "Shin2017A")
TARGET_SHARES = (0.25, 0.50, 0.80, 0.95, 1.00)
WORLD = 8.1e9  # people alive, as a sanity bound on any projected training set

# dataset -> [(accuracy, number of scored windows or trials)], one entry per person
Scores = dict[str, list[tuple[float, int]]]


def load(path: Path) -> dict:
    """Read one result JSON, refusing an incomplete or missing run."""
    if not path.is_file():
        sys.exit(f"missing result file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("complete") is False:
        sys.exit(f"{path} is an incomplete run")
    return data


def score(row: dict) -> float:
    """A person's balanced accuracy, unrounded when the file has it."""
    return float(row.get("balanced_accuracy_raw", row["balanced_accuracy"]))


def shrink(acc: np.ndarray, n: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Empirical-Bayes shrinkage of per-person accuracies toward their dataset mean.

    Each observed score = true accuracy + binomial noise with variance p(1-p)/n. The true
    between-person variance tau^2 is the observed variance minus the mean noise variance (floored
    at a small positive value), and each person moves toward the mean by tau^2 / (tau^2 + noise).
    Returns (shrunk scores, observed SD, true SD).
    """
    if acc.size < 3 or np.any(n <= 0):
        raise ValueError("need >= 3 people with positive window counts")
    noise = acc * (1 - acc) / n
    tau2 = max(acc.var(ddof=1) - noise.mean(), 1e-6)
    weight = tau2 / (tau2 + noise)
    return acc.mean() + weight * (acc - acc.mean()), float(acc.std(ddof=1)), math.sqrt(tau2)


def cue_free_scores(root: Path) -> tuple[Scores, Scores]:
    """The cue-free people's two scores: (today, ceiling).

    - **today:** zero calibration after 120 s of use (E2's ``e2_stream_120s``), with the number
      of windows scored;
    - **ceiling:** their own calibrated model on half a session (the calibration curve's
      ``n_per_class == "all"`` entry), with the number of test trials summed over blocks.
    """
    e2_path = sorted((root / "e2").glob("e2-*.json"))
    if not e2_path:
        sys.exit(f"no E2 result in {root / 'e2'}")
    now: Scores = defaultdict(list)
    for task in load(e2_path[-1])["tasks"]:
        for row in task.get("extra_scores", {}).get("e2_stream_120s", []):
            dataset = row["person"].split(":")[0]
            if dataset in CUE_FREE:
                now[dataset].append((score(row), int(row["n_windows"])))

    ceiling: Scores = defaultdict(list)
    cal_paths = sorted((root / "calibration").glob("*.json"))
    if not cal_paths:
        sys.exit(f"no calibration-curve result in {root / 'calibration'}")
    for path in cal_paths:
        cal = load(path)
        tests: dict[str, int] = defaultdict(int)
        for block in cal["blocks"]:
            tests[block["person"]] += int(block["n_test"])
        full = next(e for e in cal["curve"] if e["n_per_class"] == "all")
        for row in full["per_person"]:
            ceiling[row["person"].split(":")[0]].append((score(row), tests[row["person"]]))

    missing = [d for d in CUE_FREE if d not in now or d not in ceiling]
    if missing:
        sys.exit(f"no scores for {missing} in {root}")
    return now, ceiling


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results/aws")

    # ---- 1. Gain per doubling of training people (E1, three seeds) -----------------------
    by_person: dict[tuple[str, str], list[float]] = defaultdict(list)  # (condition, person)
    n_train: dict[tuple[str, str], list[int]] = defaultdict(list)  # (condition, dataset)
    e1_files = sorted((root / "e1").glob("e1-*.json"))
    if len(e1_files) < 2:
        sys.exit(f"expected E1's seed files in {root / 'e1'}")
    for path in e1_files:
        for task in load(path)["tasks"]:
            if task["condition"] not in ("single", "pooled") or task["data"] != "task":
                continue
            for row in task["scores"]:
                dataset = row["person"].split(":")[0]
                by_person[(task["condition"], row["person"])].append(score(row))
                n_train[(task["condition"], dataset)].append(task["n_train_people"])

    print("1. Gain per doubling of training people (E1, 3 seeds; single-dataset vs pooled model)")
    betas, weights = {}, {}
    for dataset in sorted({d for (c, d) in n_train if c == "single"}):
        people = [p for (c, p) in by_person if c == "single" and p.startswith(dataset + ":")]
        paired = [p for p in people if ("pooled", p) in by_person]
        single = np.mean([np.mean(by_person[("single", p)]) for p in paired])
        pooled = np.mean([np.mean(by_person[("pooled", p)]) for p in paired])
        n_s = float(np.mean(n_train[("single", dataset)]))
        n_p = float(np.mean([n for (c, d), ns in n_train.items() if c == "pooled" for n in ns]))
        doublings = math.log2(n_p / n_s)
        betas[dataset] = (pooled - single) * 100 / doublings
        weights[dataset] = len(paired)
        print(
            f"  {dataset:14s} N {n_s:5.0f} -> {n_p:4.0f} ({doublings:3.1f} doublings): "
            f"{single * 100:5.1f}% -> {pooled * 100:5.1f}%  = "
            f"{betas[dataset]:+5.2f} points/doubling"
        )
    w = np.array([weights[d] for d in betas])
    b = np.array([betas[d] for d in betas])
    beta_all = float((w * b).sum() / w.sum())
    keep = np.array([d != "BNCI2014_001" for d in betas])
    beta_no_tiny = float((w[keep] * b[keep]).sum() / w[keep].sum())
    print(
        f"  people-weighted: {beta_all:+.2f} points/doubling; without BNCI2014_001 (trained on ~6 "
        f"people alone): {beta_no_tiny:+.2f}"
    )

    # ---- 2. Where cue-free people stand today (E2: zero calibration after 120 s) ---------
    now, ceiling = cue_free_scores(root)
    print("\n2. Cue-free people today (zero calibration, 120 s) and their own calibrated ceiling")
    today, best = [], []
    for dataset in CUE_FREE:
        for label, table, out in (("today", now, today), ("ceiling", ceiling, best)):
            acc = np.array([a for a, _ in table[dataset]])
            n = np.array([k for _, k in table[dataset]], dtype=float)
            shrunk, sd_obs, sd_true = shrink(acc, n)
            out.extend(shrunk)
            print(
                f"  {dataset:13s} {label:8s} n={acc.size:2d} mean {acc.mean() * 100:5.1f}%  "
                f"SD {sd_obs * 100:4.1f} -> {sd_true * 100:4.1f} without noise;  >= 70%: "
                f"{(acc >= USABLE).sum():2d} observed, "
                f"{(shrunk >= USABLE).sum():2d} noise-corrected"
            )
    today_a, best_a = np.sort(np.array(today)), np.sort(np.array(best))
    print(
        f"  all {today_a.size} people: today {np.mean(today_a >= USABLE):.0%} >= 70%; at their own "
        f"calibrated ceiling {np.mean(best_a >= USABLE):.0%}; ceiling + 4 points (our fine-tuning "
        f"beat this calibration by 3-6) {np.mean(best_a + 0.04 >= USABLE):.0%}"
    )

    # ---- 3. The projection ----------------------------------------------------------------
    print("\n3. Training people needed for a share of people at >= 70% (uniform lift, log-linear)")
    n_now = 244
    print(f"   (today's pooled model: {n_now} training people; lift per doubling as measured)")
    betas_shown = (beta_no_tiny, beta_all, 1.0)  # measured (two ways), and an optimistic 1.0
    print("  share  lift needed   " + "   ".join(f"beta {x:.2f}" for x in betas_shown))
    for share in TARGET_SHARES:
        # The lift that brings the (1 - share) quantile person up to 70% (the weakest for 100%).
        person = today_a[0] if share >= 1 else np.quantile(today_a, 1 - share)
        lift = max(0.0, (USABLE - person) * 100)
        cells = []
        for beta in betas_shown:
            people = n_now * 2 ** (lift / beta) if beta > 0 else math.inf
            cells.append(f"{people:9.1e}" + ("*" if people > WORLD else " "))
        print(f"  {share:4.0%}   {lift:5.1f} points   " + "   ".join(cells))
    print(f"  * more training people than the {WORLD:.1e} alive")
    for share in TARGET_SHARES:
        person = best_a[0] if share >= 1 else np.quantile(best_a, 1 - share)
        print(
            f"  ceiling check, {share:4.0%} of people: the weakest of them reaches "
            f"{person * 100:4.1f}% even fully calibrated"
        )


if __name__ == "__main__":
    main()
