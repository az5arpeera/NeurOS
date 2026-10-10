"""Q30 (fine-tuning) and Q31 (resting rhythm) against their pre-registered bars (10 Oct).

Usage: uv run --no-sync python scripts/analysis/q30_q31.py <E2 json> <e2_finetune json> <smr json> <calibration json>...

Q31 bar: within-dataset Spearman >= 0.4 with p < 0.01 (pooled over the cue-free datasets),
resting SMR (first 60 s) vs zero-calibration accuracy (E2 pooled, full-session alignment).
Within-dataset: each variable is ranked within its dataset (percentile ranks), then pooled;
p from 20,000 permutations of SMR within each dataset.

Q30 bar: on Yang2025 and BNCI2014-001, fine-tuned with 10 per class at or above the
half-session calibration, not significantly worse (paired Wilcoxon), on the same test trials.
Reports a strict reading (mean at or above) and a lenient one (or p >= 0.05), plus 95% bootstrap
CIs over people. Results: knowledge-base/results/2026-10-10-q30-finetune-q31-fit-check.md
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr, wilcoxon


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


e2 = load(sys.argv[1])
ft = load(sys.argv[2])
smr = load(sys.argv[3])
cal_files = [load(p) for p in sys.argv[4:]]

dataset_of = lambda p: p.split(":", 1)[0]  # noqa: E731


def task_scores(payload, kind=None):
    out = {}
    for task in payload["tasks"]:
        if task["condition"] != "pooled":
            continue
        rows = task["scores"] if kind is None else task.get("extra_scores", {}).get(kind, [])
        for s in rows:
            out[s["person"]] = s["balanced_accuracy_raw"]
    return out


zero_full = task_scores(e2)
zero_120 = task_scores(e2, "e2_stream_120s")
calibrated = {}
for cal in cal_files:
    for point in cal["curve"]:
        for s in point["per_person"]:
            calibrated.setdefault(str(point["n_per_class"]), {})[s["person"]] = s["balanced_accuracy"]

# ------------------------------------------------------------------ determinism check
again = task_scores(ft)
common = sorted(set(again) & set(zero_full))
same = sum(abs(again[p] - zero_full[p]) < 1e-12 for p in common)
print(f"E2 reproduction: {same}/{len(common)} people identical")

# ------------------------------------------------------------------ Q31
print("\n=== Q31: resting SMR (first 60 s) vs accuracy")
people = sorted(smr["people"])
smr_first = {p: smr["people"][p]["first"]["smr_db"] for p in people}
smr_whole = {p: smr["people"][p]["whole_session"]["smr_db"] for p in people}


def within(values: dict, targets: dict, label: str, rng=np.random.default_rng(0)):
    keep = [p for p in values if p in targets]
    ds = np.array([dataset_of(p) for p in keep])
    x = np.array([values[p] for p in keep])
    y = np.array([targets[p] for p in keep])

    def pct(v):
        out = np.empty_like(v, dtype=float)
        for d in np.unique(ds):
            m = ds == d
            out[m] = rankdata(v[m]) / (m.sum() + 1)
        return out

    rx, ry = pct(x), pct(y)
    r = float(np.corrcoef(rx, ry)[0, 1])
    perms = []
    for _ in range(20000):
        shuffled = rx.copy()
        for d in np.unique(ds):
            m = np.flatnonzero(ds == d)
            shuffled[m] = rx[rng.permutation(m)]
        perms.append(np.corrcoef(shuffled, ry)[0, 1])
    p = float((np.sum(np.abs(perms) >= abs(r)) + 1) / (len(perms) + 1))
    per = {d: spearmanr(x[ds == d], y[ds == d]) for d in np.unique(ds)}
    detail = ", ".join(f"{d} rho={res.statistic:+.2f} (p={res.pvalue:.2g}, n={int((ds == d).sum())})" for d, res in per.items())
    print(f"  {label:42s} within-dataset r={r:+.3f} p={p:.4f} n={len(keep)}  [{detail}]")
    return r, p


r_primary, p_primary = within(smr_first, zero_full, "first 60 s vs zero-cal (full) [PRIMARY]")
within(smr_first, zero_120, "first 60 s vs zero-cal (120 s)")
if "all" in calibrated:
    within(smr_first, calibrated["all"], "first 60 s vs calibrated half-session")
within(smr_whole, zero_full, "whole first session vs zero-cal (full)")
q31 = r_primary >= 0.4 and p_primary < 0.01
print(f"  Q31 bar (r >= 0.4, p < 0.01): {'PASS' if q31 else 'FAIL'}")
for d in sorted({dataset_of(p) for p in people}):
    vals = np.array([smr_first[p] for p in people if dataset_of(p) == d])
    print(f"  {d}: SMR first 60 s median {np.median(vals):.2f} dB (IQR {np.percentile(vals, 25):.2f}-{np.percentile(vals, 75):.2f})")

# ------------------------------------------------------------------ Q30
print("\n=== Q30: fine-tuned vs zero calibration (same reference) vs conventional calibration")
tuned, zero_same = {}, {}
for task in ft["tasks"]:
    for n, entry in task.get("finetune", {}).items():
        for s in entry["finetuned"]:
            tuned.setdefault(n, {})[s["person"]] = s["balanced_accuracy_raw"]
        for s in entry["zero_calibration"]:
            zero_same.setdefault(n, {})[s["person"]] = s["balanced_accuracy_raw"]


def paired(a: dict, b: dict, keep):
    keep = [p for p in keep if p in a and p in b]
    x = np.array([a[p] for p in keep])
    y = np.array([b[p] for p in keep])
    p = wilcoxon(x, y).pvalue if len(keep) > 5 and np.any(x != y) else float("nan")
    return x.mean() * 100, y.mean() * 100, (x - y).mean() * 100, int((x > y).sum()), len(keep), p


verdict = {}
for d in ("Yang2025", "BNCI2014_001", "Shin2017A"):
    print(f" {d}:")
    for n in sorted(tuned, key=int):
        keep = sorted(p for p in tuned[n] if dataset_of(p) == d)
        if not keep:
            continue
        ft_mean, zs_mean, gain, better, k, p_gain = paired(tuned[n], zero_same[n], keep)
        line = f"   n={n:>2}/class: fine-tuned {ft_mean:5.1f}  zero-cal(same ref) {zs_mean:5.1f}  gain {gain:+5.1f} ({better}/{k}, p={p_gain:.2g})"
        if n in calibrated:
            a, b, diff, better2, k2, p2 = paired(tuned[n], calibrated[n], keep)
            line += f" | TS+LR same n {b:5.1f} ({diff:+.1f}, p={p2:.2g})"
        if "all" in calibrated:
            a, b, diff, better3, k3, p3 = paired(tuned[n], calibrated["all"], keep)
            line += f" | vs half-session {b:5.1f} ({diff:+.1f}, {better3}/{k3}, p={p3:.2g})"
            if n == "10":
                verdict[d] = (diff, p3)
            # 95% CI of the paired difference, bootstrapped over people (seeded).
            d_pp = np.array([tuned[n][q] - calibrated["all"][q] for q in keep if q in calibrated["all"]])
            boots = np.random.default_rng(0).choice(d_pp, (20000, d_pp.size)).mean(axis=1) * 100
            line += f" CI [{np.percentile(boots, 2.5):+.1f}, {np.percentile(boots, 97.5):+.1f}]"
        ft_vals = np.array([tuned[n][p] for p in keep])
        line += f" | >=70%: {(ft_vals >= 0.7).sum()}/{len(keep)}"
        print(line)
strict = all(verdict.get(d, (-1, 0))[0] >= 0 for d in ("Yang2025", "BNCI2014_001"))
lenient = all(
    verdict.get(d, (-1, 0))[0] >= 0 or verdict.get(d, (-1, 0))[1] >= 0.05
    for d in ("Yang2025", "BNCI2014_001")
)
print(f"\n  Q30 bar, strict (mean at or above the half-session calibration on both): {'PASS' if strict else 'FAIL'}")
print(f"  Q30 bar, lenient (or not significantly worse on both): {'PASS' if lenient else 'FAIL'}")
for d, (diff, p) in verdict.items():
    print(f"    {d}: fine-tuned(10) - half-session = {diff:+.1f} points, p = {p:.2g}")

# Calibration time actually used (from the run's report)
report = ft.get("finetune", {}).get("datasets", {})
for d, entry in report.items():
    for n in ("5", "10", "20"):
        secs = [b["calibration_s"] for person in entry["people"].values() for b in person.get(n, []) if "calibration_s" in b]
        wins = [b["n_stream_windows"] for person in entry["people"].values() for b in person.get(n, []) if "n_stream_windows" in b]
        if secs:
            print(f"  {d} n={n}: calibration time median {np.median(secs) / 60:.1f} min, stream windows median {np.median(wins):.0f}")
