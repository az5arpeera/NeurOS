"""The guided calibration's replay test against its pass bar (plan, 10 Oct; set before running).

Usage: uv run --no-sync python scripts/analysis/calibration_bar.py <folder of simulate --calibrate JSONs>

Pass (both): the engine's leaning accuracy (balanced argmax of its smoothed probabilities on every
decoded task decision), quick minus zero calibration, mean over people >= +2 points with paired
Wilcoxon p < 0.05; and at least 30 of 34 calibrations complete. Committed accuracy and commit
rate are reported, not part of the bar.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(sys.argv[1]).glob("*.json"))]
print(f"replays: {len(rows)} (expected 34)")
done = [r for r in rows if r["calibration"]["ok"]]
print(f"calibrations complete: {len(done)} of {len(rows)} (bar >= 30)")
for r in rows:
    if not r["calibration"]["ok"]:
        print(f"  not complete: {r['person']}: {r['calibration']['message']}")



def table(subset, label):
    if not subset:
        return None
    q = np.array([r["score"]["balanced_argmax_accuracy"] for r in subset], dtype=float)
    z = np.array([r["zero_calibration"]["balanced_argmax_accuracy"] for r in subset], dtype=float)
    keep = np.isfinite(q) & np.isfinite(z)
    diff = (q - z)[keep] * 100
    p = wilcoxon(q[keep], z[keep]).pvalue if keep.sum() > 5 and np.any(diff != 0) else float("nan")
    qc = np.array([r["score"]["balanced_task_accuracy"] for r in subset], dtype=float)
    zc = np.array([r["zero_calibration"]["balanced_task_accuracy"] for r in subset], dtype=float)
    qr = sum(r["score"]["task_commands"] for r in subset) / max(1, sum(r["score"]["task_decisions"] for r in subset))
    zr = sum(r["zero_calibration"]["task_commands"] for r in subset) / max(1, sum(r["zero_calibration"]["task_decisions"] for r in subset))
    print(f"\n{label} ({keep.sum()} people)")
    print(f"  leaning accuracy: zero {np.nanmean(z) * 100:.1f}% -> quick {np.nanmean(q) * 100:.1f}%  "
          f"gain {diff.mean():+.1f} points ({(diff > 0).sum()}/{diff.size} better, p = {p:.2g})")
    print(f"  committed accuracy: zero {np.nanmean(zc) * 100:.1f}% -> quick {np.nanmean(qc) * 100:.1f}%; "
          f"commit rate {zr * 100:.0f}% -> {qr * 100:.0f}%")
    return diff.mean(), p


gain, p = table(done, "all completed calibrations")
for d in ("BNCI2014_004", "GuttmannFlury2025"):
    table([r for r in done if r["person"].startswith(d)], d)
fits = [r["calibration"]["fit_on_trials"] for r in done]
print(f"\nfit on the calibration trials (sanity, not accuracy): median {np.median(fits):.2f}")
ok = (gain >= 2.0 and p < 0.05, len(done) >= 30)
print(f"VERDICT: {'PASS' if all(ok) else 'FAIL'} (gain >= +2 with p < 0.05: {ok[0]}; >= 30 complete: {ok[1]})")
