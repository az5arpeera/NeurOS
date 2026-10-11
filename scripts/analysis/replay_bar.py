"""The mvp-v0 real-data replay against its pass bar (plan, 10 Oct; set before running).

Usage: uv run --no-sync python scripts/analysis/replay_bar.py <replay folder> <E2 json>

Pass (all four): 34/34 replays finish; committed commands' balanced accuracy, mean over people,
>= E2's offline accuracy for the same people minus 5 points; each class >= 30% of committed
commands and >= 20% of task-window decisions committed; latency p99 < 50 ms.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

folder, e2_path = Path(sys.argv[1]), Path(sys.argv[2])
e2 = json.loads(e2_path.read_text(encoding="utf-8"))
offline = {
    s["person"]: s["balanced_accuracy_raw"]
    for t in e2["tasks"]
    if t["condition"] == "pooled"
    for s in t["scores"]
}
rows = []
for path in sorted(folder.glob("*.json")):
    r = json.loads(path.read_text(encoding="utf-8"))
    rows.append((r["person"], r["score"]))
print(f"replays: {len(rows)} (expected 34)")
people = [p for p, _ in rows]
committed = np.array([s["balanced_task_accuracy"] for _, s in rows], dtype=float)
lean = np.array([s.get("balanced_argmax_accuracy", np.nan) for _, s in rows], dtype=float)  # after c8764e1
e2_same = np.array([offline[p] for p in people])
has = np.isfinite(committed)
print(f"people with committed task commands: {has.sum()} of {len(rows)}")
per_class = {}
for _, s in rows:
    for label, (_, n) in s["per_class"].items():
        per_class[label] = per_class.get(label, 0) + n
decisions = sum(s["task_decisions"] for _, s in rows)
commands = sum(s["task_commands"] for _, s in rows)
p99 = max(s["latency_p99_ms"] for _, s in rows)
bar = e2_same[has].mean() * 100 - 5
mean_committed = np.nanmean(committed) * 100
print(f"\n1. finished: {len(rows)}/34")
print(f"2. committed balanced accuracy, mean over people: {mean_committed:.1f}% "
      f"(bar >= {bar:.1f}% = E2 offline {e2_same[has].mean() * 100:.1f}% - 5)")
print(f"   pooled committed accuracy: {sum(c for _, s in rows for c, _n in s['per_class'].values()) / commands * 100:.1f}% of {commands}")
if np.isfinite(lean).sum() > 5:
    print(f"   leaning (argmax) balanced accuracy, mean: {np.nanmean(lean) * 100:.1f}% vs E2 offline "
          f"{e2_same.mean() * 100:.1f}% (paired p = {wilcoxon(lean[np.isfinite(lean)], e2_same[np.isfinite(lean)]).pvalue:.2g})")
shares = {k: v / max(1, sum(per_class.values())) for k, v in per_class.items()}
print(f"3. class shares of committed commands: { {k: round(v, 3) for k, v in shares.items()} } "
      f"(bar each >= 0.30); commit rate {commands / decisions * 100:.1f}% of {decisions} task decisions (bar >= 20%)")
print(f"4. latency p99, worst person: {p99:.2f} ms (bar < 50)")
for d in ("BNCI2014_004", "GuttmannFlury2025"):
    m = np.array([p.startswith(d) for p in people])
    print(f"   {d}: committed {np.nanmean(committed[m]) * 100:.1f}%, "
          f"E2 offline {e2_same[m].mean() * 100:.1f}%, commit rate "
          f"{sum(s['task_commands'] for (p, s) in rows if p.startswith(d)) / sum(s['task_decisions'] for (p, s) in rows if p.startswith(d)) * 100:.1f}%")
ok = (len(rows) == 34, mean_committed >= bar, min(shares.values()) >= 0.30 and commands / decisions >= 0.20, p99 < 50)
print(f"\nVERDICT: {'PASS' if all(ok) else 'FAIL'} {ok}")
