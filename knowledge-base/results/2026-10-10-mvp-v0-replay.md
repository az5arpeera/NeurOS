# The shipping bundle (mvp-v0) replayed live on 34 never-seen people (engine check)

**Date:** Sat 10 Oct 2026, AWS g6.2xlarge.
- The bundle was trained at commit `6a6968f`.
- The replays ran at `c8764e1`, after the scoring fix below.
- The decision and pass bar were written in the [plan](../plan/road-to-nov-1.md) before running.

**Question:** is `mvp-v0` the bundle for the live demo and the founders' headset test? Or does the live engine lose accuracy compared with the offline numbers?

**Status:**
- The people are never-seen: BNCI2014_004 (9) and GuttmannFlury2025 (25), the test-only datasets the engine reads without derived channels. Neither trained the bundle.
- Both datasets keep the class on screen during the task. So this checks the engine; it is **not** a cue-free headline number.
- The locked test set never loaded.

## Verdict: PASS (all four checks)
| Check (bar) | Result |
|---|---|
| 1. Every replay finishes | 34 of 34, no errors |
| 2. Committed commands right, balanced, mean over people (≥ E2's offline accuracy for the same people − 5 = 57.2%) | **61.4%** (E2 offline 62.2%); pooled 67.7% of 690 commands |
| 3. No stuck output: each class ≥ 30% of commands, ≥ 20% of task decisions committed | left 49%, right 51%; 32.5% committed |
| 4. Latency p99 < 50 ms per decision | worst person 1.6 ms (median about 1 ms) |

Per dataset:

| Dataset | Committed, right | E2 offline | Commit rate |
|---|---|---|---|
| BNCI2014_004 (9 people; 3 channels C3/Cz/C4) | 65.0% | 61.9% | 20% |
| GuttmannFlury2025 (25 people; 19 channels) | 60.1% | 62.3% | 46% |

**What it means:**
- The live engine reproduces the offline accuracy on real EEG from people it never saw.
- The engine gives no command when it isn't confident, and it isn't stuck on one class.
- So `mvp-v0` is the bundle for the demo and the headset test.
- When the engine does commit, it's right about 2 times in 3 (pooled). When unsure, it holds.

## Protocol
**Bundle:**
- The command was `neuros bundle --experiment configs/e2.yaml`: E2's recipe, trained on all 339 development people of the 7 training datasets, with 15% held out for early stopping.
- It stopped at its best validation epoch (9), as E2's fold models do.
- Input scale 0.953.

**Replay:**
- The command was `neuros simulate`, as fast as possible, threshold 0.6, on each person's first session.
- Each person's raw recording goes through the real engine:
  1. causal preprocessing;
  2. the quality gate;
  3. online alignment from the population prior (no calibration);
  4. the model;
  5. the policy (smoothing, threshold, debounce).
- **Scoring:** each trial's scored window is the D-030 task epoch. For these datasets it is the latest 2-s window of the task, because the class stays on screen. The decision nearest that window's end is scored (within half a hop, 62.5 ms):
  - a committed command is right or wrong;
  - a HOLD is counted, not scored.
- **Offline comparator:** E2's pooled accuracy for the same people (full-session alignment, all sessions).

**Scoring bug, found and fixed:**
- The first scoring (23:32Z) counted 0 commands on all 34 people.
- For a task epoch exactly one window long, the scorer needed a decision ending at one exact instant. Decisions come every 0.125 s at sample-quantized times, so none ever matched.
- `c8764e1` scores the nearest decision, like the offline evaluation's single window, and adds a regression test.
- The replays were rerun against the same, unchanged bar.

## Regenerate
```bash
uv run --no-sync neuros --config configs/data.yaml bundle --experiment configs/e2.yaml --out bundles/mvp-v0 --device cuda
uv run --no-sync neuros --config configs/data.yaml simulate --bundle bundles/mvp-v0 --dataset BNCI2014_004 --subject 1 --out replay.json
uv run --no-sync python scripts/analysis/replay_bar.py <replay folder> results/aws/e2/e2-20261009T224245Z.json
```
