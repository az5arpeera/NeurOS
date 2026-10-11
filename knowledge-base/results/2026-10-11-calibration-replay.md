# The engine's 3-minute guided calibration, replayed on 34 never-seen people

**Date:** Sun 11 Oct 2026 (00:10–00:28Z), AWS g6.2xlarge, on the CPU (17 minutes).
- **Code:**
  - the bundle `mvp-v0` was trained at commit `6a6968f`;
  - the replays ran at `8020544` with clean code;
  - the fine-tune recipe is [`configs/e2_finetune.yaml`](../../configs/e2_finetune.yaml), Q30's, unchanged.
- **Bar:** the decision and pass bar were written in the [plan](../plan/road-to-nov-1.md) before running.

**Question:** does the engine's quick calibration deliver Q30's gain end to end, so the demo and the founders' headset test can offer it?

**Status:**
- **Who:** the same 34 never-seen people as the [mvp-v0 replay](2026-10-10-mvp-v0-replay.md):
  - BNCI2014_004: 9 people, 3 channels (C3/Cz/C4);
  - GuttmannFlury2025: 25 people, 19 channels.
  - Neither dataset trained the bundle.
- **What it checks:** both datasets keep the class on screen during the task, so this checks the engine; it is not a cue-free headline.
- The locked test set never loaded.

## Summary in plain English
- **Verdict: FAIL.**
  - Quick calibration raised the engine's leaning accuracy by **+3.1 points** on average.
  - The bar asked for at least +2 points, statistically clear. This gain isn't: p = 0.11, 95% CI −0.5 to +6.6.
  - **Every calibration completed** (34 of 34).
- **The engine works as built.**
  - Every calibration ran, fine-tuned in about a second and switched models.
  - The average gain is the size Q30 measured offline: +2.2 to +6.1.
- **It just isn't reliable enough to promise a gain, and it is weakest where it matters most to us.**
  - **19-channel dataset (GuttmannFlury2025):** +4.6 points (CI +0.4 to +9.0, p = 0.059). Each person has only 20 scored trials, so each person's score moves in 5-point steps.
  - **3-channel dataset (BNCI2014_004):** −1.2 points.
    - 6 people gained a little, and 3 lost 9–13 points.
    - This is the dataset most like our Ganglion setup.
    - Offline, Q30's 3-channel dataset (Shin2017A) also gained least: +2.9, not significant.
- **After calibration the engine is more confident.**
  - It answers far more often: 31% → 57% of task decisions.
  - **19 channels:** more of its answers are right.
  - **3 channels:** fewer are right (65% → 59%).
- **What happens now** (the rule set before running): **the calibration button stays out of the demo until the gap is understood.** The code stays in; nothing is deleted.

## Verdict against the bar
| Check (bar) | Result |
|---|---|
| 1. Quick minus zero calibration, leaning accuracy, mean over people ≥ +2 points with paired Wilcoxon p < 0.05 | **+3.1 points, p = 0.11** (95% CI −0.5 to +6.6; 21 better, 3 same, 10 worse): **fails** |
| 2. At least 30 of 34 calibrations complete | **34 of 34**: passes |

**Overall: FAIL** (both checks were needed).

## Results
**Leaning accuracy** is the argmax of the engine's smoothed probabilities on every decoded task decision, balanced, mean over people. It is the primary measure, before the confidence threshold.

| Dataset | Zero calibration | Quick calibration | Gain (95% CI) | Better / same / worse | p | People ≥ 70% (zero → quick) |
|---|---|---|---|---|---|---|
| **All 34** | 59.1% | 62.1% | **+3.1** (−0.5 to +6.6) | 21 / 3 / 10 | 0.11 | 7 → 10 |
| BNCI2014_004 (9; 3 channels) | 59.8% | 58.6% | −1.2 (−6.1 to +3.1) | 6 / 0 / 3 | 0.89 | 1 → 1 |
| GuttmannFlury2025 (25; 19 channels) | 58.8% | 63.4% | +4.6 (+0.4 to +9.0) | 15 / 3 / 7 | 0.059 | 6 → 9 |

**Reported, not part of the bar:** committed commands at θ = 0.6 (balanced, mean over people) and the share of task decisions committed (pooled).

| Dataset | Committed, right (zero → quick) | Commit rate (zero → quick) |
|---|---|---|
| **All 34** | 59.2% → 65.2% | 31% → 57% |
| BNCI2014_004 | 65.3% → 58.7% | 22% → 51% |
| GuttmannFlury2025 | 57.0% → 67.6% | 49% → 66% |

- **Committed-accuracy tests are post hoc**, so they aren't verdicts: all 34 +6.0 points (p = 0.04), GuttmannFlury2025 +10.5 (p < 0.01), BNCI2014_004 −6.6 (p = 0.16).
- **Some zero-calibration figures rest on very few commands.** For example, GuttmannFlury2025 person 4 committed 2 of 20 at zero calibration and 6 of 20 after quick calibration.
- **Fit on the 20 calibration trials:** median 0.97, minimum 0.70. This is a sanity check, not an accuracy. The fine-tuned model nearly memorises its trials.
- **Latency** is unchanged: worst person p99 1.8 ms.

## Why it may fall short (hypotheses; nothing here was tested)
1. **Few channels, little to gain.**
   - Both 3-channel datasets gain least: BNCI2014_004 here (−1.2) and Shin2017A offline (+2.9, n.s.).
   - The 19- and 22-channel datasets gain more: GuttmannFlury2025 here (+4.6) and BNCI2014-001 offline (+6.1).
   - With 3 channels, 20 trials may teach the model more noise than signal.
2. **Overconfidence.**
   - 50 epochs on 20 trials fits them almost perfectly (0.97), and the model's probabilities sharpen.
   - Where the fine-tuned model is wrong, it is now wrong confidently: on BNCI2014_004 its commit rate doubled while committed accuracy fell.
3. **Small test sets.** GuttmannFlury2025 scores only 20 trials per person after calibration, so its +4.6 is noisy. BNCI2014_004 scores 100–140 per person, spread over the rest of a long session.
4. **One engine-vs-offline difference.**
   - Q30 fixed the alignment reference at the calibration's EEG.
   - The engine keeps updating its reference (a running mean) after the switch.
   - After 3+ minutes of data the running mean moves slowly, so this is probably minor.

## Protocol
**Command, per person:**
```bash
neuros --config configs/data.yaml simulate --bundle /mnt/neuros/bundles/mvp-v0 --dataset DATASET --subject N --calibrate 10 --out OUT.json
```
The defaults are the first session, as fast as possible, and θ = 0.6. The queue is `aws_queue_calibration.sh` (log: `calib-mvp-v0.log`).

**Two arms per person, on the same recording and the same engine settings (θ = 0.6):**
- **Zero calibration:** the bundle as is.
- **Quick calibration:**
  - the session's first 10 trials per hand, at the recording's own cues, are the calibration;
  - the engine captures its own aligned windows at the dataset's task epoch, fine-tunes a copy of the model with Q30's recipe and switches to it;
  - the recipe is 50 epochs, lr 5e-4 → 5e-5, all windows in one batch, no early stopping.
- **Scored:** both arms on the same trials, the ones after the calibration ended.
  - That is 10 per hand on GuttmannFlury2025, and 50–70 per hand on BNCI2014_004.
  - The decision nearest each trial's task epoch counts (within half a hop).

**Analysis:**
```bash
uv run --no-sync python scripts/analysis/calibration_bar.py results/aws/replay/mvp-v0-calibration
```
The 95% CIs are bootstrapped over people (10,000 resamples, seed 0).

**Files** (git-ignored): `results/aws/replay/mvp-v0-calibration/*.json` (per person: both arms' scores, calibration details, provenance) and `results/aws/logs/calib-mvp-v0.log`.

## What it changes
- **Demo:** zero calibration only. The quick-calibration button stays out of the YC demo (the fail rule set before running).
- **[D-040](../decisions/decision-log.md)'s offline headline is unaffected.**
  - It stands as an offline development result: "within 1–3 points of a 10–13-minute calibration".
  - Still not covered: live use and consumer hardware. This replay adds that the engine's version is **not reliably better than zero calibration with 3 channels**.
- **The founders' headset self-test:** its [bar](../plan/headset-self-test.md) is scored on the door game *after* quick calibration. Before any session, the founders decide whether that stays, or whether the zero-calibration arm becomes the scored one.
