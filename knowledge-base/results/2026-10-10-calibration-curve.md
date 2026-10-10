# The calibration curve: how much calibration is our zero-calibration model worth? (development numbers)

**Date:** Sat 10 Oct 2026 (AWS, CPU only; about 1 minute of compute).
**Question (Q28's lever 2; Q30):**
- A standard calibrated decoder is trained on a person's own labelled windows. How accurate is it after 5, 10, 20 or 40 trials per hand, and after half a session?
- How many trials of that calibration is our zero-calibration model worth?

**Status:**
- Development people only; the locked test set never loaded.
- The calibrated decoder is deterministic. Zero-calibration scores are E2's pooled models at seed 20261005.

**Summary:**
- **After 2 minutes of unlabeled warm-up, our model is worth about 5–20 trials per hand of conventional calibration.** That is roughly 1.5–5 minutes of labelled calibration at the usual 8 s per trial.
  - **Yang2025 (41 people):** 56.8%, between 10 and 20 trials per hand (55.5% and 57.7%; both differences n.s.). It beats 5 per hand (+2.9 points, p = 0.004).
  - **BNCI2014-001 (9 people):** 66.0%, between 5 and 10 per hand (63.3% and 68.4%, n.s.).
  - **Shin2017A (23 people):** 57.1%, below 5 per hand (59.3%, n.s.). The calibrated decoder uses all 30 of Shin2017A's channels; our model uses 3.
- **With a whole session of unlabeled data** (full-session alignment), it is worth about 10–40 trials per hand: 58.8% vs 57.7–60.1%, 69.4% vs 68.4–69.5%, and 62.3% vs 61.3% at 10 per hand.
- **A full conventional calibration still wins.**
  - The calibration used half a session: 50 trials per hand on Yang2025, 36 on BNCI2014-001 and 15 on Shin2017A; about 10–15 minutes at 8 s per trial.
  - **Against our model after 2 minutes**, it is 3.7 to 7.9 points ahead, significant on all three: Yang2025 60.5% vs 56.8% (p = 0.006), BNCI2014-001 73.8% vs 66.0% (p = 0.04), Shin2017A 64.9% vs 57.1% (p = 0.007).
  - **Against our full-session alignment**, it is 1.7 to 4.5 points ahead, n.s. on all three.
- **Even full calibration doesn't make most people usable on these datasets.**

  | Dataset | Mean with full calibration | People ≥ 70% |
  |---|---|---|
  | Yang2025 | 60.5% | 8 of 41 (20%) |
  | BNCI2014-001 | 73.8% | 6 of 9 |
  | Shin2017A | 64.9% | 10 of 23 |

  The ceiling is set by the people and the signal, not by skipping calibration.
- **What we can honestly claim:** NeurOS starts above chance at second zero. After 2 minutes of normal use it matches a few minutes of conventional calibration. Given a session, it comes within a few points of a full 10–15-minute calibration.

## Protocol
- **Run:** `neuros calibration --dataset X` (commit `bdfeb1e`; code `9aa8b87`, `neuros.eval.calibration`). Same windows as E1 (D-030's cue-free epoch), band 8–30 Hz, common average.
  - Yang2025 and BNCI2014-001 use E1's motor strip.
  - Shin2017A uses all 30 of its channels. It has no motor-strip C3/C4 (E1 derives them), and a calibrated decoder would simply use what the headset has.
- **Decoder:** the baselines' Riemannian tangent space + logistic regression (OAS covariances, no alignment), trained per person. It's the field's standard strong calibrated method.
- **Calibration and test split:**
  - **Blocks:** each recording session for Yang2025 and BNCI2014-001. For Shin2017A, each person, with sessions concatenated in order, since a session has only 10 trials per hand.
  - **Pool and test set:** each hand's trials are put in time order. The first half is the calibration pool and the second half the test set. The test set is the same for every size, so the curve is paired person by person.
  - **Calibration:** the first n trials per hand of the pool (what a calibration at the start would collect), or the whole pool ("all").
- **Minutes:** calibration minutes = 2n trials × the block's median time between consecutive left/right trials.
  - The medians are 8.0 s (Yang2025), 15.5 s (BNCI2014-001, whose runs interleave four classes) and 28.5 s (Shin2017A, with long rests).
  - Comparisons below are in **trials per hand**, converted at a standard 8 s per trial, so datasets with slower protocols don't look costlier.
- **Zero-calibration scores (E2):** the population prior alone, 120 s of streamed unlabeled EEG, and full-session alignment.
- **Caveat:** the two sides are scored on different windows of the same people. Calibration is scored on the second half of each block. E2 scores all task windows, or for warm-up those after 120 s. Comparisons are person by person but not window by window.

## The curves

Mean balanced accuracy (%); people ≥ 70% in brackets.

| Calibration | Yang2025 (41) | BNCI2014-001 (9) | Shin2017A (23, 30 ch calibrated vs our 3) |
|---|---|---|---|
| 5 trials per hand (~1.3 min) | 53.9 (2) | 63.3 (2) | 59.3 (3) |
| 10 per hand (~2.7 min) | 55.5 (3) | 68.4 (4) | 61.3 (6) |
| 20 per hand (~5.3 min) | 57.7 (6) | 69.5 (4) | – |
| 40 per hand (~10.7 min) | 60.1 (8) | – | – |
| Half a session | 60.5 (8), 50/hand | 73.8 (6), 36/hand | 64.9 (10), 15/hand |
| **Ours, prior only (0 s)** | 54.4 (3) | 60.8 (3) | 51.2 (0) |
| **Ours, 120 s warm-up** | **56.8** (5) | **66.0** (3) | **57.1** (2) |
| **Ours, full-session alignment** | 58.8 (6) | 69.4 (4) | 62.3 (5) |

Ours minus calibrated, person by person (p, paired Wilcoxon):

| Ours | vs 5/hand | vs 10/hand | vs 20/hand | vs half a session |
|---|---|---|---|---|
| Yang2025, 120 s | +2.9 (0.004) | +1.3 (0.10) | −0.8 (0.62) | −3.7 (0.006) |
| Yang2025, full | +4.9 (< 0.001) | +3.2 (0.003) | +1.1 (0.11) | −1.7 (0.10) |
| BNCI2014-001, 120 s | +2.6 (0.73) | −2.4 (0.91) | −3.5 (0.36) | −7.8 (0.04) |
| BNCI2014-001, full | +6.0 (0.13) | +0.9 (0.57) | −0.2 (0.82) | −4.5 (0.20) |
| Shin2017A, 120 s | −2.2 (0.52) | −4.2 (0.11) | – | −7.9 (0.007) |
| Shin2017A, full | +3.0 (0.19) | +1.0 (0.84) | – | −2.6 (0.31) |

## What this means
- **The value of NeurOS today is time, not a higher ceiling.**
  - Conventional calibration needs labelled trials before it is any good: 5 per hand scores about chance + 4 to + 13 points.
  - Ours starts above chance with no labels, and after 2 minutes of normal use it matches about 5–20 labelled trials per hand.
  - A full calibration session is still 4–8 points better than our 2-minute number.
- **The low ceilings are the bigger finding.**
  - On Yang2025, 13 minutes of calibration reach only 60.5%, and 1 person in 5 reaches 70%. Calibration can't fix a person whose imagery barely shows on the scalp.
  - This explains E2's "low level, not low speed" result: the level is low for calibrated systems too.
- **The next experiment that could change the headline:** our model plus a short calibration (fine-tuning on 5–10 trials per hand, about 1–3 minutes).
  - **If** it reaches the half-session calibration level, the claim becomes "the accuracy of a 15-minute calibration in 3 minutes".
  - Needs about 1–2 GPU-hours; pass bar set in [Q30](../open-questions.md).

## Regenerate
```bash
uv run --no-sync neuros --config configs/data.yaml calibration --dataset Yang2025 --channels motor --reference car --band 8 30 --block session
uv run --no-sync neuros --config configs/data.yaml calibration --dataset BNCI2014_001 --channels motor --reference car --band 8 30 --block session
uv run --no-sync neuros --config configs/data.yaml calibration --dataset Shin2017A --channels all --reference car --band 8 30 --block person --sizes 5 10
```
