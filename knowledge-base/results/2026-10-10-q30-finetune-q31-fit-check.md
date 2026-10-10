# Q30 and Q31: a 3-minute calibration on top of our model, and the 1-minute fit check (development numbers)

**Date:** Sat 10 Oct 2026, AWS g6.2xlarge (L4). Commit `033bc50`, clean code (`git_dirty_code: false`). E2's config plus a `finetune` section ([`configs/e2_finetune.yaml`](../../configs/e2_finetune.yaml), sha256 `2ddb3387…`). Seed 20261005.

**Run time:** the resting-rhythm check took 30 min (mostly rebuilding Yang2025's cache on CPU). Training plus fine-tuning took 24 min on the GPU.

**Status:**
- Development people only; the locked test set never loaded.
- Both pass bars were written in [open-questions](../open-questions.md) before running.
- The fine-tune recipe (50 epochs, lr 5e-4 → 5e-5) was fixed in advance and never tuned on test people.
- **Determinism check:** the pooled models reproduced E2 exactly (396 of 396 people identical), so the comparisons below use the same models E2 reported.

## Summary in plain English
- **Q30, the 3-minute calibration:**
  - We trained our model on other people, then fine-tuned it with the new person's first **10 trials per hand** (about 3 minutes).
  - That came within 0.7 points (Yang2025) and 2.5 points (BNCI2014-001) of a **10–13-minute** conventional calibration.
  - **Neither gap is statistically significant, but both are below.** The bar passes on its lenient reading (not significantly worse) and fails on its strict reading (at or above).
  - **With 20 trials per hand (about 5 minutes), Yang2025 is above the 13-minute calibration** (+1.3 points, n.s.). BNCI2014-001 is still 2.0 points below.
  - **With the same trials, fine-tuning our model beats conventional calibration by 3–6 points.** For example, Yang2025 at 10 per hand: 59.8% vs 55.5% (p = 0.001).
- **Q31, the 1-minute fit check: fails. Drop it.**
  - The resting rhythm in a person's first minute predicts their accuracy only weakly: within-dataset r = 0.23, p = 0.056; the bar was r ≥ 0.4, p < 0.01.
  - Secondary readings reach r ≈ 0.34 (p < 0.01), which is real but far too weak to tell an individual whether NeurOS will work for them.
- **What doesn't change:** the level. Fine-tuned, Yang2025 is still about 60% and only 7 of 41 people reach 70%. **The gain is in time, not in ceiling.**

## Q30: fine-tuning on a short calibration
### Protocol
The module is `neuros.eval.finetune`, and the `finetune` section of `neuros e1`.

For each test person of Yang2025, BNCI2014-001 and Shin2017A (5 folds, the person's own fold model):
- **Split:** exactly the [calibration curve](2026-10-10-calibration-curve.md)'s split. Each hand's trials are put in time order within a block (a session; for Shin2017A, the person). The first half is the calibration pool and the second half is the test set.
- **Calibration:** the first n = 5, 10 or 20 trials per hand of the pool.
- **Reference:** our alignment uses only the EEG recorded during the calibration itself: from 4 s before the first calibration window to the end of the last one. It is blended with the population prior, as in E2. "3 minutes" means 3 minutes.
- **Fine-tuning:** the whole model, 50 epochs on those windows only. The learning rate is 5e-4, cosine down to 5e-5. There is no early stopping (no validation data exists in 3 minutes).
- **Scored:** on the same test windows, three ways:
  - our model as is (zero calibration, same reference);
  - our model fine-tuned;
  - the conventional calibrated decoder from the calibration curve (Riemannian TS+LR, trained on the same n trials, or on the whole pool).

All comparisons are paired person by person, with a Wilcoxon test. The 95% confidence intervals are bootstrapped over people.

### Results
Mean balanced accuracy (%); people ≥ 70% in brackets.

| | Yang2025 (41) | BNCI2014-001 (9) | Shin2017A (23) |
|---|---|---|---|
| Ours, zero calibration (same windows and reference, n = 10) | 57.6 | 65.3 | 62.8 |
| **Ours + 5 per hand** | 59.4 (7) | 67.4 (3) | 63.3 (8) |
| **Ours + 10 per hand** | **59.8** (7) | **71.4** (4) | **65.7** (8) |
| **Ours + 20 per hand** | **61.8** (9) | 71.8 (4) | – |
| Conventional, 5 per hand | 53.9 | 63.3 | 59.3 |
| Conventional, 10 per hand | 55.5 | 68.4 | 61.3 |
| Conventional, 20 per hand | 57.7 | 69.5 | – |
| **Conventional, half a session** | **60.5** (8), 50/hand | **73.8** (6), 36/hand | **64.9** (10), 15/hand |

**The pass bar:** ours + 10 per hand minus the half-session calibration.

| Dataset | Difference | 95% CI | Better / total | p |
|---|---|---|---|---|
| Yang2025 | −0.7 | −2.7 to +1.2 | 18 / 41 | 0.48 |
| BNCI2014-001 | −2.5 | −5.3 to +0.3 | 4 / 9 | 0.30 |
| Shin2017A (not in the bar) | +0.7 | −3.8 to +5.5 | 11 / 23 | 0.96 |

**Verdict against the bar set before running:**
- **Strict reading** (mean at or above on both): **FAIL**.
- **Lenient reading** (not significantly worse on both): **PASS**.
- Yang2025's interval rules out a gap larger than about 3 points. BNCI2014-001's 9 people can't rule out about 5.

**Other paired differences:**

| Comparison | Yang2025 | BNCI2014-001 | Shin2017A |
|---|---|---|---|
| Fine-tuning vs our own zero calibration, 10/hand | +2.2 (p = 0.009) | +6.1 (p = 0.04) | +2.9 (n.s.) |
| Fine-tuning vs conventional with the same trials, 5/hand | +5.5 (p < 0.001) | +4.1 (p = 0.07) | +4.1 (n.s.) |
| Fine-tuning vs conventional with the same trials, 10/hand | +4.2 (p = 0.001) | +2.9 (p = 0.07) | +4.3 (n.s.) |
| Fine-tuning vs conventional with the same trials, 20/hand | +4.1 (p = 0.002) | +2.3 (n.s.) | – |
| Yang2025, ours + 20/hand vs half a session | +1.3 (CI −0.8 to +3.2, n.s.) | | |

On Shin2017A, our model uses 3 channels and the conventional decoder uses all 30.

### Calibration time
At a standard 8 s per trial, as in the calibration curve:
- 10 per hand = 2.7 min;
- half a session = 13.3 min (Yang2025) and 9.6 min (BNCI2014-001).

The run also measured the real time spanned by each calibration (median):

| Dataset | 5/hand | 10/hand | 20/hand |
|---|---|---|---|
| Yang2025 | 1.6 min | **2.9 min** | 5.3 min |
| BNCI2014-001 | 1.8 min | 4.7 min | 10.6 min |
| Shin2017A | 4.4 min | 9.1 min | – |

BNCI2014-001's real times are inflated because its runs interleave four classes. Shin2017A's are inflated by long rests. In a 2-class protocol, 10 per hand is about 3 minutes.

## Q31: the 1-minute fit check
### Protocol
- **Command:** `neuros smr` (module `neuros.eval.smr`).
- **Data:** the first 60 s of each person's first session, unlabeled. 73 people.
- **Measure:** at C3 and C4, the height in dB of the 8–13 Hz peak above a 1/f line fitted at 4.5–7 and 25–28 Hz (after Blankertz et al. 2010).
- **Correlation:** each variable is ranked within its own dataset, then the ranks are pooled. The p-value comes from 20,000 permutations within each dataset.

### Results
| Resting rhythm vs | Within-dataset r | p | Yang2025 ρ (41) | BNCI2014-001 ρ (9) | Shin2017A ρ (23) |
|---|---|---|---|---|---|
| **Zero-calibration accuracy, full session (primary)** | **+0.23** | **0.056** | +0.32 | −0.02 | +0.13 |
| Zero-calibration accuracy, 120 s | +0.34 | 0.004 | +0.31 | −0.02 | +0.52 |
| Half-session calibrated accuracy | +0.34 | 0.004 | +0.46 | +0.48 | +0.06 |
| Whole first session's rhythm (not 1 min) vs primary | +0.36 | 0.003 | +0.49 | +0.07 | +0.23 |

**Verdict:** **FAIL** (bar: r ≥ 0.4 and p < 0.01).
- The relationship exists and points the right way. At r ≈ 0.3 it explains about 10% of the differences between people, too little to tell any one person "this will or won't work for you".
- **Pre-registered consequence:** drop the fit check. It doesn't go into the engine.

**A dataset-level aside (3 datasets, not evidence):** Yang2025's median resting rhythm is about half the others':

| Dataset | Median height (IQR) |
|---|---|
| Yang2025 | 3.95 dB (2.0–4.9) |
| BNCI2014-001 | 8.2 dB (4.5–11.3) |
| Shin2017A | 8.0 dB (4.3–10.2) |

This fits Yang2025's low ceiling, calibrated or not.

## What this means
- **The honest headline:**
  - "About 3 minutes of calibration on top of NeurOS comes within 1–3 points of a 10–13-minute conventional calibration."
  - "About 5 minutes matches it" (Yang2025).
  - "With the same few trials, NeurOS is 3–6 points better than starting from scratch."
- **What we can't say:** "the accuracy of a 15-minute calibration in 3 minutes". The means are slightly below on both bar datasets.
- **For the product:** a short guided calibration is worth offering as an option on top of zero calibration. The engine already has the parts:
  - a bundle with the model;
  - online alignment;
  - fine-tuning on 20 windows, which takes seconds.

  It needs a cue screen in the demo and a fine-tune step in the engine. The founders decide (Q30).
- **What it doesn't fix:** the level. About 60% on Yang2025 and 65–71% elsewhere, with 1 in 6 to 1 in 2 people at 70%. A short calibration saves time; it doesn't make more people usable. Q28's gate (≈ 16 Oct) still decides what the application leads with.

## Regenerate
```bash
uv run --no-sync neuros --config configs/data.yaml smr --datasets Yang2025 BNCI2014_001 Shin2017A --seconds 60
uv run --no-sync neuros --config configs/data.yaml e1 --experiment configs/e2_finetune.yaml --device cuda
```

Analysis (verdicts, tables and CIs):
```bash
uv run --no-sync python scripts/analysis/q30_q31.py results/aws/e2/e2-20261009T224245Z.json results/aws/e2_finetune/e2_finetune-20261010T225403Z.json results/aws/smr/smr_Yang2025-BNCI2014_001-Shin2017A_60s.json results/aws/calibration/*.json
```
