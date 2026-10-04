# Riemannian baselines on BNCI2014-001 (development numbers)

**Date:** Sat 3 Oct 2026 (the Day-2 plan, done a day early). **Status:** development numbers; not for the YC application as headline numbers.
**What:** the "Riemannian tangent space + logistic regression, and MDM (with and without re-centring)" rows that architecture §8.4 puts in every results table, on the E0 dataset.

## Protocol
- **Data:** BNCI2014-001 (BCI Competition IV 2a).
  - All 9 people; none is in the locked test set (D-026).
  - Left vs right hand, both sessions: **288 windows per person** (144 per class).
- **Signal:** the shared causal chain.
  - Notch + 4–30 Hz band-pass at 250 Hz, then timestamp resampling to 128 Hz.
  - Window [cue + 0.5 s, cue + 2.5 s] (256 samples).
  - No window was flagged: no extreme amplitudes, no filter transients.
- **Features:** OAS-shrinkage covariance per window.
  - With CAR, the window is first projected onto the zero-sum subspace (21 or 17 dimensions), so the covariances stay positive definite.
- **Classifiers:**
  - **TS+LR:** tangent space at the training mean (Riemannian metric), then logistic regression with C = 1.
  - **MDM:** minimum distance to the class means (Riemannian metric).
- **Alignment (per person, label-free):**
  - **none.**
  - **EA:** our `neuros.align.ea`, γ = 1e-3.
  - **RA:** Riemannian re-centring.
  - EA and RA both use **all** of the person's windows, the "full-session" reference. That's the upper bound the online warm-up (E2) approaches.
- **Protocols:**
  - **Cross-person:** leave one person out (train on 8, test on 1).
  - **Within-person sanity check:** train on session 1, test on session 2.
- **Statistics:**
  - Balanced accuracy (chance 50%), mean ± SD over people, 95% bootstrap CI over people.
  - A paired Wilcoxon test against "none".

## Results (balanced accuracy %, mean over 9 people)

| Channels, reference | TS+LR none | TS+LR EA | TS+LR RA | MDM none | MDM EA | MDM RA | Within-person TS+LR | Within-person MDM |
|---|---|---|---|---|---|---|---|---|
| All 22, as recorded | 65.0 | 66.7 | 66.9 | 54.3 | 65.5 | 65.5 | 75.8 | 74.5 |
| All 22, CAR | 65.0 | 67.6 | 67.3 | 54.7 | 65.6 | 65.9 | 76.5 | 74.8 |
| C-core 18, as recorded | 66.3 | 67.4 | 67.3 | 55.5 | 66.0 | 66.4 | 74.7 | 73.2 |
| C-core 18, CAR | 66.4 | 68.2 | **68.8** | 56.4 | 65.9 | 66.1 | 75.3 | 73.4 |

- **Spread between people is large:** SD 11–17 points.
  - Best cross-person row (C-core, CAR, TS+LR+RA): 68.8 ± 15.8, 95% CI [58.9, 78.6], 5 of 9 people ≥ 70%.
- **Alignment vs none** (paired Wilcoxon, n = 9):
  - **MDM:** +9.5 to +11.2 points, p = 0.004–0.016 in every variant.
  - **TS+LR:** +1 to +2.6 points. Significant only for all 22 channels with CAR (EA p = 0.023, RA p = 0.043).

**Per person** (all 22 channels, as recorded):

| Person | TS+LR none | MDM none | TS+LR EA | MDM EA | TS+LR RA | MDM RA | Within TS+LR | Within MDM |
|---|---|---|---|---|---|---|---|---|
| 1 | 71.2 | 55.6 | 71.2 | 59.0 | 71.9 | 63.5 | 86.1 | 85.4 |
| 2 | 49.6 | 49.3 | 53.1 | 51.7 | 53.5 | 47.6 | 55.6 | 52.8 |
| 3 | 88.5 | 71.5 | 85.1 | 76.4 | 88.2 | 75.3 | 93.8 | 91.0 |
| 4 | 62.2 | 53.8 | 67.7 | 66.3 | 68.1 | 68.1 | 66.7 | 67.4 |
| 5 | 50.7 | 50.0 | 49.6 | 54.9 | 48.6 | 54.2 | 59.7 | 59.0 |
| 6 | 59.7 | 50.0 | 62.2 | 67.0 | 62.5 | 66.7 | 62.5 | 71.5 |
| 7 | 46.5 | 50.3 | 47.9 | 53.8 | 44.4 | 50.3 | 74.3 | 56.2 |
| 8 | 89.6 | 55.2 | 88.5 | 78.8 | 91.3 | 82.3 | 96.5 | 97.2 |
| 9 | 67.4 | 53.1 | 75.3 | 81.6 | 73.3 | 81.2 | 87.5 | 89.6 |

## What it means
1. **No sign of a pipeline bug.** Within-person accuracy is 73–77%, which is ordinary for 2-s windows on this dataset (longer windows score higher). People 2 and 5 are near chance even within themselves, the same people other papers flag; the [EA + EEGNet paper](../research/papers/2026-sagayaraju-ea-swa-calibration-bottleneck.md) reports them at ~63%.
2. **Alignment matters most for simple classifiers.**
   - MDM measures distance to class means, so a person-specific offset breaks it (54% → 66% with alignment).
   - TS+LR is already partly robust, so it gains about 2 points.
   - EA (our implementation) and RA are within 1 point of each other everywhere. That's a real-data check of `neuros.align.ea`.
3. **C-core loses nothing here.** The 18 C-core channels score the same as, or slightly better than, all 22. That supports the canonical montage (§4.3).
4. **CAR vs as recorded: a tie** on these 9 people (within ±1 point). E0b decides on development folds.
5. **The bar for E0 and our model.**
   - Best Riemannian cross-person result: **68.8%**.
   - E0's target is EA + EEGNet + SWA ≈ **74.3% ± 3** (literature, same protocol family).
   - The literature range for cross-person left/right is 77–83% ([synthesis](../research/synthesis.md)).
   - Our deep model should beat these baselines by about 5 points to be worth its complexity.
6. **4 Oct, after the arrow:** these windows include the arrow's last 0.75 s on screen. On [cue + 1.5, cue + 3.5 s], after the arrow has gone, cross-person TS+LR + EA falls from 66.9% to 60.4%, and the calibrated TS+LR from 75.8% to 72.2%. Before the cue, everything is at chance. Details in [E0](2026-10-03-e0-eegnet-bnci2014-001.md).

## Caveats
- **9 people:** CIs are ±10 points wide. These numbers guide development; they prove nothing about 380 people.
- **The alignment reference uses each test person's full session** (offline, unlabeled). The product only gets a growing stream; E2 measures how much that costs.
- **Within-person training uses 144 labeled trials:** a calibration session, which is exactly what NeurOS removes. It's a pipeline check, not a competitor.

## Regenerate
```bash
uv run neuros --config configs/data.yaml baseline --dataset BNCI2014_001 [--channels c-core] [--reference car] [--alignment-unit session] [--window START END]
```
- Each run takes about 1.5 minutes on the laptop.
- Output: `results/baselines/BNCI2014_001_left_hand-right_hand_<channels>_<reference>.json`, with per-person scores, Wilcoxon tests and provenance.
- **Reproducibility:** first computed from uncommitted Day-2 code, then re-run from the committed code (`dc59206`, `824560b`) the same evening. All four tables were **identical to the last digit**: the pipeline is deterministic.
