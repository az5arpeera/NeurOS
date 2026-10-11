# A5: does lining up each person's brain-rhythm frequency help zero-calibration decoding?

**Date:** Sun 11 Oct 2026, 22:23 UTC. **Verdict: FAIL** by the bar set before any code ([plan](../plan/a5-frequency-alignment.md)).

**Run:** `results/a5/a5-20261011T222312Z.json` (AWS, code `b06bb80`, clean; about 2 CPU-minutes).

## In plain English
- **It doesn't help.** Measuring each person's rhythm frequency from their first unlabeled minute, and setting the frequency bands relative to it, changed accuracy by **−0.24 points** on average (p = 0.41).
  - **Yang2025:** exactly unchanged (21 people better, 20 worse).
  - **BNCI2014-001:** 1.5 points worse.
- **So, by the rule set beforehand:** it isn't built into the model or the engine. The frozen recipe stays as it is, and the idea goes back to the post-YC agenda with this result attached.

## The numbers
**Protocol:**
- **People:** the development people of the two trainable cue-free datasets.
- **Windows:** D-030 windows, left vs right, motor strip, 4–30 Hz.
- **Features and model:** log band power in five bands relative to the person's peak, or to 10 Hz for everyone, z-scored per person, then logistic regression.
- **Evaluation:** leave one person out within each dataset; the same windows and folds for both conditions.

| Dataset | People | Fixed bands | Aligned bands | Gain | Better / worse |
|---|---|---|---|---|---|
| Yang2025 | 41 | 55.9% | 55.9% | +0.03 | 21 / 20 |
| BNCI2014-001 | 9 | 63.2% | 61.7% | −1.50 | 2 / 7 |
| **All** | **50** | | | **−0.24** (p = 0.41) | |

**The bar:**

| Check | Needed | Got | |
|---|---|---|---|
| Mean gain | ≥ +2.0 points | −0.24 | ❌ |
| Wilcoxon p | < 0.05 | 0.41 | ❌ |
| No dataset below 0 | ≥ 0 each | BNCI2014-001 −1.50 | ❌ |

**Peak frequencies** (first 60 s, C3/C4, unlabeled):
- **Clear peaks:** 44 of 50 people had one (≥ 1 dB above the 1/f background).
- **Raw peaks:** median 11.5 Hz, middle half 10.5–12 Hz.
- **Clipping:** 20 people's peaks were above 11.5 Hz and were clipped to it, so 27 people used 11.5 Hz.
- **Gain vs distance from 10 Hz:** unrelated (Spearman 0.13, p = 0.35).

## What it does and doesn't show
- **Doesn't show the idea is dead:**
  - **A proxy model.** It's a linear model on band power, not EEGNet. EEGNet learns its own frequency filters across 339 people and may already absorb some of this variation.
  - **Mostly clipped.** Peaks cluster at 11–12 Hz, so most people were clipped at the plan's 11.5-Hz limit. The test mostly compared "10 Hz" with "11.5 Hz" bands rather than truly individual ones.
  - **Clipping or mis-measurement?** The peaks may be genuinely high at C3/C4, or the 60-s estimate may be catching low beta. A wider clip or a better estimator would be a new experiment with its own bar; neither is run here.
- **Does show:** a simple, cheap version of frequency normalisation gives nothing measurable on cue-free people. That is consistent with D-039: small input changes don't move zero-calibration accuracy.

## Reproduce
```bash
uv run --no-sync neuros --config configs/data.yaml freq-align
```
