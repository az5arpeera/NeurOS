# How much data would zero calibration need to work for everyone? (desk projection)

**Date:** Sun 11 Oct 2026.
- **Cost:** a calculation on results we already had. It trains nothing and spent no compute.
- **Script:** [`scripts/analysis/scaling_projection.py`](../../scripts/analysis/scaling_projection.py).
- **Why no pass bar:** it isn't a test. It informs what the application says about scaling, and which levers to pull after YC.

**Question:** as we add training data, how many people would we need before zero-calibration decoding works (≥ 70%, two commands) for most people, or for nearly everyone?

## Summary in plain English
- **More data helps, slowly.**
  - Each doubling of training people adds about **half a point** of accuracy.
  - We measured this ourselves: three seeds, seven datasets, single-dataset vs pooled models.
  - It matches the literature: INCEPT's 11,000 hours of pretraining buys about +4 points over EEGNet ([synthesis](../research/synthesis.md)).
- **At that rate, scaling can't get there.**
  - Today 11% of cue-free people are at ≥ 70% after 2 minutes.
  - **25% of people:** about **1–3 billion** training people.
  - **Half of people:** more training people than are alive.
  - **Optimistic case** (a full point per doubling, twice what we measured): 25% needs ~600,000 people, 50% ~2 million, 80% ~40 million.
- **The real wall is the signal, not the data.**
  - Give each person a model calibrated on their *own* half session, as if transfer were perfect.
  - Even then only **30% of people reach 70%** (34% with 4 more points, which is what our fine-tuning added).
  - The median person's ceiling is 60%. The weakest 5% stay below 50%.
  - With this hardware and this task (imagined hand movement), no amount of training data gets past that. The projection flattens at about a third of people.
- **So "close to 100% of people with zero calibration" is not a data-scaling problem for motor imagery.** It needs:
  - a stronger signal (better electrodes, more channels);
  - a different kind of signal for the people motor imagery can't serve;
  - product design that turns a weak signal into reliable actions.
  - (See "What could get near everyone" below.)

## Step 1: how fast accuracy grows with training people
**Method:** each person was scored by a model trained only on their own dataset, and by the pooled model trained on all seven datasets (~244 people). Three seeds, each person's score averaged. Gain per doubling = gain ÷ log2(N pooled / N single).

| Dataset | Training people: own dataset → pooled | Accuracy | Points per doubling |
|---|---|---|---|
| BNCI2014-001 | 6 → 244 (5.3 doublings) | 59.9% → 68.8% | +1.67 |
| Cho2017 | 31 → 244 (3.0) | 68.8% → 69.8% | +0.35 |
| Dreyer2023 | 50 → 244 (2.3) | 64.0% → 64.9% | +0.41 |
| Lee2019_MI | 31 → 244 (3.0) | 68.6% → 70.3% | +0.58 |
| PhysionetMI | 60 → 244 (2.0) | 63.8% → 66.5% | +1.33 |
| Stieger2021 | 36 → 244 (2.8) | 70.5% → 68.4% | −0.76 |
| Yang2025 | 30 → 244 (3.0) | 57.7% → 58.8% | +0.35 |
| **People-weighted** | | | **+0.51** (+0.48 without BNCI2014-001, which starts from 6 people) |

The biggest gains come from the smallest starting sets. The rate is falling as N grows, so assuming a constant rate is generous.

## Step 2: where people stand, and their ceiling
**Who:** the 73 cue-free people (Yang2025 41, BNCI2014-001 9, Shin2017A 23).

**Two measures:**
- **Today:** zero calibration after 120 s of use (E2).
- **Ceiling:** each person's own calibrated model on half a session (the [calibration curve](2026-10-10-calibration-curve.md), TS+LR, 15–50 trials per hand).

**Noise correction:** sampling noise is removed by empirical-Bayes shrinkage within each dataset, because a score from 60 trials is noisier than one from 600.

| Dataset | Today: mean, ≥ 70% | Ceiling: mean, ≥ 70% |
|---|---|---|
| Yang2025 (41) | 56.8%, 5 people | 60.5%, 8 |
| BNCI2014-001 (9) | 66.0%, 3 | 73.8%, 6 |
| Shin2017A (23) | 57.1%, 0 after noise correction (2 observed) | 64.9%, 8 after noise correction (10 observed) |
| **All 73** | **11%** | **30%** (34% with +4 points) |

**Shin2017A's spread at 120 s is all noise.** With 60 trials per person, the observed spread (7 points) is what chance alone would give, so after correction its people sit together at 57%. That's why the projected curve jumps when they cross 70% together.

## Step 3: the projection
**Method:** lift everyone by the same amount, at log-linear gains from today's 244 training people.

| Share of people at ≥ 70% | Lift needed | At 0.48 points/doubling | At 0.51 | At an optimistic 1.0 |
|---|---|---|---|---|
| 25% | 11.3 points | 3.3 billion people | 1.2 billion | 600,000 |
| 50% | 12.9 | 38 billion* | 12 billion* | 1.9 million |
| 80% | 17.4 | 2.7 × 10¹³* | 5.5 × 10¹²* | 43 million |
| 95% | 21.1 | 6 × 10¹⁵* | 9 × 10¹⁴* | 570 million |
| 100% | 29.5 | 10²¹* | 10²⁰* | 190 billion* |

\* More training people than are alive (8.1 billion).

**The ceiling caps it.** Cap each person at their own calibrated ceiling + 4 points and the curve flattens at **~30–34% of people**, whatever the data.

**At the ceiling, the weakest person in each target share scores:**

| Share of people | Weakest person's calibrated ceiling |
|---|---|
| 25% | 71.7% |
| 50% | 60.1% |
| 80% | 54.7% |
| 95% | 49.5% |

## What the numbers don't cover
- **A different kind of model could change the slope.** For example, self-supervised pretraining on large unlabeled EEG from deployed users. Nothing published shows it yet for cross-person motor imagery (INCEPT: +4 points at 11,000 hours).
- **The ceiling is a research-grade cap with gel electrodes.** Consumer dry headsets sit lower. More own data per person would raise it a little: Yang2025 was still rising from 40 to 50 trials per hand.
- **People can learn.** Motor-imagery skill improves with feedback over sessions, which moves people, not the model. We haven't measured it.
- **The two assumptions are both generous to scaling:** a constant rate per doubling, and an equal lift for everyone.

## What could get near everyone
*Worked out, with time-per-reliable-command numbers and 12 experiments, in the [post-YC research agenda](../plan/post-yc-research-agenda.md) (parked until after 1 Nov).*

1. **A stronger signal:**
   - More channels and better contact.
   - Shin2017A's ceiling uses 30 channels; our zero-calibration model reads 3 there.
   - This is a hardware partner question.
2. **Evoked responses for people motor imagery can't serve:**
   - The brain's response to a flickering target (SSVEP, c-VEP, P300) is strong in most people.
   - Calibration-free decoding of it is an active research line (e.g. [Thielen et al. 2024, c-VEP](https://arxiv.org/abs/2403.15521)).
   - A "select by looking" mode would be the universal fallback. It's the cut list's "SSVEP demo insurance", now with a product reason.
3. **Product design:**
   - Hold when unsure, gather evidence over several windows, and confirm or undo.
   - These trade speed for reliability, so a 60% per-window signal can still drive a usable, slower interface.
4. **The honest claim:** "zero calibration for everyone whose signal supports it, and a clear answer, within minutes of use, for those it doesn't."
   - Q31's resting-rhythm check failed to predict that answer. The engine's own decisions in the first minutes would have to supply it, which is untested.

## To measure the rate properly (proposal, not run)
- **The run:** a learning curve, pooled models on 15, 30, 60, 120 and 244 training people, two seeds.
- **What it would replace:** the single-vs-pooled estimate, which mixes dataset size with dataset match.
- **Cost:** about 1–2 GPU-hours.
- **Decision:** whether data partnerships (more people) or signal and paradigm work gets the post-YC budget.
- **Before running:** it needs a written bar.

## Reproduce
```bash
uv run --no-sync python scripts/analysis/scaling_projection.py results/aws
```
**Inputs:**
- E1's three seed files (`results/aws/e1/`);
- E2 (`results/aws/e2/`);
- the calibration curve (`results/aws/calibration/`).
