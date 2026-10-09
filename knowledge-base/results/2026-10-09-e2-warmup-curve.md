# E2: the warm-up curve, rest vs streamed seeding, and Shin2017A's first numbers (development numbers)

**Date:** Fri 9 Oct 2026 (AWS g6.2xlarge, one NVIDIA L4).
**Question (architecture §8.5, E2; the headline's S, §8.6; the D-031 gate):**
- How fast does a new person's alignment reference converge, using only unlabeled EEG from normal use?
- Does seeding it with rest work as well as streaming everything (H1, §5)?
- What does the model score on Shin2017A, a cue-free dataset from different hardware that it never trained on, now that D-037 derives its C3/C4?

**Status:**
- Development numbers on development people only; the locked test set never loaded.
- One seed (20261005), the same models as E1.
- **Determinism check:** E2's pooled models reproduce E1's per-person scores exactly, for all 373 people the two runs share.

**Summary:**
- **The warm-up works, but slower than the §8.6 hypothesis (S ≤ 45 s).**
  - Two minutes of streamed, unlabeled EEG close **52–65%** of the gap between the population prior and full-session alignment, on all three cue-free datasets.
  - At 45 s the stream is still 2.7 (Yang2025), 4.6 (BNCI2014-001) and 8.5 (Shin2017A) points below full-session alignment.
  - **S** (the first budget within 2 points of full-session) is **120 s on Yang2025**, and beyond our longest budget (120 s) on BNCI2014-001 (−2.8 at 120 s) and Shin2017A (−5.4).
- **The headline numbers at S = 120 s** (cue-free, zero calibration, X₀ → X):
  - **Yang2025** (41 people, trained on other people of this dataset): **54.4% → 56.8%** (+2.4, p = 0.003); full session 58.8%.
  - **BNCI2014-001** (9 people): **60.8% → 66.0%** (+5.2, 6 of 9 better, p = 0.25); full session 68.8%.
  - **Shin2017A** (23 people; never trained on, different hardware, 3 derived channels): **51.2% → 57.1%** (+5.9, 17 of 23 better, p < 0.001); full session 62.5%.
- **The bigger problem is the level the warm-up converges to, not its speed.** Even with a full session's alignment, only **15 of 73** cue-free people reach 70%, the field's usual threshold for usable binary motor imagery. After 120 s, 10 of 73 do.
  - This is the input to D-031's gate (≈ 16 Oct): "if E2 shows that adaptation can't reach usable accuracy within a minute or two…". See Next.
- **Shin2017A: first evidence on cue-free data from unseen hardware.**
  - **62.3%** with E1's full-session alignment (23 people, 3 channels). Its cue-only control is at chance (49.2%), so this is motor imagery, not the cue.
  - With only the population prior it is **at chance (51.2%)**. The prior from other hardware doesn't fit Shin2017A's derived channels, and only the person's own data rescues it (+11.3 over a session).
- **H1 (rest-seeded as good as streamed) isn't supported; streaming everything is as good or better.**
  - Seeding from rest scores below streaming at the same budget: −0.2 to −2.4 points (significant on Yang2025 only).
  - On Yang2025, rest seeding has nothing to use in the first two minutes: rest windows are anchored to pre-cue fixations, and each session's first trial comes 163–234 s in.
  - The engine already streams every quality-gated window (§5), so no separate rest phase is needed.

## Protocol
- `neuros e1 --experiment configs/e2.yaml`:
  - E1's data, recipe, folds and seed (20261005), with conditions pooled and cue_only;
  - the warm-up sets are scored by each person's pooled model.
- **Reference estimate** (architecture §5): R̂(t) = (n₀·R_pop + n_t·R_user(t)) / (n₀ + n_t).
  - R_pop: the mean task-window covariance of the fold's training people, on the dataset's channel profile.
  - n₀: 10 s worth of stream windows (80).
  - R_user(t): the running mean over quality-gated 2-s windows (hop 0.125 s, peak-to-peak ≤ `ptp_flag_uv`), skipping each run's first 0.5 s.
- **Stream mode:** every window that ends by the budget, counted in session time (runs concatenated).
- **Rest mode (H1):** only windows inside a pre-cue fixation, weighted by the seconds they cover.
- **Budgets:** 0 (prior only), 5, 10, 20, 30, 45, 60 and 120 s, plus "all" (the whole session).
- **Full session** = E1's placement (the session's mean task-window covariance), the offline upper bound.
- **Scored windows:** each person's task windows that start after 120 s of session time. Every budget is scored on the same windows, so comparisons are paired.
  - Scored: Yang2025 24,592 of 24,592 (no trial in a session's first 120 s); BNCI2014-001 2,407 of 2,592; Shin2017A 1,104 of 1,380.
- **Channel profiles:** the layout channels each dataset records. Yang2025 has 19, BNCI2014-001 16, and Shin2017A 3 (C3, Cz, C4, with C3/C4 derived per D-037).
- **Reported:** balanced accuracy per person; bootstrap 95% CIs over people; paired Wilcoxon.
- **Provenance:** commit `66cd9d8` on clean code; 10 models (5 pooled, 5 cue-only) in 26 min on 4 workers.

## The warm-up curve (streamed, unlabeled EEG)

Mean balanced accuracy (%), [95% CI]; difference from the prior and from full-session alignment, with people better and p.

| Budget | Yang2025 (41) | BNCI2014-001 (9) | Shin2017A (23, never trained on) |
|---|---|---|---|
| 0 s (prior) | 54.4 [52.2, 56.9] | 60.8 [53.8, 68.9] | 51.2 [50.0, 52.7] |
| 5 s | 54.7 | 62.3 | 50.8 |
| 10 s | 55.1 | 63.1 | 51.4 |
| 20 s | 55.3 | 63.6 | 52.3 |
| 30 s | 55.7 (+1.3 vs prior, p = 0.01) | 64.3 (+3.5, p = 0.16) | 52.4 (+1.2, p = 0.22) |
| 45 s | 56.1 | 64.2 | 54.0 (+2.8, p = 0.01) |
| 60 s | 56.3 (+1.8, p = 0.004) | 64.4 (+3.5, p = 0.43) | 55.2 (+4.0, p = 0.007) |
| **120 s** | **56.8** [54.2, 59.6] (+2.4, p = 0.003) | **66.0** [57.5, 75.2] (+5.2, p = 0.25) | **57.1** [54.4, 60.0] (+5.9, p < 0.001) |
| Whole session, streamed | 58.3 | 67.8 | 62.9 |
| **Full session (E1's placement)** | **58.8** [55.6, 62.1] | **68.8** [60.0, 78.3] | **62.5** [59.1, 65.9] |
| 120 s vs full session | −1.9 (p = 0.005) | −2.8 (p = 0.04) | −5.4 (p = 0.004) |
| Share of the prior → full gap closed at 30 / 60 / 120 s | 29% / 42% / 55% | 44% / 44% / 65% | 10% / 35% / 52% |
| People ≥ 70%: prior / 120 s / full | 3 / 5 / 6 of 41 | 3 / 3 / 4 of 9 | 0 / 2 / 5 of 23 |

- **Streaming the whole session reaches full-session alignment** (−0.5, −1.0, +0.4), although it averages every window rather than task windows only. Whatever the user does, enough streamed data finds the reference.
- **Yang2025's first two minutes are its lead-in before the first trial**, and they still move the reference toward the task's (+2.4). That is the realistic case: someone puts on the headset and the engine adapts before their first command.
- **On BNCI2014-001 the 9-person curve is noisy.** The gains over the prior are consistent early (8 of 9 people better at 5 s) and not significant later. One person's score has a standard error of about 3 points at ≈ 270 scored windows.

## Rest-seeded vs streamed (H1)

Rest-seeded minus streamed, at the same budget:

| Budget | Yang2025 | BNCI2014-001 | Shin2017A |
|---|---|---|---|
| 30 s | −1.3 (p = 0.01) | −0.7 (p = 0.65) | −0.2 (p = 0.66) |
| 60 s | −1.8 (p = 0.004) | −0.9 (p = 1.0) | −2.1 (p = 0.06) |
| 120 s | −2.4 (p = 0.003) | −2.1 (p = 0.20) | −1.9 (p = 0.08) |
| Whole session | −0.8 (p = 0.02) | −1.4 (p = 0.02) | −1.8 (p = 0.16) |

- On Yang2025, rest seeding stays at the prior until 120 s: there are no cue-anchored fixations in the lead-in.
- On BNCI2014-001 and Shin2017A, rest is close to streaming early and falls behind later, never ahead.

## Cue-only control and E1's numbers (same models)
- The pooled and cue-only scores are identical to E1's at this seed, for all 373 people in both runs (Shin2017A is new).
- **Shin2017A:** pooled 62.3%, cue-only 49.2%: the cue carries nothing, and the task window carries motor imagery.

## What this means
- **The mechanism is sound.** Unlabeled data from normal use moves every dataset toward its full-session score, and on an unseen headset it is what lifts the model from chance.
- **The current numbers don't support "usable within a minute or two".** At 120 s the cue-free means are 57–66%, and only 14% of people reach 70%. The gap to usable is mostly the model's cue-free accuracy, which full-session alignment caps at 59–69%. It isn't the speed of the warm-up.
- **Levers, in order of expected value per day of work:**
  1. **Raise the ceiling** with "use more of what we have" (already approved): training crops across the imagery period, and multi-class auxiliary training.
  2. **Measure the true calibrated gap:** a same-session calibrated ceiling on Yang2025, BNCI2014-001 and Shin2017A. Without it, "usable" has no reference point.
  3. **A better prior for new hardware:** Shin2017A's population prior is at chance. A prior per channel profile, built from training people *re-derived the same way* (D-037's averaging), could start new headsets above chance.
  4. **Longer budgets** (180, 300 s) to find S on BNCI2014-001 and Shin2017A, plus a sweep of n₀ (the prior's weight).
- **For D-031's gate (≈ 16 Oct):** recorded as [Q28](../open-questions.md). The founders decide after levers 1–2 report, not on this run alone.

## Regenerate
```bash
uv run --no-sync neuros --config configs/data.yaml e1 --experiment configs/e2.yaml --device cuda --workers 4 --threads 2
```
