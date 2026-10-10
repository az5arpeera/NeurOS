# D-039: training crops and extra classes ("use more of what we have") (development numbers)

**Date:** Sat 10 Oct 2026 (AWS g6.2xlarge, one NVIDIA L4).
**Question (Q28's lever 1):**
- Does training on more of the data we already have raise the zero-calibration ceiling? Two kinds of extra data:
  - **crops:** more windows from later in each trial;
  - **extra classes:** the datasets' other classes (feet, tongue, both hands, rest).
- Above all, does it lift the cue-free numbers after 120 s of warm-up, the input to D-031's gate?

**Status:**
- Development numbers on development people only; the locked test set never loaded.
- One seed (20261005), with E1/E2's folds. Validation and test windows are exactly E2's; only the training windows differ.

**Summary:**
- **Neither lever moves the ceiling.**
  - **Overall:** each variant scores +0.3 to +0.4 points over E2 (66.1–66.2% vs 65.8%, 203–209 of 396 people better, p = 0.01–0.03).
  - **Seed noise:** E1's dataset means moved 0.2–2.0 points between seeds, more than this gain.
- **The cue-free numbers at 120 s barely change:**
  - **Yang2025:** 56.8% in E2 and in all three variants.
  - **BNCI2014-001:** 65.5–66.0% vs 66.0%.
  - **Shin2017A** (never trained on): 58.5–59.9% vs 57.1% (+1.4 to +2.8, none significant).
  - **People at 70% or better after 120 s:** 12–13 of 73, against E2's 10.
- **Q28's bar isn't met.** The bar was about 65% at 120 s on Yang2025 and Shin2017A; the best variant reaches 56.8% and 59.9%.
- **Four times the Yang2025 training windows doesn't help Yang2025:** +0.4 at full session, unchanged at 120 s. What limits cue-free accuracy isn't the number of training windows.
- **The first crop runs (9–10 Oct) were void and are discarded** (see Incident). The cause was two glitched Yang2025 crops. A new artifact rule drops newly cut windows above 50× their person's median RMS. The numbers above are from the reruns.

## Protocol
- **Runs:** `neuros e1 --experiment configs/<variant>.yaml`, pooled condition only. Each variant is E2's experiment plus a `training_windows` section (D-039):
  - **e2_crops:** `crop_stride_s: 0.5`.
    - Crops slide later from the test window, never earlier, and must end inside the task period.
    - Only two datasets have room: **Yang2025** gets 4 windows per trial (starting 1.75, 2.25, 2.75, 3.25 s after the cue) and **BNCI2014-001** gets 2 (1.5, 2 s).
    - The other datasets' task windows already end at the task's end.
  - **e2_multiclass:** `extra_labels: [feet, tongue, both_hands, rest]`.
    - **Sources:** PhysionetMI (feet, both hands, rest), Stieger2021 (both hands, rest) and BNCI2014-001 (feet, tongue).
    - Each training window's softmax runs over its own dataset's classes, with label smoothing.
    - Validation and test score left vs right only, on the first two logits.
  - **e2_crops_multiclass:** both levers together.
- **Unchanged from E2:**
  - **Alignment:** each person's training windows are aligned with the EA references of their own test windows.
  - **Epochs:** an epoch counts one window per trial, so crops vary what a step sees rather than lengthening training.
  - **Warm-up:** the sets are E2's, scored by each person's model on windows that start after 120 s of session time.
- **Training windows per model** (mean over the 5 folds):

  | Run | Windows | Trials |
  |---|---|---|
  | E2 | 55k | 55k |
  | Crops | 110k | 55k |
  | Multi-class | 73k | 73k |
  | Both | 130k | 73k |

- **Artifact rule (8605be4):** newly cut training windows are dropped if their RMS is more than 50× their person's median window RMS. It drops 2 Yang2025 windows, and none from the multi-class set.
- **Reported:** balanced accuracy per person; paired Wilcoxon against E2's models at the same seed and folds.
- **Provenance** (clean code throughout):

  | Run | Commit | Model-minutes (5 models, 3 workers) |
  |---|---|---|
  | Crops | `8605be4` | 22 |
  | Both | `8605be4` | 37 |
  | Multi-class | `6373510` | 44 |
  | E2 | `66cd9d8` | 30 |

  Multi-class ran before the artifact rule existed. The rule drops nothing from its saved training set (measured), so it stands as run.

## Pooled test windows (every dataset)

Mean balanced accuracy (%); in brackets, difference from E2 and p.

| Dataset | E2 | Crops | Multi-class | Both |
|---|---|---|---|---|
| PhysionetMI (84) | 66.5 | 67.1 (+0.6, p = 0.26) | 67.2 (+0.7, p = 0.30) | 66.3 (−0.2, p = 0.55) |
| Dreyer2023 (70) | 64.8 | 64.8 (−0.0, p = 0.84) | 65.3 (+0.5, p = 0.11) | 65.4 (+0.6, p = 0.03) |
| Stieger2021 (50) | 68.1 | 68.4 (+0.3, p = 0.34) | 68.6 (+0.6, p = 0.05) | 68.5 (+0.4, p = 0.21) |
| Lee2019_MI (43) | 70.1 | 70.7 (+0.6, p = 0.16) | 70.3 (+0.2, p = 0.73) | 70.8 (+0.6, p = 0.15) |
| Cho2017 (42) | 70.0 | 70.6 (+0.6, p = 0.16) | 70.2 (+0.2, p = 0.45) | 70.5 (+0.5, p = 0.23) |
| **Yang2025 (41, cue-free)** | **58.8** | 59.1 (+0.4, p = 0.39) | 59.4 (+0.7, p = 0.03) | 59.3 (+0.5, p = 0.19) |
| **BNCI2014-001 (9, cue-free)** | **69.4** | 69.0 (−0.3, p = 0.52) | 67.7 (−1.6, p = 0.12) | 68.9 (−0.5, p = 0.81) |
| **Shin2017A (23, cue-free, never trained on)** | **62.3** | 61.8 (−0.5, p = 0.58) | 62.3 (−0.0, p = 0.95) | 63.3 (+1.0, p = 0.37) |
| GuttmannFlury2025 (25, test-only) | 62.3 | 63.2 (+0.9, p = 0.51) | 62.2 (−0.1, p = 0.82) | 62.0 (−0.3, p = 0.53) |
| BNCI2014-004 (9, test-only) | 61.9 | 62.3 (+0.3, p = 0.82) | 62.2 (+0.3, p = 0.71) | 63.1 (+1.1, p = 0.25) |
| **All (396)** | **65.8** | **66.2** (+0.4, 203 better, p = 0.03) | **66.2** (+0.4, 209 better, p = 0.01) | **66.1** (+0.3, 208 better, p = 0.03) |

- **Every difference is under 1.1 points,** except multi-class on BNCI2014-001 (−1.6, n.s., 9 people).
- **Seed noise:** E1's three seeds moved these dataset means by 0.2 (Yang2025) to 2.0 points (BNCI2014-004) ([follow-ups](2026-10-06-e1-followups-cue-test-seeds-ceilings.md)).
- **What the p-values cover:** the small "All" p-values compare people against one fixed set of models. They don't include the variation from training a model with a different seed.

## The cue-free warm-up (E2's sets)

Mean balanced accuracy (%) on windows starting after 120 s of session time.

| Dataset and budget | E2 | Crops | Multi-class | Both |
|---|---|---|---|---|
| Yang2025, 0 s (prior) | 54.4 | 55.1 | 54.9 | 55.7 |
| **Yang2025, 120 s** | **56.8** | **56.8** | **56.8** | **56.8** |
| Yang2025, full session | 58.8 | 59.1 | 59.4 | 59.3 |
| BNCI2014-001, 0 s | 60.8 | 61.3 | 61.4 | 61.0 |
| **BNCI2014-001, 120 s** | **66.0** | **65.9** | **65.5** | **66.0** |
| BNCI2014-001, full session | 68.8 | 68.5 | 67.1 | 68.3 |
| Shin2017A, 0 s | 51.2 | 51.0 | 52.0 | 52.0 |
| **Shin2017A, 120 s** | **57.1** | **59.5** (+2.4, p = 0.11) | **59.9** (+2.8, p = 0.11) | **58.5** (+1.4, p = 0.58) |
| Shin2017A, full session | 62.5 | 61.7 | 61.7 | 63.2 |
| **People ≥ 70% at 120 s (of 73)** | **10** | **12** | **12** | **13** |
| People ≥ 70%, full session | 15 | 14 | 14 | 19 |

- **Significance:** of 45 variant × dataset × budget comparisons, one has p < 0.01: crops on Shin2017A at 30 s, +3.3. At this many tests, one or two are expected by chance.
- **Shin2017A's 120-s gain is the most consistent shift:** 13 of 23 people better in each variant. It isn't significant, and it vanishes at full session.
- **Most of the "both" run's 19 at full session is near the threshold:** of the 4 extra people, 3 are on Yang2025, where the mean moves only +0.5.

## Incident: the first crop runs were void (9–10 Oct)
- **What happened:** the first crop runs (at `6373510`) scored about 53% on every dataset, including the five whose training windows didn't change.
- **Training looked healthy:** best-epoch validation was 64–67%. Four of five models then ended at exactly 50.0%, each answering one class for everyone.
- **Cause: two Yang2025 subject-18 crop windows.**
  - They come from just after that person's test window, probably an amplifier glitch.
  - Their RMS was 7,200× and 7,650× the person's median (peak 1.8 × 10⁵ against about 1). The person's test windows are clean.
  - The two windows held 99.8% of the training set's power.
- **They broke two things:**
  - **The input scale:** 26–28 instead of 0.95.
  - **The batch-norm statistics:** the trainer recomputes them over every training window at the end, and the two windows dominated the result.
  - The one model per run that scored normally was fold 2's. Fold 2 tests subject 18, so that model never trained on them.
- **Fix (`8605be4`): `screen_amplitude` drops newly cut windows above 50× their person's median RMS,** and non-finite windows.
  - Genuine windows reach 10.6× (a Stieger2021 rest window), and the test windows 7.7×.
  - Windows that E1/E2 already trained on never go through the rule.
  - The run report counts drops per dataset and per person.
- **Kept for the record:** the void results are in `results/aws/e2_crops*/void-9oct-glitched/` and the logs in `results/aws/logs/*-9oct-glitched.log`.
- **Closed later on 10 Oct:** the same weakness existed for any training window, because a single glitch can set the scale and the batch-norm statistics. E1/E2's own windows reach at most 7.7×, so their results are unaffected.
  - Now every model's training and validation windows pass a guard (`eval.pooled.check_amplitude`) before training.
  - The guard raises on any window above 50× its person's median, and never changes data (decision log, D-039 follow-up).

## What this means
- **More of the same data doesn't raise the cue-free ceiling.**
  - Neither lever moves the cue-free numbers, whether that's four times the windows per trial, the datasets' other classes, or both.
  - This fits E1's cue test: removing the three cue-driven training sets didn't change the cue-free numbers either.
  - The limit is in how much of each person's motor imagery a zero-calibration model can read, not in the amount of training data from these datasets.
- **Keep E2's recipe as the default** (no crops, no extra classes).
  - The +0.4 is at seed-noise level.
  - Multi-class costs about 50% more training time.
  - Confirming the small gain would need three seeds per variant, which isn't worth GPU time before the gate.
- **For Q28 (D-031's gate, ≈ 16 Oct):** lever 1 is done and doesn't meet the bar.
  - Lever 2, a same-session calibrated ceiling, measures the reference for "usable"; it doesn't lift anything.
  - Per Q28's recommendation, this points to the fallback (the rigour story plus E6's state channel). The founders decide.
- **Still worth running before the gate** (cheap):
  1. a same-session calibrated ceiling on Yang2025, BNCI2014-001 and Shin2017A;
  2. a per-channel-profile prior for new hardware (Shin2017A's prior is at chance);
  3. longer budgets (180, 300 s) to find S on BNCI2014-001 and Shin2017A.

## Regenerate
```bash
uv run --no-sync neuros --config configs/data.yaml e1 --experiment configs/e2_crops_multiclass.yaml --device cuda --workers 3 --threads 2
uv run --no-sync neuros --config configs/data.yaml e1 --experiment configs/e2_crops.yaml --device cuda --workers 3 --threads 2
uv run --no-sync neuros --config configs/data.yaml e1 --experiment configs/e2_multiclass.yaml --device cuda --workers 3 --threads 2
```
