# E1 follow-ups: the cue test, three seeds, Riemannian ceilings, and EEGNet's band check (development numbers)

**Date:** runs Tue 6 Oct 2026 (AWS g6.2xlarge, one NVIDIA L4); analysed Fri 9 Oct.
**Questions:**
- **Cue test.** Do the three cue-driven training datasets (PhysionetMI, Dreyer2023, Stieger2021; [E1](2026-10-06-e1-pooled-single-lodo-cue-only.md)) help or hurt the cue-free numbers? This also decides whether Stieger2021's sessions 3–11 (299 GB) are worth downloading.
- **Seeds.** Do E1's comparisons hold at two more seeds?
- **Ceilings.** Where does zero-calibration EEGNet sit against Riemannian models on the same people, window, band and channels, including a model calibrated on the person's own earlier session?
- **E0b, EEGNet part.** Does EEGNet also prefer 8–30 Hz to 4–30 Hz ([D-034](../decisions/decision-log.md))?

**Status:** development numbers on development people only; the locked test set never loaded.

**Summary:**
- **The cue-driven datasets don't hurt the cue-free numbers; keep them in training.**
  - Without them, Yang2025 scores 59.3% vs 58.8% with them (−0.5, p = 0.29) and BNCI2014-001 66.3% vs 69.4% (+3.0 for keeping them, 6 of 9 people, p = 0.15). Neither difference is significant.
  - Keeping them helps overall (+0.7 points, 200 of 373 people, p = 0.001), mostly on Stieger2021 itself (+2.0, p < 0.001).
- **Recommendation: don't download Stieger2021's sessions 3–11 now.**
  - Stieger2021's data mainly helps Stieger2021's own people, whose task window doesn't beat the cue alone (68.4% vs 66.1%, p = 0.13).
  - Pooling still costs Stieger2021 2.1 points against its own single-dataset model.
  - 299 GB more of it would mostly raise a number we can't read as motor imagery.
- **E1's comparisons hold at three seeds, and pooling now looks slightly better than with one seed.**
  - Pooled vs single: +0.9, +1.2, +1.8 points at seeds 20261005/6/7, each p < 0.02.
  - Averaged over the seeds: **+1.3** (66.5% vs 65.2%, 200 of 339 people, p = 1e-5).
  - **Yang2025 now leans toward pooling** (+1.1, 25 of 41 people, p = 0.058). E1's single seed said pooling didn't help; that seed's single-dataset model happened to score high (59.1% vs the 3-seed mean of 57.7%).
  - **LODO stays level with pooled** everywhere (within 0.5 points) except Stieger2021 (+1.3 for pooled).
  - **One person's score moves little between seeds:** median SD 1.2 points, 90th percentile 3.5.
- **Zero-calibration EEGNet beats the within-dataset Riemannian model on every training dataset (significant on six of seven).**
  - The Riemannian model is TS+LR + EA, trained on the same dataset's other people (LOSO). Margins: +1.3 to +5.2 points (p < 0.05); Stieger2021 is level (+0.1).
  - **On the two datasets EEGNet never saw, it matches the Riemannian model trained on those datasets' own people:** GuttmannFlury2025 +0.5, BNCI2014-004 +0.8 (both not significant).
- **A model calibrated on the person's own earlier session beats zero calibration on one of three datasets.**
  - BNCI2014-001: calibrated 73.8% vs 68.8% zero-calibration.
  - Lee2019_MI and Stieger2021: calibrated scores **below** zero calibration (62.0% vs 70.3%; 62.3% vs 68.4%).
  - **Caveat:** the calibrated model is unaligned TS+LR carried across days, which is the weakest calibrated baseline. Calibrating within the same session would score higher, and we haven't run that (see Gaps).
- **EEGNet has no band preference.** 8–30 Hz minus 4–30 Hz: +1.0 points on BNCI2014-001 (4 of 9 people, p = 1.0) and +0.2 on PhysionetMI (38 of 84, p = 0.87).
  - D-034's 8–30 Hz stands, since the Riemannian models clearly prefer it and EEGNet loses nothing. D-034 has the follow-up.

## Protocol
- **E1 recipe, unchanged** ([E1 protocol](2026-10-06-e1-pooled-single-lodo-cue-only.md#protocol)): EEGNet + EA per session; motor strip (20 channels); CAR; 8–30 Hz; 2-s windows at 128 Hz; 5 person-level folds; early stopping on 10% of training people; SWA.
- **Cue test** (`configs/e1_no_cued.yaml`, pooled only, seed 20261005):
  - trained on Lee2019_MI, Cho2017, Yang2025 and BNCI2014-001;
  - PhysionetMI, Dreyer2023, Stieger2021, GuttmannFlury2025 and BNCI2014-004 tested only;
  - compared person by person with E1's pooled models at the same seed.
- **Seeds:** `neuros e1 --experiment configs/e1.yaml --seeds 20261006 20261007`. Each seed redraws the folds and the early-stopping people. Comparisons use each person's score averaged over the three seeds.
- **Ceilings:** `neuros baseline --dataset <D> --channels motor --reference car --alignment-unit session --band 8 30`, on E1's development people.
  - LOSO: train on the dataset's other people.
  - Calibrated: train on the person's first session, test on their second. Only BNCI2014-001, Lee2019_MI and Stieger2021 have two sessions.
- **E0b, EEGNet part:** `neuros loso` with `configs/e0b_eegnet_<dataset>_{4_30,8_30}.yaml`. LOSO with 1 seed (20261005); the twin files differ only in `band_hz`.
- **Reported:** balanced accuracy per person; paired Wilcoxon over people tested under both conditions; "better k/n" counts people.
- **Provenance:** all runs on clean code.
  - Cue test, seeds and ceilings: commit `2865dbb`. E1 seed 20261005: `0a0cb6f`.
  - E0b: `742c45e`; PhysionetMI's 8–30 arm ran at `0a0cb6f`. Between the two commits, the only change on PhysionetMI's path is `unit_scale`, which is 1.0 for PhysionetMI.
- **Tables:** computed from the per-person scores in the result JSONs (`results/e1/`, `results/e1_no_cued/`, `results/baselines/`, `results/e0b_eegnet_*/`).

## Cue test: with vs without the cue-driven training datasets (seed 20261005)

| Dataset | People | E1 pooled (7 training sets) | Without the 3 cue-driven sets | Difference | Better with them | p |
|---|---|---|---|---|---|---|
| Lee2019_MI | 43 | 70.1 | 69.9 | +0.2 | 25/43 | 0.54 |
| Cho2017 | 42 | 70.0 | 70.7 | −0.7 | 18/42 | 0.37 |
| **Yang2025 (cue-free)** | 41 | 58.8 | 59.3 | −0.5 | 18/41 | 0.29 |
| **BNCI2014-001 (cue-free)** | 9 | 69.4 | 66.3 | +3.0 | 6/9 | 0.15 |
| PhysionetMI | 84 | 66.5 | 65.4 | +1.1 | 44/84 | 0.13 |
| Dreyer2023 | 70 | 64.8 | 64.1 | +0.7 | 36/70 | 0.10 |
| Stieger2021 | 50 | 68.1 | 66.1 | **+2.0** | 34/50 | **< 0.001** |
| GuttmannFlury2025 (test-only) | 25 | 62.3 | 61.3 | +1.0 | 12/25 | 0.30 |
| BNCI2014-004 (test-only) | 9 | 61.9 | 60.3 | +1.6 | 7/9 | 0.20 |
| **All** | 373 | 66.0 | 65.3 | **+0.7** | 200/373 | **0.001** |

- **Unseen-dataset check:** the three removed datasets became unseen datasets. A model that saw none of them scores 0.6–1.5 points below E1's LODO model, which saw the other two (Stieger2021 −1.2, p = 0.005; PhysionetMI −1.5, p = 0.05; Dreyer2023 −0.6, p = 0.13).
  - So these cued datasets do teach something general about each other. They just don't carry over to the cue-free sets.

## Three seeds

Mean balanced accuracy (%), each person's score averaged over seeds 20261005/6/7.

| Dataset | Pooled | Single | LODO | Cue-only | Pooled by seed |
|---|---|---|---|---|---|
| Lee2019_MI | 70.3 | 68.6 | 70.2 | 62.9 | 70.1 / 70.2 / 70.6 |
| Cho2017 | 69.8 | 68.8 | 69.3 | 49.8 | 70.0 / 69.6 / 69.9 |
| **Yang2025 (cue-free)** | **58.8** | 57.7 | 58.8 | 56.8 | 58.8 / 58.9 / 58.7 |
| **BNCI2014-001 (cue-free)** | **68.8** | 59.9 | 68.6 | 70.7 | 69.4 / 68.2 / 68.7 |
| PhysionetMI | 66.5 | 63.8 | 66.4 | 63.8 | 66.5 / 66.1 / 66.8 |
| Dreyer2023 | 64.9 | 64.0 | 64.7 | 74.4 | 64.8 / 65.0 / 64.9 |
| Stieger2021 | 68.4 | 70.5 | 67.1 | 66.1 | 68.1 / 68.4 / 68.7 |
| GuttmannFlury2025 (test-only) | 62.6 | – | – | 51.2 | 62.3 / 63.4 / 62.1 |
| BNCI2014-004 (test-only) | 62.1 | – | – | 49.3 | 61.9 / 61.2 / 63.2 |

**Pooled vs single, 3-seed average:**

| Dataset | Difference | Better | p |
|---|---|---|---|
| Lee2019_MI | +1.7 | 29/43 | 0.007 |
| Cho2017 | +1.0 | 30/42 | 0.04 |
| Yang2025 | +1.1 | 25/41 | 0.058 |
| BNCI2014-001 | **+8.9** | 9/9 | 0.004 |
| PhysionetMI | +2.7 | 55/84 | < 0.001 |
| Dreyer2023 | +0.9 | 40/70 | 0.04 |
| Stieger2021 | **−2.1** | 12/50 | < 0.001 |
| **All** | **+1.3** | 200/339 | 1e-5 |

**Cue-only control, 3-seed average:**
- The task window beats the cue clearly on Cho2017 (+20.1), GuttmannFlury2025 (+11.5), BNCI2014-004 (+12.8) and Lee2019_MI (+7.4).
- It doesn't on PhysionetMI (+2.7, p = 0.20) or Stieger2021 (+2.3, p = 0.13).
- On Dreyer2023 the cue wins (−9.5, p < 0.001).
- As in E1, the cue-free datasets' cue-only scores (Yang2025 56.8%, BNCI2014-001 70.7%) don't contaminate their task windows: the class is off screen by then. They say how much a cue-locked response alone carries.

## Ceilings: Riemannian models on the same people, window, band and channels

| Dataset | E1 pooled (zero calibration, 3 seeds) | Riemannian LOSO (TS+LR + EA) | E1 − LOSO | Better | p | Calibrated on own earlier session (TS+LR) |
|---|---|---|---|---|---|---|
| Lee2019_MI | 70.3 | 68.5 | +1.8 | 26/43 | 0.008 | 62.0 |
| Cho2017 | 69.8 | 67.3 | +2.6 | 30/42 | 0.002 | – |
| **Yang2025 (cue-free)** | 58.8 | 56.7 | +2.0 | 26/41 | < 0.001 | – |
| **BNCI2014-001 (cue-free)** | 68.8 | 63.6 | **+5.2** | 9/9 | 0.004 | **73.8** |
| PhysionetMI | 66.5 | 63.0 | +3.5 | 54/84 | < 0.001 | – |
| Dreyer2023 | 64.9 | 63.6 | +1.3 | 41/70 | 0.03 | – |
| Stieger2021 | 68.4 | 68.3 | +0.1 | 23/50 | 0.92 | 62.3 |
| GuttmannFlury2025 (never trained on) | 62.6 | 62.1 | +0.5 | 11/25 | 0.96 | – |
| BNCI2014-004 (never trained on) | 62.1 | 61.3 | +0.8 | 6/9 | 0.36 | – |

- **The Riemannian LOSO model is trained on each dataset's own other people.** On GuttmannFlury2025 and BNCI2014-004, EEGNet had never seen the dataset at all, and still matches it.
- **The day-1 calibrated model is not a true ceiling.** It is unaligned and trained on a different day, so it absorbs the full session shift.
  - On BNCI2014-001 it beats zero calibration by 5 points. On Lee2019_MI and Stieger2021 it loses by 6–8 points.
  - **For a pitch this cuts both ways.** "A model from yesterday's calibration does worse than our zero-calibration model" is true on two of three datasets. But it says nothing yet about same-day calibration, the comparison a sceptic will ask for.

## E0b, EEGNet part: 4–30 Hz vs 8–30 Hz

| Dataset | People | 4–30 Hz | 8–30 Hz | 8–30 minus 4–30 | Better / worse | p |
|---|---|---|---|---|---|---|
| BNCI2014-001 (cue-free, LOSO) | 9 | 61.0 [55.7, 66.1] | 62.0 [56.8, 67.7] | +1.0 | 4 / 5 | 1.0 |
| PhysionetMI (LOSO) | 84 | 62.4 [59.5, 65.4] | 62.7 [59.9, 65.4] | +0.2 | 38 / 45 | 0.87 |

EEGNet learns its own temporal filters, and the theta band (4–8 Hz) neither helps nor hurts it. The Riemannian models, which can't reweight frequencies, gained 1.1–4.2 points from 8–30 Hz (E0b, Riemannian part).

## Regenerate
```bash
uv run --no-sync neuros --config configs/data.yaml e1 --experiment configs/e1_no_cued.yaml --device cuda --workers 4 --threads 2
```
```bash
uv run --no-sync neuros --config configs/data.yaml e1 --experiment configs/e1.yaml --seeds 20261006 20261007 --device cuda --workers 4 --threads 2
```
```bash
uv run --no-sync neuros --config configs/data.yaml baseline --dataset Yang2025 --channels motor --reference car --alignment-unit session --band 8 30 --workers 6
```
```bash
uv run --no-sync neuros --config configs/data.yaml loso --experiment configs/e0b_eegnet_physionetmi_8_30.yaml --device cuda --workers 4 --threads 2
```
The baseline runs once per dataset, and the E0b run once per config file.

## Gaps and next steps
- **Same-session calibrated ceiling:** cross-validation within each person's session (TS+LR + EA and EEGNet fine-tuned), on every dataset. This is the number to quote against "just calibrate". It isn't run yet.
- **Calibrated across days with alignment** (EA on each session): the fair version of the day-1 column.
- **Cue-free data is still the bottleneck:** 50 cue-free development people (Yang2025 41, BNCI2014-001 9), plus Shin2017A as test-only from E2 onward.
- **E2** (the warm-up curve, Shin2017A's first numbers) started 9 Oct at commit `66cd9d8`.
