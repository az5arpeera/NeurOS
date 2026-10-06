# E1: one pooled model vs one model per dataset, unseen datasets, and the cue-only control (development numbers)

**Date:** Tue 6 Oct 2026, early hours (AWS g6.2xlarge, one NVIDIA L4).
**Question (architecture §8.5, E1):**
- Does one model trained on every dataset beat one model per dataset, person by person?
- How well does it do on a dataset it has never seen (leave-one-dataset-out, LODO)?
- How much of each number could the cue alone explain (D-030's cue-only control)?

**Status:**
- Development numbers on development people only; the locked test set never loaded.
- One seed (20261005). The model is EEGNet + EA, which E0 checked against the literature.

**Summary:**
- **Pooling helps a little overall, a lot on the smallest dataset, and costs the largest one.**
  - +0.9 points over single-dataset models across 339 people (66.4% vs 65.5%, p = 0.014).
  - BNCI2014-001 (9 people): **+8.7**, every person better. PhysionetMI +2.4 and Cho2017 +1.4 (both p < 0.05).
  - Stieger2021: **−2.7** (p < 0.001; better for only 16 of 50 people).
- **A dataset the model has never seen costs nothing.** On all seven training datasets, the LODO model scores within 1 point of the pooled model (overall 66.3% vs 66.4%, p = 0.3).
  - On BNCI2014-001, a model that never saw that dataset scores 68.4%: **7.8 points above** a model trained on its own people (all 9 better).
- **The cue-only control splits the cued datasets in two.**
  - The task window clearly beats the cue alone on Cho2017 (+20.5), Lee2019_MI (+6.9), GuttmannFlury2025 (+10.2) and BNCI2014-004 (+13.1).
  - It doesn't on PhysionetMI (+2.0, p = 0.43) or Stieger2021 (+1.5, p = 0.27). On Dreyer2023 the cue alone decodes **better** than the task window (74.4% vs 64.8%).
  - Those three datasets' numbers can't be read as motor imagery.
- **The cue-free numbers** (the only ones free of the class display):
  - BNCI2014-001: **69.4%** pooled at zero calibration, up from 60.7% single. For scale, a model calibrated on the person's own first session scored 73.4–74.8% on the same window and band (E0b, TS+LR).
  - Yang2025 (41 people): **58.8%**. Pooling doesn't help (single 59.1%).
- **Never trained on:** GuttmannFlury2025 62.3% (25 people) and BNCI2014-004 **61.9% from 3 channels** (C3, Cz, C4), with their cue-only controls at chance (52.1%, 48.8%).

## Protocol
- `neuros e1 --experiment configs/e1.yaml` ([D-034](../decisions/decision-log.md)):
  - EEGNet + EA per recording session, on C-core's motor-strip rows (20 channels, zero-filled where a dataset lacks one; D-031);
  - common average reference, 8–30 Hz, 2-s windows at 128 Hz from D-030's per-dataset task window.
- **Training datasets (7):** PhysionetMI, Dreyer2023, Stieger2021, Lee2019_MI, Cho2017, Yang2025 and BNCI2014-001.
- **Test-only (2):** GuttmannFlury2025 and BNCI2014-004. Shin2017A is out until Q27: it shares only Cz with the motor strip.
- **5 person-level folds**, drawn within each dataset. Each person is tested once per condition:
  - **pooled:** trained on every training dataset's people outside the fold (~243 people, 55k windows); tested on everyone in the fold, test-only datasets included;
  - **single:** trained on the person's own dataset, outside the fold;
  - **LODO:** trained on every *other* training dataset's people (all folds); tested on every person of the held-out dataset;
  - **cue-only:** the pooled models again, trained and tested on [cue, cue + 0.5 s] windows.
- **Training:**
  - 10% of each model's training people per dataset are held out for early stopping (patience 15, at most 100 epochs);
  - AdamW (lr 1e-3, weight decay 0.05), label smoothing 0.1, SWA over the last 25%, batches of 16 people × 8 windows, dropout 0.25.
- **Reported:** balanced accuracy per person; means with bootstrap 95% CIs; paired Wilcoxon over the people tested under both conditions.
- **Provenance:**
  - commit `0a0cb6f` on clean code;
  - L4, deterministic kernels, cuDNN TF32 on;
  - 52 models in 43 min on 4 workers; every recording came from the cache (§4.5).

| Dataset | Cue-free | Development people | Windows | Layout channels |
|---|---|---|---|---|
| PhysionetMI | no | 84 | 3,778 | 20 of 20 |
| Dreyer2023 | no | 70 | 16,712 | 20 |
| Stieger2021 (sessions 1–2) | no | 50 | 12,133 | 20 |
| Lee2019_MI | no | 43 | 8,600 | 20 |
| Cho2017 | no | 42 | 8,520 | 20 |
| Yang2025 | **yes** | 41 | 24,592 | 19 |
| BNCI2014-001 | **yes** | 9 | 2,592 | 16 |
| GuttmannFlury2025 (test-only) | no | 25 | 1,960 | 19 |
| BNCI2014-004 (test-only) | no | 9 | 6,520 | 3 |

## Results (seed 20261005)
Balanced accuracy, % [bootstrap 95% CI]. Each difference is paired (pooled minus the other condition), with the Wilcoxon p and how many people improved.

| Dataset | Pooled | Single | LODO | Cue-only | Pooled − single | Pooled − LODO | Pooled − cue-only |
|---|---|---|---|---|---|---|---|
| PhysionetMI | 66.5 [63.1, 69.9] | 64.1 [61.3, 67.0] | 66.9 [63.5, 70.2] | 64.5 [62.6, 66.2] | **+2.4** (p = 0.004, 49/84) | −0.4 (p = 0.70) | +2.0 (p = 0.43) |
| Dreyer2023 | 64.8 [61.7, 68.0] | 64.0 [61.0, 67.2] | 64.8 [61.7, 67.9] | **74.4** [72.0, 76.7] | +0.8 (p = 0.19) | 0.0 (p = 0.93) | **−9.6** (p = 0.0001, 21/70) |
| Stieger2021 | 68.1 [64.9, 71.2] | **70.8** [67.6, 73.9] | 67.3 [64.2, 70.4] | 66.6 [64.7, 68.4] | **−2.7** (p = 0.00006, 16/50) | +0.8 (p = 0.09) | +1.5 (p = 0.27) |
| Lee2019_MI | 70.1 [66.0, 74.5] | 68.9 [65.0, 72.9] | 70.3 [66.1, 74.7] | 63.2 [61.2, 65.2] | +1.3 (p = 0.10) | −0.1 (p = 0.67) | **+6.9** (p = 0.002, 29/43) |
| Cho2017 | 70.0 [65.9, 74.1] | 68.5 [64.7, 72.6] | 69.0 [64.9, 73.4] | 49.4 [48.4, 50.4] | **+1.4** (p = 0.027, 29/42) | **+0.9** (p = 0.023) | **+20.5** (p < 10⁻⁷, 41/42) |
| **Yang2025** (cue-free) | 58.8 [55.5, 62.1] | 59.1 [56.0, 62.5] | 58.7 [55.5, 61.9] | 57.3 [55.8, 58.9] | −0.3 (p = 0.48) | +0.1 (p = 0.78) | +1.5 (p = 0.38) |
| **BNCI2014-001** (cue-free) | 69.4 [60.6, 78.7] | 60.7 [53.2, 69.0] | 68.4 [59.6, 78.1] | 71.3 [67.9, 74.2] | **+8.7** (p = 0.004, 9/9) | +0.9 (p = 0.16) | −2.0 (p = 0.73) |
| GuttmannFlury2025 | 62.3 [59.0, 65.5] | – | – | 52.1 [49.6, 55.0] | – | – | **+10.2** (p = 0.0004, 22/25) |
| BNCI2014-004 (3 channels) | 61.9 [56.3, 67.4] | – | – | 48.8 [47.6, 50.0] | – | – | **+13.1** (p = 0.008, 8/9) |

**Everyone tested under both conditions:**
- pooled vs single: 66.4 vs 65.5, **+0.9** (p = 0.014, better for 179 of 339);
- LODO vs single: 66.3 vs 65.5, +0.7 (p = 0.10, 173 of 339);
- pooled vs LODO: +0.1 (p = 0.30);
- task window vs cue-only, cued datasets: 66.8 vs 63.4, +3.4 (p = 0.0003, 197 of 323).

**LODO vs single, per dataset:** BNCI2014-001 **+7.8** (p = 0.004, 9/9), PhysionetMI **+2.8** (p = 0.005), Lee2019_MI +1.4 (p = 0.044), Stieger2021 **−3.4** (p = 0.00004, 11/50); the rest within ±0.8 (not significant).

**People at or above 70% (pooled):** 15% of Yang2025; 20% of GuttmannFlury2025; 33–38% of Dreyer2023, BNCI2014-004 and PhysionetMI; 44–48% of Stieger2021, BNCI2014-001, Lee2019_MI and Cho2017.

## What it means
1. **More people beat same-dataset people, up to a point.**
   - The smallest dataset gains most. BNCI2014-001's single models train on 6 people; pooled ones on ~243, and LODO ones on ~298 from other datasets. Both lift it by about 8 points.
   - Stieger2021 is the exception: its own 36 training people beat pooling (−2.7), and leaving it out costs 3.4 points against single. Its task differs (cursor control with continuous feedback), so its own data teaches something the others don't.
   - The overall gain (+0.9) is small. Pooling isn't yet the big win the architecture assumes: it mostly helps where data is scarce.
2. **Unseen datasets are no harder than seen ones, here.** LODO matches pooled everywhere; Cho2017's −0.9 is the one significant cost (p = 0.023).
   - That's the kind of evidence the hardware-agnostic claim needs: a new lab, amplifier and montage cost nothing measurable at zero calibration.
   - Caveat: a LODO model trains on all of the other datasets' people, more than a pooled fold model sees.
3. **The cue-only control is doing its job.**
   - **Dreyer2023's cue alone decodes at 74.4%**, nearly 10 points above its task window. Its arrow and feedback bar evoke a strong response that looks alike in everyone. Its task-window number (64.8%) is not evidence of motor imagery.
   - **PhysionetMI and Stieger2021:** the task window doesn't beat the cue alone, so their numbers can't be read as motor imagery either.
   - **Cho2017, Lee2019_MI, GuttmannFlury2025 and BNCI2014-004:** the task window beats the cue alone by 7–21 points, and Cho2017's and BNCI2014-004's cues alone are at chance. Their task-window numbers aren't explained by the cue's onset. They could still carry a class that *stays* on screen: the control tests [cue, cue + 0.5 s] only.
4. **The honest motor-imagery numbers are the cue-free ones: 69.4% (BNCI2014-001, 9 people) and 58.8% (Yang2025, 41 people).**
   - On BNCI2014-001, pooling closes most of the calibration gap on the same window and band: 60.7% → 69.4%, against 73.4–74.8% calibrated (E0b: within-person TS+LR trained on the person's first session). In E0 (4–30 Hz) the gap was 11.6 points; here it's 4–5.
   - Yang2025 stays low with every model. It has no calibrated baseline yet, so we can't say how much of that is the dataset.
5. **Three channels transfer.** BNCI2014-004 (C3, Cz, C4; never trained on) scores 61.9%, with its cue alone at chance: a first look at E5's virtual headsets.

## Caveats
- **One seed**, development people only. Seeds are cheap now (43 min per run): run more before relying on any difference under ~2 points.
- **Cue-only tests the cue's onset**, not a class that stays on screen. A chance-level control rules out the onset response, not every on-screen effect. Only the cue-free datasets avoid the display altogether.
- **The cue-free evidence is 50 people from two datasets**, 9 of them in BNCI2014-001. Its +8.7 has a wide CI (pooled: [60.6, 78.7]).
- **The calibrated comparison uses different models:** EEGNet + EA here, TS+LR in E0b's calibrated row. It gives scale, not a matched comparison.
- **Shin2017A is missing** (Q27), and with it a second cue-free test set.
- cuDNN TF32 was on (recorded in the result file). It adds about 1e-4 to logits.

## How the run went
- The first two attempts stopped:
  - **Data preparation:** Yang2025's files read 1000× too small ([D-036](../decisions/decision-log.md)). All 153 recordings pass after the fix.
  - **Assembly:** Shin2017A shares only Cz with the motor strip (Q27). `neuros e1` now refuses such a dataset before loading anything.
- The third attempt reused every prepared recording from the cache: 4 minutes of loading, 43 minutes in all.
- Result file: `results/e1/e1-20261006T024559Z.json` on the AWS data volume, copied to the laptop under `results/aws/e1/`.

## Regenerate
```bash
uv run --no-sync neuros --config configs/data.yaml e1 --experiment configs/e1.yaml --device cuda --workers 4 --threads 2
```

## Next
- E0b's EEGNet check on PhysionetMI at 8–30 Hz is running. It decides the contract's band (D-034).
- **Founders:** Q27 (Shin2017A's montage), which unblocks its cue-free test numbers.
- A calibrated within-person baseline on Yang2025, to size its calibration gap the way E0 did for BNCI2014-001.
- More seeds of E1, and cue audits of the new datasets (`neuros audit`).
- The §8.6 headline (X ≈ 75%) needs revisiting with these numbers before anything goes into the application.
