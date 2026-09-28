# Datasets

Candidate data for training and evaluation. **Verified** means the spec was checked against the MOABB docs, PhysioNet or the dataset paper on 2026-09-28. **Verify** means it's from memory or a secondary source.

## Selection criteria (proposed)
1. The label set includes **left hand vs right hand** motor imagery (plus feet / rest where available).
2. **≥ 20 subjects** (cross-subject evaluation needs many subjects).
3. Covers **C3, Cz, C4** and neighbours (FC/CP), so it maps onto our canonical montage.
4. **Loadable through MOABB** (standard loaders plus the `CrossSubjectEvaluation` protocol, which reviewers recognise).
5. **The licence permits commercial model training.** This is unverified for most datasets and must be checked before the YC application claims "trained on N subjects".

## Motor imagery pool (MOABB, verified from the [dataset summary](https://moabb.neurotechx.com/docs/dataset_summary.html))

| Dataset | Subjects | Channels | Classes | Trials / class | Trial (s) | Rate (Hz) | Sessions | Role |
|---|---|---|---|---|---|---|---|---|
| **PhysionetMI** (= PhysioNet EEGMMIDB) | 109 | 64 | 4 | 23 | 3.0 | 160 | 1 | **Core**: largest subject count |
| **Dreyer2023** (A 60 / B 21 / C 6) | 87 | 27 EEG (+2 EMG, 3 EOG) | 2 | 20 per run (240 trials per subject) | 5.0 | 512 | 1 | **Core**: includes EMG/EOG |
| **Stieger2021** | 62 | 64 | 4 | 450 | 3.0 | 1000 | 7–11 | **Core**: by far the most trials per subject |
| **Lee2019_MI** | 54 | 62 | 2 | 100 | 4.0 | 1000 | 2 | **Core** |
| **Cho2017** | 52 | 64 | 2 | 100 | 3.0 | 512 | 1 | **Core** |
| Liu2024 | 50 | 29 | 2 | 20 | 4.0 | 500 | 1 | Optional (stroke patients; verify) |
| Shin2017A | 29 | 30 | 2 | 30 | 10.0 | 200 | 3 | Optional |
| BNCI2014_001 (BCIC-IV-2a) | 9 | 22 | 4 | 144 | 4.0 | 250 | 2 | **Literature comparability** |
| BNCI2014_004 (BCIC-IV-2b) | 9 | 3 | 2 | 360 | 4.5 | 250 | 5 | **Literature comparability** |
| Schirrmeister2017 (HGD) | 14 | 128 | 4 | 120 | 4.0 | 500 | 1 | Optional |
| Weibo2014 | 10 | 60 | 7 | 80 | 4.0 | 200 | 1 | Optional |
| Beetl2021_A | 4 | 63 | 4 | 224 | 4.0 | 500 | 1 | Optional |
| Zhou2016 | 4 | 14 | 3 | 160 | 5.0 | 250 | 3 | Optional |

**Core pool: 109 + 87 + 62 + 54 + 52 = 364 subjects**, plus 18 more for literature comparability (BCIC-IV-2a/2b), giving **about 380 subjects** across 7 independent recording setups. Every paper in our reading list evaluates on 9–30 subjects, so our evaluation scale is itself a differentiator.

MOABB currently lists **63 "imagery" datasets** (motor imagery and imagined speech) and 3 resting-state datasets.

## PhysioNet datasets from the founder list

### EEG Motor Movement/Imagery Dataset (EEGMMIDB): [physionet.org/content/eegmmidb](https://www.physionet.org/content/eegmmidb/1.0.0/). Verified
- 109 subjects, 64 channels (10-10 system), **160 Hz**, EDF+.
- 14 runs: 2 baseline runs of 1 minute each (eyes open, eyes closed) and 12 task runs: real/imagined × (left/right fist) or (both fists/both feet).
- Annotations: T0 = rest, T1/T2 = the two classes; their meaning depends on the run number.
- **Free resting-state data** (the baseline runs) for EA warm-up studies and RUNet-style contrastive pretraining.
- Licence: **Open Data Commons Attribution v1.0** (commercial use OK with attribution).
- Caveats: channel indices in the files start at 0 while the montage figure uses 1–64. A few subjects are commonly excluded for sampling or length irregularities (**verify which**, e.g. against MOABB's handling).

### EEG During Mental Arithmetic Tasks (EEGMAT): [physionet.org/content/eegmat](https://www.physionet.org/content/eegmat/1.0.0/). Partly verified
- **36 subjects**: 24 "good counters" and 12 "bad counters".
- 10-20 Ag/AgCl electrodes with linked-ear reference. The page describes a **23-channel** system (**verify** the scalp EEG channel count; likely 19 plus reference/ECG) at a sampling rate of **500 Hz (verify)**.
- Per subject: one **baseline (rest)** recording and one **serial subtraction** recording, each with **60 s** of artifact-free EEG.
- Licence: ODC Attribution v1.0.
- **Proposed role:** a *second paradigm* (rest vs mental workload, cross-subject) to show the layer isn't motor-imagery-specific. It's small, but it's a standard foundation-model benchmark ("Mental Arithmetic" in CBraMod / INCEPT), so our number is directly comparable. **Not the demo.**

## Multimodal (EEG + EMG + EOG)
- **Dreyer2023** is the only large MI dataset in the pool with EMG (2 wrist electrodes) and EOG (3 electrodes). Verified from *Scientific Data* 10:580. It's useful for artifact handling (EOG) and for detecting executed-vs-imagined leakage (EMG).
- **Caution:** most consumer headsets don't record EMG/EOG channels. A model that *depends* on them won't transfer. Proposal: use EOG only for training-time artifact handling, not as a model input, in the MVP.

## CLAP / contrastive candidates (EEG paired with stimuli)

| Dataset | Content | Size | Cross-subject possible? | Status |
|---|---|---|---|---|
| **NMED-T** (Naturalistic Music EEG – Tempo) | 10 full pop songs, passive listening | 20 subjects, 125 ch (at 150 Hz in the Qing 2026 preprocessing) | **Yes**: everyone heard the same 10 songs | Verified via Qing 2026; licence **verify** |
| **NMED-H** (Hindi pop) | 4 songs | 48 subjects, 125 ch | Yes (same 4 songs) | Verified via Qing 2026; licence **verify** |
| **THINGS-EEG2** | 16,540 training images × 4 reps; 200 test images × 80 reps (**verify**) | 10 subjects, 63–64 ch (**verify**) | Yes (NICE did LOSO) | Specs from memory: **verify** |
| **ELIPformer RSVP dataset** | 3 RSVP target-search tasks with stimulus images | 71 subjects | Yes | Verified from the paper; download location **verify** |
| OpenMIIR | Music perception + **imagination** of 12 fragments | ~10 subjects, 64 ch (**verify**) | Yes | Not yet checked: imagined music is directly on-thesis |
| MAD-EEG | Attention to one instrument in polyphonic music | ~8 subjects, 20 ch (**verify**) | Small | Not yet checked |

## Large unlabelled corpora (Phase 2 pretraining)
- **TUEG (Temple University Hospital EEG Corpus):** about 11,000 usable hours, 14,987 subjects, 19 channels, clinical (per INCEPT). Access requires registration / a data use agreement. **Check commercial terms.**

## Open data tasks
- [ ] Licence check for each core dataset (commercial training allowed?)
- [ ] Confirm the EEGMAT channel count and sampling rate
- [ ] Confirm the THINGS-EEG2, OpenMIIR and MAD-EEG specs and licences
- [ ] Decide the canonical montage: the channel intersection of the core pool around C3/Cz/C4
