# Datasets

Candidate data for training and evaluation. **Verified** means the spec was checked against the MOABB docs, PhysioNet or the dataset paper on 2026-09-28. **Verify** means it's from memory or a secondary source.

## MVP v0 data lock (Step 3, 2026-10-02)
Locked by D-011, D-012 and [architecture §4](architecture/mvp-architecture.md). Download in this order, small datasets first, so coding is never blocked:

| # | Dataset | What we use | Role | Size |
|---|---|---|---|---|
| 1 | BNCI2014_001, BNCI2014_004 | Everything | E0 sanity check and literature tables. 2b doubles as a 3-channel virtual headset | **≈ 0.75 GB + 0.45 GB** (18 .mat files each; HEAD requests, 3 Oct) |
| 2 | PhysionetMI | Imagined runs (left/right fist; both fists/feet); **baseline runs 1–2** (eyes open/closed, for H1; MOABB's `data_path` fetches them too); executed runs (later, onboarding tests) | Core; H1 | **≈ 1.9 GB** for imagery + baseline runs (872 EDF files) |
| 3 | Cho2017 | Motor imagery; real hand movement; **non-task recordings** (blinks, eye movements, head movement, jaw clenching) | Core; tuning the artifact gate | Verify |
| 4 | Lee2019_MI | Motor-imagery files only | Core | A subset of the 209 GB OpenBMI release, which covers 3 paradigms (verify) |
| 5 | Dreyer2023 | Motor imagery (EEG channels; EOG/EMG only for artifact analysis) | Core | Verify |
| 6 | EEGMAT | Rest vs mental arithmetic | E6 brain state | Small |
| 7 | Stieger2021 | **Sessions 1–2 per person first** | Core (62 people) | **377 GB** for all 598 sessions |

**Storage ask:** about 1 TB on the cluster (about 0.5 TB with the Stieger subset).

**In code (3 Oct):** every dataset above is declared in `src/neuros/data/registry.py`: channels (copied from MOABB 1.7.2's source), sampling rate, cue timing, subject ids, event names, licence, plus an `unverified` list of facts still to confirm on the files. `neuros datasets --check` cross-checks the declarations against MOABB's metadata (all agree). Downloads go through `neuros download`, which writes a SHA-256 manifest per dataset.

**What we learned from MOABB 1.7 on Day 1:**
- **It loads from a mirror by default.** `get_data` serves files from NEMAR's `sourcedata/` store (described as byte-identical originals), even when the upstream copies are already downloaded. We pin `MOABB_DOWNLOAD_PROVIDER=upstream` per process, so the files we load are exactly the files our manifest hashed.
- **No transport security.** BNCI redirects to `lampx.tugraz.at` over HTTPS, but MOABB downloads with certificate verification off. Our SHA-256 manifest is the integrity check from first download onwards.
- **Stim channel renamed** to `STI`. The harmoniser now treats any `stim`-typed channel as the event channel.
- **BNCI2014-001 artifact flags.** MOABB's default `artifact_handling="ignore"` keeps every trial unaltered (what we want). `"annotate"` would expose the original expert artifact flags (event 1023): a free benchmark for our quality gate later.
- **physionet.org is throttled** (~100 KB/s: 12 s per 1.2 MB file, ~5 h for PhysionetMI). PhysioNet's own AWS Open Data mirror (`physionet-open`) served the same file in 0.4 s. `neuros download` now fetches PhysioNet datasets from the mirror, checks **every file against `SHA256SUMS.txt` fetched from physionet.org**, and places files exactly where MOABB looks for them (`src/neuros/data/physionet.py`). Manifest entries record `"verified_against": "publisher SHA256SUMS.txt"`. EEGMAT will use the same path.

**Validation results (3 Oct, `results/validation/`):**
- **BNCI2014-001: 9/9 people, 108/108 recordings pass.** 250 Hz, 22 EEG channels, 2 sessions × 6 runs, 144 trials per class per person, no window over 500 µV.
- **BNCI2014-004: 9/9 people, 45/45 recordings pass.** 250 Hz, C3/Cz/C4, 5 sessions; 60–80 trials per class per session (not a fixed 360 per person).
- **PhysionetMI: 106/109 people pass (636/654 recordings).** All 872 files match PhysioNet's published SHA-256 sums. 160 Hz, 64 channels, 6 imagined runs per person (e.g. person 1: left 23, right 22, both hands 21, feet 24, rest 90). **People 88, 92 and 100 fail: recorded at 128 Hz** (person 100 also has fewer trials), so they're excluded (D-027). Typical amplitude (raw MAD) ~23 µV, the same as everyone else, so units weren't the problem.
- **Validation now measures artifacts in the contract band.** On raw PhysionetMI (unfiltered), 39% of windows exceeded 500 µV peak-to-peak because of line noise and drift, which the model never sees. The extreme-amplitude flag and the typical-amplitude (units) check now use the 4–30 Hz band (zero-phase, measurement only); flatness stays broadband.

**Quick-start option (Day 1–2):** the harmonized cross-hardware benchmark by J. Hassan ([Zenodo, June 2026](https://zenodo.org/records/20727886); CC-BY 4.0; 1.9 GB).
- **Contents:** PhysionetMI, Cho2017, Lee2019-MI, Shin2017A and Weibo2014.
- **Processing:**
  - resampled to 100 Hz and band-passed 8–32 Hz;
  - interpolated to a 20-channel 10-20 montage with spherical splines;
  - Euclidean-aligned;
  - cut into 3-s binary left/right windows, with a leave-one-dataset-out protocol.
- **Uses:**
  - an early cross-check of our LODO numbers;
  - **prior art to position against**: we're not the first to harmonize datasets across hardware.
- **Why it isn't our main pipeline:**
  - the data is interpolated and pre-aligned, and online-alignment experiments need raw data;
  - it isn't peer-reviewed.
- *Read depth: record page only.*

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

**Core pool: 109 + 87 + 62 + 54 + 52 = 364 subjects** (**361 usable** after D-027 excludes 3 PhysionetMI people), plus 18 more for literature comparability (BCIC-IV-2a/2b), giving **about 380 subjects** (379 usable) across 7 independent recording setups. Every paper in our reading list evaluates on 9–30 subjects, so our evaluation scale is itself a differentiator.

MOABB currently lists **63 "imagery" datasets** (motor imagery and imagined speech) and 3 resting-state datasets.

## PhysioNet datasets from the founder list

### EEG Motor Movement/Imagery Dataset (EEGMMIDB): [physionet.org/content/eegmmidb](https://www.physionet.org/content/eegmmidb/1.0.0/). Verified
- 109 subjects, 64 channels (10-10 system), **160 Hz**, EDF+.
- 14 runs: 2 baseline runs of 1 minute each (eyes open, eyes closed) and 12 task runs: real/imagined × (left/right fist) or (both fists/both feet).
- Annotations: T0 = rest, T1/T2 = the two classes; their meaning depends on the run number.
- **Free resting-state data** (the baseline runs) for EA warm-up studies and RUNet-style contrastive pretraining.
- Licence: **Open Data Commons Attribution v1.0** (commercial use OK with attribution).
- Caveats: channel indices in the files start at 0 while the montage figure uses 1–64. **Verified 3 Oct: people 88, 92 and 100 are recorded at 128 Hz** (and 100 has fewer trials); excluded by D-027. MOABB doesn't exclude them itself.

### EEG During Mental Arithmetic Tasks (EEGMAT): [physionet.org/content/eegmat](https://www.physionet.org/content/eegmat/1.0.0/). Partly verified (re-checked 2026-10-02)
- **36 subjects**: 24 "good counters" (group G) and 12 "bad counters" (group B).
- **23-channel** recording system, 10-20 placement, referenced to **interconnected ear electrodes**. The scalp channel names (probably 19, using the old T3/T4/T5/T6 names) and the sampling rate (**500 Hz, verify**) still need checking on the files.
- Per subject: `_1` = background (rest) and `_2` = mental arithmetic, each **60 s** of artifact-free EEG.
- **Caveats:**
  - The page reports "a high-pass filter with a 30 Hz cut-off". That's most likely a 30 Hz *low-pass*; we'll check on the spectra.
  - Artifacts were removed with ICA (**verify**), so the data is cleaner than any live stream.
  - The page doesn't say whether rest was eyes open or closed.
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

*The CLAP candidates above are on hold with the music-CLAP track until after 1 Nov (D-018).*

## Executed movement, artifacts and mobile EEG (mostly post-MVP)
Context: [beyond LEFT/RIGHT](explainers/beyond-left-right-any-action.md).
- **PhysionetMI executed runs** (real fist/feet movements). Already downloaded with the core pool.
- **Cho2017 non-task recordings** (blinks, eye movements, head movement, jaw clenching). Used in the MVP to tune the artifact gate.
- **Jeong et al. 2020** (*GigaScience*, doi:10.1093/gigascience/giaa098):
  - 25 people; 60 EEG + 7 EMG + 4 EOG channels; multiple sessions;
  - 11 upper-limb tasks: reaching in 6 directions, grasping 3 objects, wrist twisting with 2 motions;
  - post-MVP.
- **Lee et al. 2021 mobile BCI dataset**: scalp- and ear-EEG during standing, walking and running, with ERP and SSVEP paradigms (arXiv 2112.04176). Robustness testing, post-MVP.

## Large unlabelled corpora (Phase 2 pretraining)
- **TUEG (Temple University Hospital EEG Corpus):** about 11,000 usable hours, 14,987 subjects, 19 channels, clinical (per INCEPT). Access requires registration / a data use agreement. **Check commercial terms.**

## Open data tasks
- [ ] Licence check for each core dataset: is commercial training allowed? Deferred by the founder (Q7b). Every model bundle records its training datasets, so any dataset can be dropped later.
- [ ] EEGMAT: channel names and sampling rate from the files (the 23-channel system is confirmed); check the 30 Hz filter on the spectra.
- [ ] Record actual download sizes for every row marked "verify" in the data lock. (BNCI2014-001/004 and PhysionetMI measured 3 Oct.)
- [x] Compute the canonical montage (C-core) in code on Day 1: **26 channels** ([architecture §4.3](architecture/mvp-architecture.md)).
- [x] Stieger2021's "down" target: MOABB labels it `rest` (event id 4), and "up" `both_hand` (3).
- [ ] Confirm the registry's `unverified` fields on the files (Dreyer2023 cue offset; pre-cue fixation for Cho2017 / Lee2019 / Dreyer2023; EEGMAT channels and rate).
- [ ] PhysioNet publishes `SHA256SUMS.txt` for EEGMMIDB and EEGMAT: cross-check our manifests against it.
- [ ] THINGS-EEG2, OpenMIIR and MAD-EEG specs and licences: deferred with the music-CLAP track (D-018).
