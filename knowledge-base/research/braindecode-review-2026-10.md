# Braindecode: is anything worth using in NeurOS?

**Date:** Tue 13 Oct 2026. **Method:** a research pass over the Braindecode website (the stable docs describe 1.5.1; the latest release is 1.8.1, 31 Aug 2026), its GitHub repository, its Hugging Face models and three linked papers. It was reading only: Braindecode isn't approved for installation, and nothing was installed, downloaded or run. The founders asked for it ("see if anything is worth using in the stack").

**Bottom line: adopt nothing from Braindecode before the 24 Oct freeze.** Not the library, and not its models.
- **No cross-person evidence.** The site reports no cross-person motor-imagery numbers. Every tutorial trains and tests within one person (BNCI2014-001, subject 3), and its windows start before the cue disappears.
- **Models.** The zoo has 56 architectures. Our own A6 (pretrained encoders tie EEGNet) and D-039 (more windows don't help) make a +3-point gain from any of them unlikely.
- **Preprocessing.** Filters default to zero-phase, which can't run live, and there's no Euclidean alignment.
- **Weight and licence.** The library pulls in mne ≥ 1.11, mne_bids, skorch, wfdb and torchaudio, and ships some files under non-commercial licences (CC BY-NC).

**The one useful pointer is channel-subset training.** Braindecode's `ChannelsDropout` is a reminder of something **our own architecture already specifies** but the trainer never implemented: random channel-subset training (§4.3: "Training uses random channel-subset augmentation, including Ganglion-, Unicorn-, Crown- and 2b-like subsets"). E5 found the 3-channel Ganglion class 3.4 points below the full cap, against a bar of 3. If tested, it should be written in our code: re-reference and re-align on each subset, zero-fill the rest, as the engine does. Braindecode's version just zeroes channels after the fact.

## What was checked

| Item | Fit | Why |
|---|---|---|
| **The library as a dependency** (BSD-3 core) | Skip | Heavy dependencies (mne ≥ 1.11, mne_bids, skorch, wfdb, torchaudio, h5py). Some model files are CC BY-NC or CC BY-NC-SA (MetaNeuromotorHand, EMG2QwertyNet, NeuroRVQ, SleepFM), so shipping the package in a commercial SDK ships them too. If one model is ever wanted, copy its single BSD-3 file with attribution |
| **Small motor-imagery networks** (EEGNeX 56k params, ShallowFBCSPNet 46k, CTNet 27k, ATCNet 114k, EEGSimpleConv 730k, EEGConformer 790k, IFNet, FBCNet, EEGTCNet) | After YC | No cross-person numbers on the site. Kernels are tuned for 250 Hz and longer windows, so porting means rescaling to our 128 Hz, 2-s windows. All would fit the 50 ms CPU budget |
| **EEGSym** (hemisphere-mirroring, built for person-to-person differences) | After YC, full caps only | Needs left/right channel pairs, so it can't help 3 channels |
| **Foundation models** (REVE, LaBraM, EEGPT, CBraMod, BIOT, BENDR, LUNA, CodeBrain, SignalJEPA) | Skip | No accuracy numbers on the site. SignalJEPA was pretrained on Lee2019, which we already train on: the A6 situation again. Channel and rate requirements don't fit |
| **`Interpolated*` wrappers** (spline-map any montage onto a foundation model's channels) | Skip | Experimental, with no numbers, and they contradict "no interpolation in v1". They can't create C3/C4 for the EPOC X |
| **Euclidean alignment** | Keep ours | Not in Braindecode's API. Ours has shrinkage and the online prior |
| **Exponential moving standardisation** | Skip | Redundant with EA plus RMS scaling, and its start-up block uses unspecified statistics |
| **Filtering, resampling, EEGPrep** | Skip | Zero-phase by default (not causal); EEGPrep is calibrated on whole-session statistics. Our causal, chunk-identical filters are what live use needs |
| **`ChannelsDropout`** | Adopt the idea, not the code | See the bottom line |
| **`ChannelsSymmetry`** (left/right channel swap) | After YC, with a label swap | As shipped it leaves the labels unchanged, which would mislabel left/right imagery |
| **Mixup, segment recombination, FTSurrogate** | After YC | No evidence on the site. Outside, EA plus segment recombination added about 1 point over EA alone in a pooled model (arXiv 2405.14994). D-039 makes "more data" augmentations a low prior for us |
| **FrequencyShift and other generic augmentations** | Skip | A5 already failed on FrequencyShift's mechanism (individual rhythm frequency); no cross-person evidence for the rest |
| **Cropped decoding** | Skip | The benefit claimed is speed. D-039's crops gave +0.3, and the engine's p99 is already 1.6 ms |
| **Datasets** (HGD, TUH, NMT, CHB-MIT, SIENA, Sleep-Physionet, EEGDash) | Skip; EEGDash after YC | No new imagery people. HGD (14 people) is MOABB's Schirrmeister2017; the clinical sets don't apply; TUH's and EEGDash's licences aren't stated |
| **OpenEEG-Bench** (foundation-model fine-tuning benchmark) | After YC | Split protocol not stated; no leaderboard numbers on the page |

**Outside evidence the review found:**
- **Augmentation review (Rommel et al. 2022).** No single best augmentation. Its imagery gains were measured across sessions of the same people, not across people.
- **EEGSimpleConv (arXiv 2309.07159).** Reports 72.1% across people on BNCI2014-001, but with windows starting at the cue and test-set re-normalisation. Its online variant drops to 58.4%.

## Worth testing (all after YC unless the founders choose otherwise)
1. **Channel-subset training (architecture §4.3)**, in our own code.
   - *Hypothesis:* it closes most of the 3.4-point Ganglion gap and lifts C3/C4 and the Crown layout off about 51%, without costing full-cap accuracy.
   - *Cost:* about a day of code and about an hour of GPU.
   - **The only candidate that could go before the freeze, and only with a bar written first and a hard cut-off.** It would also mean a new bundle, replay and rehearsal.
2. **Left/right mirror averaging at prediction time**, on E2's existing models: average each window's prediction with its mirrored copy, labels swapped.
   - *Mechanics:* the C-core layout is mirror-symmetric, and both CAR and EA commute with the mirror.
   - *Cost:* CPU only, 1–2 h. Expected gain small (≤ 1 point).
3. **Mixup or segment recombination in pooled training.** About an hour of GPU per arm.
4. **A backbone sweep** (EEGNeX, ShallowFBCSPNet, which is already parked as S4, EEGSimpleConv, CTNet): copy single BSD-3 files rather than installing the library. Mainly for numbers comparable with the field.

## Couldn't verify
- PyPI didn't render, so the dependency list is from the master branch's `pyproject.toml`, not 1.5.1's.
- The NOTICE file returned 404, so the full list of non-commercial files is unknown. A lawyer would need to check before any vendoring.
- The provenance of re-hosted LaBraM, CBraMod and EEGPT weights on Hugging Face is unverified.
- HGD's page is inconsistent on imagined vs executed movement.

## Sources (main)
- **Braindecode site:**
  - [docs index](https://braindecode.org/stable/index.html)
  - [models table](https://braindecode.org/stable/models/models_table.html)
  - [API](https://braindecode.org/stable/api.html)
  - [what's new (stable)](https://braindecode.org/stable/whats_new.html), [dev](https://braindecode.org/dev/whats_new.html)
- **Tutorials:**
  - [pretrained models](https://braindecode.org/stable/auto_examples/model_building/plot_load_pretrained_models.html)
  - [channel interpolation](https://braindecode.org/stable/auto_examples/model_building/plot_channel_interpolation.html)
  - [cropped decoding](https://braindecode.org/stable/auto_examples/model_building/plot_bcic_iv_2a_moabb_cropped.html)
  - [augmentation](https://braindecode.org/stable/auto_examples/advanced_training/plot_data_augmentation.html)
  - [augmentation search](https://braindecode.org/stable/auto_examples/advanced_training/plot_data_augmentation_search.html)
- **Reference pages:**
  - [Filter](https://braindecode.org/stable/generated/braindecode.preprocessing.Filter.html)
  - [EEGPrep](https://braindecode.org/stable/generated/braindecode.preprocessing.EEGPrep.html)
  - [exponential_moving_standardize](https://braindecode.org/stable/generated/braindecode.preprocessing.exponential_moving_standardize.html)
  - [ChannelsSymmetry](https://braindecode.org/stable/generated/braindecode.augmentation.ChannelsSymmetry.html)
  - [EEGSym](https://braindecode.org/stable/generated/braindecode.models.EEGSym.html)
  - [ATCNet](https://braindecode.org/stable/generated/braindecode.models.ATCNet.html)
  - [EEGNeX](https://braindecode.org/stable/generated/braindecode.models.EEGNeX.html)
- **Code and models:**
  - [GitHub](https://github.com/braindecode/braindecode)
  - [pyproject.toml](https://github.com/braindecode/braindecode/blob/master/pyproject.toml)
  - [augmentation code](https://raw.githubusercontent.com/braindecode/braindecode/master/braindecode/augmentation/functional.py)
  - [OpenEEG-Bench](https://github.com/braindecode/OpenEEGBench)
  - [Hugging Face models](https://huggingface.co/braindecode)
- **Papers:**
  - [Rommel 2022](https://arxiv.org/abs/2206.14483)
  - [arXiv 2405.14994](https://arxiv.org/abs/2405.14994)
  - [EEGSimpleConv](https://arxiv.org/abs/2309.07159)
