# Taming foundation model with invariance-oriented pre-training for broad-spectrum EEG analysis (INCEPT)

**Dou, Wu, Chen, Ju, Cui, Shen** · arXiv 2608.24597 · Aug 2026
Link: [arXiv 2608.24597](https://arxiv.org/abs/2608.24597)
**Read:** targeted full text (intro, method overview, datasets, results tables for PhysioNet-MI), 2026-09-28
**Relevance:** the current state of EEG foundation models (FMs). It's the best evidence for *and against* the "why not just use a foundation model?" question YC will ask.

## What it is
- **Pretraining corpus:** Temple University Hospital EEG Corpus (TUEG), about **11,000 hours** from **14,987 subjects**, 19 channels, mostly 250 Hz, all unlabelled.
- **Objective:** masked contextual learning plus **invariance-oriented** learning ("which subject-sensitive information stays stable across different views of the same subject and recording"). It's positioned against reconstruction-centred FMs (CBraMod, CSBrain, CodeBrain).
- **Montage handling:** **spherical-harmonics electrode embeddings** encode continuous scalp positions, so the model doesn't depend on a fixed channel order or count.
- **Evaluation:** 10 downstream datasets across signal-level, brain-state and brain-health tasks. Mostly **subject-level splits** (cross-subject). Both linear probing (frozen backbone) and full fine-tuning.

## Key numbers: PhysioNet-MI (4-class, 109 subjects, subject-level split), balanced accuracy

| Model | How trained | Balanced accuracy |
|---|---|---|
| EEGNet | from scratch, task-specific | **59.79%** |
| EEG-Conformer | from scratch | 57.10% |
| CBraMod | linear probe / full fine-tune | 53.58% / 62.73% |
| **INCEPT** | linear probe / full fine-tune | **58.98% / 63.70%** |

INCEPT tops 26 of 30 linear-probing metrics and 24 of 30 fine-tuning metrics across the benchmark.

## What it means for NeurOS
- **Foundation models don't solve cross-subject motor imagery.** After full fine-tuning, the best FM beats a 2k-parameter EEGNet by about 4 points on PhysioNet-MI 4-class. Frozen FM features (linear probe) are *worse* than EEGNet. This is our strongest evidence that **lightweight alignment + adaptation + small models is a defensible stance**, and a good YC answer to "why not a foundation model?".
- A key design sentence: state effects are "expressed as **deviations from a subject-specific baseline** rather than as absolute waveform patterns." That's exactly the principle behind EA and per-user re-centring, from a foundation-model group.
- **Spherical-harmonics electrode embeddings** are a strong candidate for the hardware-agnostic input layer (any montage in, one representation out). Compare with spherical-spline interpolation and graph inputs in [`../synthesis.md`](../synthesis.md).
- The benchmark includes **Mental Arithmetic (PhysioNet EEGMAT)**. EEGMAT results can be compared directly against published FM numbers.

## Caveats
- The TUEG corpus is clinical, 19 channels and hospital-grade. Its domain differs from consumer headsets.
- FMs are a Phase 2 lever for us (pretrain on public data, then fine-tune with alignment), not an MVP dependency.

## Layer-1 references
CBraMod (Wang 2025), CSBrain, CodeBrain, EEGNet, EEG-Conformer, SPaRCNet, TUEG (Obeid & Picone 2016). See [`../layer-1-references.md`](../layer-1-references.md).
