# EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces

**Lawhern, Solon, Waytowich, Gordon, Hung, Lance** · *Journal of Neural Engineering* 15:056013 · 2018
Link: [arXiv 1611.08024](https://arxiv.org/abs/1611.08024)
**Read:** targeted full text (architecture, parameter table, protocols), 2026-09-28
**Relevance:** the default MVP backbone. It's the baseline every other paper on our list compares against.

## Architecture
1. **Temporal conv:** F1 filters with kernel length about fs/2 (a learned band-pass filter bank).
2. **Depthwise spatial conv:** D spatial filters per temporal filter across all channels, with a max-norm constraint (learned CSP-like spatial filters).
3. **Separable conv** (depthwise temporal, then pointwise), average pooling, dropout.
4. A dense softmax layer.

Notation: EEGNet-F1,D (e.g. EEGNet-8,2).

## Numbers
**Parameter counts (Table 3):**

| Model | P300 (1 s) | SMR (2 s) |
|---|---|---|
| DeepConvNet | 174,127 | 152,219 |
| ShallowConvNet | 104,002 | 40,644 |
| EEGNet-4,2 | 1,066 | 796 |
| **EEGNet-8,2** | **2,258** | **1,716** |

- It works across four paradigms: P300, error-related negativity (ERN), movement-related cortical potentials (MRCP), and sensorimotor rhythm (SMR, i.e. BCI IV 2a motor imagery). Both within- and cross-subject protocols were used.
- Within-subject SMR performance was similar to FBCSP, the classic hand-engineered pipeline.
- Cross-subject SMR is much lower than within-subject. The results are reported as figures only.
- For more recent cross-subject reference points: BCIC-IV-2a 4-class 56.85% (from the EAMCCA paper) and PhysioNet MI 4-class 59.79% balanced accuracy with a subject-level split (from INCEPT).

## What it means for NeurOS
- **About 2k parameters.** The founder notes' "~3,000 parameters" is the right order of magnitude. Inference is a few milliseconds on CPU, so the <30 ms latency target is realistic.
- It's tiny, so a pool of about 370 subjects won't overfit it. The question is whether it has **enough capacity** for 5+ pooled datasets. Plan an EEGNet → bigger-backbone ablation (EEG-Conformer, ATCNet, EAMCCA-style) rather than assuming.
- Its depthwise spatial filters are inspectable (topographic maps). That's useful for the "the model uses motor cortex, not eye blinks" slide, together with Grad-CAM.

## Caveats
- The 2018 cross-subject SMR results are weak. EEGNet alone is **not** a zero-calibration solution; alignment and adaptation do the heavy lifting.
- The kernel length assumes a fixed sampling rate. Resample every dataset to one rate (e.g. 128 or 250 Hz) before pooling.

## Layer-1 references
Schirrmeister 2017 (Shallow/Deep ConvNet); BCI Competition IV 2a. See [`../layer-1-references.md`](../layer-1-references.md).
