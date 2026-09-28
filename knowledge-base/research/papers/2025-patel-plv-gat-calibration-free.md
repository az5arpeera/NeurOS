# Towards Stable and Calibration-Free Motor Imagery BCIs via Graph-Based Transfer Learning

**Patel, Zhu, Bryson, Carlson, Jiang, Demosthenous** (UCL) · TechRxiv preprint, 27 Jun 2025 · **not peer-reviewed** · CC-BY-SA 4.0
Link: [TechRxiv 10.36227/techrxiv.175099867.76418399](https://www.techrxiv.org/doi/full/10.36227/techrxiv.175099867.76418399/v1) · related code: [github.com/rishannp](https://github.com/rishannp/Motor-Imagery---Graph-Attention-Network)
**Read:** full text (PDF extracted in the browser), 2026-09-28
**Relevance:** graph (connectivity) inputs as a montage-flexible, subject-independent representation. It's the paper the founder notes cite for "graphs beat CNNs". The actual numbers are more modest.

## Method
1. For each trial, take instantaneous phases from the Hilbert transform. For every electrode pair compute the **phase-locking value**: PLV = |(1/T) Σₜ e^(i(φ₂(t) − φ₁(t)))|. This gives an n×n graph per trial.
2. A stability analysis using the coefficient of variation of PLV across trials finds electrode pairs whose connectivity is stable across trials, sessions and subjects. Fronto-central and parietal-occipital hubs persist.
3. **rPLV-GAT:** 3 GATv2 layers (32×8 heads → 16×8 → 8), global mean pool, then linear(8 → 2). Adam, lr 1e-3, 25 epochs, batch 32.
4. **Protocol:** leave-one-subject-out, with no target data.

## Data
- **ALS dataset:** 8 patients, 19 channels, 4 sessions over 1–2 months, about 150–160 trials each, left/right cursor control.
- **Stieger 2021:** the first **20 of 62** subjects, the first 4 sessions, left/right trials only (about 800–900 trials per subject), 64 channels.

## Numbers (binary, LOSO)

| Dataset | rPLV-GAT | Notes |
|---|---|---|
| ALS (8 patients) | **62.65%** | Significantly better than 5 of 7 baselines. **Not significantly different from EEGNet (p = 0.46).** ShallowConvNet marginal (p = 0.055). |
| Stieger (20 subjects) | **59.21%** | Significantly better than EEGNet, FTL and 1D-CNN (p < 0.001). Comparable to GCNs-Net using PLV (p = 0.20). |

## What it means for NeurOS
- **Graph inputs are one of three ways to be montage-agnostic**: nodes = whatever electrodes exist. The others are interpolation to a virtual montage, and electrode-position embeddings as in INCEPT. Graphs also handle missing or bad channels well.
- **Absolute accuracies are modest** (about 60% binary). The founder notes' "consistently outperformed standard convolutional models" overstates this. See [`../../claims-audit.md`](../../claims-audit.md).
- **Stieger 2021 is a gold-mine dataset** for us: 62 subjects × 7–11 sessions × 450 trials/class, in MOABB. The authors used a third of it.
- The authors' future-work section (online adaptation with consistency regularisation or self-supervised proxy tasks on top of a calibration-free initial model) is exactly our test-time adaptation plan.

## Caveats
- It's a preprint. Several baselines couldn't run on the larger dataset because of GPU limits, so the comparisons are uneven.
- The per-trial PLV (Hilbert) computation adds latency and needs band-limited signals. That's fine offline but must be measured live.

## Layer-1 references
Stieger 2021 (dataset), Han, Wei & Faisal 2023 (GNN transfer across heterogeneous electrode configurations; **high priority, not yet read**), Hou 2024 (GCNs-Net), Autthasan 2022 (MIN2Net). See [`../layer-1-references.md`](../layer-1-references.md).
