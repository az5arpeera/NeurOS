# RUNet: A Zero-Calibration Framework for Cross-Domain EEG Decoding via Riemannian and Unsupervised Representation Learning

**Jin, Wang, Xu, He, Wu, Li** (ECUST) · *IEEE Transactions on Biomedical Engineering* 73(10):3384–3399 · published online 12 Jan 2026
Links: [IEEE 11346009](https://ieeexplore.ieee.org/document/11346009) (paywalled; read through UVA) · [code: github.com/Chw777/RUNet](https://github.com/Chw777/RUNet)
**Read:** full text (sections I–V), 2026-09-28
**Relevance:** peer-reviewed, code-available, zero-calibration MI framework with Riemannian layers **inside** the network and contrastive pretraining. It's the closest published relative of what the founder notes describe.

## Definition of zero-calibration (important)
"No data from the target subject or target session, **whether resting-state or task-related**, is used during model training." That's **stricter** than EA-style alignment, which uses *unlabeled* target data. At inference RUNet doesn't adapt to the new user at all.

## Architecture
1. **MSCM (multi-scale spatiotemporal convolution):** 3 parallel branches, each with a local spatial conv (KE×1), a global CSP-like depthwise spatial conv (C×1, weights constrained below 1) and temporal kernels of 15, 55 and 75. The branches are concatenated and fused by a pointwise conv.
2. **PCOM (covariance optimisation):** sinusoidal positional encoding, then the sample covariance, then three regularised versions fused with learnable softmax weights:
   - α·Σ + (1−α)I
   - ReLU(Σ/‖Σ‖_F − (1−α)), a soft threshold
   - Σ/(‖Σ‖₂ + ε)
3. **RALM (Riemannian affine log mapping):**
   - Affine-invariant transform Σ̃ = M^(−1/2) Σ M^(−1/2), where **M^(−1/2) is an exponential moving average over training batches** (α = 0.1). At inference the *stored training* reference is used, so this is a Riemannian-batch-norm-style global re-centring, not a per-user one.
   - Then a log-Euclidean map, log(Σ̃) = U log(Λ) Uᵀ, with a custom gradient.
4. Vectorise, then a fully connected layer.

## Training
- **Stage 1: SimCLR contrastive pretraining on resting-state EEG** from source subjects. Two augmentation paths: Gaussian noise (σ 0.1) plus scaling [0.8, 1.2]; and scaling plus 10% temporal masking. 64-d projection, τ = 0.5, 20 epochs, lr 1e-3, batch 64.
- **Stage 2:** full retraining with cross-entropy on task data (30 epochs cross-subject, lr 9e-4, batch 32).
- Hardware: a laptop RTX 4070; about 1–1.6 s per epoch on BCIC-IV-2a.

## Numbers

| Dataset | Cross-session | **Cross-subject (LOSO)** | vs EEGNet (cross-subject) |
|---|---|---|---|
| BCIC-IV-2a (9 subjects, 4-class) | 87.19% | **68.09%** | +7.84 |
| BCIC-IV-2b (9 subjects, 2-class, 3 channels) | 88.03% | **78.29%** | +3.94 |
| PhysioNet MI (**first 30 subjects**, left vs right fist) | — | **78.14%** | +11.48 |
| Lab dataset (11 subjects, 16 channels, 2-class) | 85.45% | **87.25%** | +10.52 |

**Ablations (cross-subject):** removing the pretraining costs 1.2–1.6 points, removing PCOM 2.0–2.6, and removing AIT (the re-centring) 2.1–3.3.

## What it means for NeurOS
- **Report two numbers, not one.** "Strict zero" (no target data, RUNet's definition) and "passive warm-up" (unlabeled target data, as with EA and online adaptation). Both are honest; the second will be higher. The YC text must say which one it quotes.
- Resting-state contrastive pretraining is a cheap, modest gain (+1–2 points). It's a good Phase 1.5 item, since PhysioNet's baseline runs give free resting data.
- The RALM idea (EMA re-centring inside the network) is the in-network cousin of EA. Kobler 2022 makes it **per-domain** (per subject), which is the version we'd want.
- The public code makes it a good external baseline to run on our dataset pool.

## Caveats
- The PhysioNet result uses 30 of 109 subjects and binary classes, so it isn't comparable to the 4-class 109-subject numbers.
- The lab-dataset cross-subject result exceeds its cross-session result, which the authors attribute to more training data. It's a small cohort.

## Layer-1 references
Brooks 2019 (Riemannian BatchNorm), Chen 2020 (SimCLR), Lawhern 2018 (EEGNet), Song 2022 (EEG-Conformer), Altaheri 2022 (ATCNet). See [`../layer-1-references.md`](../layer-1-references.md).
