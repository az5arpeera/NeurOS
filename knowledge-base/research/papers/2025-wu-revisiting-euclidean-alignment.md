# Revisiting Euclidean Alignment for Transfer Learning in EEG-Based Brain-Computer Interfaces

**Wu, D.** (Huazhong Univ. of Science & Technology; co-author of the original EA paper) · arXiv 2502.09203 (v2, May 2025)
Link: [arXiv 2502.09203](https://arxiv.org/abs/2502.09203)
**Read:** targeted full text (math, placement experiments, comparison table, tabulated gains), 2026-09-28
**Relevance:** the authoritative guide to *how* to use EA. It's the justification for EA as our first pipeline stage.

## The math
Per domain (subject or session), with N trials Xₙ ∈ ℝ^(C×T):

R̄ = (1/N) Σₙ XₙXₙᵀ  →  X̃ₙ = R̄^(−1/2) Xₙ

After alignment the mean covariance of every domain is the identity matrix I. This is what "the average brain" means in our pipeline: a statistical origin, not an anatomical template.

## Key findings
- **Placement matters:** use temporal filter → (optional common-average re-reference) → **EA** → spatial filter / classifier. EA should come after band-pass filtering and before any spatial filtering. Adding average re-referencing before EA helped slightly on average.
- It's **unsupervised**: no target labels needed. It works with any downstream model (CSP, Riemannian classifiers, CNNs) and works both offline and online.
- Tabulated motor imagery gains across published studies run from low single digits up to **27.36 points** (FBCSP 52.62 → 79.98, Amorim et al. 2024).
- It's been applied across about 10 paradigms (MI, P300/ERP, SSVEP, emotion, sleep staging, seizure, cognitive load) and beyond EEG (MEG, intracranial). The author argues EA "should be a standard pre-processing step" for cross-subject work.
- A comparison table of alignment variants (RA, EA, RPA, CORAL, adaptive BN, ...) lists which need labeled target trials, which work online, and what they use as reference. EA and RA need no labels.

## What it means for NeurOS
- EA is a **deterministic, label-free first stage**. It cancels each person's (and each headset's) average spatial covariance: electrode placement, impedance and amplitude scale all collapse to I.
- EA is a *first-order* fix. It re-centres, but it doesn't rotate class structure between subjects or fix non-stationarity within a session. That's what the test-time adaptation (UDA) layer and better features must handle.
- The reference must be **estimated from the new user's stream**. How many trials that needs is in the layer-1 notes (Junqueira 2024: 24 trials; 16–32 recommended).

## Caveats
- It's a review, so the gains are heterogeneous across studies, protocols and pipelines. Don't quote an "average EA gain" without naming the setting.
- Estimating R̄^(−1/2) becomes unstable with many channels and few trials. Regularise it: shrinkage, or ε·I plus eigenvalue clamping.

## Layer-1 references
He & Wu 2020 (original EA), Zanini 2018 (Riemannian re-centring), Rodrigues 2019 (RPA), Junqueira 2024 (EA + deep learning), Wimpff 2024 (online test-time adaptation), Amorim 2024. See [`../layer-1-references.md`](../layer-1-references.md).
