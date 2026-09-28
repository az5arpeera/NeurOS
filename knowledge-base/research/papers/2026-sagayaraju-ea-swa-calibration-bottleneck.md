# Overcoming the BCI Calibration Bottleneck: A Clinically-Grounded Architecture using Riemannian Alignment and Stochastic Weight Averaging

**Sagayaraju, I. P.** (single author, MSc student, Univ. of Bern) · arXiv 2607.16225 · Jun 2026 · **not peer-reviewed**
Link: [arXiv 2607.16225](https://arxiv.org/abs/2607.16225)
**Read:** full text, 2026-09-28
**Relevance:** the closest paper to the founder notes' MVP stack (EA + EEGNet). It's the source of the "90.97%" figure, which is **one subject**.

## Pipeline (5 stages)
1. Band-pass 1–40 Hz.
2. **Per-session ICA** (11 components). Components correlated with the EOG channels are removed.
3. **EA:** R̄ = mean of XᵢXᵢᵀ over **all trials of the session**; X̃ᵢ = R̄^(−1/2) Xᵢ. Then a per-trial z-score.
4. **EEGNet**, trained with AdamW (lr 1e-3, weight decay 0.05), dropout 0.5, 300 epochs.
5. **Stochastic Weight Averaging (SWA):** average the weights over the last K epochs.

## Numbers (BNCI2014-001, left vs right hand, 22 channels + 3 EOG, 9 subjects)

| Configuration (**Subject 1 only**) | Accuracy |
|---|---|
| Raw time series | 57.56% |
| EEGNet with band-pass | 76.74% |
| + alignment | 83.10% |
| + ICA + SWA | **90.97%** (peak single epoch 93.75%) |

- **Leave-one-subject-out (LOSO) across all 9 subjects: 74.31% ± 10.94%** (κ 0.486). Subjects 2 and 5 were about 63%, and subject 3 reached 91.3%.
- **Cross-dataset** (train BNCI2014-001, test PhysioNet MI, 22-channel subset, 160 → 250 Hz): **60.0%**.

## What it means for NeurOS
- EA and SWA are cheap, compatible wins: +6.4 points from alignment and more from SWA, on the one subject they ablate. Both belong in our training recipe.
- **The honest benchmark from this paper is 74% binary cross-subject on 9 subjects.** The 90.97% number must never appear in our materials as a cross-subject result.
- **Cross-dataset transfer (60%) is the real open problem.** It's also exactly our "any headset" claim. Our MVP must measure it head-on.

## Caveats
- **The EA reference uses all target-session trials**, computed offline and transductively. That isn't how a live stream works: we won't have future trials. Online EA estimates the reference incrementally (see Junqueira 2024 and Wimpff 2024 in the layer-1 notes).
- ICA needs EOG channels and manual or offline fitting, so it isn't real-time as described.
- The ablation covers one hand-picked subject. The t-SNE "perfect overlap" claims are qualitative.

## Layer-1 references
Ganin 2016 (DANN; the author calls it "highly unstable" given EEG data scarcity), Izmailov 2018 (SWA), Barachant 2010 (Riemannian BCI), Jayaram & Barachant 2018 (MOABB). See [`../layer-1-references.md`](../layer-1-references.md).
