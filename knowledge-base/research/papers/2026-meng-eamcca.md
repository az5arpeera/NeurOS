# EAMCCA: Euclidean Alignment based multiscale convolution and channel attention network for MI-EEG decoding

**Meng, Hou, Wu, Zuo, Ma, Liu** · *Biomedical Signal Processing and Control* vol. 128, art. 111272 · Dec 2026
Link: [ScienceDirect S1746809426018264](https://www.sciencedirect.com/science/article/pii/S1746809426018264) (paywalled; read through UVA)
**Read:** full text, 2026-09-28
**Relevance:** the most recent "EA in front of a bigger network" design. It gives current state-of-the-art cross-subject numbers on the standard motor imagery (MI) benchmarks.

## Pipeline
1. Scale each trial to [−1, 1] (divide by max |x|).
2. **EA:** R̄ = mean of xᵢxᵢᵀ over all trials of a subject; x̃ᵢ = R̄^(−1/2) xᵢ.
3. Two parallel branches:
   - **Multiscale temporal convolution:** kernels (1×15, 1×25, 1×51, 1×75), 8 filters each, then spatial conv (22×1) and average-pool (50, stride 15), giving 32×64.
   - **Channel-weighted branch:** learnable channel weights, temporal conv 1×25, spatial conv (22×1), then log-variance pooling (50, stride 15), giving 32×64.
4. A per-branch Transformer encoder (4 layers, 8 heads, dim 32, FFN 128).
5. Fusion (stack the branches, 2×1 conv with 64 filters), then a fully connected softmax layer.
Training: Adam, lr 0.002, batch 20, 200 epochs, dropout 0.5.

## Numbers
**Subject-specific (session 1 → session 2):** BCIC-IV-2a 86.61% (4-class), BCIC-IV-2b 89.42% (2-class), HGD 93.61% (4-class).

**Cross-subject** (the table's protocol is "cross-subject"; how the target's EA reference was computed isn't stated):

| Model | BCIC-IV-2a (4-class) | BCIC-IV-2b (2-class) |
|---|---|---|
| ShallowConvNet | 56.75 | 74.28 |
| DeepConvNet | 60.15 | 75.18 |
| EEGNet | 56.85 | 75.13 |
| EEG-Conformer | 53.41 | 73.52 |
| CTNet | 58.64 | 76.27 |
| **EAMCCA** | **66.24** (κ 0.566) | **77.11** (κ 0.542) |

**Ablation (subject-specific only):** removing EA on 2a costs 8.1 points (86.61 → 78.55). Removing the channel-attention or multiscale branch costs 4.2–4.5 points.

## What it means for NeurOS
- EA is worth about 8 points even *within* subject across sessions. It's cheap insurance and belongs in the pipeline no matter what else we do.
- The **realistic cross-subject ceiling on BCIC-IV-2a 4-class is about 66–68%** (EAMCCA 66.2, RUNet 68.1, online adaptation 67.3). Two-class left/right is about 77–83%. Our pitch numbers must sit in these ranges or explain why they're higher.
- The architecture (multiscale convolutions plus attention) is a reasonable "bigger backbone" candidate if EEGNet saturates on our 370-subject pool.

## Caveats
- **Abstract and table disagree:** the abstract says 67.48% cross-subject on 2a; Table 6 says 66.24%.
- The EA ablation is reported only for subject-specific training, not cross-subject.
- 9–14 subjects per dataset, so there's high variance between subjects (2a cross-subject per-subject range: 45.5–87.0%).

## Layer-1 references
He & Wu 2020 (EA), Lawhern 2018 (EEGNet), Song 2022 (EEG-Conformer), Schirrmeister 2017 (HGD). See [`../layer-1-references.md`](../layer-1-references.md).
