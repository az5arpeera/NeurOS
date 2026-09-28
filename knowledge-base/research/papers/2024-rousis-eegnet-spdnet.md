# Combining EEGNet with SPDNet towards an end-to-end architecture for imagined speech decoding

**Rousis, Kalaganis, Nikolopoulos, Kompatsiaris, Petrantonakis** · EUSIPCO 2024, pp. 1531–1535
Links: [IEEE 10715364](https://ieeexplore.ieee.org/document/10715364) · [free copy (EURASIP)](https://eurasip.org/Proceedings/Eusipco/Eusipco2024/pdfs/0001531.pdf). The two IEEE links in the original list are the same paper.
**Read:** full text (EURASIP PDF), 2026-09-28
**Relevance:** the reference design for putting Riemannian (SPD) layers *inside* the network ("Option B" in the founder notes). **No cross-subject evidence.**

## Architecture
1. **EEGNet temporal convolution:** 2D filters of shape (1, fs/2), then BatchNorm, ReLU, mean-pool (1,4) and dropout 0.5. Each filter output is a band-passed copy of the multichannel EEG.
2. **Covariance per feature map:** a channels × channels SPD matrix (they call these cross-covariance, CCV).
3. **BiMap:** Y = W X Wᵀ, with W full row rank (Stiefel-constrained), so the output stays SPD while the dimension shrinks.
4. **ReEig:** eigenvalues clamped to at least ε (the SPD analogue of ReLU).
5. **LogEig:** matrix log maps to a flat tangent space, then flatten and a fully connected layer.

## Numbers
**Protocol: subject-specific.** One model is trained per subject, and there is no cross-subject test.

| Dataset | Task | EEGNet | MFCC+SVM | EEGNet-SPDNet | Chance |
|---|---|---|---|---|---|
| Kara One (11 of 14 subjects, 64 ch, 1 kHz, 132 trials/subject) | 11 imagined prompts | 14.05% | 17.84% | **24.79%** | 9.1% |
| 2020 BCI Competition imagined speech (15 subjects, 64 ch, 5 words, 400 trials/subject) | 5 imagined words | 50.26% | 37.86% | **66.93%** | 20% |

## What it means for NeurOS
- Learned temporal filters feeding SPD layers are a strong inductive bias. It beat EEGNet by 10–17 points on imagined speech. It's a credible **v2 encoder**, especially for connectivity-heavy tasks like imagined speech.
- The eigendecomposition cost is O(C³). For C ≤ 64 channels that's well under a millisecond per window, so latency isn't the blocker the founder notes suggest. Training stability and Stiefel-manifold optimisation (e.g. `geoopt`) are the real costs.
- It says **nothing about zero-calibration**. Don't cite it for cross-subject claims.

## Caveats
- Small datasets and per-subject models; imagined-speech accuracy is still far from usable.
- The imagined speech paradigm itself has large subject variability, with some subjects at or below 60% on 5 classes.

## Layer-1 references
Huang & Van Gool 2017 (SPDNet), Schirrmeister 2017 (Shallow/Deep ConvNet), Lawhern 2018 (EEGNet). See [`../layer-1-references.md`](../layer-1-references.md).
