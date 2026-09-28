# Characterizing the dynamics of mental representations: the temporal generalization method

**King, J-R. & Dehaene, S.** · *Trends in Cognitive Sciences* 18(4):203–210 · 2014
Links: [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1364661314000199) · [free author manuscript (PMC5635958)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5635958/)
**Read:** full text (PMC), 2026-09-28
**Relevance:** evaluation and interpretability tool; informs window length and live-decoding design. Not an architecture.

## What it is
Train a decoder on brain data at time *t*, test it at every other time *t′*. The resulting **train-time × test-time matrix** shows *how* a neural code evolves:

| Pattern in the matrix | What it means |
|---|---|
| Thin diagonal | A chain of short-lived codes: each processing stage has its own pattern |
| Square / block | One stable code sustained over time |
| Off-diagonal reactivation | The same code comes back later |
| Oscillating stripes | The code recurs cyclically |
| Below-chance regions | The code flips polarity (reversal) |

It extends beyond time: train in condition A, test in condition B (**generalization across conditions**) to ask whether two conditions share a code.

## What it means for NeurOS
- **Choosing the decoding window.** For the live engine we need to know from when after intent onset, and for how long, a user's intent code is stable. A temporal generalization matrix on our training data answers this directly and justifies the sliding-window length and update rate.
- **Cross-subject generalization is the same idea.** "Train on subjects A–Y, test on subject Z" is generalization across conditions. The paper gives vocabulary and figures for showing that our aligned representation is shared across people.
- **A demo and diagnostics asset.** A square block in the matrix after alignment, and not before, is a clean visual that alignment produces a *stable, shared* code.

## Caveats
- The method comes from evoked-response (ERP) research. Motor imagery is oscillatory (mu/beta power changes), so apply it to band-power or covariance features computed in sliding windows, not raw voltages.
- It measures *when* information is decodable, not *how well* a product works. It's not a user-facing metric.

## Layer-1 references
None followed. This is a methods review, and the method is what we need.
