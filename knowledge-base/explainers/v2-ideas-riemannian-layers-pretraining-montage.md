# Post-MVP ideas explained: Q13, Q14, Q15

*Founder asked for explanations on 2026-10-01.*

## Q13. Riemannian math *inside* the network
**Today (MVP):** alignment is a fixed preprocessing step. We compute each person's reference matrix, whiten their signal, then hand it to an ordinary CNN.

**The idea:** make the geometry part of the network itself, with layers that operate directly on covariance (SPD) matrices and *learn*:
- **SPDNet layers** (BiMap → ReEig → LogEig): learned, shape-preserving transforms of covariance matrices (see the [EEGNet+SPDNet note](../research/papers/2024-rousis-eegnet-spdnet.md)).
- **Per-person Riemannian batch norm** (SPD-DSMBN, Kobler 2022): the network keeps a separate running "average brain" per person and re-centres on it, which is alignment as a trainable layer.

**Pros:** end-to-end; the network chooses which frequency bands' covariances to align; per-person statistics live inside the model.
**Cons:** harder optimisation (weights constrained to special manifolds; libraries like `geoopt`), more engineering, harder debugging.
**Why v2:** measure the simple pipeline first, and add complexity only if it beats it on our benchmark.

## Q14. Resting-state contrastive pretraining (RUNet-style)
**The idea:** before teaching the network any commands, teach it *what EEG looks like* using lots of unlabeled resting data:
1. Take a resting EEG window.
2. Make two distorted copies (add noise, rescale amplitude, blank out 10% of time points).
3. Train the network to recognise that the two copies are the same window and different from all other windows (SimCLR contrastive loss).
4. Then train on labeled task data starting from these weights.

**Result in RUNet:** +1.2–1.6 points cross-subject.
**Data cost:** almost free. Every dataset has rest periods, and PhysioNet has 1-minute eyes-open and eyes-closed runs for 109 people.
**Synergy with our 30–45 s target:** the same resting data is what we use for the rest-seeded alignment reference (see [calibration explainer](calibration-time-commands-and-metrics.md)).
**Why "Phase 1.5":** a modest gain for an extra training stage. Add it once the core pipeline works.

## Q15. Montage-agnostic input (making the model work with *any* headset)
**The problem:** every headset has a different set of electrodes (see the [hardware explainer](demo-hardware-lsl-and-compute.md)). Unicorn has Fz, C3, Cz, C4, Pz, PO7, Oz, PO8; Crown has CP3, C3, F5, PO3, PO4, F6, C4, CP4; our training datasets have 22–128 channels. A normal network expects a **fixed list of channels in a fixed order**.

Three solutions:

| Approach | How | Analogy | When |
|---|---|---|---|
| **1. Canonical montage + interpolation** | Pick a standard channel list; estimate any missing electrode from its neighbours (spherical-spline interpolation); drop extras | Resampling every photo to 224×224 | **MVP**: enough for 1–2 demo headsets |
| **2. Graph input** | Each electrode is a node; edges connect electrodes (by distance or by signal synchrony, like PLV); graph networks accept any number of nodes | Social network: works whoever is in it | After the MVP baseline |
| **3. Electrode-position embeddings** | Feed each signal *together with its 3-D scalp location* (e.g. spherical-harmonics encoding, as in INCEPT), so the model knows where every signal comes from | Words + positional encoding in a Transformer | **The long-term "universal" input layer** |

**Why not 2 or 3 right away?** For the MVP we control which headsets appear in the demo, so approach 1 is enough and much simpler. Approaches 2/3 become essential when *customers* bring arbitrary headsets. That's the product moment where "universal layer" has to be literally true, so it belongs on the post-MVP roadmap.
