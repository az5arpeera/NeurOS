# Research synthesis: what the evidence supports (as of 28 Sep 2026)

Built from the [primary papers](papers/) and the [layer-1 references](layer-1-references.md). Each claim links to its source note. **Read this before writing any pitch number.**

## Bottom line
1. **Zero-calibration motor imagery works, within honest bounds.** Published cross-subject accuracy is about **77–83% for left/right** and about **66–68% for 4-class**, with no labels from the new user. That's usable for a binary or ternary control channel, not for high-bandwidth control.
2. **Alignment is the cheapest, most reliable lever.** EA or Riemannian re-centring is label-free, costs microseconds, and is worth about 4–8 points. It needs about **16–32 trials of the new user's unlabeled data**, not 3 seconds.
3. **The evidence-backed "UDA" is test-time adaptation, not GANs or DANN.** Online alignment, adaptive batch norm and entropy minimisation raise 4-class cross-subject accuracy from **57.4% to 67.3%** with no calibration task (Wimpff 2024).
4. **CLAP-style embeddings help when the system knows the candidates** (stimuli on screen or in the ears, or a list of commands). No non-invasive result supports decoding arbitrary imagined content into an open vocabulary. Cross-subject EEG → CLIP is **6.2% top-1 on 200 images** (12× chance). EEG → CLAP music has only been shown **within-subject**.
5. **Foundation models don't solve cross-subject motor imagery.** 11,000 hours of pretraining buys about **+4 points** over a 2k-parameter EEGNet on PhysioNet-MI 4-class. Small aligned models are a defensible stance.

## 1. Realistic accuracy (no labeled data from the target user)

| Benchmark (protocol) | Best published | Source |
|---|---|---|
| BCIC-IV-2a, 4-class, LOSO, 9 subjects | 66.2% (EAMCCA), **68.1%** (RUNet, strict zero), 67.3% (online adaptation) | [EAMCCA](papers/2026-meng-eamcca.md), [RUNet](papers/2026-jin-runet.md), [Wimpff 2024](layer-1-references.md) |
| BCIC-IV-2b, left/right, LOSO, 9 subjects, 3 channels | 77.1% (EAMCCA), 78.3% (RUNet), **83.5%** (online adaptation) | same |
| BNCI2014-001 left/right subset, LOSO, 9 subjects | 74.3% ± 10.9 (EA + EEGNet + SWA) | [EA+SWA paper](papers/2026-sagayaraju-ea-swa-calibration-bottleneck.md) |
| PhysioNet MI left/right fist, LOSO, first 30 subjects | 78.1% (RUNet) | [RUNet](papers/2026-jin-runet.md) |
| PhysioNet MI 4-class, subject-level split, 109 subjects | 63.7% BA (INCEPT fine-tuned); EEGNet 59.8% | [INCEPT](papers/2026-dou-incept-foundation-model.md) |
| Stieger2021 left/right, LOSO, 20 subjects | 59.2% (rPLV-GAT) | [graph paper](papers/2025-patel-plv-gat-calibration-free.md) |
| **Cross-dataset** (train BNCI2014-001, test PhysioNet), left/right | **60.0%** | [EA+SWA paper](papers/2026-sagayaraju-ea-swa-calibration-bottleneck.md) |

Chance is 50% (binary) and 25% (4-class). Subject spread is huge: per-subject cross-subject accuracy on 2a ranges from **45% to 87%**. About 15–30% of people are "BCI-inefficient" for motor imagery.

**Implication:** a credible MVP headline is binary or ternary (left / right / rest), cross-subject and **cross-dataset**, on **hundreds of held-out subjects**, with an adaptation curve. The scale of evaluation is where we can beat every paper above, since each uses 9–30 subjects.

## 2. Three levels of "zero calibration" (proposed house definitions)

| Level | Target-user data used | Who uses this definition | Expected accuracy |
|---|---|---|---|
| **L0: strict zero** | None, not even unlabeled | RUNet, graph paper | Lowest |
| **L1: passive warm-up** | Unlabeled data streamed during normal use (alignment reference, BN statistics, entropy minimisation). **No calibration task or screen.** | EA papers, Wimpff 2024 | +5–10 points over L0 |
| **L2: few-shot** | A handful of labeled trials | Many "reduced calibration" papers | Highest |

**Proposal:** the product claims **L1** ("no calibration session; it adapts silently while you use it"), and we always report L0 alongside it. That's both the strongest honest claim and the best UX.

## 3. The EA warm-up in practice
> **Founder target (D-009, 2026-10-01):** usable within seconds, full accuracy within **30–45 s**, no calibration task. Plan and hypothesis H1 (a resting-state-seeded reference) are in [the calibration explainer](../explainers/calibration-time-commands-and-metrics.md).

- The reference R̄ must come from the **new user's own data**. Junqueira 2024 used the first **24 trials** and cites **16–32**. Wimpff 2024 shows **EMA or linear online updates match batch alignment** (62.45% vs 62.48%).
- **Resting-state re-centring has precedent** (Zanini et al. 2018 re-centred on resting-state trials). If a 30-s relaxed "settling" period gives a good-enough reference, the 30–45 s target is reachable. This is the first experiment to run (H1).
- **Design:** start from a population reference (identity after training-time alignment), update it with an EMA over sliding windows from the first second, and let accuracy climb over about 1–2 minutes. Show that climb in the demo; it's a feature, not a flaw.
- EA placement: band-pass → optional average reference → **EA** → spatial filtering / network ([Revisiting EA](papers/2025-wu-revisiting-euclidean-alignment.md)).

## 4. Where UDA actually belongs

| Layer | Technique | Needs target data? | Evidence | Proposed role |
|---|---|---|---|---|
| Input | EA / RA re-centring per subject | Unlabeled | Strong (every paper) | **Core** |
| Training | Multi-source domain generalisation: per-subject EA, SWA; optional DANN / Deep CORAL / MMD losses | No | Mixed. DANN reported unstable on small EEG data; MMD baselines weak in the graph paper. | **Ablation / experiment** |
| Network | Per-domain Riemannian BN (SPDDSMBN, Kobler 2022) or EMA re-centring (RUNet) | Unlabeled (per-domain statistics) | Strong (NeurIPS 2022, TBME 2026) | v2 candidate |
| **Test time** | **Online alignment + adaptive BN + entropy minimisation** (source-free) | Unlabeled, streamed | **+10 points** on 2a cross-subject (Wimpff 2024) | **Core: this is our "UDA"** |

Strictly speaking, DANN or CORAL across *training* subjects is domain **generalisation**. UDA proper adapts to the *target*, which in a live product means test-time adaptation. Both can be in the MVP; only the second is proven for live use.

## 5. Where CLAP-style contrastive heads belong

| Option | What gets embedded | Evidence | Zero-calibration? | Demo-ability | Risk |
|---|---|---|---|---|---|
| **A. Language-anchored intent head** | Text embeddings of command phrases ("imagine moving your left hand", "rest", ...). EEG embeddings are trained contrastively against them. | Same mechanism as CLIP zero-shot classification. It **unifies label vocabularies across 5+ datasets**, and lets developers name commands. | Yes (sits on top of the aligned encoder) | High | Low–medium |
| **B. CLAP audio head: music / attention** | CLAP audio embeddings of the sounds the user hears; retrieval among candidates | Within-subject 48.7% on 50-way (Qing 2026). Headphone attention decoding at 64–71% with per-user models. **No cross-subject result exists.** | Would be **novel** (NMED-T allows LOSO) | Medium (needs audio playback) | High |
| **C. CLIP image head** | Image embeddings (THINGS-EEG2) | Cross-subject 6.2% top-1 / 21.4% top-5 on 200-way | Yes, weakly | Low (needs a stimulus screen) | Medium |
| ✗ **Open-vocabulary imagined actions** ("jump over the wall") | — | **No non-invasive evidence** | — | — | Credibility risk |

**Proposal:** A is in the core MVP. B is a research track that could become the "wow" result (first cross-subject EEG → CLAP music identification). Say plainly in the application that open-vocabulary decoding is the long-term vision, not today's capability.

## 6. Foundation models vs small models
- INCEPT (11,000 h of pretraining) is +3.9 points over EEGNet on PhysioNet-MI 4-class after full fine-tuning, and *worse* with a frozen backbone. The MOABB benchmark finds Riemannian pipelines strongest with limited data.
- **Stance:** small aligned models plus test-time adaptation for the MVP; FM pretraining is a Phase 2 lever. Include **Riemannian baselines** (tangent space + logistic regression, MDM) in every benchmark table, since reviewers expect them.

## 7. Hardware-agnostic input: three options

| Option | How | Pros | Cons |
|---|---|---|---|
| Canonical montage + interpolation | Map any headset to a fixed virtual montage (e.g. 16–22 channels around motor cortex) by spherical-spline interpolation; mark missing channels | Simple; works with any CNN | Interpolation invents data when coverage is poor |
| Graph input | Nodes = actual electrodes, edges = connectivity (PLV) or distance | Handles any count/layout natively | More compute; modest accuracy so far |
| Electrode-position embeddings | Spherical-harmonics encoding of each electrode's 3-D position (INCEPT) | Principled; FM-grade | More complex; needs more data |

**MVP proposal:** canonical montage (the channel intersection of the training datasets, around C3/Cz/C4) + interpolation, and pick a demo headset that covers C3/C4.

## 8. Real-time budget
Sliding windows of 1–2 s, decisions every ~62.5 ms, CNN inference ~2 ms on CPU (Wimpff 2025). **The <30 ms latency target is realistic** with the whole stack on a laptop.

## 9. Straw-man MVP pipeline (for discussion, not decided)
```
[Any headset] --LSL--> Ingest: resample, band-pass/notch, map to canonical montage
      --> A. Online alignment (EA/RA via EMA; starts from population reference)      label-free
      --> B. Encoder (EEGNet-family; trained on ~370 public subjects with per-subject EA + SWA;
             optional DANN/CORAL loss as an ablation)
      --> C. Test-time adaptation (adaptive BN stats + entropy minimisation; source-free)   = "UDA"
      --> D. Language-anchored head (EEG embedding vs text embeddings of command phrases)   = "CLAP"
      --> Confidence gating (entropy threshold: act or hold) --> intent events (JSON / WebSocket) --> apps
Research track: D' = EEG vs CLAP-audio embeddings of candidate sounds (music ID / attention), LOSO on NMED-T.
```

## 10. Top risks
1. **Cross-dataset transfer** (60% in the only published test) is exactly the "any headset" claim, and it's the weakest link. Measure it first.
2. **BCI inefficiency:** a live demo user may simply not produce usable motor imagery. Screen demo users in advance.
3. **Dataset licences:** commercial use of some public datasets may be restricted. Check each one ([`../datasets.md`](../datasets.md)).
4. **Scope:** doing alignment + adaptation + a language head + a music research track + an SDK + a demo + the YC application in ~33 days needs ruthless prioritisation. See [`../open-questions.md`](../open-questions.md).
