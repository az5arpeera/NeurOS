# UDA: what it is, what's available, and train-time vs test-time

*Answers open question Q2 (founder asked 2026-10-01).*

## The problem in one picture
```
 Source domains (training)                    Target domain (deployment)
 ~380 people from public datasets             the new person who just put the headset on
 their EEG + labels (left/right/rest)         their EEG, NO labels
            │                                              │
            └── same intent, but the raw signal looks different for every person ──┘
                (skull thickness, electrode placement, impedance, baseline rhythms)
```
A **domain** is a specific *person + headset + session*. A model trained on the sources and applied to a new person drops sharply. RUNet gets 87.2% when tested on the same people on another day, but 68.1% on new people (BCIC-IV-2a, 4-class). That drop is **domain shift**, and closing it without a calibration session is the whole company.

## Vocabulary
| Term | Uses target data? | Plain meaning |
|---|---|---|
| **Domain generalisation (DG)** | No | Train so the model already works on people it has never seen. |
| **Unsupervised domain adaptation (UDA)** | Yes, **unlabeled** | Adapt using the new person's raw signal, without ever asking them to "now imagine left". "Unsupervised" means *no labels*, not *no data*. |
| **Supervised adaptation (calibration / few-shot)** | Yes, **labeled** | The old way: make the user perform cued trials. This is what we eliminate. |

## When the adaptation happens

| | **Train-time UDA** | **Test-time adaptation (TTA / online TTA)** |
|---|---|---|
| When | While training the model | Live, while the person uses the product |
| Target data needed | A batch of the target's unlabeled recordings, mixed into training | The person's incoming stream, a second at a time |
| What changes | All weights (retraining) | Only cheap parts: the alignment matrix, normalisation statistics, a few scale/shift parameters |
| Cost | Minutes to hours on a GPU, **per target** | Milliseconds on a CPU |
| Needs the training data on the device? | Yes | **No** ("source-free"), which is good for privacy and protects our IP |
| Good for | **A new headset type** (adapt once per device model) | **A new person** (adapt every session) |

The business translation: **headset onboarding** (a BCI company brings a new device; we adapt to it once, offline) uses train-time UDA. **User onboarding** (zero calibration for every new person) uses test-time adaptation.

## The menu of available methods

| Family | Methods | What it does | When | EEG evidence | Our use |
|---|---|---|---|---|---|
| **Input alignment** | EA, RA (Riemannian re-centring), RPA (re-centre + stretch + rotate), resting-state re-centring (Zanini 2018) | Whiten each person's signal so their average looks like everyone else's | Test-time (and per-subject at training) | Strong, in every paper we read | ✅ **Core** |
| **Normalisation adaptation** | AdaBN (recompute batch-norm statistics on the target), domain-specific BN, SPD-DSMBN (Kobler 2022) | Re-centre the network's internal activations for this person | Test-time | Strong (+1.7–2.0 points on top of alignment in Wimpff 2024) | ✅ **Core** (AdaBN) |
| **Entropy minimisation** | Tent; label-smoothed variant (Wimpff 2024) | Nudge a few parameters so predictions on this person become confident | Test-time | +1.5–1.9 points on top (to 67.3% on 2a, 83.5% on 2b) | ✅ **Core**, with safeguards (confidence gating, auto-reset) to avoid drift |
| **Pseudo-labelling / self-training** | SHOT, mean teacher, self-ensembling | Use confident predictions as temporary labels | Test- or train-time | Some EEG use | ◐ Later |
| **Distribution-matching losses** | MMD / DAN, Deep CORAL, Wasserstein | Penalise differences between people's feature statistics | Train-time | Mixed (MMD baselines weak in the graph paper) | ◐ Ablation |
| **Adversarial** | DANN, CDAN, ADDA | A discriminator tries to guess *whose* brain a feature came from; the encoder learns to fool it | Train-time | Reported unstable on small EEG data (EA+SWA paper) | ◐ Ablation |
| **Generative translation** | GAN / CycleGAN style transfer between people | Synthesise "person B" versions of "person A" data | Train-time | Weak for cross-subject gains | ✗ Not in MVP |
| **DG tricks** (no target data) | Per-subject EA on the sources, cross-subject contrastive learning, mixup across subjects, SWA | Make the base model person-agnostic before it ever meets the user | Train-time | Solid | ✅ **Core** |

## Published evidence for the test-time stack (Wimpff 2024, BCIC-IV-2a, 4-class, new subjects, no calibration)
```
source model, no adaptation ............ 57.4%
+ online alignment (EA / RA) ........... 62.5% / 63.6%
+ adaptive batch-norm .................. 65.4–65.6%
+ entropy minimisation (label-smoothed)  67.3%        (+9.9 points total, zero labels)
```
Two-class (BCIC-IV-2b): 78.8% → 83.5%.

## Proposed NeurOS stack (for decision)
```
TRAIN (offline, once):   per-subject EA on all sources → encoder → cross-subject contrastive + text-anchored head
                         (see contrastive explainer) + SWA      [ablation: + DANN or Deep CORAL]
TEST (live, every user): resting-state-seeded alignment → AdaBN → entropy-min on confident windows → output
NEW HEADSET (later):     train-time UDA with unlabeled recordings from that headset model
```
