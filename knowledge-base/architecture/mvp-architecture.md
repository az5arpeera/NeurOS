# NeurOS MVP architecture, v1 (Step 4)

**Status:** **signed off 2026-10-03 (D-022)** · builds on decisions D-001…D-026.
- **Amended the same day:** D-024 (the command set is configurable; decoys and `HOLD("unrecognized")`), D-026 (which datasets feed the locked test set), and the Day-1 computation of C-core (26 channels, §4.3).
- **Day 2 (D-028):** the filter order, notch Q, start-up behaviour and resampling method are fixed in code (§4.2), and the CAR/EA interaction is documented (§5). The contract itself is unchanged.
- **Day 3 (D-029):** the encoder and training details are fixed in code (§6.1, §6.4), and offline EA now uses one reference per recording session (§5).
- 🔒 sections change only through the decision log. Everything else can evolve as experiments report.

This document *is* the "architectural guidelines, mathematical constraints, and data engineering patterns" that the founder's coding rule refers to: *"All generated code must be clean, heavily commented, production-grade, and include thorough error handling and data boundary checks."*

---

## 0. 🔒 What the MVP is, and what we'll claim

**The MVP** is a Python package plus a local engine. It turns any LSL EEG stream into intent events, **with no calibration task**, and adapts silently within 30–45 s (D-009).
- **Intent channel:** `LEFT`, `RIGHT`, `REST` or `HOLD`. The command set is a configuration (D-024): apps may opt in to `FEET` and `BOTH_HANDS` (experimental, E7); trained labels an app doesn't enable act as decoys and produce `HOLD("unrecognized")`.
- **Brain-state channel:** rest vs workload.
- **Deliverables (D-014):**
  - the package and engine;
  - a dataset-playback simulator (the "virtual headset");
  - a JavaScript client example.

**Proof package for YC:**
1. **Offline benchmark at scale:** ~380 people the model never saw, from 7 recording setups, under the protocol in §8.
2. **The adaptation curve:** accuracy vs seconds of normal use.
3. **Virtual headsets:** one model evaluated on Ganglion-, Unicorn-, Crown- and BCIC-IV-2b-style channel subsets.
4. **A live, unedited demo** on a random volunteer.

**Not in the MVP:**
- open-vocabulary decoding;
- whole-body action decoding ([why](../explainers/beyond-left-right-any-action.md));
- multimodal inputs (D-013);
- cloud services;
- medical claims;
- music-CLAP (D-018).

**Claim levels (P-8, sign off here):**

| Level | What the new user provides | Our use |
|---|---|---|
| L0 strict zero | Nothing | Always reported (the floor) |
| **L1 passive warm-up** | Unlabeled EEG during normal use; **no calibration task or screen** | **The product claim:** "adapts silently within 45 s" |
| L2 few-shot | A few labeled trials | Not used in the MVP; at most an upper-bound reference |

## 1. What's ours vs. what's commodity

| Commodity (we use it but don't claim it) | Ours (what we build and claim) |
|---|---|
| EEGNet, EA/RA, MOABB loaders, pyRiemann baselines | The **integrated zero-calibration runtime**: rest-seeded online alignment, test-time adaptation, and confidence/artifact gating, measured as an adaptation curve |
| Aligning EEG with label text (CLIP/CLAP; LEAF 2025) | **One intent vocabulary** across datasets, paradigms and apps (if P-1 passes E4) |
| Harmonizing public MI datasets (e.g. Hassan 2026: 5 datasets, 20-channel montage, LODO benchmark) | **Hardware profiles and virtual-headset evaluation** down to 3–4-channel layouts, plus a live engine that actually streams |
| — | **Evaluation at scale** (~380 people, LODO) and the SDK/developer experience |

## 2. System overview
```
OFFLINE
 public datasets ─► registry ─► harmonize ─► validate ─► cache (µV, 128 Hz) ─► trainer ─► model bundle ─┐
 (MOABB + EEGMAT)   (download,  (channels,   (units, NaN,                     (EA, aug,  (weights, vocab, │
                     checksums)  labels,      flat, amplitude)                 SupCon,    priors,         │
                                 montage)                                      SWA)       provenance)     │
 eval harness: splits · metrics · replay through the REAL engine ◄──────────────────────────────────────┤
                                                                                                        ▼
ONLINE (local CPU)
 headset/simulator ─LSL─► ingest ─► preprocess ─► montage ─► quality ─► online ─► encoder ─► TTA ─► head ─► policy ─► WebSocket ─► apps
                          ring       causal        map +      gate      align              AdaBN +   linear or  smooth,     JSON v0
                          buffer,    filters,      zero-fill  (HOLD     (rest-seeded,       entropy   text-      threshold,
                          gaps       resample                 reasons)  EMA + prior)        min       anchored   debounce
```

## 3. 🔒 Engineering principles (non-negotiable)
1. **Train–serve parity.**
   - One preprocessing implementation, causal and streaming, is used for training data *and* live data.
   - No zero-phase `filtfilt` offline if it can't run online.
   - A golden test asserts that the offline pipeline and the online engine produce identical windows from the same recording.
2. **Person-level isolation.**
   - No data from a test person touches training, normalisation fitting, model selection or early stopping.
   - Splits are by person, never by trial.
3. **Typed, validated boundaries.**
   - Every array carries its metadata: units, sampling rate, channel names, dataset/subject/session ids.
   - Every public function validates shapes, dtypes, finiteness and ranges.
   - On failure it raises a typed error with an actionable message.
4. **Fail safe, never silent.** The online engine never crashes its loop and never guesses. Any invalid input becomes `HOLD` with a logged reason code.
5. **Reproducible numbers.**
   - Every result row records the git SHA, config hash, data-cache hash and seed.
   - Every number in the YC application regenerates with one command.
6. **Provenance in every model bundle.** The bundle lists exactly which datasets and people trained it. If a licence requires it (Q7b), a dataset can be dropped and the model retrained.
7. **The founder's coding rule, verbatim:** "clean, heavily commented, production-grade, … thorough error handling and data boundary checks". Comments explain *why* (the math, with KB links), not just *what*.

## 4. 🔒 Data layer (Step 3 lock)

### 4.1 Sources
The training pool (D-011, D-012), each dataset's role and the download order are in [`datasets.md` → MVP v0 data lock](../datasets.md#mvp-v0-data-lock-step-3-2026-10-02).

### 4.2 Canonical signal contract

| Property | Value | Why |
|---|---|---|
| Units / dtype | **µV, float32** | MNE/MOABB return volts, and unit mistakes are the #1 silent bug. The loader asserts plausible amplitudes (median \|x\| within 0.5–200 µV). |
| Sampling rate | **128 Hz** | Motor-imagery information sits at 8–30 Hz. It's EEGNet's default, and every source (160–1000 Hz) and headset (200–256 Hz) can reach it. |
| Band | **4–30 Hz** causal IIR (Butterworth, second-order sections, state carried across chunks) + 50/60 Hz notch. **D-028:** order 4 per edge, notch Q = 30, steady-state start; ≈ 35 ms delay at 10 Hz, settled within 0.22 s (`neuros filters` prints it per rate) | The standard motor-imagery band, and consistent across datasets (EEGMAT's page reports a 30 Hz filter, to check on the data). It excludes most muscle activity, so we decode brain rhythms, not EMG. E0b checks 8–30 and 4–40. |
| Resampling | Causal and timestamp-based: filter at the native rate, then interpolate onto a 128 Hz grid using sample timestamps. **D-028:** linear interpolation; gaps ≤ 0.5 s are bridged and counted, longer ones reset the chain | Handles awkward ratios (250 → 128) and LSL clock jitter identically offline and online |
| Window | **2.0 s** (256 samples); online hop **0.125 s** (8 decisions/s) | 2 s is standard for motor imagery; the hop is configurable |
| Offline MI epochs | [cue + 0.5 s, cue + 2.5 s] | Skips the visual response to the cue itself |
| Reference | Default: common average over the canonical channels present. **E0b decides** against "as recorded". Applied per window after the channel subset is chosen (D-028); it commutes with the filter chain, so the result is exact | Datasets use different references, and this must also work for 4-channel headsets |
| Artifact features | Computed in parallel on the **native-rate broadband** signal | Muscle and motion energy live above 30 Hz |

### 4.3 Canonical montage and missing channels
- **C-core** is the intersection of the core datasets' channels within the sensorimotor region (rows F, FC, C, CP, P), computed in code on Day 1 (`neuros montage`; committed in `src/neuros/data/montages/c_core.yaml`).
  - **Computed 2026-10-03: 26 channels:** F3, Fz, F4, FC5, FC3, FC1, FC2, FC4, FC6, C5, C3, C1, Cz, C2, C4, C6, CP5, CP3, CP1, CPz, CP2, CP4, CP6, P3, Pz, P4.
  - The earlier estimate of ~18 had also intersected BNCI2014-001's 22-channel cap. BNCI2014-001 is a literature dataset, not a core one, so it enters as an 18-of-26 subset like any headset.
  - FCz is absent because Lee2019_MI doesn't record it.
  - Coverage: Ganglion 4 (C3, Cz, C4, Pz), 2b 3, Unicorn 5, Crown 4, EEGMAT 9.
- **Channel names are normalised:** case, dots and spaces (PhysioNet's `C3..`), and old 10-20 names (T3→T7, T4→T8, T5→P7, T6→P8).
- **A headset covers a subset S of C-core.**
  - Alignment runs on S only, as an |S|×|S| problem.
  - The aligned channels go into the C-core tensor, and **missing channels are zero-filled**.
  - Training uses **random channel-subset augmentation**, including Ganglion-, Unicorn-, Crown- and 2b-like subsets, so the model has seen every deployment shape.
- **No interpolation in v1.** Interpolated channels make covariances singular and invent data.
- **Headset profiles are YAML files** mapping:
  - device channel names → canonical names;
  - native sampling rate and units;
  - LSL stream type;
  - accelerometer channels.

### 4.4 Labels (canonical vocabulary)

| Canonical label | Anchor text (if P-1 passes) | Sources |
|---|---|---|
| `left_hand` | "imagine squeezing your left hand" | All MI datasets ("left fist", "left hand", "left") |
| `right_hand` | "imagine squeezing your right hand" | All MI datasets |
| `rest` | "relax with your eyes open" | Explicit rest labels where they exist (PhysionetMI T0; Stieger2021 "down", to verify); otherwise pre-cue fixation windows |
| `feet`, `both_hands` | Descriptive phrases | PhysionetMI, BNCI2014-001, Stieger2021 ("up" = both hands). **Decoys** by default: trained, and if one wins the output is `HOLD("unrecognized")`; apps may opt in to them as `FEET` / `BOTH_HANDS` (D-024) |
| `tongue` | "imagine moving your tongue" | BNCI2014-001 only (9 people). **Held out of training entirely** (D-024): never-seen activity for the false-command metric |
| `state_rest`, `state_arithmetic` | "rest quietly" / "keep subtracting the same number in your head" | EEGMAT |

- Every window records its `rest_source` (`explicit`, `pre_cue` or `baseline_run`), so E1 can test whether pre-cue fixation helps or hurts.
- The per-dataset event maps are code-reviewed config with tests: `src/neuros/data/labels.yaml`.

### 4.5 Cache and boundary checks
- **Cache** (per dataset and person):
  - Preprocessed **continuous** signals, for replay and the simulator.
  - **Windows**, for training.
  - Both are keyed by dataset version and preprocessing-config hash.
  - Metadata is stored in Parquet, with a SHA-256 manifest of the raw files (`data/manifests/<dataset>.json`, written by `neuros download`; built 3 Oct). MOABB 1.7 downloads with TLS verification off and no known hashes, so this manifest is our integrity check.
- **Checks at load:**
  - expected sampling rate;
  - known channel names;
  - plausible units;
  - no NaN/Inf;
  - no flat channels (std < 0.1 µV);
  - extreme amplitudes (peak-to-peak > 500 µV) flag the window;
  - every event code is mapped;
  - trial length ≥ window length.
- **On failure:** raise `NeurOSDataError` with dataset/subject/run context. Flagged windows are counted and reported, never silently dropped.

## 5. 🔒 Alignment math (offline and online)
For a window X ∈ ℝ^{|S|×T}, the covariance is C = XXᵀ / T.

**Offline (training, per person):** since D-029, one reference **per recording session** (the paper's protocol, and the offline twin of the online engine, which starts a new reference each session). Per person and per session differ by < 1 point on BNCI2014-001.
```
R̄  = mean of C over the person's windows
R̄  ← (1−γ)·R̄ + γ·(tr R̄ / |S|)·I                    γ = 1e-3 (shrinkage)
R̄  = U·diag(λ)·Uᵀ,  λᵢ ← max(λᵢ, ε·mean λ)          eigenvalue floor
X̃  = R̄^(−1/2)·X                                    Euclidean Alignment
```
**Reference jitter:** during training, R̄ is sometimes computed from a *random subset of 8–64 windows*. That makes the model robust to the noisy references it will see in the first seconds online.

**Online (new user, label-free):**
```
R_user(t) = running mean of C over GOOD windows (quality-gated), switching to an EMA (~60 s memory) once full
R̂(t)      = (n₀·R_pop + n_t·R_user(t)) / (n₀ + n_t)      shrinkage toward the population prior
            n_t = windows seen so far,  n₀ ≈ 10 s worth of windows
R̂^(−1/2)  recomputed once per second (an 8×8 to 18×18 eigendecomposition: microseconds)
R_pop     stored in the model bundle for each headset profile (from training people, identical preprocessing)
```
**Hypothesis H1:** seeding R_user with the first ~30 s of relaxed rest (the "settling" phase) works as well as seeding with task data. E2 tests it.

**After CAR** (rank |S|−1): R̄ has a zero eigenvalue along the all-ones direction, which the shrinkage lifts. The windows carry no energy there, so the aligned mean covariance is I − 11ᵀ/|S| rather than I (tested). Riemannian methods, which need strictly positive-definite matrices, first project onto the zero-sum subspace (D-028).

**Numerical safety:**
- symmetrise: (R + Rᵀ)/2;
- apply the eigenvalue floor;
- if the condition number exceeds 10⁶, raise γ;
- NaN/Inf anywhere → keep the previous R̂ and emit `HOLD("signal")`.

## 6. Model layer

### 6.1 Encoder
- **Architecture:** an EEGNet-family network (temporal conv → depthwise spatial conv → separable conv), then a linear map to a 64-dimensional embedding z.
  - Settings: F1 = 16, D = 2, F2 = 32, temporal kernel 64 samples (0.5 s), dropout 0.25–0.5.
  - Size: tens of thousands of parameters (< 1 MB). ShallowConvNet is the alternative (S4).
- **Input:** [batch, 26 C-core channels, 256 samples], with zeros for missing channels.
- **As built (D-029):** `neuros.models.eegnet`, EEGNet exactly as published: Keras batch-norm settings, Glorot initialisation, max-norm 1 on spatial filters and 0.25 on the head. **20,002 parameters** at 22 channels × 2 s.
- **Batch-norm design:**
  - Batches are **K people × M windows** (e.g. 8 × 32; E0 uses 8 × 8 = 64).
  - So far (E0) standard batch norm; per-person statistics come with E3b.
  - BN statistics are computed **per person within the batch** (domain-specific BN). This matches what AdaBN does at test time.
  - Ablation E3b uses standard BN instead.

### 6.2 Heads: one interface, two implementations
- **`LinearHead`** (baseline): logits = W·z + b.
- **`TextAnchoredHead`** (P-1): logits_k = cos(z, P·t_k) / τ. The text vectors t_k are pre-computed offline with a frozen sentence-embedding model and shipped as a table. See the [text-anchored head explainer](../explainers/text-anchored-head.md).
- **`ProjectionHead`** (training only, for the contrastive loss): a small MLP → 64-d, L2-normalised.

### 6.3 Losses
```
L = L_cls + λ·L_SupCon   [ + μ·L_DANN or μ·L_CORAL : side experiments S1/S2, D-019 ]
```
- **L_cls:** cross-entropy (linear head) or cosine-softmax cross-entropy (text head), with label smoothing 0.1.
- **L_SupCon with cross-person positives:** for window i, the positives P(i) are windows with the same canonical label **from a different person** in the batch.
  ```
  L_i = −(1/|P(i)|) · Σ_{p∈P(i)} log[ exp(s_ip/τ) / Σ_{a≠i} exp(s_ia/τ) ]
  ```
  This term directly teaches "same intent, different person → same embedding". λ is swept over {0, 0.1, 0.3, 1}.

### 6.4 Training recipe
- **Sampling:** subject-balanced, so each person is equally likely and classes are balanced within each person. This stops PhysionetMI's 109 people from dominating.
- **Augmentations:** channel subsets, amplitude scaling ±20%, time shifts ±0.25 s, Gaussian noise.
- **Optimisation:**
  - AdamW with a cosine schedule;
  - early stopping on **validation people** (never test people);
  - SWA over the last 25% of epochs.
- **As built (D-029):** `neuros.train`.
  - SWA averages the weights after each of the last 25% of the run's epochs. With early stopping, the run ends at the best validation epoch.
  - Batch-norm statistics are always recomputed for the final weights.
  - Inputs are divided by the RMS of the fold's training windows.
  - Augmentations, the reference jitter (§5) and SupCon come with E1–E4.

### 6.5 Model bundle
The bundle contains:
- weights (`.pt`) and config;
- the C-core channel list;
- the vocabulary and anchors;
- R_pop for each headset profile;
- decision thresholds;
- provenance: datasets, person ids, git SHA, data hash.

The runtime is PyTorch on CPU, because test-time adaptation needs gradients.

## 7. Online engine

### 7.1 Stages

| # | Stage | Key behaviour | Budget |
|---|---|---|---|
| 1 | Ingest | `pylsl` inlet (or BrainFlow → LSL). Validates stream metadata against the headset profile; 10 s ring buffer; LSL clock correction. A gap > 0.5 s resets filter state and emits `HOLD("signal_gap")`. | — |
| 2 | Preprocess | Causal notch + band-pass, timestamp resampling to 128 Hz (the same code as training) | < 1 ms |
| 3 | Montage | Profile mapping → subset S of C-core; zero-fill the rest | — |
| 4 | Quality gate | Checks each channel for flatline, saturation, peak-to-peak amplitude, high-frequency (EMG) power ratio and sudden jumps. Outputs `ok` or reason codes (`motion`, `muscle`, `flat`, `saturated`). Bad windows are **never** used for adaptation. Thresholds are tuned on Cho2017's non-task recordings, targeting < 5% false HOLD on clean data. | < 1 ms |
| 5 | Online alignment | As in §5 | < 1 ms |
| 6 | Encoder | Forward pass | ~2 ms |
| 7 | Test-time adaptation (D-019) | **AdaBN:** BN statistics update on good windows. **Entropy minimisation:** once per second, one gradient step on the BN affine parameters only, using confident windows only, with an L2 anchor to the source weights. **Watchdog:** if predictions collapse onto one class (> 95% for 60 s on clean signal) while confidence rises, roll back to the last good state (snapshots every 10 s) and halve the step size. Can be disabled by config. | ~5–15 ms, once per second |
| 8 | Head + policy | Smooths probabilities (EMA over ~0.5 s) over **all trained labels**. If the winner is a decoy (a label the app didn't enable) or the window is far from every anchor, emits `HOLD("unrecognized")` (D-024). Otherwise emits the top command if p ≥ θ_k, else `HOLD("low_confidence")`. Debounce: 2 consecutive decisions. θ_k and the distance threshold are tuned on validation people to a target precision. | < 1 ms |
| 9 | Output | WebSocket on `localhost:8765`, schema `neuros.intent.v0` | < 1 ms |

**Latency target:** < 10 ms typical and < 30 ms at p99 per decision on a laptop CPU. To be measured.

### 7.2 Engine states
`DISCONNECTED → SETTLING → ADAPTING → STEADY`, plus `DEGRADED`:
- **SETTLING** covers 0–30 s. Decisions flow from about t = 3 s and are tagged `settling`.
- **DEGRADED** is entered when the quality gate fails for more than 2 s.

"Usable within seconds" means decisions start in the first few seconds and are honestly tagged.

### 7.3 Event schema v0
```json
{
  "type": "intent", "schema": "neuros.intent.v0", "seq": 1234, "t": 5321.125,
  "command": "LEFT", "confidence": 0.82,
  "probs": {"LEFT": 0.82, "RIGHT": 0.11, "REST": 0.07},
  "hold_reason": null,
  "calibration": {"state": "adapting", "seconds_of_data": 37.5, "level": "L1"},
  "quality": {"ok": true, "bad_channels": []},
  "latency_ms": 6.1
}
```
- `t` is the LSL clock in seconds.
- When `command` is `"HOLD"`, `hold_reason` is one of `low_confidence | unrecognized | motion | muscle | flat | saturated | signal_gap | warmup | adaptation_reset`.
- `probs` lists the enabled commands; decoy probabilities appear under `"other"`.
- A separate `status` event reports engine state changes.

### 7.4 Simulator ("virtual headset")
**What it does:**
- replays any held-out person's continuous recording as an LSL stream, in real time or faster;
- cuts it to a chosen headset profile;
- optionally injects noise;
- publishes a second LSL stream of ground-truth markers.

**What it's for:**
- end-to-end tests;
- demo rehearsal before hardware arrives;
- **all warm-up and adaptation measurements** (E2/E3), so those numbers come from the exact engine code users run.

### 7.5 Developer surface
```bash
neuros download --datasets core           # MOABB + EEGMAT, with checksums
neuros prepare  --config configs/data.yaml
neuros train    --config configs/e1_pooled.yaml
neuros eval     --config configs/e1_pooled.yaml
neuros simulate --dataset PhysionetMI --subject 12 --headset ganglion
neuros run      --headset ganglion --model bundles/mvp-v0
```
JavaScript: `new NeurOSClient("ws://localhost:8765").on("intent", e => ...)` (D-014).

## 8. 🔒 Evaluation protocol (decides every number we publish)

### 8.1 Splits
- **Locked test set:**
  - 20% of people (~75), stratified by dataset.
  - Chosen by seed **on Day 1**.
  - Untouched until the freeze on Sat 24 Oct.
  - **As locked on 2026-10-03 (D-026):** `configs/splits/locked_test_v1.yaml`, seed 20261003, SHA-256-protected. 72 motor-imagery people from the five core datasets (PhysionetMI 22, Dreyer2023 17, Stieger2021 12, Lee2019_MI 11, Cho2017 10; Dreyer stratified A/B/C 12/4/1) + 7 EEGMAT people.
  - BNCI2014-001/004 are **not** in it: E0 uses all of their people during development. They're evaluated by their standard leave-one-subject-out protocol, and in the post-freeze 10-fold and LODO runs.
  - `LockedTestSet.assert_disjoint()` guards every place data enters training or tuning.
- **Development:** subject-wise 5-fold cross-validation on the other 80%, with inner validation people for early stopping and thresholds.
- **Headline runs (after the freeze):**
  1. The frozen recipe on the locked test set.
  2. The frozen recipe in 10-fold over **all ~380 people**: each person is tested once, by a model that never saw them.
  3. **Leave-one-dataset-out:** train on 6 setups and test on the 7th. This backs the "any setup" claim.
- **Literature comparability:** BNCI2014-001 and BNCI2014-004 leave-one-subject-out with the standard protocols. These go in comparison tables only.

### 8.2 Metrics
- balanced accuracy (primary);
- Cohen's κ;
- per-person distribution (median, IQR, share of people ≥ 70%);
- ITR in bits/min at the real decision rate;
- the coverage-vs-accuracy curve for HOLD;
- false-command rate during REST;
- **false commands per minute during non-command activity** (D-024): people never seen, doing activity the model never trained on (tongue imagery, held out entirely; EEGMAT mental arithmetic);
- time-to-usable: seconds until rolling left/right accuracy reaches 70%;
- latency at p50 and p99.

### 8.3 Statistics
- A paired Wilcoxon signed-rank test over people for every A-vs-B claim.
- 95% bootstrap confidence intervals over people.
- N stated with every number.
- All runs logged, including failures.

### 8.4 Baselines in every table
First numbers, BNCI2014-001 left/right, 9 people (3 Oct): [results log](../results/2026-10-03-bnci2014-001-riemannian-baselines.md). Best cross-person Riemannian result: 68.8%.

1. Chance.
2. Riemannian tangent space + logistic regression, and MDM (with and without re-centring).
3. EEGNet without alignment.
4. EEGNet + EA.
5. Ours.

### 8.5 Experiment ladder

| ID | Question | Go / no-go |
|---|---|---|
| **E0** | Does our pipeline reproduce the literature? BNCI2014-001 left/right LOSO, EA + EEGNet + SWA ≈ 74.3% (Sagayaraju 2026) | Within ±3 points; otherwise it's a pipeline bug, so stop and fix |
| E0b | Reference scheme (CAR over present channels vs as recorded) and band (4–30 / 8–30 / 4–40 Hz) | Pick on dev folds; record in the decision log |
| **E1** | Does pooling ~380 people beat single-dataset training? Includes the first LODO numbers. | If pooling doesn't help, fix harmonisation before adding heads |
| **E2** | Warm-up curve: accuracy at 0/5/10/20/30/45/60/120 s of unlabeled data. **H1:** rest-seeded vs task-seeded vs population prior vs full-session EA (the upper bound). | Sets S in the headline |
| E3 | Test-time adaptation on replay: + AdaBN, + entropy minimisation. E3b: domain-specific BN in training. | Keep each piece only if it helps on dev folds |
| E4 | Heads and contrastive: linear vs text-anchored (the P-1 rule); SupCon λ sweep. **E4-zs:** hold out "feet" entirely and ask whether the text head recognises it; report either way. | P-1 rule: text head within 1 point, or better |
| E5 | Virtual headsets: C-core, Ganglion-4, Unicorn-8, Crown-8, 2b-3 | Shows which hardware we can honestly support |
| E6 | EEGMAT: rest vs mental arithmetic across people | Viability of the backup demo |
| E7 | Command sets (D-024): default 3 commands vs opt-in 5 (+FEET, +BOTH_HANDS) on never-seen people, full C-core and Ganglion-4; decoy and distance thresholds; false commands per minute on tongue and arithmetic windows | 5-command mode goes in the demo only if it works on held-out people; reported either way |
| S1–S4 | Side experiments: DANN and Deep CORAL (D-019), RA vs EA, ShallowConvNet | Reported, but not on the critical path |

### 8.6 Headline (P-4): targets are hypotheses, not promises
The headline sentence:

> "**X%** balanced accuracy, left vs right, on **N** people the model never saw, from **M** recording setups, usable after **S** seconds of normal use with no calibration task."

It's published together with the adaptation curve, the LODO number and the virtual-headset table.

**Working hypotheses:**
- X ≈ 75% (pooled);
- LODO ≈ 65–70%;
- S ≤ 45 s to get within 2 points of full-session alignment.

Targets change only through the decision log, never silently.

## 9. Repository layout and tooling
```
NeurOS/
  pyproject.toml              # deps, ruff, mypy, pytest config
  src/neuros/
    config/                   # pydantic schemas + YAML loading
    data/                     # registry (MOABB + EEGMAT), harmonize, labels.yaml, montages/*.yaml, cache, validate
    signal/                   # causal filters, timestamp resampler, windows, artifacts (quality gate)
    align/                    # covariance (shrinkage, SPD checks), ea (offline + online with prior)
    models/                   # encoders, heads, losses, domain-specific BN / AdaBN
    train/                    # sampler (K people × M windows), augment, trainer (SWA, early stop, checkpoints)
    adapt/                    # test-time adaptation + watchdog
    engine/                   # lsl_in, pipeline (state machine), policy, server (WebSocket), simulator
    eval/                     # splits, metrics, replay through the engine, reports
    cli.py
  configs/                    # one YAML per experiment ID
  tests/                      # unit + property tests (pytest, hypothesis) + golden parity test
  clients/js/                 # minimal JS client (D-014)
  knowledge-base/
```
- **Tooling:**
  - Python ≥ 3.11 (CI tests 3.11 and 3.13; dev laptop 3.13), with `uv` and a committed lockfile (`uv.lock`);
  - **ruff** for lint and format, **mypy** for types, **pytest + hypothesis** for tests, plus pre-commit;
  - GitHub Actions runs lint and unit tests on CPU with synthetic data, in under 5 minutes.
- **Outputs:**
  - `data/`, `cache/`, `results/` and `bundles/` are git-ignored.
  - Each result is one JSON file: metrics per person plus provenance. Summary tables are generated from these.
- **Test data:** a **synthetic EEG generator** simulates mu-rhythm desynchronisation on C3/C4 with known labels.
  - It powers fast unit tests.
  - It also drives an end-to-end test: simulator → engine → WebSocket → the client receives the right intents.
- **Dependencies:** all permissive licences, fine for a commercial SDK.
  - numpy, scipy, mne, moabb, pyriemann, torch;
  - pylsl, brainflow, websockets;
  - pydantic, h5py, pyarrow;
  - sentence-transformers (only for building anchors offline).

## 10. Privacy and safety (MVP level)
- **Local only.** All processing is local, there's no network egress by default, and the WebSocket binds to localhost.
- **Recording.** Session recording is off by default. It's on only with written consent, and no names are stored (D-006).
- **Participants.** Live tests are with adults only.

## 11. Open items that do **not** block Saturday

| Item | Default until decided | Decided by |
|---|---|---|
| ~~P-1 text-anchored head~~ | Adopted under the rule (D-023); E4 decides | Decided 3 Oct |
| ~~P-4 commands~~ | Configurable command set (D-024) | Decided 3 Oct |
| ~~P-5 headset~~ | Ganglion + gold cups, ordered (D-025); simulator until it arrives | Decided 3 Oct |
| Reference scheme, band | CAR over present channels, 4–30 Hz | E0b |
| SSVEP demo insurance | Not built | Founder, week 3 |
