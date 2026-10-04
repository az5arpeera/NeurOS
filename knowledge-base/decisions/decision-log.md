# Decision log

Newest first. Format: **ID · date · decision** — why · who. Superseded decisions stay and are marked ~~struck~~ with a pointer.

## Decided

- **D-029 · 2026-10-03 · Deep-model and training details, fixed in code (Day 3).** These fill in what architecture §6.1, §6.4 and E0 left open:
  - **Encoder:** EEGNet exactly as published (Lawhern 2018).
    - Keras's batch-norm settings (running statistics move 1% per step, ε = 1e-3), Glorot initialisation, no convolution biases.
    - Max-norm caps: 1 on each spatial filter, 0.25 on each class of the head.
    - F1/D/F2 = 16/2/32 with a 64-sample temporal kernel (§6.1), then our 64-d embedding and the linear head.
    - **20,002 parameters** for 22 channels × 2 s.
  - **Training recipe:**
    - batches of 8 people × 8 windows, class-balanced within each person;
    - AdamW (lr 1e-3, weight decay 0.05); the learning rate follows a cosine down to 0.
    - **SWA** = the plain average of the weights after each of the last 25% of the run's epochs.
    - **Early stopping** (optional, validation people only): the run then ends at the best validation epoch, and SWA averages the last quarter before it.
    - **Batch-norm statistics are always recomputed for the final weights** (one shuffled pass over the training windows). Averaged weights need their own statistics, and running statistics that move 1% per step lag the weights.
  - **Input scale:** the inputs of each fold are divided by one number, the RMS of its training windows (training people only, principle 2).
  - **EA reference: one per recording session** (it was one per person on Day 2).
    - It's the paper's protocol, and the offline twin of the online engine, which starts a new reference every time the headset goes on.
    - Measured on BNCI2014-001 with the Riemannian baselines: per session and per person differ by **< 1 point**, so the Day-2 numbers stand.
    - Amends §5's "per person" (🔒).
  - **E0 has a second config with the paper's 4-s window** (cue + 0–4 s), besides our 2-s contract window (cue + 0.5–2.5 s), so a shortfall could be traced to the window (a protocol choice) or to a bug. The 2-s run passed (75.0%), so the 4-s run is optional.
  - **Tooling:**
    - CI installs PyTorch's CPU build on Linux: 196 MB instead of 3.3 GB of CUDA wheels.
    - Folds train in parallel processes: 9 × 2 threads is ≈ 2× faster than 1 × 16 on the laptop.
    - Every result records its thread count, because parallel sums can change the last bits.
  - · Claude (implementation within the signed-off architecture; founder may veto)
- **D-028 · 2026-10-03 · Signal-chain details, fixed in code (Day 2).** These fill in what architecture §4.2 and §5 left open, without changing the contract:
  - **Band-pass:** Butterworth **order 4 per edge** (8 poles). Measured at every native rate in scope (160–1000 Hz): ≈ 35 ms delay at 10 Hz, 99% settled within 0.22 s, ≥ 30 dB down at 64 Hz. Order 2 would halve the delay but leave only 15–29 dB at 64 Hz, and the band-pass is also the resampler's anti-alias filter.
  - **Notches:** 50 and 60 Hz, **Q = 30**.
  - **Start-up:** the filter starts in its steady state for the first sample, so DC offsets don't ring. Windows starting within **0.5 s** of a (re)start are flagged.
  - **Resampling:** **linear interpolation on timestamps** (≤ 1 dB loss at 30 Hz; identical offline and online).
  - **CAR:** applied **per window, after the channel subset is chosen** (exact, because CAR commutes with the linear chain).
  - **Riemannian baselines:** OAS covariances; with CAR, a projection onto the zero-sum subspace (covariances must be positive definite).
  - **Evidence:** offline and streaming outputs are bit-identical for any chunking (the golden parity test).
  - · Claude (implementation within the signed-off contract; founder may veto)
- **D-027 · 2026-10-03 · PhysionetMI people 88, 92 and 100 are excluded for data quality.** Their files are recorded at 128 Hz while the dataset is 160 Hz, and person 100 has only 6 trials per class per run; validation fails them (`sampling_rate`). Excluding them is common practice for this dataset. None of the three is in the locked test set; the PhysionetMI development pool shrinks from 87 to 84 people. **Usable core pool: 361 people** (379 including the BNCI datasets). Recorded in the registry's `excluded_subjects`, with reasons. · Claude (validation result; founder may veto)
- **D-026 · 2026-10-03 · The locked test set covers the 5 core motor-imagery datasets + EEGMAT, not BNCI2014-001/004.** E0 (the pipeline sanity check) uses all 9 people of BNCI2014-001 during development, so they can't also be pristine test people. The BNCI datasets are evaluated with their standard leave-one-subject-out protocol (comparison tables), and still appear in the post-freeze 10-fold and LODO runs. Locked test = round(20%) of each dataset's people, seed `20261003` (the lock date, chosen before any data was examined): **72 motor-imagery people + 7 EEGMAT people**, written to `configs/splits/locked_test_v1.yaml`. This clarifies architecture §8.1 (still "~75 people, stratified by dataset"). · Claude (protocol clarification; founder may veto)
- **D-025 · 2026-10-03 · Headset: OpenBCI Ganglion + dongle + gold-cup electrodes + Ten20 paste** (≈ $715, plus ≈ $30 of skin-prep gel, tape measure and an elastic net cap). Ordered by the founder on 3 Oct. Electrode plan: **C3, Cz, C4, Pz**, with reference and ground on the earlobes. Rejected: OpenBCI's EEG Electrode Cap Kit ($999.99 for the cap alone; ≈ $1,710 with a Ganglion, over the D-020 cap); the Electro-Cap 19-channel ($455.95, but it appears to need a 25-pin breakout wired to the Ganglion); the Ultracortex (dry electrodes, 3D printing). The board carries over if a cap is added later ($39.99 touch-proof adapter). Resolves P-5. · Founder
- **D-024 · 2026-10-03 · Commands: `LEFT / RIGHT / REST + HOLD`, and the command set is a configuration.** Every trained intent label keeps its anchor at run time. The app chooses which labels are commands (default: LEFT, RIGHT, REST). A trained label the app didn't enable acts as a **decoy**: if it wins, the output is `HOLD("unrecognized")`; a window far from every anchor gets the same. FEET and BOTH_HANDS are evaluated as an experimental opt-in 5-command mode (E7), reported either way. **Tongue imagery is held out of training entirely**; with EEGMAT mental arithmetic it serves as never-seen activity for a new metric, false commands per minute during non-command activity. (This refines the chat proposal that called tongue a decoy: 9 people are too few to learn from, and more useful as an unseen test.) Mental arithmetic stays on the brain-state channel. Resolves P-4. · Founder (accepted recommendation)
- **D-023 · 2026-10-03 · Text-anchored head adopted under the rule.** Build both heads; ship the text-anchored head if it scores within 1 point of the linear head, or better, on never-seen people (E4). Otherwise ship the linear head and keep the vocabulary table for naming. Resolves P-1. · Founder
- **D-022 · 2026-10-03 · Architecture v1 signed off**: all 🔒 sections, including the L0/L1/L2 claim levels (P-8) and the evaluation protocol with the locked 20% test set. It is now the coding guideline. Resolves P-21 and P-8. · Founder
- **D-021 · 2026-10-02 · Coding starts Sat 3 Oct 2026.** Steps 3–4 were drafted on Fri 2 Oct: the [data lock](../datasets.md#mvp-v0-data-lock-step-3-2026-10-02), [architecture v1](../architecture/mvp-architecture.md) and the [30-day plan](../plan/road-to-nov-1.md). Only the 🔒 sections of the architecture need sign-off before coding. This supersedes the P-20 proposal ("Steps 3–4 by Sat 4 Oct, coding Mon 6 Oct"), whose weekdays were wrong: in 2026, 3 Oct is a Saturday. · Founder
- **D-020 · 2026-10-02 · Headset budget cap: about $1,000–1,200.** Supersedes D-017's "budget isn't a constraint". Model choice is pending (P-5); see the [hardware explainer](../explainers/demo-hardware-lsl-and-compute.md). · Founder
- **D-019 · 2026-10-02 · UDA stack: test-time adaptation is the core; DANN and Deep CORAL run as side experiments.** The core is rest-seeded online alignment → adaptive batch-norm → entropy minimisation, all source-free, plus training-time domain generalisation (per-subject EA, cross-subject contrastive loss, SWA). Resolves P-2. · Founder
- **D-018 · 2026-10-02 · The music-CLAP research track is deferred until after 1 Nov.** All resources go to the MVP, business value / market fit, and the YC application. · Founder
- **D-017 · 2026-10-01 · Buy a research-grade EEG headset for the demo; the live demo uses a random volunteer, not a cofounder.** ~~Budget isn't a constraint~~ (superseded by D-020). Model choice is pending ([Q5](../open-questions.md)). · Founder
- **D-016 · 2026-10-01 · Positioning: "a zero-calibration layer enabling a generalized neural operating system that any BCI company can use."** · Founder
- **D-015 · 2026-10-01 · First users and outreach: UVA labs and OpenBCI forum developers.** · Founder
- **D-014 · 2026-10-01 · MVP deliverable:** a Python package + local engine (LSL in → intent events as JSON over WebSocket, with confidence and HOLD), a dataset-playback "virtual headset" simulator, and a JavaScript client example. · Founder (accepted recommendation)
- **D-013 · 2026-10-01 · EEG-only model input for the MVP**; multimodal EEG + EOG + EMG is a later module. · Founder
- **D-012 · 2026-10-01 · Include PhysioNet EEGMAT** (rest vs mental arithmetic, 36 subjects) as a brain-state paradigm alongside motor imagery. · Founder
- **D-011 · 2026-10-01 · Core motor imagery training pool: PhysionetMI, Dreyer2023, Stieger2021, Lee2019_MI, Cho2017**, plus BNCI2014-001/004 for literature comparability (~380 subjects). Licence check still to do. · Founder
- **D-010 · 2026-10-01 · Compute: the school cluster via Proxmox.** GPU/storage details pending. · Founder
- **D-009 · 2026-10-01 · Calibration target: usable within seconds, full accuracy within 30–45 s at most, with no calibration task.** To be validated by the warm-up-curve experiment (H1: a resting-state-seeded reference). · Founder
- **D-007 · 2026-09-28 · Research depth = primary list + one citation layer (targeted).** · Founder
- **D-006 · 2026-09-28 · User testing is informal and consented (not IRB research).** · Founder
- **D-005 · 2026-09-28 · The knowledge base lives in this repo under `knowledge-base/`; git history is the version record.** · Founder
- **D-004 · 2026-09-28 · Restart the repo from scratch**; the prior prototype code and `claudeinstructions.md` are discarded. · Founder
- **D-003 · 2026-09-28 (clarified 2026-10-01) · The MVP includes UDA and CLAP-style contrastive learning, for "true zero-phase calibration"** (no calibration phase). The intent of the contrastive part is to **increase generalizability across people**. The UDA side is decided (D-019); the CLAP head design is pending ([Q1](../open-questions.md)). · Founder
- **D-002 · 2026-09-28 · Training data comes from public datasets only.** No self-collected training sessions. · Founder
- **D-001 · 2026-09-28 · Target the YC Winter 2027 batch; the 1 Nov 2026 deadline is hard.** · Founder

## Pending (see [open questions](../open-questions.md))
- P-6b: cluster GPU/storage. Founder is working on it. (Q6b)
- P-22: role split. Founder (3 Oct): "TBD on specific roles". Proposal in the [plan](../plan/road-to-nov-1.md). (Q22)
- Resolved on 3 Oct: P-21 and P-8 → D-022; P-1 → D-023; P-4 → D-024; P-5 → D-025.
