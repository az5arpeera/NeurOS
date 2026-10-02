# Decision log

Newest first. Format: **ID · date · decision** — why · who. Superseded decisions stay and are marked ~~struck~~ with a pointer.

## Decided

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
- P-21: **sign off architecture v1**, specifically the 🔒 sections ([architecture](../architecture/mvp-architecture.md)). This now includes P-8 (§0 claim levels) and the evaluation protocol (§8). (Q21)
- P-8: adopt the L0/L1/L2 definitions of zero-calibration for all reporting. The product claim would be "no calibration task; adapts within 30–45 s", with L0 (no target data) always reported alongside. Folded into P-21.
- P-1: text-anchored head. Proposed rule: adopt it as the MVP's CLAP layer if it scores within 1 point of the linear head, or better, on never-seen people (experiment E4). Cross-subject supervised contrastive is the generalization engine either way. Explained in the [text-anchored head explainer](../explainers/text-anchored-head.md). (Q1)
- P-4: command vocabulary (`LEFT / RIGHT / REST + HOLD`, plus a motion/artifact HOLD) and headline format. Whole-body actions are out of scope; see [beyond LEFT/RIGHT](../explainers/beyond-left-right-any-action.md). (Q4)
- P-5: headset within the D-020 cap. Recommended: OpenBCI Ganglion kit (≈ $715, wet electrodes, C3/Cz/C4). Order today. (Q5)
- P-6b: cluster GPU/storage. Founder is working on it. (Q6b)
- P-22: role split (proposal in the [plan](../plan/road-to-nov-1.md)). (Q22)
