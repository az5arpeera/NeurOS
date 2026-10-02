# Decision log

Newest first. Format: **ID · date · decision** — why · who. Superseded decisions stay and are marked ~~struck~~ with a pointer.

## Decided

- **D-017 · 2026-10-01 · Buy a research-grade EEG headset for the demo; the live demo uses a random volunteer, not a cofounder.** Budget isn't a constraint. Model choice is pending ([Q5](../open-questions.md)). · Founder
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
- **D-003 · 2026-09-28 (clarified 2026-10-01) · The MVP includes UDA and CLAP-style contrastive learning, for "true zero-phase calibration"** (no calibration phase). The intent of the contrastive part is to **increase generalizability across people**. Exact design pending ([Q1](../open-questions.md), [Q2](../open-questions.md)). · Founder
- **D-002 · 2026-09-28 · Training data comes from public datasets only.** No self-collected training sessions. · Founder
- **D-001 · 2026-09-28 · Target the YC Winter 2027 batch; the 1 Nov 2026 deadline is hard.** · Founder

## Pending (see [open questions](../open-questions.md))
- P-8: adopt the L0/L1/L2 definitions of zero-calibration for all reporting ([synthesis §2](../research/synthesis.md)). The product claim would be "no calibration task; adapts in ≤ 30–45 s", with L0 (no target data) reported alongside. Proposed by Claude and consistent with D-009; **founder to confirm**.
- P-1: CLAP design. Cross-subject supervised contrastive + text-anchored head? Audio-CLAP research track now or later? (Q1)
- P-2: UDA stack. Test-time core + training-time domain generalisation? (Q2)
- P-4: command vocabulary (LEFT/RIGHT/REST + HOLD) and headline format (Q4)
- P-5: headset model(s) and order date (Q5)
- P-6b: cluster GPU/storage (Q6b)
- P-20: timeline lock. Steps 3–4 done by 4 Oct, coding from 6 Oct (Q20)
