# Decision log

Newest first. Format: **ID · date · decision** — why · who. Superseded decisions stay and are marked ~~struck~~ with a pointer.

## Decided

- **D-007 · 2026-09-28 · Research depth = primary list + one citation layer (targeted).** Two layers across 19 papers is about 10,000 papers; one targeted layer covers the decisions we face. · Founder
- **D-006 · 2026-09-28 · User testing is informal and consented (not IRB research).** · Founder
- **D-005 · 2026-09-28 · The knowledge base lives in this repo under `knowledge-base/`; git history is the version record.** Founders can edit any doc freely and recover earlier wording from history. · Founder
- **D-004 · 2026-09-28 · Restart the repo from scratch.** The prior prototype code (`DataIngest.py`, `algorithmic_alignment_layer.py`, `eegnet-classifier_validation.py`, `eventbridge_sdk.py`) and `claudeinstructions.md` were deleted and are disregarded. · Founder
- **D-003 · 2026-09-28 · The MVP must include UDA and CLAP**, aiming at "true zero calibration". The exact meaning of each is still open (see [Q1](../open-questions.md#q1-what-does-clap-mean-for-us-and-what-experience-should-the-mvp-decode-into-it), [Q2](../open-questions.md#q2-which-uda-do-we-mean), [Q3](../open-questions.md#q3-which-zero-calibration-claim-do-we-make)). · Founder
- **D-002 · 2026-09-28 · Training data comes from public datasets only** (MOABB, PhysioNet, OpenNeuro, ...). No self-collected cap sessions for training; a headset may be bought for the demo only. · Founder
- **D-001 · 2026-09-28 · Target the YC Winter 2027 batch; the application deadline of 1 Nov 2026 is hard.** · Founder

## Pending (blocking; see [open questions](../open-questions.md))
- P-1: meaning and scope of CLAP in the MVP (Q1)
- P-2: which UDA: test-time, train-time, or both (Q2)
- P-3: zero-calibration claim level, L0 vs L1 (Q3)
- P-4: intent vocabulary and headline metric (Q4)
- P-5: demo form and headset purchase (Q5)
- P-6: team roles, hours and compute (Q6)
