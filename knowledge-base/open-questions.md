# Open questions (Step 2)

Questions that must be answered before the MVP architecture is locked (Step 4). Answered items move to the bottom with their answer and a link to the decision log. Plain-language background is in [`explainers/`](explainers/).

Priority: 🔴 blocking · 🟠 this week · 🟡 can wait until after the MVP.

---

## 🔴 Still open (round 2, 2026-10-01)

### Q1. Confirm the "CLAP" design
Founder intent (2026-10-01): use CLAP / contrastive learning **to increase generalizability**, delivering "UDA and CLAP for true zero-phase calibration as an MVP". Explained in [contrastive explainer](explainers/contrastive-learning-and-clap-for-generalization.md).
**Proposal:** the MVP's contrastive component = **(a) a cross-subject supervised contrastive loss** (same intent, different people → same embedding; this is the generalization engine) **+ (b) a text-anchored, CLAP-style intent head** (one shared coordinate system for all people and datasets; merges label vocabularies; commands by name).
**Also decide:** run the **CLAP-audio research track** (EEG ↔ CLAP music embeddings, cross-subject on NMED-T, which would be a first) in parallel, or defer it until after 1 Nov?

### Q2. Confirm the UDA stack
Explained in [UDA explainer](explainers/uda-train-time-vs-test-time.md). **Proposal:** test-time adaptation as the core (rest-seeded alignment → adaptive batch-norm → entropy minimisation; source-free), plus training-time domain generalisation (per-subject EA, cross-subject contrastive, SWA). DANN/Deep CORAL run as ablations only.

### Q4. Confirm the command vocabulary and headline format
Explained in [calibration/commands explainer](explainers/calibration-time-commands-and-metrics.md). **Proposal:** `LEFT` / `RIGHT` / `REST` + confidence-gated `HOLD`. Headline: "X% on N never-seen people from M setups, usable after S seconds, no calibration task", plus the adaptation curve. What would you be proud to write for X and S?

### Q5. Which headset(s) to order, and when
See [hardware explainer](explainers/demo-hardware-lsl-and-compute.md). **Proposal:** g.tec **Unicorn Hybrid Black** (primary, reliable on random people) **+ OpenBCI Ultracortex** (second headset proves "hardware-agnostic", and speaks to the OpenBCI community). **Order this week.**

### Q6b. Cluster details
What GPU (model/VRAM) and how much storage does the Proxmox cluster give us? Can we start downloading datasets there now?

### Q20 (new). Timeline lock
We have 31 days. **Proposal:** Steps 3–4 (dataset lock + architecture/MVP spec) are finished by **Sat 4 Oct**. Coding starts **Mon 6 Oct**, and dataset downloads start **now**. OK?

---

## 🟡 After the MVP
- **Q13.** Riemannian layers inside the network (SPDNet / per-person Riemannian BN). Explained in [v2 ideas](explainers/v2-ideas-riemannian-layers-pretraining-montage.md). Rec: v2.
- **Q14.** Resting-state contrastive pretraining (RUNet-style, +1–2 points). Explained in [v2 ideas](explainers/v2-ideas-riemannian-layers-pretraining-montage.md). Rec: Phase 1.5.
- **Q15.** Montage-agnostic input (graph or electrode-position embeddings). Explained in [v2 ideas](explainers/v2-ideas-riemannian-layers-pretraining-montage.md). Rec: MVP uses canonical montage + interpolation; position embeddings are the long-term universal layer.
- **Q16.** Foundation-model pretraining on TUEG. Rec: Phase 2.
- **Q17.** Privacy posture statement (on-device processing, consent, neural-data laws). Rec: one sentence in the application.
- **Q18.** Sourced market sizing. Rec: before the final application draft.
- **Q19.** IP / patents. Rec: talk to a startup-law clinic after the MVP.
- **Q7b.** Licence check on each core dataset (commercial training allowed?). Founder said skip for now; it must be done before the application cites the dataset pool.

---

## Answered (2026-10-01)

| Q | Answer | Decision |
|---|---|---|
| Q3: calibration claim | **Usable within seconds; full accuracy within 30–45 s at most; no calibration task.** Leave the door open for BCI companies to see the appeal. | [D-009](decisions/decision-log.md) |
| Q6: compute | School compute cluster via **Proxmox** (GPU details pending, see Q6b) | [D-010](decisions/decision-log.md) |
| Q7: data pool | Agree with the proposed core: PhysionetMI, Dreyer2023, Stieger2021, Lee2019_MI, Cho2017 (+ BNCI2014-001/004 for comparability) | [D-011](decisions/decision-log.md) |
| Q8: EEGMAT | **Include EEGMAT** in the dataset (brain-state paradigm: rest vs mental arithmetic) | [D-012](decisions/decision-log.md) |
| Q9: modalities | **EEG only** for the MVP; multimodal EEG + EOG + EMG later | [D-013](decisions/decision-log.md) |
| Q10: deliverable | Agree: Python package + local engine (LSL in → JSON/WebSocket intents with confidence + HOLD), dataset-playback simulator, JS client example | [D-014](decisions/decision-log.md) |
| Q11: first users | **UVA labs and OpenBCI forum developers** | [D-015](decisions/decision-log.md) |
| Q12: positioning | "A **zero-calibration layer** enabling a **generalized neural operating system** that any BCI company can use" | [D-016](decisions/decision-log.md) |
| Step 0 sentence | "UDA and CLAP for true zero-phase calibration as an MVP" | [D-003](decisions/decision-log.md) (clarified) |
| Q5 (partial) | Buying an EEG headset is fine; prefer a demo on a **random volunteer**, not a cofounder | [D-017](decisions/decision-log.md) |
