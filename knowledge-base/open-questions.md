# Open questions (Step 2)

Questions that must be answered before the MVP architecture is locked (Step 4).
- Answered items move to the bottom, with their answer and a link to the decision log.
- Plain-language background is in [`explainers/`](explainers/).
- The architecture draft is [`architecture/mvp-architecture.md`](architecture/mvp-architecture.md).

Priority: 🔴 blocking · 🟠 this week · 🟡 can wait until after the MVP.

---

## 🔴 Still open (round 3, 2026-10-02): needed for Saturday's coding start

### Q21. Sign off architecture v1
Read the 🔒 sections of [architecture v1](architecture/mvp-architecture.md):
- §0: what we claim, including the **L0/L1/L2 claim levels** (P-8);
- §3: engineering principles;
- §4: data contract;
- §5: alignment math;
- §8: evaluation protocol, including the locked 20% test set.

Edit anything you disagree with. Everything else can evolve.

### Q1. Text-anchored head: adopt it under the rule?
Explained in the [text-anchored head explainer](explainers/text-anchored-head.md).

**Proposal:**
- Build both heads.
- Adopt the text-anchored head as the MVP's CLAP layer **if it scores within 1 point of the linear head, or better, on never-seen people** (experiment E4).
- Cross-subject supervised contrastive is the generalization engine either way.

The music-CLAP track is deferred (D-018).

### Q4. Confirm the command vocabulary
**Proposal:** `LEFT / RIGHT / REST` + `HOLD`.
- `HOLD` now also fires on detected movement or muscle artifacts.
- Whole-body actions (walk, punch) are out of scope; see [beyond LEFT/RIGHT](explainers/beyond-left-right-any-action.md).
- Headline targets are working hypotheses in [architecture §8.6](architecture/mvp-architecture.md).

### Q5. Pick the headset within the $1.2k cap, and order today
See the [hardware explainer](explainers/demo-hardware-lsl-and-compute.md).
- **Recommended:** OpenBCI Ganglion kit (≈ $715: wet electrodes, C3/Cz/C4, BrainFlow/LSL, our first-user community).
- **Alternative:** BrainAccess MINI (€900 excl. VAT; 8 dry channels).
- **Optional upside:** email g.tec about a student/startup price for the Unicorn.

### Q6b. Cluster details (founder working on it)
- GPU (≥ 8 GB VRAM is plenty);
- storage (~1 TB);
- SSH;
- Docker/conda.

### Q22. Role split
**Proposal** (in the [plan](plan/road-to-nov-1.md)):
- **Founder:** data pipeline, engine/SDK, evaluation harness, outreach, the YC application.
- **Cofounder:** model training and experiments.

---

## 🟡 After the MVP
- **Q13.** Riemannian layers inside the network (SPDNet / per-person Riemannian BN). Explained in [v2 ideas](explainers/v2-ideas-riemannian-layers-pretraining-montage.md). Rec: v2.
- **Q14.** Resting-state contrastive pretraining (RUNet-style, +1–2 points). Explained in [v2 ideas](explainers/v2-ideas-riemannian-layers-pretraining-montage.md). Rec: Phase 1.5.
- **Q15.** Montage-agnostic input (graph or electrode-position embeddings). Explained in [v2 ideas](explainers/v2-ideas-riemannian-layers-pretraining-montage.md). The MVP uses a canonical channel set with zero-fill + channel-subset training ([architecture §4.3](architecture/mvp-architecture.md)); position embeddings are the long-term universal layer.
- **Q16.** Foundation-model pretraining on TUEG. Rec: Phase 2.
- **Q17.** Privacy posture statement (on-device processing, consent, neural-data laws). Rec: one sentence in the application.
- **Q18.** Sourced market sizing. Rec: before the final application draft.
- **Q19.** IP / patents. Rec: talk to a startup-law clinic after the MVP.
- **Q7b.** Licence check on each core dataset (is commercial training allowed?). The founder said skip for now. It must be done before the application cites the dataset pool. Model bundles record their training datasets, so a dataset can be dropped.
- **Q23.** Music-CLAP research track: cross-subject EEG ↔ CLAP audio on NMED-T. Deferred (D-018).
- **Q24.** Decoding richer actions: feet/tongue, fine arm movements, movement intention; weak labels from context. See [beyond LEFT/RIGHT](explainers/beyond-left-right-any-action.md).

---

## Answered

| Date | Q | Answer | Decision |
|---|---|---|---|
| 2026-10-02 | Q2: UDA stack | **Test-time adaptation is the core** (rest-seeded alignment → AdaBN → entropy minimisation); **DANN / Deep CORAL are side experiments** | [D-019](decisions/decision-log.md) |
| 2026-10-02 | Q1 (part): music-CLAP track | **Deferred until after 1 Nov.** All resources go to the MVP, business value / market fit, and applying | [D-018](decisions/decision-log.md) |
| 2026-10-02 | Q5 (part): budget | **$1,000–1,200 max** for the headset | [D-020](decisions/decision-log.md) |
| 2026-10-02 | Q20: timeline | **Start coding Sat 3 Oct.** Steps 3–4 drafted 2 Oct | [D-021](decisions/decision-log.md) |
| 2026-10-01 | Q3: calibration claim | **Usable within seconds; full accuracy within 30–45 s at most; no calibration task.** Leave the door open for BCI companies to see the appeal. | [D-009](decisions/decision-log.md) |
| 2026-10-01 | Q6: compute | School compute cluster via **Proxmox** (details in Q6b) | [D-010](decisions/decision-log.md) |
| 2026-10-01 | Q7: data pool | Agree with the proposed core: PhysionetMI, Dreyer2023, Stieger2021, Lee2019_MI, Cho2017 (+ BNCI2014-001/004 for comparability) | [D-011](decisions/decision-log.md) |
| 2026-10-01 | Q8: EEGMAT | **Include EEGMAT** (brain-state paradigm: rest vs mental arithmetic) | [D-012](decisions/decision-log.md) |
| 2026-10-01 | Q9: modalities | **EEG only** for the MVP; multimodal EEG + EOG + EMG later | [D-013](decisions/decision-log.md) |
| 2026-10-01 | Q10: deliverable | Python package + local engine (LSL in → JSON/WebSocket intents with confidence + HOLD), dataset-playback simulator, JS client example | [D-014](decisions/decision-log.md) |
| 2026-10-01 | Q11: first users | **UVA labs and OpenBCI forum developers** | [D-015](decisions/decision-log.md) |
| 2026-10-01 | Q12: positioning | "A **zero-calibration layer** enabling a **generalized neural operating system** that any BCI company can use" | [D-016](decisions/decision-log.md) |
| 2026-10-01 | Step 0 sentence | "UDA and CLAP for true zero-phase calibration as an MVP" | [D-003](decisions/decision-log.md) (clarified) |
| 2026-10-01 | Q5 (part) | Buy an EEG headset; demo on a **random volunteer**, not a cofounder | [D-017](decisions/decision-log.md) |
