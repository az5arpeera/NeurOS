# Open questions (Step 2)

Questions for the founders, each with a recommendation. (The architecture was locked on 3 Oct, D-022.)
- Answered items move to the bottom, with their answer and a link to the decision log.
- Plain-language background is in [`explainers/`](explainers/).
- The architecture draft is [`architecture/mvp-architecture.md`](architecture/mvp-architecture.md).

Priority: 🔴 blocking · 🟠 this week · 🟡 can wait until after the MVP.

---

## 🟠 Still open (2026-10-03): coding has started; these don't block it

### Q6b. Cluster details (fallback since D-032: training moves to AWS credits)
**AWS checklist (founders; Claude never handles credentials):** step-by-step in the [AWS runbook](plan/aws-runbook.md) (4 Oct).
- Request the GPU quota now: 16 vCPUs of "G and VT" instances. New accounts often start at 0.
- Set a budget alarm.
- Launch a g6.4xlarge in us-east-1 with a 500 GB data volume.
- Run `scripts/aws/bootstrap.sh`.
- Approve the downloads in step 5 (≈ 282 GB).

If the school cluster is still wanted as a fallback, its details are:
- GPU (≥ 8 GB VRAM is plenty);
- storage (~1 TB);
- SSH;
- Docker/conda.

### Q22. Role split
Founder (3 Oct): "TBD on specific roles". **Proposal** (in the [plan](plan/road-to-nov-1.md)):
- **Founder:** data pipeline, engine/SDK, evaluation harness, outreach, the YC application.
- **Cofounder:** model training and experiments.

### Q27. Shin2017A's montage (raised 4 Oct, D-033)
**Why:** Shin2017A's cap samples the motor strip at 10-5 half positions (FCC3h, CCP3h, …) and has no C3 or C4.
- The model's input (C-core) shares only 6 of its 26 positions with it: F3, F4, Cz, P3, Pz and P4.
- As things stand, the model would read it like a 6-channel headset *without* the channels that carry left vs right. A test on it would say little.

**Options:**
1. **Two-neighbour averages onto C-core positions (recommended).**
   - C3 ≈ mean(FCC3h, CCP3h) and C5 ≈ mean(FCC5h, CCP5h); C4 and C6 likewise. Each pair sits half a step in front of and behind its target.
   - Test-only, recorded as a montage profile, and never used in training.
   - It replaces 8 channels with 4, so covariances stay full rank. (The worry about interpolation in §4.3 is rank deficiency, when an interpolated channel sits next to its own sources.)
2. **Nearest-position relabelling** (FCC3h → C3). Simpler, but it silently shifts each electrode by about 1.5 cm.
3. **Leave it.** Use Shin2017A only in internal diagnostics (the cue audit), not as a test of the model.

**Rec:** 1. It's an exception to "no interpolation in v1" (§4.3), so it needs your decision.

**Related, for week 3:** Shin2017A's download also holds **dataset B** (29 people, rest vs mental arithmetic, GPL-3.0). It's a candidate second dataset for testing E6's mental-state head across datasets, with EEGMAT training it.

### Optional: email g.tec
Ask about a student/startup price for the Unicorn (a second headset for the hardware-agnostic story).

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
| 2026-10-04 | Q26: add cue-free datasets | **Approved, all three.** Yang2025 trains and tests (cue-free); Shin2017A tests only (GPL-3.0); GuttmannFlury2025 is a gaze diagnostic. The locked test set grows to v2 (+22 people, v1 unchanged). Brandl2020, HefmiIch2025 and Chang2025 stay out (non-commercial licences) | [D-033](decisions/decision-log.md) |
| 2026-10-04 | Q25: cue-controlled evaluation windows | **Adopted.** Every motor-imagery result uses windows that start after the class has left the screen (train and test), next to a cue-only control; datasets whose cue stays up carry a flag. Prompted by E0: 75.0% with the arrow on screen, 60.6% after it | [D-030](decisions/decision-log.md) |
| 2026-10-03 | Q21: architecture sign-off | **Signed off**, including the L0/L1/L2 claim levels and the evaluation protocol | [D-022](decisions/decision-log.md) |
| 2026-10-03 | Q1: text-anchored head | **Adopt under the rule** (ship it if within 1 point of the linear head, or better, on never-seen people) | [D-023](decisions/decision-log.md) |
| 2026-10-03 | Q4: commands | **`LEFT / RIGHT / REST + HOLD`, as a configurable command set**: labels the app doesn't enable act as decoys → `HOLD("unrecognized")`; FEET / BOTH_HANDS as an experimental opt-in mode | [D-024](decisions/decision-log.md) |
| 2026-10-03 | Q5: headset | **OpenBCI Ganglion + gold cups + Ten20** (≈ $715), ordered; electrodes at C3, Cz, C4, Pz | [D-025](decisions/decision-log.md) |
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
