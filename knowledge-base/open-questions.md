# Open questions (Step 2)

Questions for the founders, each with a recommendation. (The architecture was locked on 3 Oct, D-022.)
- Answered items move to the bottom, with their answer and a link to the decision log.
- Plain-language background is in [`explainers/`](explainers/).
- The architecture draft is [`architecture/mvp-architecture.md`](architecture/mvp-architecture.md).

Priority: 🔴 blocking · 🟠 this week · 🟡 can wait until after the MVP.

---

## 🟠 This week (added 9 Oct)

### Q32. Does the name "NeurOS" survive a trademark check? (added 11 Oct) 🔴 before the application names the company
**What we found** (USPTO records read on 11 Oct through TSDR; a quick search, **not a legal clearance**):

| Mark | Owner | Status | Goods and services | How close |
|---|---|---|---|---|
| **NEUOS** ([reg. 6719026](https://tsdr.uspto.gov/statusview/sn87551964)) | Arctop, Inc. ([competitors](business/competitors.md#arctop)) | **Registered 3 May 2022, live** | **Class 9:** software for "detecting, measuring and interpreting" electrical parameters, identifying and authenticating users, and "**controlling software programs and devices**". **Class 42:** SaaS including "brain-activity-based profiling". **Class 45:** authentication services | **Highest risk.** One letter apart, sounds alike, and the same kind of goods (brain-signal software that controls devices) |
| **NEUROS** ([reg. 7235956](https://tsdr.uspto.gov/statusview/sn97657556), and stylized [reg. 7724724](https://tsdr.uspto.gov/statusview/sn97661246)) | Neuros Medical, Inc. | **Registered 5 Dec 2023 and 11 Mar 2025, live** | **Class 10:** neurostimulation medical devices (chronic pain, motor control, spasticity) | **Medium.** Identical letters in a related field (neurotechnology); different goods (a medical device, not software). The risk grows if NeurOS pitches medical or patient uses |
| NEUROS (reg. 2944404) | (digital audio players) | Cancelled 2011 | Class 9 audio players and software | None (dead) |
| NEUROS (serial 87371021) | (transdermal nerve stimulator) | Abandoned 2018 | — | None (dead) |
| **NeeuroOS** | Neeuro Pte Ltd (Singapore; an A*STAR spin-off) | No US record found; Singapore not checked | An AI platform analysing brain signals from its own headband, open to third-party developers ("NeeuroOS Agent Studio") | **Worth checking.** Close spelling and the same business: a brain-signal platform for developers |

**Why it matters now:** the YC application names the company. Renaming before the application, the domain, the SDK's package name and the first customers is cheap. Renaming after is expensive, and Arctop has the registration to object with.

**Recommendation:**
1. **Get a clearance opinion from a trademark attorney this week**: US, plus Singapore and the EU/UK if we'll sell there. Ask specifically about NEUOS (Class 9/42) and NEUROS (Class 10).
2. **In parallel, shortlist 3–5 backup names** and run the same quick check on each (USPTO, plus a domain and the package names on PyPI and npm). The code's internal package name (`neuros`) can stay until a new name is chosen.
3. **If the opinion is "risky", rename before submitting.** A YC application under a new name is normal; one that must be renamed after a demand letter isn't.

**Not legal advice.** This is a first screen of public records; only an attorney's search and opinion can clear the name.

### Q28. D-031's gate (≈ 16 Oct): does the application still lead with time-to-usable intent?
**What E2 found** ([results](results/2026-10-09-e2-warmup-curve.md)):
- The warm-up works. Two minutes of unlabeled EEG close 52–65% of the gap to full-session alignment on all three cue-free datasets.
- **The level is too low to call usable.**
  - After 120 s: 56.8% (Yang2025), 66.0% (BNCI2014-001) and 57.1% (Shin2017A, unseen hardware).
  - Only 10 of 73 cue-free people reach 70%; even full-session alignment reaches 15 of 73.
- The §8.6 hypothesis S ≤ 45 s fails: S is 120 s on Yang2025 and longer elsewhere.

**Recommendation:** don't decide on this run. By Thu 15 Oct, run the two levers that could move the ceiling, then decide on 16 Oct as planned:
1. "Use more of what we have": training crops and multi-class auxiliary training (approved 6 Oct);
2. a same-session calibrated ceiling, so "usable" has a reference.

Meanwhile, prepare the fallback in parallel: E6's rest vs mental-effort head (EEGMAT, plus Shin2017A's dataset B). If the levers don't lift cue-free accuracy at 120 s to around 65% or more on Yang2025 and Shin2017A, lead with the rigour story and the state channel.

**Update 10 Oct: lever 1 is done and doesn't meet the bar** ([results](results/2026-10-10-d039-crops-multiclass.md)).
- Crops, extra classes or both: +0.3 to +0.4 overall, within seed noise.
- At 120 s: Yang2025 56.8% in every variant; Shin2017A at best 59.9% (E2 57.1%, n.s.); BNCI2014-001 unchanged at 66.0%.
- 12–13 of 73 cue-free people reach 70% (E2 10).
- Lever 2, the same-session calibrated ceiling, measures the reference; it can't lift the number. On the bar above, this points to the fallback. The founders decide on 16 Oct.

**Update 11 Oct: the fallback's state channel failed E6** ([results](results/2026-10-11-e6-state-channel.md)).
- 70.9% on a full cap, but only 18/29 people at ≥ 60%; 56.1% on the Ganglion's channels.
- What's left to lead with: "calibration saved" (D-040) plus the rigour story (the cue finding, bars set before every run). Mental states become the stated next step, not a claim.

### Q30. Reframe the promise from "no calibration" to "calibration saved"? (added 10 Oct)
**Answered 10 Oct: the headline is adopted ([D-040](decisions/decision-log.md)).** Still open: the optional 3-minute guided calibration in the engine (proposed below).

**Why now:** the core promise is "any headset, no calibration, usable within a minute or two". After E2 and D-039 it holds only in part:
- **No calibration:** above chance for most people, but only 1 in 7 cue-free people reach 70% after 2 minutes (57–66% means).
- **Any headset:** three other lab systems give about 62%; no consumer headset has been tested, and nothing has run live.

More public data or model variants didn't move the ceiling (D-039), so the claim has to change, not the tuning.

**Proposal (Claude):** measure and claim *calibration saved*: "minute zero with NeurOS equals N minutes of calibration elsewhere", plus how well a short calibration on top of our model does. One result already points this way: a standard model calibrated on the person's earlier day scored *below* our zero-calibration model on Lee2019_MI and Stieger2021 ([follow-ups](results/2026-10-06-e1-followups-cue-test-seeds-ceilings.md)).

**Evidence needed, cheapest first:**
1. **Calibration curve:** a person's own calibrated decoder after 5/10/20/40 trials per class and half a session (`neuros calibration`, CPU, about 1 hour on AWS; code ready 10 Oct).
2. **Short calibration on top of our model:** fine-tune with 1–2 minutes of labelled data (GPU, about 1–2 hours).
3. **The founders' headset:** a cue-free self-test, the only consumer-hardware evidence we can get.
4. **Interviews:** what accuracy and setup time would people pay for?

**Update 10 Oct, evidence 1 done** ([calibration curve](results/2026-10-10-calibration-curve.md)):
- After 2 minutes of unlabeled use, our model matches about 5–20 labelled trials per hand (≈ 1.5–5 min of calibration).
- A full 10–15-minute calibration is 4–8 points better.
- Even calibrated, most people stay below 70% on these datasets.

**Next, evidence 2 (needs a go-ahead, about 1–2 GPU-hours):** our model plus 5 or 10 trials per hand of fine-tuning. **Pass bar, set before running:** on Yang2025 and BNCI2014-001, fine-tuned with 10 per hand (~3 min) at or above the half-session calibration (60.5% and 73.8%, paired, not significantly worse). If it passes, the headline is "a 15-minute calibration's accuracy in 3 minutes". If not, the claim stays "zero calibration is worth a few minutes of calibration".

**Update 10 Oct, evidence 2 done** ([results](results/2026-10-10-q30-finetune-q31-fit-check.md)): **passes the lenient reading of the bar, fails the strict one.**
- Our model plus 10 trials per hand (~3 min) is **within 0.7 points** of a 13-minute calibration on Yang2025 (59.8% vs 60.5%; 95% CI −2.7 to +1.2, p = 0.48) and **within 2.5** of a 10-minute one on BNCI2014-001 (71.4% vs 73.8%; CI −5.3 to +0.3, p = 0.30). Not significantly worse, but below.
- With 20 per hand (~5 min), Yang2025 is above it (+1.3, n.s.).
- With the same trials, it beats conventional calibration by 3–6 points (Yang2025, 10 per hand: +4.2, p = 0.001).
- **Recommended headline:** "3 minutes of calibration on top of NeurOS comes within 1–3 points of a 10–13-minute calibration; 5 minutes matches it." Not "a 15-minute calibration's accuracy in 3 minutes": the means are slightly below.
- **Recommended product step:** offer an optional ~3-minute guided calibration on top of zero calibration. It needs a cue screen in the demo and a fine-tune step in the engine; the fine-tune itself takes seconds.
- **Unchanged:** the level (about 60% on Yang2025; 7 of 41 people at 70%). Q28's gate still decides the lead.

**Also for the founders:** whether the demo should lean on a more reliable signal: mental state (E6) or SSVEP. SSVEP needs on-screen flicker and electrodes at the back of the head; check the headset first.

### Q31. A one-minute "fit check": can unlabeled EEG predict who NeurOS will work for? (added 10 Oct; bar set before running)
**Why:** even with full calibration, most people stay below 70% on our cue-free datasets. A product should tell a new user early whether motor imagery will work for them, instead of letting them struggle.

**Evidence from the literature:** the height of the resting ~10 Hz sensorimotor rhythm above the background predicted later BCI accuracy (Blankertz et al. 2010, NeuroImage; r ≈ 0.5).

**Test:** `neuros smr` (`neuros.eval.smr`). For each of the 73 cue-free development people, compute the peak height in dB:
- **Data:** the first 60 s of their first session, unlabeled.
- **Channels:** C3/C4.
- **Measure:** the 8–13 Hz peak above a 1/f line fitted at 4.5–7 and 25–28 Hz.

Then correlate it with:
- **Primary:** their zero-calibration accuracy (E2's pooled model, full-session alignment).
- **Secondary:** E2's accuracy after 120 s, and the half-session calibrated accuracy.

**Decision and pass bar (set 10 Oct, before running):**
- **Pass:** within-dataset Spearman correlation ≥ 0.4 with p < 0.01, pooled across the three datasets. Each person is ranked within their own dataset, so differences between datasets can't drive it. If it passes, the engine gets a fit check (week 3).
- **Fail:** drop the idea.
- **Cost:** CPU, minutes.

**Result 10 Oct: FAIL. The fit check is dropped** ([results](results/2026-10-10-q30-finetune-q31-fit-check.md)).
- First-minute resting rhythm vs zero-calibration accuracy: within-dataset r = +0.23 (p = 0.056), against a bar of r ≥ 0.4 with p < 0.01.
- Secondary readings reach r ≈ 0.34 (p < 0.01): a real but weak link, too weak to tell one person whether NeurOS will work for them.

### Q29. Stieger2021's sessions 3–11 (299 GB)
**Recommendation: don't download them** ([cue test](results/2026-10-06-e1-followups-cue-test-seeds-ceilings.md)).
- Stieger2021's data mostly helps its own people. Their task window doesn't beat the cue alone (68.4% vs 66.1%, p = 0.13).
- Removing all three cue-driven training sets didn't change the cue-free numbers.

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

### Note: Shin2017A's dataset B (week 3)
Shin2017A's download also holds **dataset B** (29 people, rest vs mental arithmetic, GPL-3.0). It's a candidate second dataset for testing E6's mental-state head across datasets, with EEGMAT training it.

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
  - **Progress (4 Oct), confirmed at the repositories:**
    - Dreyer2023: CC BY 4.0 (Zenodo).
    - Stieger2021: CC BY 4.0 (Figshare).
    - D-033's datasets: Yang2025 CC BY 4.0; GuttmannFlury2025 CC0; Shin2017A GPL-3.0.
  - **Still to confirm:** Cho2017 and Lee2019_MI (GigaDB, presumably CC0; the page didn't say), BNCI2014-001/004.
- **Q23.** Music-CLAP research track: cross-subject EEG ↔ CLAP audio on NMED-T. Deferred (D-018).
- **Q24.** Decoding richer actions: feet/tongue, fine arm movements, movement intention; weak labels from context. See [beyond LEFT/RIGHT](explainers/beyond-left-right-any-action.md).

---

## Answered

| Date | Q | Answer | Decision |
|---|---|---|---|
| 2026-10-10 | Q30: "calibration saved" | **Adopted.** NeurOS works from minute zero; about 3 minutes of calibration on top of it comes within 1–3 points of a 10–13-minute conventional calibration ("approximately there"). Never "the same accuracy" | [D-040](decisions/decision-log.md) |
| 2026-10-06 | Q27: Shin2017A's montage | **Option 1, corrected.** Averages of the recorded positions around each missing one, test-only. Checking the geometry showed that two-neighbour pairs land on C3h/C5h (about 19 mm off), so C3 and C4 are each the mean of their four surrounding half positions (2.6 and 1.9 mm off). The motor strip reads C3, Cz and C4 | [D-037](decisions/decision-log.md) |
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
