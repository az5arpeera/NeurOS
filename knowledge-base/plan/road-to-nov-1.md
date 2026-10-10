# Road to 1 November 2026

**v1 · 2026-10-02 (Fri).**
- **Deadline:** the YC W27 application is due **Sun 1 Nov 2026** (D-001). We aim to submit by **Fri 30 Oct**, keeping the weekend as buffer. Check YC's posted deadline time.
- **Coding starts Sat 3 Oct** (D-021).
- **All resources go to three tracks** (D-018): the MVP, business value / market fit, and the application.

> Weekday check: in 2026, 3 Oct is a **Saturday** and 1 Nov is a **Sunday**. An earlier draft (P-20) said "Sat 4 Oct / Mon 6 Oct"; those weekdays were wrong.

## Proposed role split (confirm, P-22)
| Track | Owner | Why |
|---|---|---|
| Data pipeline, engine/SDK, evaluation harness | Founder (Azhar) | Systems + math; owns train–serve parity and the protocol |
| Model training and experiments (E0–E6, S1–S4) | Cofounder | NN research background |
| Customer discovery, YC application | Founder (lead), cofounder reviews | One voice for the narrative |

Claude Code pairs on all engineering. **Founders send all outreach;** Claude drafts.

## Fri 2 – Sat 3 Oct: decisions
- [x] **Sign off** [architecture v1](../architecture/mvp-architecture.md) (D-022).
- [x] **Decide** P-1 (D-023: text-anchored head under the rule) and P-4 (D-024: configurable command set).
- [x] **Order the headset** (D-025): OpenBCI Ganglion + gold cups + Ten20, ≈ $715.
- [ ] **Email g.tec** about a student/startup price for the Unicorn (optional upside).
- [ ] **Ask the cluster admins** about GPU, storage (~1 TB) and SSH (Q6b).
- [x] **Start downloads**: BNCI2014-001/004 and PhysionetMI on 3 Oct (≈ 3.1 GB).
- [ ] **Roles** (P-22): founder says TBD.

### Sat 3 Oct, done
- Package scaffold: `pyproject.toml` + `uv.lock`, ruff, strict mypy, pytest + hypothesis, CI (3.11 and 3.13), pre-commit.
- Data layer: typed errors, channel normaliser, vocabulary + event maps (`labels.yaml`), dataset registry (cross-checked with MOABB), harmonisation, validation (§4.5), headset profiles, synthetic EEG generator, provenance stamps, download + SHA-256 manifest, `neuros` CLI.
- **C-core computed: 26 channels.** **Locked test set written** (79 people, D-026).
- Real data (≈ 3.2 GB, SHA-256 manifests): BNCI2014-001 108/108 recordings pass; BNCI2014-004 45/45; PhysionetMI 106/109 people (88, 92, 100 recorded at 128 Hz → excluded, D-027), every file verified against PhysioNet's published checksums.
- Business: [customer-interview script](../business/customer-interview-script.md) drafted.

### Day 2 (planned Sun 4 Oct), done Sat 3 Oct evening
- **Signal chain** (`src/neuros/signal/`), with details fixed in D-028:
  - causal notch + band-pass with state carried across chunks;
  - timestamp resampler to 128 Hz (bridges short gaps, raises on long ones);
  - common average reference;
  - windows: offline epochs, sliding windows and the streaming window buffer, with quality flags.
- **Golden parity test:** offline preprocessing and chunked streaming give **bit-identical** samples and windows (architecture §3, principle 1).
- **EA** (`src/neuros/align/`): covariances with the §5 safety rules (shrinkage, eigenvalue floor, condition guard).
  - Acceptance test passes: after alignment, each person's mean covariance = I.
  - Exact without shrinkage, within 1e-2 with it. After CAR it is I − 11ᵀ/n, as expected.
- **Riemannian baselines on BNCI2014-001** (`neuros baseline`). Cross-person left/right **65–69%**; within-person sanity check 73–77%. See the [results log](../results/2026-10-03-bnci2014-001-riemannian-baselines.md).
- **Fixed:** a Day-1 encoding bug in `validate.py` ("§" and "µ" were double-encoded, also in user-facing error messages). A new test checks every text file in the repo for clean UTF-8.
- **Tests:** 422 pass (9 more need downloaded data, and pass with it).
- **Business:** [15 target contacts](../business/target-contacts.md), across UVA labs, developer communities and BCI companies, with message drafts. ⏰ g.tec's virtual BR41N.IO hackathon runs **Sun 4 – Mon 5 Oct**: the founders decide whether to join.

### Day 3 (planned Mon 5 – Tue 6 Oct), done Sat 3 Oct night
- **PyTorch** 2.14.1 (CPU) on the laptop. CI installs PyTorch's CPU build on Linux: 196 MB instead of 3.3 GB of CUDA wheels.
- **The model layer** (details fixed in D-029):
  - **EEGNet encoder** + 64-d embedding + linear head (`src/neuros/models/`): 20,002 parameters, built to EEGNet's published settings.
  - **Trainer** (`src/neuros/train/`): subject-balanced batches (8 people × 8 windows), AdamW with a cosine learning rate, optional early stopping on validation people, SWA over the last 25% of epochs, NaN/Inf losses stop the run.
  - **Leave-one-person-out runner** (`neuros loso --experiment configs/e0.yaml`): the 9 people's folds train in parallel (≈ 50 min per 300-epoch run on the laptop) and results carry full provenance.
- **EA per recording session** (D-029): the paper's protocol, and what the live engine does. On the Riemannian baselines it changes results by < 1 point.
- **Tests:** 539 pass (CI green on 3.11 and 3.13, PyTorch tests included).
- **✅ E0's pipeline check passes:** EEGNet + EA **75.0%** against the literature's 74.3% ± 3, on our own causal 2-s pipeline (9 people, one seed). See the [E0 results log](../results/2026-10-03-e0-eegnet-bnci2014-001.md).
  - EEGNet without alignment: 72.1%. EA adds 2.9 points (better for 8 of 9 people; p = 0.074).
  - On this window the deep model beat the best Riemannian baseline by 8.1 points (p = 0.039). It doesn't survive the cue control below.
- **⚠️ 4 Oct cue diagnostics: the 75.0% is not a motor-imagery number.**
  - The paper's 4-s window gave 83.7%, but **the arrow alone (its first 0.5 s) decodes at 75.8%**.
  - **After the arrow has gone** ([1.5, 3.5] s): zero-calibration EEGNet scores **60.6%**, level with TS+LR + EA (60.4%) and **11.6 points below a calibrated model** (72.2%, p = 0.012).
  - Before the cue, every model scores at chance: nothing leaks.
  - Retracted: the 83.7%, the deep model's 8.1-point lead, and "zero-calibration on par with calibrated".
  - **Adopted ([D-030](../decisions/decision-log.md)):** every motor-imagery result uses windows that start after the class has left the screen, next to a cue-only control. Built in on 4 Oct: the registry declares how long each dataset shows the class, `neuros datasets` prints each window, and every result file records its window and a cue-free flag. **Only BNCI2014-001 has a cue-free window;** all five core datasets keep the class on screen for the whole task.
  - Without the arrow's first half-second, the 4-s window's 83.7% falls to 72.3% ([0.5, 4.0] s), no better than our 2-s window: its gain was the cue's onset, not the window's length.
- **4 Oct evening (D-031: intent first, a headline led by adaptation, one mental state; D-032: AWS):**
  - `neuros audit` measures when and where on the head each cued dataset's class is readable. BNCI2014-001, BNCI2014-004 and PhysionetMI: no leaks; the motor strip carries the class in the training windows; PhysionetMI's eyes still decode it (56.6%). See the [audit results](../results/2026-10-04-cue-audit-and-cue-free-datasets.md).
  - Cue-free candidates checked against their papers. Yang2025 (51 people, CC BY) is the clean addition: [Q26](../open-questions.md) is for the founders.
  - Riemannian folds now run in parallel (the first PhysionetMI audit hit the 2-hour limit).
  - **Q26 approved → [D-033](../decisions/decision-log.md):** Yang2025, Shin2017A (test only) and GuttmannFlury2025 (gaze diagnostic) are in the registry.
    - Their timing was checked at the papers; three facts were corrected along the way.
    - The locked test set grew to v2 (+22 people, v1 unchanged).
    - Downloads now check the publishers' MD5s, and GPL-3.0 must be accepted per download.
    - The downloads themselves wait for approval and for AWS. Open: Shin2017A's montage ([Q27](../open-questions.md)).
- **Fixed:** result files took their provenance stamp at the end of a run instead of the start. They now record the code that actually ran, plus a code-only "dirty" flag.
- **Business:** [ready-to-adapt drafts for the first 10 outreach messages](../business/first-10-messages.md); the founders send them.
- **4 Oct late evening: ready for AWS (D-032).**
  - PyTorch's CUDA 12.6 build installs with `--extra ml-cuda`. `neuros loso --device cuda` spreads folds over the GPUs, with deterministic kernels by default.
  - The [AWS runbook](aws-runbook.md) and `scripts/aws/bootstrap.sh` are ready. The download list is measured at ≈ 282 GB, and each download waits for approval.
  - The preprocessed-recording cache (§4.5) makes every person's loading and filtering a one-time cost.
  - **E0b, Riemannian part**, running on the laptop: BNCI2014-001 (cue-free) × reference × band × channel set. PhysionetMI's development people follow.
- **5 Oct, early hours: E0b's Riemannian part decides the band, and the E1 runner is built.**
  - **8–30 Hz beats 4–30 Hz** with every method on both datasets: +2.4 to +4.2 points on BNCI2014-001 and +1.1 to +2.5 on PhysionetMI (84 people). The reference makes no difference, and the motor strip matches all channels.
    - [Results](../results/2026-10-04-e0b-riemannian-band-reference-channels.md); [D-034](../decisions/decision-log.md).
    - E1 runs at 8–30 Hz on the motor strip. The contract's default changes once EEGNet's check on AWS agrees.
  - **`neuros e1`** (`configs/e1.yaml`): pooled vs single-dataset over 5 development folds, leave-one-dataset-out, and the cue-only control for every dataset.
    - Test-only datasets are never trained on.
    - Workers share the assembled windows read-only, and the result file is saved after every model.
    - Smoke-tested on the laptop's three datasets.
  - **Every command that loads data** now refuses people that `neuros download` didn't fetch. MOABB would otherwise download them unasked.
- **5–6 Oct: on AWS (g6.2xlarge, one L4; D-032), E1 done.**
  - All approved data is downloaded (10 datasets plus Shin2017A). Every dataset E1 uses passes validation, after two unit fixes: Cho2017 ([D-035](../decisions/decision-log.md)) and Yang2025 (D-036).
  - **E0b, EEGNet part:**
    - BNCI2014-001: 4–30 vs 8–30 Hz, 61.0 vs 62.0 (p = 1.0);
    - PhysionetMI: 62.4% at 4–30 Hz (84 people); its 8–30 Hz arm is running.
  - **E1** ([results](../results/2026-10-06-e1-pooled-single-lodo-cue-only.md)):
    - pooling gains +0.9 points overall; unseen datasets cost nothing;
    - cue-free motor imagery: 69.4% (BNCI2014-001) and 58.8% (Yang2025);
    - Dreyer2023's cue alone decodes at 74.4%.
  - Shin2017A waits for Q27: it shares only Cz with the motor strip.
- **6–9 Oct** ([follow-ups](../results/2026-10-06-e1-followups-cue-test-seeds-ceilings.md)):
  - Q27 answered (D-037: Shin2017A's C3/C4 derived from its half positions). Liu2024 added as test-only consumer-grade hardware (D-038; downloaded, checksums match, 50/50 recordings validate). Locked test set v3.
  - **Cue test:** the cue-driven training datasets don't hurt the cue-free numbers, so they stay. Stieger2021's sessions 3–11 aren't worth 299 GB (recommendation; the founders decide).
  - **Three seeds:** pooling +1.3 points (p = 1e-5).
  - **Ceilings:** zero-calibration EEGNet beats within-dataset Riemannian LOSO on 6 of 7 training datasets. A day-1 calibrated model beats it only on BNCI2014-001.
  - **E0b, EEGNet part:** no band preference. 8–30 Hz stands; the contract default switches at the freeze (D-034).
  - **E2 done 9 Oct** ([results](../results/2026-10-09-e2-warmup-curve.md)):
    - the warm-up closes 52–65% of the gap in 2 minutes, but cue-free accuracy at 120 s is 57–66% and only 10 of 73 people reach 70%;
    - S = 120 s on Yang2025, longer elsewhere;
    - Shin2017A 62.3% (unseen hardware, cue-only at chance).
- **10 Oct: D-039 done** ([results](../results/2026-10-10-d039-crops-multiclass.md)):
  - Crops, extra classes or both: +0.3 to +0.4 overall (within seed noise). The cue-free numbers at 120 s are unchanged (Yang2025 56.8% in every variant). Q28's bar isn't met; E2's recipe stays.
  - The first crop runs were void: two glitched Yang2025 crops broke 4 of 5 models. An artifact rule now drops them (`8605be4`). A general guard now refuses any glitched training window (`32f22ad`).
- **10 Oct, evening:** the critical path moves to the product ([Q30](../open-questions.md): reframe to "calibration saved").
  - **Engine core done** (`9c7e00d`, `src/neuros/engine/`):
    - the chain: preprocessing → quality gate → online alignment (E2's) → layout → decoder → policy;
    - states and events (`neuros.intent.v0`), and the simulator.
    - Live windows equal the training pipeline's bit for bit. On a new synthetic person, 92% of task-window commands are right with no calibration; latency is 0.3 ms median.
    - **Next:** model bundle + EEGNet decoder, LSL in / WebSocket out (needs the `engine` extra installed), `neuros simulate` / `neuros run`, the JS client.
  - **Calibration curve done** ([results](../results/2026-10-10-calibration-curve.md)):
    - zero calibration after 2 min ≈ 5–20 labelled trials per hand of conventional calibration;
    - a full 10–15-min calibration is 4–8 points better, and still leaves most people below 70%.
    - Next proposed: fine-tuning our model on 5–10 trials per hand (Q30).
  - **Q30 and Q31 done** ([results](../results/2026-10-10-q30-finetune-q31-fit-check.md)):
    - our model + 10 trials per hand (~3 min) is within 1–3 points of a 10–13-min calibration (not significantly worse; strict bar fails, lenient passes); 20 per hand (~5 min) matches it on Yang2025;
    - the 1-minute fit check fails (r = 0.23); dropped.
    - Next: the shipping bundle and a real-data replay through the engine (bar below), then an optional 3-minute guided calibration in the engine if the founders take Q30's headline.
  - **The shipping bundle and the real-data replay (bar set 10 Oct, before running):**
    - **Decision:** is `mvp-v0` (E2's recipe trained on all 7 training datasets' development people) the bundle for the live demo and the founders' headset test, or is there a gap between the engine and the offline numbers to fix first?
    - **Run:**
      - `neuros bundle --experiment configs/e2.yaml` (GPU);
      - then `neuros simulate` as fast as possible, θ = 0.6, on every development person of the two test-only datasets the engine reads without derived channels: BNCI2014_004 (9) and GuttmannFlury2025 (25), first session each.
      - Neither dataset trained the bundle. Both keep the class on screen, so these numbers check the engine; they are not a cue-free headline.
    - **Pass (all four):**
      1. All 34 replays finish without an error.
      2. Committed commands in the scored task windows are right at least as often as E2's offline accuracy for the same people, minus 5 points (the largest gap E2 measured between streaming and full-session alignment). Balanced; mean over people; E2 61.9% and 62.3%, so the bar is ≥ 57.2% across the 34.
      3. No stuck output: across people, each class gets at least 30% of the committed commands, and the engine commits on at least 20% of task-window decisions.
      4. Engine latency p99 < 50 ms per decision.
    - **Fail:** no live demo or headset test with this bundle until the gap is found.
    - **Cost:** about 15 GPU-minutes plus about 30 CPU-minutes.
  - The engine extra (pylsl, websockets, brainflow) is installed on the laptop (approved 10 Oct).
- **Next (before D-031's gate, ≈ 16 Oct; Q28):**
  - ~~Training crops and multi-class auxiliary training~~: done 10 Oct, no lift.
  - A same-session calibrated ceiling on Yang2025, BNCI2014-001 and Shin2017A (the number to quote against "just calibrate").
  - E2 v2: budgets of 180 and 300 s, a sweep of the prior's weight, and a per-profile prior for new hardware.
  - The fallback in parallel: E6's rest vs mental-effort head.
  - The founders: Q28 (the gate), Q29 (Stieger2021), outreach (drafts ready), the 16-vCPU quota request.

## Week 1: Sat 3 → Fri 9 Oct · data and baselines
| Day | Engineering | Business / YC |
|---|---|---|
| **Sat 3** | Scaffolding (pyproject, CI, src layout, error types). Data registry and harmonisation for BNCI2014-001/004 + PhysionetMI. `validate.py`. Compute C-core. Fix the locked test set by seed. Tests. | Draft the customer-interview script (calibration pain, current workflow, willingness to pay) |
| Sun 4 | ✅ (done Sat 3 Oct evening) Causal filters, timestamp resampler, windows, offline EA (unit tests: aligned mean covariance = I). Riemannian baselines on BNCI2014-001 | ✅ List 15 target contacts: UVA EEG/BCI labs, OpenBCI forum, BCI startups |
| Mon 5 – Tue 6 | ✅ (done Sat 3 Oct night) EEGNet encoder, trainer, subject-wise splits → **E0 sanity** (≈ 74% EA + EEGNet + SWA): **pipeline check passed, 75.0%; cue-controlled 60.6%** (4 Oct) | Founders send the first 10 outreach messages ([drafts ready](../business/first-10-messages.md)) |
| Wed 7 – Fri 9 | On AWS (D-032): downloads, then validation of the remaining loaders (Cho2017, Lee2019_MI, Dreyer2023, EEGMAT, Stieger subset, and D-033's Yang2025, Shin2017A, GuttmannFlury2025), checking each one's declared class display (D-030) against its paper, and a cue audit of each. **E0b**, **E1** pooled + first LODO on D-030 windows, each with its cue-only control (automate that in the E1 runner) | One-page landing site + waitlist. YC: one-liner, founder bios, "why now" |

## Week 2: Sat 10 → Fri 16 Oct · live engine and adaptation
- **Engineering:**
  - the streaming engine (LSL in, quality gate, online alignment, policy with the configurable command set and `HOLD("unrecognized")` (D-024), WebSocket out);
  - the simulator and the JS client;
  - the golden train–serve parity test.
- **Experiments:** **E2** warm-up curve + **H1** (rest-seeded), then **E3** test-time adaptation.
- **Hardware:** headset arrives → profile YAML, signal checks, founders' self-test. Design the self-test so the cue leaves the screen before the analysed window (D-030): it's our only cue-free evidence beyond BNCI2014-001.
- **Business:** 8–10 interviews done; write up the pains; ask for pilot commitments or letters of intent.
- **YC:** all written answers v1; a rough demo recorded on the simulator.

## Week 3: Sat 17 → Fri 23 Oct · generalization and hardware-agnostic proof
- **Experiments:**
  - **E4:** heads (the P-1 rule) and SupCon;
  - **E5:** virtual headsets;
  - **E6:** EEGMAT;
  - **E7:** command sets (3 vs 5 commands) and false commands per minute on never-seen activity (D-024);
  - side experiments S1/S2 if time allows.
- **Pilot sessions:** 3–5 consented volunteers on the real headset, giving the first live numbers.
- **Lab test:** try NeurOS on a UVA lab's research EEG system (second-hardware proof).
- **Optional:** SSVEP demo insurance (founder's call).
- **Business:** 15+ interviews in total; 2–3 pilot commitments or letters of intent; a pricing hypothesis.
- **YC:** answers v2 and the founder-video script.

## Week 4: Sat 24 → Fri 30 Oct · freeze, demo, submit
- **Sat 24, freeze:**
  - final locked-test, 10-fold and LODO runs;
  - no tuning after the freeze;
  - results page plus a one-page technical report.
- **Sun 25 – Tue 27:**
  - demo video: a random volunteer, real time, unedited UI capture, 2–3 takes;
  - the founder video.
- **Wed 28 – Fri 30:** final answers, a final number audit against [claims-audit.md](../claims-audit.md), and **submit**.
- **Sat 31 – Sun 1 Nov:** buffer only.

## Cut list (if we fall behind, drop in this order)
1. Side experiments S1–S4.
2. SSVEP demo insurance.
3. E4-zs (zero-shot test).
4. The text-anchored head (ship the linear head; keep the vocabulary table for naming).

**Never cut:** E0, E1, E2, the engine + simulator, the live demo, the interviews, and since D-031 (4 Oct) one validated mental state (E6, kept small).

## What YC will want to see (track weekly)
| Metric | Target by 30 Oct |
|---|---|
| Headline number with protocol | Frozen 24 Oct |
| Adaptation curve (seconds → accuracy) | Frozen 24 Oct |
| Live demo on a random person | Recorded by 27 Oct |
| Customer conversations | 15+ |
| Pilot commitments / letters of intent | 2–3 |
| Waitlist sign-ups / developer interest | Track from week 1 |
