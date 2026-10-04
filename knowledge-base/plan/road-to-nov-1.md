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
  - **Proposed protocol change ([Q25](../open-questions.md)):** cue-controlled windows plus a cue-only control for every motor-imagery result. The founders decide.
  - Without the arrow's first half-second, the 4-s window's 83.7% falls to 72.3% ([0.5, 4.0] s), no better than our 2-s window: its gain was the cue's onset, not the window's length.
- **Fixed:** result files took their provenance stamp at the end of a run instead of the start. They now record the code that actually ran, plus a code-only "dirty" flag.
- **Business:** [ready-to-adapt drafts for the first 10 outreach messages](../business/first-10-messages.md); the founders send them.
- **Next:** cluster access (Q6b) before E1, which is too heavy for the laptop.

## Week 1: Sat 3 → Fri 9 Oct · data and baselines
| Day | Engineering | Business / YC |
|---|---|---|
| **Sat 3** | Scaffolding (pyproject, CI, src layout, error types). Data registry and harmonisation for BNCI2014-001/004 + PhysionetMI. `validate.py`. Compute C-core. Fix the locked test set by seed. Tests. | Draft the customer-interview script (calibration pain, current workflow, willingness to pay) |
| Sun 4 | ✅ (done Sat 3 Oct evening) Causal filters, timestamp resampler, windows, offline EA (unit tests: aligned mean covariance = I). Riemannian baselines on BNCI2014-001 | ✅ List 15 target contacts: UVA EEG/BCI labs, OpenBCI forum, BCI startups |
| Mon 5 – Tue 6 | ✅ (done Sat 3 Oct night) EEGNet encoder, trainer, subject-wise splits → **E0 sanity** (≈ 74% EA + EEGNet + SWA): **pipeline check passed, 75.0%; cue-controlled 60.6%** (4 Oct) | Founders send the first 10 outreach messages ([drafts ready](../business/first-10-messages.md)) |
| Wed 7 – Fri 9 | Remaining loaders (Cho2017, Lee2019_MI, Dreyer2023, EEGMAT, Stieger subset), recording how long each dataset's cue stays on screen. **E0b**, **E1** pooled + first LODO, on cue-controlled windows if Q25 is adopted | One-page landing site + waitlist. YC: one-liner, founder bios, "why now" |

## Week 2: Sat 10 → Fri 16 Oct · live engine and adaptation
- **Engineering:**
  - the streaming engine (LSL in, quality gate, online alignment, policy with the configurable command set and `HOLD("unrecognized")` (D-024), WebSocket out);
  - the simulator and the JS client;
  - the golden train–serve parity test.
- **Experiments:** **E2** warm-up curve + **H1** (rest-seeded), then **E3** test-time adaptation.
- **Hardware:** headset arrives → profile YAML, signal checks, founders' self-test.
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
5. E6 / EEGMAT backup demo.

**Never cut:** E0, E1, E2, the engine + simulator, the live demo, the interviews.

## What YC will want to see (track weekly)
| Metric | Target by 30 Oct |
|---|---|
| Headline number with protocol | Frozen 24 Oct |
| Adaptation curve (seconds → accuracy) | Frozen 24 Oct |
| Live demo on a random person | Recorded by 27 Oct |
| Customer conversations | 15+ |
| Pilot commitments / letters of intent | 2–3 |
| Waitlist sign-ups / developer interest | Track from week 1 |
