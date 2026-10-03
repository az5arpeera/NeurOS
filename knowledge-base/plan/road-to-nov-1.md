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

## Week 1: Sat 3 → Fri 9 Oct · data and baselines
| Day | Engineering | Business / YC |
|---|---|---|
| **Sat 3** | Scaffolding (pyproject, CI, src layout, error types). Data registry and harmonisation for BNCI2014-001/004 + PhysionetMI. `validate.py`. Compute C-core. Fix the locked test set by seed. Tests. | Draft the customer-interview script (calibration pain, current workflow, willingness to pay) |
| Sun 4 | Causal filters, timestamp resampler, windows, offline EA (unit tests: aligned mean covariance = I). Riemannian baselines on BNCI2014-001 | List 15 target contacts: UVA EEG/BCI labs, OpenBCI forum, BCI startups |
| Mon 5 – Tue 6 | EEGNet encoder, trainer, subject-wise splits → **E0 sanity** (≈ 74% EA + EEGNet + SWA) | Founders send the first 10 outreach messages |
| Wed 7 – Fri 9 | Remaining loaders (Cho2017, Lee2019_MI, Dreyer2023, EEGMAT, Stieger subset). **E0b**, **E1** pooled + first LODO | One-page landing site + waitlist. YC: one-liner, founder bios, "why now" |

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
