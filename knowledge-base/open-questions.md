# Open questions (Step 2)

Questions that must be answered before the MVP architecture is locked (Step 4). Each has context, options, and a recommendation (**Rec**). Answer inline or in chat; answered items move to [`decisions/decision-log.md`](decisions/decision-log.md).

Priority: 🔴 blocking (answer before any architecture work) · 🟠 this week · 🟡 can wait until after the MVP.

---

## 🔴 Blocking

### Q1. What does "CLAP" mean for us, and what experience should the MVP decode into it?
The founder notes use "CLAP" for three different things: **(i)** Contrastive Language–*Audio* Pretraining specifically, **(ii)** any CLIP-style alignment to a pretrained embedding space (text, audio or image), and **(iii)** "contrastive learning" in general. The evidence (see [synthesis §5](research/synthesis.md)) supports these options:
- **A. Language-anchored intent head:** EEG embeddings trained against text embeddings of command phrases. Zero-calibration compatible, unifies the label sets of 5+ datasets, lets developers name commands. It is *not* open-vocabulary.
- **B. CLAP audio head:** EEG matched against CLAP embeddings of the sounds the user hears (music identification, or which instrument or stream they're attending to). This matches the headphones thesis. Only within-subject results exist, so **cross-subject would be a first**.
- **C. CLIP image head** (THINGS-EEG2): works cross-subject, but at 6.2% top-1 on 200-way. Hard to demo live.

**Rec:** A in the core MVP; B as a parallel research track that could become the headline result. **Pushback:** "true zero" open-vocabulary thought decoding has no non-invasive evidence. Claiming it in a YC application will cost credibility with technical partners. Keep it as the stated long-term vision. *Also:* your Step 0 sentence ended at "UDA and CLAP for true zero phase as…". Please finish it; it may change this answer.

### Q2. Which "UDA" do we mean?
- **Train-time** domain generalisation across training subjects (DANN / Deep CORAL / MMD). No target data needed; mixed evidence; DANN is reported unstable on small EEG datasets.
- **Test-time** source-free adaptation on the new user's unlabeled stream: online alignment + adaptive batch norm + entropy minimisation. **+10 points** on 4-class cross-subject (57.4 → 67.3%) in published work.

**Rec:** both in the MVP, with **test-time adaptation as the core** (it's what makes the live demo work) and DANN/CORAL as a reported ablation.

### Q3. Which zero-calibration claim do we make?
Definitions are in [synthesis §2](research/synthesis.md): L0 = no target data at all; **L1 = passive warm-up on unlabeled data during normal use, no calibration screen**; L2 = a few labeled trials.
**Rec:** claim **L1** ("no calibration session; adapts silently as you use it") and always report L0 alongside it. Kill the "3 seconds" and "perfect on the first frame" claims ([claims audit](claims-audit.md) #2 and #13).

### Q4. What is the MVP's intent vocabulary and headline metric?
- **Vocabulary:** left / right (binary) or left / right / rest (ternary). 4-class (+ feet, tongue) is only about 66% cross-subject.
- **Headline:** accuracy on **held-out subjects across held-out datasets** (e.g. train on 4 datasets, test on the 5th), an **adaptation curve** (accuracy vs seconds of use), information transfer rate (ITR, bits/min), latency, and the fraction of users above chance.

**Rec:** ternary (left / right / rest), plus a confidence-gated "hold" output. Headline = cross-dataset LOSO on ~370 subjects plus the adaptation curve. **What target numbers would you be proud to put in the application?** Realistic: ~75–80% binary, ~65–70% ternary.

### Q5. Demo form, and do we buy a headset now?
- **Playback demo:** stream held-out public subjects through LSL and switch "virtual users" live. Safe and reproducible.
- **Live demo:** real people, no calibration screen. Most persuasive, but risky (15–30% of people are MI-inefficient).
- **Headset requirements:** covers **C3 / C4** (motor cortex), streams over LSL, arrives by mid-October. Candidates to price-check: OpenBCI Cyton (+Daisy), g.tec Unicorn Hybrid Black, Neurosity Crown. **Muse can't do motor imagery** (no motor-cortex electrodes).

**Rec:** both. Playback for the numbers, and a live clip with 2–3 **pre-screened** people. **Order a headset this week** if the live clip is in scope. What's the budget?

### Q6. Team, time and compute
- Who owns what between you and your cofounder? (Data pipeline + evaluation / modelling / live engine + SDK / demo app / YC writing.)
- Hours per week each until 1 Nov? Is either of you full-time?
- Compute: a local NVIDIA GPU? Colab? Budget? The core pool is small by ML standards (a laptop GPU works), but **LOSO over ~370 subjects** means hundreds of training runs. Use subject-grouped k-fold plus leave-one-dataset-out instead.

---

## 🟠 This week

### Q7. Training data pool and licences
Proposed core: PhysionetMI (109), Dreyer2023 (87), Stieger2021 (62), Lee2019_MI (54), Cho2017 (52), plus BNCI2014-001/004 for literature comparability ([datasets](datasets.md)). Anything to add or remove? **Licences must allow commercial training.** OK for me to check them?

### Q8. EEGMAT as a second paradigm?
Rest vs mental arithmetic, 36 subjects, cross-subject. It shows the layer isn't MI-only, and it's comparable to published foundation-model numbers. **Rec:** yes, as a secondary table, not the demo.

### Q9. EMG / EOG in the MVP?
The old spec had a multimodal EEG+EOG+EMG network. Consumer headsets rarely record EMG/EOG. **Rec:** EEG-only model input; use EOG (Dreyer2023, BNCI) only for training-time artifact handling. Multimodal fusion becomes a later module.

### Q10. What exactly does a developer get on 1 Nov?
**Rec:** a Python package plus a local engine: LSL in → intent events out (JSON over WebSocket, with confidence and "hold"), a dataset-playback "virtual headset" simulator (the Trojan-horse dev tool), and a ~10-line JavaScript client example. Unity/C#: later?

### Q11. First customer persona for the YC narrative
BCI app developers and hackers (OpenBCI community) / headset OEMs / research labs / assistive-tech companies (ALS, stroke). **Rec:** developers + OEMs as the business, with assistive tech as the "why it matters" story. Can you get **letters of intent or user quotes** from any of these before 1 Nov (e.g. UVA labs, OpenBCI forum developers)?

### Q12. One positioning sentence
"The zero-calibration intent layer for non-invasive neurotech" vs "Android for neuro-wearables" vs something else. **Rec:** lead with the concrete one (zero-calibration intent layer) and use "Android for neuro-wearables" as the analogy.

---

## 🟡 Can wait until after the MVP
- Q13. Move Riemannian math **inside** the network (SPDNet / SPDDSMBN) in v2? **Rec:** v2, after the EA + test-time adaptation baseline is measured.
- Q14. RUNet-style resting-state contrastive pretraining (+1–2 points)? **Rec:** Phase 1.5.
- Q15. Graph or electrode-position-embedding inputs for arbitrary montages? **Rec:** after the canonical-montage baseline.
- Q16. Foundation-model pretraining on TUEG? **Rec:** Phase 2.
- Q17. Privacy posture statement (on-device processing, consent, neural-data laws). **Rec:** one sentence in the application; details later.
- Q18. Sourced market sizing (the current TAM/SAM/SOM is unsourced). **Rec:** before the final application draft.
- Q19. IP: patent the alignment/adaptation approach? **Rec:** talk to a UVA tech-transfer / startup-law clinic after the MVP. EA itself is published prior art.

---

## Answered
_(Move answered questions here with date + answer, and log the decision.)_
