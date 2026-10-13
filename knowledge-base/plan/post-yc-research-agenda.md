# Post-YC research agenda: reliable control for (nearly) everyone on 3–4 electrodes

**Status: parked until after the 1 Nov 2026 deadline.** Written Sat 10 Oct from two founder questions:
- "How do we get over this bar? My goal is ~100% accuracy for all people."
- "Not the realistic steps: what research and which specific experiments would it take to decode thought, imagined movement and mental states accurately, with a fine-tuned architecture, without needing data from billions of people, on only 3–4 electrodes?"

**Before anything here runs:**
- Each experiment gets its decision and pass bar written in the KB first, and founder approval.
- Every download is asked about first (file, source, size).
- Nothing here changes the YC application or the demo.

**Background:**
- [Data-scaling projection](../results/2026-10-11-data-scaling-projection.md): more training people alone can't get motor imagery near everyone.
- [Calibration replay](../results/2026-10-11-calibration-replay.md): 20-trial fine-tuning gave +3 points, not significant, and made the model overconfident.

---

## Part 1: what "~100% for everyone" can mean

**No brain-computer interface is right 100% of the time per decision, implants included.** The engineering target that can approach "everyone" is different: **every person gets commands that are ~99% reliable.** The open question is how long each reliable command takes.

**Reliability can be bought with time.** If the engine waits for agreeing evidence before acting, errors fall fast. The price depends on how often each 2-s decision is right.

*Wald's sequential test, independent decisions every 2 s; [`reliability_time.py`](../../scripts/analysis/reliability_time.py).*

| Per-decision accuracy | Time for a 99%-reliable command | For 99.9% |
|---|---|---|
| 55% | 458 s | 688 s |
| 60% | 113 s | 170 s |
| 65% | 49 s | 74 s |
| 70% | 27 s | 41 s |
| 75% | 17 s | 25 s |
| 80% | 11 s | 17 s |
| 90% | 5 s | 8 s |
| 95% | 3 s | 5 s |

**Applied to our 73 cue-free people.** The scores are noise-corrected as in the projection. "Never" means at or below 52%, effectively chance.

| Share with a 99%-reliable command within | Zero calibration (after 120 s of use) | Own calibrated ceiling |
|---|---|---|
| 10 s | 3% | 10% |
| 30 s | 14% | 30% |
| 60 s | 16% | 38% |
| 120 s | 23% | 58% |
| Median time | 229 s | 111 s |
| Never | 18% | 14% |

**Both assumptions are generous.** Real 2-s decisions are correlated (overlapping windows, slow drifts), and sustained imagery tires people. Real times are longer.

**So imagined movement alone can't give everyone reliable control at a usable speed.** The four levers, roughly in order of impact:
1. **A different signal for the people imagined movement can't serve.**
   - Evoked responses to flickering or coded visual targets (SSVEP, c-VEP) or to recognized targets (P300) are strong in most people, and calibration-free decoders exist.
   - In the S-JEPA paper the same seven people scored 65% on imagined movement and 94% on flicker responses (within-person, Lee2019).
   - At 90–95% per decision, a 99%-reliable selection takes 3–5 s.
   - **Flicker can trigger seizures in photosensitive people:** screen and warn first.
2. **A stronger signal per person:** more channels over motor cortex, better electrode contact.
   - +10 points of per-person accuracy cuts the time per reliable command roughly 4×.
   - This is a hardware-partner question as much as a software one.
3. **Product design.**
   - **Already built:** hold when unsure.
   - **To add:**
     - accumulate evidence before acting;
     - confirm and undo;
     - a hybrid mode (flicker to select and confirm, imagined movement for continuous control);
     - measure each person's accuracy live in the first minutes and route them to the mode that works. This replaces the failed Q31 fit check.
4. **Train the person and improve the model.**
   - Imagined movement is a learnable skill.
   - Better models (JEPA-style pretraining, foundation models) add a few points.

---

## Part 2: the research program

### The bottleneck, in our own numbers
**The data need is huge because a supervised model must learn every way one head differs from another purely from examples.** There are three ways to cut it:
- **Simulation:** tell the model how people differ, by simulating heads from physics.
- **Inference:** let it measure each new person from their unlabeled signal.
- **Labels from use:** take them free from normal use.

But first, measure how much information 3–4 electrodes carry. No model decodes what isn't in the signal.

**Today's zero-calibration score is only ~4–8 points below each person's own calibrated score:**

| Dataset | Zero calibration (120 s) | Own calibrated ceiling | Gap |
|---|---|---|---|
| Yang2025 (41) | 56.8% | 60.5% | 3.7 |
| BNCI2014-001 (9) | 66.0% | 73.8% | 7.8 |
| Shin2017A (23) | 57.1% | 64.9% | 7.8 |

*Means; [projection](../results/2026-10-11-data-scaling-projection.md), step 2.*

A perfect architecture closes at most that gap, unless the "ceiling" is too low. It may be, because it was measured:
- with a simple model (TS+LR);
- on 15–50 trials per hand;
- on people who'd had no training.

**Whether the ceiling is real is the most important unknown in the program.** Experiments 1, 2 and 10 settle it.

### What 3–4 scalp electrodes can carry

| Goal | What the signal allows | Research frontier |
|---|---|---|
| Imagined movement | Left vs right hand (maybe feet) over C3/Cz/C4. Not fingers or many classes. | Close the transfer gap. Find out how many "no signal" people can be trained or reached by moving electrodes. |
| Mental states | Drowsiness, workload and engagement shift broad rhythms that almost any electrode sees. | Zero calibration works by measuring change from each person's own baseline. The hard part is ground truth (what the true state was), not channel count. |
| Thought (words, inner speech) | Reading words directly from 3–4 scalp electrodes isn't realistic (evidence below). | Don't read the thought: let a language model propose candidates and read the brain's recognition response to them (P300, error potentials). |

**Why reading words directly isn't realistic** *(search-result level; read the full texts before quoting a number externally)*:
- **Implants:** arrays in motor cortex decode inner speech at up to 74% in the best case (Kunz et al., Cell, Aug 2025).
- **Meta's Brain2Qwerty:**
  - EEG got 67% of characters wrong, even while people physically typed memorized sentences;
  - MEG got 32% wrong (35 people, Feb 2025 preprint).
- **EEG imagined-speech studies:** reported accuracies of 70–80% on small vocabularies drop to ~20–50% once the vocabulary grows or evaluation gets stricter.

### The architecture to build: a person-conditioned decoder pretrained on simulated heads
1. **Position-aware input.** Each channel carries its 3D scalp position, so any 3–4-electrode headset feeds one model.
2. **Person-normalized filters.** Frequency bands are set relative to the person's own rhythm peaks, measured from unlabeled rest. Speech recognition does the same with vocal-tract-length normalization, a one-parameter fix for speaker differences.
3. **Pretraining on simulated people.** A physics model of a template head generates unlimited "people", randomizing:
   - where the brain source sits and how it's oriented;
   - skull conductivity;
   - electrode positions;
   - rhythm frequency and strength (including none);
   - artifacts.

   This is robotics' sim-to-real trick (domain randomization).
4. **Self-supervised pretraining (JEPA-style)** on unlabeled real EEG cut down to 3–4-channel montages. It learns the real noise and artifacts the simulator misses.
5. **A person encoder.** It turns the first 30–60 s of unlabeled signal into a short "person vector" that tunes the decoder, like speaker embeddings in speech. It's trained one person per training episode (meta-learning).
6. **Labels from use.** Two sources give noisy labels while the person simply uses the app:
   - the brain's automatic "that was wrong" response (error potential, at FCz/Cz);
   - the app's own context.

   The decoder updates on these, with guards against drift and the overconfidence that sank the calibration replay.

**The bet:** differences between people come down to a handful of physical parameters. If so, simulation covers them almost free and the encoder estimates them without labels, replacing billions of labeled people with a parameter estimate. Experiment 3 tests that bet.

### The experiments, in the order they unlock each other
**Stage 1: measure the wall.** Uses data we have: Lee2019, Cho2017 and PhysionetMI have 62–64 channels.

1. **The true ceiling at 3–4 channels.**
   - **Method:** the strongest within-person model, each person's learning curve extrapolated to its plateau, at 3, 4, 8 and all channels.
   - **Answers:** whether the ~60% median ceiling comes from the signal itself or from our weak model.
   - Stieger2021's later sessions would show what months of a person's own data do. They aren't downloaded by founder decision, so using them needs a fresh one.
2. **Why people with no signal fail.**
   - **Method:** for people at chance on C3/Cz/C4, search the full cap for any change during the task.
   - **Signal elsewhere:** an electrode-placement and transfer problem, which is fixable.
   - **Signal nowhere:** these people need another signal or training.
   - We found no large published study that splits people this way, and it decides how close imagined movement can get to "everyone".
3. **Do people differ in only a few ways?**
   - **Method:** describe each person with a few parameters (where on the scalp the response shows, peak frequency, response strength). Hand a decoder those true values.
   - **If it reaches the calibrated ceiling:** the problem reduces to estimating a few numbers.
   - **Then:** test whether unlabeled rest data estimates them. Q31 showed rest doesn't predict *accuracy*; this is a different question.
4. **The best four electrodes.**
   - **Method:** search channel subsets and the reference choice for zero-calibration accuracy across people.
   - **Hypothesis:** C3, Cz, C4 plus one at the back of the head (Oz/POz) covers:
     - imagined movement;
     - error and recognition responses;
     - flicker responses;
     - drowsiness.
   - **If it holds:** a "NeurOS reference montage" for hardware partners.

**Stage 2: make transfer cheap**

5. **Frequency normalization.** Align each person's rhythm peaks before the pooled model sees the data. CPU-only, on data we have: the quickest test of the whole "tell the model how people differ" idea.
6. **Sim-to-real.**
   - **Method:** train on simulated people only and test on real ones. Then compare simulation pretraining plus real fine-tuning against real data only.
   - **The number that matters:** accuracy against the count of *simulated* people. A steep slope there is what replaces billions of real ones.
   - **Precedent:**
     - an error-potential decoder trained on simulated subjects reached ~63–73% balanced accuracy on four real datasets (Fidêncio et al. 2024);
     - simulated imagined-movement EEG has been used for data augmentation (Galvan et al. 2024).
   - We found no one pretraining on randomized simulated heads and testing on unseen real imagined-movement users. That's open ground.
7. **Person encoder.** Score it on the E2 test (zero calibration after 120 s) against mvp-v0.
8. **JEPA at few channels.** Does unlabeled pretraining beat the measured +0.5 points per doubling of training people? Unlabeled hours cost far less to get than labeled people.

**Stage 3: labels from use, and training the person**

9. **Error responses as free labels.**
   - Published decoders that need no per-person calibration detect error responses about 72–74% of the time (Bhattacharyya et al. 2017; Schönleitner et al. 2020).
   - **Desk simulation first:** on our replay engine, find the lowest error-detection accuracy at which self-training helps instead of hurts.
   - **Then:** test on a public error-potential dataset using only FCz/Cz.
10. **User learning.**
    - **Method:** feedback sessions where the model and the person adapt together, with people who start at chance.
    - **Measure:** how many become usable over about 5 sessions.
    - **Precedent:** this worked for two tetraplegic Cybathlon pilots (Perdikis et al. 2018, PLOS Biology).
    - Needs real participants and their consent.

**Stage 4: thought and mental states**

11. **Language model plus recognition responses.**
    - **Method:** an offline simulation on public P300 data at Cz/Pz/Oz, with a language model proposing candidates. Measure zero-calibration words per minute.
    - **Back-of-envelope rate:** words per minute ≈ 60 × brain bits per second ÷ bits of uncertainty left per word. At 0.5–1 bit/s and 6–8 bits per word, that's roughly 4–10 words per minute.
    - **Today:** P300 spellers manage about 1–2 words per minute. Several 2025 papers started on this combination.
12. **Mental states against each person's own baseline.**
    - **Method:** drowsiness and workload from 3–4 electrodes. Labels come from behavior (reaction-time lapses, task difficulty), not self-report.
    - **Scoring:** across people, with each person's first minutes as their baseline.
    - Emotion is the weakest science here (inflated published numbers are a known problem), so it goes last.

### Finish line
On a 4-electrode headset, after 60 s of unlabeled use, **≥ 90% of people have at least one mode that is right ≥ 90% of the time per decision**. That's about 5 s per 99%-reliable command (Part 1). Experiments 1, 2 and 10 decide whether imagined movement can ever carry that, or whether evoked modes would do most of the work.

### Cost and data

| Experiments | Needs |
|---|---|
| 1–5 and the desk part of 9 | Data we already have; mostly CPU |
| 6 | A template head model (e.g. MNE's fsaverage): a download, ask first |
| 7, 8 | GPU hours; 8 also a large unlabeled corpus. TUH needs a registration the founders would do. |
| 9, 11, 12 | Public error-potential, P300 and drowsiness or workload datasets: downloads, ask first |
| 10 | Real participants, consent, our headset |

## Experiments cut from the YC sprint
Parked by [D-041](../decisions/decision-log.md) (11 Oct). These were in the architecture's experiment table (§8) and aren't dropped. Each still needs its decision and bar written before it runs.

| Experiment | Question | Where it fits now |
|---|---|---|
| **E3:** test-time adaptation on replay (AdaBN, entropy minimisation; E3b domain-specific batch norm) | Does unsupervised adaptation during use lift accuracy beyond E2's alignment warm-up? | Part 2's "labels from use" stage: a cheaper cousin of experiment 9. Watch for the overconfidence the calibration replay showed |
| **E4:** heads (linear vs text-anchored, the P-1 rule) and the SupCon λ sweep | Does a language-anchored head or contrastive training help cross-person accuracy? | Model-side; D-039 suggests small gains at best. Run after Stage 1 says whether the ceiling is real |
| **E4-zs:** hold out "feet" entirely | Can the text head recognise a command it never trained on? | The long-term "name your own commands" claim (claims audit row 8). Report either way |
| **S1–S4:** DANN, Deep CORAL, RA vs EA, ShallowConvNet | Do adversarial or moment-matching alignment, Riemannian alignment, or another backbone beat EA + EEGNet? | Comparison rows for a paper; low expected product impact |
| **The text-anchored head** | Ship-ready naming of intents by text | Ships only if E4 passes the P-1 rule |
| **Pretrained encoders, after A6** (13 Oct: [result](../results/2026-10-13-a6-pretrained-backbone.md)) | Does a pretrained encoder help with (a) a gentler fine-tuning recipe (A6's kept model was epoch 1 every fold), (b) little training data (a new headset class, one person's calibration), (c) REVE on broadband input? | A6 found a tie (+0.9) with the authors' recipe on 339 people. Bundle v2 can already ship one |
| **SSVEP demo insurance** | A calibration-free flicker mode as a fallback demo | Becomes Part 1's lever 1 (the evoked fallback). Needs an SSVEP dataset download and a photosensitivity warning |

## Sources
All search-result level unless marked. Read the full text before citing a number externally.
- **Implants and speech decoding:**
  - Kunz, Willett et al. 2025, inner speech in motor cortex, *Cell* (DOI 10.1016/j.cell.2025.06.015): [Stanford release](https://med.stanford.edu/news/all-news/2025/08/brain-computer-interface.html).
  - Lévy et al. 2025, Brain2Qwerty: [Meta AI](https://ai.meta.com/research/publications/brain-to-text-decoding-a-non-invasive-approach-via-typing/), [arXiv 2502.17480](https://arxiv.org/html/2502.17480v1).
  - Imagined-speech EEG review (IJACSA 2026): [PDF](https://thesai.org/Downloads/Volume17No1/Paper_36-EEG_Based_Imagined_Speech_Decoding.pdf).
- **Simulation:**
  - Fidêncio, Klaes & Iossifidis 2024, generic error-potential classifier from simulated subjects: [Frontiers](https://frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2024.1390714/full).
  - Galvan et al. 2024, simulated MI-EEG for augmentation, *IEEE TNSRE*: [DOAJ](https://doaj.org/article/d9067cc0dfc844de88495be9afb91f75).
  - Closed-loop MI-EEG simulator, 2022: [PMC9428352](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9428352/).
- **Error potentials:** Bhattacharyya et al. 2017, transferable decoder: [Frontiers](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2017.00226/text).
- **Language models with P300 and SSVEP spellers:**
  - [bioRxiv 2025](https://www.biorxiv.org/content/10.1101/2025.11.06.686984.full.pdf);
  - [MindChat (SSVEP)](https://arxiv.org/html/2507.21435v1);
  - [arXiv 2410.15161](https://arxiv.org/pdf/2410.15161).
- **Known, not re-read:**
  - Perdikis et al. 2018, Cybathlon mutual learning, *PLOS Biology*.
  - Schönleitner et al. 2020, adaptive subject-independent error-potential decoders.
  - S-JEPA (motor imagery vs SSVEP), from the 11 Oct discussion.

## Reproduce Part 1
```bash
uv run --no-sync python scripts/analysis/reliability_time.py results/aws
```
