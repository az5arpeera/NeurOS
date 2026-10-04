# Pretraining on 50,000 Hours of Unlabeled Brain Data (Neuralink update)

**Neuralink** (company engineering blog, no named authors) · 1 Oct 2026 · **not peer-reviewed; company-reported results from an investigational device**
Link: [neuralink.com/updates/pretraining-on-50000-hours](https://neuralink.com/updates/pretraining-on-50000-hours/)
**Read:** full text, 4 Oct 2026
**Relevance:** the market leader in brain implants says publicly that calibration is the core user-experience problem of BCIs, and names "zero-shot calibration" and "an API for the motor cortex" as open goals. Both are NeurOS's pitch, on a different signal.

## What they did
- **Signal:** spikes from Neuralink's implant ("Link"), recorded in the hand area of the motor cortex. Intracortical, not scalp EEG.
- **Data:** clinical-trial participants have streamed **over 50,000 hours** of unlabeled, free-form neural data in two years. The first participant alone recorded **over 9,000 hours** (22.4 billion spikes).
- **Model:** one encoder **per participant**, pretrained on thousands of hours of that participant's own recordings. It turns drifting spike data into stable embeddings, and the decoders are trained on those embeddings.
  - Mamba (a state-space network): causal, constant low latency.
  - Each spike is a token (POYO-style), read by a Perceiver input layer.
  - **Self-supervised objective:** "spatially masked auto-Poisson regression". The model sees half the channels and predicts the other half's activity.

## Reported results (company-reported)
- **Calibration burden before:** users spent an average of **55 minutes per week** calibrating, typically 10 minutes at the start of each day.
- **Calibration burden after:**
  - Decoders lasted more than a week; some kept strong performance for over three weeks.
  - Some users went from 10 minutes a day to **10 minutes a week**.
  - 30 s of labelled embeddings ≈ 3.5 min of raw spikes.
  - One decoder gave usable control 20 months after its calibration.
- **Performance:** six participants set personal records; three beat the previous record of 10.39 bits/s, and one reached **11.32 bits/s** (cursor control). The median user is about 10 bits/s.
- **Robotic arm simulation** (7+ dimensions): decoders on embeddings stayed controllable a week later.

## What they say is unsolved
- **Pooling people doesn't help yet.** "All results from live use in this post came from models pretrained on a single participant's data." Online, multi-participant models "perform no better than their single-participant counterparts".
  - One offline-style transfer worked: a model trained on one participant (P9) adapted to another (P2) with 99% of its weights frozen, and beat P2's existing decoder.
- **Zero-shot calibration** ("able-bodied control without ever calibrating") and **one-shot calibration** are stated as future directions. In their words, "None of these problems are solved."
- **"An API for the motor cortex":** one shared decoder across applications, so developers build on a common interface for the Link.

## What it means for NeurOS
1. **It validates the problem, from the best-funded player.** Even with implanted electrodes (far cleaner signals than EEG), calibration costs users nearly an hour a week, and zero-calibration is their stated goal. Use this in the YC application's "why now", with the citation.
2. **It's a different market, not the same product.**
   - Neuralink: an implant, surgery, a clinical trial (this post names participants up to P15), and decoders that work only with the Link.
   - NeurOS: non-invasive EEG, any headset, no surgery, people the model has never seen.
   - Their data can't train an EEG model: different signal, physics and scale.
3. **Their hardest open problem is our core bet.** Cross-person transfer doesn't yet help them online, and they have 50,000 hours. Our method is cross-person by construction: per-session alignment, pooled training on ~380 people, testing on people the model never saw.
   - E0 (3 Oct) is a first, small sign it works on EEG: 75.0% for never-seen people, on par with a model calibrated on each person's own session (9 people, development numbers).
4. **"Behind on data" is the wrong axis.**
   - Their 50,000 hours are *deep* (thousands of hours from each of a few people); what we need is *broad* (many people, many headsets).
   - In EEG, scale has bought little so far: INCEPT's 11,000 hours of pretraining gives about +4 points over EEGNet ([synthesis §6](../synthesis.md)).
   - The data that matters most is unlabeled EEG from normal use. A hardware-agnostic layer is the one place where that accumulates *across* headset brands, with opt-in consent (D-006 rules; local-only by default, §10).
5. **A technique to borrow later.** Their objective (predict half the channels from the other half) is a natural self-supervised task for our hardware-agnostic goal: it teaches a model to infer missing electrodes. It's a candidate side experiment after the MVP (public unlabeled EEG exists; licences to check), not on the Nov 1 critical path.

## Caveats
- Company blog: no methods paper, no confidence intervals, no number of participants per result. Figures are described, not tabulated.
- Bits per second on a cursor task isn't comparable with our balanced accuracy on left vs right imagery. **Never put the two numbers side by side in YC materials.**
- Re-check every quoted figure against the page before using it (claims-audit rule).
