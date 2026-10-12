# E5: virtual headsets: which hardware can we honestly say NeurOS supports?

**Written:** Mon 12 Oct 2026, before any code or run. **Status:** bar set; waiting for founder approval to run.

## The decision
Which headset classes do the application, the docs and the demo call **supported**, **experimental** or **not supported** for commands with zero calibration?
- **What it changes:**
  - the "any headset" wording;
  - how the demo video describes the Ganglion;
  - which hardware makers we approach first.
- **At the freeze (24 Oct):** the same table is computed once on the locked test set. It's the "virtual-headset table" the architecture (§8.6) publishes with the headline. This development run decides the labels; the locked run gives the numbers we quote.

## Hypothesis
- Our model reads the motor strip, so what matters is how much of it a headset covers:
  - most 4–8-channel headsets give it only C3, Cz and C4;
  - a few add CP3/CP4.
- Fewer channels should cost accuracy. 3 channels may stay close to the full cap because the hand areas sit under C3 and C4. On Shin2017A, 3 channels gave 57.1% after 120 s.

## Channel sets
What the model can read: its 20-channel motor-strip layout ∩ the headset.

| Set | Model channels | Headsets in this class |
|---|---|---|
| Research cap | every layout channel the dataset has (E2's baseline) | lab caps |
| 4-channel strip | CP3, C3, C4, CP4 | Neurosity Crown |
| **3-channel** | **C3, Cz, C4** | **OpenBCI Ganglion (our demo headset)**, g.tec Unicorn, BCI IV-2b layouts |
| 2-channel (report only) | C3, C4 | the bare minimum |
| No motor strip | none | Muse (frontal/temporal): not supported for commands by construction; not run |

## Data
- **People:** E2's 73 cue-free development people, scored by E2's own fold models (trained on the full layout by the E2 recipe, 5 development folds):
  - Yang2025 (41) and BNCI2014-001 (9);
  - Shin2017A (23), test-only and never trained on.
- **The scored number:** each person's zero-calibration accuracy after **120 s** of unlabeled streaming (E2's `e2_stream_120s`). Their windows are restricted to the set's channels, and the missing ones are zero-filled exactly as the live engine places them.
- **Shin2017A only has C3, Cz and C4 in the layout.** Its research-cap number *is* its 3-channel number, so it counts towards the 3- and 2-channel sets but not the paired comparison.
- **Never touched:** the locked test set.

## Pass bar (set before running), for each set
- **Supported:**
  - paired against the research cap on Yang2025 and BNCI2014-001 (50 people), the mean is no more than **3 points** lower;
  - **and** the set is above chance over every person who has it (Wilcoxon vs 50%, p < 0.05).
- **Experimental:** above chance (p < 0.05) but more than 3 points below the research cap.
- **Not supported:** not above chance.

**The headline check:** the 3-channel set (the Ganglion's class) gets its label from this bar. The demo and application describe the Ganglion accordingly:
- **Supported:** "works on a 4-electrode board".
- **Experimental:** "works on a 4-electrode board, about N points below a full cap".

## Also reported (not part of the bar)
- Each set's accuracy per dataset, and how many people reach 70%.
- The 2-channel set.

## What this can't show
- **Virtual, not real.** Real headsets differ in electrode type (dry vs gel), noise and placement. The founders' Ganglion session is the real check for the 3-channel class.
- **One model for all.** It was trained on full caps; a model trained per headset class might do better on few channels. That's a later experiment.

## Cost
- **Compute:** one E2-style GPU run on AWS, about 30–40 minutes (E2 took 26).
- **Code:** add channel-set scoring to E2's warm-up evaluation (`neuros.eval.warmup`), with a config `configs/e5.yaml`. No downloads.
- **AWS:** the founders start the machine and allow the laptop's current address.
