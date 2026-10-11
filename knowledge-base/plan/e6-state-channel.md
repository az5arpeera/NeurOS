# E6: the mental-state channel (rest vs mental effort): decision and pass bar

**Written:** Sun 11 Oct 2026, before any EEGMAT data is loaded (the rule: every run answers a decision, with its bar set first).

**Status:** bar set; approved by the founders 11 Oct ("yes run E6").

> **Amendment, 11 Oct, after inspecting the files and before any model ran:**
> - **Rest recordings are longer than the page says:** 80–188 s (median 182 s). The task is 62 s, and the rate is 500 Hz.
> - **The EDF headers give a 0.5–45 Hz filter.** That suggests the page's "30 Hz high-pass" is an error; the spectrum check still runs.
> - **The bar is unchanged:** baseline 0–20 s, scored rest 20–60 s, drift control 20–40 vs 40–60 s.
> - **One report-only control added: late rest.** It scores rest 20–60 s against the last 40 s of rest, for the people with ≥ 120 s of rest. Two minutes of drift is closer to the gap between the two recordings than 20 s is.
> - **Code:** `neuros.data.eegmat`, `neuros.eval.state`, `configs/e6.yaml`. Run with `neuros --config configs/data.yaml e6`.

## The decision
Can NeurOS tell **rest from mental effort across people, with no calibration** (only the person's first 20 s at rest), well enough to:
1. **ship it as the MVP's one mental state** (D-031) on research-grade headsets? (arm A)
2. **show it in the live demo** on the Ganglion's four channels? (arm B)

It also feeds [Q28](../open-questions.md): the state channel is half of the fallback headline.

- **Pass A:** `neuros.intent.v0` gains one state event (`rest` / `effort`, with a confidence); the application can say "one validated mental state".
- **Fail A:** no state ships. The claims audit's row 27 stands ("add one state only after E6 validates it"), and the application leads on intent and rigour only.
- **Arm B only decides the demo.** Fail B means the demo shows intent only, even if A passes.

## Data
**EEGMAT** (PhysioNet; 36 people; 19 scalp channels, linked-ear reference; ODC-By 1.0).
- Per person: one 60-s recording at rest ("background, before the task") and one 60-s recording of mental serial subtraction. Both were cleaned with ICA by the authors.
- **Only the 29 development people are used.** The 7 locked-test people (`configs/splits/locked_test_v3.yaml`) are untouched until the 24 Oct freeze.

**Before any training, three facts get checked on the files and written into the [results log](../results/README.md) entry:**
- **Channels and sampling rate:** the registry's `unverified` fields.
- **The filter:** the page says "a high-pass filter with a 30 Hz cut-off". Check the spectra to see which it really is. If there's no power below 30 Hz, theta and alpha are gone and E6 stops here.
- **Eyes open or closed at rest:** the page doesn't say. If the median person's parietal alpha at rest is more than twice their task alpha, rest was probably eyes closed.
  - The number would then partly measure "eyes closed vs open", not mental effort.
  - In that case the **frontal arm (C) becomes the claim** and the result is labelled accordingly.

## Run
**Pipeline:** the engine's contract (causal filters, CAR over the channels present, 2-s windows), except the band is **4–30 Hz**, because the state lives partly in theta (4–8 Hz). Windows hop 0.5 s.

**The person's baseline (the zero-calibration condition):**
- The first 20 s of each person's rest recording, standing in for the "relax for 20 s" start of a live session.
- It is used only to normalize that person; it is never scored.
- **Scored:** rest 20–60 s against the whole effort recording.
- A **no-baseline arm** (no person information at all) is reported too.

**Two models, fixed now:**
1. **Band power + logistic regression:** log power in theta (4–8), alpha (8–13) and beta (13–30) per channel, z-scored against the person's baseline.
2. **TS+LR:** covariances re-centred on the baseline's mean covariance, then tangent space and logistic regression.

The one with the higher arm-A mean is the primary model. That choice is made on development people and confirmed on the locked test at the freeze.

**Evaluation:** leave one person out over the 29 development people.

**Channel arms:**

| Arm | Channels | Why |
|---|---|---|
| **A** | All 19 | Research-grade headsets |
| **B** | **C3, Cz, C4, Pz** | The Ganglion montage in the [headset self-test](headset-self-test.md) |
| C (report only) | Fz, F3, F4, Pz, P3, P4 | D-031's state channels: frontal theta and parietal alpha. The fallback claim if rest was eyes closed |

**Primary measure:** per-person balanced accuracy over scored 2-s windows, mean over people.

**Also reported:**
- AUROC;
- each person's accuracy;
- accuracy on 10-s blocks (mean probability over five non-overlapping windows; the unit a live state display would use);
- the no-baseline arm.

## Pass bar (set before running)
**Arm A passes (ship the state) only if all three hold:**
1. **Mean balanced accuracy ≥ 70%**, with Wilcoxon signed-rank p < 0.05 against 50%.
2. **At least 20 of the 29 people at ≥ 60%.** Most people must get it, not just a strong few.
3. **The drift control is at chance.**
   - **The control:** the same pipeline trained to tell rest 20–40 s from rest 40–60 s. Both are rest, so anything it decodes is drift over time, not effort.
   - **The bar:** a mean of ≤ 60%. Above that, the verdict is **FAIL (confounded)** whatever the main number.

**Arm B passes (state in the live demo):** mean balanced accuracy ≥ 65%, p < 0.05, and the same drift control at ≤ 60%.

**Why these numbers:**
- **70%** is the usual two-class "usable" line, as for intent.
- **The published reference:** cross-person EEGMAT with no per-person baseline gives EEGNet 67.7% and the CBraMod foundation model 72.6% balanced accuracy. Those are 5-s samples and only 4 test people ([CBraMod, Table 12](https://arxiv.org/pdf/2412.07236); search-result level, verify on the PDF).
  - Our baseline is legitimate: the engine sees the person's first seconds.
  - So 70% asks for parity with the best published model, using a linear model on a few dozen features.
- **65% for arm B:** four channels lose information, and the demo only needs a clearly visible effect.

## What this can't show (goes into the results entry, and the claims audit if it passes)
- **Order is confounded with class.** Rest was always recorded before the task, in a separate file.
  - The drift control covers drift within a recording, not between the two recordings.
  - **The real control is live:** add 4 blocks (rest, arithmetic, rest, arithmetic; 60 s each, the order chosen by coin flip) to the founders' headset session. It costs about 5 minutes and no extra hardware.
- **The data is ICA-cleaned,** so live data will be noisier. The number is an upper estimate for a live stream.
- **One dataset.** Shin2017's mental-arithmetic set would be a second; it needs a download approval and isn't required for this bar.
- **"Mental effort" means serial subtraction.** Wording rules if it passes:
  - say "rest vs mental effort (arithmetic)";
  - never "stress", "focus" or "cognitive load" (claims audit row 27).

## What must be built first (no results; about half a day)
1. **The EEGMAT loader:**
   - read the EDF files;
   - rename T3–T6 to 10-10 names;
   - verify channels and rate against the registry;
   - add a `neuros validate` entry.
2. **The download:**
   - 76 files, 0.18 GB, from PhysioNet's mirror, each checked against PhysioNet's SHA256SUMS;
   - EEGMAT is on the approved download list (datasets 1–10), so this is that same download, never fetched only because the loader didn't exist.
3. **`neuros e6`** with `configs/e6.yaml`: the two models, the three arms, the drift control and the result file with provenance. Unit tests on synthetic data.

**Cost:** CPU only; minutes on the laptop (on mains power) or on AWS.

**Timing:** build on Mon 12 Oct, run on Tue 13, and the verdict feeds Q28 by Thu 15.
