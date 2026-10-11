# E6: can NeurOS tell rest from mental effort across people, with no calibration?

**Date:** Sun 11 Oct 2026. **Verdict: FAIL** on both arms, by the bar set before any data was loaded ([plan and bar](../plan/e6-state-channel.md)).

**Cost:** about 2 CPU-minutes on the laptop (mains power). **Result file:** `results/e6/e6-20261011T202026Z.json` (code `50304e3`, clean).

## In plain English
- **On a full 19-electrode cap it works for about two people in three, not for everyone.**
  - Average 70.9% right on 2-s windows, after a 20-s rest at the start (29 people the model never saw).
  - That's level with the best published model on this dataset.
  - But only 18 of 29 people reach 60%. The bar asked for 20.
  - The rest sit at chance: it's a works-or-doesn't split, not a weak signal everywhere.
- **On the Ganglion's four motor-strip electrodes (C3, Cz, C4, Pz) it barely works:** 56.1%, against a bar of 65%.
- **So, by the rule set beforehand:**
  - no mental state ships in `neuros.intent.v0`;
  - claims-audit row 27 stands;
  - the YC demo shows commands only;
  - the application doesn't claim a validated mental state.
- **What it does tell us:**
  - The signal is real where it exists. Drift controls sit at chance, and the 20-s baseline is worth 8 points.
  - It lives over the front and back of the head, not the motor strip. A headset that wants both commands and states needs electrodes in both places (post-YC agenda, experiment 4).

## The numbers
**Protocol:**
- **People:** EEGMAT's 29 development people (7 locked test people untouched), leave one person out.
- **Data:** rest 20–60 s against the 62-s task recording, 2-s windows every 0.5 s.
- **Score:** per-person balanced accuracy, mean over people.
- **Baseline:** "baseline" means the person's first 20 s of rest normalise their data; "none" means no information about the test person at all.

| Model / condition | Arm | Mean | People ≥ 60% | p vs 50% | AUROC | Drift control |
|---|---|---|---|---|---|---|
| **TS+LR / baseline (primary)** | **A: all 19** | **70.9%** | **18/29** | 9.5e-5 | 0.746 | 48.7% |
| TS+LR / baseline | B: C3, Cz, C4, Pz | 56.1% | 10/29 | 0.01 | 0.655 | 52.3% |
| TS+LR / baseline | C: Fz, F3, F4, Pz, P3, P4 | 66.7% | 23/29 | 2.6e-7 | 0.758 | 56.0% (p = 0.018) |
| Band power / baseline | A | 67.0% | 21/29 | 5.1e-7 | 0.744 | 50.7% |
| Band power / baseline | B | 53.0% | 7/29 | 0.088 | 0.598 | 52.9% |
| Band power / baseline | C | 59.9% | 14/29 | 3.7e-4 | 0.696 | 51.9% |
| TS+LR / none | A | 62.9% | 14/29 | 3.2e-4 | 0.724 | — |
| Band power / none | A | 60.2% | 11/29 | 1.2e-5 | 0.689 | — |

**The bar** (primary model = the higher arm-A mean, TS+LR):

| Bar | Needed | Got | |
|---|---|---|---|
| **Ship, arm A:** mean | ≥ 70% | 70.9% | ✅ |
| **Ship, arm A:** p vs 50% | < 0.05 | 9.5e-5 | ✅ |
| **Ship, arm A:** people at ≥ 60% | ≥ 20 of 29 | 18 | ❌ |
| **Ship, arm A:** drift control | ≤ 60% | 48.7% | ✅ |
| **Demo, arm B:** mean | ≥ 65% | 56.1% | ❌ |
| **Demo, arm B:** p vs 50% | < 0.05 | 0.01 | ✅ |
| **Demo, arm B:** drift control | ≤ 60% | 52.3% | ✅ |

**Spread (TS+LR, arm A):**
- Nine people sit at 43–52%: chance.
- Two are at 55–57%.
- Eighteen are at 61–100%; the median of all 29 is 78%.
- Someone who works tends to work well. Someone who doesn't, doesn't.

**Report only:**
- **10-s blocks** (the mean probability over each 10-s block, the unit a live state display would use): arm A 74.6% (TS+LR) and 77.4% (band power), per-person balanced. Arm B 53.0%.
- **Late-rest control:** rest 20–60 s against the last 40 s of the same rest recording, about 2 minutes later. 49.6% (arm A) and 48.8% (arm B): two minutes of within-recording drift isn't decodable.
- **Published reference** (different protocol: 5-s samples, 4 test people, no per-person baseline): EEGNet 67.7%, CBraMod 72.6% ([CBraMod, Table 12](https://arxiv.org/pdf/2412.07236); search-result level).

## Data checks (run first)
- **The filter:** power at 4–13 Hz is 55× the power at 30–45 Hz (median over people), so theta and alpha are present. The EDF headers give 0.5–45 Hz. The page's "30 Hz high-pass" is wrong.
- **Eyes:** parietal-occipital alpha at rest is 1.87× the task's (median; 14 of 29 people above 2×). That's under the bar's 2× flag, so the primary arm stands.
  - Some people may have closed their eyes at rest. It isn't what's being decoded, though: accuracy doesn't track the ratio (Spearman 0.13, p = 0.50).
- **File facts:** rest recordings run 170–188 s for these people (80 s for one locked person), not the page's 60 s. The plan was amended before any model ran.

## What this can't show
- **Order is still confounded with class.** Rest was always recorded before the task, in a separate file.
  - The within-recording drift controls are at chance, but the gap *between* the files is untestable here.
  - The live control (rest / arithmetic blocks in random order) belongs in the founders' headset session.
- **The data is ICA-cleaned,** so a live stream will be noisier.
- **One dataset; "effort" means serial subtraction.**

## What it means
1. **The rule set beforehand applies:**
   - nothing ships;
   - `neuros.intent.v0` stays a command, a confidence and a hold with its reason;
   - the demo shows commands only.
2. **For Q28's fallback headline:** "a validated mental state" is not available. The rigour story and "calibration saved" carry the application.
3. **For the long game (the founders want passive states, to compete with Arctop):**
   - **Cross-person passive state decoding works for about two-thirds of people on a full cap.** Arctop and Zander don't publish a number like this.
   - **The failing third and the headset are the work.** Electrode placement is the cheap lever: the frontal-parietal arm got 23 of 29 people to 60% with six electrodes.
   - The post-YC agenda's experiments 2 (why some people fail), 4 (the best four electrodes for commands *and* states) and 12 (mental states against each person's own baseline) now have a concrete starting point.

## Reproduce
```bash
uv run --no-sync neuros --config configs/data.yaml download --dataset EEGMAT
uv run --no-sync neuros --config configs/data.yaml validate --dataset EEGMAT
uv run --no-sync neuros --config configs/data.yaml e6
```
