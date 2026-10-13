# The 24 Oct freeze: what we measure, once, on the locked test set

**Written:** Tue 13 Oct 2026, before any freeze code or run. **Status: draft for the founders' approval** (the founders asked for it: "yes start the freeze plan"). Nothing touches the locked test set until this is approved and the rehearsal below passes.

## The decision
**Which numbers do the YC application and the technical report cite?** Every number marked ‡ in the [application draft](../yc/application-responses-v1.md) is a development number. Each is replaced by its locked-test counterpart, measured **once** with frozen code and the frozen model.

The model isn't on trial here: there's no pass/fail on it. What's fixed in advance is the measurement and the **wording each claim may use** given the number that comes out.

## What is frozen (by Fri 23 Oct night)
- **The model: `mvp-v0`.** A6 failed, so there's no `mvp-v1` ([result](../results/2026-10-13-a6-pretrained-backbone.md)).
  - `model.pt` SHA-256 `ab91416d…8011c7`.
  - Trained at `6a6968f` on 306 + 33 development people from 7 datasets.
  - Contract: 128 Hz, 8–30 Hz, common average, 2-s windows every 0.125 s.
- **Engine settings:** threshold θ 0.6, 0.5-s smoothing, debounce 2. These are the [10 Oct replay](../results/2026-10-10-mvp-v0-replay.md)'s settings.
- **The code:**
  - a git tag `freeze-2026-10-24` on the commit that runs, with CI green;
  - before tagging, the contract's default band switches to 8–30 Hz (D-034). That's housekeeping: `mvp-v0` already records 8–30 Hz.
- **The locked test set v2** (`configs/splits/locked_test_v2.yaml`): 101 people, unchanged since 4 Oct. Its loader checks the SHA-256.

## Who is scored
| Group | People | Use |
|---|---|---|
| **Cue-free locked people** | **16:** Yang2025 10, Shin2017A 6 | **The headline.** The only locked people whose windows don't show the class on screen |
| GuttmannFlury2025 locked | 6 | Replay only, reported separately (lateral cue, eye tracking: diagnostic) |
| Cued locked people | 72: PhysionetMI 22, Dreyer2023 17, Stieger2021 12, Lee2019_MI 11, Cho2017 10 | Technical report only: task-window accuracy next to the cue-only control. Never cited as a brain number |
| EEGMAT locked | 7 | Not used: no mental state ships (E6 failed) |

**16 people is a small headline.**
- With the development spread (about 9 points per person), a mean's 95% interval is roughly **±4–5 points**. Every locked number is quoted with its n and interval.
- BNCI2014-001 has no locked people, so its development numbers stay development-only.

## What is measured (each row replaces a ‡ number)
Every method is the development one, unchanged; only the model (the bundle instead of fold models) and the people differ.

| # | Measure | Method | Replaces |
|---|---|---|---|
| 1 | **Zero calibration over the first minutes** | `mvp-v0` with E2's warm-up protocol: unlabeled streaming at E2's budgets (0 to 120 s, and the full session), prior weight 10 s, 2-s windows every 0.125 s. Balanced accuracy per person; mean with 95% interval; people ≥ 70% | "Level about 60–70% for two commands"; the warm-up claim |
| 2 | **Calibration curve** | A per-person calibrated decoder (TS+LR) on 5/10/20/40 trials per hand and on half a session. Same time-ordered split as [10 Oct](../results/2026-10-10-calibration-curve.md) | "2 min of use ≈ 5–20 labelled trials per hand" |
| 3 | **3-minute calibration on top** | `mvp-v0` fine-tuned on the person's first 10 (and 20) trials per hand, with Q30's recipe and reference ([Q30](../results/2026-10-10-q30-finetune-q31-fit-check.md)), against the half-session calibration | "3-min calibration within 1–3 points of 10–13 min" |
| 4 | **Headset channel sets** | Measure 1 at 120 s on C3/Cz/C4, CP3/C3/C4/CP4 and FC5/FC6 vs the full layout, as in [E5](../results/2026-10-12-e5-virtual-headsets.md) | Claims audit row 29's wording ("about 3 points below a full research cap") |
| 5 | **Replay through the real engine** | `neuros simulate` on each replayable locked person, first session. Committed accuracy, commit rate, class shares, p99 latency, as in the 10 Oct replay | "61% of committed commands right; under 2 ms" |
| 6 | **Counts** | People per dataset in training and in the locked test | "339 training people … 73 cue-free test people" |

## Rules fixed now
1. **One run.**
   - The locked set is opened by one command (`neuros freeze`, to be built). It records the time, the code tag and the bundle checksum in its results, and leaves a marker on the AWS data disk. It refuses a second run.
   - If it crashes before writing any score (the log shows this), the crash may be fixed and the run repeated. Analysis code can't change after the tag.
2. **Nothing is tuned on it.** No threshold, prior weight, budget, model or window changes after seeing any locked number.
3. **Every number is reported**, including bad ones, in a results doc the same day.
4. **The application's wording follows the number:**

   | Claim | What it may say |
   |---|---|
   | Warm-up ≈ calibration | "Two minutes of use matched *N* labelled trials per hand." *N* is the largest calibrated size whose locked mean the 120-s mean reaches. Below 5 per hand, the claim becomes "fewer than 5" or is cut |
   | Calibration saved | The locked gap between the 3-minute fine-tune and the half-session calibration, with its interval. "Within 1–3 points" only if the locked gap is ≤ 3 points |
   | Level | The locked 120-s mean with its interval and n ("on 16 people it had never seen") |
   | Small headsets | The locked drop for C3/Cz/C4 vs the full layout |
   | Replay | Locked committed accuracy and p99 latency |
   | The cue finding (E0) | Kept as is: a finding about the benchmark (9 people, BNCI2014-001), quoted with its protocol |

5. **Development numbers** (the 73 people) stay in the knowledge base and the technical report, labelled as development, but not in the application.

## Dress rehearsal (by Tue 20 Oct; bar set now)
The same command, run on people `mvp-v0` never trained on and who aren't locked:
- **Shin2017A's 23 development people** (test-only, cue-free) for measures 1–4;
- **GuttmannFlury2025's 25 development people** for the replay. This also checks whether the simulator can play Shin2017A (its C3/C4 are derived channels); if it can't, Shin2017A drops out of measure 5.

**The rehearsal passes if all hold:**
- it runs end to end on AWS with no manual step;
- the guards fire when tested: it refuses a person who trained the bundle, a second locked run, and an edited locked file;
- on Shin2017A, the calibrated decoder reproduces the 10 Oct curve's numbers exactly (it's deterministic);
- `mvp-v0`'s 120-s mean is within 3 points of E2's fold models (57.1%). Both never saw these people; a bigger gap means a bug in the new path.

If it fails, it gets fixed before the tag. The freeze may slip to Sun 25 Oct, not later.

## Dropped from the original plan (founders' call)
**The "10-fold and LODO runs"** in the [road plan](road-to-nov-1.md) are dropped. The application cites neither, and E1 already reports leave-one-dataset-out. The time goes to the rehearsal instead.

## Schedule
| When | What |
|---|---|
| Tue 13 Oct | This plan; founders approve |
| Wed 14 – Sat 17 | Build `neuros freeze` (measures 1–6, the guards), with tests on synthetic data. Delete the old E2 scratch on AWS (approved 13 Oct) |
| Sun 18 – Tue 20 | Rehearsal on development people (AWS, about 1–2 h) |
| Wed 21 – Fri 23 | Buffer; tag `freeze-2026-10-24` on Fri 23 |
| **Sat 24** | **The one run** (AWS, about 1–2 h); results doc the same day |
| Sun 25 | Results page, a one-page technical report, and the application's ‡ numbers replaced |

## Cost
- About 2–3 days of code.
- About 3–4 hours on AWS in total, mostly data preparation. Every dataset needed is already on the AWS data disk.

## What this can't show
- **16 people is a thin headline.** The interval will be wide, and the application says n.
- **Yang2025 counts as cue-free on an assumption:** that the screen shows no class during imagery. Its paper doesn't say so; its white cross is the break cue (D-033).
- **This is offline, on recorded lab EEG.** The live headset (the founders' self-test and the pilot sessions) is measured separately, with its own bar.
