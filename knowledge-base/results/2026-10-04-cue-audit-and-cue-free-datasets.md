# Cue audit of three datasets, and the cue-free datasets we could add (development numbers)

**Date:** Sun 4 Oct 2026, the two tasks after D-030 and D-031.
**Status:** development numbers (no locked test people). **Q26 approved the same evening:** all three candidates were added ([D-033](../decisions/decision-log.md)). Three facts in §2 were corrected while registering them (see the note under the table).

**Summary:**
- **Nothing leaks.** Before the cue, every dataset is at chance. Two small effects (1.4 and 2.4 points) are reported as warnings, below the 3-point leak threshold.
- **The cue's onset is readable, from where you'd expect.** PhysionetMI's side target is read best from the **eye channels (66.5%)**: people glance toward it. BNCI2014-004's central arrow isn't readable from C3, Cz and C4 at all (50.6%).
- **In the training window, the motor strip carries the class:**
  - 60.9% on BNCI2014-001 (cue-free) and 61.4% on PhysionetMI (target still on screen).
  - On both, the motor strip alone matches or beats all channels (60.9 vs 60.4; 61.4 vs 59.7). That supports D-031's motor-strip intent head.
- **PhysionetMI's training window still carries some eye signal** (56.6% from the eye channels). It stays usable for training on the motor strip, and is never quoted without its controls.
- **Cue-free data exists.**
  - Yang2025 (51 people, CC BY 4.0, 65.6 GB) is the clean addition. Its paper doesn't say what the screen shows during imagery, so the audit confirms it on the data.
  - Shin2017A (29 people) is cue-free but GPL-3.0.
  - GuttmannFlury2025 (31 people, CC0) isn't cue-free, but it records gaze.
  - Three other candidates are non-commercial.

## 1. The audit (`neuros audit`, D-030)
- **Classifier:** TS+LR with EA per recording session, leave-one-person-out over the development people: E0's Riemannian baseline.
- **Windows,** in seconds from the cue's onset: pre-cue [−0.5, 0] (a leak check), cue onset [0, 0.5], and each dataset's D-030 task window.
- **Scalp regions:** eyes (Fp, AF, F7/F8, F9/F10), motor (the FC, C and CP rows), parietal (P row), visual (PO, O, Iz), and all channels. A region with fewer than 2 channels is skipped.
- **Rules:**
  - a pre-cue effect is a *leak* if its CI excludes chance by at least 3 points, and a *warning* if smaller;
  - a dataset is *cue-dominated* if the eyes or visual region come within 2 points of the motor strip on the task window.

Balanced accuracy %, mean over people.

### BNCI2014-001 (9 people, 22 channels; no eye or occipital channels)
| Window | Cue-free | All (22) | Motor (17) | Parietal (3) |
|---|---|---|---|---|
| Pre-cue [−0.5, 0] | ✅ | 48.0 | 46.3 | 49.5 |
| Cue onset [0, 0.5] | ❌ | 59.1 | 56.5 | 56.4 |
| **Task [1.5, 3.5]** | ✅ | 60.4 | **60.9** | 54.7 |

The task row reproduces diagnostic 3's Riemannian number (TS+LR + EA, 60.4%). POz, the only visual channel, is skipped.

### BNCI2014-004 (9 people, C3, Cz and C4 only)
| Window | Cue-free | Motor = all (3) |
|---|---|---|
| Pre-cue [−0.5, 0] | ✅ | 51.4 ⚠️ (CI [50.7, 52.2]) |
| Cue onset [0, 0.5] | ❌ | 50.6 |
| Task [2.5, 4.5] | ❌ smiley feedback in sessions 3–5 | 65.6 |

- **The pre-cue warning:**
  - The first run (code `3ceaa50`) called it a leak, which led to the 3-point threshold (`4ceda70`).
  - Classes switch on 50.5% of consecutive trials, so trial order carries nothing.
  - A timing or label bug would appear at the task window's size (65.6%). And the cue onset itself is at chance here.
  - With 720 trials per person, a 1.4-point effect passes the CI test. Across the ≈10 pre-cue cells tested, one false alarm like this is expected.
- **The cue onset isn't readable from the motor strip alone,** consistent with the onset response living outside it.

### PhysionetMI (84 development people, 64 channels)
| Window | Cue-free | All (64) | Eyes (10) | Motor (21) | Parietal (9) | Visual (9) |
|---|---|---|---|---|---|---|
| Pre-cue [−0.5, 0] | ✅ | 51.0 | 50.1 | 50.3 | 52.4 ⚠️ | 51.6 |
| Cue onset [0, 0.5] | ❌ | 63.0 | **66.5** | 58.2 | 53.6 | 53.0 |
| **Task [1, 3]** | ❌ target on screen | 59.7 | 56.6 | **61.4** | 53.3 | 51.6 |

95% CIs: cue-onset eyes [64.4, 68.6]; task motor [58.5, 64.5]; task eyes [54.8, 58.5]; pre-cue parietal [50.9, 53.9].

**What it shows:**
- **The target's first half-second is read best from the eye channels.** People glance toward the side the target appears on.
  - The visual channels barely decode it (53.0%): covariance features see band power, not the brief evoked response.
  - The motor strip's 58.2% may partly be that same eye signal: its front row sits close to the eyes.
- **In the training window, the motor strip leads (61.4%),** but the eyes still decode the class (56.6%): sustained or repeated glances while the target stays up.
- **Extra channels don't help:** all channels give 59.7%, against 61.4% for the motor strip alone.
- **The pre-cue parietal warning (52.4%)** may mean the logged event time lags the target's appearance slightly: the regions that respond to the target are the ones nudged. Check with narrower pre-cue windows before relying on PhysionetMI's exact timing.

## 2. The cue-free candidates (checked against papers and repositories, 4 Oct)
| Dataset | Left/right people | On screen during imagery | Cue-free? | Licence | Size | Use |
|---|---|---|---|---|---|---|
| **Yang2025** (WBCIC-SHU) | 51 (+11 three-class), 3 sessions × 200 trials | A 1.5-s video-and-sound cue, then imagery with no feedback. What the screen shows during imagery isn't stated (corrected: the white cross on black marks the break *after* it) | ✅ likely ("no unnecessary visual and auditory stimuli"); the audit checks it | CC BY 4.0 (paper, Figshare) | 65.6 GB | **Training and cue-free testing** |
| Shin2017A | 29 | A central arrow for 2 s, then a fixation cross for 10 s | ✅ per MOABB's account; the paper is paywalled | GPL-3.0 (TU Berlin page) | 6.26 GB (six zips; corrected from "small"; they also hold dataset B) | **Testing only**, until a licence decision |
| GuttmannFlury2025 | 31 | A red rectangle on the left or right for the whole 4 s; a 300 Hz eye tracker | ❌ (paper) | CC0 for the data (paper's Data Records, Zenodo); the *article* is CC BY-NC-ND | 26.2 GB (31 per-person zips with gaze and eye video) | **Diagnostic**: how much decoding follows gaze |
| Brandl2020 | 16 | A spoken cue, nothing on screen | ✅ | CC BY-NC-ND 4.0 | | ❌ non-commercial |
| HefmiIch2025 | 37 patients | A 2-s arrow and beep, then 10 s of imagery | likely | CC BY-NC-ND 4.0 | | ❌ non-commercial |
| Chang2025 | 28 | A square whose side signals the class; persistence not stated | ❌ likely | CC BY-NC-ND 4.0 (paper; MOABB says CC BY) | | ❌ |

**Corrections made while registering the datasets (D-033, 4 Oct evening):**
- **Yang2025's screen during imagery:** the table first said "a white fixation cross". Re-reading the paper: the white cross on black is the *break* cue after imagery. During imagery the paper only says there are "no unnecessary visual and auditory stimuli", and its figure is too small to read. So Yang2025 is cue-free *if* the screen is blank or neutral then; the audit's eye and visual channels will show it.
- **Shin2017A's size:** six zips, 6.26 GB in all (each also holds dataset B), not "small".
- **GuttmannFlury2025's licence:** the data is CC0 (Data Records section), but the article is CC BY-NC-ND. We use the data, not the article's figures. MOABB's Zenodo re-host includes the Tobii gaze files and eye video.
- **GuttmannFlury2025's electrodes:** its authors recommend excluding 11 faulty or bridged electrodes, including F3, FC3 and P4. The registry drops them.

**Lessons:**
- **MOABB's metadata isn't a source of record.** It gets Chang2025's licence wrong and repeats BNCI2014-001's text for BNCI2014-004.
- **Every timing and licence fact is checked at the paper or repository** before a dataset enters the registry.

**With Yang2025 and Shin2017A,** the cue-free pool is ≈ 89 people from 3 setups (98 from 4 with BNCI2014-004's screening sessions), against 9 today.

## 3. What it means
1. **The audit works, and nothing leaks.** Every dataset added from now on gets an audit before use (D-030).
2. **Motor-strip channels for the intent head** (D-031) are supported on both multi-region datasets. E0b checks the same with EEGNet.
3. **PhysionetMI stays in the training pool,** on the motor strip and D-030's late window, flagged. Its eye contamination is now measured (56.6%), not assumed.
4. **Cue-free evidence:** Yang2025 first, Shin2017A for testing only, GuttmannFlury2025 to measure gaze.

## Caveats
- **Riemannian features see band power, not waveforms.** On BNCI2014-001's cue onset, EEGNet read 75.8% where TS+LR read 59.1%. These audits are therefore a lower bound on how readable the cue is to a deep model. Repeat the audit with EEGNet once the E1 runner exists (cheap on AWS).
- **Region scores are evidence, not proof:** volume conduction spreads every source.
- **Development people only:** PhysionetMI's 84 of 106 usable people (the locked test set's 22 are excluded).

## Regenerate
```bash
uv run neuros --config configs/data.yaml audit --dataset BNCI2014_001
uv run neuros --config configs/data.yaml audit --dataset BNCI2014_004
uv run neuros --config configs/data.yaml audit --dataset PhysionetMI
```
- Outputs: `results/audit/<dataset>_left_hand-right_hand_as_recorded_session_aftercue.json`, with per-person scores per cell, region channel lists, findings and provenance.
- **Provenance:** commit `78b82e9`, clean; 3, 2 and 19 minutes with 11 fold workers.
  - The first PhysionetMI attempt ran its folds one at a time and was stopped at the 2-hour background limit after 10 of 15 cells.
  - That run is why folds now run in parallel and the file is saved after every cell.
