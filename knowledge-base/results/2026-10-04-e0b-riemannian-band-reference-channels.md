# E0b, Riemannian part: band, reference and channel set (development numbers)

**Date:** Sun 4 Oct 2026, late evening (laptop, on mains power).
**Question (architecture §8.5, E0b):**
- the band (4–30, 8–30 or 4–40 Hz);
- the reference (common average over the channels present, or as recorded);
- since D-031, whether the motor strip alone is enough.

**Status:**
- Development numbers on development people only.
- These use Riemannian models, which see band power. EEGNet's version of E0b runs on AWS.

**Summary:**
- **8–30 Hz beats our 4–30 Hz contract band** on both datasets and with every method.
  - BNCI2014-001, cue-free, 9 people: +2.4 to +4.2 points.
  - PhysionetMI, 84 people: +1.1 to +2.5 points; p < 0.05 in 7 of 10 tests and p < 0.06 in 9.
  - 4–40 Hz is no better than 4–30.
- **The reference doesn't matter** (within ±0.5 points; no test significant).
  - Common average stays the default: E1 mixes datasets recorded against different references, and it must work for 4-channel headsets (§4.2).
- **The motor strip alone matches all 22 channels** on BNCI2014-001 (−0.1 points, p = 0.74). That's the third time, after both audits, that the motor strip holds up (D-031).
- **Decision:** [D-034](../decisions/decision-log.md). E1 uses 8–30 Hz on the motor strip with common average reference. The contract's default band changes once EEGNet confirms it.

## Protocol
- `neuros baseline`: TS+LR and MDM, without alignment and with EA or RA per recording session. Leave-one-person-out over development people.
- **Window:** D-030's task window.
  - BNCI2014-001: [1.5, 3.5] s after the cue (cue-free).
  - PhysionetMI: [1, 3] s (the target is still on screen).
- **Reported:** mean balanced accuracy over people. Tests are paired Wilcoxon over people.
- **Provenance:** BNCI2014-001 ran at commit `9015280`, PhysionetMI at `78b82ba`; both on clean code.
- **The cache** (§4.5) served the repeat runs: PhysionetMI runs took 276 s instead of 389 s once filled.

## BNCI2014-001 (9 people, cue-free window)
Ranges are over the reference × channel-set combinations (all 22 channels or the motor strip, 17).

| Band | TS+LR + EA | MDM + EA | TS+LR + RA | Calibrated TS+LR (within person) |
|---|---|---|---|---|
| 4–30 Hz (contract) | 60.4–61.2 | 58.5–59.4 | 59.7–61.1 | 70.6–72.7 |
| 4–40 Hz | 60.8–61.2 | 58.9–59.3 | 60.1–61.5 | 70.5–72.9 |
| **8–30 Hz** | **63.6–64.2** | **62.2–63.0** | **63.9–65.4** | **73.4–74.8** |

**8–30 vs 4–30 Hz**, per combination:
- TS+LR + EA: +2.4 to +3.9 points (p = 0.05–0.25, better for 5–6 of 9 people).
- MDM + EA: +3.4 to +4.2 (p = 0.012–0.078, 6–8 of 9).
- Every one of the 8 comparisons points the same way, but 9 people can't make it certain.

**Reference:** −0.5 points (p = 0.38). **Motor strip vs all channels:** −0.1 (p = 0.74).

## PhysionetMI (84 development people, motor strip, 21 channels)
| Reference | Band | TS+LR | TS+LR + EA | MDM + EA | TS+LR + RA | MDM + RA |
|---|---|---|---|---|---|---|
| common average | 4–30 Hz | 58.0 | 61.1 | 62.5 | 61.4 | 62.4 |
| common average | 4–40 Hz | 57.2 | 61.1 | 62.5 | 61.5 | 62.2 |
| common average | **8–30 Hz** | **60.0** | **63.0** | **64.1** | **63.1** | **64.8** |
| as recorded | 4–30 Hz | 57.5 | 61.4 | 62.2 | 62.1 | 61.5 |
| as recorded | 4–40 Hz | 57.5 | 61.4 | 61.9 | 62.5 | 61.7 |
| as recorded | **8–30 Hz** | **59.9** | **62.5** | **63.7** | **63.6** | **63.8** |

**8–30 vs 4–30 Hz:**
- **Common average:** +1.7 to +2.4 points.
  - Significant (p < 0.05) for TS+LR + EA (p = 0.027), MDM + EA (0.039), TS+LR + RA (0.028) and MDM + RA (0.0003).
  - TS+LR without alignment: p = 0.058.
- **As recorded:** +1.1 to +2.5 points.
  - Significant for TS+LR without alignment (p = 0.005), MDM + EA (0.023) and MDM + RA (0.002).
  - Not significant for TS+LR + EA (0.13) or TS+LR + RA (0.051).
- 44–50 of the 84 people improve, a few tie, and the rest get worse.

**4–40 vs 4–30 Hz:** −0.8 to +0.4 points. Nothing gained, and one test is significantly worse (TS+LR without alignment, common average: −0.8, p = 0.011).

**Reference:** common average vs as recorded, −0.3 to +0.5 points (p = 0.08–0.74).

## What it means
1. **Narrow the band to 8–30 Hz** for cross-person intent decoding. Two readings, which may both hold:
   - mu and beta carry the imagery, and 4–8 Hz adds activity that differs between people;
   - 4–8 Hz also carries slow eye and cue-evoked activity that the cue audit worries about. PhysionetMI's target is on screen during its window.
   Either way the narrower band is the safer one for cue-controlled numbers.
2. **Common average stays.** Its value is practical, not a measured gain: it puts datasets with different recording references on one footing.
3. **The motor strip is enough** (D-031). E1's input is C-core's motor-strip rows (20 channels), not all 26.
4. **Riemannian ≠ EEGNet.** EEGNet learns its own filters, so it may use a wider band differently. E0b's EEGNet part runs on AWS, with the same windows and the same people, before the contract's default band changes.

## Caveats
- Development numbers only, from two datasets. BNCI2014-001 has just 9 people.
- PhysionetMI's window isn't cue-free. Part of the 8–30 Hz gain there could come from shedding cue-related slow activity rather than from motor imagery.
- One seed, one split (leave-one-person-out).

## Regenerate
```bash
# BNCI2014-001: every reference x band x channel-set combination
uv run neuros --config configs/data.yaml baseline --dataset BNCI2014_001 --channels motor --reference car --alignment-unit session --band 8 30
# PhysionetMI: motor strip
uv run neuros --config configs/data.yaml baseline --dataset PhysionetMI --channels motor --reference car --alignment-unit session --band 8 30
```
Drop `--band` for 4–30 Hz, or use `--band 4 40`, `--reference as_recorded` and `--channels all` for the other cells. Results land in `results/baselines/<dataset>_left_hand-right_hand_<channels>_<reference>_session_aftercue[_band…].json`.
