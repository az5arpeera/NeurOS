# A4: which four electrodes serve both commands and mental states?

**Date:** Sun 11 Oct 2026, run 22:24–22:51 UTC. **Verdict: FAIL** by the bar set before any code ([plan](../plan/a4-four-electrodes.md)). **Keep C3, Cz, C4, Pz** on the Ganglion.

**Run:** `results/a4/a4-20261011T225132Z.json` (AWS, code `b06bb80`, clean; 27 CPU-minutes).

## In plain English
- **On the selection half the best pick swapped one electrode: Cz for Fz.** That pick was C3, C4, Fz, Pz.
  - Mental states: 72.7% against today's 65.1%.
  - Commands: about the same (54.2% vs 52.9%).
- **It didn't hold up on the other half of the people.** On the confirmation half, both layouts scored at chance on mental states: 50.5% vs 49.4% (+1.1 points, p = 0.81). The bar needed +5 points with p < 0.05.
- **So, by the rule set beforehand:**
  - the Ganglion keeps C3, Cz, C4, Pz;
  - the headset runbook doesn't change;
  - we recommend no new layout to partners.
- **The honest reading: inconclusive more than negative.**
  - The confirmation half's 14 people sit at chance on mental states *even with today's layout*. Across all 29 people, E6 measured 56.1% for it.
  - So that half seems to hold most of the people for whom the state doesn't work (E6: about one in three), and with only 13 training people per fold, it had little chance of showing a difference either way.
  - Splitting 29 people in half made each half too small. That's a lesson for the next design, not a reason to re-run this one.

## The numbers
**Protocol:**
- **Candidates:** 136 four-electrode layouts, each C3 and C4 plus 2 of 17 positions.
- **States:** EEGMAT, E6's pipeline (TS+LR, 20-s rest baseline), leave one person out within each half.
- **Commands:** Yang2025's cue-free windows, zero-calibration TS+LR with each person's own unlabeled alignment, leave one person out within each half.
- **Halves:** fixed by seed. EEGMAT 15 / 14, Yang2025 21 / 20.

**Selection half (all 136 layouts):**

| Layout | States | Commands |
|---|---|---|
| Today: C3, Cz, C4, Pz | 65.1% | 52.9% |
| **The pick: C3, C4, Fz, Pz** (best states within 1 point of today's commands) | **72.7%** | 54.2% |
| Range over all 136 | 50.9–72.7% (median 66.2%) | 50.3–55.2% (median 54.0%) |

The next best states layouts (C3, C4, P3, T8; C3, C4, F8, Fp2; C3, C4, Cz, P8) score within 1 point of the pick: many layouts are about as good, so the ranking is noisy.

**Confirmation half (only the pick and today's layout):**

| | Pick | Today | Difference | Bar | |
|---|---|---|---|---|---|
| States (14 people) | 50.5% | 49.4% | +1.1 points, p = 0.81 | ≥ +5, p < 0.05 | ❌ |
| Commands (20 people) | 54.6% | 51.0% | +3.6 points, p = 0.008 | ≥ −1 | ✅ |

**Report only:**
- On the confirmation half, the pick (Fz in place of Cz) was also 3.6 points better on commands (p = 0.008).
- Commands with any four electrodes are only 51–55% with this simple model, so it's a small effect near chance. It's worth checking with our real model in E5 (virtual headsets), not a reason to move electrodes.

## What it means
1. **No change to the Ganglion layout before the founders' sessions.**
2. **For the mental-state direction (passive states, the Arctop goal):** four electrodes can't yet carry the state for most people with these methods. The selection half's 73% with one frontal electrode is encouraging, but it didn't replicate.
3. **For the next design** (post-YC agenda, experiment 4, needs its own bar):
   - score every layout with leave-one-person-out over all people, inside an outer split, instead of two small halves;
   - add a second states dataset, so selection and confirmation aren't 14 people each.

## Reproduce
```bash
uv run --no-sync neuros --config configs/data.yaml four-electrodes
```
