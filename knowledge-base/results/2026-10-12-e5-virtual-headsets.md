# E5: virtual headsets: which hardware can we honestly say NeurOS supports?

**Date:** Mon 12 Oct 2026, run 18:41–18:57 UTC on AWS (16 GPU-minutes). **Verdict by the bar set before any code** ([plan](../plan/e5-virtual-headsets.md)): **no headset class is "supported".** The 3-channel class (Ganglion, Unicorn, 2b) is **experimental**, missing "supported" by 0.4 points. The **EPOC X class is not supported.** The founders approved the run on 12 Oct ("start the run on the computer and aws").

**Run:** `results/aws/e5/e5-20261012T184115Z.json`, code `f7601f3` (code clean; untracked data links only). E2's recipe, seed and folds.
- **The comparison is sound:** the research cap reproduced E2 exactly: 56.8% (Yang2025), 66.0% (BNCI2014-001) and 57.1% (Shin2017A) after 120 s.

## In plain English
- **On three electrodes (C3, Cz, C4: the Ganglion's class), the model keeps most of its accuracy.**
  - 55.7% over 73 never-seen people, clearly above chance.
  - **3.4 points** below a full research cap on the same people (55.1% vs 58.5%).
  - The bar for "supported" was at most 3 points, so it's **experimental**.
  - On BNCI2014-001 it's 62.0% (cap 66.0%); 2 of 9 people reach 70%.
- **The Emotiv EPOC X class doesn't work for commands.**
  - On its two motor-strip sites (FC5, FC6), the model scores **50.0%**: chance (p = 0.75), on all 41 people.
  - **So the EPOC X can't carry the commands demo** with this model, as expected from its layout ([EPOC X note](../explainers/emotiv-epoc-x.md)).
- **Losing Cz hurts more than expected.**
  - The Crown's layout (CP3, C3, C4, CP4) scores 51.7% and bare C3/C4 51.6%.
  - Both are statistically above chance but practically useless: nobody reaches 70% on CP3/C3/C4/CP4, and 1 person of 73 on C3/C4.
  - The model was trained on full caps; with Cz missing, what's left after the common average looks unlike anything it learned.
  - A model trained on few-channel inputs might do better ([A6](../plan/a6-pretrained-backbone.md) and the post-YC agenda).

## The numbers
**Protocol:**
- E2's pooled EEGNet fold models (5 folds, 339 development people).
- **Scored:** each cue-free development person after 120 s of unlabeled streaming.
- **Each set:** the windows cut to its channels (common average over the set, zeros elsewhere: the live engine's placement), aligned with a stream reference and population prior built on the same channels.
- **Research cap:** each dataset's own full profile at 120 s, paired on Yang2025 and BNCI2014-001 (50 people). Shin2017A's full profile *is* the 3-channel set.

| Set (headsets) | People | Mean | vs chance | Paired vs research cap (50 people) | Label |
|---|---|---|---|---|---|
| **C3, Cz, C4** (Ganglion, Unicorn, BCI IV-2b) | 73 | **55.7%** | p = 1e-8 | 55.1% vs 58.5%: **−3.4 points** (p = 5e-5) | **Experimental** (bar: ≤ 3) |
| CP3, C3, C4, CP4 (Crown) | 50 | 51.7% | p = 0.001 | 51.7% vs 58.5%: −6.7 (p = 2e-5) | Experimental (barely) |
| C3, C4 (report only) | 73 | 51.6% | p = 4e-4 | 51.5% vs 58.5%: −7.0 (p = 8e-6) | Experimental (barely) |
| **FC5, FC6** (Emotiv EPOC, EPOC+, EPOC X) | 41 (Yang2025 only) | **50.0%** | p = 0.75 | 50.0% vs 56.8%: −6.9 (p = 9e-6) | **Not supported** |
| Muse (no motor strip) | — | — | — | — | Not supported by construction (not run) |

**Per dataset** (mean; people at ≥ 70%):

| Set | Yang2025 (41) | BNCI2014-001 (9) | Shin2017A (23) |
|---|---|---|---|
| Research cap (E2) | 56.8% | 66.0% | 57.1% |
| C3, Cz, C4 | 53.6% (1) | 62.0% (2) | 57.1% (2) |
| CP3, C3, C4, CP4 | 51.6% (0) | 52.5% (0) | — |
| C3, C4 | 50.8% (0) | 54.4% (1) | 51.8% (0) |
| FC5, FC6 | 50.0% (0) | — | — |

## What it means
1. **The application and demo describe the Ganglion class as the plan set:** "works on a 4-electrode board, about 3 points below a full research cap." Not "supported", and not "any headset".
   - "Any headset" stays the mission, not a claim (D-042).
   - The claims audit's "any headset" wording now carries these numbers.
2. **The EPOC X:** not for commands. The [EPOC X note](../explainers/emotiv-epoc-x.md)'s option 1 holds:
   - use it for the live-pipeline bring-up (stream, quality, stage view);
   - keep a headset with C3, Cz and C4 for the commands demo, or the labelled replay ([Q33](../open-questions.md)).
3. **Cz matters for this model.** If the demo board has a spare channel, Cz is non-negotiable. The Ganglion runbook already places C3, Cz, C4 and Pz.
4. **Few-channel headsets need a model trained for few channels.** Two candidates:
   - A6's pretrained encoders (position-aware, so missing channels are absent rather than zero);
   - training with random channel dropout (a post-YC agenda idea).
   - Either needs its own bar.
5. **At the freeze (24 Oct):** the same table on the locked test set is the headline headset table (architecture §8.6).

## What this can't show
- **Virtual, not real.** No real consumer headset was tested: electrode type, noise and fit all differ. The EPOC X verdict rests on its layout, which no recording can change.
- **One model trained on full caps.** This measures our current model on fewer channels, not what's possible on fewer channels.
- **The EPOC row is Yang2025 only** (the only cue-free dataset with FC5/FC6).

## Reproduce
```bash
uv run --no-sync neuros --config configs/data.yaml e1 --experiment configs/e5.yaml --device cuda --workers 4 --threads 2
```
