# A6: does a pretrained encoder beat our from-scratch EEGNet?

**Date:** run Mon 12 Oct 2026 22:45–23:55 Eastern (02:45–03:55 UTC on 13 Oct) on AWS (L4); scored Tue 13 Oct. **Verdict: FAIL** by the bar set before any run ([plan](../plan/a6-pretrained-backbone.md)). **`mvp-v0` stays** the freeze candidate. Pretrained encoders move to the [post-YC agenda](../plan/post-yc-research-agenda.md) with these numbers.

**History:**
- Approved 12 Oct ([D-043](../decisions/decision-log.md)).
- REVE was parked after its port check ([D-044](../decisions/decision-log.md); [result](2026-10-12-a6-port-check.md)).
- MIRepNet passed its port check ([result](2026-10-12-a6-mirepnet-port-check.md)), and the founders asked for its full run.

**Run:** `results/aws/a6/a6_mirepnet-20261013T024545Z.json`, code `bbdfc11` (clean), recipe `configs/a6_mirepnet.yaml`, 2 workers. Scored with `scripts/analysis/a6_bar.py` against E2's models: E5's file, which reproduced E2 exactly.

## In plain English
- **A published motor-imagery foundation model, fine-tuned on our data, ties our own small model.** Fine-tuned MIRepNet scored **58.9%** on E2's 73 cue-free never-seen people after 120 s of unlabeled streaming. EEGNet trained from scratch on the same people and windows scored **58.0%**. That's **+0.9 points, p = 0.23**: not the +3 the bar asked for, and not significant.
  - It didn't lose on any dataset: Yang2025 +0.6, BNCI2014-001 +0.2, Shin2017A +1.6.
  - It's fast enough: p99 9.3 ms against the 50 ms bar.
  - **But on gain it fails, so `mvp-v0` stays.**
- **Same story on smaller headsets.** On C3/Cz/C4 (the Ganglion class), MIRepNet scored 56.6% vs EEGNet's 55.7% (+0.9, p = 0.34). On the EPOC X's FC5/FC6, both sit at chance. With C3/C4 alone, MIRepNet is slightly worse (−1.4, p = 0.015).
- **Pretraining carries left/right information; our 339 people already teach the model about as much.**
  - With its encoder frozen, MIRepNet's published features beat random ones by 10 points ([port check](2026-10-12-a6-mirepnet-port-check.md)). So the pretraining isn't empty.
  - MIRepNet was pretrained on about 330 people, about as many as we train on, which may be why it adds little here.
  - For context, the field's benchmarks find the same: foundation models' gains over EEGNet in cross-person motor imagery are small or not significant (the [pretrained-model review](../research/pretrained-eeg-models-2026.md)).
- **One caveat changes how far this generalises: the kept model in every fold was the epoch-1 model.**
  - MIRepNet's own recipe fine-tunes every weight at lr 1e-3. Validation loss was lowest after the first epoch, rose after that, and early stopping ended each fold at epoch 11.
  - So this tested "MIRepNet plus one epoch of its authors' fine-tuning", as the plan specified. A gentler recipe is untested: a lower learning rate for the encoder, or freezing it first. That's a post-YC question, not a reason to re-run before the freeze.
- **The cost would have been real anyway:** 5.15M parameters vs EEGNet's 20k (258×), and 9.3 ms vs 1.1 ms per decision.

## The numbers
**The bar (all must hold; set 12 Oct before any download):**

| Check | MIRepNet | Bar | |
|---|---|---|---|
| Gain over E2, 73 cue-free people at 120 s | **+0.90 points** (58.9% vs 58.0%), wins 39 of 73 | ≥ +3.0 | ❌ |
| Paired Wilcoxon (two-sided) | p = 0.23 | < 0.05 | ❌ |
| No dataset loses | Yang2025 57.5% vs 56.8%; BNCI2014-001 66.2% vs 66.0%; Shin2017A 58.7% vs 57.1% | each ≥ E2 | ✅ |
| p99 latency, laptop CPU (median of 3 runs) | 9.3 ms (runs 9.0, 9.3, 10.7) | < 50 ms | ✅ |
| Licence allows shipping a fine-tuned copy | MIT | yes | ✅ |

People at or above 70%: Yang2025 6 vs 5, BNCI2014-001 3 vs 3, Shin2017A 2 vs 2.

**Also reported: E5's headset channel sets** (after 120 s, paired with E2's models; E5's labels for MIRepNet's numbers):

| Set | People | MIRepNet | EEGNet (E5) | Difference |
|---|---|---|---|---|
| C3, Cz, C4 (Ganglion class) | 73 | 56.6% | 55.7% | +0.9 (p = 0.34) |
| CP3, C3, C4, CP4 (Crown layout) | 50 | 51.9% | 51.7% | +0.2 (p = 0.66) |
| C3, C4 | 73 | 50.2% | 51.6% | −1.4 (p = 0.015) |
| FC5, FC6 (EPOC X) | 41 | 50.1% | 50.0% | +0.1 (p = 0.99) |

**Also reported: training:**
- **Folds:** each took 26 minutes with 2 workers sharing the L4 (12 minutes for the last fold, alone). E5's EEGNet folds took 3.5–9 minutes with 4 workers.
- **Stopping:** the kept epoch was 1 in every fold, with 11 epochs trained (patience 10).
- **Validation and held-out accuracy:** final validation was 66.0% vs EEGNet's 65.9%. Held-out people on task windows (cued datasets included) were 66.1% vs 65.8%.

## What this can't show
- **Other recipes.** This is one recipe (the authors') and one seed, as planned. The epoch-1 stopping says the recipe overfits our data quickly, so a lower encoder learning rate or a frozen start might do better. Untested.
- **REVE fine-tuned.** It was parked after its frozen port check failed, so a fully fine-tuned REVE wasn't run.
- **Small training sets.** Pretraining might matter more when there's little data: a new headset class with few recorded people, or one person's calibration. With only 10 people per dataset, the frozen MIRepNet probe reached held-out accuracy in the same range as EEGNet trained on everyone. But those were different people and datasets, so it's only a hint for the post-YC agenda.

## What happens next
**By the rule set beforehand:**
- **No `mvp-v1`.** No bundle and no replay; the founders' "if it passes … bundle it" doesn't trigger.
- **`mvp-v0` stays** for the 24 Oct freeze.
- **Bundle version 2** (which can hold a pretrained encoder) stays in the code for later.

The post-YC agenda gets the open questions, each needing its own bar:
1. a gentler fine-tuning recipe;
2. pretraining's value with little data;
3. REVE on broadband input.
