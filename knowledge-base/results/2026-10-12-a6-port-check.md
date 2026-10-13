# A6 port check: do REVE's published weights help on our windows?

**Date:** Mon 12 Oct 2026 (evening), run 00:30–00:40 UTC on 13 Oct on AWS (L4). **Verdict: FAIL** by the bar set before the run ([plan, 'Port check'](../plan/a6-pretrained-backbone.md)). The founders approved A6 on 12 Oct ([D-043](../decisions/decision-log.md)) and started AWS on 12 Oct ("AWS is started, go ahead").

**Runs:** `results/aws/a6/a6_reve_probe_pretrained-20261013T003031Z.json` and `results/aws/a6/a6_reve_probe_random-20261013T003031Z.json`, code `e08ef91` (clean), scored with `scripts/analysis/a6_port_check.py`.

## In plain English
- **The check:** REVE with its encoder frozen and only a small add-on layer trained (a "linear probe"), once with the published weights and once with random weights. A faithful port of a useful encoder should beat its random twin clearly. The bar was at least 5 points of validation accuracy.
- **The result:** pretrained **54.7%** vs random **52.6%**: **+2.2 points**. The bar was +5, so this fails.
- **What the failure means, by the rule set beforehand:** the REVE arm of A6 stops until the cause is found.
- **The code review since found no difference from the published model.** Every block of REVE's published model file (`modeling_reve.py` at `dc2a075`) was compared with ours, line by line:
  - the position embedding;
  - patching;
  - attention, norms and feed-forward;
  - the final layer, which is an identity there, as we have it;
  - attention pooling.
- **The more likely explanation is that REVE's frozen features carry little left/right information on our windows.** Our windows are filtered to 8–30 Hz and 2 s long; REVE was pretrained on unfiltered signals.
  - The sign: the training loss stayed at chance (0.688–0.691 against ln 2 = 0.693) in every fold, pretrained or random. The add-on layer couldn't even fit the people it was trained on.
  - The pretrained weights did win all 5 folds, which a badly broken port wouldn't usually do. But five folds can't prove it (sign test p = 0.06), and a subtle port error that only weakens the features can't be ruled out without running the authors' code. That was never approved, and nothing downloaded is executed.
- **MIRepNet isn't covered by this check.** It's a separate model and a separate port, pretrained on left/right imagery with our own filtering and alignment.

## The numbers
**Protocol:**
- **Data:** 10 people per dataset (`--max-subjects 10`), with E2's datasets, 5 folds, seed, windows and alignment.
- **Training:** the encoder frozen for all 4 epochs; the head trained with AdamW at lr 1e-3, label smoothing 0.1 and SWA.
- **Scored:** each fold's kept model on the fold's validation people (balanced accuracy).

| Fold | Pretrained | Random | Gain | Held-out people (task windows) |
|---|---|---|---|---|
| 0 | 55.9% | 54.2% | +1.7 | 56.6% vs 54.3% |
| 1 | 51.7% | 50.5% | +1.2 | 53.2% vs 54.0% |
| 2 | 56.0% | 54.0% | +2.0 | 54.4% vs 52.5% |
| 3 | 55.0% | 52.6% | +2.3 | 55.6% vs 53.9% |
| 4 | 55.0% | 51.5% | +3.5 | 55.2% vs 52.9% |
| **Mean** | **54.7%** | **52.6%** | **+2.2** ❌ (bar ≥ +5) | 55.0% vs 53.5% |

The held-out column includes cued datasets, so it's context, not part of the bar.

## What this can't show
- **Whether fine-tuning the whole encoder would help.** The full A6 run updates every REVE weight. But it was premised on REVE's pretraining carrying over, and here the frozen features are only 2 points better than random ones.
- **Whether unfiltered input would change the picture.** That's a different window from E2's, so even a pass there wouldn't give a paired comparison with E2.

## What happens next (founder decision, not yet taken)
**Proposed:**
- Park REVE with this result. It's also the slowest run (2–3 h), borderline on the 50 ms latency bar, and its published 64% rests on 2 test people with the cue on screen.
- Run the same check on MIRepNet before its full A6 run: pretrained vs random weights, frozen, ≥ 5 points, about 5 minutes. Then, if it passes, run MIRepNet's full A6 against the unchanged bar.
