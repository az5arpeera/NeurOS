# A6 MIRepNet port check: do MIRepNet's published weights work through our port?

**Date:** Mon 12 Oct 2026 (late evening), run 02:36–02:45 UTC on 13 Oct on AWS (L4). **Verdict: PASS** by the bar written before the run ([plan, 'MIRepNet port check'](../plan/a6-pretrained-backbone.md); [D-044](../decisions/decision-log.md)). The founders approved it on 12 Oct ("yes park reve and run the mirepnet check") and asked for the full run on a pass ("if it passes start the full mirepnet run").

**Runs:**
- `results/aws/a6/a6_mirepnet_probe_pretrained-20261013T023656Z.json`
- `results/aws/a6/a6_mirepnet_probe_random-20261013T024151Z.json`

Code `bbdfc11` (clean); scored with `scripts/analysis/a6_port_check.py`.

## In plain English
- **The check:** MIRepNet with its encoder frozen and only a new output layer trained, published weights against random weights. It's the same check REVE failed ([result](2026-10-12-a6-port-check.md)), with the same recipe and people.
- **The result:** pretrained **64.1%** vs random **54.0%** validation accuracy: **+10.0 points** (bar +5). It won all 5 folds, by +4.4 to +14.1.
- **So the port works,** and the full A6 MIRepNet run started at 02:45 UTC.
- **It also shows real transfer, not just memory.** MIRepNet was pretrained with labels on four of our training datasets (PhysionetMI, Dreyer2023, Lee2019, Cho2017), which can flatter this check. But on all six datasets it never saw, frozen pretrained features beat random ones too, by +3.7 to +12.8 (held-out people below). That includes the three that A6 scores:
  - Yang2025 +3.7, the smallest gain;
  - BNCI2014-001 +9.0;
  - Shin2017A +10.2.
- **The contrast with REVE is the point.** MIRepNet's frozen features already separate left from right (training loss 0.61–0.63, against 0.69 at chance). REVE's didn't (0.69). MIRepNet was built for imagined movement with our own preprocessing; REVE is general-purpose and was pretrained on unfiltered signals.
- **This is not an A6 result.** The A6 bar compares the fine-tuned model with E2's EEGNet, person by person after 120 s of streaming, on the full data. The held-out numbers here come from task windows (several datasets keep the cue on screen), 10 people per dataset and a frozen encoder.

## The numbers
**Protocol:** 10 people per dataset (`--max-subjects 10`), with E2's datasets, 5 folds, seed, windows and alignment. The encoder was frozen for all 4 epochs, and the head trained with AdamW at lr 1e-3, label smoothing 0.1 and SWA. The REVE check's recipe was unchanged. Each fold's kept model was scored on the fold's validation people (balanced accuracy).

| Fold | Pretrained | Random | Gain | Held-out people (task windows) |
|---|---|---|---|---|
| 0 | 62.7% | 53.4% | +9.3 | 66.8% vs 56.6% |
| 1 | 56.2% | 51.9% | +4.4 | 65.6% vs 55.8% |
| 2 | 69.6% | 55.7% | +13.9 | 67.7% vs 54.9% |
| 3 | 68.6% | 54.5% | +14.1 | 64.1% vs 56.2% |
| 4 | 63.1% | 54.7% | +8.5 | 62.7% vs 54.3% |
| **Mean** | **64.1%** | **54.0%** | **+10.0** ✅ (bar ≥ +5) | |

**Held-out people by dataset** (context, not the bar):

| Dataset | People | Pretrained | Random | Gain | In MIRepNet's labelled pretraining?¹ |
|---|---|---|---|---|---|
| Yang2025 | 10 | 55.8% | 52.0% | +3.7 | no |
| BNCI2014-001 | 9 | 68.3% | 59.4% | +9.0 | no |
| Shin2017A | 10 | 63.5% | 53.3% | +10.2 | no |
| PhysionetMI | 10 | 64.5% | 54.3% | +10.2 | yes |
| Dreyer2023 | 10 | 61.8% | 53.5% | +8.4 | yes |
| Lee2019_MI | 10 | 71.0% | 56.0% | +15.1 | yes |
| Cho2017 | 10 | 73.0% | 59.4% | +13.6 | yes |
| Stieger2021 | 10 | 67.7% | 54.9% | +12.8 | no |
| BNCI2014-004 | 9 | 62.0% | 55.9% | +6.1 | no |
| GuttmannFlury2025 | 10 | 66.3% | 57.1% | +9.2 | no |

¹ Per the [12 Oct review](../research/pretrained-eeg-models-2026.md): 7 MOABB datasets, 330 people, 4 of them ours. Stieger2021 and GuttmannFlury2025 aren't among the datasets its released code handles. BNCI2014-004 is one of its downstream test sets, not a pretraining one.

## Operational notes
- **The first random-weights run crashed** on GPU memory. Both arms ran at once (4 MIRepNet processes on one 22 GB GPU), and a validation pass needs several GB each. It was re-run alone, unchanged. The crashed run's partial file (`…-20261013T023656Z.json`, incomplete) isn't used, and its log was kept as `a6_mprobe_random-oom.log`.
- **So the full run uses 2 workers, not the config comment's 4.** A MIRepNet worker peaks near 8 GB, so 4 wouldn't fit. Worker count changes only speed, because each fold trains independently and deterministically.
- **Disk:** the finished checks' scratch folders were moved to the AWS root disk (`~/pooled-archive`), not deleted.
