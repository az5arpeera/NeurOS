# Experiment register

**Every experiment in one place:** the decision it answers, the hypothesis, the pass bar, the verdict, and where the full write-up lives. Started 11 Oct 2026; the earlier rows are collected from where their bars were first written.

## The rule
- **Before a run:**
  - write its decision and pass bar;
  - add a row here marked *bar set*;
  - get the founders' go-ahead.
- **After the run:**
  - add the verdict here;
  - write the full entry in [`results/`](results/README.md).
- Failed runs stay, marked as such.

## Where the pieces live

| What | Where |
|---|---|
| This index | `knowledge-base/experiments.md` (you are here) |
| The full decision and bar of a new experiment | **`knowledge-base/plan/<id>-<name>.md`**, one file each, as in [E6](plan/e6-state-channel.md) and the [headset self-test](plan/headset-self-test.md) |
| Older bars (before 11 Oct) | Inline in the [road plan](plan/road-to-nov-1.md), the [open questions](open-questions.md) (Q28, Q30, Q31), and the [architecture's §8 table](architecture/mvp-architecture.md) |
| Results, with protocol, N and the command that regenerates them | [`results/`](results/README.md), one file per run |
| The numbers per person | `results/<experiment>/*.json` (git-ignored; each records the code version, config hash and package versions) |
| The code | `src/neuros/eval/` (one module per experiment family), `configs/<experiment>.yaml` |
| Decisions a result triggered | [Decision log](decisions/decision-log.md) |

## Done

| ID | Decision / question | Hypothesis or bar (set before running) | Verdict | Write-up |
|---|---|---|---|---|
| E0 | Does our pipeline reproduce the literature? | EEGNet + alignment on BNCI2014-001 within ±3 points of 74.3% | **PASS** (75.0%). But after the arrow leaves the screen, **60.6%**: the 75% was mostly the cue (D-030) | [results](results/2026-10-03-e0-eegnet-bnci2014-001.md) |
| E0b | Which frequency band, reference and channels? | Pick on development folds | **8–30 Hz, motor strip** (D-034) | [results](results/2026-10-04-e0b-riemannian-band-reference-channels.md) |
| Cue audit | Do cued datasets leak the class before or around the cue? | Chance before the cue | **No leaks.** Eye channels still decode PhysionetMI at 56.6% | [results](results/2026-10-04-cue-audit-and-cue-free-datasets.md) |
| E1 | Does pooling datasets beat single-dataset training? Do unseen datasets cost accuracy? | Pooling must help, or fix harmonisation first | **PASS:** +0.9 points (+1.3 over 3 seeds); unseen datasets cost nothing | [results](results/2026-10-06-e1-pooled-single-lodo-cue-only.md), [follow-ups](results/2026-10-06-e1-followups-cue-test-seeds-ceilings.md) |
| E2 + H1 | How fast does the warm-up reach usable accuracy? Is starting from rest better? | S ≤ 45 s to get within 2 points of full-session alignment | **FAIL:** S = 120 s or longer; 10 of 73 people at 70% after 2 min. H1 not supported | [results](results/2026-10-09-e2-warmup-curve.md) |
| D-039 | Do training crops or extra classes lift cue-free accuracy at 120 s? | About 65% on Yang2025 and Shin2017A (Q28) | **FAIL:** +0.3 to +0.4 | [results](results/2026-10-10-d039-crops-multiclass.md) |
| Calibration curve | How much conventional calibration is our zero-calibration model worth? | A measurement, no bar | 2 min of use ≈ **5–20 labelled trials per hand** | [results](results/2026-10-10-calibration-curve.md) |
| Q30 | Does our model plus 10 trials per hand (~3 min) match a 10–13-min calibration? | Not significantly worse (lenient); at or above (strict) | **Lenient PASS, strict FAIL**, giving "calibration saved" (D-040) | [results](results/2026-10-10-q30-finetune-q31-fit-check.md) |
| Q31 | Can a 1-minute resting check predict who NeurOS works for? | Spearman ≥ 0.4, p < 0.01 | **FAIL:** r = 0.23. Dropped | [results](results/2026-10-10-q30-finetune-q31-fit-check.md) |
| Replay | Does the live engine match the offline numbers? Is mvp-v0 the demo model? | All 34 finish; committed commands ≥ 57.2% right; no stuck output; p99 latency < 50 ms | **PASS:** 61.4%, 1.6 ms; mvp-v0 is the demo model | [results](results/2026-10-10-mvp-v0-replay.md) |
| Calibration replay | Does the engine's 3-min calibration deliver Q30's gain? | +2 points with p < 0.05; ≥ 30 of 34 complete | **FAIL:** +3.1, p = 0.11. Button out of the demo | [results](results/2026-10-11-calibration-replay.md) |
| Projection | How much data would zero calibration need to work for everyone? | A desk calculation, no bar | **Not reachable by data alone** for motor imagery | [results](results/2026-10-11-data-scaling-projection.md) |
| E6 | Can we tell rest from mental effort across people, without calibration? | Full cap ≥ 70% mean and ≥ 20/29 people at ≥ 60%; Ganglion's channels ≥ 65%; drift at chance | **FAIL:** 70.9% but 18/29; Ganglion 56.1%. No state ships | [plan](plan/e6-state-channel.md), [results](results/2026-10-11-e6-state-channel.md) |
| A5 | Do we build per-person frequency alignment into the model before the freeze? | Aligned minus fixed bands ≥ +2.0 points over 50 cue-free people, p < 0.05, no dataset below 0 | **FAIL:** −0.24 points (p = 0.41); not built. Most peaks clipped at 11.5 Hz | [plan](plan/a5-frequency-alignment.md), [results](results/2026-10-11-a5-frequency-alignment.md) |
| A4 | Which four electrodes go on the Ganglion (and to partners) for commands and states? | On the confirmation half: states ≥ today's + 5 points (p < 0.05) and commands ≥ today's − 1 point | **FAIL:** the pick (C3, C4, Fz, Pz; 72.7% on states in the selection half) scored +1.1 points (p = 0.81) on the confirmation half, where both layouts sat at chance. Keep C3, Cz, C4, Pz. Underpowered halves | [plan](plan/a4-four-electrodes.md), [results](results/2026-10-11-a4-four-electrodes.md) |

## Bar set, not run

| ID | Decision | Bar | Waiting for | Plan |
|---|---|---|---|---|
| E7 | Can the application and demo say NeurOS holds when you aren't trying to give a command? | On 29 never-seen EEGMAT people (C3/Cz/C4, mvp-v0, θ 0.6): commit rate ≤ 10% at rest and ≤ 10% during mental arithmetic, half of the 20% during real imagery on 3 channels | Founder approval; held until the demo headset is settled (Q33: it tests the C3/Cz/C4 class, which the EPOC X isn't); then CPU on the laptop | [plan](plan/e7-quiet-when-idle.md) |
| E5 | Which headset classes are supported, experimental or not supported for commands? | Supported: ≤ 3 points below the research cap (paired, 50 people) and above chance (p < 0.05); experimental: above chance only; the 3-channel (Ganglion) class and, since 12 Oct, the EPOC X class (FC5/FC6, 41 Yang2025 people) are the headline checks | Founder approval; then GPU on AWS (~30–40 min) | [plan](plan/e5-virtual-headsets.md) |
| Headset self-test | Is the YC demo video a live headset session or a labelled replay? | At least one founder: ≥ 70% of answered doors right (p < 0.05), with at least half the doors answered | The founders' session on the Ganglion, and their call on which doors the bar scores (zero calibration recommended) | [plan](plan/headset-self-test.md) |

## In the sprint, bar not yet written (before the 24 Oct freeze)

| ID | Question | Why it matters | Data |
|---|---|---|---|
| Freeze runs | The locked test set, 10-fold and leave-one-dataset-out, on the frozen recipe (24 Oct) | The only numbers allowed into the application | In hand |

## Parked (D-041: after the deadline)

| ID | Question | Where |
|---|---|---|
| E3 | Test-time adaptation (AdaBN, entropy minimisation) | [post-YC agenda](plan/post-yc-research-agenda.md#experiments-cut-from-the-yc-sprint) |
| E4, E4-zs | Language-anchored head, contrastive training; recognising a command never trained on | same |
| E7's 3 vs 5 commands | Does an opt-in 5-command mode (+FEET, +BOTH_HANDS) work on never-seen people? | [E7 plan](plan/e7-quiet-when-idle.md#scope-change-from-the-architectures-e7-8); parked with E4 (D-039: extra classes don't help) |
| S1–S4 | DANN, Deep CORAL, RA vs EA, ShallowConvNet | same |
| SSVEP | A calibration-free "select by looking" mode | same, and the agenda's Part 1 |
| Agenda experiments 1–3, 6–12 | The ceiling, why some people fail, simulated heads, person encoder, JEPA, error-response labels, user training, language model plus recognition, states against each person's baseline | [post-YC agenda](plan/post-yc-research-agenda.md) |
| Learning curve | Data partnerships vs signal work | [projection](results/2026-10-11-data-scaling-projection.md#to-measure-the-rate-properly-proposal-not-run) |

## Template for a new plan file (`plan/<id>-<name>.md`)

```markdown
# <ID>: <question in plain words>
**Written:** <date>, before any run.
**Status:** bar set / approved / run (verdict).

## The decision
What changes if it passes, and what changes if it fails.

## Hypothesis
What we expect and why, with any published reference number.

## Data
Which people (development only; never the locked test set) and what windows.

## Run
Models, arms, controls, and the command that runs it.

## Pass bar (set before running)
Numbered checks, each with a number and a test.

## What this can't show
Confounds and limits, stated before the result.

## Cost
CPU/GPU time and any downloads (asked for separately).
```
