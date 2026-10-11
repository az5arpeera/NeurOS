# A4: which four electrodes serve both commands and mental states?

**Written:** Sun 11 Oct 2026, before any code or run. **Status:** bar set; approved by the founders 11 Oct ("good for the run both tests").

**From:** the [post-YC agenda](post-yc-research-agenda.md), experiment 4. Pulled forward because [E6](../results/2026-10-11-e6-state-channel.md) showed the mental state lives over the front and back of the head, while our Ganglion montage (C3, Cz, C4, Pz) sits on the motor strip.

## The decision
Do the founders move the Ganglion's four electrodes, and which four do we recommend to hardware partners?
- **Pass:**
  - the winning four go on the Ganglion for every founder and pilot session; the headset runbook and montage file are updated;
  - it becomes our recommended 4-electrode layout;
  - live sessions then record both commands and the randomised rest/arithmetic blocks (E6's open control) with one setup.
- **Fail:** keep C3, Cz, C4, Pz.
- **Either way:** no mental state enters the YC demo or the application as a validated feature. E6 failed its ship bar (D-042).

## Hypothesis
Two electrodes over the hand areas (C3, C4) carry the commands. The other two are better spent where the mental state shows (frontal or parietal) than on Cz and Pz. That should raise state accuracy a lot and cost commands little.

## Data
- **States:** EEGMAT's 29 development people, E6's exact pipeline. TS+LR with the 20-s rest baseline; rest 20–60 s vs the task; leave one person out.
- **Commands:** Yang2025's 41 development people, cue-free windows (D-030), left vs right. The largest cue-free dataset, and the only one with all 19 positions EEGMAT has.
  - Zero calibration: TS+LR with each person's own unlabeled alignment (EA, full session), common average over the four, leave one person out.
- **Never touched:** the locked test set.

**Candidates:** C3 and C4, plus any 2 of the other 17 positions in both datasets (Fp1, Fp2, F7, F3, Fz, F4, F8, T7, Cz, T8, P7, P3, Pz, P4, P8, O1, O2). That's **136 montages**, including today's C3, Cz, C4, Pz.

**Picking without fooling ourselves:** with 136 tries, the best score is inflated by luck. So each dataset's people are split in two with a fixed seed:
- **a selection half:** EEGMAT 15, Yang2025 21;
- **a confirmation half:** EEGMAT 14, Yang2025 20.

Each half runs its own leave-one-person-out.

## Run
1. **On the selection half:** score all 136 montages on states and commands.
   - **The pick, W:** among montages whose commands score is within 1 point of today's montage, the one with the best states score (ties broken by commands).
2. **On the confirmation half only:** score W and today's montage.

## Pass bar (set before running), on the confirmation half
Both must hold, and W must differ from today's montage:
1. **States:** W's mean ≥ today's + **5 points**, with paired Wilcoxon p < 0.05 (two-sided, by person).
2. **Commands:** W's mean ≥ today's − **1 point**.

If the pick *is* today's montage, the verdict is **keep** (nothing to switch).

## Also reported (not part of the bar)
- The top 10 montages on each measure, and the best trade-offs between them, on the selection half.
- For context, how close the best four come to E6's full cap (70.9%) and frontal-parietal six (66.7%).

## What this can't show
- **The commands score is a proxy** (TS+LR), not our EEGNet model. mvp-v0 reads only motor-strip channels, so if W drops Cz, the live model sees C3 and C4 only. E5 (virtual headsets) measures our model on the winning layout before any headset session.
- **States inherit E6's limits:** one dataset, ICA-cleaned, order confounded with class.
- **Electrode positions are 10-20 only** (EEGMAT has no Oz, POz or FCz).

## Cost
CPU, about 30–60 minutes on AWS: Yang2025 lives there, and EEGMAT (0.18 GB, on the approved list) gets downloaded there too. Code: `neuros.eval.four_electrodes`, `neuros four-electrodes`.
