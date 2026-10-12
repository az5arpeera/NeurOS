# E7: does NeurOS stay quiet when you're not trying to give a command?

**Written:** Mon 12 Oct 2026, before any code or run. **Status:** run 12 Oct: **FAIL** ([results](../results/2026-10-12-e7-quiet-when-idle.md)). Approved by the founders 12 Oct ("start the run on the computer and aws").

**Note (12 Oct):** this tests the C3/Cz/C4 class (Ganglion, Unicorn, 2b). If the demo headset becomes the founders' Emotiv EPOC X, it doesn't apply as written: EEGMAT has no FC5/FC6, the only model channels the EPOC X has. So hold E7 until the headset question is settled ([E5's EPOC row](e5-virtual-headsets.md) and the [EPOC X note](../explainers/emotiv-epoc-x.md)).

## Scope change from the architecture's E7 (§8)
- **Kept:** "false commands per minute on never-seen activity."
- **Dropped from the sprint, parked with E4:** "3 vs 5 commands."
  - D-039 showed extra classes don't lift accuracy.
  - The demo and the application use two commands.
  - The shipping model (mvp-v0) only knows left and right.

## The decision
Can the application and the demo say "NeurOS holds when you aren't trying to give a command"?
- **Pass:** they may say it, with these numbers attached, after the 24 Oct freeze.
- **Fail:** they don't say it.
  - The demo keeps the door game's scoring window (only commands 3.5–7 s after a door count), and the stage view shows HOLD and its reason.
  - "Knowing when the person is trying" becomes a post-YC item. It's a passive-state question (E6's territory): a natural first use of the state channel.

## Hypothesis
- A two-class model must lean left or right on every window. While someone rests or does mental arithmetic, the smoothing, the 60% threshold θ and the debounce should keep most of that leaning below the line.
- If the engine commits about as often at rest as during real imagery, it can't tell "trying" from "not trying".

## Data
- **People:** EEGMAT's 29 development people: never seen by mvp-v0, no class cue on any screen. The locked test set is never touched.
- **Each person's session,** replayed through the real engine in recording order:
  - their rest recording (~3 min, relaxed);
  - a 5-s pause;
  - their mental-arithmetic recording (62 s of busy thinking, no hand imagery).
- **What the engine reads:** EEGMAT has only C3, Cz and C4 from the model's motor-strip layout. That's exactly the 3-channel class of our Ganglion.
- **The comparison:** the same engine during real imagery on 3 channels. On BNCI2014-004 (9 never-seen people, C3/Cz/C4) the [mvp-v0 replay](../results/2026-10-10-mvp-v0-replay.md) committed on **20%** of task decisions at θ = 0.6 (fixed on 10 Oct).

## Run
- `neuros simulate`-style replays, as fast as possible, with the bundle mvp-v0 and the engine's defaults: θ = 0.6, 0.5-s smoothing, debounce 2.
- **Scored:** every decision after the engine's 30-s settling period (§7.2).
  - **Rest:** decisions inside the rest recording.
  - **Arithmetic:** decisions inside the task recording.

**Measures:**
1. **Commit rate** (primary): the share of decisions that are LEFT or RIGHT rather than HOLD, pooled over people. It's the same measure as the replay's 20%.
2. **False command onsets per minute:** a decision that emits LEFT or RIGHT when the previous decision emitted something else. Median over people.

## Pass bar (set before running)
Both must hold, at θ = 0.6:
1. **Rest:** commit rate ≤ **10%**, half of the 20% during real imagery on the same channels.
2. **Arithmetic:** commit rate ≤ **10%**.

## Also reported (not part of the bar)
- Onsets per minute, with each person's numbers.
- The left/right split of false commands: a lopsided split means a biased model, not noise.
- **Thresholds 0.7, 0.8 and 0.9:** rest and arithmetic commit rates, next to the BNCI2014-004 replay's commit rate and committed accuracy at the same θ. This shows what a stricter threshold would cost during real use. Changing θ would be its own decision.

## What this can't show
- **Not a live control.** Rest and arithmetic come from EEGMAT's lab recordings (ICA-cleaned). A live session also has blinks, chewing and fidgeting, which the quality gate should catch but which this doesn't test.
- **Different people.** The imagery baseline (BNCI2014-004) and the idle data (EEGMAT) come from different datasets and amplifiers.
- **Three channels only.** A full cap may behave differently.

## Cost
CPU, about 15–30 minutes on the laptop (mains power); EEGMAT and mvp-v0 are both local. Code: a small replay runner (`neuros idle`) reusing `neuros.engine` and `neuros.data.eegmat`.
