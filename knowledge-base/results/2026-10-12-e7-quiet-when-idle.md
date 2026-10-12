# E7: does NeurOS stay quiet when you're not trying to give a command?

**Date:** Mon 12 Oct 2026, run 18:31–19:00 UTC on the laptop (mains power). **Verdict: FAIL** by the bar set before any code ([plan](../plan/e7-quiet-when-idle.md)). The founders approved the run on 12 Oct ("start the run on the computer and aws").

**Run:** `results/e7/e7-20261012T190046Z.json`, code `6381bf2` (clean), bundle `mvp-v0`.

## In plain English
- **When nobody is trying, the engine fires commands as often as when someone is.**
  - At the shipping threshold (θ = 0.6), it committed a LEFT or RIGHT on **24.7%** of decisions at rest and **22.9%** during mental arithmetic.
  - During real imagined movement on the same three electrodes it commits on **20.4%** (BNCI2014-004, reproducing the 10 Oct replay's 20%).
  - That's about **8 false commands a minute** at rest (median person).
- **So the application and demo can't say "NeurOS holds when you aren't trying."** By the rule set beforehand:
  - the demo keeps the door game's scoring window (only commands 3.5–7 s after a door count);
  - the stage view keeps showing HOLD and its reason;
  - knowing when someone is trying becomes a post-YC item. It's a passive-state question, a natural first job for the state work (E6's territory).
- **A stricter threshold silences everything equally; it doesn't separate trying from not trying.** At θ = 0.7, rest drops to 2.2% and arithmetic to 3.7%, but real imagery drops to 4.1% too. Those commands are right more often (80% vs 70%), but there are five times fewer of them.
- **At rest, the false commands lean LEFT** (78% of them). The two-class model has a resting bias on these people's C3/Cz/C4, not just noise.

## The numbers
**Protocol:**
- **People:** 29 EEGMAT development people, never seen by mvp-v0, no class cue anywhere.
- **The replay:** each person's rest recording (~3 min), a 5-s pause, then mental arithmetic (62 s), through the real engine on C3/Cz/C4 (θ, 0.5-s smoothing, debounce 2).
- **Scored:** every decision whose 2-s window starts after the 30-s settling period and lies inside one recording. That's 34,781 rest decisions and 13,920 arithmetic decisions.
- **Imagery context:** BNCI2014-004's 9 never-seen people, first session, scored on task windows. That dataset keeps the class on screen, so it's an engine comparison, not a cue-free number.

| θ | Rest: committed | Arithmetic: committed | Rest: false commands/min (median person) | Arithmetic: /min | Imagery: committed | Imagery: committed and right |
|---|---|---|---|---|---|---|
| **0.6 (shipping; the bar)** | **24.7%** ❌ (bar ≤ 10%) | **22.9%** ❌ (bar ≤ 10%) | 8.0 | 7.0 | 20.4% | 69.9% (balanced 65.0%) |
| 0.7 | 2.2% | 3.7% | 0.8 | 1.0 | 4.1% | 80.4% |
| 0.8 | 0.0% | 0.3% | 0.0 | 0.0 | 0.7% | 87.5% (8 commands) |
| 0.9 | 0.0% | 0.0% | 0.0 | 0.0 | 0.1% | 1 command |

**Per person at θ = 0.6:**
- **Rest:** median 20.7%; only 3 of 29 people at or under 10%; all 29 committed at least once. Range 2.1–71.7%.
- **Arithmetic:** median 20.2%; 10 of 29 at or under 10%.
- **The split of false commands:** at rest, LEFT 6,725 and RIGHT 1,881 (78% LEFT); in arithmetic, LEFT 1,869 and RIGHT 1,322 (59% LEFT).
- **Why the engine held otherwise:** almost always low confidence (26,140 of 26,175 rest holds). Signal-quality holds were rare (35), as expected on ICA-cleaned lab data.

## What it means
1. **The claim is out.** Only commands inside a task window count, which the door game already enforces. The claims audit gets a "not allowed" note.
2. **The threshold is a volume knob, not a trying detector.** Raising θ to 0.7 would cut false commands at rest about tenfold (8 → 0.8 per minute). It would also cut real commands about fivefold, though more of them would be right.
   - Changing the demo's θ is its own decision. The headset self-test bar needs at least half the doors answered, which 0.7 may not reach.
3. **For the product:** a real app needs an "is the user trying?" signal before the left/right decoder. Two routes:
   - **Product design:** push-to-intend, a scoring window, a dwell confirmation.
   - **A trained idle-vs-intent detector:** the passive-state direction.
   - The resting LEFT bias also suggests checking the model's lean on each person's rest data during warm-up. That's a post-YC idea and needs its own bar.

## What this can't show
- **Not a live session.** EEGMAT is ICA-cleaned lab data. A live idle user also blinks, chews and moves, which the quality gate should catch but this didn't test.
- **Different people and amplifiers** for the idle and imagery numbers.
- **Three channels only** (the Ganglion class). The EPOC X class can't be tested on EEGMAT (no FC5/FC6).

## Reproduce
```bash
uv run --no-sync neuros --config configs/data.yaml idle
```
