# Founders' headset self-test (Ganglion): runbook, decision and pass bar

**Written:** 11 Oct 2026, before any headset session (the rule: every run answers a decision with its bar set first).

**Who runs it:** the founders, on their laptop with the Ganglion (D-025). Claude can't touch the headset; it can read the saved session afterwards and score it.

## The decision
Does NeurOS work live, on our Ganglion, on a founder's own brain?
- **Yes:** the YC demo video is a live headset session.
- **No:** the video uses a replay of a never-seen dataset person, labelled as a replay, and the application says live headset results are pending.

This is also the only consumer-hardware evidence we can get before 1 Nov (Q30's evidence 3).

## Pass bar (set before the first session)
For at least one founder, after the quick calibration, the door game must reach both:
1. **Committed accuracy ≥ 70%** (right among the doors the engine answered), one-sided binomial p < 0.05 against 50%.
2. **At least half the doors answered** (the engine didn't hold back on most of them).

Each founder plays 40 doors after the quick calibration. Example: 21 right of 30 answered is 70% with p = 0.02, and 30 of 40 is 75% answered.

**Also reported, not part of the bar:** the zero-calibration door game (40 doors) and the signal-quality holds.

**Why 70%:** with two commands, about 70% is the usual threshold where a person can steer an app. Our offline expectation for this channel set is about 66% after a quick calibration (Shin2017A, 3 channels), so passing is possible but not certain.

## Why the game is built this way
- **The door is shown for 1 second, then hidden.** It's scored only on decisions whose 2-s window started after it was gone (3.5–7 s after the cue).
- **This is the D-030 rule:** with the target still on screen, the engine could score by picking up eye movements toward it instead of motor imagery. That inflated our first numbers (E0, 75% vs 61%).
- **Doors come in shuffled blocks of 5 left and 5 right.**

## Before the session (once, when the board arrives)
1. **Check the channel labels.** In the OpenBCI GUI, note the channel names the Ganglion's stream publishes. Set them in `src/neuros/data/montages/ganglion.yaml` (`device:` fields), set `verified: true`, and commit.
2. **Check signal quality.** In the OpenBCI GUI, check impedance and look at the raw traces: no flat lines, no 50/60 Hz wall. Blinks should show; jaw clenches should show as bursts.

## The session (about 25 minutes per founder)
**Electrodes:** gold cups with Ten20 paste at C3, Cz and C4 (motor strip) and Pz; reference and ground on the earlobes. Sit still, hands in your lap, the screen at arm's length.

**Steps:**
1. **Start the engine and record the session** (one new folder per session; it is never overwritten):
   ```bash
   uv run --no-sync neuros --config configs/data.yaml run --bundle bundles/mvp-v0 --brainflow GANGLION_BOARD --serial-port COM3 --headset ganglion --record sessions/2026-10-XX-founder-a
   ```
   Replace `COM3` with the dongle's port.
2. **Open the demo page** at `http://127.0.0.1:8765/` and wait 3 minutes, still, eyes open. The engine settles and adapts; the state goes settling → adapting → steady.
3. **Play 40 doors with zero calibration:** press **Start** in "Practice: open the door". Each door is on screen for 1 s; when it disappears, imagine squeezing that hand until "Relax". That's about 6 minutes; press **Stop** after 40 doors (right + wrong + held = 40).
4. **Run the quick calibration:** press **Quick calibration · 3 min** and follow the arrows (20 trials). If it says it failed, fix the electrodes and run it again. Don't count a failed attempt as a result.
5. **Play 40 more doors** with the quick calibration (the pill says "quick calibration").
6. **Stop the engine** (Ctrl+C). It prints `session saved: …`.

**Score it** from the saved session, not the page's counter:
```bash
uv run --no-sync neuros session sessions/2026-10-XX-founder-a
```
This prints right / wrong / held, the accuracy among answered doors, the share answered and p, separately for zero and quick calibration.

## Privacy
- **Your own brain data:** the session folder holds raw EEG. It stays on the laptop: `sessions/` is git-ignored and nothing uploads it.
- **Sharing:** decide yourselves whether to share a session with Claude for analysis. Claude reads it only from the laptop.
- **Consent (D-006):** informal, consented testing. Don't record anyone else without their consent.

## Afterwards
- Write the result in `knowledge-base/results/` with the date, founder (A/B), the door scores and the verdict against this bar, including a failure.
- Check the levels in the saved `events.jsonl` if anything looks off.
