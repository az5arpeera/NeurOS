# Calibration time, "commands", and the headline number

*Answers open questions Q3 and Q4 (founder asked 2026-10-01).*

## Commands = the intent vocabulary
A **command** is one of the outputs the decoder can produce. Apps map each command to an action.

| Command | What the user does | Typical app action |
|---|---|---|
| `LEFT` | Imagine squeezing the **left** hand (no real movement) | move cursor or character left, previous track |
| `RIGHT` | Imagine squeezing the **right** hand | move right, next track |
| `REST` | Relax; think of nothing in particular | do nothing / idle |
| `HOLD` *(not a mental task)* | Nothing. The **decoder** emits HOLD when its confidence is below a threshold | ignore; wait for a clear signal |

`HOLD` is like a voice assistant ignoring audio that isn't clearly its wake word. It prevents false triggers. It's the "confidence-gated" output from the founder notes, and it makes a demo look calm instead of jittery.

*Update 2026-10-02:* `HOLD` also fires, with a reason code, when the quality gate detects movement or muscle artifacts. Why we don't decode whole-body actions (walking, punching) is covered in [beyond LEFT/RIGHT](beyond-left-right-any-action.md).

**Why so few commands?** Accuracy falls as the number of choices rises: about 77–83% for 2 commands across people, about 66–68% for 4. What users feel is throughput, measured by the **information transfer rate (ITR)**:

ITR per decision (bits) = log₂N + P·log₂P + (1−P)·log₂((1−P)/(N−1)), for N commands at accuracy P.

| N | Accuracy P | Bits per decision | At 1 decision / 2 s |
|---|---|---|---|
| 2 (left/right) | 80% | 0.278 | **8.3 bits/min** |
| 3 (left/right/rest) | 70% | 0.404 | **12.1 bits/min** |

A third command can carry more information even at lower accuracy. That's why left/right/rest is the proposed MVP vocabulary, with HOLD on top.

## The headline number
The single sentence and figure that goes in the YC application and on the demo slide. Proposed shape:

> "**X% accuracy on N people the model has never seen, from M different recording setups, usable after S seconds, with no calibration task.**"

plus one chart: **accuracy vs seconds of use** (the adaptation curve). Every number carries its protocol (house rule #1).

## Calibration-time target (founder decision 2026-10-01)
**Usable within seconds; full accuracy within 30–45 s at most; no calibration task.**

How we plan to hit it:
```
t = 0 s        Model works immediately with population defaults (strict zero, L0). Lower accuracy.
0 – ~30 s      "Settling" while the headset connects: the user just sits relaxed, eyes open.
               We estimate their resting-state reference covariance and blend it with the population
               reference:  R̂(t) = (1 − α_t)·R_population + α_t·R_user(t),  with α_t growing as data arrives.
               No instructions, no labels, no task. It looks like a "connecting…" screen.
continuous     Online alignment (EMA) + adaptive batch-norm + entropy-min keep improving it while in use.
```

**Is 30–45 s realistic?**
- **Statistically, yes.** A 16-channel covariance from 30 s at 250 Hz uses 7,500 samples to estimate 136 numbers.
- **The real question is neurophysiological.** Does a reference computed from *resting* data re-centre *task* data as well as the 16–32 task trials used in the literature (≈1–2 min)? There's precedent: Zanini et al. 2018 re-centred on resting-state trials, and later EA studies compare against it. But it's unproven on our pool.
- **This is Hypothesis H1 and our first experiment.** We measure accuracy at 0 / 10 / 20 / 30 / 45 / 60 / 120 s of warm-up, comparing a resting reference, a task-stream reference, and the population default. **We claim only what that curve shows.**

**Why this framing sells to BCI companies:** "works at second 0, reaches near-full accuracy in ~30 s, and the user never sees a calibration screen". That's the experience BCI companies can't offer today, and the adaptation curve makes it visible.
