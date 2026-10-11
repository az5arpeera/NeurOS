# A5: does lining up each person's brain-rhythm frequency help zero-calibration decoding?

**Written:** Sun 11 Oct 2026, before any code or run. **Status:** run 11 Oct: **FAIL** ([results](../results/2026-10-11-a5-frequency-alignment.md)). Approved by the founders 11 Oct ("good for the run both tests").

**From:** the [post-YC agenda](post-yc-research-agenda.md), experiment 5, pulled forward because it's cheap and could still reach the 24 Oct freeze.

## The decision
Do we build per-person frequency alignment into the model and the engine before the freeze?
- **Pass:** build it into the EEGNet recipe and the engine, with its own GPU bar (E2's cue-free accuracy at 120 s) written before that run. Only if both land before 24 Oct does it enter the frozen recipe.
- **Fail:** park it with the agenda; the frozen recipe stays as is.

## Hypothesis
- People's motor rhythm (mu, ~10 Hz) peaks at different frequencies: roughly 8.5–11.5 Hz.
- A model with fixed frequency bands sees the same brain event in different places for different people.
- Measuring each person's peak from their first minute (no labels) and expressing frequencies relative to it should make people look more alike, so a cross-person model transfers better. Speech recognition does the same with vocal-tract-length normalisation.
- **Expected effect:** a few points at most; D-039 showed small model changes move little.

## Data
- **People:** the development people of the two trainable cue-free datasets: Yang2025 (41) and BNCI2014-001 (9). The locked test set is never touched.
- **Excluded:** Shin2017A is test-only (D-033), so it can't be trained on.
- **Windows:** the D-030 cue-free task windows (one 2-s window per trial, after the class has left the screen), left vs right hand.
- **Channels:** the motor strip (D-031).
- **Band:** the contract's 4–30 Hz preprocessing.

## Run
1. **Each person's peak frequency (unlabeled):**
   - **Data:** the first 60 s of their first session at C3 and C4, the same data and spectrum as Q31.
   - **Peak:** the frequency of the largest height above the 1/f background within 7–14 Hz.
   - **Fallback:** if that height is under 1 dB, there's no clear peak and the person keeps 10 Hz.
   - **Clipping:** peaks are clipped to 8.5–11.5 Hz.
2. **Features:**
   - per window, common average over the motor strip, Hann periodogram;
   - log mean power in five bands with edges 0.8, 1.0, 1.2, 1.6, 2.0 and 2.6 × *f*: theta-alpha edge, mu, upper mu, low beta, high beta;
   - *f* is the person's peak (**aligned**) or 10 Hz for everyone (**fixed**). At 10 Hz the bands span 8–26 Hz; at the clipping limits, 6.8–29.9 Hz, inside the 4–30 Hz band;
   - each person's features are z-scored over their own windows (unlabeled, both conditions).
3. **Model:** standardise, then logistic regression. Leave one person out within each dataset; the same folds for both conditions.

**Measure:** each person's balanced accuracy; the difference aligned minus fixed, paired by person.

## Pass bar (set before running)
All three must hold:
1. **Gain:** mean gain over the 50 people ≥ **+2.0 points**.
2. **Significance:** paired Wilcoxon p < 0.05 (two-sided).
3. **No dataset loses:** neither dataset's mean gain is below 0.

## Also reported (not part of the bar)
- The distribution of peak frequencies, and how many people had a clear peak.
- Whether the gain grows with the distance of the person's peak from 10 Hz (Spearman).
- Both conditions' absolute accuracy, for context against E2.

## What this can't show
- **It's a proxy.** A linear model on band power, not EEGNet. A pass says the idea is worth building, not that EEGNet gains the same; the GPU bar after a pass decides that.
- **A minute of rest is a noisy estimate.** The peak from 60 s at C3/C4 is noisy, and Q31 found the peak's *height* weakly linked to accuracy. This tests its *frequency*.
- **Only two datasets** have cue-free windows and can be trained on.

## Cost
CPU, about 15–30 minutes on AWS (the data lives there), from the preprocessed cache. No downloads. Code: `neuros.eval.frequency`, `neuros freq-align`.
