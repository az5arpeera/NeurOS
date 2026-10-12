# Emotiv EPOC X: what it can and can't do for NeurOS

**Written:** 12 Oct 2026, when the founders got access to an EPOC X and asked to use it instead of the Ganglion. **Status:** the headset choice is the founders' ([Q33](../open-questions.md)); recommendation below.

## In short
- **The EPOC X is a solid research headset, but none of its 14 sensors sits over the hand areas.** They ring the head: forehead, temples, the sides and the back. Left vs right imagery shows most clearly at C3 and C4, on top of the head.
- **Our model reads the motor strip (20 positions). The EPOC X gives it 2 of them: FC5 and FC6,** at the strip's front corners. After the engine's common average, that's effectively one signal (FC5 minus FC6).
- **So expect commands at or near chance** until [E5](../plan/e5-virtual-headsets.md) says otherwise. E5 now scores this exact case, on 41 never-seen people, in the same GPU run.
  - Our own literature review found nothing better: a study on an EPOC X without C3/C4 decoded six classes with kappa near or below zero ([consumer-headset report](../research/consumer-headset-mi-datasets.md)).
- **What it's good for right now:** bringing up the live pipeline on real hardware (stream, signal quality, stage view, the engine's warm-up), so the first motor-strip session isn't also the first time anything touches a real headset.
- **What it's good for later:** passive mental states. Its sites cover the front and back of the head, where E6 found the mental state shows. 10 of its 14 sites are also in EEGMAT. That's the post-YC passive-state direction.

## The facts
From Emotiv's [EPOC X manual](https://emotiv.gitbook.io/epoc-x-user-manual/introduction/technical-specifications), checked 12 Oct:

| | EPOC X | What it means for us |
|---|---|---|
| Sensors | AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4 | No C3, Cz or C4. The profile is `src/neuros/data/montages/epocx.yaml` |
| References | CMS/DRL at P3/P4 | Fine: the engine re-references (common average) |
| Sampling | 128 or 256 Hz (2048 Hz internally) | 128 Hz is exactly our model's rate |
| Electrodes | Ag/AgCl with saline-soaked felt | Closer to our gel training data than dry pins |
| Filters | 0.2–45 Hz, notches at 50 and 60 Hz | Covers our 8–30 Hz band |

**Raw EEG costs a licence.**
- Without a paid licence, the signal never leaves Emotiv's software, and NeurOS can't read it.
- Emotiv staff say the free tier only shows the signal, and LSL streaming is a paid EmotivPRO feature ([forum](https://forum.emotiv.com/?p=2178)).
- Emotiv's knowledge base also describes a separate Developer API licence for raw EEG through its Cortex API ([Emotiv](https://www.emotiv.com/knowledge-base/i-am-getting-the-error-invalid-license-key)).
- **First check:** does the account you have access to (a lab's, perhaps) already include EmotivPRO? If not, ask Emotiv which licence covers raw EEG over LSL for an EPOC X, and the price.

**Privacy.**
- EmotivPRO can sync recordings to Emotiv's cloud. You can opt out under Settings → Recording ([EmotivPRO docs](https://emotiv.gitbook.io/emotivpro-v3/managing-your-eeg-data-recordings/cloud-synchronization)).
- Our rule is that brain data stays on the laptop. So opt out, don't record in EmotivPRO, and record with `neuros run --record` into `sessions/` (git-ignored).
- Signing in to Emotiv's software is yours to do; Claude never handles the account.

## The options for the YC demo
1. **Use both (recommended).**
   - The EPOC X now, for the live-pipeline bring-up below.
   - Commands on a headset with C3/C4: the Ganglion kit (about $715 at list price on 2 Oct; re-check), or a supervised session on Professor Sun's 16-channel OpenBCI cap (it decodes imagined fist closure, so it likely covers C3/C4; confirm with her).
   - Fallback: the labelled replay of a never-seen person, already planned in the [self-test](../plan/headset-self-test.md).
2. **The EPOC X only.**
   - Run E5 first. If its EPOC row is "not supported", the live demo can show the pipeline running on a real headset, but not working commands. The commands evidence becomes the labelled replay.
3. **Train a model for the EPOC X's own 14 sites.**
   - Five of our training datasets have all 14 (PhysionetMI, Cho2017, Lee2019_MI, Stieger2021, Yang2025), so it's buildable.
   - It needs its own bar and a GPU run, plus about a day or two of work, 12 days before the freeze.
   - Only worth it if E5's EPOC row shows something above chance. Without the hand areas, a big gain is unlikely.

**Why option 1:** the demo's job is working commands, and those need the hand areas. The EPOC X still saves real time: every integration bug it finds is one the motor-strip session won't hit.

## Bring-up check: is the live pipeline ready for a headset session?
This is an engineering checklist, not an accuracy test: **no commands are scored**, so it needs no statistical bar.
- **Pass:** all of these hold on one 10-minute session:
  - the stream connects and maps all 14 channels;
  - every trace is live in the stage view;
  - FC5's and FC6's quality dots are green while sitting still;
  - the engine reaches "steady";
  - nothing crashes.
- **Fail:** whatever broke gets fixed before any motor-strip session.

**Steps (the founders, once the licence is sorted):**
1. **Fit the headset.** Soak the felt pads in saline. In EmotivPRO, set the rate to 128 Hz and wait until contact quality is green.
2. **Set privacy.** In Settings → Recording, opt out of cloud sync.
3. **Turn on the stream.** In Settings → Lab Streaming Layer, enable the EEG outlet.
4. **Start the engine:**
   ```bash
   uv run --no-sync neuros --config configs/data.yaml run --bundle bundles/mvp-v0 --lsl --headset epocx --telemetry --record sessions/2026-10-XX-epocx-bringup
   ```
   - **If it refuses** with "stream channels [...] aren't devices of profile 'epocx'", paste me that line. It lists Emotiv's channel labels, no EEG. I'll update the profile.
   - **The expected warning:** it will say there's no electrode at C3/C4.
5. **Open the stage view** at `http://127.0.0.1:8765/stage`. Sit still with eyes open for 3 minutes. Check that the state goes settling → adapting → steady.
   - Emotiv's raw values typically ride on a large constant offset. The engine's filter removes it, so the stage view's RMS should look like ordinary EEG (tens of µV). If it doesn't, tell me.
6. **Stop the engine** with Ctrl+C. The session stays in `sessions/`.

## What we don't know yet
- **The stream's channel labels and extra columns.** The profile assumes the position names. Set `verified: true` after the first stream.
- **Whether you already have the licence,** and what it costs if not.
- **Whether the commands are truly at chance.** That's E5's EPOC row.
