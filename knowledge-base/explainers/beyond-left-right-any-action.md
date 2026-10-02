# Beyond LEFT / RIGHT / REST: could NeurOS pick up *any* action without labels?

*Answers the founder's question (2026-10-02): "How much more would it be to extend the calibration so it pulls out any action the subject does, without labels: getting up, moving around, walking, punching the air?"*

## Short answer
- **More imagined body parts** (feet, both hands, tongue):
  - the data exists, and it's about a week of work;
  - but accuracy drops with every command added.
- **Real (executed) movements:**
  - the data exists, and they're easier to detect;
  - but someone who can move doesn't need a BCI to detect the movement;
  - the exception is *intention*: detecting a movement just before it happens.
- **Whole-body actions** (stand up, walk, punch):
  - EEG mostly records the movement's **artifacts** (muscle activity, cable and electrode motion), not the brain's plan;
  - a cheap accelerometer detects these movements better;
  - doing this properly takes months of research and new data, so it isn't a credible BCI claim before 1 Nov.
- **"Without labels":** the system can *notice* that something different is happening without labels, but it can't *name* it. Names have to come from one of three places:
  - labeled training data from other people (zero calibration already works this way);
  - context from other sensors or the app;
  - one example from the user.

## Three meanings of "without labels"

| Meaning | Possible? | In NeurOS |
|---|---|---|
| The **new user** gives no labels (zero calibration) | Yes, *if* the action is labeled in training data from other people | **MVP.** This is how LEFT / RIGHT / REST work. |
| The system **discovers** actions by itself | Partly. Clustering and novelty detection can find "a different state", but not *which* action. EEG clusters first by artifacts, eyes open/closed and alertness, not by intent. | Research |
| Labels arrive **for free** from context (the headset accelerometer says "punch"; a game knows the user was steering left) | Yes. This is *weak supervision*, a real and scalable route. | After the MVP: the long-term data engine |

## Action families, ranked by feasibility

| Action family | Training data | What the EEG actually measures | Zero-calibration reliability | Extra effort | Recommendation |
|---|---|---|---|---|---|
| Imagined left / right hand + rest | Yes: ~380 people | Brain (mu/beta rhythms over motor cortex) | Moderate: 77–83% binary across people (literature, 9-subject benchmarks) | In the MVP | **MVP** |
| Imagined feet / both hands / tongue | Yes: PhysionetMI, BNCI2014-001, Stieger2021 | Brain | Drops with each command (66–68% for 4 classes) | ~1 week | Optional, after the core works |
| Real hand movements (squeeze, open/close fists) | Yes: PhysionetMI executed runs; Cho2017 real-movement recordings | Brain **plus** muscle and movement artifacts | Higher, but partly artifact | Days | Use in **onboarding** ("squeeze first, then imagine"); test as extra training data |
| Fine arm/hand actions (reach direction, grasp type, wrist twist) | Some: Jeong et al. 2020, 25 people, 11 upper-limb tasks, 60 EEG + 7 EMG + 4 EOG channels | Brain (weak) | Low across people | Weeks to months | Research, after 1 Nov |
| Whole-body actions (stand up, walk, punch) | Little: mobile-EEG datasets are small and artifact-heavy | **Mostly artifacts.** Kline et al. 2015 blocked the brain signal with a silicone swim cap under a gel-coated wig and *still* recorded gait-related "EEG" from 9 people walking at 0.4–1.6 m/s. | Detecting the movement: easy (an accelerometer does it). Decoding the brain's plan: low. | Months + new data | **Don't claim it.** Detect the movement and pause (HOLD). |
| Visual-attention commands (SSVEP: look at one of several flickering targets) | None needed: standard training-free math (CCA) | Brain (visual cortex) | High for most people | 1–2 days | Optional **demo insurance** |
| Eye / jaw commands (blink, glance, clench) | None needed | Not brain (EOG / EMG) | Very high | Days | Later multimodal module (D-013) |

## What this means for the MVP
1. **Keep `LEFT / RIGHT / REST + HOLD`** as the novel zero-calibration core.
2. **Add a movement/artifact detector that pauses decoding.**
   - It emits `HOLD` with a reason code: `motion`, `muscle` or `signal_gap`.
   - If the volunteer laughs, nods or punches the air, the screen holds steady instead of firing random commands.
   - That robustness is product value, and it's EEG-only. Thresholds will be tuned on Cho2017's recorded blinks, eye movements, head movements and jaw clenches.
3. **Onboard with real squeezes**, then switch to imagined ones.
4. **Optional SSVEP demo insurance** in week 3, if time allows (founder's call).

## The long-term path toward "any action" (post-MVP story for YC)
1. **Self-supervised pretraining** on large unlabeled EEG, which learns structure without labels (RUNet / foundation-model style).
2. **Weak labels from context:** headset accelerometers, app and game events, and the brain's own error signals provide labels automatically during normal use. With consent, every session can improve the model.
3. **A vocabulary that grows with data:** each new labeled dataset adds named actions through the [text-anchored head](text-anchored-head.md).
4. **Movement-intention decoding:** signals that precede a movement, for exoskeletons and rehab. A real, fundable direction.

## Sources (read depth)
- Kline, Huang, Snyder & Ferris 2015, *J. Neural Eng.*: "Isolating gait-related movement artifacts in electroencephalography during human walking". Silicone cap + conductive wig; 9 participants; treadmill at 0.4–1.6 m/s. *Abstract level.*
- Jeong et al. 2020, *GigaScience*: "Multimodal signal dataset for 11 intuitive movement tasks from single upper extremity during multiple recording sessions" (doi:10.1093/gigascience/giaa098). *Abstract level.*
- Cho et al. 2017, *GigaScience*: "EEG datasets for motor imagery brain–computer interface". Includes real hand movement plus non-task recordings (blinks, eye movements, head movement, jaw clenching). *Abstract level.*
- Lee et al. 2021: a mobile BCI dataset of scalp- and ear-EEG recorded while standing, walking and running (arXiv 2112.04176). *Title level*; relevant to robustness, post-MVP.
