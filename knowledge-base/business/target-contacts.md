# Target contacts: the first 15 (v1, 2026-10-03)

**Purpose:** who to approach first for the [customer interviews](customer-interview-script.md), across the script's three segments: A labs, B developers, C companies (D-015). The plan's target is the first 10 messages on Mon 5 – Tue 6 Oct, and 15+ conversations by 23 Oct.

**Ground rules:**
- **Founders send every message;** Claude drafts only.
- **No personal contact data is stored here.** Names are as shown on public lab and company pages; get contact details from the linked page when you write.
- Every fact below comes from the linked public page, checked on 3 Oct 2026. Re-check before quoting it to the person.

## ⏰ Time-critical: BR41N.IO hackathon, Sun 4 – Mon 5 Oct (virtual, free)
g.tec's BCI designers' hackathon runs **online during IEEE SMC 2026, 4–5 Oct (PDT)**, over Zoom ([event page](https://www.gtec.at/hackathon/ieee-smc-2026/)).
- **Projects:** games and programming on the Unicorn headset, plus **data-analysis projects that need no hardware**.
- **Why go:** dozens of BCI developers hit calibration in real time, in one place. Join a data-analysis team or just the Zoom, and ask teams how they got each person working.
- **Decision for the founders:** register or skip. Claude doesn't register.

## The 15

**Priority:** ★★★ = message first (Mon–Tue), ★★ = week 2, ★ = backup or connector.

### A. Research labs (UVA)

| # | Who | Why they fit | What to learn first | Reach via | Pri |
|---|---|---|---|---|---|
| 1 | **WEARLab** (Wearable Electronics and Robotics Lab), UVA Engineering; led by Dr. Sarah Sun (MAE/ECE) | Builds "brain-computer interface technologies for motion assistance, neurorehabilitation" on wearables ([lab page](https://engineering.virginia.edu/labs-groups/wearlab)). Closest UVA match to motor-imagery BCI | How they train a decoder for each new participant or patient, and how long it takes; whether their hardware changes between studies | Lab page | ★★★ |
| 2 | **Long Term Memory Lab**, UVA Psychology; director Nicole Long | Uses scalp EEG (and fMRI) in human studies ([UVA Psychology lab list](https://psychology.as.virginia.edu/research-opportunities-working-lab), [lab site](https://longtermmemorylab.com)) | Per-participant setup and decoding time; how many participants they exclude for bad data | Lab site | ★★★ |
| 3 | **Sensing Systems for Health Lab (S2He)**, UVA Systems & Information Engineering; director Laura Barnes, also Associate Director of Link Lab | Wearable and phone sensing for mental health (anxiety, depression, traumatic brain injury) ([lab site](https://s2helab.com/), [faculty page](https://engineering.virginia.edu/faculty/laura-barnes)). Fits our **brain-state channel** (rest vs workload) and the founder's department | Whether brain sensing is on their roadmap and what stops it (setup, calibration); who at Link Lab works with EEG | Lab site | ★★★ |
| 4 | **Social Neuroscience Lab**, UVA Psychology; director James Morris | EEG and event-related potentials, plus fMRI ([lab site](https://socialneuroscience.as.virginia.edu/)) | Participant setup time and data loss; whether any study needs real-time decoding | Lab site | ★★ |
| 5 | **UVA Brain Institute** (connector) | Runs the neuroscience seed and pilot grants (up to $35k; the 2026 cycle is in review) and keeps the affiliated-faculty directory ([funding](https://braininstitute.virginia.edu/funding), [researchers](https://braininstitute.virginia.edu/researchers)) | Intros to EEG and neurotech faculty we haven't found. Later: a pilot-grant collaboration with a lab | Website | ★ |

**Backup:** Virginia Affective Neuroscience Laboratory (UVA Psychology, director James Coan), EEG + fMRI ([psychology lab list](https://psychology.as.virginia.edu/research-opportunities-working-lab)).

### B. Developers and communities

| # | Who | Why they fit | What to learn first | Reach via | Pri |
|---|---|---|---|---|---|
| 6 | **OpenBCI community: motor-imagery builders.** Authors of recent posts such as "EEG Controlled Robotic Arm in Real-time" (Apr 2026) and "Controlling an Exoskeleton Arm in Two Directions" (Mar 2026) | They use OpenBCI boards, our demo hardware family, for exactly our task ([community](https://openbci.com/community/)) | How long it took to make the decoder work for a second person; what they hacked to avoid recalibrating | Forum or the post's comments | ★★★ |
| 7 | **BR41N.IO hackathon participants** (see the time-critical note above) | Developers building BCI apps on the Unicorn, one of our headset profiles | Calibration in the wild, under time pressure | The event's Zoom | ★★★ |
| 8 | **NeuroTechX**: global neurotech community (≈ 9,500-member Slack, Discord, 30+ chapters, Global NeuroHack 2026) ([site](https://neurotechx.org/education)) | The largest pool of hobbyist and student BCI developers | Post one Mom-Test question ("how do you handle calibration for a new user?") and interview the people who answer | Slack or Discord | ★★★ |
| 9 | **Neurosity Crown developer community** (8-channel, 256 Hz headset with JS/Python SDKs, LSL and BrainFlow ([Crown](https://neurosity.co/crown))) | App developers on a consumer headset that is one of our virtual-headset profiles | Do apps use raw EEG or only Neurosity's focus/calm metrics? Has anyone tried motor imagery? | Community projects page, Discord | ★★ |
| 10 | **MOABB maintainers** (weekly developer meeting hosted by NeuroTechX ([site](https://neurotechx.org/education))) | Expert, not customer: they run the benchmark our evaluation builds on. We have two concrete MOABB 1.7 findings to report (downloads default to a NEMAR mirror; TLS verification is off), a credible way to start the conversation | How reviewers judge cross-subject claims; whether our ~380-person LODO protocol would be welcome upstream | A GitHub issue, then the weekly meeting | ★★ |

### C. BCI companies

| # | Who | Why they fit | What to learn first | Reach via | Pri |
|---|---|---|---|---|---|
| 11 | **OpenBCI** (company) | Makes our demo board (Ganglion). A hardware-agnostic intent layer makes their boards more useful: partner and distribution (a featured community project) | Which questions about new-user setup their support gets; whether they'd feature an open SDK | Contact page | ★★★ |
| 12 | **g.tec medical engineering** (Unicorn Hybrid Black; runs BR41N.IO) ([hackathon](https://www.gtec.at/hackathon/ieee-smc-2026/)) | Unicorn is one of our headset profiles. The plan already has an email to g.tec about a student/startup price, so combine the two | How Unicorn users get from unboxing to a working BCI; what their calibration workflow looks like | Combined with the planned price email | ★★★ |
| 13 | **Cognixion** (Axon-R: AR headset with EEG, up to 16 channels, for ALS communication; Apple Vision Pro study; pivotal trial planned for 2026) ([site](https://www.cognixion.com/), [news](https://www.massdevice.com/cognixion-study-bci-apple-vision-pro/)) | Assistive BCI is where calibration hurts most: patients tire fast | Minutes of calibration per patient session; how often they re-calibrate | Website contact | ★★ |
| 14 | **Neurable** (Boston; launched Neurable One, $499 EEG headphones with 12 sensors, on 1 Oct 2026; raised a $35M Series A) ([Engadget](https://www.engadget.com/2271652/neurable-one-eeg-headphones/), [Technology Org](https://www.technology.org/2026/10/03/neurable-one-eeg-headphones-launch/)) | Consumer EEG at scale: onboarding friction is their problem every day | Onboarding completion rate; whether a per-user baseline period exists today | Website / LinkedIn | ★★ |
| 15 | **Arctop** (cognition-decoding SDK: focus, cognitive load, attention from EEG in headphones, decoded in Arctop's cloud; works with IDUN) ([site](https://arctop.com/), [SDK](https://github.com/arctop/iOS-SDK)) | The closest analogue to our brain-state channel. A possible partner, or a competitor to understand. We run locally; they run in the cloud | How they handle new users (population model vs personal baseline); which customers ask for intent, not just state | Website contact | ★ |

**Backups for segment C:**
- IDUN Technologies: Guardian 4 in-ear EEG, 2026; workload estimates ([site](https://iduntechnologies.com/idun-guardian)).
- Emotiv: EPOC X 14-channel; Cortex SDK over JSON/WebSockets ([developer docs](https://emotiv.gitbook.io/cortex-api)).

## Message drafts (founders adapt and send)
Short, about their work, and no pitch, per the Mom Test. Personalise the first line every time.

**Ready-to-adapt versions for the first 10 contacts:** [first-10-messages.md](first-10-messages.md) (one per contact, with the channel to use).

**A, labs:**
> Subject: 20 minutes on your EEG setup? (UVA students)
>
> Hi Dr. [name], I'm [name], a [year/major] at UVA. With my cofounder I'm researching how EEG labs handle the setup and decoder-training step with each new participant. I read about [one specific thing from their page]. Could I ask you or someone in your lab a few questions about how that step goes today? 20 minutes, any time this or next week. I'm not selling anything.

**B, developers** (forum reply or DM):
> Really nice [project]. Question from someone building in the same space: when a *second* person tried it, how long did it take to get the decoder working for them, and what did you have to redo? I'm collecting how people deal with per-user calibration. Happy to share our cross-person benchmark results on public EEG data as they come in.

**C, companies:**
> Hi [name], we're two UVA founders working on calibration-free intent decoding that works across EEG headsets. Before we build further, we're trying to understand how teams like yours handle new-user setup: how long it takes, where people drop off. Would you have 20 minutes for a few questions? In return, we'll share our cross-headset benchmark (public data, 7 recording setups) when it's done.

## Tracking
After each conversation, log the §6 note template from the [interview script](customer-interview-script.md) and update the status here.

| # | Status | Date | Next step |
|---|---|---|---|
| 1–15 | Not contacted | | |
