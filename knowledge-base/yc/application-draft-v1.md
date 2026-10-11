# YC W27 application: draft v1

**Written:** Sun 11 Oct 2026 by Claude, from the knowledge base. **The founders rewrite it in their own voice:** YC reads for clear thinking, not polish.

**Deadline:** YC's apply page (read 11 Oct) gives the W27 deadline as **Mon 2 Nov 2026, 8:00 PM PT**, with decisions by 11 Dec. We still aim to submit **Fri 30 Oct**.

## Before this goes anywhere
- **The questions are from YC's recent application form,** not the live W27 portal. The portal (apply.ycombinator.com) needs your login. Open it, paste the current questions and their limits here, and I'll re-map the answers.
- **Every number marked ‡ is a development number.**
  - Development numbers can't go in the application ([results rules](../results/README.md)).
  - Each one is replaced by its frozen headline number after the **24 Oct freeze**, then checked against the [claims audit](../claims-audit.md).
- **Three founder decisions change the draft:**
  - **Q28, the lead:** this draft follows D-040, "calibration saved" plus rigour, and adds the mental-state channel only if E6 passes.
  - **Q32, the name:** NEUOS is a live mark one letter away. If you rename, every answer changes.
  - **Interviews and letters of intent:** the "how do you know people need it" answers are placeholders until you've done interviews.
- **Brackets [like this]** are for you to fill. **Never invent** traction, users or quotes.

---

## Founders
**Who writes code, or does other technical work on your product? Was any of it done by a non-founder?**
> [Founder 1: Azhar] built the data pipeline, the evaluation harness and the live engine; [Founder 2] leads the model experiments. We write code with AI coding agents (Claude Code), which we direct and review. No non-founder has written code.

*Note: YC asks about AI tools in the tech-stack question too. Be plain about it; it's normal now.*

**Are you looking for a cofounder?** [No.]

**Founder profiles** (each founder fills their own):
- **Education and work:** Azhar: University of Virginia, Systems Engineering and Mathematics (expected May 2028). [Founder 2: …]
- **"The time you most successfully hacked some (non-computer) system to your advantage":** [One concrete story each, with the result. Not a brag.]
- **"The most impressive thing, other than this startup, you have built or achieved":** [One or two sentences, with a link to the shipped thing.]

**Founder video (one minute, both founders on camera):**
- Who you are, what you're building in one sentence, and why you two.
- Don't read a script.
- Record after the 24 Oct freeze, so any number you say is final.

---

## Company
**Company name:** NeurOS [pending Q32].

**Describe what your company does in 50 characters or less.**
> Calibration-free brain input for any EEG headset

(48 characters.)

**Company URL:** [landing page, if live].

**Demo (up to 3 min):**
- **The plan:** a random volunteer, real time, unedited screen capture, recorded 25–27 Oct.
- **If the founders' headset test fails its bar:** a replay of a never-seen dataset person, **labelled as a replay**.

**What is your company going to make? Please describe your product and what it does or will do.**
> NeurOS is a software engine that turns brain signals from any EEG headset into commands an app can use, with no calibration session. Today every brain-computer interface makes each new user sit through a calibration session (typically 10–15 minutes of training trials) before it works, and each headset maker builds its own decoder from scratch. NeurOS is one model trained across hundreds of people and many headsets. It works from minute zero, adapts silently while you use it, and holds rather than guessing when it's unsure. Developers embed it with a few lines of Python or JavaScript, test against real recorded brains in a simulator, and the decoding runs on the user's own machine.
>
> On 73 people it had never seen, from three datasets with no cue on screen, two minutes of normal use with no labels match what 5–20 labelled trials per hand of conventional calibration give ‡. A 3-minute calibration on top comes within 1–3 points of a full 10–13-minute session ‡.

*Wording rules (D-040):*
- Always pair the calibration claim with the level (about 60–70% two-class ‡).
- Never say "the same accuracy as a full calibration".
- Never say "zero-shot" (claims audit row 25).

**Where do you live now, and where would the company be based after YC?** [Charlottesville, VA (UVA) → San Francisco?] **Explain:** [your reason].

---

## Progress
**How far along are you?**
> We have a working engine and a measured model, and we're starting live tests.
> - **The engine** runs live: headset stream in (Lab Streaming Layer or BrainFlow), signal-quality gating, online alignment to the new person, decoding, a confidence policy that holds when unsure, and commands out over WebSocket. Its input windows match our offline training pipeline bit for bit.
> - **The model** was trained on 339 people from 7 datasets, with a locked test set nobody has touched.
> - **The replay test:** we replayed 34 never-seen people from two other lab systems through the live engine. Committed commands were right 61% of the time ‡, at under 2 ms latency.
> - **Next:** a consumer headset (OpenBCI Ganglion) is in our live tests now [update], and [N] customer interviews are done [update].

**How long have each of you been working on this? How much of that has been full-time?**
> [Since late September 2026: the idea work was in September; coding started 3 Oct. Say honestly how much of it was full-time alongside classes.]

**What tech stack are you using, or planning to use? Include AI models and AI coding tools you use.**
> Python (PyTorch, MNE, pyRiemann, MOABB), a compact convolutional decoder (EEGNet-style, ~20k parameters) with Riemannian alignment, Lab Streaming Layer and BrainFlow for headsets, WebSocket and a JavaScript client for apps. Training runs on one AWS GPU instance. We write code with Claude Code (Anthropic) as a pair programmer; every experiment has its pass bar written before it runs, and every number regenerates from a committed command.

**Are people using your product?** [No / Yes: who, how many.] **Do you have revenue?** [No.]

**If you've participated in an incubator or accelerator:** [None / …].

---

## Idea
**Why did you pick this idea to work on? Do you have domain expertise? How do you know people need what you're making?**
> [Founders' own story: why brain-computer interfaces, why the two of you.]
>
> The bottleneck we kept hitting is that BCIs don't transfer: a decoder built for one person fails on the next, so every user starts with a calibration session and every headset maker rebuilds the same software. We went looking for how good "no calibration" really is, and found the standard benchmark's ~75% zero-calibration result is mostly the on-screen arrow, not the brain. The arrow alone decodes at 75.8%; after it leaves the screen, the same model scores 60.6% ‡ (one dataset, 9 people). So we built the evaluation first: every number we publish starts after the cue has left the screen.
>
> **How we know people need it:** [interviews: who, how many, the pain in their words; any letters of intent. Fill only with what happened.]

**Who are your competitors? What do you understand about your business that they don't?**
> - **Zander Labs** reads mental states on its own hardware and is collecting thousands of recordings for universal classifiers.
> - **Arctop** sells read-only access to mental-state scores computed in its own cloud.
> - **Neurable and Muse** build headsets with their own apps.
> - **Neuralink** and the other implant companies serve surgical patients.
>
> None of them gives a developer an engine they can embed that turns *any* headset's signal into commands on the user's device, and none publishes calibration-free accuracy with the cue controlled.
>
> What we understand:
> 1. **The leaks.** Much of the field's "calibration-free" accuracy is cue and eye leakage. Honest numbers are lower, but they're the ones that survive a live demo.
> 2. **Calibration is a time cost.** It can be bought down, from minutes to seconds of normal use, without ever claiming perfection.
> 3. **The market is fragmented.** A dozen headset makers each need decoding software and can't afford a research team, the way phone makers couldn't each build an OS.

*Prepare for the interview: "Isn't this Arctop?" → "Arctop reads mental-state scores out of its own cloud. NeurOS is an engine developers embed, turning any headset's EEG into commands on the user's machine, with calibration-free accuracy measured in cue-controlled tests." ([competitors](../business/competitors.md))*

**Why not build your own hardware, lock down your platform, or compete with Neuralink?** (Likely interview question, not on the form.)
> Hardware is a crowded, capital-heavy race we'd enter years behind. The open gap is the layer between the hardware and the apps. Every new headset makes the layer more valuable, and every app built on it makes headsets more useful.

**How do or will you make money? How much could you make?**
> - **Headset makers:** a per-device licence for the embedded engine, so their device ships with working brain input on day one.
> - **App developers:** free for research and prototypes, paid per active user in commercial apps.
> - **Enterprise and defence:** custom intents and on-premise deployment.
>
> [Market size: build it bottom-up from sourced numbers (EEG headsets sold per year × licence fee, plus developer seats). Don't reuse the founder notes' $30B / $4.5B / $150M: they're unsourced (claims audit row 11).]

**Which category best applies?** [B2B / Developer Tools, or Healthcare: choose B2B.]

**Other ideas you considered:**
> - Brain-signal sign-on (an EEG "neuralprint").
> - A communication aid where a language model proposes words and the brain's recognition response picks them.
> - A passive focus and workload API on consumer headphones.

---

## Equity
**Legal entity?** [No / yes: type, state.] **Investment taken?** [No.] **Fundraising?** [No.]

## Curious
**What convinced you to apply? Did someone encourage you? YC events?** [Honest answer.] **How did you hear about YC?** [Honest answer.]

---

## Numbers to replace after the 24 Oct freeze

| ‡ Development number in this draft | Source | Replaced by |
|---|---|---|
| 339 training people from 7 datasets; 73 cue-free test people from 3 | [mvp-v0 replay](../results/2026-10-10-mvp-v0-replay.md), E1, E2 | Frozen counts (locked test included) |
| 2 min of use ≈ 5–20 labelled trials per hand | [Calibration curve](../results/2026-10-10-calibration-curve.md) | Locked-test calibration curve |
| 3-min calibration within 1–3 points of 10–13 min | [Q30](../results/2026-10-10-q30-finetune-q31-fit-check.md), D-040 | Locked-test Q30 (offline only: the engine's version failed its replay bar) |
| Level about 60–70% two-class | E2, Q30 | Locked-test cue-free numbers |
| 61% of committed commands right; latency under 2 ms | [mvp-v0 replay](../results/2026-10-10-mvp-v0-replay.md) | Final replay on the locked test |
| 75.8% from the arrow alone; 60.6% after it | [E0](../results/2026-10-03-e0-eegnet-bnci2014-001.md) | Keep as-is: it's a finding about the benchmark, not our headline. Give the protocol |
| Mental-state channel | [E6](../plan/e6-state-channel.md), not run | Only if E6 passes |
