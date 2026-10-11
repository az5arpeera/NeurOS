# YC W27 application: responses v1

**Updated:** Sun 11 Oct 2026. The questions are copied from the live W27 portal (pasted by Azhar). The answers follow the founders' [writing guide](writing-guide.md): plain English, answer first, numbers instead of adjectives.

**Deadline:** Mon 2 Nov 2026, 8:00 PM PT (YC's apply page). We aim to submit **Fri 30 Oct**.

## Before submitting
- **‡ marks a development number.**
  - Development numbers can't go in the application ([results rules](../results/README.md)).
  - Each is replaced by its frozen number after the **24 Oct freeze**, and every claim is re-checked against the [claims audit](../claims-audit.md).
- **[Brackets]** are for the founders. Never invent traction, users or quotes.
- **The company name:**
  - The portal draft says **"NuerOS"**; the repo and every document say **NeurOS**. Is the spelling intentional?
  - Either way, Q32 still applies: Arctop's live NEUOS mark is one letter from both. Settle the name before filming the video.
- **Q28, the lead:** these answers lead with "calibration saved" plus rigour.
  - E6 failed its bar ([results](../results/2026-10-11-e6-state-channel.md)), so no answer claims a validated mental state.
  - Mental states appear only as what we're working on next.

---

## Founders

**How long have the founders known one another and how did you meet? Have any of the founders not met in person?**
> [How long, where you met (e.g. UVA, which class or project), and something you've built or done together before NeurOS. Say "We've both met in person" if true.]

**Who writes code, or does other technical work on your product? Was any of it done by a non-founder? Please explain.**
> We do. [Azhar] built the data pipeline, the evaluation harness and the live engine. [Founder 2] leads the model experiments. We write code with an AI coding agent (Claude Code) that we direct and review: every change goes through our tests (864 today) and continuous integration. No non-founder has written code or done technical work.

*Why this matters: YC's #1 applicant tip is that one founder can clearly build the product. Keep "we built it" first.*

**Are you looking for a cofounder?** [No.]

**Founder video (one minute, all founders):**
- Who you are; what NeurOS does in one sentence; why you two; one thing you've built.
- Unscripted, energetic, under 60 s.
- Record after the 24 Oct freeze and after the name is settled.

---

## Company

**Company name:** [NeurOS / NuerOS: see above]

**Describe what your company does in 50 characters or less.**
> Brain-signal input for any EEG headset, no setup

(48 characters.)

*Alternatives:*
- "Calibration-free brain input for any EEG headset" (48)
- "Lets apps read brain signals from any headset" (45)

**Company URL:** [landing page, if live]

**Demo (3 min / 100 MB):**
- **The plan:** a real person using it live, unedited screen capture, recorded 25–27 Oct: raw EEG in, the decoder's confidence, commands out, and "hold" when unsure.
- **If the founders' headset test fails its bar:** a replay of a never-seen dataset person, **labelled as a replay**.

**Link to the product / login credentials:** [GitHub or docs link if we open a developer preview; otherwise leave blank.]

**What is your company going to make? Please describe your product and what it does or will do.**
> We're building the software that lets apps read brain signals from any EEG headset, with no setup session. Today a brain-computer interface makes every new user sit through 10–15 minutes of training trials before it works, and every headset maker writes its own decoding software from scratch. NeurOS is one model trained on hundreds of people. It works from the first minute and adapts quietly while you use it. Developers add it with a few lines of Python or JavaScript, test against real recorded brains in our simulator, and ship. It runs on the user's own computer, so brain data doesn't have to leave it. We start with commands (left, right, and "hold" when it isn't sure) and are adding mental states such as rest vs focused effort.
>
> On 73 people it had never seen, from three datasets with no cue on screen, two minutes of ordinary use matched what 5–20 labelled training trials give ‡. A 3-minute calibration on top came within 1–3 points of a full 10–13-minute session ‡.

*Wording rules:*
- Always pair the calibration claim with the level (about 60–70% for two commands ‡; D-040).
- Never say "zero-shot" (claims audit row 25).
- Never say "the same accuracy as a full calibration".

**Where do you live now, and where would the company be based after YC?** (format: City A, Country A / City B, Country B)
> [Charlottesville, USA] / San Francisco, USA

**Explain your decision regarding location.**
> [Draft:] The people we sell to, BCI and wearable startups and the investors behind them, are concentrated in the Bay Area, and YC is there. [Add anything personal, such as school plans.]

---

## Progress

**How far along are you?**
> We have a working engine and a tested model, and live tests on our own headset start this month.
> - **The engine runs end to end:** headset stream in, signal-quality checks, alignment to the new person, decoding, a confidence policy that holds when unsure, and commands out to any app. Its input matches our training pipeline bit for bit.
> - **The model** was trained on 339 people from 7 public datasets; a test set of people nobody has touched is locked until our 24 Oct freeze.
> - **A replay test on 34 never-seen people from two other lab systems:** committed commands were right 61% of the time ‡, at under 2 ms per decision.
> - **We found a flaw in the standard benchmark:** its ~75% "no-calibration" result mostly reads the arrow on screen, not the brain. The arrow alone decodes at 75.8%; after it's gone, 60.6% ‡. Every number we publish is measured after the cue leaves the screen.
> - **Customers:** [N] interviews with BCI labs and headset companies; [letters of intent].

**How long have each of you been working on this? How much of that has been full-time? Please explain.**
> [Example shape: "Since September 2026. Research and design in September; we started coding on 3 October. [Azhar]: [hours/week] alongside classes at UVA. [Founder 2]: [...]." Be exact and honest.]

**What tech stack are you using, or planning to use, to build this product? Include AI models and AI coding tools you use.**
> - **Language and model:** Python. A compact convolutional network (EEGNet-style, about 20,000 parameters) with per-person alignment (Riemannian geometry), trained in PyTorch on public datasets via MNE and MOABB.
> - **Headsets:** Lab Streaming Layer and BrainFlow, so most research and consumer headsets plug in.
> - **Apps:** a local WebSocket server and a JavaScript client.
> - **Training:** one AWS GPU instance.
> - **AI tools:** we write code with Claude Code as a pair programmer. Every experiment has its pass bar written before it runs, and every number can be regenerated from a committed command.

**Are people using your product?** [No.]

**When will you have a version people can use?**
> [Suggested: "A developer preview (the engine, Python and JavaScript SDKs, and the simulator) in [December 2026]; the first live sessions with volunteers on our own headset are this month." Pick a date you'll hit.]

**Do you have revenue?** [No.]

**If you are applying with the same idea as a previous batch …** [First application: leave blank, or "This is our first application."]

**If you have already participated or committed to participate in an incubator, accelerator or pre-accelerator, please tell us about it.**
> [Jefferson Trust: say exactly what it is. It's UVA's alumni grant programme, not an accelerator. Applied, received, amount, for what? If it's a grant, say so plainly; if nothing applies, leave this blank.]

---

## Idea

**Why did you pick this idea to work on? Do you have domain expertise in this area? How do you know people need what you're making?**
> [Founders' own story, first: why brain-computer interfaces, and why you two.]
>
> Brain-computer interfaces don't carry over from one person to the next. A decoder built on one brain fails on the next, so every user starts with a training session, and every headset company rebuilds the same software. We set out to measure how good "no calibration" really is, and found the field's standard number was mostly the on-screen cue. So we built the evaluation first, then the engine.
>
> **Expertise:** [Azhar: systems engineering and mathematics at UVA; Founder 2: neural-network research]. We've built the full pipeline on [~400] people from [10+] public datasets.
>
> **How we know people need it:** [N interviews: who (labs, headset makers, developers), the pain in their words, how they handle calibration today and what it costs them; any letters of intent. Fill only with what happened.]

**Who are your competitors? What do you understand about your business that they don't?**
> - **Arctop** scores mental states (focus, cognitive load) in its own cloud for research teams.
> - **Zander Labs** reads mental states on its own hardware.
> - **Neurable and Muse** sell headsets with their own apps.
> - **Neuralink** and other implant companies serve surgical patients.
>
> None of them gives developers an engine to put inside their own app that works on any headset, decodes commands, and runs on the user's device. None publishes how well it works on a new person with the cue controlled.
>
> What we understand:
> 1. **The leaks.** Much of the field's "calibration-free" accuracy is the cue or eye movements. Honest numbers are lower, but they're the ones that hold up in a live demo.
> 2. **Calibration is a time cost.** It can be bought down, from 15 minutes to a few minutes of normal use, without ever claiming perfection.
> 3. **The market is fragmented.** A dozen headset makers each need decoding software and can't afford a research team, the way phone makers couldn't each build an OS.
>
> Long term, we want one layer for both commands and mental states, which is where we meet Arctop. We start with commands, where nobody is.

**How do or will you make money? How much could you make?**
> - **Headset makers:** a per-device licence for the embedded engine, so their device ships with working brain input on day one.
> - **App developers:** free for research and prototypes; paid per active user in commercial apps.
> - **Enterprise and defence:** custom commands and on-premise deployment.
>
> [Bottom-up estimate, sourced: (EEG headsets sold per year) × (licence per device) + (developers) × (seat price). Find the headset numbers before writing this; don't reuse the founder notes' unsourced $30B / $4.5B / $150M (claims audit row 11).]

**If you had any other ideas you considered applying with, please list them.**
> - **Mental-state monitoring for operators** (pilots, drivers, defence) on consumer headsets: we measured rest vs effort across people at about 71% ‡, working for two in three people so far.
> - **A communication aid for people who can't speak or type:** a language model proposes words, and the brain's "that's it" response picks them.
> - **Brain-signal sign-in.**

*The first idea's number comes from E6, a development result whose bar failed. Keep it or cut it after the freeze; never call it validated.*

---

## Equity
**Have you formed ANY legal entity yet?** [Yes / No]

**If not, the planned equity breakdown among founders, employees and other stockholders, with each founder's title:**
> [Founders decide together: e.g. "[Azhar], CEO, [x]% / [Founder 2], CTO, [y]%; no employees or other stockholders yet." YC notes this question is "as much for you as us": decide it explicitly, in writing, before submitting.]

**Have you taken any investment yet?** [No.] **Are you currently fundraising?** [No.]

## Curious
**What convinced you to apply to Y Combinator? Did someone encourage you to apply? Have you been to any YC events?** [Honest answer: Startup School videos, events, who encouraged you.]

**How did you hear about Y Combinator?** [Honest answer.]

## Batch preference
**What batch do you want to apply for?** Winter 2027.

---

## Development numbers to replace after the 24 Oct freeze

| ‡ Number in these answers | Source | Replaced by |
|---|---|---|
| 339 training people, 7 datasets; 73 cue-free test people, 3 datasets | [mvp-v0 replay](../results/2026-10-10-mvp-v0-replay.md), E1, E2 | Frozen counts (locked test included) |
| 2 min of use ≈ 5–20 labelled trials per hand | [Calibration curve](../results/2026-10-10-calibration-curve.md) | Locked-test calibration curve |
| 3-min calibration within 1–3 points of 10–13 min | [Q30](../results/2026-10-10-q30-finetune-q31-fit-check.md), D-040 | Locked-test Q30 (offline only: the engine's version failed its replay bar) |
| Level about 60–70% for two commands | E2, Q30 | Locked-test cue-free numbers |
| 61% of committed commands right; under 2 ms | [mvp-v0 replay](../results/2026-10-10-mvp-v0-replay.md) | Final replay on the locked test |
| Arrow alone 75.8%; after it, 60.6% | [E0](../results/2026-10-03-e0-eegnet-bnci2014-001.md) | Keep: a finding about the benchmark (9 people, BNCI2014-001), quoted with its protocol |
| Rest vs effort about 71%, two in three people | [E6](../results/2026-10-11-e6-state-channel.md) (bar failed) | Other-ideas answer only, or cut |
| 864 tests | CI | The count on submission day |
