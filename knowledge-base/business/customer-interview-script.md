# Customer-interview script (v1, 2026-10-03)

**Purpose:** learn who feels calibration pain most, what it costs them today, and whether they'd pilot NeurOS. Every conversation also counts toward the YC metric "customer conversations" (target 15+ by 30 Oct; [plan](../plan/road-to-nov-1.md)).

**Founders run every conversation and send every message; Claude drafts only.** Informal and consented (D-006): ask before recording, store no names in shared notes, and never collect health information.

## 1. Who to talk to (D-015)

| Segment | Examples | What we most need to learn |
|---|---|---|
| **A. Research labs** | UVA EEG/BCI/neuroscience labs: PIs, postdocs, grad students, lab managers | How much participant and RA time calibration eats; whether failed sessions hurt their studies |
| **B. Developers** | OpenBCI forum builders, hackathon teams, indie BCI app makers | Whether calibration stops people from using their apps, and what they've hacked together instead |
| **C. BCI companies** | Startups shipping EEG products (headsets or apps) | Onboarding drop-off, support load, and whether a calibration-free SDK is something they'd license |

Aim for 8–10 conversations by Fri 16 Oct and 15+ by Fri 23 Oct, at least 3 per segment.

## 2. Rules (from *The Mom Test*, Rob Fitzpatrick)

1. **Talk about their life, not our idea.** The pitch comes last, if at all.
2. **Ask about specific past events,** not opinions or the future. "Last time…" beats "Would you…".
3. **Listen 80% of the time.** Silence is fine; they'll fill it.
4. **Compliments are noise.** "Sounds cool" means nothing; time, money or an intro means something.
5. **Every conversation ends with a concrete next step** (or we learn there isn't one).

## 3. Opening (≈ 2 minutes)

> "Thanks for making time. I'm [name]; my cofounder [name] and I are working on EEG brain-computer interfaces. We're trying to understand how people actually set up EEG with a new person: what takes time, what goes wrong. We're not selling anything today. Is it OK if I take notes? [If recording: Is it OK if I record, just for our notes?]"

## 4. Core questions (≈ 20 minutes)

Pick 6–8. Follow the energy: when they get animated, dig there ("Tell me more about that", "Why was that hard?").

**Their work**
1. What are you working on with EEG right now?
2. Who uses your setup: you, participants, customers? How many new people a month?

**The last time (the heart of the interview)**
3. Walk me through the last time you set up EEG with someone new, from putting the cap on to getting usable data.
4. How long did each part take? Which part was calibration or training a decoder?
5. What went wrong that time? What happened to that session or user?
6. How often does someone just not work? (Motor imagery "BCI inefficiency" is 15–30% of people.) What do you do then?

**Cost and workarounds**
7. What does an hour of a session cost you? (Participant pay, RA time, equipment time, user patience.)
8. What have you tried to shorten or skip calibration? (Transfer learning, reusing old models, fewer trials, SSVEP instead.) How did it go?
9. If calibration disappeared tomorrow, what would you do with the time? *(Use near the end; it's hypothetical.)*

**Segment-specific**
- **A (labs):** Do failed or short calibrations ever force you to exclude participants? How many per study?
- **B (developers):** Have users quit your app at the calibration step? What did they say?
- **C (companies):** What share of new users finish onboarding? What do support tickets say about setup?

**Decision and budget**
10. Who decides on tools like this? What did you last pay for software or hardware in this area, and how was it approved?

## 5. Closing and the ask (≈ 5 minutes)

Only now show the idea, in one sentence:

> "We're building a layer that decodes intent from any EEG headset without a calibration session: it adapts silently during the first ~45 seconds of normal use."

Then climb the commitment ladder and stop at the highest rung they accept:

1. "Can we show you a 3-minute demo when it's ready, around mid-October?" (time)
2. "Who else should we talk to?" (intro: ask for two names)
3. "Would you try it on your setup in a short pilot?" (reputation and time)
4. "Would you sign a one-paragraph, non-binding letter of intent to pilot it?" (written commitment, see §7)

## 6. After each call (fill within 1 hour)

| Field | Notes |
|---|---|
| Segment, role, setup (headset, channels, paradigm) | |
| New people per month; minutes of calibration each | |
| Biggest pain, in their words (exact quote) | |
| What it costs them (time / money / data lost) | |
| Workarounds tried | |
| Pain score 1–5 (5 = they've spent money or built workarounds) | |
| Commitment reached (none / demo / intro / pilot / LOI) | |
| Follow-up promised, by whom, by when | |

**Strong signals:** they've built workarounds; they quote a cost; they ask "when can I try it?"; they offer an intro unprompted.
**Weak signals:** "interesting", "cool", "I'd probably use that", hypotheticals about the future.

## 7. Letter of intent (outline, for the founders to adapt)

One paragraph on the pilot partner's letterhead or email: who they are; that they're interested in piloting NeurOS's calibration-free decoding on their EEG setup during Oct–Dec 2026; what they'd provide (a setup, a few test sessions, feedback); that it's **non-binding** with no payment obligation; name, role, date. YC values specific, signed interest over vague enthusiasm.

## 8. Patterns to watch for across interviews

- Which segment reports the **highest pain score** and the **most new people per month**? That's the beachhead.
- Is the pain **calibration time**, **calibration failure** (people who never work), or **re-calibration across sessions**? Each points to a different headline.
- Do they care about **any headset** (hardware-agnostic) or only their own? This tests D-016's positioning.
