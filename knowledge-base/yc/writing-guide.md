# How to write the YC application: the founders' notes, condensed

**Source:** Azhar's notes (`yc application notes.md`, 11 Oct 2026): YC talks, slides and Startup School transcripts. The notes stay on the founder's laptop; this page keeps only the lessons, with a few published examples added. The answers themselves are in [application-responses-v1.md](application-responses-v1.md).

## The rules we write by
1. **Plain English.**
   - Talk to one person about the problem, then the solution.
   - No jargon: not "hardware-agnostic", "zero-shot", "unbundling" or "middleware".
   - If someone can't understand it, we can't sell it.
2. **Conclusion first (inverted pyramid).**
   - The first sentence of every answer is the answer. Proof follows.
   - Reviewers may skip answers or read them out of order, so each one stands alone.
3. **Show, don't tell.**
   - What we built, who uses it, who pays, and the special sauce, with a number wherever possible.
   - Cut any sentence that has neither a number nor a named thing in it.
4. **Simple statements, side by side.** Strung together, they should add up to a big idea. A great pitch never needs a brain teaser.
5. **Obfuscation reads as hiding.** Weak applications are vague. Be honest about what isn't done: YC funds companies at every stage.
6. **It has to be big.**
   - Show a path to a billion-dollar problem and $100M in revenue.
   - Starting niche is fine: Airbnb started with air beds.
7. **At least one founder must clearly be able to build it** (YC's #1 applicant tip). This is our strongest card: we built the engine and the evaluation ourselves.
8. **Tell a story:**
   - who we are;
   - what we're working on;
   - why;
   - what we've done so far;
   - why we'll make it work.
9. **Detail matters:**
   - punctuation and grammar;
   - a one-minute video with every founder in it, unscripted;
   - no novels.

**The odds:** per YC's slide, a well-written application that follows directions, technical cofounders and evidence of seriousness give about 4× the average applicant's chance of an interview.

## What good answers look like
**"What is your company going to make?"**
- **Remix** (transit mapping; the notes' good example): what it is, then who has the pain ("transit agencies take years to plan simple bus changes"), then the outcome ("we help them do it in weeks"), then the concrete things it does.
- **GitLab:** what it is, how it started, what it does, then one hard number: over 100,000 organisations.
- **Basedash** (S20, [founder's write-up](https://www.basedash.com/blog/how-to-write-a-successful-yc-application)): one sentence of what, plus one well-known comparison ("like Airtable but for your existing database").
- **The notes' "bad example"** (Fakeco): "revolutionizing … utilizing a blockchain-based machine learning framework …". Every word is abstract.

**"How far along are you?"**
- Dates and numbers: what's built, when it launched, users, money.
- With small top-line numbers, lead with depth (retention, named teams, a bug you caught) rather than totals ([Flowjam's collection](https://www.flowjam.com/blog/yc-application-examples-12-that-got-in-2026); secondary, unverified quotes).

**Competitors:**
- Name them.
- Say what you understand that they don't, ideally something you learned by building or from users.
- Make today's alternative sound painful.

**Money:**
- Size the market bottom-up: users × price.
- Never quote a top-down TAM you can't source (our founder notes' $30B / $4.5B / $150M are unsourced; claims audit row 11).

## The interview (10 minutes)
- **Fast and intense.** Expect to be interrupted, so practise being interrupted. Do a mock interview.
- **What they test:**
  - mastery of the business: the numbers, the biggest problems, what's next;
  - clear thinking;
  - self-awareness about risk.

  Memorised scripts backfire.
- **Know our numbers cold.** Before revenue, that means:
  - the accuracy numbers with their protocols;
  - the people counts;
  - monthly spend (AWS);
  - runway;
  - the interview count and letters of intent.
- **Hard-tech framing:**
  - **Market risk** (do people want it): interviews and letters of intent answer it.
  - **Technical risk** (can we build it): our results log answers it.
- **"Could you run your customer's business?"**
  - Could we teach a class on the problem?
  - Do we know a headset maker's or BCI lab's daily crises, and what calibration costs them?

  Interviews are how we get there.

**Prepared answers for likely pushback:**
- **"Isn't this Arctop?"**
  - Arctop scores mental states in its own cloud, for research teams.
  - We're an engine developers put inside their own apps, on any headset, on the user's device.
  - We start with commands, which Arctop doesn't do, and add mental states. E6: works for about two people in three on a full cap; not shipped yet.
- **"Why not your own hardware, or a locked platform, or compete with Neuralink?"**
  - Hardware is a crowded, capital-heavy race we'd join years behind.
  - The open gap is the layer between headsets and apps, and every new headset makes it more valuable.
- **"YC says ideas should verticalise and sell an outcome. Why a horizontal layer?"** The notes' own transcript makes this point (Corgi: be the insurer, not insurance software).
  - **Our honest answer:** the layer is the wedge. The outcome we own is "brain input that works on a stranger in the first minute".
  - **The vertical bet:** one vertical where we'd own the whole product. Candidates:
    - accessibility controls;
    - training simulators;
    - focus or fatigue monitoring for operators.
  - Decide which before the interview.

## Getting the first customers (for "how do you know people need it")
- **Customers 1–3:** our warm network: UVA labs, classmates, professors, people one intro away.
- **Customers 4–10:** things that don't scale:
  - show up in person;
  - small conferences;
  - the communities where the pain is posted (OpenBCI forum, r/BCI, NeurotechX);
  - founder dinners.
- **Outreach:**
  - Framed as asking for advice, and only if we genuinely want it.
  - Under 75 words, with one clear ask.
  - Give something first, e.g. run their public dataset through our cue audit and send them what we find.
  - Follow up 3–4 times.
- **A letter of intent:** a non-binding note from a company saying it intends to pilot or buy. Its confidentiality clauses can bind.
- **Only after 10–20 customers** do tools like Apollo, Clay or LinkedIn sequences start to pay off.

## Moats (for the competitor answer and the interview)
Speed, process power, cornered resources, switching costs, counter-positioning, branding, network economies, scale economies.

Don't over-think moats early. **Ours, honestly:**
- **Speed.**
- **Process power:** a cue-controlled evaluation harness and a live engine with bit-exact parity between training and live use.
- **Later, network economies:** opt-in data across headset brands improves the shared model.
