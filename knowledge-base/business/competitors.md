# Competitors and adjacent companies (v2, 2026-10-11)

**Purpose:** what each company actually does, where it overlaps with NeurOS, and whether it competes directly or sits next to us. Feeds the YC application's competition answer.

**Rules** (as in the [claims audit](../claims-audit.md)):
- Every fact below comes from the linked source, checked on 4 Oct 2026 (Arctop and Zander's January demo: 11 Oct).
- Company claims are reported *as claims*: none of these companies has published evidence for its calibration claims that we found.
- Re-check before quoting.

## Summary

| Company | What they sell | Signal / hardware | Calibration story | Versus NeurOS |
|---|---|---|---|---|
| **Zander Labs** (NL/DE) | A platform that reads **mental states** (workload, focus, error, surprise, stress, "intent") and makes software and AI react to them. B2B building block, plus professional services | Scalp EEG on **their own hardware**: the Zypher EEG patch, a 24-channel mini amplifier, a classification chip | Building **"universal classifiers"** that work for any person, headset and environment, with no individual calibration. Collecting 4,000 people's datasets (1,000 done in 2025). **Jan 2026: an integrated "plug-and-play" demo** (cognitive load adapts a UI, decoded locally). Still no published accuracy | **Direct competitor on our core thesis** (calibration-free, any person, any headset) and on our brain-state channel. **Adjacent** on intent commands |
| **Synaptrix Labs** (NYC) | A **medical device**: brain-controlled wheelchairs for people with paralysis, via the FDA route. Also licenses artefact-removal AI to an EEG maker | Scalp EEG on **their own hardware**: the registered device uses 4 dry sensors over the visual cortex; new "Portal" hardware is high-density over the motor cortex | Per-user personalisation in the cloud; third-party articles say "minimal recalibration". **No zero-calibration claim** | **Adjacent:** one clinical application on their own hardware. Overlaps on deep-learning EEG decoding, aggregating public datasets, and licensing software to hardware makers |
| **Arctop** (San Francisco) | **"An API for the human inside"**: software that decodes mental states (cognitive load, attention, enjoyment, sleep) from EEG in everyday wearables. A self-serve developer API, custom classifiers, its own app, and an EEG-adaptive AI agent | **No hardware of its own.** Names Master & Dynamic's MW75 Neuro headphones (12 channels, 500 Hz). Decodes in **its cloud** | **No calibration claim** on its site; no published accuracy | **Direct competitor on the business model** (the layer between headsets and apps, sold to developers). **Adjacent on signal:** passive states, not commands |
| **Neuralink** | Brain implants | Intracortical spikes | Per-person pretraining; zero-shot is a stated, unsolved goal ([note](../research/papers/2026-neuralink-pretraining-50000-hours.md)) | **Different market** (surgery). Validates the problem |

## Zander Labs
**What they are:**
- A Dutch-German company founded by Prof. Thorsten O. Zander, who introduced the concept of the *passive* BCI (2008). "Passive" means reading mental states in the background; the user gives no commands.
- Offices in Delft, Berlin, Munich and Cottbus. CEO Jonathan Zwaan.
- The team more than doubled in 2025: 26 new colleagues and 9 interns.

**Products** ([solutions](https://www.zanderlabs.com/solutions)):
- **SAMANAI® platform:**
  - multi-dimensional mental state recognition, linked to context ("CAT");
  - a framework to make third-party software neuroadaptive ("NHCI");
  - an "AI learning engine" that lets AI learn from human brain responses.
- **Zypher EEG suite**, their own hardware:
  - a mobile EEG patch (V2 in 2025);
  - a 24-channel miniature amplifier (with Sencure);
  - a "BCI-on-a-chip" that classifies on the device and sends only results over Bluetooth (privacy).
  - Launch in 2026: partner institutions first, universities in Q2, then companies ([2025 review](https://www.zanderlabs.com/blog/2025-in-review)).
- **Professional services.**

**Customers and money:**
- **The €30M NAFAS contract** from Germany's Agency for Innovation in Cybersecurity (Cyberagentur). They call it the largest of its kind in the EU: neuroadaptive prototypes for human–machine interaction and AI learning ([NAFAS](https://www.zanderlabs.com/nafas)).
- **Customers:** the site names Volkswagen and Siemens.
- **Recognition:** a BPI France award (defence and security); Deloitte Technology Fast 50, fastest-growing.
- **Equity funding:** small or unclear in the databases (PitchBook/Tracxn show about $1.5M in seed rounds and grants). The €30M is a contract, not equity.

**The calibration claim** ([scientific](https://www.zanderlabs.com/scientific), [foundation-model blog](https://www.zanderlabs.com/blog/foundation-model-in-eeg-research)):
- **The goal:** universal classifiers giving a plug-and-play experience regardless of person, headset or environment.
- **The data:** 4,000 datasets from people at their Dutch and German sites, by 2027 (1,000 reached in 2025).
- **Approach:** they're watching EEG foundation models closely.
- **Status:** in development. We found no published accuracy for the universal classifiers.
- **January 2026 demo** ([release, 7 Jan 2026](https://www.zanderlabs.com/resources/fully-integrated-system)):
  - their first fully integrated system: the Zypher electrodes and amplifier plus an on-chip decoder;
  - it detected cognitive load and adapted a user interface in real time, with all processing local;
  - they claim the algorithms are "plug-and-play" and decode any individual "without prior personalization", from pre-training on large EEG datasets;
  - **not stated:** any accuracy, number of users or readiness level. Only cognitive load is named.
- **Hardware roll-out** ([2025 review](https://www.zanderlabs.com/blog/2025-in-review)): the Zypher patch goes to partner institutions first, universities and knowledge institutes in Q2 2026, then more companies later in the year; prototypes go to partners through 2026. No third-party SDK or API is mentioned.

**US entry:** announced 12 Mar 2026, seeking partners on the West Coast (Bay Area). Target sectors: defence, mobility, robotics, healthcare, specialist jobs such as air traffic control. The CEO describes the technology as a building block for partners, not an end product ([Business Wire](https://www.businesswire.com/news/home/20260312113517/en/DeepTech-Company-Zander-Labs-Enters-the-U.S.-Market-With-Groundbreaking-pBCI-Technology)).

**Same as NeurOS:**
- The core thesis: calibration is what blocks BCIs from real use, and the fix is models that generalise across people *and headsets*.
- A B2B "layer" sold to partners.
- Privacy by processing locally.

**Different from NeurOS:**

| | Zander Labs | NeurOS |
|---|---|---|
| Output | Passive **mental states** for enterprise and defence systems | **Active intent commands** (LEFT, RIGHT, REST, HOLD) for app developers, plus a smaller brain-state channel |
| Hardware | Their own (patch, amplifier, chip), classifiers on-chip | **Any headset** over LSL, including open and consumer boards |
| Data | Proprietary recordings (4,000 people, their paradigms) | Public datasets (~380 people, 7 setups) |
| Distribution | Professional services and partnerships | Developer SDK (Python, JavaScript), simulator |
| Approach | Universal classifiers out of the box (L0) | Silent adaptation in seconds of normal use (L1) on top of a pooled model |

**Verdict:**
- **Direct competitor for the calibration-free narrative and for our brain-state channel.** They have more money, more proprietary data and a head start there.
- **Adjacent for intent commands,** their stated secondary target.

## Arctop
**What they are:**
- Founded in spring 2016 in San Francisco by Dan Furman, PhD (CEO, neuroscientist; his Technion PhD decoded imagined finger movements from EEG) and Eitan Kay (CTO) ([site](https://www.arctop.com), [interview, Apr 2026](https://theneurotechnewsletter.substack.com/p/arctop-and-the-intelligence-layer)).
- **The CEO's framing is ours:** Arctop works on "the intelligence layer between brain-sensing hardware and the applications built on top of it." Its tagline is "An API for the human inside", and it says it "does not make hardware."

**Products** ([site](https://www.arctop.com)):
- **Arctop API:** a public developer API with self-serve keys ([docs](https://docs.arctop.com)). Apps on connected devices read decoded scores in real time.
- **Custom classifiers** for partners: states a study can define and measure.
- **The Arctop app** on Master & Dynamic's MW75 Neuro headphones, measuring cognitive load live. The headphones themselves were built with Neurable (see below).
- **Isaac:** an EEG-adaptive conversation agent in the app.
- **RLbF** ("reinforcement learning from brain feedback"): decoded EEG as a reward signal to adapt a language model to its user. It's in a preprint (Aug 2026), not peer-reviewed.
- **History:** the earlier platform was called **Neuos**. Its first product, Neuos Eureka (Aug 2020), measured audiences' reactions to media content ([Research Live](https://www.research-live.com/article/news/arctop-launches-brain-decoding-platform/id/5072892)).

**Outputs:** cognitive load, visual and auditory attention, enjoyment, sleep, memory and identity. All passive states; no commands.

**Privacy:** EEG is encrypted on the device and **decoded in Arctop's cloud**. Language models receive only the derived scores.

**Customers and money:**
- **$10M Series A** (Oct 2023), co-led by Fifth Growth Fund and Supermoon Capital ([VCA Online](https://www.vcaonline.com/news/2023101017/arctop-a-cognition-company-building-software-that-decodes-the-human-brain-announces-10m-series-a-funding-co-led-by-fifth-growth-fund-and-supermoon-capital)).
  - Its site also lists In-Q-Tel and Script Capital as investors; neither is confirmed elsewhere.
- **US Air Force:** a $1.25M AFWERX Direct-to-Phase II SBIR (Sep 2024): cognition-adaptive training simulations, with MetaTeq and BigBear.ai ([release](https://www.defence-industries.com/index.php/pressreleases/arctop-secures-125m-afwerx-contract)).
- **Patents:** 13 granted, according to its site (US 6, Europe 2, Japan 2, Israel 3).
- **Customers:** no enterprise customers named.

**Calibration and evidence:**
- **No calibration claim** on its site, either way, and no accuracy figures.
- **Evidence cited:** only the RLbF preprint. The 2022 focus result with Endel is a company white paper.

**Same as NeurOS:**
- **The business model:** a hardware-agnostic layer between headsets and apps, sold to developers through an API.
- Consumer form factors (headphones, earbuds, headbands) as the target hardware.

**Different from NeurOS:**

| | Arctop | NeurOS |
|---|---|---|
| Output | **Passive states** (load, attention, enjoyment) | **Active intent commands** (LEFT, RIGHT, HOLD with a reason), plus one state channel |
| Where decoding runs | **Their cloud** | **On the user's machine** (local by default, §10) |
| Hardware | Supported wearables (one named) | Any headset streaming over LSL or BrainFlow, including open boards |
| Calibration | No claim | Zero calibration by default, adapting from normal use; measured |
| Evidence | Company white papers, one preprint | Cue-controlled results on ~380 public-data people, with bars set before each run |
| Ships to developers | API keys | Open SDK (Python, JavaScript) and a simulator that replays real datasets |

**Verdict:**
- **Direct competitor on the business model.** They already sell what our thesis describes: a self-serve brain-decoding API with no hardware of its own. They're seven years in, with a Series A and defence contracts.
- **Adjacent on the signal:** they decode passive states; we decode commands.
- **Where we're different:** commands (they have none), local processing, measured zero-calibration and published cue-controlled evidence.
- **Naming risk:** their former platform name, **Neuos**, is close to **NeurOS**, in the same field.
  - A quick search found no trademark record for either name.
  - The founders should run a proper trademark search before the name goes into the application.

## Synaptrix Labs
**What they are:**
- Founded in 2023 in New York by Aryan Govil (CEO; previously five years at an NYU Langone clinical research lab) and Eric Yao (CTO; background in time-series forecasting and quantitative modelling).
- Seven full-time employees in Oct 2025.
- Seed round led by Mark Cuban, size undisclosed. "Industry experts" are quoted estimating $8–10M; that's speculation, not a reported figure.
- Pre-seed from Underdog Labs. Creative Destruction Lab participant.
- Sources: [Forbes / Naveen Rao](https://www.forbes.com/sites/naveenrao/2025/10/24/noise-to-signal-cuban-backs-synaptrix-labs-non-invasive-bci-platform/) via [its newsletter copy](https://neurotechnology.substack.com/p/synaptrix-cuban), [citybiz](https://www.citybiz.co/article/764275/synaptrix-labs-gets-seed-funding-from-mark-cuban-to-advance-brain-computer-interface/).

**Products** ([site](https://www.synaptrix-labs.com/)): they describe themselves as a vertically integrated "neuroAI" platform.
- **Neuralis / Portal:** their own EEG hardware. Portal is new: high-density, over the motor cortex, wireless.
- **NeuroDiffusion™:** a real-time deep-learning model whose primary purpose is cleaning EEG.
  - It's licensed to IDUN Technologies, an EEG hardware maker, for artefact removal.
- **Cortex™:** a "brain data lake" combining published datasets with data from their deployed devices, for foundation models.
- **They also run an EEG recording study** that recruits healthy volunteers to collect training data.
- **The stated ambition:** non-invasive systems that match implants within the decade.

**Clinical:**
- **EMPOWER** early feasibility study, [NCT06169696](https://ctv.veeva.com/study/empower-early-feasibility-study-non-invasive-bci-to-control-a-wheelchair-for-people-with-paralysis): wheelchair control for people with paralysis, 6 participants planned.
- The registration (last updated Jan 2024) describes the device as **4 dry sensors over the visual cortex**, controlled through focused attention and eye movements, with app-side calibration settings and cloud-side "model training and personalization".
- Its listed status was "not yet enrolling"; we don't know the current status.
- Articles mention a Columbia University trial and an FDA De Novo pathway.

**Same as NeurOS:**
- Non-invasive EEG and deep learning.
- Aggregating published datasets.
- Selling software to hardware makers (the IDUN licence).
- Assistive control as a use case.

**Different from NeurOS:**
- They build their own hardware and take one medical application through the FDA.
- Their decoding is personalised per user; we found no zero-calibration claim.
- Their first device appears to use visual attention and eye movements, not motor imagery.
- Their software product today is denoising, not intent across headsets.

**Verdict: adjacent.** They'd become direct if NeuroDiffusion grew from denoising into intent decoding sold across headsets.

## What this means for NeurOS
1. **We can't claim to be the only calibration-free EEG company.** Zander Labs says the same thing, with €30M of government money and 1,000+ proprietary datasets.
   - Our honest edges:
     - **active intent for developers** (they do passive states for enterprises);
     - **any headset, including the cheap open ones,** versus their own hardware;
     - **an open developer SDK and simulator** versus professional services. (A self-serve API alone is no longer an edge: Arctop has one.)
     - **published, protocol-controlled evidence** (public data, leave-one-dataset-out, and the cue-artefact controls we just added) versus claims without published numbers;
     - **adaptation in seconds** (L1) rather than betting everything on one universal model.
2. **The brain-state channel overlaps Zander's core.** Lead with intent and ship one validated state (rest vs mental effort) as the second channel; no parity claim (D-031, 4 Oct).
3. **Investor appetite is real:** Mark Cuban backing Synaptrix, the €30M contract for Zander. Both are "why now" evidence.
4. **Hardware makers do buy EEG software** (Synaptrix → IDUN). That supports our B2B channel. IDUN is on our backup contact list; it may already have a software partner.
5. **"They'll lock down their ecosystems, we're open" doesn't hold as stated** (checked 11 Oct).
   - Zander calls its technology a building block for partners, and Arctop already sells an open developer API without hardware.
   - **The edge has to be specific:**
     - commands, not only states;
     - any headset, including cheap open boards;
     - local processing;
     - measured zero-calibration with cue-controlled evidence, where both competitors publish no accuracy.
   - **Hardware makers are building their own models too:**
     - Neurable built the brain sensing in Master & Dynamic's MW75 Neuro headphones.
     - Muse (Interaxon) announced its own "foundation brain model" in June 2025, per its press release.
     - Both are to profile next.
6. **Data:** Zander's proprietary recordings and Synaptrix's "Cortex" both aggregate data. Public-dataset aggregation (which we do too) is not a moat for anyone. Ours has to come from the cross-headset evaluation, the adaptation runtime and, later, opt-in data from deployed users.
