# UVA faculty outreach: email drafts (v1, 6 Oct 2026)

**Status:** drafts for the founders to adapt and send from a UVA address. Claude drafts only; the founders send.
- Every fact in the drafts comes from a public page, checked on 6 Oct 2026 (links below). Re-check before sending.
- **No personal contact data is stored here.** Get addresses from the linked profile pages at send time.

## Before you send
- **It's Rasero, not Rosero.** Javier Rasero is Assistant Professor of Data Science. No Javier Rosero is at UVA (the only UVA Rosero is in Astronomy).
- **Be ambitious in the mission and the ask, never in the numbers.**
  - These are rigour-minded scientists. Van Horn has said publicly that classifiers can reveal more about the algorithm than about the brain ([UVA Data Points, 2023](https://uvadatapoints.castos.com/episodes/brain-science-is-data-science)).
  - One honest, surprising result beats an accuracy claim. So the drafts lead with E1's cross-hardware finding and the cue artifact we caught ourselves, and quote no accuracy.
- **Van Horn and Rasero don't run EEG recording labs, and neither has motor-imagery data.** Asking what EEG systems they own would show we hadn't read their pages. Their drafts ask about their data work and ask them for introductions to the labs that do record. Liu and Puglia do record EEG with people, so the user-testing questions go to them.
- **Data access goes through agreements, not favours.**
  - De-identified EEG leaves UVA only under a data use agreement.
  - Testing with volunteers needs IRB review.
  - So the drafts ask how they would approach this, rather than asking them for data.
- **One clear ask:** a 20–30 minute call. Offer two or three time windows. Follow up once, after about 5 working days.
- **Van Horn and Rasero work together** (2025–26 autism papers; the Oct 2024 brain and data science symposium), so you can offer a joint call.
- **Claims check** (against the [claims audit](../claims-audit.md)): every claim below is a development result from [E1](../results/2026-10-06-e1-pooled-single-lodo-cue-only.md) and the [E0 cue diagnostics](../results/2026-10-03-e0-eegnet-bnci2014-001.md). "Within a point" means the leave-one-dataset-out mean over 339 people (−0.1 points, one seed).

## 1. John Darrell (Jack) Van Horn
Quantitative Foundation Distinguished Professor of Data Science and Professor of Psychology; Editor-in-Chief of *Neuroinformatics*. [SDS profile](https://datascience.virginia.edu/people/john-van-horn) · [lab](https://datascience.virginia.edu/labs/brain-and-data-science) · [jackvanhorn.org](https://www.jackvanhorn.org/)

**Why him:**
- He ran the data coordinating centre for the multi-site ACE GENDAAR autism network: 128-channel EEG and MRI from four sites, archived in NDAR #2021 ([Neuhaus et al. 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8439051/)).
- He writes on the economics of data sharing (2023 editorial; 2024 talk "Pain and Pleasure in Human Neuroimaging Data Sharing").
- He has led industry collaborations before (an NIH STTR with Kitware).
- He can open doors: UVA neuroscience compute, the BDSIL training network, and EEG labs.

**Subject:** UVA student building calibration-free EEG decoding: 20 minutes on pooling and validation?

> Dear Professor Van Horn,
>
> I'm Azhar Peera, a second-year Systems Engineering and Mathematics student at UVA. I'm building NeurOS: software that lets any EEG headset decode what its wearer intends from the moment it's put on, with no calibration session. Today that means left- versus right-hand motor imagery. We're applying to Y Combinator's Winter 2027 batch, and the goal is for this to be the layer every BCI app and headset runs on.
>
> The hardest parts are your field: pooling data across sites, and validation that holds up. Our first large experiment pooled seven public EEG datasets, about 340 people from different labs and amplifiers. A model that never saw a dataset decoded it to within a point of one trained on it: the hardware wasn't the bottleneck, the people were. We also found that our own early 75% headline was the brain's response to the on-screen arrow, not imagery, so every result now carries a cue-only control. Your point that a classifier can reveal more about the algorithm than about the brain is one we learned the hard way.
>
> Could I have 20–30 minutes of your time? I'd most like to ask:
> - what mattered most when your team coordinated the ACE GENDAAR EEG and MRI across sites, and what you would watch for when pooling public EEG from different headsets;
> - what a startup training on public data owes the community: licences, data-use certifications, releasing benchmarks or code;
> - who at UVA records EEG with volunteers, and how you would set up a first test of calibration-free decoding with people.
>
> I'm free [two or three time windows]. I'm happy to come to the School of Data Science, or to meet together with Professor Rasero if a joint conversation is easier.
>
> Thank you,
> Azhar Peera
> Systems Engineering and Mathematics, Class of 2028

## 2. Javier Rasero
Assistant Professor of Data Science; Brain, Body & Networks Lab (EEG, stereo-EEG and MEG among its methods). [SDS profile](https://datascience.virginia.edu/people/javier-rasero) · [lab](https://datascience.virginia.edu/labs/brain-body-networks-lab) · [GitHub](https://github.com/jrasero)

**Why him:**
- He's the closest technical peer we found at UVA. His June 2026 preprint analyses 1,404 public clinical EEG recordings (TUH): aperiodic features separate the groups best and stay most stable over time, and medication shifts many EEG measures ([medRxiv](https://doi.org/10.64898/2026.06.02.26354693)).
- He argues for cleaning biases and harmonising cohorts to reach large samples ([SDS news, Oct 2024](https://datascience.virginia.edu/news/uva-symposium-highlights-how-data-science-can-unlock-mysteries-brain)).
- He publishes null results and practises open science.
- His lab takes students.

**Subject:** Pooling public EEG across labs and headsets: would you have 20 minutes?

> Dear Professor Rasero,
>
> I'm Azhar Peera, a second-year Systems Engineering and Mathematics student at UVA. I'm building NeurOS, software that decodes intent from any EEG headset with no per-user calibration. The bet is that a model which has learned from enough people on enough hardware lets a new user put on a headset and go. We're applying to Y Combinator's Winter 2027 batch.
>
> Your June 2026 preprint on 1,404 TUH recordings is close to the problem I'm working on: which EEG features stay stable across people, sessions and recording setups, and which drift with confounds. We pooled seven public motor-imagery datasets, about 340 people. After per-session alignment, a model that never saw a dataset's hardware did as well on it as one trained on it. The hard limit now is between people, so trait-like structure is exactly what we need.
>
> Could I have 20–30 minutes of your time? I'd most like to ask:
> - how you handled TUH's mix of montages and sampling rates, and which harmonisation methods you trust for site and device effects;
> - whether aperiodic or excitability measures could normalise people before decoding, and which would survive on low-density consumer headsets;
> - whether you or your EEG, sEEG or MEG collaborators have recordings that include motor tasks, even executed movements or rest blocks.
>
> I'd also love to know whether an undergraduate collaboration on an open, cross-dataset EEG benchmark would interest you. I'm free [two or three time windows], and Professor Van Horn may join if a joint conversation suits you both.
>
> Best regards,
> Azhar Peera
> Systems Engineering and Mathematics, Class of 2028

## 3. Jeff Changchia Liu, PhD
Research Associate Professor of Neurosurgery; Human Brain Function Lab (scalp EEG with healthy volunteers, intracranial EEG, machine learning). [Lab page](https://www.humanbrainlab.net/people/jeff-c-liu)

**Why him:**
- His lab runs scalp-EEG studies with healthy volunteers (laser-evoked potentials, 2025–26).
- His 2026 *Brain Communications* paper separates stable cohort-level patterns from day-to-day fluctuation in multiday intracranial EEG ([PubMed 42582622](https://pubmed.ncbi.nlm.nih.gov/42582622/)).
- He's co-PI on an NIH-funded insula DBS trial.
- He's the strongest technical and user-testing contact in the medical school.

**Subject:** Calibration-free EEG decoding: could a short motor-imagery block fit your volunteer sessions?

> Dear Dr. Liu,
>
> I'm Azhar Peera, a second-year Systems Engineering and Mathematics student at UVA. I'm building NeurOS, software that decodes intent from any EEG headset with no per-user calibration: the user puts it on and it works. We're applying to Y Combinator's Winter 2027 batch.
>
> Your 2026 Brain Communications paper caught my attention. Separating stable cohort-level patterns from day-to-day fluctuation is the problem we face across people. In our first large experiment (seven public datasets, about 340 people), a model that never saw a dataset decoded it to within a point of one trained on it. The remaining gap is between people, not machines, and closing it needs recordings from real volunteers.
>
> Your lab records scalp EEG from healthy volunteers, which is where we most need guidance. Could I have 20 minutes to ask:
> - whether a 10-minute motor-imagery block could ever be added to a healthy-volunteer protocol, and what the IRB route would be;
> - what a data use agreement for de-identified scalp EEG looks like at UVA;
> - what you would measure in a first usability test of calibration-free decoding?
>
> I'm free [two or three time windows] and happy to come to you.
>
> Thank you,
> Azhar Peera
> Systems Engineering and Mathematics, Class of 2028

## 4. Meghan Puglia, PhD
Associate Professor of Neurology; Director of Community Engagement at the UVA Brain Institute; Developmental Neuroanalytics Lab (infant EEG). [Brain Institute profile](https://braininstitute.virginia.edu/news/meet-meghan-puglia-uva-brain-institute-director-community-engagement)

**Why her:**
- Infant EEG has to work without per-person calibration.
- Her 2026 NICU paper uses brain-signal entropy, with clinical measures, to predict later autism risk ([PubMed 41576766](https://pubmed.ncbi.nlm.nih.gov/41576766/)).
- As the Brain Institute's Director of Community Engagement, she's the right person to ask about recruiting volunteers properly.
- She has co-authored with Systems & Information Engineering (Gerling) ([PubMed 39181012](https://pubmed.ncbi.nlm.nih.gov/39181012/)), the founder's own department.

**Subject:** A calibration-free EEG question from a Systems Engineering student

> Dear Dr. Puglia,
>
> I'm Azhar Peera, a second-year Systems Engineering and Mathematics student at UVA. I'm building NeurOS, software that decodes intent from any EEG headset with no calibration session, so a new user can put it on and go. We're applying to Y Combinator's Winter 2027 batch.
>
> Your field lives with the constraint we've built the product around: babies can't be calibrated. Your 2026 NICU paper, which finds individual signal in brain-signal entropy with no calibration step, is the kind of result we're chasing. I also noticed your work with Professor Gerling in my own department.
>
> Could I have 20 minutes to ask:
> - how you handle artifacts and differences between people when no calibration is possible;
> - as the Brain Institute's Director of Community Engagement, how a student team should recruit volunteers for headset usability tests the right way (IRB, community channels);
> - whether the Brain Institute supports undergraduate neurotech projects?
>
> I'm free [two or three time windows].
>
> Thank you,
> Azhar Peera
> Systems Engineering and Mathematics, Class of 2028

## 5. Ye (Sarah) Sun — send first
Associate Professor of Mechanical & Aerospace Engineering (joint in Electrical & Computer Engineering); WEARLab. [Profile](https://engineering.virginia.edu/faculty/ye-sarah-sun) · [WEARLab](https://engineering.virginia.edu/labs-groups/wearlab)

**Why her:**
- Hers is the only motor-imagery EEG lab at UVA. NeuroFlex (*Sensors* 2025) decodes imagined fist closure on a **16-channel dry OpenBCI Cyton + Daisy cap** to drive a soft rehabilitation glove ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11820135/)).
- **She's also the most useful sceptic.** NeuroFlex trained a model per person and concludes that "individually trained models are needed", the opposite of our bet.
- A fair head-to-head on her rig, a dry consumer-style headset we don't control, would be stronger evidence than anything we run alone.
- She's the same person as the "Sarah Sun" already #1 in [target-contacts.md](target-contacts.md).
- **Don't allege leakage.** NeuroFlex doesn't describe its train/test split, so ask about it, don't imply anything.

**Subject:** Per-user vs pooled motor-imagery models: a head-to-head on your OpenBCI rig?

> Dear Professor Sun,
>
> I'm Azhar Peera, a second-year Systems Engineering and Mathematics student at UVA. I'm building NeurOS, a startup applying to Y Combinator's Winter 2027 batch. It's software that decodes motor imagery from any EEG headset with no per-user calibration: one pooled model, trained on many people and many headsets, that works for someone it has never seen.
>
> Your NeuroFlex paper argues the opposite: that people imagine the same movement differently, so each user needs their own model. That is exactly the question I want to settle honestly. In our first large experiment (seven public datasets, about 340 people), a model that never saw a dataset's lab or amplifier decoded it to within a point of one trained on it. Every one of those datasets, though, was recorded on research-grade wet EEG, never on a dry 16-channel OpenBCI cap like yours.
>
> Could I have 20–30 minutes? I'd like to propose a fair head-to-head with people held out: your per-subject transformer against our zero-calibration model, on your hardware. I'd also like to learn:
> - what dry-electrode artifacts and motion problems you fight on the NeuroFlex rig;
> - how you split the overlapping epochs between training and testing;
> - what it would take, under your IRB protocol and a data use agreement, to run or share that comparison.
>
> I'm free [two or three time windows] and happy to come to the lab.
>
> Thank you,
> Azhar Peera
> Systems Engineering and Mathematics, Class of 2028

## 6. Per B. Sederberg
Professor of Psychology; Director of the Cognitive Science Program; Computational Memory Lab. [Profile](https://psychology.as.virginia.edu/people/sederberg) · [compmem.org](https://compmem.org/)

**Why him:** he co-authored Schwemmer et al., *Nature Medicine* 2018 ([PubMed](https://pubmed.ncbi.nlm.nih.gov/30250141/)). That intracortical decoder kept its accuracy for over a year without daily retraining, through unsupervised updating, and learned new functions through transfer learning. It's the closest UVA precedent for "no calibration", and the same idea as our E2/E3, which adapt to a new user from unlabelled data. He also co-authored Bouton et al., *Nature* 2016, and has worked with Battelle.

**Subject:** Unsupervised updating for a calibration-free EEG decoder: 20 minutes?

> Dear Professor Sederberg,
>
> I'm Azhar Peera, a second-year Systems Engineering and Mathematics student at UVA. I'm building NeurOS, a startup applying to Y Combinator's Winter 2027 batch: software that decodes intent from any EEG headset with no calibration session.
>
> Your 2018 Nature Medicine decoder, which held its accuracy for over a year by updating itself without supervision, is the closest thing I know to what we're attempting. We do it across people rather than days, on scalp EEG. In our first large experiment (seven public datasets, about 340 people), a model that never saw a dataset decoded it as well as one trained on it, but the gap between people remains. Our next experiments adapt the model to a new person from their first minutes of unlabelled EEG.
>
> Could I have 20–30 minutes to ask:
> - which parts of your unsupervised-updating approach you'd expect to carry over to a new user, as opposed to a new day;
> - whether you'd critique our evaluation design (people, datasets and devices held out);
> - what you learned from building a BCI with an industry partner?
>
> I'm free [two or three time windows].
>
> Thank you,
> Azhar Peera
> Systems Engineering and Mathematics, Class of 2028

## 7. Nicole Long
Long Term Memory Lab, Psychology. [Psychology labs page](https://psychology.as.virginia.edu/research-opportunities-working-lab)

**Why her:**
- Her 2026 *iScience* paper validated decoding of memory brain states across participants and across studies, on a **64-channel BrainVision actiCAP** at 1,000 Hz under protocol SBS-2115 ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13543892/)).
- She holds NINDS R01NS132872 (2023–2028) on multivariate decoding.
- Her research-grade rig would be the wet reference against which a dry headset gets judged.
- **Her OSF datasets are CC BY-NC (non-commercial).** Don't ask to train on them; a collaboration would be the route.

**Subject:** Cross-participant EEG decoding: a student startup's question about your 64-channel rig

> Dear Dr. Long,
>
> I'm Azhar Peera, a second-year Systems Engineering and Mathematics student at UVA. I'm building NeurOS, a startup applying to Y Combinator's Winter 2027 batch: software that decodes intent from any EEG headset with no per-person calibration.
>
> Your 2026 iScience paper, which decodes memory brain states across participants and across studies, is the strongest evidence I've found at UVA that cross-participant decoding can work. That's the premise of our company. In our first large experiment (seven public datasets, about 340 people), a model that never saw a dataset decoded it as well as one trained on it. What we can't yet show is the same person on a research-grade cap and on a consumer headset.
>
> Could I have 20 minutes to ask:
> - what made your classifier hold up across people and studies;
> - whether one or two supervised sessions recording a short motor-imagery block on your actiCAP and on a dry headset side by side could ever fit under your protocol or an amendment;
> - how you'd recommend a student team recruit volunteers properly?
>
> I'm free [two or three time windows].
>
> Thank you,
> Azhar Peera
> Systems Engineering and Mathematics, Class of 2028

## 8. Tanya Evans
Co-Director of the UVA Brain Institute (since Dec 2025); School of Education and Human Development. [Brain Institute news](https://braininstitute.virginia.edu/news/meet-tanya-evans-uva-brain-institute-co-director)

**Why her:**
- Her group records with **dry wireless Cognionics Quick-20m headsets** (19 channels), a second consumer-style hardware class ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10518595/)).
- As co-director, she's the route to Brain Institute introductions and to partners for faculty pilot grants. She has said she wants to lower barriers to collaboration across schools.

**Subject:** A student neurotech startup and the Brain Institute: could I ask for 20 minutes?

> Dear Professor Evans,
>
> I'm Azhar Peera, a second-year Systems Engineering and Mathematics student at UVA. I'm building NeurOS, a startup applying to Y Combinator's Winter 2027 batch: software that lets any EEG headset decode its wearer's intent with no calibration session. In our first large experiment, across seven public datasets and about 340 people, a model that never saw a dataset's lab or hardware decoded it as well as one trained on it.
>
> You've said you want the Brain Institute to lower the barriers to working across schools, and this project sits right across them: systems engineering, data science, psychology and neurology. I'd love 20 minutes to ask:
> - which Brain Institute faculty you'd connect a student neurotech project with, and how the Institute supports student-led work;
> - whether your dry wireless Quick-20m headsets could ever host a supervised session comparing them with research-grade caps;
> - how you'd approach representative recruitment for headset testing, given your work on representative samples.
>
> I'm free [two or three time windows].
>
> Thank you,
> Azhar Peera
> Systems Engineering and Mathematics, Class of 2028

## UVA rules that shape every ask (from the [research report](uva-contacts-research.md); not legal advice)
- **UVA owns lab recordings** (policy RES-002). Faculty can't share EEG informally. Sharing takes IRB approval plus a data use agreement signed by the Office of Sponsored Programs, so never ask for raw data in a first email.
- **EEG testing with people probably needs a faculty PI under IRB-HSR**, where students can't be PI. Purely internal product testing is a grey area: get a written determination with the IRB's "Evaluate My Project" tool.
- **Keep NeurOS's code and data outside any lab.** Building it with lab equipment, lab data or a paid lab role risks a UVA IP claim (RES-006 for patents, RES-001 for copyright, including software). Ask for supervised sessions, not loaned equipment (PRM-011).
- **Say up front that NeurOS is a company.** Faculty equity or advising may need conflict-of-interest disclosure (RES-005).
- **Free help:** the Foundry (walk-in), the Law School's Entrepreneurial Law Clinic, Engineering's VentureForward grants (up to $1K), I-Corps (up to $2,700 for customer discovery).
- **⏰ E-Cup Concept stage: due 23 Oct at noon** (final 13 Nov), with an Engineering & Healthcare Technologies track and $15K+ in perks for applicants. The year is inferred from the page; confirm at the Foundry.
- **neurotech@uva** (founded 2026) builds EEG headsets and BCI apps: the nearest student community for collaborators and testers. Testing on members can still need IRB review.

## Sending order and timing
1. **This week:** Sun, Rasero and Van Horn (or one joint note to Van Horn and Rasero), and Liu.
2. **Next:** Long, Evans, Sederberg and Puglia.
3. **Later:**
   - **Quigg**, for the clinical EEG archive. That's a 2027 horizon, not a YC-timeframe ask.
   - **Carlson**, for stroke work and a precedent for partnering with a device startup.
   - **Riggs and Brown**, both in SIE. Reach them through Gregory Gerling, the interim SIE chair.
4. UVA advises following up after at least ten business days. For mid-October emails that lands around 26–28 Oct, so send the first wave this week if replies are to inform the YC application.
