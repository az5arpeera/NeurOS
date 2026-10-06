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

## Next contacts (drafts after the remaining research)
- **Ye Sun (Engineering): NeuroFlex**, a soft rehabilitation glove driven by motor-imagery EEG ([PubMed 39943246](https://pubmed.ncbi.nlm.nih.gov/39943246/)). This is the only motor-imagery EEG decoding found at UVA, and probably WEARLab, already #1 in [target-contacts.md](target-contacts.md); confirm the identity. They would be the closest technical match for user testing.
- **Mark Quigg, MD (Neurology):** used about 800 routine EEGs from UVA's clinical archive in a 2026 study, and co-leads NIH clinical-trial methods training. Ask about data and pilot design.
- **Andrew Carlson, MD (Neurosurgery, stroke):** co-published in 2025 with a small EEG-device company, a precedent for clinician–startup partnerships.
- **Introductions:** Jaideep Kapur and Tanya Evans, co-directors of the UVA Brain Institute (announced Dec 2025).
- Psychology, Engineering and UVA-resources research (IRB and data-use-agreement routes, the Licensing & Ventures Group, student programmes) is being re-run: the first pass was cut off by a usage limit on 6 Oct.
