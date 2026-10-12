# Data thread: reply draft to David Chen (v2, 12 Oct 2026)

**Status:** a draft for Azhar to adapt and send. Claude drafts only; the founders send. No contact details are stored here.

**v2 (founder's direction):**
- Low-cost headset recordings are a real ask, not something to collect alone.
- The timing is real: early conversations would strengthen the accelerator application.

## The thread so far
- **Azhar asked** David for introductions to neurosurgeons or neurologists open to a data-sharing or research agreement (requirements, costs, de-identified EEG), and whether anyone had synthetic EEG. Azhar mentioned the accelerator deadline at the end of October.
- **David replied** that he can ask around. First he asked how the data we need differs from public sources:
  - the [public EEG data list](https://sccn.ucsd.edu/~arno/fam2data/publicly_available_EEG_data.html) (NEMAR);
  - an [NIH Common Data Elements](https://cde.nlm.nih.gov/deView?tinyId=DN2pHb7FSxc) page (it wouldn't load for us);
  - the [BNCI Horizon](https://bnci-horizon-2020.eu/database/data-sets) datasets.

## Before you send
- **Answer his question directly.** He's offering to spend his own goodwill on introductions. Two named gaps and two named kinds of people make that easy.
- **We already use exactly this public data.** Two of our datasets come from the BNCI Horizon page he linked: BNCI 2014-001 (training) and BNCI 2014-004 (the never-seen replay test). Say so; it shows the homework.
- **The two gaps are real** (from our [consumer-headset data review](../research/consumer-headset-mi-datasets.md)):
  - **Patients.** Public motor-imagery EEG is almost all healthy volunteers. The largest patient set we found is Liu2024: 50 acute stroke patients, one session each.
  - **Low-cost headsets.** A 2023 review found all 25 public motor datasets recorded on research-grade systems. A dozen or so consumer sets have appeared since. **None has the same people imagining movements on both a research cap and a low-cost headset.** That pairing is the cleanest way to measure what cheaper hardware costs.
- **At UVA**, Professor Sun's lab (16-channel dry OpenBCI) is the obvious first source of low-cost recordings, and you've already contacted her. neurotech@uva builds EEG headsets too.
- **Drop the synthetic-data ask.** Synthetic EEG is modelled on existing recordings, so it can't add new people or new headsets.
- **In the application, describe only what has happened.** "In conversation with UVA neurologists about a data agreement" is fine once a conversation has happened. Don't call it a partnership before anything is signed.
- **UVA rules** (from our [research report](uva-contacts-research.md); not legal advice):
  - De-identified EEG leaves UVA only under a data use agreement.
  - Recording volunteers needs IRB review with a faculty PI.
  - So the draft asks how it would work, not for data.
- **Numbers:** "about 340 people, seven public datasets" is the same claim the faculty drafts use ([claims audit](../claims-audit.md)). No accuracy numbers.
- **The name:** a registered NEUROS mark covers neurostimulation medical devices ([Q32](../open-questions.md)). Pitching patient uses raises that risk: one more reason to get the attorney's opinion before the name spreads.

## The draft

**Subject:** Re: [keep the existing subject]

> Hi David,
>
> Thank you, that's exactly the right question, and the links help. We already build on that public data. Our model is trained on about 340 people from seven public motor-imagery datasets and tested on people it has never seen. Two of the datasets we use come from the BNCI Horizon page you linked.
>
> What public data doesn't have falls into two gaps:
> - **The people a BCI is for.** Nearly all public motor-imagery EEG comes from healthy volunteers in a single lab session. People with paralysis from ALS, spinal cord injury or stroke produce different signals when they try to move. The largest public patient set we've found has 50 stroke patients, each recorded once.
> - **The headsets people will actually wear.** Public recordings are almost all made on research-grade gel caps; very few use low-cost or consumer headsets. We found none where the same people imagined movements on both a research cap and a low-cost headset, which is the cleanest way to measure what cheaper hardware costs in accuracy.
>
> So the most useful introductions would be:
> - a neurologist, neurosurgeon or rehabilitation physician who works with patients with paralysis;
> - anyone at UVA beyond Professor Sun's lab who records EEG on low-cost headsets (OpenBCI, Emotiv, Muse and similar), especially alongside a research-grade system.
>
> With either, we'd like to learn:
> - whether existing de-identified recordings could be shared under a data use agreement;
> - what an IRB-approved study would involve: steps, timeline and costs;
> - for clinicians, whether a no-calibration, non-invasive BCI would be useful to their patients.
>
> We're applying to an accelerator at the end of October, so even an early conversation in the next two weeks would help us a great deal. And we can drop the synthetic-data question: synthetic EEG is modelled on existing recordings, so it can't add the new people and headsets we're missing.
>
> Thanks again for your help.
>
> Best,
> Azhar
