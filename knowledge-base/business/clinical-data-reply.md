# Clinical data thread: reply draft to David Chen (v1, 12 Oct 2026)

**Status:** a draft for Azhar to adapt and send. Claude drafts only; the founders send. No contact details are stored here.

## The thread so far
- **Azhar asked** David for introductions to neurosurgeons or neurologists open to a data-sharing or research agreement (requirements, costs, de-identified EEG), and whether anyone had synthetic EEG. Azhar mentioned the accelerator deadline at the end of October.
- **David replied** that he can ask around. First he asked how the data we need differs from public sources:
  - the [public EEG data list](https://sccn.ucsd.edu/~arno/fam2data/publicly_available_EEG_data.html) (NEMAR);
  - an [NIH Common Data Elements](https://cde.nlm.nih.gov/deView?tinyId=DN2pHb7FSxc) page (it wouldn't load for us);
  - the [BNCI Horizon](https://bnci-horizon-2020.eu/database/data-sets) datasets.

## Before you send
- **His question is the right one, so answer it directly.** He's offering to spend his own goodwill on introductions. A specific, small ask makes that easy for him.
- **We already use exactly this public data.** Two of our datasets come from the BNCI Horizon page he linked: BNCI 2014-001, used in training, and BNCI 2014-004, the never-seen replay test. Several are also mirrored on NEMAR. Say so, so he sees we've done the homework.
- **What public data really lacks, that only clinicians can provide:** people who need a BCI.
  - Public motor-imagery data comes almost entirely from healthy volunteers. The largest patient set we've found is Liu2024 (50 acute stroke patients, one session each).
  - The other gap, recordings on low-cost headsets outside the lab, we can collect ourselves. Don't ask clinicians for that.
- **Correct the urgency.** The YC application and demo don't depend on clinical data; they use public data. Saying so keeps anyone from rushing for a deadline that doesn't need them.
- **Drop the synthetic-data ask.**
  - Synthetic EEG is modelled on existing recordings, so it can't add the new people we're short of.
  - For prototyping, we already replay real public recordings through the engine.
- **UVA rules** (from our [research report](uva-contacts-research.md); not legal advice):
  - De-identified EEG leaves UVA only under a data use agreement.
  - Testing with volunteers needs IRB review, with a faculty PI.
  - So ask how it would work, not for data.
- **Numbers:** "about 340 people, seven public datasets" is the same claim the faculty drafts use ([claims audit](../claims-audit.md)). Add no accuracy numbers.
- **The name:** a registered NEUROS mark covers neurostimulation medical devices ([Q32](../open-questions.md)). Pitching patient uses raises that risk, so this thread is one more reason to get the attorney's opinion before the name spreads.

## The draft

**Subject:** Re: [keep the existing subject]

> Hi David,
>
> Thank you, that's exactly the right question, and the links help. We already build on that public data. Our model is trained on about 340 people from seven public motor-imagery datasets and tested on people it has never seen. Two of the datasets we use come from the BNCI Horizon page you linked.
>
> What public data doesn't have is the people a BCI is for. Nearly all public motor-imagery EEG comes from healthy volunteers in a single lab session. People with paralysis from ALS, spinal cord injury or stroke produce different signals when they try to move. The largest public patient set we've found has 50 stroke patients, each recorded once. That's the gap only clinicians can help with. (The other gap, recordings on low-cost headsets outside the lab, we can collect ourselves.)
>
> So the most useful introduction would be a neurologist, neurosurgeon or rehabilitation physician who works with patients with paralysis and might have 20–30 minutes. We'd like to learn:
> - whether a no-calibration, non-invasive BCI would be useful to their patients;
> - whether de-identified EEG of patients attempting movement already exists at UVA;
> - what an IRB-approved study and a data use agreement would involve: steps, timeline and costs.
>
> Two corrections to my last note. Our accelerator application doesn't depend on this; it's for our next stage, so there's no rush on anyone's side. And we can drop the synthetic-data question: synthetic EEG is modelled on existing recordings, so it can't add the new people we're missing.
>
> Thanks again for your help.
>
> Best,
> Azhar
