# Ten UVA Faculty Can Stress-Test Zero-Calibration EEG

UVA has no brain-computer-interface program. Only one lab on Grounds decodes motor imagery from scalp EEG: Ye (Sarah) Sun's WEARLab in Engineering. Its NeuroFlex glove ran on a 16-channel dry OpenBCI cap, and its authors concluded that each user needs an individually trained model. That is exactly the assumption NeurOS sets out to break.

The two named contacts both make the shortlist, for different reasons:

- **Javier Rasero** is almost certainly the "Javier Rosero" you asked about. He is an Assistant Professor of Data Science and NeurOS's closest methods peer at UVA. His 2026 work pools 1,404 public clinical EEG recordings and finds that aperiodic features are the most stable, trait-like markers.
- **John Darrell Van Horn** is a senior neuroinformatics figure. He ran the data coordinating center for a multi-site autism network that collected 128-channel EEG, and he holds a roughly $2M NIH compute instrument. His value is data-governance credibility, rigor in validation, and introductions. He is not a BCI researcher.

The rest of the top ten:

- **Per Sederberg**, co-author of a 2018 *Nature Medicine* decoder that stayed accurate for more than a year without daily retraining.
- **Nicole Long**, who records 64-channel EEG and has published cross-participant decoding.
- **Tanya Evans**, Brain Institute co-director, whose lab owns dry wireless headsets.
- **Sara Riggs**, who does EEG decoding inside the founder's own department.
- **Donald Brown**, who uses data augmentation to make EEG models generalize across subjects.
- **Jeff Liu** and **Mark Quigg**, the clinical EEG leads.

Together these labs run four classes of EEG hardware: dry 16-channel, dry wireless 19-channel, wet 64-channel and high-density 128-channel. That is enough to back a hardware-agnostic claim on a single campus.

The constraints are firm:

- UVA owns lab recordings.
- Any data transfer needs IRB approval plus an agreement signed by the Office of Sponsored Programs (OSP).
- EEG user testing with a UVA partner probably needs a faculty member as principal investigator (PI).
- Building NeurOS inside a lab can give UVA an IP claim.

The near-term moves are:

- Apply to the E-Cup Concept stage (due October 23 at noon), ahead of the YC W27 deadline on November 1.
- Send outreach emails that ask for validation advice and small supervised sessions on faculty hardware, not for raw data.

## One motor-imagery lab, four hardware classes: where NeurOS fits at UVA

The underlying research covered:

- a keyword scan of all 353 Engineering (SEAS) faculty profiles;
- OpenAlex and PubMed queries of UVA-affiliated EEG and BCI work from 2022 to 2026;
- NIH RePORTER and NSF award searches, and ClinicalTrials.gov;
- lab, department and policy pages.

The headline finding is what is missing. RePORTER returned 81 UVA projects for fiscal 2023-2026 matching EEG, BCI, stereo-EEG, deep brain stimulation, TMS, neurofeedback, focused ultrasound or motor imagery. **None was a BCI or motor-imagery project.** The hits were dominated by focused ultrasound, epilepsy and pain ([RePORTER search](https://reporter.nih.gov/search/WFbv9_4Wf0ST5W5ms-RqpA/projects)).

A PubMed query for UVA-affiliated papers on Muse, Emotiv, OpenBCI, dry electrodes, wearable EEG, BCI, motor imagery or neurofeedback returned seven records. The only scalp-EEG BCI among them was Sun's NeuroFlex glove ([PubMed query](https://pubmed.ncbi.nlm.nih.gov/?term=%28Muse%5Btiab%5D+OR+Emotiv%5Btiab%5D+OR+OpenBCI%5Btiab%5D+OR+%22dry+electrode%2A%22%5Btiab%5D+OR+%22consumer-grade%22%5Btiab%5D+OR+%22wearable+EEG%22%5Btiab%5D+OR+%22mobile+EEG%22%5Btiab%5D+OR+%22portable+EEG%22%5Btiab%5D+OR+%22brain-computer+interface%22%5Btiab%5D+OR+%22motor+imagery%22%5Btiab%5D+OR+neurofeedback%5Btiab%5D%29+AND+%28%22University+of+Virginia%22%5Bad%5D%29+AND+2018%3A2026%5Bdp%5D)).

In Engineering, only two groups decode EEG at all: Sun (motor imagery) and Riggs (event-related potential, or ERP, decoding of touch) ([OpenAlex](https://api.openalex.org/works?filter=institutions.id:I51556381)). Outside the School of Medicine, human EEG sits in a handful of Psychology and Education labs, each with its own system. The only EEG core facility at UVA is a provisional one for animal EEG ([Animal EEG core](https://med.virginia.edu/electrophysiology-core/about/)).

That shapes the pitch. NeurOS would be bringing a new paradigm to UVA, not tapping an existing store of motor-imagery data. The strongest ask is therefore "help me validate on your hardware and your population", not "share your BCI data".

It also points to UVA's real asset, which is hardware diversity:

| Lab | EEG hardware | Source |
|---|---|---|
| Sun | 16-electrode dry OpenBCI cap | [NeuroFlex](https://pmc.ncbi.nlm.nih.gov/articles/PMC11820135/) |
| Evans | dry wireless 19-channel Cognionics headsets | [Evans et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC10518595/) |
| Long | 64-channel active BrainVision actiCAP | [Long et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC13543892/) |
| Autism network (coordinated by Van Horn, led locally by Pelphrey) | 128-sensor nets | [UVA Today](https://news.virginia.edu/content/uva-researchers-find-brain-signal-linked-communication-challenges-autism) |

These four classes are exactly what a hardware-agnostic claim must hold up across.

**Two identity questions are settled.**

- **No Javier Rosero holds a UVA appointment.** A search of virginia.edu found only Viviana Rosero, a research associate in Astronomy ([UVA Astronomy](https://astronomy.as.virginia.edu/people/profile/vr8gx)). The other Javier Roseros online are at the University of Florida and at an engineering firm ([Google Scholar](https://scholar.google.com/citations?user=O2JniyUAAAAJ&hl=en); [Stantec](https://www.stantec.com/en/people/r/rosero-javier)).
- **Javier Rasero is the intended person.** He shares the first name, works in neuroscience and data science, and co-authors with Van Horn ([Cerebral Cortex 2026](https://doi.org/10.1093/cercor/bhag156)). Use "Rasero" in any email.
- **"Ye Sun" and "Sarah Sun" are one person.** The Ye Sun of NeuroFlex and the Sarah Sun of WEARLab are the same; UVA lists her as Ye (Sarah) Sun ([UVA profile](https://engineering.virginia.edu/faculty/ye-sarah-sun); [WEARLab](https://engineering.virginia.edu/labs-groups/wearlab)).

The ranking below weighs four things:

1. direct technical overlap with decoding EEG across subjects and devices;
2. access to hardware, participants or shareable data;
3. leverage for introductions and institutional support;
4. likely bandwidth for a student collaborator.

| Rank | Person | Title, department | Lab | What they can offer NeurOS | Public contact page |
|---|---|---|---|---|---|
| 1 | Ye (Sarah) Sun | Associate Professor, Mechanical & Aerospace Engineering; joint Associate Professor, Electrical & Computer Engineering | WEARLab | The only motor-imagery BCI lab; a dry 16-channel OpenBCI rig; a partner for a head-to-head benchmark | [Profile](https://engineering.virginia.edu/faculty/ye-sarah-sun) · [WEARLab](https://engineering.virginia.edu/labs-groups/wearlab) |
| 2 | Javier Rasero ("Rosero") | Assistant Professor of Data Science | Brain, Body & Networks Lab | Pooled public clinical EEG, stable aperiodic features, cohort harmonization | [SDS profile](https://datascience.virginia.edu/people/javier-rasero) · [Lab](https://datascience.virginia.edu/labs/brain-body-networks-lab) |
| 3 | John Darrell Van Horn | Quantitative Foundation Distinguished Professor of Data Science; Professor of Psychology | Brain and Data Science | Multi-site EEG data governance, rigor in validation, high-performance computing (HPC), introductions | [SDS profile](https://datascience.virginia.edu/people/john-van-horn) · [jackvanhorn.org](https://www.jackvanhorn.org/) |
| 4 | Per B. Sederberg | Professor of Psychology; Director, Cognitive Science Program | Computational Memory Lab | BCI experience with decoders that stay stable and transfer to new tasks; multivariate pattern analysis (MVPA) tools | [Profile](https://psychology.as.virginia.edu/people/sederberg) · [compmem.org](https://compmem.org/) |
| 5 | Nicole Long | Psychology (rank not captured in notes) | Long Term Memory Lab | A 64-channel wet EEG system to serve as a reference; cross-participant decoding; an approved IRB-SBS EEG protocol | [Psychology labs page](https://psychology.as.virginia.edu/research-opportunities-working-lab) |
| 6 | Tanya Evans | Co-Director, UVA Brain Institute; School of Education and Human Development | Evans lab (SEHD) | Dry wireless 19-channel headsets; Brain Institute introductions | [Brain Institute](https://braininstitute.virginia.edu/news/meet-tanya-evans-uva-brain-institute-co-director) |
| 7 | Sara Lu Riggs | Associate Professor, Systems & Information Engineering | No public lab site | EEG/ERP decoding in the founder's department; workload paradigms; an undergraduate research route | [Profile](https://engineering.virginia.edu/faculty/sara-lu-riggs) |
| 8 | Donald E. Brown | Quantitative Foundation Distinguished Professor of Data Science; W.S. Calcott Professor of Systems & Information Engineering | None listed | Cross-subject EEG augmentation; industry experience; a capstone route in Systems & Information Engineering (SIE) | [SDS profile](https://datascience.virginia.edu/people/donald-brown) |
| 9 | Jeff Changchia Liu | Research Associate Professor of Neurosurgery | Human Brain Function Lab | Scalp EEG protocols with healthy volunteers and intracranial protocols; machine learning; a bridge to functional neurosurgery | [Lab page](https://www.humanbrainlab.net/people/jeff-c-liu) |
| 10 | Mark S. Quigg | T.R. Johns Professor in Neurology (Epilepsy) | Clinical neurophysiology / EEG labs | Gatekeeper to UVA's clinical EEG volume; advice on trial methodology | [Neurology faculty](https://med.virginia.edu/neurology/faculty/) |

## Profiles: where the hardware, data and participants actually sit

### 1. Ye (Sarah) Sun: the only motor-imagery BCI lab, and the most useful skeptic

**Role and lab.** Sun is an Associate Professor of Mechanical & Aerospace Engineering with a joint appointment in Electrical & Computer Engineering. She joined UVA in 2021 from Michigan Technological University, holds a 2018 NSF CAREER award, and was General Chair of IEEE/ACM CHASE 2022 ([UVA profile](https://engineering.virginia.edu/faculty/ye-sarah-sun)). WEARLab describes its mission as human-centered wearable electronics, soft robotics and BCI technologies for motion assistance and neurorehabilitation ([WEARLab](https://engineering.virginia.edu/labs-groups/wearlab)).

**NeuroFlex** (*Sensors* 2025) is the most relevant paper at UVA for NeurOS ([PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC11820135/)):

- **Setup:** three healthy adults wore a cap with **16 dry ThinkPulse electrodes on an OpenBCI Cyton + Daisy board**, recorded through BrainFlow under UVA IRB protocol 7070.
- **Task:** imagined fist closure vs rest, 18 trials per subject over three sessions.
- **Model:** an FFT front end feeding a transformer.
- **Results:** the best subject reached **85.3%**. The three-subject mean, computed from the paper's Table 4, is about **80.5%**, with mean AUC 0.88.
- **Training was within-subject.** The authors write that "individually trained models are needed" because people imagine the same movement differently, and they list individualized calibration as future work.

**Other EEG work:**

- A two-page CHASE 2024 paper trained a CNN-transformer on BCI Competition IV-2a (left vs right hand) and reported a **per-subject mean of 75.3 ± 7.1%** ([NSF PAR](https://par.nsf.gov/biblio/10570673)).
- A 2026 *Smart Health* paper on CNN decoding of motor-imagery EEG exists, but its abstract could not be retrieved ([record](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1016/j.smhl.2026.100647)).

**People** ([WEARLab people](https://engineering.virginia.edu/labs-groups/wearlab/people)):

- Soroush Zare, PhD candidate and NeuroFlex first author.
- Faisal Rehman, who works on low-impedance embroidered skin-contact electrodes.
- A 2026 summer intern who built a simulated real-time motor-imagery EEG control system on Arduino.

**Funding:**

- A $1M NSF Smart and Connected Health award (2025-2029) on soft exosuits, with Shawn Russell as co-PI ([NSF 2501041](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2501041)).
- The NSF award that funded NeuroFlex ([NSF 2135620](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2135620)).
- Earlier NSF work on the physics of triboelectric motion artifacts in wearables ([NSF 2219375](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2219375)).

**Why she ranks first.** She works on NeurOS's task family (left/right motor imagery; rest vs effort) and on the same benchmark dataset family. Her lab runs a realistic dry headset that NeurOS does not control, so tests there are independent. She also understands dry-electrode artifacts.

**She is also the most persuasive skeptic.** Her own paper argues that per-user models are necessary. A head-to-head result on her rig would therefore carry more weight than any number NeurOS generates alone.

**Gaps and cautions:**

- **No shareable data.** The NeuroFlex GitHub repository contains only a README and two images, with no code or EEG data ([GitHub](https://github.com/SoroushZare/NeuroFlex)).
- **Possible leakage, unconfirmed.** The paper does not describe its train/test split. With 1-second epochs overlapping by 0.75 s, an epoch-level split could leak between training and test data. Treat this as something to ask about, not a claim to make.
- **No public statements on commercialization** were found, and her August 2025 spotlight video was not transcribed.

**Contact:** her [profile](https://engineering.virginia.edu/faculty/ye-sarah-sun) and [WEARLab](https://engineering.virginia.edu/labs-groups/wearlab).

### 2. Javier Rasero: the "Rosero" on your list, and the closest methods peer

**Role and lab.** Rasero has been Assistant Professor of Data Science at the School of Data Science (SDS) since 2023 ([SDS profile](https://datascience.virginia.edu/people/javier-rasero)).

**Background** ([CV](https://api.dsi.virginia.edu/sites/default/files/2023-08/cv_RASERO.pdf)):

- PhD in physics (University of Valencia, 2014).
- C++ developer for 3D applications at a Bilbao company, 2015-2016.
- Postdocs in Bari, at Biocruces, and in Carnegie Mellon's CoAx Lab (2019-2023), where he built models predicting cardiovascular risk from neuroimaging.

**His Brain, Body & Networks Lab** ([lab page](https://datascience.virginia.edu/labs/brain-body-networks-lab)):

- treats the brain as a hub interacting with whole-body physiology;
- lists **EEG, stereo-EEG and MEG** among its modalities, alongside MRI, transcriptomics and peripheral physiology;
- takes students, but states it has no current funding.

**Most relevant work: a June 2026 medRxiv preprint** with Duma, Valencia, Bonanni and Pellegrino ([medRxiv](https://doi.org/10.64898/2026.06.02.26354693)):

- **Data:** **1,404 EEG recordings** from 96 people with epilepsy and 85 controls, all from the public **Temple University Hospital (TUH) EEG corpus**.
- **Aperiodic parameters** gave the most robust separation between patients and controls, and were the most stable, trait-like markers.
- **Medication burden** shifted several EEG metrics.

**Other EEG/MEG and methods work:**

- **MEG excitability (bioRxiv, January 2026):** whole-head MEG at rest and during 40 Hz auditory stimulation yielded six separable dimensions of cortical excitability. The analysis code is public on his GitHub ([bioRxiv](https://doi.org/10.64898/2026.01.31.703018); [GitHub](https://github.com/jrasero?tab=repositories)).
- **MEG connectivity (*NeuroImage* 2023):** how the method used to extract region-of-interest signals changes MEG connectivity estimates ([DOI](https://doi.org/10.1016/j.neuroimage.2023.120424)).
- **Autism subtypes (*Biological Psychiatry* 2023):** a stable hypoconnected and a stable hyperconnected subtype among 657 autistic individuals ([DOI](https://doi.org/10.1016/j.biopsych.2023.04.014)).
- **Autism brain-morphology networks (*Cerebral Cortex* 2026, with Van Horn):** 337 youth from the Autism Centers of Excellence (ACE) network ([DOI](https://doi.org/10.1093/cercor/bhag156)).
- **Published null results.** For example, resting connectivity did not predict perceived stress in 417 adults ([DOI](https://doi.org/10.1097/PSY.0000000000001358)).

**Equipment, data and funding:**

- He does not run an EEG acquisition lab, and the lab page names no in-house equipment.
- None of his datasets involve motor imagery.
- No grants with him as PI were found; his funded work appears under other investigators' NIH awards ([CV](https://api.dsi.virginia.edu/sites/default/files/2023-08/cv_RASERO.pdf)).

**Public views:**

- He calls himself an open-science advocate who builds tools for the community ([Brain Institute news](https://braininstitute.virginia.edu/news/meet-javier-rasero-new-faculty-member-school-data-science)).
- At UVA's October 2024 brain and data science symposium, he argued that data must be cleaned of biases and that multiple cohorts must be harmonized to reach large samples ([SDS news](https://datascience.virginia.edu/news/uva-symposium-highlights-how-data-science-can-unlock-mysteries-brain)).

**Why he matters.** He is working on NeurOS's core problem in a clinical setting: pooling heterogeneous public EEG and asking which features stay stable across sessions and which shift with confounds. He is early-career, his lab is unfunded, and he has industry coding experience. All of this makes him plausibly receptive to an undergraduate collaborator on a reproducible cross-dataset benchmark.

**Gaps:**

- The full texts of the 2026 preprints could not be opened, so his handling of TUH montages, his channel subsets and his code licences are unverified.
- His UVA teaching is unconfirmed.
- No statements on industry or BCI were found.

**Contact:** [SDS profile](https://datascience.virginia.edu/people/javier-rasero), [lab page](https://datascience.virginia.edu/labs/brain-body-networks-lab), [jrasero.github.io](https://jrasero.github.io/).

### 3. John Darrell Van Horn: data governance, compute and the best introductions

**Role and standing.** Van Horn ("Jack") has been Quantitative Foundation Distinguished Professor of Data Science since August 2025 and is a tenured Professor of Psychology ([SDS profile](https://datascience.virginia.edu/people/john-van-horn)). According to his self-reported CV ([CV](https://www.jackvanhorn.org/s/JOHN-DARRELL-VAN-HORN-WINTER-2026.pdf)), he:

- has been Editor-in-Chief of *Neuroinformatics* since 2022;
- is an OHBM Fellow with more than 300 peer-reviewed papers;
- previously ran operations at Dartmouth's fMRI Data Center and worked at UCLA's and USC's Laboratory of Neuro Imaging (LONI).

**Lab.** SDS lists his lab as "Brain and Data Science". Graduate students there work with large-scale MRI and electrophysiology datasets ([lab page](https://datascience.virginia.edu/labs/brain-and-data-science)). His personal site invites upper-division student volunteers through a survey ([jackvanhorn.org](https://www.jackvanhorn.org/)).

**EEG involvement** comes through data coordination, not acquisition:

- **Data coordinating center (DCC).** He was site PI of the DCC for an NICHD-funded ACE network grant (2022-2025, marked "ended early" on his CV). The DCC handled QA, archiving and informatics for all neuroimaging and electrophysiology data from seven sites, including UVA ([CV](https://www.jackvanhorn.org/s/JOHN-DARRELL-VAN-HORN-WINTER-2026.pdf)).
- **First-cycle EEG** ([Neuhaus et al. 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8439051/)):
  - **Hardware:** **EGI 128-channel HydroCel nets at 500 Hz**.
  - **Paradigm:** eyes-open rest while watching slow videos.
  - **Participants:** **142 autistic and 138 typically developing youth** aged 8-17, at four sites.
  - **Availability:** archived in the National Database for Autism Research (NDAR) as study #2021.
- **Co-authored papers, 2023-2025,** on frontal alpha asymmetry, gamma oscillations, alpha-peak count and trial-to-trial spectral variability, plus a submitted paper on aperiodic power ([CV](https://www.jackvanhorn.org/s/JOHN-DARRELL-VAN-HORN-WINTER-2026.pdf)).

**Compute.** NIH RePORTER confirms his roughly **$2.0M S10 instrument grant**, S10OD038249. It funds a high-performance computing platform for neuroimaging, electrophysiological and genetic data and machine learning, with a project period of July 2025 to July 2026 ([RePORTER](https://reporter.nih.gov/project-details/11101913)).

**Training and teaching:**

- He directs the NIH-funded Biomedical Data Science Innovation Labs and its free virtual seminar series ([KnowInnovation](https://knowinnovation.com/inside-the-uva-bdsil-seminar-series/)).
- He teaches a PhD course that covers electrophysiology, and supervises Master of Science in Data Science (MSDS) capstones that work with outside clients ([CV](https://www.jackvanhorn.org/s/JOHN-DARRELL-VAN-HORN-WINTER-2026.pdf)).

**Industry precedent** ([CV](https://www.jackvanhorn.org/s/JOHN-DARRELL-VAN-HORN-WINTER-2026.pdf)):

- an NIH STTR grant with Kitware on assessing traumatic brain injury from imaging;
- IBM and HP sponsorship of meetings he organized.

**Public views:**

- **On machine learning** (UVA Data Points podcast, 2023): he is skeptical of purely data-driven ML, warning that a classifier can reveal more about the algorithm than about the brain. He also urged undergraduates to learn programming and data management early ([podcast](https://uvadatapoints.castos.com/episodes/brain-science-is-data-science)).
- **On data sharing:** he wrote a *Neuroinformatics* editorial on its economics ([PubMed](https://pubmed.ncbi.nlm.nih.gov/37966621)).
- **Recognition:** he won OHBM's 2025 Education in Neuroimaging Award ([SDS news](https://datascience.virginia.edu/news/john-van-horn-recognized-ohbm-2025-education-neuroimaging-award)).
- **Recent media:** he co-hosted an April 2026 Data Points episode on digital twins ([SDS news](https://datascience.virginia.edu/news/uva-data-points-podcast-digital-twins)).

**Why he matters.** He knows from the inside how multi-site EEG data are QA'd and harmonized. He can speak to the norms for commercial use of public data, and he can say what validation design a neuroinformatics reviewer would trust. His skepticism of black-box classifiers fits NeurOS's own discovery that an early 75% result was driven by cues rather than brain signal; that story will land with him. He is also the natural route to Kevin Pelphrey, the autism network's local lead, and to other SDS computational neuroscientists.

**Gaps:**

- Most details come from his self-reported CV.
- His University of Washington eScience affiliation ran through May 14, 2026. He is probably back in Charlottesville this fall, but this is unconfirmed.
- Whether the S10 compute platform is open to undergraduate or commercial work is unknown.
- No public views on startups were found.
- He runs no EEG acquisition lab.

**Contact:** [SDS profile](https://datascience.virginia.edu/people/john-van-horn), [lab page](https://datascience.virginia.edu/labs/brain-and-data-science), [jackvanhorn.org](https://www.jackvanhorn.org/).

### 4. Per B. Sederberg: the closest UVA precedent for a decoder that skips recalibration

**Role and lab.** Sederberg is Professor of Psychology and Director of the Cognitive Science Program, and joined UVA in 2017 ([profile](https://psychology.as.virginia.edu/people/sederberg)). His Computational Memory Lab combines multivariate fMRI and EEG analysis with computational models of memory ([compmem.org](https://compmem.org/)).

**BCI credentials:**

- **Schwemmer et al., *Nature Medicine* 2018** (co-author) ([PubMed](https://pubmed.ncbi.nlm.nih.gov/30250141/)). The paper built a deep-neural-network decoder of intracortical signals for a participant with tetraplegia. The decoder:
  - **sustained high accuracy for more than a year without explicit daily retraining**, through an unsupervised updating procedure;
  - **learned new functions with minimal retraining** through transfer learning;
  - drove functional electrical stimulation of the participant's forearm in real time.
- **Bouton et al., *Nature* 2016** (co-author), on restoring cortical control of movement ([PubMed](https://pubmed.ncbi.nlm.nih.gov/27074513/)).
- **Related patent.** The patent with the same title as the 2018 paper is assigned to Battelle, and he is not a named inventor ([Google Patents](https://patents.google.com/patent/US11752349B2/en)).

**Recent work.** His only 2023-2026 EEG paper links drift-diffusion decision parameters to frontal-midline theta and gaze ([PubMed](https://pubmed.ncbi.nlm.nih.gov/36682508/)). His 2025-2026 output is mostly computational and behavioral.

**Tools:**

- co-author of the PyMVPA toolbox ([PubMed](https://pubmed.ncbi.nlm.nih.gov/19212459/));
- maintains the SMILE Python experiment framework under the GPL-3.0 licence ([GitHub](https://github.com/compmem/smile)).

**Participants.** He is listed on the site of the Virginia Cognitive Aging Project (VCAP), an older-adult cohort; his role there is not stated ([VCAP](https://uva.theopenscholar.com/vcap/people/sederberg)).

**Why he matters.** His decoder is the nearest UVA precedent for NeurOS's "no calibration session" claim. The analogy is partial: his decoder stayed stable across days for one intracortical user, whereas NeurOS needs transfer across users on scalp EEG. He is well placed to stress-test NeurOS's evaluation design, and his Battelle collaboration gives him first-hand experience of academic-industry BCI work.

**Gaps:**

- Whether his lab currently acquires EEG is unclear; its site still advertises Summer 2023 openings.
- No public views on neurotech startups were found.
- SMILE's GPL licence would require NeurOS to release its own source code under the same terms if SMILE were bundled into a commercial product.

**Contact:** [Psychology profile](https://psychology.as.virginia.edu/people/sederberg), [compmem.org](https://compmem.org/).

### 5. Nicole Long: a 64-channel reference rig and published cross-participant decoding

The research notes treated Long as an already-known contact, so her profile is thin. Her academic rank and a direct profile URL were not captured.

**Equipment and protocol.** Her Long Term Memory Lab in Psychology records with a **BrainVision system and a 64-channel active actiCAP** at 1,000 Hz ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13543892/)). The same 2026 *iScience* paper:

- **validated cross-participant and cross-study decoding of memory brain states**;
- ran under the IRB-SBS protocol SBS-2115.

**Funding.** She holds NINDS R01NS132872, "Multivariate Decoding of the Retrieval Brain State Across Cognition", running 2023-2028 ([RePORTER](https://reporter.nih.gov/project-details/11357643)).

**Recruitment:**

- middle-aged and older adults through VCAP, and young adults through campus flyers ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12050195/));
- the lab also appears on Psychology's list of community participant registries ([Participate page](https://psychology.as.virginia.edu/participate-research-uva-psychology)).

**Data.** Her datasets on OSF carry a **CC BY-NC 4.0 (non-commercial) licence** ([OSF m6knv](https://osf.io/m6knv/); [OSF qmzfv](https://osf.io/qmzfv/)).

**Why she matters:**

- She has peer-reviewed UVA evidence that pooled, cross-participant decoders work.
- Her research-grade wet 64-channel system is a reference against which dry headsets can be judged.
- She has an IRB-SBS EEG pathway already in place.

**The licence is a hard constraint.** NeurOS should not train commercial models on her public data without written permission.

**Contact:** her lab appears on the [Psychology labs page](https://psychology.as.virginia.edu/research-opportunities-working-lab).

### 6. Tanya Evans: dry wireless headsets and the Brain Institute's front door

**Role.** Evans became Co-Director of the UVA Brain Institute in December 2025, alongside Jaideep Kapur ([Brain Institute news](https://braininstitute.virginia.edu/news/meet-tanya-evans-uva-brain-institute-co-director)). The institute's team page lists her as an Associate Professor with neuroimaging and EEG expertise; her home school is Education and Human Development ([team page](https://braininstitute.virginia.edu/team); [SEHD directory](https://education.virginia.edu/about/directory/tanya-evans)).

**EEG equipment and study.** Her group ran a child-parent "hyperscanning" study (recording two people's EEG at once) ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10518595/)):

- **Hardware:** two **dry wireless Cognionics Quick-20m headsets**, 19 channels at 500 Hz, with a Bluetooth wireless trigger.
- **Recruitment:** families from Albemarle County, as part of a larger autism study.

**Public views:**

- **Research style.** She stresses theoretically grounded, longitudinal work and fluency in team science ([podcast](https://uvadatapoints.castos.com/episodes/brain-science-is-data-science)).
- **Representative samples.** She has argued for representative samples in educational neuroscience, with community engagement and transparent reporting of demographics ([PubMed](https://pubmed.ncbi.nlm.nih.gov/38879200/)).
- **Priorities as co-director:** translational team science, and lowering barriers to collaboration across schools ([Brain Institute news](https://braininstitute.virginia.edu/news/meet-tanya-evans-uva-brain-institute-co-director)).

**Commercialization climate.** The institute is openly pro-commercialization. It has invested more than $6.5M in seed funding since 2016, and five of its PIs have struck licensing deals with UVA Licensing & Ventures ([C-VILLE](https://c-ville.com/is-uva-getting-smarter-about-how-it-plays-its-biotech-research-cards/)).

**Why she matters:**

- Her dry wireless headsets are a second consumer-style hardware class for cross-device tests.
- As co-director, she is the best route to Brain Institute introductions, partners for faculty pilot grants, and funding calls.

**Gaps.** Her current grants, and whether her headsets are free for outside sessions, were not researched. SEHD pages blocked automated fetching.

**Contact:** [Brain Institute profile news](https://braininstitute.virginia.edu/news/meet-tanya-evans-uva-brain-institute-co-director) and [SEHD directory](https://education.virginia.edu/about/directory/tanya-evans).

### 7. Sara Riggs: EEG decoding inside the founder's own department

**Role and research.** Riggs is an Associate Professor of Systems & Information Engineering ([profile](https://engineering.virginia.edu/faculty/sara-lu-riggs)). She studies:

- attention and interruption management;
- multimodal displays, with a focus on touch;
- adaptive displays that adjust to sensed conditions.

She belongs to SIE's Human Technology Interaction research area with Gerling, Bolton and Iqbal ([HTI page](https://engineering.virginia.edu/department/systems-and-information-engineering/research/human-technology-interaction)).

**EEG work:**

- **Horde & Riggs, HFES 2025.** A support-vector machine was trained on ERP voltages over somatosensory cortex. It classified tactile change vs no change at **63% at 390-410 ms** after the stimulus, and the authors framed the work as relevant to brain-machine interfaces ([DOI](https://doi.org/10.1177/10711813251369805)).
- **Haptics papers** (2025 and 2026) on tactile change detection measured with EEG/ERP ([World Haptics 2025](https://doi.org/10.1109/WHC64065.2025.11123332)).
- **Lauren Horde's April 2026 dissertation.** It covers **72 participants across three EEG experiments**, analysed with ERP components and time-resolved decoding. It is CC BY-licensed but embargoed until April 1, 2027 ([LibraETD](https://libraetd.lib.virginia.edu/public_view/oid-d7668qo1qb2s73jant0g)).

**Workload measurement.** In a 2023 study using the MATB-II workload task (a standard multi-task battery), Apple Watch and Fitbit heart signals **failed to detect workload differences** that participants' own ratings did detect ([DOI](https://doi.org/10.1177/21695067231192629)).

**Funding and students:**

- She held an NSF CAREER award (2019-2025) ([NSF 2008680](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2008680)).
- An April 2026 UVA story features an undergraduate working on her noninvasive brain-sensing research ([UVA news](https://engineering.virginia.edu/news-events/news/another-dimension-learning-uva-systems-engineers-take-undergraduate-research)).

**Why she matters:**

- Hers is the only research-grade EEG decoding pipeline in the founder's department.
- MATB-II is a ready-made way to manipulate effort when validating NeurOS's rest-vs-effort output.
- Joining her lab is an easy route for an SIE undergraduate.

**Gaps:**

- Her EEG amplifier and channel count are not public.
- Whether her decoding was evaluated within or across participants is unknown.
- She has no lab website.

**Contact:** [profile](https://engineering.virginia.edu/faculty/sara-lu-riggs).

### 8. Donald E. Brown: augmentation for cross-subject generalization, plus a founder's perspective

**Roles** ([SDS profile](https://datascience.virginia.edu/people/donald-brown)):

- Quantitative Foundation Distinguished Professor of Data Science.
- W.S. Calcott Professor of Systems & Information Engineering.
- Founding Director of UVA's Data Science Institute.
- IEEE Fellow.
- President of Commonwealth Computer Research, Inc., which provides data analysis for private and government clients.

His profile also lists a co-directorship of iTHRIV, the translational research institute that wound down in October 2025, so that title is out of date ([CTU newsletter](https://med.virginia.edu/clinicalresearch/ctu-newsletter/october-2025-edition/ithriv-sunset-in-october/)).

**EEG work: AugUNet1D** (*Scientific Reports*, July 2026, with the School of Medicine lab of Mark Beenhakker) ([PubMed](https://pubmed.ncbi.nlm.nih.gov/42481563/)):

- **Task:** segmenting spike-wave discharges (absence-seizure events) in single-channel EEG.
- **Data:** **961 hours of annotated mouse EEG** containing 22,637 events.
- **Model:** a residual 1D U-Net trained with **amplitude scaling, Gaussian noise injection and signal inversion** to generalize across subjects.
- **Results:** F1 0.90, a 29% relative improvement over the baseline.
- **Availability:** the model and dataset are stated to be public.

**Why he matters:**

- **Augmentation as a cross-headset hypothesis.** His augmentation recipe mimics gain and noise differences, and those are the kinds of differences that separate EEG headsets. Whether it transfers across headsets is an untested hypothesis worth asking him about.
- **A founder's perspective.** He runs a company alongside his faculty role, so he can speak to that dual life.
- **An SIE route.** He is in SIE, which opens the capstone and independent-study route.

**Gaps:**

- The data come from mice, not humans.
- Where the model and data are hosted, and under what licence, were not found.
- The arXiv preprint and the journal abstract disagree on how many classifiers were compared (15 vs 16) ([arXiv](https://arxiv.org/abs/2601.00459)).

**Contact:** [SDS profile](https://datascience.virginia.edu/people/donald-brown).

### 9. Jeff Changchia Liu: active human EEG protocols in the School of Medicine

**Role.** Liu is a Research Associate Professor of Neurosurgery in the Human Brain Function Lab and is on the Biomedical Sciences graduate faculty ([lab page](https://www.humanbrainlab.net/people/jeff-c-liu); [BIMS](https://med.virginia.edu/bims/faculty/?facbio=1&id=1730860)).

**Research and methods.** He studies human pain perception and neuromodulation, using:

- intracranial and single-neuron recordings;
- scalp EEG;
- deep brain stimulation, focused ultrasound and TMS;
- laser stimulation;
- machine learning and nonlinear dynamics.

**Insular DBS grant and trial.** He is co-PI with Jeff Elias on NINDS UH3NS123308, which runs to 2028 (FY2025 award $901,805) ([RePORTER](https://reporter.nih.gov/project-details/10918114)):

- Twelve patients with refractory neuropathic pain receive electrodes along the insula.
- A randomized, sham-controlled DBS crossover trial follows.
- The linked trial is recruiting ([NCT05404581](https://clinicaltrials.gov/study/NCT05404581)).

**Intracranial EEG (*Brain Communications*, 2026).** The study recorded multiday bilateral iEEG from six patients (34 days in total). Mixed-effects models separated a **stable cohort-level spectral gradient from day-to-day fluctuation** ([PubMed](https://pubmed.ncbi.nlm.nih.gov/42582622/)).

**Scalp EEG in healthy volunteers (2025-2026):**

- reliable laser-evoked-potential biomarkers of descending pain inhibition ([PubMed](https://pubmed.ncbi.nlm.nih.gov/41328665/));
- a study of how pain catastrophizing amplifies parietal responses ([PubMed](https://pubmed.ncbi.nlm.nih.gov/40910668/)).

**Why he matters:**

- He has approved scalp EEG protocols with healthy volunteers in the School of Medicine.
- He frames his findings as "stable across people despite daily variability", which is close to NeurOS's own thesis.
- He links to functional neurosurgery (Elias, Moosa) and to Quigg.

**Gaps.** His EEG hardware is not named, and no public views on startups were found.

**Contact:** [lab page](https://www.humanbrainlab.net/people/jeff-c-liu).

### 10. Mark S. Quigg: the gate to UVA's clinical EEG volume

**Role.** Quigg is T.R. Johns Professor in Neurology (Epilepsy) ([Neurology faculty](https://med.virginia.edu/neurology/faculty/)). A 2017 article named him director of UVA's EEG, intensive-monitoring and evoked-potential laboratories; that title needs re-verifying ([UVA Health 2017](https://www.uvahealth.com/healthy-practice/uva-earns-abret-accreditation-for-eeg-units/)).

**Clinical EEG volume** ([fellowship page](https://med.virginia.edu/neurology/subspecialty-divisions/epilepsy/clinical-neurophysiologyeeg-and-epilepsy-fellowships/program-description/)):

- about **3,000 routine EEGs** a year;
- about **1,200 continuous ICU EEGs** a year;
- about 450 epilepsy-monitoring admissions a year.

**Retrospective EEG study (2026).** He co-authored a review of 806 routine UVA EEGs from 659 patients. It found epileptiform abnormalities in 27% and identified recording factors that raise the yield ([PubMed](https://pubmed.ncbi.nlm.nih.gov/41705827/)).

**Clinical trials training.** He co-leads NINDS R25NS138633, a clinical-trials methodology course running to 2029 ([RePORTER](https://reporter.nih.gov/project-details/11413469)).

**Why he matters:**

- The clinical archive mixes routine, ambulatory and ICU systems. That heterogeneity is a real-world hardware-agnostic test.
- He can give pilot-design advice.

**Timeline caution.** Access to the archive would need an IRB-HSR protocol, a HIPAA de-identification check and a data agreement. That is a horizon of months, not weeks.

**Gaps.** His current directorship and his clinical EEG vendors are unverified.

**Contact:** [Neurology faculty](https://med.virginia.edu/neurology/faculty/).

### Second-tier contacts fill specific gaps

These people did not make the top ten but each solves a specific problem: departmental access, recruitment, test beds, ethics, or commercialization mentoring.

| Person | Role | Specific value to NeurOS | Contact page |
|---|---|---|---|
| Kevin Pelphrey | Harrison-Wood Jefferson Scholars Foundation Professor of Neurology; multi-PI, ACE Network | A July 2026 study of **306 youth recorded with 128-sensor EEG**, with UVA among the sites ([UVA Today](https://news.virginia.edu/content/uva-researchers-find-brain-signal-linked-communication-challenges-autism)); the ACE grant runs to 2027 ([RePORTER](https://reporter.nih.gov/project-details/11415604)). Best reached via Van Horn | [Neuroscience page](https://neuroscience.virginia.edu/kevin-pelphrey) |
| Gregory Gerling | Professor and interim chair of SIE (April 2026) | Departmental gatekeeper for credit, capstone and space; co-authored infant EEG with Puglia ([PubMed](https://pubmed.ncbi.nlm.nih.gov/39181012/)) | [Profile](https://engineering.virginia.edu/faculty/gregory-j-gerling) |
| Meghan Puglia | Associate Professor of Neurology; Brain Institute Director of Community Engagement | Infant EEG analysed without per-subject calibration; NICU EEG entropy predicted later autism risk with 88% accuracy ([PubMed](https://pubmed.ncbi.nlm.nih.gov/41576766/)); community channels for usability testing | [Brain Institute](https://braininstitute.virginia.edu/news/meet-meghan-puglia-uva-brain-institute-director-community-engagement) |
| Zachary Irving | Associate Professor of Philosophy; directs the HooCS Hub | His inter-subject EEG classifier (AUC 0.667) slightly beat the per-subject one (0.641) ([PubMed](https://pubmed.ncbi.nlm.nih.gov/40048826/)); a forum on agency and AI for intent-decoding ethics | [HooCS Hub](https://hoocs.science/) |
| Teague Henry | Assistant Professor of Data Science and Psychology | A 2025 method for fast leave-one-cluster-out cross-validation ([PubMed](https://pubmed.ncbi.nlm.nih.gov/40534513/)); warns that data-driven searches always find something ([podcast](https://uvadatapoints.castos.com/episodes/brain-science-is-data-science)) | [SDS profile](https://datascience.virginia.edu/people/teague-henry) |
| Arsalan Heydarian | Associate Professor, Civil & Environmental Engineering / Link Lab; co-director, Omni-Reality & Cognition Lab (ORCL) | Four simulators, 25 XR headsets, heart-rate, skin-conductance and eye-tracking sensors, **no EEG**; explicitly invites outside studies ([ORCL facilities](https://engineering.virginia.edu/labs-groups/omni-reality-cognition-lab/facilities)) | [ORCL](https://engineering.virginia.edu/labs-groups/omni-reality-cognition-lab) |
| Tariq Iqbal | Associate Professor, SIE and Computer Science | Trust model (n=30) built from a Franka robot arm, Empatica E4 wristband and gaze data ([arXiv](https://arxiv.org/abs/2504.05291)); NSF CAREER ([NSF](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2441587)); a robot to drive in a demo | [Lab](https://www.collabrobotics.com/) |
| Afsaneh Doryab | Associate Professor, SIE and Computer Science; Human-AI Technology (HAI) Lab | A 2026 wearable study found physiological changes **did not generalize across participants** ([DOI](https://doi.org/10.1109/SIEDS69358.2026.11540300)): the counter-thesis NeurOS must answer | [HAI Lab](https://hai-lab-uva.github.io/) |
| Matthew Bolton | Associate Professor, SIE | Critiques of the NASA-TLX workload questionnaire for designing user-study measures ([DOI](https://doi.org/10.1109/THMS.2026.3664262)) | [drbolton.org](https://drbolton.org) |
| Benton Calhoun | Professor, Electrical & Computer Engineering | Co-founded PsiKick, now Everactive, which raised $30M ([EEPower](https://eepower.com/news/psikick-rebrands-to-everactive-raises-30m-to-grow-batteryless-solutions-for-iiot/)); a UVA Licensing & Ventures venture ([LVG](https://lvg.virginia.edu/ventures/everactive)) | [Lab](https://engineering.virginia.edu/labs-groups/robust-low-power-vlsi) |
| Silvia Blemker | Professor, Biomedical Engineering | Springbok's MuscleView AI won FDA 510(k) clearance after training on MRI from different manufacturers and field strengths, the closest UVA analogue to a cross-headset claim ([UVA news](https://engineering.virginia.edu/news-events/news/uva-biomedical-engineering-professor-receives-fda-clearance-ai-muscle-health-analysis-tool)) | [M3 Lab](https://engineering.virginia.edu/labs-groups/multiscale-muscle-mechanophysiology-m3-lab) |
| Andrew Carlson | Professor of Neurosurgery (Stroke) | Co-published a 29-electrode forehead EEG patch with a small device company ([PubMed](https://pubmed.ncbi.nlm.nih.gov/40800892/)); stroke electrocorticography (ECoG) R01 | [UVA Health profile](https://www.uvahealth.com/providers/Andrew-Carlson-1811036825) |
| Shawn Russell | Assistant Professor, Orthopaedic Surgery; Motion Analysis Lab | EMG and motion capture as objective labels for effort; a data bank of 600+ children with cerebral palsy ([lab page](https://med.virginia.edu/orthopaedic-surgery/research/the-motion-analysis-and-motor-performance-lab/)) | [Lab page](https://med.virginia.edu/orthopaedic-surgery/research/the-motion-analysis-and-motor-performance-lab/) |
| Rose Nevill | Directs the Neurodevelopmental Research Core | Recruitment, coordination and assessment services for neurodevelopmental studies ([Brain Institute](https://braininstitute.virginia.edu/neurodevelopment)) | [Brain Institute](https://braininstitute.virginia.edu/neurodevelopment) |

## UVA owns lab data, so every route runs through the IRB, OSP and OVPR

### Faculty cannot hand over lab EEG informally

**Who owns the data.** UVA Policy RES-002 makes research records University property whenever significant University resources produced them. "Significant" includes laboratories and specialized equipment, and "records" explicitly include computer files, consent forms and data ([RES-002](https://uvapolicy.virginia.edu/policy/RES-002)). Unpublished records count as sensitive data and must stay on University-controlled systems.

**What a transfer requires.** Any transfer of sensitive human-subjects records needs approval from the responsible office, which for human data is the IRB. A temporary copy approved by a PI confers no rights to future use.

**Who signs.** OSP negotiates and executes data use agreements (DUAs), not the faculty member ([OSP DUAs](https://osp.research.virginia.edu/research-administration/research-contracts/material-transfer-agreements-data-use-agreements)). For health-system data the rules are stricter ([IRB-HSR: Leaving UVA](https://hrpp.research.virginia.edu/teams/irb-hsr/researcher-guide-irb-hsr/leaving-uva)):

- No data may leave without a signed Data Transfer Agreement through OSP or the School of Medicine's grants office.
- Identifiable information collected without consent may not leave at all.
- Data claimed to be de-identified must first be checked by a health-system office.

**Commercialization of UVA-owned work.** The VP for Research decides whether University-owned copyrights are marketable, and UVA Licensing & Ventures Group (LVG) handles commercialization ([RES-001](https://uvapolicy.virginia.edu/policy/RES-001)). LVG handles every startup built on UVA intellectual property ([UVA Innovates: LVG](https://innovates.virginia.edu/ecosystem/lvg)).

**What faculty can share freely:** published papers, methods and protocols, general advice, and pointers to datasets they have already deposited publicly.

**Licences on public deposits still bind:**

- Long's OSF EEG data are non-commercial (CC BY-NC 4.0) ([OSF](https://osf.io/m6knv/)).
- The autism network's EEG sits in NDAR study #2021 ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8439051/)). NDAR access normally requires an institutionally sponsored data use certification, which in practice means a faculty sponsor. NDA's current terms for startups were not verified.
- The licence terms for the TUH corpus, which Rasero uses, were not checked either.

**Joining a lab vs staying outside.** Joining a lab as a research assistant gives access to data but turns the work into University research. Accessing data as a company keeps the IP clean but needs a DUA, and DUAs take weeks to months.

### EEG user testing probably needs a faculty PI and IRB-HSR

**When IRB review applies.** UVA requires IRB approval before any human-subjects research begins, and "research" includes advertising, recruiting and screening. Research means a systematic investigation designed to contribute to generalizable knowledge, whether or not it is published ([IRB-SBS guidance](https://hrpp.research.virginia.edu/teams/irb-sbs/researcher-guide-irb-sbs/activities-require-irb-sbs-review)). Quality-assurance work not meant to be shared beyond UVA, and classroom training exercises, typically fall outside it. If a device study aims to establish safety or effectiveness, FDA regulations add their own IRB requirement ([HRPP non-human-subjects page](https://hrpp.research.virginia.edu/non-human-subject-research)).

**Which board.** UVA's guide lists physiological measures and FDA-regulated products as examples for the health-sciences board (IRB-HSR), and behavior, interviews and surveys for the social and behavioral board (IRB-SBS) ([Which side](https://hrpp.research.virginia.edu/which-side-irb-should-i-submit-my-study)). Practice is murkier: Long's cognitive EEG studies ran under IRB-SBS ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13543892/)). Ask both boards before choosing.

**The student-PI rule.** On IRB-HSR **a student cannot be principal investigator**. A faculty member or UVA employee must be PI, with the student as study coordinator. Alternatively, a student can be added to an existing approved protocol, and consultations with IRB coordinators are free ([Student researchers](https://hrpp.research.virginia.edu/student-researchers)).

**Getting a formal "not research" determination.** HRPP's Evaluate My Project tool issues a formal letter, and HRPP announced an update to the tool for September 2026 ([HRPP home](https://hrpp.research.virginia.edu/)). The next full board meetings are:

- IRB-HSR: October 13 and 27.
- IRB-SBS: October 14 and 28.

**Virginia state law.** Human research not covered by federal rules falls under Virginia Code §32.1-162.16 and following sections. These require witnessed written consent and approval by a human research review committee, and physiological recording is not among the exemptions ([Code of Virginia](https://law.lis.virginia.gov/vacodefull/title32.1/chapter5.1/)).

**Three realistic paths.** These are inferences from the policies above, not rulings:

| Path | What it looks like | Main catch |
|---|---|---|
| **A. Internal product testing** | NeurOS runs it on company hardware, with no UVA channels and no intent to publish | Probably outside "research", but confirm in writing first |
| **B. Faculty-PI study** | A UVA faculty member runs it under IRB-HSR or IRB-SBS, with the founder as coordinator | The founder's equity must be disclosed |
| **C. Company-sponsored generalizable research** | NeurOS runs a study meant to produce publishable findings, outside UVA | Brings in the Virginia statute; commercial IRBs commonly fill this role, but that was not verified for Virginia |

**Decide early whether the zero-calibration results will be published.** A public benchmark or paper is generalizable knowledge and so counts as research.

### Keep NeurOS's code outside the lab, and let faculty disclose before they advise

**Who owns student work.** The 2026-27 Undergraduate Record says students generally own what they create. The exceptions are work made through significant use of University resources, or within UVA employment such as a research-assistant job. UVA claims "University Research", defined to include any activity that uses University-controlled facilities. Students facing conflicts with third-party contracts must raise them with the Office of the Vice President for Research (OVPR) before starting research ([Undergraduate Record](https://records.ureg.virginia.edu/content.php?catoid=72&navoid=6592)).

**Patents vs software:**

- The patent policy requires a patent agreement from anyone engaged in University research. UVA normally relinquishes claims to inventions the VP for Research judges not to be products of University research, usually within 60 days of a complete disclosure ([RES-006](https://uvapolicy.virginia.edu/policy/RES-006)).
- Software falls under the copyright policy. It treats office computers and occasional shared facilities as not significant use, but specialized equipment and dedicated staff help as significant ([RES-001](https://uvapolicy.virginia.edu/policy/RES-001)).

**Practical upshot.** Keep developing NeurOS on personal or company hardware and cloud. Keep dated git history. Talk to OVPR before taking any paid or for-credit lab role on closely related work.

**Conflict-of-interest rules that bind faculty:**

- **Faculty are public officials** who must disclose or recuse where they have a special stake in an outcome ([PROV-007](https://uvapolicy.virginia.edu/policy/PROV-007)).
- **Any equity in a private company** is a Significant Financial Interest for a research investigator ([RES-005](https://uvapolicy.virginia.edu/policy/RES-005)).
- **Institutional conflicts** in human-subjects research are covered separately ([RES-008](https://uvapolicy.virginia.edu/policy/RES-008)).
- **Equipment.** University employee time and equipment may not be used for commercial purposes without a contract. Students may use equipment for scholarship or approved extracurriculars with the equipment supervisor's approval ([PRM-011](https://uvapolicy.virginia.edu/policy/PRM-011)).

**Practical upshot.** Ask for supervised sessions under an existing protocol rather than to borrow equipment. Offer equity or advisory roles only after a faculty member has checked with their conflict-of-interest office.

**Free founder-side legal advice** is available from UVA Law's Entrepreneurial Law Clinic ([UVA Innovates: Law](https://innovates.virginia.edu/ecosystem/law)).

### No human EEG core: participants and hardware live in individual labs

**Shared facilities:**

- **MRI:** two Siemens 3T research scanners are open to UVA and outside researchers after screening and safety training ([MRI research services](https://med.virginia.edu/radiology-research/imaging-services/mri-research-services/)).
- **Biomarker and Neuroimaging Core:** a new provisional core ([core page](https://med.virginia.edu/neuroimaging-core/)).
- **MEG:** no system was found at UVA.
- **TMS:** appears to be clinical (depression treatment) only.

**Participant pools:**

- **Psychology's SONA pool** draws on undergraduates in introductory psychology courses and closes in early December ([SONA](https://uva-psych.sona-systems.com/)). Whether researchers outside Psychology can use it is unverified.
- **Psychology's community registries** cover infants through older adults ([Participate page](https://psychology.as.virginia.edu/participate-research-uva-psychology)).
- **ResearchMatch.** After iTHRIV wound down, recruitment support moved to the School of Medicine's Clinical Trials Unit. Over three years, 44 UVA studies used ResearchMatch to contact more than 466,000 volunteers, and more than 9,200 expressed interest ([CTU newsletter](https://med.virginia.edu/clinicalresearch/ctu-newsletter/october-2025-edition/ithriv-sunset-in-october/)).

**Research credit.** PSYC 3590 gives 2-3 credits for 7-10 hours a week of lab work; supervision by a faculty member outside Psychology needs prior approval ([Psychology labs page](https://psychology.as.virginia.edu/research-opportunities-working-lab)).

**Inclusive dry-electrode testing.** SEHD Dean Stephanie Rowley co-authored a paper showing that hair density and texture disproportionately exclude Black and Latinx participants from EEG studies ([PubMed](https://pubmed.ncbi.nlm.nih.gov/38565857/)). This is directly relevant to how NeurOS recruits for dry-electrode testing.

## Sixteen days to the E-Cup deadline: programs, funding and neurotech communities

### Student entrepreneurship programs

**The Foundry** is UVA's front door for student founders. It is at 9 Elliewood Ave, open to any school and major, with no application ([UVA Innovates: students](https://innovates.virginia.edu/students)). It runs mentor matching and the university-wide E-Cup ([Foundry](https://innovates.virginia.edu/ecosystem/foundry)).

**E-Cup** runs three stages a year across five tracks, with about $100,000 in total prizes, including an Engineering & Healthcare Technologies track. Teams can have up to six people, with at least one current UVA student ([E-Cup](https://innovates.virginia.edu/ecosystem/ecup)). The current Concept window ([Batten funding page](https://www.darden.virginia.edu/batten-institute/entrepreneurship/ventures/funding-competitions)):

- gives every entrant more than **$15K in AI credits, software and vendor perks just for applying**;
- requires only an early-stage idea;
- is **due October 23 at noon**, with the Concept Final on **November 13**.

**Two caveats on the dates.** The page does not state the year; 2026 is inferred from upload dates and weekdays. The old E-Cup site was down when checked, so confirm the dates at the Foundry.

**Other programs:**

| Program | What it offers | Source |
|---|---|---|
| Batten i.Lab | Ten-week summer accelerator, $5,000-10,000 per venture; the 2026 cohort had 22 teams | [i.Lab](https://innovates.virginia.edu/ecosystem/batten-ilab) |
| VentureForward (Engineering) | Up to $1,000 for new teams and $2,500 for returning teams | [VentureForward](https://innovates.virginia.edu/ecosystem/ventureforward) |
| LVG's I-Corps program | Up to $2,700 for customer discovery | [Batten funding page](https://www.darden.virginia.edu/batten-institute/entrepreneurship/ventures/funding-competitions) |
| Galant Challenge | Spring investor showcase; finalists pitched for up to $6M in April 2026 | [Galant Challenge](https://innovates.virginia.edu/ecosystem/galant-challenge) |
| 434 Accelerator | $20K plus services for early-stage founders | [434 Accelerator](https://innovates.virginia.edu/ecosystem/434-accelerator) |

**Not open to NeurOS directly:**

- **LVG's seed funds** ([LVG Seed](https://innovates.virginia.edu/ecosystem/lvg-seed)) and the **Coulter Center's** device-translation funding ([Coulter](https://innovates.virginia.edu/ecosystem/coulter)) are aimed at UVA-IP and faculty-led projects.
- **Batten Innovation Grants** are limited to Darden students ([Batten Grants](https://innovates.virginia.edu/ecosystem/batten-grants)).

### Brain Institute and Link Lab

**Brain Institute pilot grants.** The Transformative Neuroscience Pilot Grants need at least **two faculty PIs** ([2026 RFP](https://braininstitute.virginia.edu/transformative-neuroscience-pilot-grants-2026)):

- The 2026 cycle offered up to **$35,000** and has already been awarded. Proposals involving human subjects could get extra support.
- The 2025 cycle offered up to $50,000 ([2025 RFP](https://braininstitute.virginia.edu/transformative-neuroscience-pilot-grants-2025)).
- NeurOS's route to this money is a proposal by two faculty partners. A natural pairing is one from SDS or SIE with one from Psychology or SEHD.

**Not available this year:**

- The undergraduate Future Leaders Summer Fellowship is not expected to run in 2026 ([fellowship](https://braininstitute.virginia.edu/uva-brain-institute-future-leaders-summer-fellowship)).
- The 2026 Brain Symposium (May 19-21) has passed. The institute's newsletter sign-up has an undergraduate option, and funding calls go out through it ([Brain Institute](https://braininstitute.virginia.edu/); [Funding](https://braininstitute.virginia.edu/funding)).

**Link Lab events.** The 2026 Research Day was September 25; its spring open house was held in early April ([Link Lab events](https://engineering.virginia.edu/labs-groups/link-lab/upcoming-events)).

### Student neurotech groups

**neurotech@uva** is the best-fit community ([HoosInvolved](https://hoosinvolved.virginia.edu/organization/neurotech)). Founded in 2026 with open membership, it:

- builds EEG headsets and BCI applications;
- runs journal clubs;
- runs workshops on writing outreach emails to UVA faculty.

Its officers and hardware are visible only after signing in.

**Other groups:**

- the Neuroscience Enthusiasts Club, which runs workshops with UVA faculty ([HoosInvolved](https://hoosinvolved.virginia.edu/organization/theneuroscienceenthusiastsclubatuvanec));
- Nu Rho Psi and the Society for Cognitive Science ([Cognitive Science groups](https://cogsci.as.virginia.edu/student-groups)).

**Not found:** an official NeuroTechX chapter at UVA ([NeuroTechX community](https://neurotechx.com/community/)), or a UVA-based scout for Dorm Room Fund or Contrary.

**IRB still applies.** Recruiting club members for any testing that counts as research needs IRB approval before the first message.

## Outreach angles: lead with validation questions, not a product demo

### General guidance

**UVA's own advice.** UVA's undergraduate research office recommends ([Undergraduate Research](https://undergraduateresearch.virginia.edu/our-opportunities/getting-started)):

- reading a mentor's current work before writing;
- connecting the email to that specific work;
- being transparent about goals and capacity (volunteer, for credit or paid);
- following up politely after at least ten business days.

**Startup-specific additions:**

- **State early that NeurOS is a student startup applying to YC.** This lets the faculty member check conflict-of-interest rules before engaging.
- **Make one small ask,** such as a 20-minute call or a specific methods question.
- **Never ask for raw data in a first email.** RES-002 means they cannot send it informally anyway.
- **Offer to go through OSP and the IRB** for anything formal.
- **Bring evidence:** NeurOS's own leave-one-dataset-out and cue-free results, and the founder's story of catching the cue-driven 75% result. Together these show the rigor these faculty value.

**Timing.** A follow-up ten business days after an email sent the week of October 12 lands near October 26-28. Send the first wave this week if any replies are to inform the November 1 YC application.

**Pairings and warm routes:**

- **Van Horn and Rasero** co-author papers and co-organized UVA's October 2024 symposium, so mention each in the other's email or propose a joint meeting.
- **Gerling**, as interim SIE chair, is the warm route to Riggs, Iqbal, Doryab and Bolton.
- **Evans** is the warm route to the Brain Institute.
- **Liu** is the warm route to Quigg and the neurosurgeons.

### What to reference and ask, person by person

| Person | Reference this | Core ask | Also worth asking (and what to avoid) |
|---|---|---|---|
| Sun | NeuroFlex's conclusion that per-user models are necessary; the CHASE 2024 per-subject 75.3% on BCI IV-2a | A head-to-head test: NeurOS's pooled zero-calibration model vs their per-subject transformer, on their dry 16-channel OpenBCI rig, with subjects held out | Whether NeuroFlex EEG could be shared under IRB 7070 plus a DUA; how they split overlapping epochs (ask, don't allege leakage); dry-electrode motion artifacts; whether the exosuit project will add EEG intent |
| Rasero | The June 2026 TUH preprint (aperiodic features most stable; medication effects); his October 2024 point on harmonizing cohorts | A methods call on harmonizing heterogeneous public EEG (montages, sampling rates, channel subsets), and whether aperiodic features could normalize across subjects and devices | Whether he would co-supervise a reproducible cross-dataset EEG benchmark; how his published nulls shape his leakage checks; whether his EEG/sEEG collaborators have motor recordings |
| Van Horn | His ACE data coordinating center (EGI 128-channel, NDAR #2021); his data-sharing economics editorial; his podcast caution about classifiers | What validation design a neuroinformatics reviewer would trust, e.g. leave-one-dataset-out plus physiological checks; share the cue-confound story | What a startup owes the community when training on public EEG (licences, NDA certifications, releasing benchmarks); an introduction to Pelphrey; whether a benchmark could be a capstone or volunteer project; whether the S10 compute is open to students |
| Sederberg | The 2018 decoder that held accuracy for more than a year via unsupervised updating, plus transfer learning | Whether unsupervised updating could adapt a pooled EEG model to a new user in the first minutes, replacing calibration | A critique of NeurOS's subject-, dataset- and device-held-out splits; lessons from Battelle on authorship and IP; Cognitive Science students for a small pilot |
| Long | Her 2026 cross-participant, cross-study decoding paper and R01 | One or two supervised sessions recording NeurOS's protocol on her 64-channel rig alongside a dry headset, under her protocol or an amendment | Written permission for commercial use of her CC BY-NC data, or a collaboration instead; whether her cross-participant classifier code can serve as a baseline |
| Evans | Her dry wireless Cognionics hyperscanning; her stated priority on cross-school collaboration | Introductions to Brain Institute neurotech faculty and pilot-grant partners | Whether her dry headsets could host a supervised cross-device session; her representative-samples work applied to inclusive testing |
| Riggs | The HFES 2025 tactile ERP decoding (63%); smartwatches missing MATB-II workload | Advice on validating rest-vs-effort decoding against controlled MATB-II workload | Her EEG system and channel count; whether her decoding was within or across participants; joining her lab as an SIE undergraduate |
| Brown | AugUNet1D's augmentation recipe for cross-subject generalization; its public model and data | Whether amplitude-scaling and noise augmentation would transfer across EEG headsets | Where the data are hosted and under what licence; an SIE/SDS capstone or independent study on cross-dataset motor-imagery benchmarking; running a company alongside a faculty role |
| Liu | The 2026 insula paper separating a stable cohort-level pattern from day-to-day fluctuation | Whether a short motor-imagery block could ride along on his healthy-volunteer scalp EEG protocols | What a DUA path for de-identified scalp EEG would look like; an introduction to the stereo-EEG team |
| Quigg | The 806-EEG archive study; his trial-methodology course | How montages and hardware vary across UVA's routine, ambulatory and ICU systems, plus pilot-design advice | The IRB-HSR and DUA path for a de-identified archive subset (treat as a 2027 horizon, not a YC-timeframe ask) |

## Thin and unverified areas to check before relying on this report

All five note files had complete section structures, but several sections rest on search snippets or pages that blocked automated access. The items below are the ones that would change a decision.

| Area | What is uncertain | How to resolve |
|---|---|---|
| Rasero identity | High confidence, but inferred from name, field and co-authorship | Confirm from the SDS profile before sending |
| Van Horn availability | His University of Washington affiliation ended May 14, 2026; presence in Charlottesville this fall not confirmed | Check recent SDS events, or ask in the email |
| S10 compute access | Whether undergraduate or company work can use the $2M platform | Ask Van Horn |
| Long and Evans profiles | Long's rank and profile URL, and Evans's current grants and headset availability, were not captured | Check department pages; ask directly |
| EEG hardware | Systems used by Riggs, Morris, Coan and the clinical labs are unnamed | Ask in emails |
| NeuroFlex details | Train/test split undescribed; the 2026 *Smart Health* abstract not retrieved | Ask Sun's group |
| Rasero preprints | Full texts inaccessible (TUH montage handling, code licence); TUH licence unchecked | Read the preprints; check TUH terms |
| NDAR/NDA access | Whether a startup can obtain a data use certification, and on what terms | Check the NDA site; ask Van Horn |
| IRB board | IRB-HSR's examples include physiological measures, yet Long's EEG ran under IRB-SBS | Ask both boards, or use Evaluate My Project |
| Outgoing DUAs to companies | Whether OSP alone handles them or LVG must be involved | Ask OSP office hours and LVG |
| E-Cup dates | Year inferred; canonical fall 2026 page returned 404 | Confirm at the Foundry |
| Brain Institute pilot cap | $35K (2026) vs $50K (2025); next cycle dates unknown | Watch the Brain Institute mailing list |
| SONA access | Whether non-Psychology researchers can recruit | Ask Psychology |
| Brown's data | Hosting location and licence not found; data are mouse EEG | Ask Brown |
| Faculty attitudes | No public statement on student startups was found for any profiled person | Disclose early and let them say |
| Grossmann data | The public OSF node returned an authentication error | Low priority |

## Conclusion

The research changes how NeurOS should think about UVA in two ways.

**UVA's asset is hardware variety, not data volume.** UVA has no motor-imagery data trove to mine. It does have four EEG hardware classes within a mile of each other, and that makes a "four headsets, one campus" validation the distinctive thing NeurOS can produce here. The internal evidence is split in a way that sharpens the pitch:

- **For pooling:** Long's cross-participant decoders, Irving's inter-subject classifier matching per-subject ones, and Brown's augmentation for cross-subject generalization all suggest pooled models can work.
- **Against pooling:** Sun's NeuroFlex and Doryab's wearable study both conclude that individuals need their own models.

Sun is therefore both NeurOS's best collaborator and its most valuable skeptic. Winning a fair head-to-head on her rig would be stronger evidence than any benchmark NeurOS runs alone.

**Aperiodic EEG features are an emerging UVA theme.** Several groups are converging, apparently without a shared program:

- Rasero's TUH work found aperiodic parameters the most stable markers ([medRxiv](https://doi.org/10.64898/2026.06.02.26354693)).
- Pelphrey's 306-youth study linked aperiodic "neural noise" to communication ([UVA Today](https://news.virginia.edu/content/uva-researchers-find-brain-signal-linked-communication-challenges-autism)).
- Ribic and Puglia found a flatter 1/f slope (the aperiodic component of the EEG spectrum) in preterm infants ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10225509/)).
- Van Horn's CV lists a submitted ACE-network paper on aperiodic power during speech perception ([CV](https://www.jackvanhorn.org/s/JOHN-DARRELL-VAN-HORN-WINTER-2026.pdf)).

Features that are stable within a person yet comparable across people are exactly what a zero-calibration decoder might normalize on. That gives NeurOS a scientific question to bring to these faculty, rather than only a product to show them.

**Run UVA work on a separate track from the core IP.** The institutional rules point to two tracks. NeurOS's core IP should be built company-side on public datasets. UVA collaborations should run as faculty-PI studies that produce publishable validation, with conflicts of interest disclosed up front. Before November 1, the realistic yield is conversations, advisers and a planned study, not new data.
