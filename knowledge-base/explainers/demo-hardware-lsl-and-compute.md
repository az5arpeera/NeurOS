# Demo hardware, Muse, LSL, and compute

*Answers open questions Q5 and Q6 (founder asked 2026-10-01). Prices are as listed by vendors in Oct 2026; **verify at checkout**.*

## What's Muse, and why it's excluded
Muse (by Interaxon) is a consumer meditation headband with **4 dry electrodes**: two on the forehead (AF7, AF8) and two behind the ears (TP9, TP10). It's good for relaxation and focus feedback, but it has **nothing over the motor cortex** (C3 / Cz / C4, on top of the head), so it can't detect motor imagery.

## Does LSL cost anything?
**No.** Lab Streaming Layer is **free and open-source** (MIT licence), maintained by the open-source community that originated at UCSD/SCCN. It's the standard way EEG devices stream time-stamped data to software, through the `pylsl` Python package. Most research headsets ship an LSL app or work through BrainFlow.

## Headset shortlist (must cover the motor cortex and stream over LSL)

| Headset | Channels (positions) | Electrodes | LSL | Price (vendor, Oct 2026) | Notes |
|---|---|---|---|---|---|
| **g.tec Unicorn Hybrid Black** | 8: Fz, **C3, Cz, C4**, Pz, PO7, Oz, PO8 · 250 Hz · 24-bit | Hybrid: dry **or** gel | Yes (UnicornLSL app, Windows) | ~$1.5k ([g.tec](https://www.gtec.at/product/unicorn-hybrid-black/)) | Exactly the classic motor-imagery trio; gel mode gives reliable signal on strangers; independently validated (e.g. [Oostenveld review](https://robertoostenveld.nl/unicorn-hybrid-black/)). Pz/PO/Oz also allow visual paradigms as a backup. |
| **OpenBCI Ultracortex Mark IV + Cyton/Daisy** | Up to 16, positions configurable; dense coverage over motor and visual cortex | Dry comb electrodes | Yes (OpenBCI GUI, BrainFlow) | ~$3.0k complete ([OpenBCI](https://shop.openbci.com/products/the-complete-headset-eeg)) | **Strategic fit**: OpenBCI forum developers are a target customer. Dry combs on hair need fiddling. More DIY. |
| Neurosity Crown | 8: CP3, **C3**, F5, PO3, PO4, F6, **C4**, CP4 · dry | Dry | Via BrainFlow / LSL | ~$1.5k (per Neurosity) | Consumer look, fast setup; **no Cz**. |

**Recommendation:** buy the **Unicorn Hybrid Black** as the primary demo headset (reliable on a random person). Consider the **OpenBCI Ultracortex** as a second headset: "**same model, two different headsets, two random people, no calibration**" proves the hardware-agnostic claim on camera, and it speaks directly to our first community. Order this week: shipping plus setup plus pilot sessions need about 2 weeks before demo recording.

**Architecture link:** the headset fixes the model's input channels. BCIC-IV-2b uses only **C3, Cz, C4** and still reaches 78–83% left/right across people, so a 3-motor-channel headset is viable. The canonical montage will be chosen to match the demo headset(s).

## Demoing on a random person: risks and mitigations
- About 15–30% of people don't produce usable motor imagery at first ("BCI inefficiency"). A random volunteer might be one of them.
- **Mitigations:**
  - 20 seconds of coaching ("imagine squeezing a stress ball");
  - start with real hand squeezes, then switch to imagining them;
  - HOLD gating so the screen stays calm;
  - 2–3 volunteers lined up;
  - a robust second demo: **focus vs relax** (EEGMAT-style mental arithmetic vs rest), which works for most people;
  - the dataset-playback demo for the hard numbers.
- **Informal testing hygiene:** a written consent form, no names stored with recordings, adults only.

## Compute (founder: school cluster via Proxmox)
Fine for this project. Training EEGNet-scale models is light (CPU-feasible). The heavy part is **evaluation sweeps** (leave-one-subject-out / leave-one-dataset-out over ~380 subjects means hundreds of runs), which is where a GPU helps. To confirm with the cluster admins:
- **GPU model and VRAM**, and whether PCIe passthrough into the VM is set up.
- **Storage:** several core datasets are large (Stieger2021 and Lee2019 in particular). Check sizes before downloading; plan hundreds of GB.
- Docker/conda availability, job time limits, remote access (SSH).
- **Start downloading the agreed datasets now.** It's a long-lead task that doesn't depend on architecture decisions.
