# Demo hardware, Muse, LSL, and compute

*Answers open questions Q5 and Q6 (founder asked 2026-10-01). Updated 2026-10-02 for the founder's **$1,000–1,200 cap** (D-020). Prices are vendor list prices checked on 2026-10-02; **verify at checkout**.*

## What's Muse, and why it's excluded
Muse (by Interaxon) is a consumer meditation headband with **4 dry electrodes**: two on the forehead (AF7, AF8) and two behind the ears (TP9, TP10). It's good for relaxation and focus feedback. But it has **nothing over the motor cortex** (C3 / Cz / C4, on top of the head), so it can't detect motor imagery.

## Does LSL cost anything?
**No.** Lab Streaming Layer is **free and open-source** (MIT licence), maintained by the open-source community that originated at UCSD/SCCN. It's the standard way EEG devices stream time-stamped data to software, through the `pylsl` Python package. Most research headsets ship an LSL app or work through BrainFlow.

## Headset shortlist under the $1,000–1,200 cap

| Option | Channels | Electrodes | Price | Within the cap? | Notes |
|---|---|---|---|---|---|
| **OpenBCI Ganglion kit**: Ganglion board $624.99 + required dongle $19.99 + 10 gold-cup electrodes $44.99 + Ten20 paste $24.99 | 4 at 200 Hz. You place them: C3, Cz, C4 + one spare (Pz/Oz) | **Wet** (paste) | **≈ $715** + tax/shipping | ✅ ≈ $450 headroom | C3/Cz/C4 is the BCIC-IV-2b montage (78–83% left/right across people in the literature). Wet contact matches our training data. BrainFlow + LSL (OpenBCI GUI). It's our first-user community (D-015), and it ships from the US. **Cons:** only 4 channels, paste in hair, ~10 min setup, hobby-grade amplifier. |
| **BrainAccess MINI kit** (Neurotechnology, Lithuania) | 8 at 250/500 Hz, 24-bit; the standard layout includes C3 and C4 | **Dry** gold-plated spikes in a cap | **€900 excl. VAT** (≈ $1,000–1,100) + shipping/import duty | ✅ tight | Fast setup with no paste, 8 channels, accelerometer. Free software suite with LSL and a Python SDK. **Cons:** dry contact is noisier and varies with hair; ships from the EU; we found little independent validation. |
| OpenBCI Cyton + gold cups + paste | 8 at 250 Hz | Wet | ≈ $1,320 | ❌ ~10% over | The research-community standard, if you can stretch. |
| Neurosity Crown | 8 (CP3, C3, F5, PO3, PO4, F6, C4, CP4) | Dry | $1,499 | ❌ | No Cz. |
| g.tec Unicorn Hybrid Black | 8 (Fz, C3, Cz, C4, Pz, PO7, Oz, PO8); IMU | Dry or gel | $1,518 | ❌ | Still the best technical fit. **Ask g.tec about a student/startup discount**; if it lands under the cap, prefer it. |
| Muse, Emotiv Insight | No motor-cortex channels | — | — | ❌ | Can't do motor imagery. |

## Recommendation: the OpenBCI Ganglion kit, ordered today
1. **Wet electrodes match our training data.** Every public dataset we train on was recorded with gel electrodes. Dry electrodes would add noise *and* a hardware shift on top of the person-to-person shift we're already fighting.
2. **C3 / Cz / C4 is a proven zero-calibration montage** (BCIC-IV-2b).
3. **OpenBCI users are our first customers** (D-015). A demo on their hardware speaks their language.
4. **It's the cheapest and fastest to arrive.** It leaves ≈ $450 for spares: electrodes, skin-prep gel, a headband to hold the cups.

**Second-hardware proof without spending:** ask a UVA EEG lab to run NeurOS on their research system for an afternoon.
- It doubles as customer discovery (D-015).
- It makes "two different headsets, same model, no calibration" possible on camera.
- Until any hardware arrives, **virtual headsets** cover the hardware-agnostic claim: public data cut down to a headset's channels ([architecture §6](../architecture/mvp-architecture.md)).

**Architecture link:** the headset fixes the input channels. BCIC-IV-2b uses only **C3, Cz, C4** and still reaches 78–83% left/right across people, so a 3–4-channel motor setup is viable.

## Demoing on a random person: risks and mitigations
- About 15–30% of people don't produce usable motor imagery at first ("BCI inefficiency"). A random volunteer might be one of them.
- **Mitigations:**
  - 20 seconds of coaching ("imagine squeezing a stress ball");
  - start with real hand squeezes, then switch to imagining them;
  - HOLD gating, plus the motion/artifact detector, so the screen stays calm;
  - 2–3 volunteers lined up;
  - a robust second demo: **focus vs relax** (EEGMAT-style mental arithmetic vs rest). It needs one electrode over Pz/Oz, which the Ganglion's spare channel covers;
  - optional SSVEP "demo insurance" ([beyond LEFT/RIGHT](beyond-left-right-any-action.md));
  - the dataset-playback demo for the hard numbers.
- **Informal testing hygiene:** a written consent form, no names stored with recordings, adults only.

## Compute (founder: school cluster via Proxmox)
Fine for this project. Training EEGNet-scale models is light (tens of thousands of parameters; CPU-feasible). The heavy part is **evaluation sweeps**: subject-wise K-fold plus leave-one-dataset-out runs over ~380 people, repeated for every experiment. A GPU helps there.

**What to ask the cluster admins:**
- **GPU:** any NVIDIA card with **≥ 8 GB VRAM** is plenty. Is PCIe passthrough into the VM set up?
- **Storage: ask for ~1 TB.**
  - Stieger2021 alone is **377 GB** (62 people, 598 sessions); we'll start with its first 1–2 sessions per person.
  - The full Lee2019 OpenBMI release is **209 GB** across three paradigms; we need only the motor-imagery part.
  - The rest of the pool is a few GB per dataset.
- **Access and limits:** SSH access, Docker or conda, and job time limits.

**Coding starts Saturday on a laptop regardless.** Days 1–3 use the small datasets (BNCI2014-001/004, PhysionetMI). The cluster only matters for the full sweeps from week 2.
