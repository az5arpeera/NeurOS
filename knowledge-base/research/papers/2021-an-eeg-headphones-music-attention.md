# Decoding Music Attention from "EEG Headphones": A User-Friendly Auditory Brain-Computer Interface

**An, Shinn-Cunningham, Gamper, Emmanouilidou, Johnston, Jalobeanu, Cutrell, Wilson, Chiang, Tashev** (CMU, UCSD, Microsoft Research) · ICASSP 2021
Links: [Microsoft Research page](https://www.microsoft.com/en-us/research/publication/decoding-music-attention-from-eeg-headphones-a-user-friendly-auditory-brain-computer-interface-2/) · [PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2021/07/ICASSP2021_Music_BCI.pdf) · DOI 10.1109/ICASSP39728.2021.9414492
**Read:** targeted full text (hardware, participants, method, results), 2026-09-28
**Relevance:** proof that a **headphone form factor** can run an auditory BCI. It's the consumer path the founder notes bet on (Neurable-style headphones).

## Setup
- **Hardware:** "Smartfones", a saline-based EEG device built into headphones, with **11 sensors** (3 on top of the head, 4 around each ear).
- **Participants:** 9 adults; approved by the Microsoft Research IRB.
- **Task:** a polyphonic piece with three instruments (vibraphone, piano, harmonica), each playing a melody adapted from a different pop song, from different spatial directions. The user attends to one instrument.
- **Decoder:** auditory attention decoding (AAD). A linear backward model reconstructs the attended audio envelope from EEG (2–8 Hz band, downsampled to 64 Hz). The reconstruction is correlated with each instrument's envelope. **Window: 8 s.** Models are **subject-specific** (leave-one-out cross-validation within each subject).

## Numbers
- Mean decoding accuracy **63.77%** without feature selection; **71.23%** with segment-based feature selection.
- Information transfer rate about **1.01 bits/min**, versus ≤ 0.50 bits/min in earlier music-BCI work.

## What it means for NeurOS
- **Headphones can work, but slowly.** About 70% with 8-second decisions is a low-bandwidth channel. It's fine for "which stream am I focused on?" or "skip track", not for continuous control.
- **This is a natural CLAP demo.** Replace the hand-crafted envelope with **CLAP audio embeddings** of each candidate stream, and match the EEG embedding against them. That's the ELIPformer pattern (stimulus-referenced) applied to audio, and it's a clear upgrade story.
- **Zero-calibration is the gap.** These decoders are per-user. A cross-subject AAD or CLAP decoder with alignment would be a genuine NeurOS contribution.

## Caveats
- 9 participants; attention compliance isn't verified behaviourally for every trial.
- Saline electrodes aren't dry electrodes. Consumer dry-electrode signal quality will be worse.

## Layer-1 references
O'Sullivan 2015 (AAD), Treder 2014 (music oddball BCI). See [`../layer-1-references.md`](../layer-1-references.md).
