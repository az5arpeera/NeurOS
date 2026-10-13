# A6: does a pretrained encoder beat our from-scratch EEGNet?

**Written:** Mon 12 Oct 2026, before any code, download or run. **Status:** **approved 12 Oct with both downloads** ([D-043](../decisions/decision-log.md)); work starts 13 Oct.

**From:** the founder's question "do we really have to train from scratch?" and the [pretrained-model review](../research/pretrained-eeg-models-2026.md).

## The decision
Which model do we freeze on 24 Oct for the headline numbers and the demo: `mvp-v0` (EEGNet, trained from scratch) or a fine-tuned pretrained encoder?
- **Pass:**
  - bundle the winner as `mvp-v1`;
  - re-run the engine replay check (the [10 Oct replay](../results/2026-10-10-mvp-v0-replay.md) bar) on it;
  - freeze it if that passes by 22 Oct.
- **Fail:** `mvp-v0` stays. Pretrained encoders move to the post-YC agenda with these numbers attached.

## Hypothesis
- A pretrained encoder has seen far more brains than our 339 people, which should help on people it has never seen.
- REVE's published cross-person result is the reason to try: BCI IV-2a, 4-class, fine-tuned, 64.0% vs EEGNet 44.8%.
- The counter-evidence says expect less:
  - INCEPT: about +4 over EEGNet;
  - the ICLR 2026 benchmark: often no significant gain;
  - our own finding that zero calibration is only 4–8 points below each person's own calibrated ceiling.

## Candidates (licences checked 12 Oct; not legal advice)
1. **MIRepNet** (5.2M parameters, MIT).
   - Built for motor imagery; preprocessed like us (8–30 Hz, Euclidean alignment).
   - Fast enough for the laptop.
2. **REVE-base** (69M parameters, REVE Responsible Use License: commercial use and shipping fine-tuned copies allowed with notices).
   - Takes electrode positions, so every dataset uses all its channels instead of our 20-channel strip.
   - Needs 200 Hz input (our windows are resampled from 128 Hz).

## Data and protocol (E2's, so the comparison is paired)
- **Training:** E2's 7 training datasets (339 development people), E2's 5 folds and seed, E2's windows and per-session alignment.
- **The encoder:** each candidate's pretrained weights, fully fine-tuned on each fold's training people, with E2's early stopping on the fold's validation people.
- **Scored:** E2's 73 cue-free development people (Yang2025 41, BNCI2014-001 9, Shin2017A 23), after **120 s** of unlabeled streaming (E2's `e2_stream_120s`), with the identical windows.
- **Never touched:** the locked test set.

## Pass bar (set before any run), for each candidate
All must hold:
1. **Gain:** mean ≥ E2 + **3.0 points** over the 73 people, paired Wilcoxon p < 0.05 (two-sided).
2. **No dataset loses:** each of the three datasets' mean ≥ its E2 mean.
3. **Real-time:** p99 decision latency < **50 ms** on the founders' laptop CPU, the engine's existing bar.
4. **Shippable:** the licence allows a fine-tuned copy inside our SDK (both candidates do, per the 12 Oct reading).

If both pass, the larger gain wins; within 1 point of each other, the smaller model wins.

## Also reported (not part of the bar)
- The E5 channel sets for each candidate: the 3-channel Ganglion class and the EPOC class. For REVE, also the EPOC X's full 14 sites, which only a position-aware model can use.
- Training time and model size.

## What this can't show
- **One seed.** E2's seed noise was 0.2–2.0 points, which is why the bar is 3.
- **Cue-free people only.** The other datasets keep the class on screen.
- **Possible leakage.** REVE's pretraining corpus includes public motor-imagery datasets, possibly some of our development people (it lists Lee2019). Pretraining was unlabeled (masked reconstruction), so no class labels leaked, but the people aren't strictly unseen. MIRepNet's *supervised* pretraining used labels from PhysionetMI, Dreyer2023, Lee2019 and Cho2017, all training-only for us; none of our 73 scored people (Yang2025, BNCI2014-001, Shin2017A) are in its pretraining list. The result writes this up either way.

## Code review and implementation (13 Oct, before any run)
**Downloads** (all checksums match the published ones; in the git-ignored `data/pretrained/`):
- REVE-base weights at revision `dc2a075`.
- The REVE position bank at `befa5b5`.
- MIRepNet weights at `9bac043`.
- Source to read: MIRepNet at `b35d113`, REVE's official repository at `06a7059` (MIT).

**What REVE's code does:**
- **Patches:** each channel is cut into 1-s patches (200 samples at 200 Hz, 0.1-s overlap). A 2-s window gives 2 patches per channel.
- **Positions:** each patch gets a 4-D position embedding: Fourier features of the electrode's x, y, z (metres) and the patch's time.
  - REVE's position bank is identical to MNE's standard 10-05 montage (148 of 148 names to 1e-8 m). So we ship our own table generated from MNE (`src/neuros/models/positions_1005.json`) and don't depend on theirs.
- **Transformer:** a 22-layer transformer (RMS norm, gated GELU, no biases) over all channel × patch tokens. There's no final norm.
- **Size:** 69,189,632 parameters, encoder only. No decoder was published, so the masked-reconstruction check isn't possible.
- **Dependencies:** `transformers`, `einops` and optionally `flash_attn`. Our port needs only torch.
- **The authors' imagined-movement recipe** (BCI IV-2a): CAR, 0.3–50 Hz, 0–4 s after the cue (the class is on screen), µV/100, resampled to 200 Hz.
  - **Their split:** 5 training people, 2 validation, **2 test** (A08, A09). So REVE's headline 64.0% is cross-person, but on two people, with the cue on screen. That's weaker evidence than it looked.
  - **Pretraining:** z-scored, clipped at 15 SD, 0.5–99.5 Hz.

**Our implementation** (`src/neuros/models/pretrained.py`; no downloaded code is imported or run):
- **Weights:** both models are re-implemented in torch; only the published weights are loaded, strictly (every name and shape must match).
  - REVE's come from safetensors, read as a JSON header plus raw tensors.
  - MIRepNet's come from a pickle opened with `weights_only=True`.
- **REVE:**
  - an attention-pooled linear head (the authors' "last" pooling), so it doesn't depend on the channel count;
  - **absent channels are left out of attention.** A test proves a zero-filled channel gives exactly the output of a layout without it.
- **MIRepNet:** absent channels are filled by its own inverse-distance interpolation onto its 45-channel template, after alignment, as its pipeline does.
- **Resampling:** both resample internally (FFT) from our 128 Hz.
- **What differs from pretraining:** our windows are 8–30 Hz and Euclidean-aligned (roughly unit variance, like z-scoring). REVE was pretrained on broadband signals, so this is a shift that fine-tuning has to absorb. It keeps the windows identical to E2's, which the paired comparison needs.
- **Latency on the founders' laptop** (16 threads, one 2-s window): MIRepNet p99 8.9 ms, REVE p99 33.8 ms, EEGNet 1.1 ms. **Both pass the < 50 ms bar** (`scripts/analysis/a6_latency.py`).

**The fine-tuning recipes** (inside E2's: AdamW, label smoothing 0.1, SWA, early stopping on validation people):
- **MIRepNet** (`configs/a6_mirepnet.yaml`): its own `finetune.py`, all weights at lr 1e-3; up to 40 epochs, patience 10.
- **REVE** (`configs/a6_reve.yaml`): the authors' approach.
  - The encoder is frozen for 2 epochs (a linear probe), then fine-tuned at 0.1× the head's 1e-3.
  - Dropout 0.5, weight decay 0.01, up to 15 epochs, patience 5.
  - Their mixup and LoRA aren't used.

**Port check (engineering, before the A6 runs).** On a 10-person-per-dataset subset, with the encoder frozen throughout (a linear probe), the pretrained REVE must beat the same architecture with **random weights** by at least 5 points of validation accuracy. If it doesn't, the port is suspect and A6 stops until it's found.

## Cost
- **Downloads (permission needed; sizes from Hugging Face on 12 Oct):**
  - `MIRepNet.pth`, 27.0 MB, from huggingface.co/starself/MIRepNet (MIT), plus its model code from github.com/staraink/MIRepNet.
  - `model.safetensors`, 276.8 MB, plus model code, from huggingface.co/brain-bzh/reve-base. The position bank (`positions.json`, 60 KB) is from huggingface.co/brain-bzh/reve-positions.
  - Possibly the `transformers` package if REVE's model code needs it (checked before asking).
- **Code:** about a day. A backbone option in the pooled trainer (`neuros.eval.pooled`), resampling for REVE, a latency check, and a bundle/decoder path in the engine.
- **GPU:** about 1 h for MIRepNet and about 2–3 h for REVE on AWS (5 folds each).
- **Time:** to make the freeze, code by 15 Oct, runs by 17 Oct, replay check and bundle by 20 Oct.
