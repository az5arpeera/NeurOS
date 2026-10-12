# Pretrained EEG models and toolkits: what we can reuse (12 Oct 2026)

**Why:** the founder shared a Gemini conversation and a list of links (REVE, LUNA, ST-EEGFormer, ZUNA, MIRepNet, ITSA, TorchEEG, Braindecode, Meta's neuroai suite). The question: do we have to train from scratch, or is there a faster, better starting point?

**Read 12 Oct:** each project's own page, model card, licence or paper (links below). Web content was treated as data. Numbers are the papers' own; no file was downloaded.

## In short
- **Nobody should pretrain a foundation model from scratch at our stage.** REVE used 60,000+ hours from 25,000 people. Our model is small (EEGNet) and trains on 339 people in under an hour on the GPU, so training isn't our bottleneck. What matters is the starting point and the evaluation.
- **The real shortcut is transfer learning.** Start from a pretrained encoder and fine-tune it on our 339 people, inside our existing pipeline (same folds, alignment, warm-up scoring, engine). Two candidates fit:
  - **REVE-base:** handles any electrode layout; commercial use allowed with conditions.
  - **MIRepNet:** built for motor imagery, small, MIT licence, preprocessed almost exactly as we do.
- **Expect a few points, not a leap.**
  - The best foundation models beat EEGNet by about 4 points on cross-person motor imagery ([INCEPT note](papers/2026-dou-incept-foundation-model.md)).
  - An ICLR 2026 benchmark found they "often fail to significantly outperform compact neural networks or even classical non-neural decoders in data-scarce scenarios" ([ICLR 2026](https://iclr.cc/virtual/2026/poster/10011450)).
  - Our own ceiling analysis says zero calibration is only 4–8 points below each person's own calibrated ceiling.
  - The one result in our favour: REVE's cross-person BCI IV-2a number (below).
- **Gemini's pipeline** (3-D electrode positions → Riemannian alignment → pretrained encoder → few-shot fine-tune) is already our architecture, apart from the encoder. We align per person (EA), have a guided quick calibration (Q30, D-040), and place channels by name. The only new piece is swapping EEGNet for a pretrained, position-aware encoder.

## Model by model

| Model | What it is | Licence (weights) | Evidence relevant to us | Fit |
|---|---|---|---|---|
| **REVE** (NeurIPS 2025) | 69M-parameter transformer; electrode **positions** as input, so any layout works; pretrained on 60,000+ h, 92 datasets, 25,000 people | **REVE Responsible Use License v1.0:** research **and commercial** use; fine-tuned copies may be shipped with the licence, notices and a "modified from REVE" label. Prohibits identifying people from EEG and inferring sensitive states "without a valid research, medical, legal, or other appropriate basis" | Cross-person splits (CBraMod's), full fine-tune, 4-class: **BCI IV-2a 64.0% vs EEGNet 44.8%**, PhysioNet-MI 64.8% vs 58.1%. No zero-shot results. Input 200 Hz, ≥ 1 s | **Best candidate.** Strongest published cross-person gain; could read every EPOC X site. 277 MB weights; latency on a laptop CPU needs measuring |
| **MIRepNet** (KBS 2026) | 5.2M-parameter motor-imagery model; masked-token plus supervised pretraining | **MIT** (weights on Hugging Face, 27 MB) | Pretrained on 330 people from 7 MOABB datasets, **4 of them ours** (PhysionetMI, Dreyer2023, Lee2019, Cho2017). **Same preprocessing as ours:** 8–30 Hz and Euclidean alignment, plus a channel template. Every result uses the target person's labels (30% of a session, 12–86 trials); **no zero-calibration result**. BNCI2014-004 82.4% vs EEGNet 73.6% | **Cheapest drop-in;** real-time on the laptop. **Also prior art** for "calibration saved" (see below) |
| **ST-EEGFormer** (ICLR 2026, KU Leuven) | ViT with masked-autoencoder pretraining; small/base/large (> 300M); built for **128 Hz** (our rate); up to 142 channels | Code MIT; **weights' licence not stated** (default: all rights reserved) | Its own paper is the "Are EEG foundation models worth it?" benchmark, a sceptical result. The NeurIPS 2025 win was **Challenge 1** (response time in an SSVEP task, children's data), not motor imagery | Not before the licence is clarified |
| **LUNA** (NeurIPS 2025, ETH) | 7M-parameter topology-agnostic encoder (learned queries compress any channel set) | **CC BY-ND 4.0:** modified (fine-tuned) weights can't be redistributed | Not checked in depth | **Poor fit:** shipping a fine-tuned copy in an SDK would be redistribution |
| **ZUNA** (Zyphra, Feb 2026) | 380M diffusion autoencoder: **denoises and reconstructs missing channels** from scalp coordinates; ~2M channel-hours | **Apache 2.0** (research-use disclaimer) | No classification results; beats spherical-spline interpolation, more so when > 75% of channels are dropped | **Not a decoder.** One EPOC X idea below. 1.5 GB |
| **ITSA** (NeurIPS 2025) | Not a network: a pre-alignment (per-person re-centring, rescaling, then a **supervised** rotation) with CSP + Riemannian features | Code "will be released"; not found | Leave-one-person-out gains on its own pipeline | We already do the re-centring (EA). The rescaling step is unsupervised and cheap to test. The rotation needs labels, so it belongs to calibration |

## Toolkits

| Toolkit | What it gives us | Notes |
|---|---|---|
| **Braindecode** 1.8.1 (BSD-3) | 65+ published architectures with one constructor, `from_pretrained` for foundation weights, every MOABB dataset, EEGDash's 700+ BIDS datasets | The quickest way to try other backbones (ShallowFBCSPNet, ATCNet, EEGNeX, Conformer) in a pooled setup. Our EEGNet note planned that ablation; it was never run. Needs mne ≥ 1.11, torchaudio and more: an optional extra, not a core dependency |
| **TorchEEG** | Implementations (EEGNet, ATCNet, Conformer, LaBraM and more) | No pretrained weights stated; overlaps Braindecode, which fits our MOABB/MNE stack better |
| **Meta neuroai** (MIT; NeuralSet, NeuralFetch, NeuralTrain, NeuralBench) | **NeuralBench:** 36 EEG tasks incl. motor imagery; compares REVE, LUNA, LaBraM, CBraMod, BIOT, BENDR. **NeuralFetch:** 144 studies from 12 repositories, incl. TUEG and a 203-person Muse set ("Interaxon2026Muse") | Later: a standard benchmark gives outside-checkable numbers (credibility after YC). Python 3.12+. NeuralFetch's dataset licences aren't shown; check each before use |

## What Gemini got wrong or overstated
- **"EEGNet completely fails across headsets."** Our E1 found a model that never saw a dataset's hardware did within a point of one trained on it (EEGNet + per-session alignment + named-channel placement). Fixed layouts are a limit only for channels outside the layout (the EPOC X's case).
- **"Works flawlessly on a 51st person."** No published zero-calibration motor-imagery result comes close. REVE's best cross-person number is 64% (4-class), after fine-tuning on that dataset's other people.
- **ST-EEGFormer "dominated" and "excels at generalizing across subjects."** It won Challenge 1 (a response-time regression), and its own paper is sceptical of foundation models when data is scarce.
- **ZUNA "integrates via the BrainAccess ecosystem."** BrainAccess's site doesn't mention it, and ZUNA is a denoiser/reconstructor, not a classifier.
- **MIRepNet's "< 30 trials".** True, but those are labelled trials from the new user: calibration, not zero calibration.
- **ITSA "natively compensates for channel mismatches."** Its key step needs labels, and the code isn't out.
- **A LUNA citation** (papers.cool/arxiv/2610.05503) couldn't be matched; LUNA is [arXiv 2510.22257](https://arxiv.org/abs/2510.22257).

## What we can pull in, in order of value per day
1. **A pretrained backbone (REVE-base, MIRepNet) fine-tuned in our pipeline.** One experiment, bar drafted: [A6](../plan/a6-pretrained-backbone.md). It decides whether the frozen model stays EEGNet.
2. **Prior art to cite and position against: MIRepNet.** It pools public MOABB motor-imagery data with Euclidean alignment and claims fast adaptation with < 30 trials per class. Our "calibration saved" story must name it. Our difference is a usable start with zero labels, the live engine and headset-agnostic placement. Added to the competitor/prior-art list.
3. **For the EPOC X** (after E5):
   - **(a)** REVE can read all 14 sites instead of zero-filling 12.
   - **(b)** ZUNA could reconstruct C3/Cz/C4 from the EPOC X's sites for mvp-v0. That's a virtual test on Yang2025 first. Reconstruction can invent a plausible but wrong hand-area signal, so it needs its own bar.
4. **Braindecode's model zoo** for the never-run backbone ablation (ShallowFBCSPNet, ATCNet, EEGNeX) on our pooled folds. Cheap once installed.
5. **ITSA's rescaling step** on top of EA: a CPU-only proxy check (TS+LR), minutes.
6. **After YC:** NeuralBench for outside-checkable numbers; NeuralFetch/EEGDash for more datasets (licences first).

## Sources
- REVE: [paper (NeurIPS 2025, arXiv 2510.21585)](https://arxiv.org/html/2510.21585v1), [model card](https://huggingface.co/brain-bzh/reve-base), [licence](https://huggingface.co/brain-bzh/reve-base/blob/main/LICENSE), [code](https://github.com/elouayas/reve_eeg)
- MIRepNet: [GitHub](https://github.com/staraink/MIRepNet), [paper (arXiv 2507.20254)](https://arxiv.org/html/2507.20254), weights `starself/MIRepNet`
- ST-EEGFormer: [GitHub](https://github.com/LiuyinYang1101/STEEGFormer), [ICLR 2026 paper](https://iclr.cc/virtual/2026/poster/10011450), [challenge write-up](https://www.vscentrum.be/post/scaling-eeg-foundation-models-on-vsc-an-iclr-2026-benchmark-and-a-neurips-2025-eeg-challenge-win), [challenge](https://arxiv.org/pdf/2506.19141)
- LUNA: [NeurIPS 2025](https://neurips.cc/virtual/2025/poster/115483), [weights card](https://huggingface.co/thorir/LUNA), [code](https://github.com/pulp-bio/biofoundation)
- ZUNA: [Zyphra](https://www.zyphra.com/post/zuna), weights `Zyphra/ZUNA`
- ITSA: [NeurIPS 2025](https://neurips.cc/virtual/2025/133882), [arXiv 2508.08216](https://papers.cool/arxiv/2508.08216)
- BrainAccess: [site](https://www.brainaccess.ai/) (8-ch MINI kit €900, 4-ch HALO €400, free SDK with LSL)
- Toolkits: [Braindecode](https://braindecode.org/stable/index.html), [TorchEEG models](https://torcheeg.readthedocs.io/en/latest/torcheeg.models.html), [Meta neuroai](https://github.com/facebookresearch/neuroai): [NeuralBench](https://facebookresearch.github.io/neuroai/neuralbench/index.html), [NeuralFetch](https://facebookresearch.github.io/neuroai/neuralfetch/index.html), [NeuralTrain](https://facebookresearch.github.io/neuroai/neuraltrain/index.html), [NeuralSet](https://facebookresearch.github.io/neuroai/neuralset/index.html)
