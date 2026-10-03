# Glossary

| Term | Plain-language meaning |
|---|---|
| **Channel / electrode** | One scalp sensor's voltage stream (relative to a reference). Positions follow the 10-20 / 10-10 system (e.g. C3/C4 sit over the left/right motor cortex). |
| **Montage** | The set and layout of electrodes a headset has. Different headsets have different montages. |
| **Motor imagery (MI)** | Imagining a movement (e.g. squeezing the left hand) without doing it. It changes mu (8–12 Hz) and beta (13–30 Hz) power over the motor cortex (event-related desynchronisation, ERD). |
| **Covariance matrix / SPD** | A C×C matrix of how every pair of channels co-varies in a time window. It's always symmetric positive-definite (SPD); these matrices live on a curved space (a Riemannian manifold). |
| **EA (Euclidean Alignment)** | Whitening each user's data by R̄^(−1/2), where R̄ is that user's mean covariance, so every user's average becomes the identity matrix. Label-free and cheap. |
| **RA (Riemannian Alignment / re-centring)** | Like EA but uses the Riemannian (geometric) mean of the covariance matrices. Slightly better in some online results. |
| **Tangent space / log map** | A way to "flatten" SPD matrices into ordinary vectors so standard classifiers can use them. |
| **Within-subject / cross-session / cross-subject / cross-dataset** | Train and test on the same person's same session / same person on different days / different people / different recording setups. Each is harder than the previous. |
| **LOSO** | Leave-one-subject-out: train on all subjects but one, test on the held-out one, repeat for each. The standard zero-calibration protocol. |
| **L0 / L1 / L2 zero-calibration** | Our house definitions: no target data at all / unlabeled data during normal use (no calibration screen) / a few labeled trials. See [synthesis §2](research/synthesis.md). |
| **Domain generalisation (DG)** | Training so the model works on unseen domains (new users) without seeing them. |
| **UDA (unsupervised domain adaptation)** | Adapting to a new domain using its *unlabeled* data. In a live product this happens at test time. |
| **OTTA (online test-time adaptation)** | Adapting the model on the fly to a user's incoming unlabeled data: alignment, updating batch-norm statistics, entropy minimisation. Source-free, so the training data isn't needed. |
| **DANN** | Domain-adversarial neural network: a gradient-reversal trick that makes features indistinguishable across subjects. |
| **Deep CORAL / MMD** | Losses that match feature statistics between domains (covariances / kernel mean embeddings). |
| **SWA** | Stochastic weight averaging: average the model weights over the last training epochs, for flatter, better-generalising minima. |
| **Contrastive learning / InfoNCE** | Training so matching pairs (e.g. an EEG window and its label text) have similar embeddings and non-matching pairs don't. |
| **CLIP / CLAP / MuLan** | Pretrained contrastive models with a shared embedding space for image↔text / audio↔text / music↔text. |
| **Stimulus-referenced decoding** | The system knows the candidate stimuli (images on screen, sounds in the ears, command list) and the EEG picks among them. Every working "zero-shot" EEG result is of this kind. |
| **ITR** | Information transfer rate (bits/min): combines accuracy, number of choices, and decision speed. The standard BCI throughput metric. |
| **BCI inefficiency ("illiteracy")** | The 15–30% of people who don't produce decodable motor imagery, at least at first. |
| **LSL (Lab Streaming Layer)** | The de-facto open protocol for streaming time-synced biosignals between devices and software. |
| **MOABB** | "Mother of All BCI Benchmarks": a Python library with standard loaders and evaluation protocols for public BCI datasets. |
| **Foundation model (EFM)** | A large model pretrained on many hours of unlabeled EEG, then adapted to tasks (e.g. CBraMod, INCEPT). |
| **Encoder / embedding / head** | The encoder turns an EEG window into a short list of numbers (the embedding). The head turns the embedding into a decision. |
| **Text-anchored head** | A head whose per-command reference vectors come from a frozen text model reading a description of each command ("imagine squeezing your left hand"), plus a small learned adapter. It's CLIP/CLAP-style, and it gives all datasets one shared vocabulary. See [explainer](explainers/text-anchored-head.md). |
| **SupCon (supervised contrastive loss)** | Pulls together embeddings with the same label. We make the positives come from *different people*, which teaches person-invariance. |
| **AdaBN / domain-specific BN** | Batch-norm statistics computed per person (domain). AdaBN swaps in the new user's statistics at test time. |
| **LODO** | Leave-one-dataset-out: train on all recording setups but one, then test on that one. Our "any setup" test. |
| **Virtual headset** | Public data cut down to a specific headset's channels and replayed as a live stream, so we can test a headset before owning it. |
| **Train–serve parity** | The same preprocessing code runs on training data and on the live stream, so the model never sees a mismatch. |
| **SSVEP** | Steady-state visually evoked potential: looking at a target flickering at a fixed frequency makes the visual cortex oscillate at that frequency. Decodable with no training (CCA). |
| **Weak supervision** | Labels that arrive for free from context (an accelerometer, app events) instead of from a person labelling data. |
| **Artifact** | Signal not from the brain: eye blinks (EOG), muscle activity (EMG), electrode or cable motion. |
| **C-core** | The model's fixed 26-channel input layout: the sensorimotor channels every core dataset recorded (F3 … P4). A headset fills the rows it has; the rest are zeros. See [architecture §4.3](architecture/mvp-architecture.md). |
| **Command set** | Which trained labels an app turns into commands. Default `LEFT / RIGHT / REST`; `FEET` and `BOTH_HANDS` are opt-in (D-024). |
| **Decoy label** | A label the model is trained on but the app hasn't enabled as a command. If it wins, NeurOS says `HOLD("unrecognized")` instead of guessing a command. The same idea as "unknown" in voice-assistant wake-word models. |
| **Locked test set** | The 79 people (20% per dataset) set aside on 3 Oct 2026 and not touched until the freeze on 24 Oct, so our published numbers come from people no decision was tuned on (D-026). |
| **Harmonisation** | Converting each dataset's own conventions (channel spellings, units, event names, cue timing) into one canonical format before anything else touches the data. |
