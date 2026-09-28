# Research

Start with **[`synthesis.md`](synthesis.md)**: the cross-cutting conclusions and realistic numbers. Then dip into the per-paper notes as needed.

## Primary reading list (founder-supplied)

| # | Paper | Access used | Depth | Note |
|---|---|---|---|---|
| 1 | King & Dehaene 2014, temporal generalization (TICS) | PMC author manuscript | Full | [note](papers/2014-king-dehaene-temporal-generalization.md) |
| 2 | Rousis et al. 2024, EEGNet + SPDNet (EUSIPCO). The IEEE link and "stamp" link are the same paper. | Free EURASIP PDF | Full | [note](papers/2024-rousis-eegnet-spdnet.md) |
| 3 | Xiao et al. 2025, Densing law of LLMs (Nature MI) | arXiv 2412.04315 (Nature redirected to login) | Full text available; summary-level use | [note](papers/2020-2025-scaling-laws-chinchilla-densing.md) |
| 4 | Meng et al. 2026, EAMCCA (BSPC) | ScienceDirect via UVA | Full | [note](papers/2026-meng-eamcca.md) |
| 5 | Lawhern et al. 2018, EEGNet (JNE) | arXiv | Targeted full | [note](papers/2018-lawhern-eegnet.md) |
| 6 | Li et al. 2025, ELIPformer (cross-task zero-calibration RSVP) | arXiv | Targeted full | [note](papers/2025-li-elipformer.md) |
| 7 | MOABB documentation (v1.7.1) | Web | Docs + dataset summary | [datasets](../datasets.md) |
| 8 | Sagayaraju 2026, EA + SWA calibration bottleneck | arXiv | Full | [note](papers/2026-sagayaraju-ea-swa-calibration-bottleneck.md) |
| 9 | PhysioNet EEGMAT (mental arithmetic) | Web | Dataset page | [datasets](../datasets.md) |
| 10 | PhysioNet EEGMMIDB (motor movement/imagery) | Web | Dataset page | [datasets](../datasets.md) |
| 11 | Wu 2025, Revisiting Euclidean Alignment | arXiv | Targeted full | [note](papers/2025-wu-revisiting-euclidean-alignment.md) |
| 12 | Jin et al. 2026, RUNet (IEEE TBME) | IEEE Xplore via UVA | Full | [note](papers/2026-jin-runet.md) |
| 13 | Kaplan et al. 2020, Scaling laws | arXiv | Full text available; summary-level use | [note](papers/2020-2025-scaling-laws-chinchilla-densing.md) |
| 14 | Dou et al. 2026, INCEPT foundation model | arXiv | Targeted full | [note](papers/2026-dou-incept-foundation-model.md) |
| 15 | Patel et al. 2025, graph-based calibration-free MI (TechRxiv) | PDF extracted in-browser | Full | [note](papers/2025-patel-plv-gat-calibration-free.md) |
| 16 | Hoffmann et al. 2022, Chinchilla | arXiv | Full text available; summary-level use | [note](papers/2020-2025-scaling-laws-chinchilla-densing.md) |
| 17 | Denk et al. 2023, Brain2Music | ar5iv HTML | Full | [note](papers/2023-denk-brain2music.md) |
| 18 | An et al. 2021, EEG headphones music attention (ICASSP) | Microsoft Research PDF | Targeted full | [note](papers/2021-an-eeg-headphones-music-attention.md) |

## Layer 1
See [`layer-1-references.md`](layer-1-references.md): 25 references across alignment/UDA, backbones, contrastive/CLAP decoding and datasets, each with its read depth.

## Reading backlog (high priority)
- [ ] Han, Wei & Faisal 2023: GNN transfer across heterogeneous electrode configurations (JNE). Directly on the "any headset" claim.
- [ ] Kobler et al. 2022: mine the full text for SPDDSMBN numbers and what target data it needs online.
- [ ] Banville et al. 2025: scaling laws for brain decoding. Needed before any "scales with data" claim.
- [ ] OpenMIIR and MAD-EEG dataset papers: imagined music and music attention (CLAP track).
- [ ] Neural-data privacy law status (US states, federal proposals): one paragraph for the application.

## How papers were accessed
- Open-access PDFs: fetched, then text extracted locally (`pdfplumber`) so full text could be searched.
- Paywalled (IEEE, ScienceDirect): read through the founder's UVA library session in Chrome.
- TechRxiv (bot-protected): the PDF was parsed inside the browser session.
- PDFs are **not** stored in this repo (copyright); notes and links only.
