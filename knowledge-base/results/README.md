# Results log

Every number we compute, with its protocol, in the order we computed it.

**Two kinds of numbers live here:**
- **Development numbers** (everything until the freeze on Sat 24 Oct): computed on development people only, never on the [locked test set](../architecture/mvp-architecture.md#81-splits). They guide decisions. They are **not** headline numbers.
- **Headline numbers** (after the freeze): the frozen recipe on the locked test set, the 10-fold run over all ~380 people, and leave-one-dataset-out. Only these go into the YC application, after a check against the [claims audit](../claims-audit.md).

**Rules:**
- Every entry states the protocol, N (people) and the command that regenerates it.
- The raw result file (`results/…json`, git-ignored) holds per-person scores and provenance: git SHA, config hash and package versions.
- Failed and disappointing runs are logged too (architecture §8.3).

| Date | Entry | What it shows |
|---|---|---|
| 3 Oct 2026 | [Riemannian baselines on BNCI2014-001](2026-10-03-bnci2014-001-riemannian-baselines.md) | The first table rows for E0. Cross-person left/right: **65–69%**. Alignment adds about 2 points to tangent space and about 11 points to MDM. Within-person sanity check: 73–77% |
| 3 Oct 2026 | [E0: EEGNet on BNCI2014-001](2026-10-03-e0-eegnet-bnci2014-001.md) | **Pipeline check passes** (EEGNet + EA 75.0% vs the literature's 74.3% ± 3; chance before the cue). **But 75.0% is not a motor-imagery number** (4 Oct): the arrow is still on screen in our window, and the cue alone decodes at 75.8%. After the arrow has gone, zero-calibration EEGNet scores **60.6%**: level with Riemannian TS+LR + EA (60.4%), and **11.6 points below a calibrated model** (72.2%, p = 0.012). The 8.1-point deep-model lead and the "on par with calibrated" observation are retracted; so is the 4-s window's 83.7%, which falls to 72.3% without the arrow's first half-second |
