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
