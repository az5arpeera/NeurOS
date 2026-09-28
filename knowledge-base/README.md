# NeurOS knowledge base

The single source of truth for what we know, what we've decided, and what's still open. Every document here is **living**: edit it in place whenever thinking changes. Git keeps every earlier version.

## Map

| Path | What it holds | Status |
|---|---|---|
| [`open-questions.md`](open-questions.md) | Step 2: questions that must be answered before the MVP is locked, each with a recommendation | **Needs founder answers** |
| [`decisions/decision-log.md`](decisions/decision-log.md) | Dated decisions (what, why, who), plus decisions still pending | Living |
| [`research/synthesis.md`](research/synthesis.md) | What the literature actually supports for zero-calibration, UDA and CLAP, with realistic numbers | Read this second |
| [`research/README.md`](research/README.md) | Reading list: every paper, how it was accessed, how deeply it was read | Living |
| [`research/papers/`](research/papers/) | One note per primary paper (method, numbers with protocol, meaning for NeurOS, caveats) | Living |
| [`research/layer-1-references.md`](research/layer-1-references.md) | Citations followed one layer out from the primary papers | Living |
| [`datasets.md`](datasets.md) | Candidate training/evaluation datasets with verified specs | Living |
| [`claims-audit.md`](claims-audit.md) | Claims from the founder notes checked against evidence, with safer wording | Living |
| [`glossary.md`](glossary.md) | Plain-language definitions (EA, LOSO, OTTA, DANN, CLAP, ITR, ...) | Living |
| [`founder-notes/`](founder-notes/README.md) | The founders' original working notes (27 Sep 2026) | Editable; first commit is byte-identical to the originals |

## House rules

1. **Every number carries its protocol.** Write "74.3% binary, leave-one-subject-out, 9 subjects", never just "74%". Numbers without a protocol (within-subject vs cross-subject vs cross-session, number of classes, number of subjects, whether unlabeled target data was used, offline vs online) don't go into the YC application.
2. **Say how deeply a source was read.** "full text", "targeted full text", "abstract only", or "known, not re-read". Don't cite an abstract-only source for a specific number.
3. **Decisions go in the decision log** with a date and a reason. Superseded decisions stay, marked as superseded.
4. **Edit freely.** Nothing is lost. To recover an older phrasing:
   ```bash
   git log -p -- knowledge-base/<file>.md
   ```
   Or use the **History** button on the file in GitHub.
