# NeurOS knowledge base

The single source of truth for what we know, what we've decided, and what's still open. Every document here is **living**: edit it in place whenever thinking changes. Git keeps every earlier version.

## Map

| Path | What it holds | Status |
|---|---|---|
| [`architecture/mvp-architecture.md`](architecture/mvp-architecture.md) | **Step 4: MVP architecture v1.** Claims, data contract, alignment math, model, online engine, evaluation protocol, repo layout. The coding guidelines. | **Signed off 3 Oct (D-022)**; code started the same day |
| [`plan/road-to-nov-1.md`](plan/road-to-nov-1.md) | Day-by-day plan to the 1 Nov deadline: engineering, customer discovery, YC application, cut list | Living |
| [`yc/application-draft-v1.md`](yc/application-draft-v1.md) | **YC W27 application, draft v1:** every answer drafted from the KB; development numbers marked ‡ for replacement after the freeze; founder placeholders in brackets | Draft for the founders (11 Oct) |
| [`plan/e6-state-channel.md`](plan/e6-state-channel.md) | E6's decision and pass bar: rest vs mental effort across people (EEGMAT), full cap and the Ganglion's four channels, with a drift control | Bar set 11 Oct; awaiting approval |
| [`plan/post-yc-research-agenda.md`](plan/post-yc-research-agenda.md) | After 1 Nov: reliable control for (nearly) everyone on 3–4 electrodes. Reliability-through-time numbers, the architecture to build (simulated heads, person encoder, labels from use) and 12 gated experiments | **Parked until after 1 Nov** |
| [`results/`](results/README.md) | **Results log:** every number with its protocol, N and the command that regenerates it. Development numbers until the 24 Oct freeze; headline numbers after | Living (first entry 3 Oct: Riemannian baselines) |
| [`business/`](business/customer-interview-script.md) | Customer discovery: the [interview script](business/customer-interview-script.md) (segments, questions, commitment ladder, note template, LOI outline) the [first 15 target contacts](business/target-contacts.md) with message drafts, [ready-to-adapt drafts for the first 10](business/first-10-messages.md), and the [competitor analysis](business/competitors.md) (Zander Labs, Synaptrix, Neuralink) | Drafts for the founders |
| [`open-questions.md`](open-questions.md) | Open questions, each with a recommendation; answered ones at the bottom | Two open (cluster, roles) |
| [`decisions/decision-log.md`](decisions/decision-log.md) | Dated decisions (what, why, who), plus decisions still pending | Living |
| [`research/synthesis.md`](research/synthesis.md) | What the literature actually supports for zero-calibration, UDA and CLAP, with realistic numbers | Background |
| [`explainers/`](explainers/) | Plain-language explanations: [UDA](explainers/uda-train-time-vs-test-time.md), [contrastive learning & CLAP](explainers/contrastive-learning-and-clap-for-generalization.md), [text-anchored head](explainers/text-anchored-head.md), [calibration time & commands](explainers/calibration-time-commands-and-metrics.md), [beyond LEFT/RIGHT](explainers/beyond-left-right-any-action.md), [demo hardware & compute](explainers/demo-hardware-lsl-and-compute.md), [post-MVP ideas](explainers/v2-ideas-riemannian-layers-pretraining-montage.md) | Living |
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
