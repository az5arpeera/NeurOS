# What is a text-anchored head?

*Answers the founder's question on P-1 (2026-10-02): "What is a text-anchored head? Educate me so I can make a firm decision." Background: [contrastive learning & CLAP](contrastive-learning-and-clap-for-generalization.md).*

## 1. Every decoder has two halves
```
EEG window ──► ENCODER ──► embedding z ──► HEAD ──► one score per command ──► LEFT / RIGHT / REST / HOLD
(e.g. 8 ch × 2 s)  (learns what       (~64–128 numbers:   (compares z with one
                    matters in EEG)    a summary of the     reference vector per
                                       window)              command; picks the closest)
```
- The **encoder** (EEGNet-style network) does the hard work. It turns a noisy window into a compact summary `z`.
- The **head** is the last, small step. It holds **one reference vector per command** and scores how close `z` is to each one.

The whole question is **where the head's reference vectors come from.**

## 2. A normal head: reference vectors learned from scratch
```
score_k = w_k · z + b_k          p(command k | window) = softmax(scores)_k
```
- `w_LEFT`, `w_RIGHT` and `w_REST` are whatever numbers the optimiser found. They mean nothing outside this one model.
- They're tied to **one label list**:
  - PhysioNet's "left fist", BNCI's "left hand" and Stieger's "left" are three unrelated slots unless a human hand-maps them.
  - A dataset with a new label ("feet", "mental arithmetic") needs a new slot and retraining.

## 3. A text-anchored head: reference vectors computed from the *meaning* of each command
1. **Describe** each command in words:
   - "imagine squeezing your left hand"
   - "imagine squeezing your right hand"
   - "relax with your eyes open"
   - "count backwards by sevens in your head"
2. **Embed** each description with a frozen, pretrained text model. This is an open sentence-embedding model, such as `all-MiniLM-L6-v2` (Apache-2.0). It turns each description into a vector `t_k`. "Frozen" means we never change it; it's a fixed dictionary.
3. **Adapt.** A small learned adapter `P` maps the text vectors into the EEG embedding space: anchor `a_k = P · t_k`.
4. **Score** by the similarity between the EEG embedding and each anchor:
   ```
   score_k = cos(z, a_k) / τ          p(command k | window) = softmax(scores)_k        (τ = temperature)
   ```
5. **Train.** Each EEG embedding is pushed toward the anchor of its true command and away from the others. That's the CLIP/CLAP contrastive objective with EEG in place of images or audio. With a fixed command list it reduces to cross-entropy over the cosine scores.

**At run time** the anchors are a small table of pre-computed numbers, and the text model isn't loaded. The runtime cost is zero.

### Analogy
A normal head is a multiple-choice exam where every dataset uses its own answer key: "B" means *left fist* in one dataset and *feet* in another. A text-anchored head writes the answers out in words, so every dataset, every person and every app shares **one dictionary**.

### Why the adapter `P` is needed
- **The mismatch:** a text model sees "imagine squeezing your **left** hand" and "…your **right** hand" as nearly the same sentence, because only one word differs. The brain treats them as opposites: they drive opposite hemispheres.
- **The fix:** the adapter lets training push those two anchors apart while keeping "left fist" and "left hand" together.

This is standard practice in CLIP-style models.

## 4. What it buys NeurOS

| Benefit | Why it matters |
|---|---|
| **Merges datasets with different label names** | "Left fist" (PhysioNet), "left hand" (BNCI, Cho, Lee, Dreyer) and "left" (Stieger) land on one anchor, so all ~380 people train one model. More people in training is our biggest generalization lever. |
| **One model, several paradigms** | Motor imagery (LEFT / RIGHT / REST) and brain state (rest vs mental arithmetic, EEGMAT) share one vocabulary and one head. |
| **Commands by name, any subset** | An app asks for `["left", "right"]` or `["rest", "focus"]`. The head scores only those, with no retraining. That's the "neural OS" API. |
| **Labels carry protocol detail** | "Rest, eyes closed" and "rest, eyes open" are different (the alpha rhythm differs a lot) but related. Integer labels can't express that; text can. |
| **An honest "CLAP" in the pitch** | It *is* the CLIP/CLAP recipe applied to intents. It's also the bridge to audio and image embeddings later. |

## 5. What it does *not* buy
- **No accuracy jump on its own.** With 2–4 commands, a text-anchored head behaves almost like a normal head with fixed directions. Expect about the same accuracy (±1–2 points). The cross-person gains come from:
  - merging data;
  - the **cross-subject contrastive loss**;
  - alignment;
  - test-time adaptation.
- **No reliable zero-shot to unseen actions.** The text model knows "imagine kicking a ball", but the EEG encoder has never seen kicking. The head would map it to the nearest trained anchor, probably *feet*. We'll *test* this honestly (experiment E4-zs in the [architecture](../architecture/mvp-architecture.md): hold out one label), but we won't claim it.
- **No new information.** If the EEG doesn't carry a distinction, no label text can create it.

## 6. Prior art: position it honestly
- **CLIP / CLAP / MuLan** are the original image-text, audio-text and music-text versions.
- **LEAF** (Jiang et al., [arXiv 2509.24302](https://arxiv.org/abs/2509.24302), Sep 2025, revised Mar 2026) is a language-aligned EEG foundation model. It aligns EEG with **textual label embeddings** and reports state-of-the-art results on 12 of 16 downstream datasets, motor imagery included. *Read depth: abstract only.* The abstract doesn't isolate how much the text alignment itself adds across people.
- **Conclusion:** text anchoring is a **credible, mainstream design, not a moat on its own**. Our edge is the zero-calibration runtime around it:
  - alignment and adaptation within 45 s;
  - any headset;
  - the SDK;
  - evaluation on hundreds of people.

## 7. Cost, risk, decision rule
- **Cost:** about 2–3 days on top of the normal head (anchor table, adapter, loss, tests).
- **Risk:** low. We train the normal head anyway as the baseline, so the comparison costs nothing (experiment E4).
- **Proposed decision rule:** adopt the text-anchored head as the MVP's CLAP layer **if it scores within 1 point of the normal head, or better, on never-seen people.** If it's worse, ship the normal head and keep the text table only as the naming/vocabulary layer.

## 8. Recommendation
**Adopt it under the rule above**, together with the cross-subject contrastive loss, which is the generalization engine:
```
total loss = L_text-anchored (CLAP-style)  +  λ · L_cross-subject-contrastive   [ + μ · L_DANN / L_CORAL : side experiments only, D-019 ]
```
