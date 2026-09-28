# Scaling laws, compute-optimal training, and the densing law (three LLM papers)

| Paper | Authors · venue · year | Link | Read |
|---|---|---|---|
| Scaling Laws for Neural Language Models | Kaplan et al. (OpenAI) · arXiv · 2020 | [2001.08361](https://arxiv.org/abs/2001.08361) | full text available; key results known |
| Training Compute-Optimal Large Language Models ("Chinchilla") | Hoffmann et al. (DeepMind) · NeurIPS · 2022 | [2203.15556](https://arxiv.org/abs/2203.15556) | full text available; key results known |
| Densing Law of LLMs | Xiao et al. (Tsinghua / OpenBMB) · *Nature Machine Intelligence* · 2025 | [Nature](https://www.nature.com/articles/s42256-025-01137-0) · [arXiv 2412.04315](https://arxiv.org/abs/2412.04315) | arXiv version (Nature redirected to login) |

**Relevance:** narrative and strategy ("why now", "why small models", "why data is the moat"). **These are not EEG results**, so don't cite them as evidence about brain decoding.

## What each says
- **Kaplan 2020:** language-model loss falls as a smooth power law in parameters N, data D and compute C, over many orders of magnitude. Bigger models are more sample-efficient.
- **Chinchilla 2022:** for a fixed compute budget, parameters and training tokens should grow **together** (about 20 tokens per parameter). Many large models were over-sized and under-trained.
- **Densing law 2025:** define *capability density* = effective parameter size ÷ actual parameter size. Density of open models has grown exponentially, doubling roughly every **3 months** in the arXiv version. The same capability keeps getting cheaper to run.

## What it means for NeurOS
1. **EEG decoding is data-bound, not compute-bound.** All public EEG together is about 10⁴ hours, and labelled motor imagery is a small fraction of that. Chinchilla logic says a model should be sized to its data. That supports **small models (10³–10⁶ parameters) plus strong priors** (alignment, covariance geometry) for the MVP, and argues against "we'll scale a giant model".
2. **The empirical EEG evidence agrees.** INCEPT (11,000 h of pretraining) only buys about 4 points over EEGNet on cross-subject motor imagery. See [`2026-dou-incept-foundation-model.md`](2026-dou-incept-foundation-model.md).
3. **"Why now" (densing):** capability per parameter keeps rising, so decoders and embedding models that need a GPU today will run on a headset or phone chip soon. That supports an **edge SDK** business, not a cloud-only API.
4. **The moat is data and evaluation, not parameters.** Whoever sees the most diverse real-world headsets and users (with consent) can build the best alignment and adaptation. That's the data-flywheel argument, stated honestly.

## Caveats
- Don't write "scaling laws show brain decoding will improve with scale" unless citing EEG-specific scaling work. A candidate is Banville et al. 2025 on scaling laws for brain decoding (not yet read; see [`../layer-1-references.md`](../layer-1-references.md)).
