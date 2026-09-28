# Brain2Music: Reconstructing Music from Human Brain Activity

**Denk, Takagi, Matsuyama, Agostinelli, Nakai, Frank, Nishimoto** (Google and academic collaborators) · arXiv 2307.11078 · 2023
Links: [arXiv 2307.11078](https://arxiv.org/abs/2307.11078) · [project page](https://google-research.github.io/seanet/brain2music/)
**Read:** full text (ar5iv HTML; the PDF exceeds the 10 MB fetch limit), 2026-09-28
**Relevance:** the template for "brain signal → frozen multimodal embedding → off-the-shelf generator". **fMRI, not EEG.**

## What they did
- **Data:** fMRI from **5 subjects** listening to **540 music clips** (15 s each, 10 GTZAN genres); 480 clips for training, 60 for test.
- **Target embeddings:** **MuLan** (128-d joint music–text embedding, high-level semantics), w2v-BERT (1024-d, low-level acoustics), and SoundStream tokens.
- **Decoding:**
  1. **Linear regression** from fMRI voxels to the embedding.
  2. Then either **retrieval** (nearest clip in the FMA music library) or **generation** with **MusicLM** conditioned on the predicted embedding.

## Numbers
- Identification accuracy for predicted MuLan embeddings: **87.6%**.
- Reconstructions keep **genre, instrumentation and mood**. Temporal structure (melody, rhythm) is poorly preserved.
- High-level semantic embeddings (MuLan) are predicted better than low-level acoustic ones.

## What it means for NeurOS
- **The architectural pattern is ours:** don't generate audio from brain data directly. Map brain data into a **frozen, pretrained embedding space** (MuLan / CLAP / CLIP), then reuse existing retrieval or generation models. This is how the "CLAP bridge" in the founder notes should work.
- **Semantic embeddings are the right target.** Coarse semantics are much more decodable than fine acoustic detail. The same should hold, even more so, for EEG.

## Caveats (why its numbers don't transfer)
- fMRI has millimetre spatial resolution and whole-brain coverage; scalp EEG has neither. EEG music decoding results are far weaker (see Qing et al. 2026 in [`../layer-1-references.md`](../layer-1-references.md): 48.7% 50-way identification, within-subject).
- The models are subject-specific (5 subjects). There's no zero-calibration component at all.

## Layer-1 references
Huang 2022 (MuLan), Agostinelli 2023 (MusicLM), Nakai 2022 (music-genre fMRI dataset). See [`../layer-1-references.md`](../layer-1-references.md).
