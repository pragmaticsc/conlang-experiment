# conlang-experiment

**Does morphological regularity improve LLM pre-training efficiency? A controlled comparison on English vs. Esperanto Wikipedia.**

## The Question

Language models are trained on text written for humans. Human languages evolved under constraints unrelated to machine learning — irregular morphology, ambiguous syntax, high-frequency polysemy. Does a more *regular* language reduce the capacity cost of modeling it?

Wang & Wen (2025, [arXiv:2502.04488](https://arxiv.org/abs/2502.04488)) argue from theory that a morphologically regular language should reduce training overhead, and explicitly leave empirical validation as future work. This project provides that validation.

Esperanto provides a natural test: it is a real language with real speakers and native content (eo.wikipedia.org, ~370K articles), but was designed to be morphologically regular and agglutinative. If regularity matters for modeling efficiency, Esperanto should yield lower validation bits-per-byte (val_bpb) at equivalent compute and equivalent model architecture.

## The Experiment

2-cell × 2-size factorial on Wikipedia corpora:

```
                  English           Esperanto
                  (Simple Wiki)     (eo.wikipedia)
              ┌──────────────────┬──────────────────┐
  ~10M params │    3 seeds       │    3 seeds       │
              ├──────────────────┼──────────────────┤
  ~30M params │    3 seeds       │    3 seeds       │
              └──────────────────┴──────────────────┘
```

Plus a **byte-level baseline** for each language to isolate tokenization effects from language-structure effects.

**Primary metric:** `val_bpb` (validation bits-per-byte on held-out articles).

**Success criterion:** Pre-registered in `docs/preregistration.md` before the first training run.

## Status

| Phase | Status |
|-------|--------|
| Project scaffolding | ✅ |
| Pre-registration | ⏳ |
| Data download | ⏳ |
| Tokenizer training | ⏳ |
| Model training runs | ⏳ |
| Analysis and write-up | ⏳ |

## Why This Setup

- **Real corpora for both languages** — no machine translation, no "garbage in, garbage out" confound
- **Native Esperanto content** — human-written, not simplified or translated
- **Byte-level baseline** — separates tokenization efficiency from language-structure efficiency
- **3 seeds per cell** — small-model training is noisy; CIs are mandatory
- **Pre-registered success criterion** — no goal-post shifting post hoc

## Quickstart

```bash
pip install -e ".[train,dev]"

# 1. Download both corpora (~2GB total)
bash data/download-english.sh
bash data/download-esperanto.sh

# 2. Train BPE tokenizers and compare
python -m tokenizer.tokenizer_train train --vocab-size 8192

# 3. Clone autoresearch-mlx for training runs
git clone https://github.com/trevin-creator/autoresearch-mlx train/autoresearch
# Follow train/program.md

# 4. Analyze results
python -m eval.benchmark summary
```

## Hardware

Developed and intended for Apple M4 Max (64GB). MLX-native training via autoresearch-mlx. No GPU required.
