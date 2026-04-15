# Autoresearch Program: English vs. Esperanto Pre-Training Efficiency

## Experiment Objective

Testing a single hypothesis via a 2-language × 2-size × 3-seed factorial, with a byte-level ablation:

**H1**: A transformer pre-trained on Esperanto Wikipedia achieves lower `val_bpb` than an architecturally-identical transformer pre-trained on Simple English Wikipedia, with non-overlapping 95% CIs across 3 seeds at ≥5% reduction, at both 10M and 30M parameter scales.

**Null (H0)**: CIs overlap, or point-estimate reduction < 5%.

**Primary metric**: `val_bpb` (validation bits-per-byte, UTF-8 byte-normalized — language-invariant and tokenization-invariant).

Pre-registration: see `docs/preregistration.md`. **Success criterion is fixed and not adjustable after training begins.**

---

## Experimental Setup

### Primary matrix (BPE, vocab 8192)

| Run ID              | Corpus         | Size   | Seed |
|---------------------|----------------|--------|------|
| `EN-10M-s1/s2/s3`   | English        | ~10M   | 1/2/3 |
| `EO-10M-s1/s2/s3`   | Esperanto      | ~10M   | 1/2/3 |
| `EN-30M-s1/s2/s3`   | English        | ~30M   | 1/2/3 |
| `EO-30M-s1/s2/s3`   | Esperanto      | ~30M   | 1/2/3 |

12 primary runs.

### Ablation (byte-level, no learned vocabulary)

| Run ID                   | Corpus         | Size   | Seed |
|--------------------------|----------------|--------|------|
| `EN-byte-10M-s1/s2/s3`   | English        | ~10M   | 1/2/3 |
| `EO-byte-10M-s1/s2/s3`   | Esperanto      | ~10M   | 1/2/3 |
| `EN-byte-30M-s1/s2/s3`   | English        | ~30M   | 1/2/3 |
| `EO-byte-30M-s1/s2/s3`   | Esperanto      | ~30M   | 1/2/3 |

12 ablation runs. Total: **24 training runs.**

### Matched conditions (identical across every run except language and size tier)

- Architecture (layers, heads, d_model, ffn_dim) fixed per size tier
- Optimizer (AdamW), LR schedule (cosine decay), warmup
- Total training token budget (whichever corpus is smaller binds both)
- BPE vocab size 8,192 for primary; raw UTF-8 bytes for ablation
- Validation split: held-out 5% of articles, `seed=42` RNG call used for both corpora
- Context length, batch size (largest that fits 64GB unified memory)

---

## What autoresearch Should Explore

**Tune once on English 10M, apply to all other cells.** Do not re-tune per language — that would confound the experiment.

Tune exclusively on `train.py`. Do NOT modify `prepare.py`.

Axes to explore (on the English 10M tuning cell only):

### 1. Architecture (once, then lock)
- `n_layer`: 4, 6, 8
- `n_head`: 4, 6, 8
- `n_embd`: 128, 256, 384
- `block_size`: 256, 512
- Goal: pick the config closest to 10M params that achieves lowest val_bpb

### 2. Optimizer and LR schedule (once, then lock)
- `learning_rate`: 1e-4, 3e-4, 6e-4, 1e-3
- `lr_decay`: cosine
- `warmup_iters`: 50, 100, 200
- `weight_decay`: 0.0, 0.01, 0.1

### 3. Regularization (once, then lock)
- `dropout`: 0.0, 0.1
- Gradient clipping: max_norm 1.0

### 4. Batch size
- Largest that fits 64GB unified memory without OOM
- Use same batch size across all cells

**After tuning:** lock the hyperparameters and run all 24 cells with only `corpus`, `size`, `seed`, and `tokenizer` varying.

---

## Experiment Protocol

### Phase 1 — Tuning (English 10M only)

Run each tuning variation for exactly **5 minutes** of wall-clock training time. Record `val_bpb` at the 5-minute mark. Accept a change if val_bpb improves by ≥ 0.01; revert otherwise. Stopping condition: 50 experiments on tuning track, or val_bpb plateau.

### Phase 2 — Fixed-config runs (all 24 cells)

Run each cell for a **fixed token budget** (set by Phase 1 plateau — whichever gives stable val_bpb estimates, typically 15–30 minutes per run).

Record to `results/runs.tsv` per seed:
- run_id, language, size, seed, tokenizer_type (bpe/byte), val_bpb_final, val_bpb_curve, training_wallclock, training_tokens, hyperparameters_json

### Phase 3 — Analysis

Run `python -m eval.benchmark significance` on `results/runs.tsv`. This computes bootstrap CIs across seeds per cell and compares against the pre-registered criterion.

---

## Stopping Condition

- Phase 1 tuning: 50 experiments OR val_bpb plateau for 20 consecutive experiments
- Phase 2 runs: all 24 cells complete with per-seed val_bpb logged

---

## Hardware Notes

**Target hardware**: Apple MacBook Pro M4 Max, 64GB unified memory
**Framework**: MLX (Apple Silicon native)
**autoresearch fork**: https://github.com/trevin-creator/autoresearch-mlx (clone to `train/autoresearch/`)

MLX-specific guidance:
- Use `mlx.core` for tensor ops
- Unified memory allows larger batch sizes than typical GPU setups
- Memory ceiling: ~50GB usable for model + activations; leave 14GB for OS
- Lazy evaluation: call `.eval()` before timing measurements

---

## Key Files

```
data/raw/
  english/simplewiki-articles.jsonl
  english/simplewiki-sentences.txt
  esperanto/eowiki-articles.jsonl
  esperanto/eowiki-sentences.txt

data/prepared/
  english/train.bin, val.bin, meta.pkl
  esperanto/train.bin, val.bin, meta.pkl
  english-byte/train.bin, val.bin   # byte-level ablation
  esperanto-byte/train.bin, val.bin

tokenizer/
  english-bpe/tokenizer.json
  esperanto-bpe/tokenizer.json

train/
  train.py              # model + training loop (MUTABLE in Phase 1, IMMUTABLE in Phase 2)
  prepare.py            # data pipeline (IMMUTABLE)
  program.md            # this file
  autoresearch/         # cloned autoresearch-mlx

results/
  runs.tsv              # all 24 cells, one row per seed
  tuning.tsv            # Phase 1 tuning experiments
```

---

## Interpreting Results

`val_bpb = (total_nll_nats) / (ln(2) * total_validation_bytes_utf8)`

Lower is better. UTF-8 byte normalization makes this language-invariant.

### Expected Decision Tree

- **H1 holds at both sizes, BPE and byte-level**: clean positive result. Morphological regularity improves pre-training efficiency beyond tokenization.
- **H1 holds at BPE only, null at byte-level**: gain is tokenization-driven. Reframe paper accordingly.
- **H1 fails at both sizes**: honest null. Report as such. No post-hoc sub-group hunting.
- **Esperanto significantly worse**: surprising; investigate and report honestly.

See `docs/preregistration.md §10` for the full outcome matrix.
