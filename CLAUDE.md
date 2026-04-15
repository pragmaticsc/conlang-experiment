# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**conlang-experiment** is a narrowly-scoped empirical study: does pre-training an LLM on Esperanto Wikipedia yield lower validation bits-per-byte than pre-training on Simple English Wikipedia at matched compute and matched architecture?

This paper directly answers the open question raised in Wang & Wen (2025, arXiv:2502.04488), a position paper that proposes an "AI-centric language system" with regular morphology but leaves empirical validation as future work. Paper 1 is framed as the empirical answer to their proposal — cite them in the introduction.

This is Paper 1 of a planned sequence. **Paper 1 contains no constructed-language work** — it uses Esperanto (a real language with native speakers) to avoid translation confounds. Loga (the author's constructed language) is developed in a sibling project at `~/Dev/loga` and is deferred to Paper 2.

## Scope Discipline

Things that are **out of scope** for this project and should not be added:
- Ternary weight quantization (BitLinear) — Paper 2 concern
- Head-level sparsity analysis — Paper 2 concern
- Constructed-language design, translation pipelines, or Loga-related anything
- Architectures beyond the 2-size × 2-language × 3-seed factorial
- Speculation about "language design as optimization" in the paper text

If you find yourself wanting to add any of the above, stop. Note it as "future work" and keep going.

## Common Commands

**Install:**
```bash
pip install -e ".[train,dev]"
```

**Pipeline (run in order):**
```bash
bash data/download-english.sh                                   # Simple English Wikipedia
bash data/download-esperanto.sh                                 # Esperanto Wikipedia
python -m tokenizer.tokenizer_train train --vocab-size 8192     # Train BPE tokenizers + efficiency report
# Training runs via autoresearch-mlx — see train/program.md
python -m eval.benchmark significance                           # Bootstrap CIs across seeds
```

**Tests and linting:**
```bash
pytest                              # tests/ does not exist yet
ruff check .
ruff format .
```

## Architecture

```
data/download-english.sh            # Simple English Wikipedia → data/raw/english/
data/download-esperanto.sh          # Esperanto Wikipedia → data/raw/esperanto/
tokenizer/tokenizer_train.py        # BPE training + chars/bytes/tokens efficiency metrics
train/program.md                    # autoresearch-mlx experiment protocol (2x2x3 seeds)
eval/benchmark.py                   # Evaluation: learning curves, bootstrap CI significance
docs/preregistration.md             # Pre-registered hypothesis & success criterion (committed before training)
docs/paper.md                       # Paper draft (written AFTER results)
results/                            # val_bpb TSVs and plots from training runs
```

## Key Design Decisions

- **Corpora are native to each language** — no translation step. Esperanto articles are written by Esperantists. This deliberately eliminates the largest vulnerability flagged in the Paper 2 reviewer correspondence (`~/Dev/loga/notes/grok-review_32726.md`, `~/Dev/loga/notes/chatgpt-review.md`): translation confound.
- **Byte-level baseline is mandatory**, not optional. It isolates "does tokenization help?" from "does language structure help?"
- **3 seeds minimum per cell.** Small-model training noise is real. Report CIs.
- **Pre-registration is committed before first training run.** The git timestamp on `docs/preregistration.md` is the pre-registration proof.
- **Success criterion is set before running training.** Do not adjust after seeing data.

## Relationship to `~/Dev/loga`

The parallel `~/Dev/loga` project contains:
- The Loga constructed language spec
- BitLinear (ternary weights) implementation
- Sparsity analysis tooling
- Multiple abandoned paper drafts

None of that material belongs in this project. If Paper 1 lands positively, a Paper 2 in `~/Dev/loga` can extend to the constructed-language case with empirical ammunition. If Paper 1 is null, Paper 2 strategy needs to be rethought — don't pre-commit.

## Null Result Acceptance

A well-controlled null result (English and Esperanto yield statistically indistinguishable val_bpb) is a valid, publishable outcome. Do not reframe to avoid it. Do not hunt for sub-group effects. The pre-registration document is the contract.
