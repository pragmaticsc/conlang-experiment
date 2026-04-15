# Pre-Registration: English vs. Esperanto Pre-Training Efficiency

**Author:** Shaun Russell
**Committed:** (see `git log docs/preregistration.md` — timestamp of first commit is the pre-registration proof)
**Status:** Committed before any training run. Not to be revised after training begins.

---

## 1. Research Question

Does pre-training a small transformer on Esperanto Wikipedia yield a lower validation bits-per-byte (`val_bpb`) than pre-training an architecturally-identical transformer on Simple English Wikipedia, at matched compute?

**Motivation.** Wang & Wen (2025, arXiv:2502.04488) propose a unified AI-centric language system — a constructed or regularized language optimized for machine learning — and argue from theory that morphological regularity, reduced ambiguity, and consistent structure should lower computational overhead during training and inference. Their Section 9 explicitly leaves empirical validation as future work. This pre-registration defines the protocol for such a validation, using Esperanto (a real, human-written language with regular morphology and an existing Wikipedia corpus) as a naturalistic test case. Esperanto is the strongest empirical test available today that avoids the translation confounds that would accompany any machine-generated constructed-language corpus.

## 2. Hypothesis (H1)

**H1:** At both ~10M and ~30M parameter scales, Esperanto-trained models achieve lower `val_bpb` than English-trained models, with non-overlapping 95% confidence intervals across 3 seeds per cell, at ≥5% relative reduction.

**Null (H0):** The 95% CIs of English and Esperanto `val_bpb` overlap, OR the point-estimate reduction is < 5%.

A null result is acceptable and will be reported as such. No sub-group analysis or post-hoc reframing.

## 3. Design

2 languages × 2 model sizes × 3 seeds = 12 primary training runs.

Plus ablation: byte-level tokenization × 2 languages × 2 sizes × 3 seeds = 12 additional runs.

**Total:** 24 training runs.

### Matched conditions

- Identical transformer architecture (same layers, heads, d_model, ffn_dim) per size tier
- Identical optimizer and learning rate schedule
- Identical total training tokens (wall-clock and token-count matched; whichever binds first is reported)
- Identical BPE vocab size (8,192) for primary runs
- Byte-level baseline (no learned vocabulary) for ablation

### Corpora

- **English:** Simple English Wikipedia, latest dump from `dumps.wikimedia.org/simplewiki/latest/`
- **Esperanto:** Esperanto Wikipedia, latest dump from `dumps.wikimedia.org/eowiki/latest/`
- Both parsed with the same `clean_wikitext` routine in `data/download-*.sh`
- Held-out validation set: random 5% of articles, seed=42, same RNG call for both corpora
- **Corpus size matching:** After tokenization, truncate the larger corpus to match the smaller on total training tokens. This is the critical control against the "smaller vocab = easier prediction" confound.

## 4. Primary Metric

`val_bpb` — average bits-per-byte on the held-out validation set, computed as:

```
val_bpb = (sum of NLL over validation in nats) / (ln(2) * validation_bytes_utf8)
```

Bytes are UTF-8 bytes of the *raw text* (not tokens), so the metric is language-invariant and tokenization-invariant.

## 5. Statistical Test

For each model size, compute 95% bootstrap confidence intervals on the mean `val_bpb` across 3 seeds per language. Success criterion for H1: **Esperanto CI upper bound < English CI lower bound × 0.95** (i.e., ≥5% reduction with non-overlapping CIs), at both sizes.

Bootstrap: 10,000 resamples, percentile method.

## 6. Confounds Explicitly Addressed

1. **Vocabulary-size confound:** Matched BPE vocab size (8,192). Also reported: byte-level baseline (no learned vocabulary).
2. **Corpus-size confound:** Corpora matched on total training tokens (smaller truncates larger).
3. **Tokenization confound:** Byte-level ablation isolates "does tokenization help?" from "does language structure help?"
4. **Seed variance confound:** 3 seeds per cell, bootstrap CIs reported.
5. **Architecture/hyperparameter confound:** Identical architecture, identical optimizer, identical schedule. Any hyperparameter tuning happens once and applies to both languages.
6. **Validation-set confound:** Same RNG call for both corpora's held-out split.

## 7. Reporting Commitments

Regardless of outcome, the paper will report:
- All 24 training runs, with per-seed `val_bpb`, learning curves, and hyperparameters
- Byte-level and BPE results side by side
- Training FLOPs and wall-clock time per cell
- Release of all code, tokenizers, and processed corpora (or download+parse scripts sufficient for reproduction)

## 8. What Would Change This Pre-Registration

Only the following edits are permitted after the first training run, and each requires explicit commit-and-push of this file:

- **Typo corrections** (tracked in git history)
- **Clarification of ambiguous language** (must not change the falsifiable criterion)
- **Protocol expansion** if a confound not listed here is discovered — but the original criterion remains binding, and the new analysis is reported as exploratory

The success criterion in §5 cannot be relaxed after training begins. If results narrowly miss it, they are reported as "below pre-registered threshold" and H0 is accepted.

## 9. Deferred Questions (Explicitly NOT Tested Here)

These are out of scope for Paper 1 and reserved for future work:
- Constructed languages beyond Esperanto (Loga, etc.)
- Ternary quantization effects
- Head-level sparsity structure
- Downstream fine-tuning performance (ARC, GSM8K, etc.)
- Model sizes outside {10M, 30M}

## 10. Anticipated Outcomes

- **Strong positive:** H1 holds at both sizes, and byte-level results are consistent → evidence that morphological regularity improves pre-training efficiency beyond tokenization effects. Clean result.
- **Tokenization-only positive:** H1 holds at BPE, but byte-level results are null → gain is tokenization-driven, not structural. Still publishable, reframed.
- **Null:** No significant difference at either size → honest negative result. Reported as such.
- **Negative:** English significantly beats Esperanto → also reported honestly. Would be a surprising finding worth investigating.
