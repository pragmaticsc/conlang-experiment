"""
tokenizer/tokenizer_train.py
============================
Train BPE tokenizers for both English and Esperanto corpora and compare
vocabulary efficiency (tokens per line / chars per token / bytes per token).

Outputs:
  tokenizer/english-bpe/     — HuggingFace tokenizers format
  tokenizer/esperanto-bpe/   — HuggingFace tokenizers format
  tokenizer/efficiency_report.json — comparison metrics

Usage:
    python -m tokenizer.tokenizer_train train \
        --english-corpus data/raw/english/simplewiki-sentences.txt \
        --esperanto-corpus data/raw/esperanto/eowiki-sentences.txt \
        --vocab-size 8192
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]


def train_bpe(
    corpus_path: Path,
    output_dir: Path,
    vocab_size: int,
    name: str,
) -> Tokenizer:
    """Train a BPE tokenizer on a text file (one sentence/paragraph per line)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        min_frequency=2,
    )

    log.info("Training %s BPE tokenizer (vocab=%d) on %s", name, vocab_size, corpus_path)
    tokenizer.train(files=[str(corpus_path)], trainer=trainer)

    save_path = output_dir / "tokenizer.json"
    tokenizer.save(str(save_path))
    log.info("Saved %s tokenizer → %s", name, save_path)
    return tokenizer


def measure_efficiency(
    tokenizer: Tokenizer,
    sample_file: Path,
    max_lines: int = 10000,
) -> dict:
    """Measure tokenizer efficiency on a corpus sample."""
    total_tokens = 0
    total_chars = 0
    total_bytes = 0
    lines_processed = 0

    with open(sample_file, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            encoding = tokenizer.encode(line)
            total_tokens += len(encoding.ids)
            total_chars += len(line)
            total_bytes += len(line.encode("utf-8"))
            lines_processed += 1

    if total_tokens == 0:
        return {"error": "no tokens produced"}

    return {
        "chars_per_token": round(total_chars / total_tokens, 4),
        "bytes_per_token": round(total_bytes / total_tokens, 4),
        "tokens_per_line": round(total_tokens / lines_processed, 2),
        "total_tokens": total_tokens,
        "total_chars": total_chars,
        "total_bytes": total_bytes,
        "lines_processed": lines_processed,
    }


def build_efficiency_report(
    english_metrics: dict,
    esperanto_metrics: dict,
    vocab_size: int,
) -> dict:
    """Compare tokenizer efficiency. Neutral framing — both languages reported equally."""
    report = {
        "english": {**english_metrics, "vocab_size": vocab_size},
        "esperanto": {**esperanto_metrics, "vocab_size": vocab_size},
    }

    if "bytes_per_token" in english_metrics and "bytes_per_token" in esperanto_metrics:
        e_bpt = english_metrics["bytes_per_token"]
        o_bpt = esperanto_metrics["bytes_per_token"]
        report["comparison"] = {
            "bytes_per_token_delta": round(o_bpt - e_bpt, 4),
            "bytes_per_token_ratio_eo_over_en": round(o_bpt / e_bpt, 4),
        }

    return report


@click.group()
def cli():
    """BPE tokenizer training and efficiency analysis for English vs. Esperanto."""
    pass


@cli.command("train")
@click.option("--english-corpus", type=click.Path(exists=True, path_type=Path),
              default="data/raw/english/simplewiki-sentences.txt")
@click.option("--esperanto-corpus", type=click.Path(exists=True, path_type=Path),
              default="data/raw/esperanto/eowiki-sentences.txt")
@click.option("--output-dir", type=click.Path(path_type=Path),
              default="tokenizer")
@click.option("--vocab-size", default=8192, show_default=True)
@click.option("--sample-lines", default=10000, show_default=True)
def train_cmd(english_corpus, esperanto_corpus, output_dir, vocab_size, sample_lines):
    """Train BPE tokenizers for English and Esperanto and report efficiency."""
    eng_dir = output_dir / "english-bpe"
    esp_dir = output_dir / "esperanto-bpe"

    eng_tok = train_bpe(english_corpus, eng_dir, vocab_size, "English")
    esp_tok = train_bpe(esperanto_corpus, esp_dir, vocab_size, "Esperanto")

    log.info("Measuring efficiency...")
    eng_metrics = measure_efficiency(eng_tok, english_corpus, sample_lines)
    esp_metrics = measure_efficiency(esp_tok, esperanto_corpus, sample_lines)

    report = build_efficiency_report(eng_metrics, esp_metrics, vocab_size)

    report_path = output_dir / "efficiency_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    log.info("Efficiency report → %s", report_path)

    print("\n" + "=" * 60)
    print(f"TOKENIZER EFFICIENCY COMPARISON (vocab={vocab_size})")
    print("=" * 60)
    print(f"  English    bytes/token: {eng_metrics.get('bytes_per_token', 'N/A')}")
    print(f"  Esperanto  bytes/token: {esp_metrics.get('bytes_per_token', 'N/A')}")
    if "comparison" in report:
        cmp = report["comparison"]
        print(f"  Delta (eo-en):     {cmp['bytes_per_token_delta']:+.4f}")
        print(f"  Ratio (eo/en):     {cmp['bytes_per_token_ratio_eo_over_en']}")
    print("=" * 60)


if __name__ == "__main__":
    cli()
