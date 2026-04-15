"""
eval/benchmark.py
=================
Evaluation harness for the English vs. Esperanto pre-training experiment.

Modes:
  tokenizer      — compare bytes/chars-per-token between English and Esperanto tokenizers
  learning-curve — plot val_bpb vs. training step from per-seed TSVs
  significance   — bootstrap 95% CIs across seeds, test the pre-registered criterion
  summary        — run all modes and write a markdown report

Input format (results/runs.tsv), tab-separated:
  run_id  language  size  seed  tokenizer_type  val_bpb  tokens  wallclock_s  hparams_json
"""

from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Pre-registered success criterion (see docs/preregistration.md §5)
PREREG_RELATIVE_REDUCTION = 0.05  # 5%
PREREG_CI_ALPHA = 0.05            # 95% CI
PREREG_BOOTSTRAP_N = 10_000


# ---------------------------------------------------------------------------
# Tokenizer efficiency
# ---------------------------------------------------------------------------

def compare_tokenizers(
    english_tokenizer_path: Path,
    esperanto_tokenizer_path: Path,
    english_corpus: Path,
    esperanto_corpus: Path,
    sample_lines: int = 5000,
) -> dict:
    from tokenizers import Tokenizer

    eng_tok = Tokenizer.from_file(str(english_tokenizer_path))
    esp_tok = Tokenizer.from_file(str(esperanto_tokenizer_path))

    def measure(tok, corpus: Path, n: int) -> dict:
        total_tokens, total_chars, total_bytes, lines = 0, 0, 0, 0
        with open(corpus, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= n:
                    break
                line = line.strip()
                if not line:
                    continue
                enc = tok.encode(line)
                total_tokens += len(enc.ids)
                total_chars += len(line)
                total_bytes += len(line.encode("utf-8"))
                lines += 1
        if total_tokens == 0:
            return {"error": "no tokens"}
        return {
            "bytes_per_token": round(total_bytes / total_tokens, 4),
            "chars_per_token": round(total_chars / total_tokens, 4),
            "tokens_per_line": round(total_tokens / lines, 2),
        }

    return {
        "english": measure(eng_tok, english_corpus, sample_lines),
        "esperanto": measure(esp_tok, esperanto_corpus, sample_lines),
    }


# ---------------------------------------------------------------------------
# Results TSV loading
# ---------------------------------------------------------------------------

def load_runs(path: Path) -> list[dict]:
    """Load results/runs.tsv into list of dicts."""
    rows: list[dict] = []
    if not path.exists():
        log.warning("Results file not found: %s", path)
        return rows
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            row["val_bpb"] = float(row["val_bpb"])
            row["seed"] = int(row["seed"])
            rows.append(row)
    return rows


def group_by_cell(rows: list[dict]) -> dict[tuple, list[float]]:
    """Group val_bpb by (language, size, tokenizer_type)."""
    groups: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        key = (r["language"], r["size"], r.get("tokenizer_type", "bpe"))
        groups[key].append(r["val_bpb"])
    return groups


# ---------------------------------------------------------------------------
# Bootstrap CIs
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: list[float],
    n_bootstrap: int = PREREG_BOOTSTRAP_N,
    alpha: float = PREREG_CI_ALPHA,
    rng_seed: int = 42,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for the mean. Returns (mean, lo, hi)."""
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(rng_seed)
    means = np.empty(n_bootstrap)
    n = len(arr)
    for i in range(n_bootstrap):
        sample = arr[rng.integers(0, n, size=n)]
        means[i] = sample.mean()
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (float(arr.mean()), lo, hi)


def evaluate_prereg(groups: dict[tuple, list[float]]) -> dict:
    """
    For each (size, tokenizer_type), compare English vs. Esperanto.
    Pre-registered criterion: Esperanto CI upper < English CI lower × (1 - 0.05).
    """
    results = {}
    sizes = sorted({key[1] for key in groups.keys()})
    tok_types = sorted({key[2] for key in groups.keys()})

    for size in sizes:
        for tok in tok_types:
            en_key = ("english", size, tok)
            eo_key = ("esperanto", size, tok)
            en_vals = groups.get(en_key, [])
            eo_vals = groups.get(eo_key, [])
            if len(en_vals) < 2 or len(eo_vals) < 2:
                results[f"{size}-{tok}"] = {
                    "status": "insufficient_seeds",
                    "english_n": len(en_vals),
                    "esperanto_n": len(eo_vals),
                }
                continue

            en_mean, en_lo, en_hi = bootstrap_ci(en_vals)
            eo_mean, eo_lo, eo_hi = bootstrap_ci(eo_vals)
            threshold = en_lo * (1 - PREREG_RELATIVE_REDUCTION)
            criterion_met = eo_hi < threshold
            relative_reduction = (en_mean - eo_mean) / en_mean if en_mean > 0 else 0.0

            results[f"{size}-{tok}"] = {
                "english": {"mean": round(en_mean, 4), "ci_lo": round(en_lo, 4), "ci_hi": round(en_hi, 4), "n": len(en_vals)},
                "esperanto": {"mean": round(eo_mean, 4), "ci_lo": round(eo_lo, 4), "ci_hi": round(eo_hi, 4), "n": len(eo_vals)},
                "relative_reduction": round(relative_reduction, 4),
                "threshold_for_criterion": round(threshold, 4),
                "criterion_met": bool(criterion_met),
                "interpretation": (
                    f"H1 SUPPORTED at {size} ({tok}): Esperanto CI upper ({eo_hi:.4f}) < threshold ({threshold:.4f})"
                    if criterion_met else
                    f"H1 NOT SUPPORTED at {size} ({tok}): Esperanto CI upper ({eo_hi:.4f}) >= threshold ({threshold:.4f})"
                ),
            }
    return results


# ---------------------------------------------------------------------------
# Learning curves
# ---------------------------------------------------------------------------

def plot_learning_curves(runs_path: Path, output: Path) -> None:
    """Plot per-seed val_bpb final values grouped by cell."""
    rows = load_runs(runs_path)
    if not rows:
        log.error("No runs found in %s", runs_path)
        return
    groups = group_by_cell(rows)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    tok_types = sorted({k[2] for k in groups})

    for ax, tok in zip(axes, tok_types):
        labels, means, lo_errs, hi_errs = [], [], [], []
        for size in sorted({k[1] for k in groups if k[2] == tok}):
            for lang in ("english", "esperanto"):
                vals = groups.get((lang, size, tok), [])
                if not vals:
                    continue
                m, lo, hi = bootstrap_ci(vals)
                labels.append(f"{lang[:2]}-{size}")
                means.append(m)
                lo_errs.append(m - lo)
                hi_errs.append(hi - m)
        x = np.arange(len(labels))
        ax.bar(x, means, yerr=[lo_errs, hi_errs], capsize=5,
               color=["steelblue" if "en" in lab else "coral" for lab in labels])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(f"val_bpb by cell ({tok})")
        ax.set_ylabel("val_bpb (lower = better)")
        ax.grid(True, alpha=0.3, axis="y")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    log.info("Saved learning curves → %s", output)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.group()
def cli():
    """Evaluation harness for English vs. Esperanto pre-training."""
    pass


@cli.command("tokenizer")
@click.option("--english-tokenizer", type=click.Path(exists=True, path_type=Path),
              default="tokenizer/english-bpe/tokenizer.json")
@click.option("--esperanto-tokenizer", type=click.Path(exists=True, path_type=Path),
              default="tokenizer/esperanto-bpe/tokenizer.json")
@click.option("--english-corpus", type=click.Path(exists=True, path_type=Path),
              default="data/raw/english/simplewiki-sentences.txt")
@click.option("--esperanto-corpus", type=click.Path(exists=True, path_type=Path),
              default="data/raw/esperanto/eowiki-sentences.txt")
@click.option("--sample-lines", default=5000, show_default=True)
def tokenizer_cmd(english_tokenizer, esperanto_tokenizer, english_corpus, esperanto_corpus, sample_lines):
    """Compare BPE tokenizer efficiency between English and Esperanto."""
    result = compare_tokenizers(english_tokenizer, esperanto_tokenizer,
                                 english_corpus, esperanto_corpus, sample_lines)
    print(json.dumps(result, indent=2))


@cli.command("significance")
@click.option("--runs", type=click.Path(exists=True, path_type=Path),
              default="results/runs.tsv")
@click.option("--output", type=click.Path(path_type=Path),
              default="results/significance.json")
def significance_cmd(runs, output):
    """Bootstrap 95% CIs across seeds. Test pre-registered criterion."""
    rows = load_runs(runs)
    if not rows:
        log.error("No runs loaded — cannot evaluate criterion.")
        return
    groups = group_by_cell(rows)
    evaluation = evaluate_prereg(groups)

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(evaluation, f, indent=2)

    print("\n" + "=" * 72)
    print("PRE-REGISTERED CRITERION EVALUATION")
    print(f"(threshold: Esperanto CI upper < English CI lower × {1 - PREREG_RELATIVE_REDUCTION:.2f})")
    print("=" * 72)
    for cell, res in evaluation.items():
        print(f"\n[{cell}]")
        if res.get("status") == "insufficient_seeds":
            print(f"  INSUFFICIENT SEEDS: en_n={res['english_n']}, eo_n={res['esperanto_n']}")
            continue
        print(f"  English:   {res['english']['mean']:.4f}  CI=[{res['english']['ci_lo']:.4f}, {res['english']['ci_hi']:.4f}]  n={res['english']['n']}")
        print(f"  Esperanto: {res['esperanto']['mean']:.4f}  CI=[{res['esperanto']['ci_lo']:.4f}, {res['esperanto']['ci_hi']:.4f}]  n={res['esperanto']['n']}")
        print(f"  Relative reduction (eo vs en): {res['relative_reduction']*100:+.2f}%")
        print(f"  {res['interpretation']}")
    print("\n" + "=" * 72)


@cli.command("learning-curve")
@click.option("--runs", type=click.Path(exists=True, path_type=Path),
              default="results/runs.tsv")
@click.option("--output", type=click.Path(path_type=Path),
              default="results/learning_curves.png")
def learning_curve_cmd(runs, output):
    """Plot per-cell val_bpb with CIs across seeds."""
    plot_learning_curves(runs, output)


@cli.command("summary")
@click.option("--runs", type=click.Path(path_type=Path),
              default="results/runs.tsv")
@click.option("--output", type=click.Path(path_type=Path),
              default="results/summary.md")
def summary_cmd(runs, output):
    """Generate a markdown summary report of all results."""
    rows = load_runs(runs)
    lines = ["# English vs. Esperanto Pre-Training: Summary Report", ""]
    if not rows:
        lines.append("_No runs available yet. Execute the experiment protocol in `train/program.md`._")
    else:
        groups = group_by_cell(rows)
        evaluation = evaluate_prereg(groups)
        lines.append("## Pre-registered criterion evaluation")
        lines.append("")
        for cell, res in evaluation.items():
            if res.get("status") == "insufficient_seeds":
                lines.append(f"- **{cell}**: insufficient seeds")
                continue
            lines.append(f"- **{cell}**: {res['interpretation']}")
        lines.append("")
        lines.append(f"Total runs loaded: {len(rows)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        f.write("\n".join(lines) + "\n")
    log.info("Summary written → %s", output)


if __name__ == "__main__":
    cli()
