"""
Populate the autoresearch cache (~/.cache/autoresearch/) from local Wikipedia corpora.

Replaces autoresearch's prepare.py download step with our own English/Esperanto
Wikipedia data. The cache layout is identical, so train.py works unmodified.

Usage (from train/autoresearch/):
    uv run python ../prepare_local.py --corpus english --clean
    uv run python ../prepare_local.py --corpus esperanto --clean
"""

import argparse
import json
import os
import pickle
import random
import shutil
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import rustbpe
import tiktoken

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "data")
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")

VOCAB_SIZE = 8192
VAL_SHARD_INDEX = 6542

SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]

VAL_FRACTION = 0.05
SPLIT_SEED = 42
DOCS_PER_SHARD = 50_000

CORPUS_FILES = {
    "english": "simplewiki-articles.jsonl",
    "esperanto": "eowiki-articles.jsonl",
}


def load_corpus(data_dir, corpus):
    jsonl_name = CORPUS_FILES[corpus]
    jsonl_path = os.path.join(data_dir, corpus, jsonl_name)
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(
            f"Expected {jsonl_path}\n"
            f"Run: bash data/download-{corpus}.sh"
        )
    docs = []
    with open(jsonl_path) as f:
        for line in f:
            obj = json.loads(line)
            text = obj.get("text", "")
            if len(text.strip()) > 50:
                docs.append(text)
    return docs


def split_train_val(docs):
    rng = random.Random(SPLIT_SEED)
    indices = list(range(len(docs)))
    rng.shuffle(indices)
    n_val = max(1, int(len(docs) * VAL_FRACTION))
    val_set = set(indices[:n_val])
    train = [docs[i] for i in range(len(docs)) if i not in val_set]
    val = [docs[i] for i in range(len(docs)) if i in val_set]
    return train, val


def write_shards(docs, start_index=0):
    paths = []
    for i in range(0, len(docs), DOCS_PER_SHARD):
        chunk = docs[i : i + DOCS_PER_SHARD]
        shard_idx = start_index + i // DOCS_PER_SHARD
        if shard_idx == VAL_SHARD_INDEX:
            shard_idx += 1
        path = os.path.join(DATA_DIR, f"shard_{shard_idx:05d}.parquet")
        pq.write_table(pa.table({"text": chunk}), path)
        paths.append(path)
    return paths


def train_tokenizer(train_docs):
    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    tokenizer_pkl = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")
    token_bytes_path = os.path.join(TOKENIZER_DIR, "token_bytes.npy")

    print("Training BPE tokenizer (rustbpe)...")
    t0 = time.time()

    tok = rustbpe.Tokenizer()
    vocab_no_special = VOCAB_SIZE - len(SPECIAL_TOKENS)

    def doc_iter():
        for doc in train_docs:
            yield doc[:10_000]

    tok.train_from_iterator(doc_iter(), vocab_no_special, pattern=SPLIT_PATTERN)

    mergeable_ranks = {bytes(k): v for k, v in tok.get_mergeable_ranks()}
    offset = len(mergeable_ranks)
    special = {name: offset + i for i, name in enumerate(SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(
        name="rustbpe",
        pat_str=tok.get_pattern(),
        mergeable_ranks=mergeable_ranks,
        special_tokens=special,
    )

    with open(tokenizer_pkl, "wb") as f:
        pickle.dump(enc, f)
    print(f"Tokenizer trained in {time.time() - t0:.1f}s (vocab={enc.n_vocab})")

    special_set = set(SPECIAL_TOKENS)
    tbytes = []
    for tid in range(enc.n_vocab):
        s = enc.decode([tid])
        tbytes.append(0 if s in special_set else len(s.encode("utf-8")))
    np.save(token_bytes_path, np.array(tbytes, dtype=np.int32))
    print(f"Saved token_bytes.npy")

    test = "Hello world! Saluton mondo! Ĉu vi parolas Esperanton?"
    assert enc.decode(enc.encode_ordinary(test)) == test, "Roundtrip failed"
    print("Tokenizer roundtrip OK")
    return enc


def count_tokens(enc, docs, label=""):
    total = 0
    for doc in docs:
        total += len(enc.encode_ordinary(doc))
    print(f"Token count ({label}): {total:,}")
    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True, choices=["english", "esperanto"])
    p.add_argument("--data-dir", default=None)
    p.add_argument("--clean", action="store_true", help="Wipe cache before populating")
    args = p.parse_args()

    if args.data_dir is None:
        here = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(here, "..", "data", "raw")
        if os.path.isdir(candidate):
            args.data_dir = os.path.abspath(candidate)
        else:
            raise FileNotFoundError("Pass --data-dir /path/to/data/raw")

    print(f"Corpus:    {args.corpus}")
    print(f"Data dir:  {args.data_dir}")
    print(f"Cache dir: {CACHE_DIR}")
    print()

    if args.clean and os.path.exists(CACHE_DIR):
        print("Cleaning cache...")
        shutil.rmtree(CACHE_DIR)
    os.makedirs(DATA_DIR, exist_ok=True)

    docs = load_corpus(args.data_dir, args.corpus)
    print(f"Loaded {len(docs):,} documents")

    train_docs, val_docs = split_train_val(docs)
    print(f"Split: {len(train_docs):,} train, {len(val_docs):,} val ({VAL_FRACTION*100:.0f}%)")

    train_paths = write_shards(train_docs)
    print(f"Wrote {len(train_paths)} training shard(s)")

    val_path = os.path.join(DATA_DIR, f"shard_{VAL_SHARD_INDEX:05d}.parquet")
    pq.write_table(pa.table({"text": val_docs}), val_path)
    print(f"Wrote validation shard ({len(val_docs):,} docs)")

    enc = train_tokenizer(train_docs)
    train_tokens = count_tokens(enc, train_docs, "train")
    val_tokens = count_tokens(enc, val_docs, "val")

    total_bytes = sum(len(d.encode("utf-8")) for d in train_docs + val_docs)
    print(f"\nCorpus bytes (UTF-8): {total_bytes:,}")
    print(f"Total tokens: {train_tokens + val_tokens:,}")
    print(f"\nCache ready at {CACHE_DIR}")
    print(f"Next: cd train/autoresearch && uv run train.py")


if __name__ == "__main__":
    main()
