"""
make_bins.py
============
Converts raw JSONL files into packed .bin shard files for nanochat training.

Usage:
    python make_bins.py \
        --pubmed-dir   /data/medical_raw/pubmed \
        --fineweb-dir  /data/medical_raw/fineweb \
        --reasoning-dir /data/medical_raw/reasoning \
        --output-dir   /data/medical_bins_65k \
        --shard-size   100000000

The output structure will be:
    /data/medical_bins_65k/
        pubmed/
            shard_000000.bin
            shard_000001.bin
            ...
        fineweb/
            shard_000000.bin
            ...
        reasoning/
            shard_000000.bin
            ...

Each .bin file is a flat array of uint16 token IDs.
The dataloader in nanochat expects exactly this structure.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path

# ── make sure we can import nanochat from the repo root ────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── CLI arguments ──────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="Convert JSONL files to nanochat .bin shards")

    p.add_argument("--pubmed-dir",    type=str, required=True,
                   help="Directory containing PubMed JSONL files")
    p.add_argument("--fineweb-dir",   type=str, required=True,
                   help="Directory containing FineWeb-Edu JSONL files")
    p.add_argument("--reasoning-dir", type=str, required=True,
                   help="Directory containing Reasoning JSONL files")
    p.add_argument("--output-dir",    type=str, required=True,
                   help="Output directory for .bin shards (subdirs created automatically)")

    p.add_argument("--shard-size",    type=int, default=100_000_000,
                   help="Tokens per shard file (default: 100M)")
    p.add_argument("--text-field",    type=str, default="text",
                   help="JSON field name containing the document text (default: text)")
    p.add_argument("--min-chars",     type=int, default=30,
                   help="Minimum document length in characters (default: 30)")
    p.add_argument("--expected-vocab",type=int, default=65536,
                   help="Expected tokenizer vocab size — crashes if mismatch (default: 65536)")

    return p.parse_args()


# ── helpers ────────────────────────────────────────────────────────────────────

def iter_jsonl(data_dir: str, text_field: str, min_chars: int):
    """Stream text from all .jsonl files in a directory."""
    files = sorted(Path(data_dir).glob("*.jsonl"))
    if not files:
        print(f"  [WARNING] No .jsonl files found in {data_dir}")
        return
    for fpath in files:
        print(f"  Reading {fpath.name} ...")
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                try:
                    text = json.loads(line).get(text_field, "")
                    if len(text) >= min_chars:
                        yield text
                except Exception:
                    continue


def flush_shard(buf: list, output_dir: Path, shard_idx: int) -> int:
    """Write buffer to a .bin file and return number of tokens written."""
    arr  = np.array(buf, dtype=np.uint16)
    path = output_dir / f"shard_{shard_idx:06d}.bin"
    arr.tofile(str(path))
    n = len(arr)
    print(f"    Saved {path.name}  —  {n:,} tokens  ({n/1e6:.1f}M)")
    return n


def process_source(
    data_dir:    str,
    subdir_name: str,
    output_base: Path,
    tokenizer,
    shard_size:  int,
    text_field:  str,
    min_chars:   int,
):
    """Tokenize all JSONL files in data_dir and write .bin shards to output_base/subdir_name/."""
    output_dir = output_base / subdir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Source  : {data_dir}")
    print(f"  Output  : {output_dir}")
    print(f"{'='*60}")

    shard_idx  = 0
    buf        = []
    buf_size   = 0
    total_docs = 0
    total_toks = 0
    t_start    = time.time()

    for text in iter_jsonl(data_dir, text_field, min_chars):
        ids = tokenizer.encode(text, bos=True, eos=True)
        buf.extend(ids)
        buf_size  += len(ids)
        total_docs += 1

        if total_docs % 200_000 == 0:
            elapsed = time.time() - t_start
            print(f"    {total_docs:>9,} docs | "
                  f"{total_toks/1e9:.3f}B tokens so far | "
                  f"{elapsed/60:.1f}m elapsed")

        if buf_size >= shard_size:
            total_toks += flush_shard(buf, output_dir, shard_idx)
            shard_idx  += 1
            buf         = []
            buf_size    = 0

    # flush the last partial shard
    if buf:
        total_toks += flush_shard(buf, output_dir, shard_idx)
        shard_idx  += 1

    elapsed = time.time() - t_start
    print(f"\n  [{subdir_name}] Done.")
    print(f"  Docs    : {total_docs:,}")
    print(f"  Tokens  : {total_toks:,}  ({total_toks/1e9:.3f}B)")
    print(f"  Shards  : {shard_idx}")
    print(f"  Time    : {elapsed/60:.1f} minutes")

    return shard_idx, total_docs, total_toks


# ── sanity check ───────────────────────────────────────────────────────────────

def sanity_check(output_base: Path, sources: list):
    """Verify the first shard from each source looks correct."""
    print(f"\n{'='*60}")
    print("  SANITY CHECK")
    print(f"{'='*60}")
    all_ok = True
    for _, subdir_name in sources:
        shards = sorted((output_base / subdir_name).glob("*.bin"))
        if not shards:
            print(f"  [FAIL] No shards found in {subdir_name}/")
            all_ok = False
            continue
        data = np.fromfile(shards[0], dtype=np.uint16)
        max_id = int(data.max())
        min_id = int(data.min())
        ok = max_id < 65536
        status = "✓ OK" if ok else "✗ FAIL — token ID exceeds vocab!"
        print(f"  {subdir_name:12s} | shards: {len(shards):4d} | "
              f"first shard tokens: {len(data):,} | "
              f"max ID: {max_id} | {status}")
        if not ok:
            all_ok = False
    return all_ok


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    args = get_args()

    # ── Load tokenizer ─────────────────────────────────────────────────────────
    print("\nLoading nanochat tokenizer...")
    try:
        from nanochat.tokenizer import get_tokenizer
        tokenizer = get_tokenizer()
    except ImportError:
        print("ERROR: Could not import nanochat.tokenizer.")
        print("Make sure you run this script from inside the nanochat repo root")
        print("and that the venv is activated:  source .venv/bin/activate")
        sys.exit(1)

    actual_vocab = tokenizer.get_vocab_size()
    print(f"Tokenizer vocab size : {actual_vocab:,}")

    if actual_vocab != args.expected_vocab:
        print(f"\nERROR: Vocab size mismatch!")
        print(f"  Expected : {args.expected_vocab:,}")
        print(f"  Got      : {actual_vocab:,}")
        print(f"  Make sure you trained the tokenizer with --vocab-size={args.expected_vocab}")
        sys.exit(1)

    print(f"✓ Vocab size matches expected ({args.expected_vocab:,})")

    # ── Define sources ─────────────────────────────────────────────────────────
    # Each tuple: (input_dir, output_subdir_name)
    sources = [
        (args.pubmed_dir,    "pubmed"),
        (args.fineweb_dir,   "fineweb"),
        (args.reasoning_dir, "reasoning"),
    ]

    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput directory : {output_base}")
    print(f"Shard size       : {args.shard_size:,} tokens ({args.shard_size/1e6:.0f}M)")
    print(f"Text field       : {args.text_field}")
    print(f"Min doc chars    : {args.min_chars}")

    # ── Process each source ────────────────────────────────────────────────────
    t_total_start = time.time()
    grand_shards = 0
    grand_docs   = 0
    grand_tokens = 0

    for data_dir, subdir_name in sources:
        n_shards, n_docs, n_toks = process_source(
            data_dir    = data_dir,
            subdir_name = subdir_name,
            output_base = output_base,
            tokenizer   = tokenizer,
            shard_size  = args.shard_size,
            text_field  = args.text_field,
            min_chars   = args.min_chars,
        )
        grand_shards += n_shards
        grand_docs   += n_docs
        grand_tokens += n_toks

    # ── Final summary ──────────────────────────────────────────────────────────
    total_elapsed = time.time() - t_total_start
    print(f"\n{'='*60}")
    print("  FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  Total docs    : {grand_docs:,}")
    print(f"  Total tokens  : {grand_tokens:,}  ({grand_tokens/1e9:.3f}B)")
    print(f"  Total shards  : {grand_shards}")
    print(f"  Total time    : {total_elapsed/60:.1f} minutes")
    print(f"\n  Output structure:")
    for _, subdir_name in sources:
        n = len(list((output_base / subdir_name).glob("*.bin")))
        print(f"    {output_base}/{subdir_name}/  →  {n} shards")

    # ── Sanity check ───────────────────────────────────────────────────────────
    all_ok = sanity_check(output_base, sources)

    if all_ok:
        print(f"\n✓ All shards look good. Ready for training!")
        print(f"\nNext step — run base training:")
        print(f"  torchrun --standalone --nproc_per_node=2 \\")
        print(f"    -m scripts.base_train -- \\")
        print(f"    --depth=34 \\")
        print(f"    --data-dir={args.output_dir} \\")
        print(f"    --device-batch-size=4 \\")
        print(f"    --model-tag=medical_3b")
    else:
        print(f"\n✗ Some shards failed the sanity check. Fix the errors above before training.")
        sys.exit(1)


if __name__ == "__main__":
    main()