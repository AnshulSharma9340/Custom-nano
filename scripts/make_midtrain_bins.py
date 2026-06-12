"""
make_midtrain_bins.py
=====================
Converts mid-training JSONL files into packed .bin shard files for nanochat.
Supports task mixture weights so each source contributes proportionally.

Usage:
    python scripts/make_midtrain_bins.py \
        --data-dir  data/custom_nano_mid_train \
        --output-dir data/midtrain_bins_65k \
        --shard-size 100000000

Output structure:
    data/midtrain_bins_65k/
        cleaned-pubmed/
            shard_000000.bin ...
        cleaned-clinical-notes/
            shard_000000.bin ...
        ...

Task mixture weights (how much each source contributes):
    cleaned-pubmed          : 30%  — core medical knowledge
    cleaned-clinical-notes  : 20%  — real clinical language
    cleaned-medqa           : 15%  — medical Q&A
    cleaned-medmcqa         : 10%  — medical MCQ
    cleaned-drug-data       : 10%  — pharmacology
    cleaned-openhermes      : 10%  — instruction following
    cleaned-fineweb         :  3%  — general language
    cleaned-metamath        :  2%  — reasoning
"""

import os
import sys
import json
import time
import random
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Task mixture weights ───────────────────────────────────────────────────────
# These control how much each source contributes to the final training mix.
# Must sum to 1.0
TASK_MIXTURE = {
    "cleaned-pubmed":          0.30,
    "cleaned-clinical;-notes": 0.20,
    "cleaned-medqa":           0.15,
    "cleaned-medmcqa":         0.10,
    "cleaned-drug-data":       0.10,
    "cleaned-openhermes":      0.10,
    "cleaned-fineweb":         0.03,
    "cleaned-metamath":        0.02,
}

# ── CLI arguments ──────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser(description="Convert mid-training JSONL files to nanochat .bin shards")
    p.add_argument("--data-dir",      type=str, required=True,
                   help="Root directory containing all mid-train source folders")
    p.add_argument("--output-dir",    type=str, required=True,
                   help="Output directory for .bin shards")
    p.add_argument("--shard-size",    type=int, default=100_000_000,
                   help="Tokens per shard file (default: 100M)")
    p.add_argument("--text-field",    type=str, default="text",
                   help="JSON field name for text (default: text)")
    p.add_argument("--min-chars",     type=int, default=30,
                   help="Minimum document length in characters (default: 30)")
    p.add_argument("--expected-vocab",type=int, default=65536,
                   help="Expected tokenizer vocab size (default: 65536)")
    p.add_argument("--seed",          type=int, default=42,
                   help="Random seed for shuffling (default: 42)")
    return p.parse_args()


# ── helpers ────────────────────────────────────────────────────────────────────

def iter_jsonl(data_dir: str, text_field: str, min_chars: int):
    """Stream text from all .jsonl files in a directory."""
    files = sorted(Path(data_dir).glob("*.jsonl"))
    if not files:
        print(f"  [WARNING] No .jsonl files found in {data_dir}")
        return
    for fpath in files:
        print(f"    Reading {fpath.name} ...")
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
    source_name: str,
    data_dir:    str,
    output_base: Path,
    tokenizer,
    shard_size:  int,
    text_field:  str,
    min_chars:   int,
    weight:      float,
):
    """Tokenize all JSONL files in data_dir and write .bin shards."""
    # sanitize folder name for output (replace ; with -)
    safe_name  = source_name.replace(";", "-")
    output_dir = output_base / safe_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Source  : {source_name}  (weight={weight:.0%})")
    print(f"  Input   : {data_dir}")
    print(f"  Output  : {output_dir}")
    print(f"{'='*60}")

    shard_idx  = 0
    buf        = []
    buf_size   = 0
    total_docs = 0
    total_toks = 0
    t_start    = time.time()

    for text in iter_jsonl(data_dir, text_field, min_chars):
        ids = tokenizer.encode(text, prepend=tokenizer.get_bos_token_id())
        buf.extend(ids)
        buf_size   += len(ids)
        total_docs += 1

        if total_docs % 100_000 == 0:
            elapsed = time.time() - t_start
            print(f"    {total_docs:>8,} docs | "
                  f"{total_toks/1e9:.3f}B tokens | "
                  f"{elapsed/60:.1f}m elapsed")

        if buf_size >= shard_size:
            total_toks += flush_shard(buf, output_dir, shard_idx)
            shard_idx  += 1
            buf         = []
            buf_size    = 0

    if buf:
        total_toks += flush_shard(buf, output_dir, shard_idx)
        shard_idx  += 1

    elapsed = time.time() - t_start
    print(f"\n  [{safe_name}] Done.")
    print(f"  Docs    : {total_docs:,}")
    print(f"  Tokens  : {total_toks:,}  ({total_toks/1e9:.3f}B)")
    print(f"  Shards  : {shard_idx}")
    print(f"  Time    : {elapsed/60:.1f} minutes")

    return shard_idx, total_docs, total_toks


def sanity_check(output_base: Path, sources: list):
    """Verify first shard from each source."""
    print(f"\n{'='*60}")
    print("  SANITY CHECK")
    print(f"{'='*60}")
    all_ok = True
    for source_name, _ in sources:
        safe_name = source_name.replace(";", "-")
        shards    = sorted((output_base / safe_name).glob("*.bin"))
        if not shards:
            print(f"  [FAIL] No shards in {safe_name}/")
            all_ok = False
            continue
        data   = np.fromfile(shards[0], dtype=np.uint16)
        max_id = int(data.max())
        ok     = max_id < 65536
        status = "✓ OK" if ok else "✗ FAIL — token ID exceeds vocab!"
        print(f"  {safe_name:30s} | shards: {len(shards):3d} | "
              f"tokens: {len(data):,} | max ID: {max_id} | {status}")
        if not ok:
            all_ok = False
    return all_ok


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    random.seed(args.seed)

    # Load tokenizer
    print("\nLoading nanochat tokenizer...")
    try:
        from nanochat.tokenizer import get_tokenizer
        tokenizer = get_tokenizer()
    except ImportError:
        print("ERROR: Could not import nanochat.tokenizer.")
        print("Activate venv first:  source .venv/bin/activate")
        sys.exit(1)

    actual_vocab = tokenizer.get_vocab_size()
    print(f"Tokenizer vocab size : {actual_vocab:,}")
    if actual_vocab != args.expected_vocab:
        print(f"ERROR: Expected {args.expected_vocab:,} but got {actual_vocab:,}")
        sys.exit(1)
    print(f"✓ Vocab size matches ({args.expected_vocab:,})")

    # Discover sources
    data_root   = Path(args.data_dir)
    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    # Build source list from TASK_MIXTURE
    sources = []
    for source_name, weight in TASK_MIXTURE.items():
        source_path = data_root / source_name
        if source_path.exists():
            sources.append((source_name, weight))
            print(f"  ✓ Found: {source_name:35s} weight={weight:.0%}")
        else:
            print(f"  ✗ Missing: {source_name:33s} — skipping")

    assert sources, f"No sources found in {data_root}"

    # Normalize weights in case some sources are missing
    total_weight = sum(w for _, w in sources)
    sources      = [(n, w/total_weight) for n, w in sources]

    print(f"\nOutput directory : {output_base}")
    print(f"Shard size       : {args.shard_size:,} tokens ({args.shard_size/1e6:.0f}M)")
    print(f"\nTask mixture (normalized):")
    for name, weight in sources:
        print(f"  {name:35s} {weight:.1%}")

    # Process each source
    t_total_start = time.time()
    grand_shards  = 0
    grand_docs    = 0
    grand_tokens  = 0

    for source_name, weight in sources:
        source_path = data_root / source_name
        n_shards, n_docs, n_toks = process_source(
            source_name = source_name,
            data_dir    = str(source_path),
            output_base = output_base,
            tokenizer   = tokenizer,
            shard_size  = args.shard_size,
            text_field  = args.text_field,
            min_chars   = args.min_chars,
            weight      = weight,
        )
        grand_shards += n_shards
        grand_docs   += n_docs
        grand_tokens += n_toks

    # Final summary
    total_elapsed = time.time() - t_total_start
    print(f"\n{'='*60}")
    print("  FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  Total docs    : {grand_docs:,}")
    print(f"  Total tokens  : {grand_tokens:,}  ({grand_tokens/1e9:.3f}B)")
    print(f"  Total shards  : {grand_shards}")
    print(f"  Total time    : {total_elapsed/60:.1f} minutes")
    print(f"\n  Output structure:")
    for source_name, weight in sources:
        safe_name = source_name.replace(";", "-")
        n = len(list((output_base / safe_name).glob("*.bin")))
        print(f"    {safe_name:35s} {weight:.1%}  →  {n} shards")

    # Sanity check
    all_ok = sanity_check(output_base, sources)

    if all_ok:
        print(f"\n✓ All shards ready for mid-training!")
        print(f"\nNext step — run mid-training:")
        print(f"  torchrun --standalone --nproc_per_node=2 \\")
        print(f"    -m scripts.mid_train -- \\")
        print(f"    --model-tag=medical_2b \\")
        print(f"    --out-model-tag=medical_2b_mid \\")
        print(f"    --data-dir={args.output_dir} \\")
        print(f"    --num-iterations=5000 \\")
        print(f"    --device-batch-size=2 \\")
        print(f"    --lr-scale=0.1 \\")
        print(f"    --save-every=500 \\")
        print(f"    --run=medical_2b_midtrain")
    else:
        print(f"\n✗ Some shards failed. Fix errors above before training.")
        sys.exit(1)


if __name__ == "__main__":
    main()
