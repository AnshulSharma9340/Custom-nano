#!/bin/bash
# =============================================================================
#  medical_train.sh
#  One-file pipeline: tokenizer → .bin shards → 3B base training
#
#  Usage:
#    bash medical_train.sh
#
#  With wandb logging:
#    WANDB_RUN=medical_3b bash medical_train.sh
#
#  In a screen session (recommended — training takes days):
#    screen -L -Logfile medical_train.log -S medical bash medical_train.sh
# =============================================================================

set -e   # stop on any error

# =============================================================================
# ██  CONFIGURE THESE 4 THINGS BEFORE RUNNING  ██
# =============================================================================

# 1. Where your raw JSONL files live on cloud storage
#    S3  example:  s3://your-bucket/pubmed_jsonl/
#    GCS example:  gs://your-bucket/pubmed_jsonl/
PUBMED_CLOUD="s3://your-bucket/pubmed_jsonl/"
FINEWEB_CLOUD="s3://your-bucket/fineweb_jsonl/"
REASONING_CLOUD="s3://your-bucket/reasoning_jsonl/"

# 2. Local working directory (needs ~2TB free for shards + cache)
BASE_DIR="/data/medical_3b"

# 3. Number of GPUs on your machine (you have 2x A100)
NUM_GPUS=2

# 4. Model depth — controls parameter count
#    depth=32 → ~2.7B params   depth=34 → ~3.1B params   depth=36 → ~3.7B params
#    Run step 3 below first to confirm exact count, then set this
DEPTH=34

# =============================================================================
# Derived paths — do not change these
# =============================================================================

RAW_DIR="$BASE_DIR/raw"
PUBMED_DIR="$RAW_DIR/pubmed"
FINEWEB_DIR="$RAW_DIR/fineweb"
REASONING_DIR="$RAW_DIR/reasoning"
BINS_DIR="$BASE_DIR/bins_65k"
CACHE_DIR="$BASE_DIR/nanochat_cache"

export NANOCHAT_BASE_DIR="$CACHE_DIR"
export OMP_NUM_THREADS=1

mkdir -p "$PUBMED_DIR" "$FINEWEB_DIR" "$REASONING_DIR" "$BINS_DIR" "$CACHE_DIR"

# =============================================================================
# wandb setup (optional)
# =============================================================================

if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy   # dummy = skip wandb logging
fi

# =============================================================================
# Python venv setup with uv
# =============================================================================

echo ""
echo "============================================================"
echo "  SETUP — Python venv"
echo "============================================================"

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env" 2>/dev/null || true

[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

echo "[setup] Python venv ready."

# =============================================================================
# Rust + BPE tokenizer build (only needed once)
# =============================================================================

echo ""
echo "============================================================"
echo "  SETUP — Rust tokenizer build"
echo "============================================================"

if ! python -c "import rustbpe" 2>/dev/null; then
    echo "[rust] Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"

    echo "[rust] Building rustbpe (this takes ~10 minutes, once only)..."
    uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
    echo "[rust] Build complete."
else
    echo "[rust] rustbpe already built — skipping."
fi

# =============================================================================
# STEP 1 — Sync raw JSONL data from cloud to local disk
# =============================================================================

echo ""
echo "============================================================"
echo "  STEP 1/4 — Syncing raw JSONL data from cloud"
echo "============================================================"

# Auto-detect S3 vs GCS
sync_from_cloud() {
    local src=$1
    local dst=$2
    if [[ "$src" == s3://* ]]; then
        aws s3 sync "$src" "$dst" --no-progress
    elif [[ "$src" == gs://* ]]; then
        gsutil -m cp -r "$src" "$dst"
    else
        echo "ERROR: CLOUD path must start with s3:// or gs://"
        exit 1
    fi
}

echo "[sync] Syncing PubMed (80% of training data)..."
sync_from_cloud "$PUBMED_CLOUD" "$PUBMED_DIR"

echo "[sync] Syncing FineWeb-Edu (12% of training data)..."
sync_from_cloud "$FINEWEB_CLOUD" "$FINEWEB_DIR"

echo "[sync] Syncing Reasoning (8% of training data)..."
sync_from_cloud "$REASONING_CLOUD" "$REASONING_DIR"

echo "[sync] All data synced to $RAW_DIR"
du -sh "$PUBMED_DIR" "$FINEWEB_DIR" "$REASONING_DIR"

# =============================================================================
# STEP 2 — Train tokenizer on your medical data
# =============================================================================

echo ""
echo "============================================================"
echo "  STEP 2/4 — Training medical tokenizer (vocab=65536)"
echo "============================================================"

# Train on 4B characters from PubMed (best for medical vocabulary)
python -m scripts.tok_train \
    --max-chars=4000000000 \
    --vocab-size=65536 \
    --data-dir="$PUBMED_DIR"

# Verify compression ratio — should be 4.5–5.5 chars/token for medical text
echo ""
echo "[tokenizer] Evaluating tokenizer compression..."
python -m scripts.tok_eval

echo "[tokenizer] Done. Check compression ratio above — expect 4.5–5.5 chars/token."

# =============================================================================
# STEP 3 — Create .bin shards from your JSONL data
# =============================================================================

echo ""
echo "============================================================"
echo "  STEP 3/4 — Creating .bin shards (this takes 2–4 hours)"
echo "============================================================"

# Write the shard-making script inline and run it
python - <<PYEOF
import json, numpy as np
from pathlib import Path
from nanochat.tokenizer import get_tokenizer

SHARD_SIZE = 100_000_000   # 100M tokens per shard
OUTPUT_DIR = Path("$BINS_DIR")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Data sources with mixing weights
SOURCES = [
    ("$PUBMED_DIR",    0.80),
    ("$FINEWEB_DIR",   0.12),
    ("$REASONING_DIR", 0.08),
]

tokenizer = get_tokenizer()
print(f"[shards] Tokenizer vocab size: {tokenizer.vocab_size}")
assert tokenizer.vocab_size == 65536, \
    f"ERROR: Expected vocab 65536, got {tokenizer.vocab_size}. Tokenizer mismatch!"

shard_idx = 0
buf       = []
buf_size  = 0
total_docs = 0

def flush_shard():
    global shard_idx, buf, buf_size
    arr  = np.array(buf, dtype=np.uint16)
    path = OUTPUT_DIR / f"shard_{shard_idx:06d}.bin"
    arr.tofile(str(path))
    print(f"[shards] Saved {path}  —  {len(arr):,} tokens")
    shard_idx += 1
    buf      = []
    buf_size = 0

for data_dir, weight in SOURCES:
    files = sorted(Path(data_dir).glob("*.jsonl"))
    print(f"\n[shards] Processing {data_dir}  ({len(files)} files,  weight={weight})")
    if not files:
        print(f"[shards] WARNING: No .jsonl files found in {data_dir} — skipping.")
        continue

    for fpath in files:
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                try:
                    text = json.loads(line).get("text", "")
                    if len(text) < 30:
                        continue
                    ids = tokenizer.encode(text, bos=True, eos=True)
                    buf.extend(ids)
                    buf_size += len(ids)
                    total_docs += 1
                    if total_docs % 500_000 == 0:
                        print(f"[shards]   {total_docs:,} docs processed, "
                              f"{shard_idx} shards written so far...")
                    if buf_size >= SHARD_SIZE:
                        flush_shard()
                except Exception:
                    continue

# flush the final partial shard
if buf:
    flush_shard()

print(f"\n[shards] Done.")
print(f"[shards]   Total docs    : {total_docs:,}")
print(f"[shards]   Total shards  : {shard_idx}")
print(f"[shards]   Approx tokens : ~{shard_idx * SHARD_SIZE / 1e9:.1f}B")
print(f"[shards]   Output dir    : {OUTPUT_DIR}")

# Quick sanity check on first shard
first = sorted(OUTPUT_DIR.glob("*.bin"))[0]
data  = np.fromfile(first, dtype=np.uint16)
print(f"\n[shards] Sanity check on {first.name}:")
print(f"[shards]   Tokens : {len(data):,}")
print(f"[shards]   Max ID : {data.max()}  (must be < 65536)")
print(f"[shards]   Min ID : {data.min()}")
assert data.max() < 65536, "ERROR: Token ID exceeds vocab size — tokenizer mismatch!"
print(f"[shards]   ✓ Shard looks good.")
PYEOF

echo ""
echo "[shards] All .bin shards created at $BINS_DIR"

# =============================================================================
# STEP 4 — Base training (3B model from scratch)
# =============================================================================

echo ""
echo "============================================================"
echo "  STEP 4/4 — Base training  (depth=$DEPTH, gpus=$NUM_GPUS)"
echo "============================================================"

# Print param count before training so you can confirm it's ~3B
python - <<PYEOF
from nanochat.gpt import GPT, GPTConfig
cfg = GPTConfig(depth=$DEPTH, vocab_size=65536)
m   = GPT(cfg)
p   = sum(x.numel() for x in m.parameters()) / 1e9
print(f"[model] depth=$DEPTH  →  {p:.2f}B parameters")
print(f"[model] vocab_size=65536")
print(f"[model] Starting training on $NUM_GPUS x A100...")
PYEOF

# Start training
torchrun \
    --standalone \
    --nproc_per_node=$NUM_GPUS \
    -m scripts.base_train -- \
    --depth=$DEPTH \
    --run="$WANDB_RUN" \
    --model-tag="medical_3b" \
    --data-dir="$BINS_DIR" \
    --device-batch-size=4

# =============================================================================
# Evaluate after training
# =============================================================================

echo ""
echo "============================================================"
echo "  Evaluating base model"
echo "============================================================"

torchrun \
    --standalone \
    --nproc_per_node=$NUM_GPUS \
    -m scripts.base_eval -- \
    --device-batch-size=4

echo ""
echo "============================================================"
echo "  Training complete!"
echo "  To chat with your model:"
echo "    python -m scripts.chat_web"
echo "  Then open  http://<your-server-ip>:8000"
echo "============================================================"