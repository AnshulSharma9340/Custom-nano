"""
Mid-training script for nanochat medical model.
Continues training from a base model checkpoint on new/additional medical data.

Run as:
    torchrun --standalone --nproc_per_node=2 -m scripts.mid_train -- \
        --model-tag=medical_2b \
        --data-dir=data/midtrain_bins \
        --resume-from-step=7812 \
        --num-iterations=5000 \
        --device-batch-size=2 \
        --run=medical_2b_midtrain

Key differences from base_train.py:
1. Loads from existing base checkpoint automatically
2. Lower learning rate (10x smaller) to avoid catastrophic forgetting
3. save-every=500 by default
4. Separate model-tag for mid-trained checkpoint
"""

import os
import numpy as np
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import json
import time
import math
import argparse
from dataclasses import asdict
from contextlib import contextmanager

import wandb
import torch
import torch.distributed as dist

from nanochat.gpt import GPT, GPTConfig, Linear
from nanochat.dataloader import bin_distributed_data_loader_with_state
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type, get_peak_flops, COMPUTE_DTYPE, COMPUTE_DTYPE_REASON, is_ddp_initialized
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from nanochat.flash_attention import HAS_FA3
from scripts.base_eval import evaluate_core
print_banner()

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Mid-train base model on additional medical data")

# Logging
parser.add_argument("--run",              type=str,   default="dummy",         help="wandb run name")
# Runtime
parser.add_argument("--device-type",      type=str,   default="",              help="cuda|cpu|mps")
# Model
parser.add_argument("--model-tag",        type=str,   required=True,           help="model tag to load AND save (e.g. medical_2b)")
parser.add_argument("--out-model-tag",    type=str,   default=None,            help="output model tag (default: {model-tag}_midtrain)")
parser.add_argument("--depth",            type=int,   default=24,              help="depth of the Transformer model")
parser.add_argument("--aspect-ratio",     type=int,   default=64,              help="model_dim = depth * aspect_ratio")
parser.add_argument("--head-dim",         type=int,   default=128,             help="target head dimension")
parser.add_argument("--max-seq-len",      type=int,   default=2048,            help="max context length")
parser.add_argument("--window-pattern",   type=str,   default="L",             help="sliding window pattern")
# Data
parser.add_argument("--data-dir",         type=str,   required=True,           help="path to directory containing .bin shard subfolders")
# Resume
parser.add_argument("--resume-from-step", type=int,   default=-1,              help="step to resume from (-1 = auto detect last step)")
# Training horizon
parser.add_argument("--num-iterations",   type=int,   default=5000,            help="number of mid-training steps")
# Optimization — lower LR than base training to avoid forgetting
parser.add_argument("--device-batch-size",type=int,   default=2,               help="per-device batch size")
parser.add_argument("--total-batch-size", type=int,   default=-1,              help="total batch size in tokens (-1 = auto)")
parser.add_argument("--lr-scale",         type=float, default=0.1,             help="LR multiplier vs base training (default: 0.1 = 10x smaller)")
parser.add_argument("--embedding-lr",     type=float, default=0.3,             help="base embedding LR (scaled by --lr-scale)")
parser.add_argument("--unembedding-lr",   type=float, default=0.008,           help="base unembedding LR (scaled by --lr-scale)")
parser.add_argument("--weight-decay",     type=float, default=0.28,            help="weight decay")
parser.add_argument("--matrix-lr",        type=float, default=0.02,            help="base matrix LR (scaled by --lr-scale)")
parser.add_argument("--scalar-lr",        type=float, default=0.5,             help="base scalar LR (scaled by --lr-scale)")
parser.add_argument("--warmup-steps",     type=int,   default=100,             help="LR warmup steps")
parser.add_argument("--warmdown-ratio",   type=float, default=0.2,             help="ratio of steps for LR warmdown")
parser.add_argument("--final-lr-frac",    type=float, default=0.1,             help="final LR as fraction of peak LR")
# Evaluation
parser.add_argument("--eval-every",       type=int,   default=250,             help="evaluate val bpb every N steps")
parser.add_argument("--eval-tokens",      type=int,   default=80*524288,       help="tokens for val loss eval")
parser.add_argument("--core-metric-every",type=int,   default=-1,              help="CORE metric every N steps (-1=disable)")
parser.add_argument("--sample-every",     type=int,   default=1000,            help="sample every N steps (-1=disable)")
parser.add_argument("--save-every",       type=int,   default=500,             help="save checkpoint every N steps")

args = parser.parse_args()
user_config = vars(args).copy()

# -----------------------------------------------------------------------------
# Output model tag
out_model_tag = args.out_model_tag if args.out_model_tag else f"{args.model_tag}_midtrain"
print0(f"Input  model tag : {args.model_tag}")
print0(f"Output model tag : {out_model_tag}")

# -----------------------------------------------------------------------------
# Compute init
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops  = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')

print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")

# wandb
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(
    project="nanochat", name=args.run, config=user_config)

# Flash Attention status
from nanochat.flash_attention import USE_FA3
if not USE_FA3:
    print0("!" * 80)
    print0("WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback")
    print0("!" * 80)

# -----------------------------------------------------------------------------
# Tokenizer
tokenizer    = get_tokenizer()
token_bytes  = get_token_bytes(device=device)
vocab_size   = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# -----------------------------------------------------------------------------
# Build model
def build_model_meta(depth):
    base_dim  = depth * args.aspect_ratio
    model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    num_heads = model_dim // args.head_dim
    config    = GPTConfig(
        sequence_len=args.max_seq_len, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=args.window_pattern,
    )
    with torch.device("meta"):
        model_meta = GPT(config)
    return model_meta

model        = build_model_meta(args.depth)
model_config = model.config
print0(f"Model config:\n{json.dumps(asdict(model_config), indent=2)}")
model.to_empty(device=device)
model.init_weights()

# -----------------------------------------------------------------------------
# Load base checkpoint
base_dir        = get_base_dir()
input_ckpt_dir  = os.path.join(base_dir, "base_checkpoints", args.model_tag)
output_ckpt_dir = os.path.join(base_dir, "base_checkpoints", out_model_tag)

# Auto-detect last step if not specified
if args.resume_from_step == -1:
    import glob
    model_files = sorted(glob.glob(os.path.join(input_ckpt_dir, "model_*.pt")))
    assert model_files, f"No checkpoints found in {input_ckpt_dir}"
    last_file         = model_files[-1]
    resume_step       = int(os.path.basename(last_file).split("_")[1].split(".")[0])
    print0(f"Auto-detected last checkpoint step: {resume_step}")
else:
    resume_step = args.resume_from_step

print0(f"Loading checkpoint from {input_ckpt_dir} at step {resume_step}")
model_data, _, _ = load_checkpoint(
    input_ckpt_dir, resume_step, device, load_optimizer=False, rank=ddp_rank)
model.load_state_dict(model_data, strict=True, assign=True)
del model_data
print0(f"✓ Loaded base model checkpoint at step {resume_step}")

# -----------------------------------------------------------------------------
# Compile
orig_model = model
model      = torch.compile(model, dynamic=False)

# -----------------------------------------------------------------------------
# Parameter counts
param_counts       = model.num_scaling_params()
num_params         = param_counts['total']
num_flops_per_token = model.estimate_flops()
print0(f"Parameter counts:")
for key, value in param_counts.items():
    print0(f"  {key:24s}: {value:,}")

# -----------------------------------------------------------------------------
# Batch size
B_REF = 2**19
if args.total_batch_size == -1:
    total_batch_size = B_REF  # keep same as base training
else:
    total_batch_size = args.total_batch_size
print0(f"Total batch size: {total_batch_size:,} tokens")

# -----------------------------------------------------------------------------
# LR scaling — key difference from base training
# Mid-training uses much lower LR to avoid catastrophic forgetting
lr_scale = args.lr_scale
print0(f"LR scale vs base training: {lr_scale} (10x smaller = safer mid-training)")

batch_lr_scale    = (total_batch_size / B_REF) ** 0.5
weight_decay_scaled = args.weight_decay * math.sqrt(total_batch_size / B_REF) * (B_REF / total_batch_size)

# -----------------------------------------------------------------------------
# Optimizer
optimizer = model.setup_optimizer(
    unembedding_lr = args.unembedding_lr * batch_lr_scale * lr_scale,
    embedding_lr   = args.embedding_lr   * batch_lr_scale * lr_scale,
    scalar_lr      = args.scalar_lr      * batch_lr_scale * lr_scale,
    matrix_lr      = args.matrix_lr      * batch_lr_scale * lr_scale,
    weight_decay   = weight_decay_scaled,
)

# -----------------------------------------------------------------------------
# DataLoaders
num_iterations   = args.num_iterations
total_tokens     = total_batch_size * num_iterations
print0(f"Mid-training iterations : {num_iterations:,}")
print0(f"Mid-training tokens     : {total_tokens:,}  ({total_tokens/1e9:.2f}B)")

train_loader = bin_distributed_data_loader_with_state(
    data_dir=args.data_dir,
    B=args.device_batch_size,
    T=args.max_seq_len,
    split="train",
    device=device,
    resume_state_dict=None,
    val_ratio=0.05,
    dtype=np.uint16
)

def build_val_loader():
    loader = bin_distributed_data_loader_with_state(
        data_dir=args.data_dir,
        B=args.device_batch_size,
        T=args.max_seq_len,
        split="val",
        device=device,
        val_ratio=0.05,
        dtype=np.uint16
    )
    for x, y, state in loader:
        yield x, y

x, y, dataloader_state_dict = next(train_loader)

# -----------------------------------------------------------------------------
# LR schedule
def get_lr_multiplier(it):
    warmup_iters   = args.warmup_steps
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * args.final_lr_frac

def get_muon_momentum(it):
    if it < 400:
        frac = it / 400
        return (1 - frac) * 0.85 + frac * 0.95
    return 0.95

def get_weight_decay(it):
    return weight_decay_scaled * 0.5 * (1 + math.cos(math.pi * it / num_iterations))

# -----------------------------------------------------------------------------
# Training loop
step             = 0
val_bpb          = None
min_val_bpb      = float("inf")
smooth_train_loss = 0
total_training_time = 0
results          = {}

tokens_per_fwdbwd       = args.device_batch_size * args.max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd

print0(f"Gradient accumulation steps: {grad_accum_steps}")
print0(f"Starting mid-training from base step {resume_step}...")

while True:
    last_step = step == num_iterations

    # Eval val bpb
    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        val_bpb    = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Val bpb: {val_bpb:.6f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({"step": step, "val/bpb": val_bpb, "total_training_time": total_training_time})
        model.train()

    # CORE metric
    if args.core_metric_every > 0 and (last_step or (step > 0 and step % args.core_metric_every == 0)):
        model.eval()
        results = evaluate_core(orig_model, tokenizer, device, max_per_task=500)
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log({"step": step, "core_metric": results["core_metric"]})
        model.train()

    # Sample
    if args.sample_every > 0 and master_process and (last_step or (step > 0 and step % args.sample_every == 0)):
        model.eval()
        prompts = [
            "The patient presented with fever and",
            "Diagnosis: Type 2 diabetes mellitus. Treatment plan:",
            "Acetaminophen is used to treat",
            "The most common cause of pneumonia is",
        ]
        engine = Engine(orig_model, tokenizer)
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=32, temperature=0.7)
            print0(tokenizer.decode(sample[0]))
        model.train()

    # Save checkpoint
    if last_step or (step > 0 and args.save_every > 0 and step % args.save_every == 0):
        save_checkpoint(
            output_ckpt_dir,
            step,
            orig_model.state_dict(),
            optimizer.state_dict(),
            {
                "step": step,
                "val_bpb": val_bpb,
                "model_config": asdict(model_config),
                "user_config": user_config,
                "base_model_tag": args.model_tag,
                "base_resume_step": resume_step,
                "dataloader_state_dict": dataloader_state_dict,
                "loop_state": {
                    "min_val_bpb": min_val_bpb,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                },
            },
            rank=ddp_rank,
        )
        print0(f"✓ Checkpoint saved at step {step} → {output_ckpt_dir}")

    if last_step:
        break

    # Training step
    synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        loss       = model(x, y)
        train_loss = loss.detach()
        loss       = loss / grad_accum_steps
        loss.backward()
        x, y, dataloader_state_dict = next(train_loader)

    lrm             = get_lr_multiplier(step)
    muon_momentum   = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"]     = muon_momentum
            group["weight_decay"] = muon_weight_decay

    optimizer.step()
    model.zero_grad(set_to_none=True)
    train_loss_f = train_loss.item()
    synchronize()
    t1 = time.time()
    dt = t1 - t0

    # Logging
    ema_beta          = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done          = 100 * step / num_iterations
    tok_per_sec       = int(total_batch_size / dt)
    flops_per_sec     = num_flops_per_token * total_batch_size / dt
    mfu               = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)

    if step > 10:
        total_training_time += dt

    steps_done = step - 10
    if steps_done > 0:
        avg_time_per_step = total_training_time / steps_done
        remaining_steps   = num_iterations - step
        eta_seconds       = remaining_steps * avg_time_per_step
        eta_str           = f" | eta: {eta_seconds/60:.1f}m"
    else:
        eta_str = ""

    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | "
           f"loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | "
           f"dt: {dt*1000:.2f}ms | tok/sec: {tok_per_sec:,} | "
           f"mfu: {mfu:.2f}{eta_str}")

    if step % 100 == 0:
        wandb_run.log({
            "step": step,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "total_training_time": total_training_time,
        })

    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif step % 5000 == 0:
        gc.collect()

    step += 1

# Final stats
print0(f"Peak memory    : {get_max_memory()/1024/1024:.2f} MiB")
print0(f"Total mid-train time: {total_training_time/60:.2f}m")
print0(f"Min val bpb    : {min_val_bpb:.6f}")
print0(f"Checkpoint saved at: {output_ckpt_dir}")

wandb_run.finish()
compute_cleanup()
