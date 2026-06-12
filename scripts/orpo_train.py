"""
ORPO (Odds Ratio Preference Optimization) training for medical model alignment.
More stable than DPO - single model training without reference model issues.

Run as:
torchrun --standalone --nproc_per_node=2 -m scripts.orpo_train -- \
    --model-tag=medical_2b_mid \
    --model-step=1358 \
    --out-model-tag=medical_2b_orpo \
    --data-path=data/dpo_data/dpo_combined.jsonl \
    --num-iterations=100 \
    --device-batch-size=1 \
    --lambda-orpo=0.01 \
    --run=medical_orpo_safe
"""

import os
import json
import math
import time
import argparse
import random
import gc

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from nanochat.common import get_base_dir, print0, compute_init, compute_cleanup, autodetect_device_type, DummyWandb, COMPUTE_DTYPE
from nanochat.checkpoint_manager import load_model, save_checkpoint
from nanochat.tokenizer import get_tokenizer

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="ORPO training for nanochat medical model")
parser.add_argument("--run",               type=str,   default="orpo_run")
parser.add_argument("--device-type",       type=str,   default="")
parser.add_argument("--model-tag",         type=str,   required=True,
                    help="input model tag (e.g. medical_2b_mid)")
parser.add_argument("--out-model-tag",     type=str,   default=None,
                    help="output model tag (default: model-tag + _orpo)")
parser.add_argument("--model-step",        type=int,   default=-1,
                    help="checkpoint step to load (-1 = latest)")
parser.add_argument("--data-path",         type=str,   required=True,
                    help="JSONL with prompt/chosen/rejected fields")
parser.add_argument("--val-ratio",         type=float, default=0.05)
parser.add_argument("--num-iterations",    type=int,   default=100)
parser.add_argument("--device-batch-size", type=int,   default=1)
parser.add_argument("--total-batch-size",  type=int,   default=-1)
parser.add_argument("--max-seq-len",       type=int,   default=512)
parser.add_argument("--lambda-orpo",       type=float, default=0.01,
                    help="ORPO preference strength (0.01-0.1)")
parser.add_argument("--embedding-lr",      type=float, default=None)
parser.add_argument("--unembedding-lr",    type=float, default=None)
parser.add_argument("--matrix-lr",        type=float, default=None)
parser.add_argument("--lr-scale",         type=float, default=0.00001,
                    help="multiply all LRs by this (very small for stability)")
parser.add_argument("--warmup-ratio",      type=float, default=0.1)
parser.add_argument("--warmdown-ratio",    type=float, default=0.2)
parser.add_argument("--eval-every",        type=int,   default=25,
                    help="validate every N steps (-1 = disable)")
parser.add_argument("--save-every",        type=int,   default=25,
                    help="save checkpoint every N steps (-1 = end only)")
parser.add_argument("--seed",              type=int,   default=42)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Distributed setup
# ---------------------------------------------------------------------------

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0

torch.manual_seed(args.seed + ddp_rank); random.seed(args.seed + ddp_rank)

base_dir       = get_base_dir()
out_model_tag  = args.out_model_tag or (args.model_tag + "_orpo")
checkpoint_dir = os.path.join(base_dir, "orpo_checkpoints", out_model_tag)
if master_process:
    os.makedirs(checkpoint_dir, exist_ok=True)

print0(f"""
ORPO TRAINING — NANOCHAT MEDICAL MODEL
  Input  model tag : {args.model_tag}
  Output model tag : {out_model_tag}
  Lambda ORPO      : {args.lambda_orpo}
  Data path        : {args.data_path}
  Max seq len      : {args.max_seq_len}
  LR scale         : {args.lr_scale}
""")

# ---------------------------------------------------------------------------
# WandB
# ---------------------------------------------------------------------------

use_wandb = False
if master_process and args.run != "dummy":
    try:
        import wandb
        wandb.init(project="nanochat-orpo", name=args.run, config=vars(args))
        use_wandb = True
    except ImportError:
        print0("wandb not installed — skipping wandb logging")

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

tokenizer = get_tokenizer()
print0(f"Tokenizer vocab size: {tokenizer.get_vocab_size():,}")

step_to_load = args.model_step if args.model_step >= 0 else None

# Load model for training
model, _, meta = load_model(
    "sft", device, phase="train",
    model_tag=args.model_tag, step=step_to_load,
)
model = model.to(device).train()

n_params = sum(p.numel() for p in model.parameters()) / 1e9
print0(f"Model: {n_params:.2f}B params (trainable)")

# DDP-wrap the model
model = DDP(model, device_ids=[ddp_local_rank])

# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

emb_lr   = (args.embedding_lr   or meta.get("embedding_lr",   0.3))   * args.lr_scale
unemb_lr = (args.unembedding_lr or meta.get("unembedding_lr", 0.008)) * args.lr_scale
mat_lr   = (args.matrix_lr      or meta.get("matrix_lr",      0.02))  * args.lr_scale

print0(f"LRs (after x{args.lr_scale}):  emb={emb_lr:.7f}  unemb={unemb_lr:.7f}  mat={mat_lr:.7f}")

optimizer = model.module.setup_optimizer(
    unembedding_lr=unemb_lr,
    embedding_lr=emb_lr, 
    matrix_lr=mat_lr,
    weight_decay=0.0
)
for group in optimizer.param_groups:
    group["initial_lr"] = group["lr"]

# ---------------------------------------------------------------------------
# ORPO Dataset
# ---------------------------------------------------------------------------

class ORPODataset:
    """
    ORPO dataset that tokenizes prompt/chosen/rejected triplets.
    Same format as DPO but uses different loss function.
    """

    def __init__(self, rows, tokenizer, max_seq_len):
        self.rows        = rows
        self.tokenizer   = tokenizer
        self.max_seq_len = max_seq_len
        self._bos    = tokenizer.get_bos_token_id()
        self._usr_s  = self._tok1("<|user_start|>")
        self._usr_e  = self._tok1("<|user_end|>")
        self._ast_s  = self._tok1("<|assistant_start|>")
        self._ast_e  = self._tok1("<|assistant_end|>")

    def _tok1(self, s):
        ids = self.tokenizer.encode(s)
        return ids[0] if ids else 0

    def _build(self, prompt, response):
        """Returns (token_ids, loss_mask) for one (prompt, response) pair."""
        p_ids  = self.tokenizer.encode(prompt.strip())
        r_ids  = self.tokenizer.encode(response.strip())

        prefix = [self._bos, self._usr_s] + p_ids + [self._usr_e, self._ast_s]
        suffix = r_ids + [self._ast_e]
        full   = prefix + suffix

        # Truncate if needed
        if len(full) > self.max_seq_len:
            max_resp = self.max_seq_len - len(prefix)
            if max_resp < 4:
                full      = full[:self.max_seq_len]
                loss_mask = [0] * self.max_seq_len
            else:
                suffix = suffix[:max_resp]
                full   = prefix + suffix
                loss_mask = [0] * len(prefix) + [1] * len(suffix)
        else:
            loss_mask = [0] * len(prefix) + [1] * len(suffix)

        assert len(full) == len(loss_mask)
        return full, loss_mask

    def get(self, idx):
        row = self.rows[idx % len(self.rows)]
        chosen_ids,   chosen_mask   = self._build(row["prompt"], row["chosen"])
        rejected_ids, rejected_mask = self._build(row["prompt"], row["rejected"])
        return {
            "chosen_ids":    chosen_ids,
            "chosen_mask":   chosen_mask,
            "rejected_ids":  rejected_ids,
            "rejected_mask": rejected_mask,
        }

    def __len__(self):
        return len(self.rows)


def load_orpo_jsonl(filepath, val_ratio=0.05, seed=42):
    """Load and split a JSONL file into train/val rows."""
    rows = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "prompt" in obj and "chosen" in obj and "rejected" in obj:
                    rows.append(obj)
            except Exception:
                continue
    if not rows:
        raise ValueError(f"No valid ORPO rows found in {filepath}")
    random.seed(seed)
    random.shuffle(rows)
    n_val  = max(1, int(len(rows) * val_ratio))
    return rows[n_val:], rows[:n_val]   # train, val


def collate(samples, device, max_seq_len):
    """Pad a list of ORPO samples and return (B,T) tensors."""
    def pad(seqs, pad_id=0):
        max_len = min(max(len(s) for s in seqs), max_seq_len)
        out = []
        for s in seqs:
            s = s[:max_len]
            s = s + [pad_id] * (max_len - len(s))
            out.append(s)
        return torch.tensor(out, dtype=torch.long, device=device)
    chosen_ids   = pad([s["chosen_ids"]    for s in samples])
    chosen_mask  = pad([s["chosen_mask"]   for s in samples])
    rejected_ids = pad([s["rejected_ids"]  for s in samples])
    rejected_mask= pad([s["rejected_mask"] for s in samples])
    return chosen_ids, chosen_mask.float(), rejected_ids, rejected_mask.float()


# ---------------------------------------------------------------------------
# Core ORPO computation
# ---------------------------------------------------------------------------

def compute_log_probs(model, input_ids, loss_mask):
    """
    Compute sum of log-probs on response tokens only.
    """
    inputs  = input_ids[:, :-1]   # (B, T-1)
    targets = input_ids[:, 1:]    # (B, T-1)
    mask    = loss_mask[:, 1:]    # (B, T-1)

    with torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
        logits = model(inputs)        # (B, T-1, V)

    log_probs       = F.log_softmax(logits.float(), dim=-1)  # (B, T-1, V)
    token_log_probs = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)  # (B, T-1)
    return (token_log_probs * mask).sum(dim=-1)  # (B,)


def orpo_loss(chosen_logp, rejected_logp, lambda_orpo=0.01):
    """
    ORPO loss: Odds Ratio Preference Optimization
    More stable than DPO as it doesn't need reference model.
    
    Returns: (loss, chosen_rewards, rejected_rewards)
    """
    # Compute log odds ratio
    log_odds = chosen_logp - rejected_logp
    
    # ORPO objective: maximize log-sigmoid of the odds ratio
    loss = -F.logsigmoid(lambda_orpo * log_odds).mean()
    
    # For logging (convert to reward-like values)
    chosen_rewards = chosen_logp.detach()
    rejected_rewards = rejected_logp.detach()
    
    return loss, chosen_rewards, rejected_rewards


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

train_rows, val_rows = load_orpo_jsonl(args.data_path, val_ratio=args.val_ratio)
train_dataset = ORPODataset(train_rows, tokenizer, args.max_seq_len)
val_dataset   = ORPODataset(val_rows,   tokenizer, args.max_seq_len)
print0(f"ORPO train: {len(train_dataset):,} pairs   |   val: {len(val_dataset):,} pairs")

total_batch_size = (args.total_batch_size if args.total_batch_size > 0
                    else args.device_batch_size * ddp_world_size)
grad_accum_steps = max(1, total_batch_size // (args.device_batch_size * ddp_world_size))
print0(f"device_batch={args.device_batch_size}  world={ddp_world_size}  grad_accum={grad_accum_steps}  total_batch={total_batch_size}")

num_iterations = args.num_iterations
if num_iterations < 0:
    num_iterations = math.ceil(len(train_dataset) / total_batch_size)
print0(f"Training for {num_iterations} steps")

# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def get_lr_scale(step, total, warmup_ratio, warmdown_ratio):
    w_up   = int(total * warmup_ratio)
    w_down = int(total * warmdown_ratio)
    if step < w_up:
        return (step + 1) / max(w_up, 1)
    if step >= total - w_down:
        return max(0.0, (total - step) / max(w_down, 1))
    return 1.0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_val():
    model.eval()
    total_loss = total_acc = total_margin = n = 0
    max_val = min(len(val_dataset), 200)
    for i in range(0, max_val, args.device_batch_size):
        idx     = list(range(i, min(i + args.device_batch_size, max_val)))
        samples = [val_dataset.get(j) for j in idx]
        c_ids, c_msk, r_ids, r_msk = collate(samples, device, args.max_seq_len)

        chosen_logp = compute_log_probs(model.module, c_ids, c_msk)
        rejected_logp = compute_log_probs(model.module, r_ids, r_msk)

        loss, cr, rr = orpo_loss(chosen_logp, rejected_logp, args.lambda_orpo)
        B = c_ids.size(0)
        total_loss   += loss.item() * B
        total_acc    += (cr > rr).float().sum().item()
        total_margin += (cr - rr).mean().item() * B
        n += B

    model.train()
    return dict(
        val_loss   = total_loss   / max(n, 1),
        val_acc    = total_acc    / max(n, 1),
        val_margin = total_margin / max(n, 1),
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

print0("Starting ORPO training...\n")

t0         = time.time()
total_time = 0.0

for step in range(num_iterations):
    # LR schedule
    lr_s = get_lr_scale(step, num_iterations, args.warmup_ratio, args.warmdown_ratio)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lr_s

    # Gradient accumulation
    optimizer.zero_grad(set_to_none=True)
    acc_loss = acc_acc = acc_chosen_r = acc_rejected_r = 0.0

    for micro in range(grad_accum_steps):
        # Each rank processes different samples
        global_micro = step * grad_accum_steps + micro
        start  = (global_micro * args.device_batch_size * ddp_world_size
                  + ddp_rank * args.device_batch_size) % len(train_dataset)
        idx    = [(start + k) % len(train_dataset) for k in range(args.device_batch_size)]
        samples= [train_dataset.get(j) for j in idx]

        c_ids, c_msk, r_ids, r_msk = collate(samples, device, args.max_seq_len)

        # Compute log-probs for both chosen and rejected
        chosen_logp = compute_log_probs(model.module, c_ids, c_msk)
        rejected_logp = compute_log_probs(model.module, r_ids, r_msk)

        # ORPO loss
        loss, chosen_rew, rejected_rew = orpo_loss(chosen_logp, rejected_logp, args.lambda_orpo)
        (loss / grad_accum_steps).backward()

        acc_loss       += loss.item()       / grad_accum_steps
        acc_chosen_r   += chosen_rew.mean().item()   / grad_accum_steps
        acc_rejected_r += rejected_rew.mean().item() / grad_accum_steps
        acc_acc        += (chosen_rew > rejected_rew).float().mean().item() / grad_accum_steps

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    torch.cuda.synchronize()
    t1         = time.time()
    dt_ms      = (t1 - t0) * 1000
    total_time += (t1 - t0) / 60.0
    t0         = t1
    avg_s      = total_time * 60 / max(step + 1, 1)
    eta_min    = (num_iterations - step - 1) * avg_s / 60.0
    pct        = (step + 1) / num_iterations * 100

    print0(
        f"step {step+1:05d}/{num_iterations} ({pct:.1f}%) | "
        f"loss: {acc_loss:.6f} | "
        f"acc: {acc_acc*100:.1f}% | "
        f"log_margin: {acc_chosen_r - acc_rejected_r:.4f} | "
        f"lrm: {lr_s:.4f} | "
        f"dt: {dt_ms:.0f}ms | "
        f"total: {total_time:.1f}m | "
        f"eta: {eta_min:.1f}m"
    )

    if use_wandb and master_process:
        wandb.log({
            "train/loss":           acc_loss,
            "train/accuracy":       acc_acc,
            "train/log_margin":     acc_chosen_r - acc_rejected_r,
            "train/chosen_logp":    acc_chosen_r,
            "train/rejected_logp":  acc_rejected_r,
            "train/lr_scale":       lr_s,
        }, step=step)

    last_step = (step + 1 == num_iterations)

    # Evaluation
    if args.eval_every > 0 and (last_step or (step + 1) % args.eval_every == 0):
        metrics = evaluate_val()
        print0(
            f"Step {step+1:05d} | Val loss: {metrics['val_loss']:.6f} | "
            f"Val acc: {metrics['val_acc']*100:.1f}% | "
            f"Val margin: {metrics['val_margin']:.4f}"
        )
        if use_wandb and master_process:
            wandb.log({"val/" + k: v for k, v in metrics.items()}, step=step)

    # Checkpoint saving
    should_save = last_step or (args.save_every > 0 and (step + 1) % args.save_every == 0)
    if should_save and master_process:
        save_checkpoint(
            checkpoint_dir,
            step + 1,
            model.module.state_dict(),
            optimizer.state_dict(),
            meta_data={
                "model_tag":    out_model_tag,
                "source_tag":   args.model_tag,
                "lambda_orpo":  args.lambda_orpo,
                "embedding_lr": emb_lr,
                "unembedding_lr": unemb_lr,
                "matrix_lr":    mat_lr,
                "model_config": dict(model.module.config.__dict__),
            },
            rank=ddp_rank,
        )
        print0(f"Checkpoint saved: {checkpoint_dir}/model_{step+1:06d}.pt")

    # Garbage collection to avoid memory leaks during long training
    if step > 0 and step % 20 == 0:
        gc.collect()

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------

print0(f"""
ORPO training complete!
  Total time   : {total_time:.1f} min
  Checkpoint   : {checkpoint_dir}

Next step — test the aligned model:
  python -m scripts.chat_web --model-tag={out_model_tag} --source=orpo
""")

if use_wandb and master_process:
    wandb.finish()

compute_cleanup()
