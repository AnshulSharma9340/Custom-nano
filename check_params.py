import torch
from nanochat.gpt import GPTConfig, GPT
from nanochat.tokenizer import get_tokenizer

# The exact math from your bash script
depth = 28
aspect_ratio = 106
head_dim = 128

# base_train.py logic
base_dim = depth * aspect_ratio
model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
num_heads = model_dim // head_dim

# Grab vocab size
tokenizer = get_tokenizer()
vocab_size = tokenizer.get_vocab_size()

config = GPTConfig(
    sequence_len=2048, 
    vocab_size=vocab_size,
    n_layer=depth, 
    n_head=num_heads, 
    n_kv_head=num_heads, 
    n_embd=model_dim
)

print(f"Testing Architecture: Layers={depth}, Emb={model_dim}, Heads={num_heads}")

# Initialize on "meta" device (instant, uses 0 RAM)
with torch.device("meta"):
    model = GPT(config)

param_counts = model.num_scaling_params()

print("\n--- Parameter Breakdown ---")
for key, value in param_counts.items():
    print(f"{key:24s}: {value:,}")