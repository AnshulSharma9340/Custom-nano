"""
Train a tokenizer using our own BPE Tokenizer library.
In the style of GPT-4 tokenizer.
"""
import os
import time
import argparse
import torch
import json
import glob
from nanochat.tokenizer import RustBPETokenizer
from nanochat.common import get_base_dir

# -----------------------------------------------------------------------------
# Parse command line arguments

parser = argparse.ArgumentParser(description='Train a BPE tokenizer')
parser.add_argument('--max-chars',  type=int, default=2_000_000_000, help='Maximum characters to train on (default: 2B)')
parser.add_argument('--doc-cap',    type=int, default=10_000,        help='Maximum characters per document (default: 10,000)')
parser.add_argument('--vocab-size', type=int, default=32768,         help='Vocabulary size (default: 32768 = 2^15)')
parser.add_argument('--data-dir',   type=str, required=True,         help='Directory containing JSONL files')
parser.add_argument('--text-field', type=str, default="text",        help='JSON field name for text (default: text)')
args = parser.parse_args()
print(f"max_chars:  {args.max_chars:,}")
print(f"doc_cap:    {args.doc_cap:,}")
print(f"vocab_size: {args.vocab_size:,}")
print(f"data_dir:   {args.data_dir}")
print(f"text_field: {args.text_field}")

# -----------------------------------------------------------------------------
# Text iterator

def text_iterator():
    """
    1) Read JSONL files from --data-dir
    2) Crop every document to args.doc_cap characters
    3) Break when we've seen args.max_chars characters
    """
    nchars = 0
    files  = sorted(glob.glob(os.path.join(args.data_dir, "*.jsonl")))
    assert files, f"No .jsonl files found in {args.data_dir}"
    print(f"Found {len(files)} JSONL files in {args.data_dir}")
    for fpath in files:
        print(f"  Reading {os.path.basename(fpath)} ...")
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                try:
                    text = json.loads(line).get(args.text_field, "")
                    if len(text) < 30:
                        continue
                    if len(text) > args.doc_cap:
                        text = text[:args.doc_cap]
                    nchars += len(text)
                    yield text
                    if nchars >= args.max_chars:
                        print(f"  Reached {args.max_chars:,} chars — stopping.")
                        return
                except Exception:
                    continue

text_iter = text_iterator()

# -----------------------------------------------------------------------------
# Train the tokenizer
t0 = time.time()
tokenizer = RustBPETokenizer.train_from_iterator(text_iter, args.vocab_size)
t1 = time.time()
train_time = t1 - t0
print(f"Training time: {train_time:.2f}s")

# -----------------------------------------------------------------------------
# Save the tokenizer to disk
base_dir = get_base_dir()
tokenizer_dir = os.path.join(base_dir, "tokenizer")
tokenizer.save(tokenizer_dir)

# -----------------------------------------------------------------------------
# Quick inline sanity check
test_text = """Hello world! This is a test.
Numbers: 123, 4567, 89
Contractions: I'm, you're, it's
Special chars: @#$%^&*()
Unicode: 你好世界 🌍"""
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)
assert decoded == test_text

# -----------------------------------------------------------------------------
# One more thing: we wish to cache a mapping from token id to number of bytes of that token
# for efficient evaluation of bits per byte. Unlike the typical mean loss, this
# allows us to report a loss that is invariant to the vocab size of the tokenizer.
# The bits per byte on the validation set is then one of the primary metrics we care about.
vocab_size = tokenizer.get_vocab_size()
special_set = set(tokenizer.get_special_tokens())
token_strings = [tokenizer.decode([token_id]) for token_id in range(vocab_size)]
token_bytes = []
for token_id in range(vocab_size):
    token_str = token_strings[token_id] # the Python string representation of this token
    if token_str in special_set:
        token_bytes.append(0) # special characters are not counted
    else:
        id_bytes = len(token_str.encode("utf-8")) # number of bytes that make up this token
        token_bytes.append(id_bytes)
token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device='cpu')
token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
with open(token_bytes_path, "wb") as f:
    torch.save(token_bytes, f)
print(f"Saved token_bytes to {token_bytes_path}")

# Log to report
from nanochat.report import get_report
token_bytes_nonzero = (token_bytes[token_bytes > 0]).to(dtype=torch.float32)
get_report().log(section="Tokenizer training", data=[
    vars(args), # argparse command line arguments
    {"train_time": train_time},
    {"num_special_tokens": len(special_set)},
    {
        "token_bytes_min": int(token_bytes_nonzero.min().item()),
        "token_bytes_max": int(token_bytes_nonzero.max().item()),
        "token_bytes_mean": token_bytes_nonzero.mean().item(),
        "token_bytes_std": token_bytes_nonzero.std().item(),
    }
])