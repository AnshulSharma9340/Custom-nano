"""
Evaluate compression ratio of the tokenizer.
"""
import os
import json
import glob
import argparse
from nanochat.tokenizer import get_tokenizer, RustBPETokenizer

# -----------------------------------------------------------------------------
# CLI arguments

parser = argparse.ArgumentParser(description='Evaluate tokenizer compression ratio')
parser.add_argument('--data-dir',   type=str, default="data/medical_raw/pubmed", help='Directory containing JSONL files')
parser.add_argument('--text-field', type=str, default="text",                    help='JSON field name for text')
parser.add_argument('--num-docs',   type=int, default=500,                       help='Number of docs to evaluate on (default: 500)')
args = parser.parse_args()

# -----------------------------------------------------------------------------
# Load sample docs from your JSONL files instead of parquet

def load_sample_docs(data_dir, text_field, num_docs):
    docs = []
    files = sorted(glob.glob(os.path.join(data_dir, "*.jsonl")))
    assert files, f"No .jsonl files found in {data_dir}"
    for fpath in files:
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                try:
                    text = json.loads(line).get(text_field, "")
                    if len(text) >= 30:
                        docs.append(text)
                        if len(docs) >= num_docs:
                            return docs
                except Exception:
                    continue
    return docs

print(f"Loading {args.num_docs} sample docs from {args.data_dir} ...")
sample_docs = load_sample_docs(args.data_dir, args.text_field, args.num_docs)
train_text = "\n".join(sample_docs)
print(f"Loaded {len(sample_docs)} docs, {len(train_text):,} chars total")

# -----------------------------------------------------------------------------
# Fixed text samples for evaluation

news_text = r"""
(Washington, D.C., July 9, 2025)- Yesterday, Mexico's National Service of Agro-Alimentary Health, Safety, and Quality (SENASICA) reported a new case of New World Screwworm (NWS) in Ixhuatlan de Madero, Veracruz in Mexico, which is approximately 160 miles northward of the current sterile fly dispersal grid, on the eastern side of the country and 370 miles south of the U.S./Mexico border. This new northward detection comes approximately two months after northern detections were reported in Oaxaca and Veracruz, less than 700 miles away from the U.S. border, which triggered the closure of our ports to Mexican cattle, bison, and horses on May 11, 2025.
""".strip()

code_text = r"""
class BasicTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            stats = get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = merge(ids, pair, idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
""".strip()

science_text = r"""
Photosynthesis is a photochemical energy transduction process in which light-harvesting pigment-protein complexes within the thylakoid membranes of oxygenic phototrophs absorb photons and initiate charge separation at the reaction center, driving the linear electron transport chain from water to NADP+ via photosystem II, the cytochrome b6f complex, and photosystem I, concomitantly generating a trans-thylakoid proton motive force utilized by chloroplastic ATP synthase. The light-dependent reactions produce ATP and NADPH, which fuel the Calvin-Benson-Bassham cycle in the stroma, wherein ribulose-1,5-bisphosphate is carboxylated by ribulose-1,5-bisphosphate carboxylase/oxygenase (RuBisCO) to form 3-phosphoglycerate.
""".strip()

medical_text = r"""
The patient presented with acute onset chest pain radiating to the left arm, diaphoresis, and dyspnea. ECG showed ST-elevation in leads II, III, and aVF consistent with inferior STEMI. Troponin I was markedly elevated at 8.4 ng/mL. The patient was immediately taken to the catheterization laboratory where coronary angiography revealed a 100% occlusion of the right coronary artery. Successful percutaneous coronary intervention was performed with placement of a drug-eluting stent. Post-procedure, the patient was started on dual antiplatelet therapy with aspirin and clopidogrel, along with a statin, beta-blocker, and ACE inhibitor. Echocardiography demonstrated an ejection fraction of 45% with inferior wall hypokinesis.
""".strip()

all_text = [
    ("medical",    medical_text),
    ("science",    science_text),
    ("news",       news_text),
    ("code",       code_text),
    ("pubmed-sample", train_text[:5000]),  # first 5000 chars of your actual PubMed data
]

# -----------------------------------------------------------------------------
# Evaluate tokenizers

tokenizer_results = {}
vocab_sizes = {}

for tokenizer_name in ["gpt2", "gpt4", "ours"]:
    if tokenizer_name == "gpt2":
        tokenizer = RustBPETokenizer.from_pretrained("gpt2")
    elif tokenizer_name == "gpt4":
        tokenizer = RustBPETokenizer.from_pretrained("cl100k_base")
    else:
        tokenizer = get_tokenizer()

    vocab_sizes[tokenizer_name] = tokenizer.get_vocab_size()
    tokenizer_results[tokenizer_name] = {}

    for name, text in all_text:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert decoded == text, f"Decode mismatch for {name} with {tokenizer_name}"
        encoded_bytes = text.encode('utf-8')
        ratio = len(encoded_bytes) / len(encoded)
        tokenizer_results[tokenizer_name][name] = {
            'bytes':  len(encoded_bytes),
            'tokens': len(encoded),
            'ratio':  ratio,
        }

# -----------------------------------------------------------------------------
# Print results

GREEN = '\033[92m'
RED   = '\033[91m'
RESET = '\033[0m'

print(f"\nVocab sizes:")
print(f"  GPT-2 : {vocab_sizes['gpt2']:,}")
print(f"  GPT-4 : {vocab_sizes['gpt4']:,}")
print(f"  Ours  : {vocab_sizes['ours']:,}")

def print_comparison(baseline_name, baseline_key, all_text):
    baseline_results = tokenizer_results[baseline_key]
    ours_results     = tokenizer_results['ours']
    print(f"\nComparison with {baseline_name}:")
    print("=" * 95)
    print(f"{'Text Type':<15} {'Bytes':<8} {baseline_name:<15} {'Ours':<15} {'Diff %':<12} {'Better'}")
    print(f"{'':15} {'':8} {'Tok':<7} {'Ratio':<7} {'Tok':<7} {'Ratio':<7}")
    print("-" * 95)
    for name, text in all_text:
        b = baseline_results[name]
        o = ours_results[name]
        diff = ((b['tokens'] - o['tokens']) / b['tokens']) * 100
        if b['ratio'] > o['ratio']:
            bc, oc, better, dc = GREEN, RED, baseline_name, RED
        elif o['ratio'] > b['ratio']:
            bc, oc, better, dc = RED, GREEN, "Ours ✓", GREEN
        else:
            bc, oc, better, dc = "", "", "Tie", ""
        print(f"{name:<15} {b['bytes']:<8} "
              f"{bc}{b['tokens']:<7}{RESET} {bc}{b['ratio']:<7.2f}{RESET} "
              f"{oc}{o['tokens']:<7}{RESET} {oc}{o['ratio']:<7.2f}{RESET} "
              f"{dc}{diff:+7.1f}%{RESET}  {better}")

print_comparison("GPT-2", "gpt2", all_text)
print_comparison("GPT-4", "gpt4", all_text)

# -----------------------------------------------------------------------------
# Overall compression ratio on your full PubMed sample
print(f"\n{'='*60}")
print(f"Overall compression on {len(sample_docs)} PubMed docs:")
full_text  = train_text
enc_ours   = tokenizer_results['ours']['pubmed-sample']
enc_gpt4   = tokenizer_results['gpt4']['pubmed-sample']
print(f"  GPT-4 ratio : {enc_gpt4['ratio']:.3f} chars/token")
print(f"  Ours ratio  : {enc_ours['ratio']:.3f} chars/token")
if enc_ours['ratio'] >= 4.5:
    print(f"  ✓ Tokenizer looks good! (>= 4.5 chars/token)")
else:
    print(f"  ⚠ Compression ratio below 4.5 — may need more training data")
print(f"{'='*60}")

# Log to report
from nanochat.report import get_report
get_report().log(section="Tokenizer evaluation", data=[
    {"vocab_size_ours": vocab_sizes['ours']},
    {"pubmed_compression_ratio": enc_ours['ratio']},
])