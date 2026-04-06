#!/bin/bash
# Medical Model Alignment with ORPO
# Safer than DPO - no reference model collapse issues
# Ultra-conservative settings for medical safety

set -e

export NANOCHAT_BASE_DIR="$HOME/Custom-nano/nanochat_cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8  # Optimize for 2 GPU setup

echo "🏥 MEDICAL MODEL ALIGNMENT WITH ORPO"
echo "======================================"
echo "Model: medical_2b_mid (SFT) → medical_2b_orpo (Aligned)"
echo "Data: Your existing DPO dataset (perfect format!)"
echo "Safety: Ultra-conservative hyperparameters"
echo ""

# Verify dataset exists
if [ ! -f "data/dpo_data/dpo_combined.jsonl" ]; then
    echo "❌ Error: data/dpo_data/dpo_combined.jsonl not found"
    echo "Please check your dataset path"
    exit 1
fi

# Check dataset size
DATASET_LINES=$(wc -l < data/dpo_data/dpo_combined.jsonl)
echo "📊 Dataset: $DATASET_LINES preference pairs"

if [ $DATASET_LINES -lt 100 ]; then
    echo "⚠️  Warning: Very small dataset ($DATASET_LINES pairs)"
    echo "Consider getting more training data for better alignment"
fi

echo ""
echo "🚀 Starting ORPO training..."
echo "Expected runtime: ~10-15 minutes on 2xA100"
echo ""

# ORPO Training - Medical Safety Configuration
torchrun --standalone --nproc_per_node=2 \
    -m scripts.orpo_train -- \
    --model-tag=medical_2b_mid \
    --model-step=1358 \
    --out-model-tag=medical_2b_orpo \
    --data-path=data/dpo_data/dpo_combined.jsonl \
    --num-iterations=100 \
    --device-batch-size=1 \
    --max-seq-len=512 \
    --lambda-orpo=0.01 \
    --lr-scale=0.00001 \
    --warmup-ratio=0.1 \
    --warmdown-ratio=0.3 \
    --eval-every=25 \
    --save-every=25 \
    --run=medical_orpo_alignment

ORPO_EXIT_CODE=$?

if [ $ORPO_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ ORPO Training Complete!"
    echo ""
    echo "📁 Model saved to: nanochat_cache/orpo_checkpoints/medical_2b_orpo/"
    echo ""
    echo "🧪 Test your aligned model:"
    echo "   python -m scripts.chat_web --model-tag=medical_2b_orpo --source=orpo"
    echo ""
    echo "🔍 Compare with original SFT model:"
    echo "   python -m scripts.chat_web --model-tag=medical_2b_mid --source=sft --step=1358"
    echo ""
    echo "💡 What to test:"
    echo "   - Ask about chest pain → Should recommend emergency care"
    echo "   - Ask for specific drug dosages → Should refuse and suggest doctor"
    echo "   - Ask about symptoms → Should express appropriate uncertainty"
    echo ""
    
    # Add orpo to checkpoint manager if not already there
    if ! grep -q '"orpo":' nanochat/checkpoint_manager.py; then
        echo "🔧 Adding ORPO to checkpoint manager..."
        sed -i 's/"dpo": "dpo_checkpoints",/"dpo": "dpo_checkpoints",\n        "orpo": "orpo_checkpoints",/' nanochat/checkpoint_manager.py
        echo "✅ ORPO source added to checkpoint manager"
    fi
    
else
    echo ""
    echo "❌ ORPO Training Failed (exit code: $ORPO_EXIT_CODE)"
    echo ""
    echo "🔍 Common issues:"
    echo "   - Check GPU memory (should work on 2xA100 40GB)"
    echo "   - Verify dataset format (prompt/chosen/rejected fields)"
    echo "   - Check model checkpoint exists (medical_2b_mid step 1358)"
    echo ""
    echo "💡 Try reducing batch size if OOM:"
    echo "   Add --device-batch-size=1 to the torchrun command"
    
    exit $ORPO_EXIT_CODE
fi
