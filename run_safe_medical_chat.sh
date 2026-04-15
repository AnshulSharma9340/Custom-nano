#!/bin/bash
# Launch Safe Medical Chat Server

export NANOCHAT_BASE_DIR="$HOME/Custom-nano/nanochat_cache"
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "🏥 Starting Safe Medical Chat Server..."
echo "🛡️ Medical Safety Layer: ENABLED"
echo "🌡️ Strict Mode: Temperature set to 0.3 (Anti-Hallucination)"
echo ""

cd ~/Custom-nano
python -m scripts.chat_web_safe \
    --model-tag=medical_2b_mid \
    --source=sft \
    --step=1358 \
    --temperature=0.3 \
    --port=8000
