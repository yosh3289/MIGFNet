#!/bin/bash
# Train + Evaluate all 4 adaptive native models sequentially
# Each model uses: adaptive gating + deep supervision + modality dropout

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

MODELS="adaptive_nnunet adaptive_unet adaptive_conv3d adaptive_swinunetr"
CONFIG="configs/adaptive_mpnet.yaml"
NUM_GPUS=4

export NCCL_P2P_DISABLE=1

echo "=========================================="
echo "Adaptive Native Models: Train + Evaluate"
echo "=========================================="
echo "Models: $MODELS"
echo "Modules: Adaptive Gating + Deep Supervision + Modality Dropout"
echo ""

for MODEL in $MODELS; do
    echo ""
    echo "=========================================="
    echo "Training: $MODEL"
    echo "=========================================="

    # Kill stale GPU processes
    pkill -9 -f "train_adaptive_native.py" 2>/dev/null || true
    sleep 2

    torchrun --nproc_per_node=$NUM_GPUS train_adaptive_native.py \
        --model "$MODEL" \
        --config "$CONFIG" \
        2>&1 | tee "outputs/logs/${MODEL}_train.log"

    CKPT="outputs/checkpoints/${MODEL}/best_model.pth"
    if [ ! -f "$CKPT" ]; then
        echo "WARNING: No checkpoint found for $MODEL, skipping evaluation"
        continue
    fi

    echo ""
    echo "=========================================="
    echo "Evaluating: $MODEL"
    echo "=========================================="

    # For adaptive models, use the MAR evaluate.py with appropriate model_name
    # We need a small eval wrapper since these models use different build functions
    python eval_adaptive_native.py \
        --model "$MODEL" \
        --checkpoint "$CKPT" \
        --config "$CONFIG" \
        --output "outputs/results/eval_${MODEL}.json" \
        2>&1 | tee "outputs/logs/${MODEL}_eval.log"

    echo "$MODEL done."
done

echo ""
echo "=========================================="
echo "All adaptive native models done!"
echo "=========================================="
