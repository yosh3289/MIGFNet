#!/bin/bash
# AMPNet-UNet and AMPNet-Mamba without Deep Supervision: 2 models × 5 seeds = 10 runs
#
# Based on ablation finding: deep supervision has negative marginal effect (-0.022)
# on AMPNet-nnUNet. Testing if the same holds for UNet and Mamba.
#
# Usage: bash run_ablation_no_deepsup.sh [SEED1 SEED2 ...]
# Default seeds: 42 123 456 789 1024

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

NUM_GPUS=4
PREFIX="ablation_no_deepsup"

if [ $# -gt 0 ]; then
    SEEDS="$@"
else
    SEEDS="42 123 456 789 1024"
fi

export NCCL_P2P_DISABLE=1

echo "=========================================="
echo "AMPNet No-DeepSup Ablation"
echo "=========================================="
echo "Models: adaptive_unet, AdaptiveMPNet"
echo "Seeds: $SEEDS"
echo "GPUs: $NUM_GPUS"
echo ""

# --- adaptive_unet (faster, ~7s/ep) ---
UNET_CONFIG="configs/ablation_unet_no_deepsup.yaml"

for SEED in $SEEDS; do
    EVAL_JSON="outputs/results/${PREFIX}/unet/seed${SEED}/adaptive_unet_eval.json"
    LOG_DIR="outputs/logs/${PREFIX}/unet/seed${SEED}"
    mkdir -p "$(dirname "$EVAL_JSON")" "$LOG_DIR"

    if [ -f "$EVAL_JSON" ]; then
        echo "[SKIP] adaptive_unet seed=$SEED — eval already done"
        continue
    fi

    echo ""
    echo "=========================================="
    echo "Training: adaptive_unet w/o DeepSup (seed=$SEED)"
    echo "=========================================="
    pkill -9 -f "train_adaptive_native.py" 2>/dev/null || true
    sleep 2

    torchrun --nproc_per_node=$NUM_GPUS train_adaptive_native.py \
        --config "$UNET_CONFIG" \
        --model adaptive_unet \
        --seed "$SEED" \
        --output_prefix "${PREFIX}/unet" \
        2>&1 | tee "${LOG_DIR}/adaptive_unet_train.log"

    CKPT="outputs/checkpoints/${PREFIX}/unet/seed${SEED}/adaptive_unet/best_model.pth"
    if [ -f "$CKPT" ]; then
        echo "Evaluating: adaptive_unet w/o DeepSup (seed=$SEED)"
        python evaluate.py \
            --config "$UNET_CONFIG" \
            --model_name adaptive_unet \
            --checkpoint "$CKPT" \
            --output "$EVAL_JSON" \
            2>&1 | tee "${LOG_DIR}/adaptive_unet_eval.log"
    else
        echo "WARNING: No checkpoint for adaptive_unet seed=$SEED"
    fi
done

echo ""
echo "adaptive_unet w/o DeepSup complete (all seeds)."
echo ""

# --- AdaptiveMPNet / Mamba (slower, ~55s/ep) ---
MAMBA_CONFIG="configs/ablation_mamba_no_deepsup.yaml"

for SEED in $SEEDS; do
    EVAL_JSON="outputs/results/${PREFIX}/mamba/seed${SEED}/AdaptiveMPNet_eval.json"
    LOG_DIR="outputs/logs/${PREFIX}/mamba/seed${SEED}"
    mkdir -p "$(dirname "$EVAL_JSON")" "$LOG_DIR"

    if [ -f "$EVAL_JSON" ]; then
        echo "[SKIP] AdaptiveMPNet seed=$SEED — eval already done"
        continue
    fi

    echo ""
    echo "=========================================="
    echo "Training: AdaptiveMPNet w/o DeepSup (seed=$SEED)"
    echo "=========================================="
    pkill -9 -f "train.py" 2>/dev/null || true
    sleep 2

    torchrun --nproc_per_node=$NUM_GPUS train.py \
        --config "$MAMBA_CONFIG" \
        --name AdaptiveMPNet \
        --seed "$SEED" \
        --output_prefix "${PREFIX}/mamba" \
        2>&1 | tee "${LOG_DIR}/AdaptiveMPNet_train.log"

    BEST_CKPT="outputs/checkpoints/${PREFIX}/mamba/seed${SEED}/AdaptiveMPNet/best_model.pth"
    if [ -f "$BEST_CKPT" ]; then
        echo "Evaluating: AdaptiveMPNet w/o DeepSup (seed=$SEED)"
        python evaluate.py \
            --config "$MAMBA_CONFIG" \
            --model_name AdaptiveMPNet \
            --checkpoint "$BEST_CKPT" \
            --output "$EVAL_JSON" \
            2>&1 | tee "${LOG_DIR}/AdaptiveMPNet_eval.log"
    else
        echo "WARNING: No checkpoint for AdaptiveMPNet seed=$SEED"
    fi
done

echo ""
echo "=========================================="
echo "All no-deepsup ablation runs complete!"
echo "=========================================="
