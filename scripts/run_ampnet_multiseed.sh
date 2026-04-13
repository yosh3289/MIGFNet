#!/bin/bash
# MAR: Multi-seed training + evaluation for 3 AMPNet variants
#
# Models (ordered by speed, fastest first):
#   adaptive_nnunet  → train_adaptive_native.py
#   adaptive_unet    → train_adaptive_native.py
#   AdaptiveMPNet    → train.py (Mamba backbone)
#
# Features:
#   - Skips completed runs (checks for eval JSON)
#   - Resumes interrupted Mamba training from checkpoint
#
# Usage: bash run_ampnet_multiseed.sh [SEED1 SEED2 ...]
# Default seeds: 42 123 456 789 1024

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="configs/adaptive_mpnet.yaml"
NUM_GPUS=4

# Seeds from args or default
if [ $# -gt 0 ]; then
    SEEDS="$@"
else
    SEEDS="42 123 456 789 1024"
fi

export NCCL_P2P_DISABLE=1

echo "=========================================="
echo "MAR: AMPNet Multi-Seed Benchmark"
echo "=========================================="
echo "Models: adaptive_nnunet adaptive_unet AdaptiveMPNet"
echo "Seeds: $SEEDS"
echo "Config: $CONFIG"
echo "GPUs: $NUM_GPUS"
echo ""

for SEED in $SEEDS; do
    echo ""
    echo "############################################"
    echo "# SEED = $SEED"
    echo "############################################"

    SEED_DIR="seed${SEED}"
    LOG_DIR="outputs/logs/${SEED_DIR}"
    mkdir -p "outputs/checkpoints/${SEED_DIR}" "outputs/results/${SEED_DIR}" "$LOG_DIR"

    # --- adaptive_nnunet (fastest, ~10s/ep) ---
    EVAL_JSON="outputs/results/${SEED_DIR}/adaptive_nnunet_eval.json"
    if [ -f "$EVAL_JSON" ]; then
        echo "[SKIP] adaptive_nnunet seed=$SEED — eval already done"
    else
        echo ""
        echo "=========================================="
        echo "Training: adaptive_nnunet (seed=$SEED)"
        echo "=========================================="
        pkill -9 -f "train_adaptive_native.py" 2>/dev/null || true
        sleep 2

        torchrun --nproc_per_node=$NUM_GPUS train_adaptive_native.py \
            --config "$CONFIG" \
            --model adaptive_nnunet \
            --seed "$SEED" \
            2>&1 | tee "${LOG_DIR}/adaptive_nnunet_train.log"

        CKPT="outputs/checkpoints/${SEED_DIR}/adaptive_nnunet/best_model.pth"
        if [ -f "$CKPT" ]; then
            echo "Evaluating: adaptive_nnunet (seed=$SEED)"
            python evaluate.py \
                --config "$CONFIG" \
                --model_name adaptive_nnunet \
                --checkpoint "$CKPT" \
                --output "$EVAL_JSON" \
                2>&1 | tee "${LOG_DIR}/adaptive_nnunet_eval.log"
        else
            echo "WARNING: No checkpoint for adaptive_nnunet seed=$SEED"
        fi
    fi

    # --- adaptive_unet (~15s/ep) ---
    EVAL_JSON="outputs/results/${SEED_DIR}/adaptive_unet_eval.json"
    if [ -f "$EVAL_JSON" ]; then
        echo "[SKIP] adaptive_unet seed=$SEED — eval already done"
    else
        echo ""
        echo "=========================================="
        echo "Training: adaptive_unet (seed=$SEED)"
        echo "=========================================="
        pkill -9 -f "train_adaptive_native.py" 2>/dev/null || true
        sleep 2

        torchrun --nproc_per_node=$NUM_GPUS train_adaptive_native.py \
            --config "$CONFIG" \
            --model adaptive_unet \
            --seed "$SEED" \
            2>&1 | tee "${LOG_DIR}/adaptive_unet_train.log"

        CKPT="outputs/checkpoints/${SEED_DIR}/adaptive_unet/best_model.pth"
        if [ -f "$CKPT" ]; then
            echo "Evaluating: adaptive_unet (seed=$SEED)"
            python evaluate.py \
                --config "$CONFIG" \
                --model_name adaptive_unet \
                --checkpoint "$CKPT" \
                --output "$EVAL_JSON" \
                2>&1 | tee "${LOG_DIR}/adaptive_unet_eval.log"
        else
            echo "WARNING: No checkpoint for adaptive_unet seed=$SEED"
        fi
    fi

    # --- AdaptiveMPNet / Mamba (slowest, ~75s/ep) ---
    EVAL_JSON="outputs/results/${SEED_DIR}/AdaptiveMPNet_eval.json"
    if [ -f "$EVAL_JSON" ]; then
        echo "[SKIP] AdaptiveMPNet seed=$SEED — eval already done"
    else
        echo ""
        echo "=========================================="
        echo "Training: AdaptiveMPNet (seed=$SEED)"
        echo "=========================================="
        pkill -9 -f "train.py" 2>/dev/null || true
        sleep 2

        # Check for resumable checkpoint (final_model means completed, skip to eval)
        FINAL_CKPT="outputs/checkpoints/${SEED_DIR}/AdaptiveMPNet/final_model.pth"
        BEST_CKPT="outputs/checkpoints/${SEED_DIR}/AdaptiveMPNet/best_model.pth"

        if [ -f "$FINAL_CKPT" ]; then
            echo "Training already completed (final_model.pth exists), skipping to eval"
        elif [ -f "$BEST_CKPT" ]; then
            echo "Resuming from best_model.pth"
            torchrun --nproc_per_node=$NUM_GPUS train.py \
                --config "$CONFIG" \
                --name AdaptiveMPNet \
                --seed "$SEED" \
                --resume "$BEST_CKPT" \
                2>&1 | tee "${LOG_DIR}/AdaptiveMPNet_train.log"
        else
            torchrun --nproc_per_node=$NUM_GPUS train.py \
                --config "$CONFIG" \
                --name AdaptiveMPNet \
                --seed "$SEED" \
                2>&1 | tee "${LOG_DIR}/AdaptiveMPNet_train.log"
        fi

        if [ -f "$BEST_CKPT" ]; then
            echo "Evaluating: AdaptiveMPNet (seed=$SEED)"
            python evaluate.py \
                --config "$CONFIG" \
                --model_name AdaptiveMPNet \
                --checkpoint "$BEST_CKPT" \
                --output "$EVAL_JSON" \
                2>&1 | tee "${LOG_DIR}/AdaptiveMPNet_eval.log"
        else
            echo "WARNING: No checkpoint for AdaptiveMPNet seed=$SEED"
        fi
    fi

    echo ""
    echo "Seed $SEED complete."
done

echo ""
echo "=========================================="
echo "All seeds complete!"
echo "=========================================="
