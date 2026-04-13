#!/bin/bash
# AMPNet-nnUNet Module Ablation: 6 configs × 5 seeds = 30 training runs
#
# A1: w/o Gating    (G=N, D=Y, M=Y)
# A2: w/o DeepSup   (G=Y, D=N, M=Y)
# A3: w/o ModDrop   (G=Y, D=Y, M=N)
# A4: Gating Only   (G=Y, D=N, M=N)
# A5: DeepSup Only  (G=N, D=Y, M=N)
# A6: ModDrop Only  (G=N, D=N, M=Y)
#
# Usage: bash run_ablation_nnunet.sh [SEED1 SEED2 ...]
# Default seeds: 42 123 456 789 1024

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

NUM_GPUS=4

if [ $# -gt 0 ]; then
    SEEDS="$@"
else
    SEEDS="42 123 456 789 1024"
fi

export NCCL_P2P_DISABLE=1

ABLATIONS="A1_no_gating A2_no_deepsup A3_no_moddrop A4_gating_only A5_deepsup_only A6_moddrop_only"

echo "=========================================="
echo "AMPNet-nnUNet Module Ablation"
echo "=========================================="
echo "Configs: $ABLATIONS"
echo "Seeds: $SEEDS"
echo "GPUs: $NUM_GPUS"
echo ""

for ABL in $ABLATIONS; do
    CONFIG="configs/ablation_nnunet_${ABL}.yaml"
    # Short tag: A1, A2, ... A6
    TAG="${ABL%%_*}"

    echo ""
    echo "############################################"
    echo "# CONFIG = $ABL"
    echo "############################################"

    for SEED in $SEEDS; do
        PREFIX="ablation_nnunet/${TAG}"
        EVAL_JSON="outputs/results/${PREFIX}/seed${SEED}/adaptive_nnunet_eval.json"
        LOG_DIR="outputs/logs/${PREFIX}/seed${SEED}"
        mkdir -p "$(dirname "$EVAL_JSON")" "$LOG_DIR"

        if [ -f "$EVAL_JSON" ]; then
            echo "[SKIP] $TAG seed=$SEED — eval already done"
            continue
        fi

        echo ""
        echo "=========================================="
        echo "Training: $ABL (seed=$SEED)"
        echo "=========================================="
        pkill -9 -f "train_adaptive_native.py" 2>/dev/null || true
        sleep 2

        torchrun --nproc_per_node=$NUM_GPUS train_adaptive_native.py \
            --config "$CONFIG" \
            --model adaptive_nnunet \
            --seed "$SEED" \
            --output_prefix "${PREFIX}" \
            2>&1 | tee "${LOG_DIR}/adaptive_nnunet_train.log"

        CKPT="outputs/checkpoints/${PREFIX}/seed${SEED}/adaptive_nnunet/best_model.pth"
        if [ -f "$CKPT" ]; then
            echo "Evaluating: $ABL (seed=$SEED)"
            python evaluate.py \
                --config "$CONFIG" \
                --model_name adaptive_nnunet \
                --checkpoint "$CKPT" \
                --output "$EVAL_JSON" \
                2>&1 | tee "${LOG_DIR}/adaptive_nnunet_eval.log"
        else
            echo "WARNING: No checkpoint for $ABL seed=$SEED"
        fi
    done

    echo ""
    echo "$ABL complete (all seeds)."
done

echo ""
echo "=========================================="
echo "All ablation configs complete!"
echo "=========================================="
