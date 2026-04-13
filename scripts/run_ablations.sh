#!/bin/bash
# Run ablation experiments sequentially (A2-A4, A1 already done)
# Each uses 4 GPUs with DDP
# NO set -e: individual failures don't kill the pipeline

export NCCL_P2P_DISABLE=1
export NCCL_ASYNC_ERROR_HANDLING=1
TORCHRUN=/workspace/.venv/bin/torchrun

run_ablation() {
    local label=$1
    local config=$2
    local name=$3

    echo ""
    echo "=========================================="
    echo "[$label] START: $(date)"
    echo "=========================================="

    # Kill any leftover GPU processes
    sleep 5

    $TORCHRUN --nproc_per_node=4 --master_port=29500 train.py \
      --config "$config" --name "$name"
    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "[$label] DONE (success): $(date)"
    else
        echo "[$label] FAILED (exit=$exit_code): $(date)"
    fi

    # Cool-down between runs
    sleep 10
    return $exit_code
}

echo "=========================================="
echo "ABLATION PIPELINE START: $(date)"
echo "A1 already complete — running A2, A3, A4"
echo "=========================================="

# A2: w/o Mamba SSM
run_ablation "A2" "configs/ablation_no_mamba.yaml" "Ablation_NoMamba"

# A3: w/o Deep Supervision
run_ablation "A3" "configs/ablation_no_deepsup.yaml" "Ablation_NoDeepSup"

# A4: w/o Modality Dropout
run_ablation "A4" "configs/ablation_no_moddropout.yaml" "Ablation_NoModDropout"

echo ""
echo "=========================================="
echo "ABLATION PIPELINE FINISHED: $(date)"
echo "=========================================="
echo "Checkpoints:"
ls -la outputs/checkpoints/Ablation_*/best_model.pth 2>/dev/null || echo "  (none found)"
