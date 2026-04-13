#!/bin/bash
# =============================================================================
# Pipeline Rebuild: Official Evaluation Protocol + Human Expert Labels
# =============================================================================
# 6 models total:
#   1. nnU-Net v2 (official, 4-GPU DDP)
#   2. U-Net (official picai_baseline, 1-GPU)
#   3. SwinUNETR (ours, 4-GPU DDP)
#   4. UMamba-concat (ours, 4-GPU DDP)
#   5. AdaptiveMPNet (ours/proposed, 4-GPU DDP)
#   6. nnDetection (official, 1-GPU) — optional, long training
#
# Labels: Human expert (merged original + Pooch25)
# Split: Official PI-CAI fold-0
# Metrics: picai_eval (AUROC, AP, Ranking Score) + Dice/Sens/Spec
# =============================================================================

set -e
source /workspace/.venv/bin/activate
cd /workspace/P1-MAR

CONFIG="configs/adaptive_mpnet.yaml"
NUM_GPUS=4

# nnU-Net environment
export nnUNet_raw="/workspace/P1-MAR/nnunet_data/nnUNet_raw"
export nnUNet_preprocessed="/workspace/P1-MAR/nnunet_data/nnUNet_preprocessed"
export nnUNet_results="/workspace/P1-MAR/nnunet_data/nnUNet_results"

mkdir -p outputs/checkpoints outputs/logs outputs/results

echo "============================================================"
echo "Pipeline: Official Evaluation + Human Expert Labels"
echo "============================================================"
echo "GPUs: ${NUM_GPUS} | Split: PI-CAI fold-0 | Labels: human expert"
echo ""

# =============================================================================
# Phase 1: Official baselines (can overlap with cache rebuild)
# =============================================================================

# --- 1. nnU-Net v2 (official, 4-GPU DDP) ---
echo "[1/6] nnU-Net v2: Planning and preprocessing..."
nnUNetv2_plan_and_preprocess -d 100 --verify_dataset_integrity -c 3d_fullres 2>&1 | tail -5
echo "[1/6] nnU-Net v2: Training fold-0 (4 GPUs)..."
nnUNetv2_train 100 3d_fullres 0 -num_gpus ${NUM_GPUS} -tr nnUNetTrainerFocalDiceLoss
echo "nnU-Net v2 training complete."

# --- 2. Official U-Net (picai_baseline, 1-GPU) ---
echo "[2/6] Official U-Net: Training..."
python train_official_unet.py --fold 0
echo "Official U-Net training complete."

# =============================================================================
# Phase 2: Our models (requires rebuilt cache)
# =============================================================================

# --- 3. AdaptiveMPNet (our proposed method, 4-GPU DDP) ---
echo "[3/6] Training AdaptiveMPNet (proposed)..."
torchrun --nproc_per_node=${NUM_GPUS} train.py --config ${CONFIG}
echo "AdaptiveMPNet training complete."

# --- 4. SwinUNETR (4-GPU DDP) ---
echo "[4/6] Training SwinUNETR..."
torchrun --nproc_per_node=${NUM_GPUS} train_baseline.py \
    --config ${CONFIG} --model SwinUNETR
echo "SwinUNETR training complete."

# --- 5. UMamba-concat (4-GPU DDP) ---
echo "[5/6] Training UMamba-concat..."
torchrun --nproc_per_node=${NUM_GPUS} train_baseline.py \
    --config ${CONFIG} --model UMamba_concat
echo "UMamba-concat training complete."

# =============================================================================
# Phase 3: Evaluation (all models on fold-0 val set)
# =============================================================================
echo ""
echo "============================================================"
echo "Evaluation Phase (PI-CAI official metrics)"
echo "============================================================"

# Evaluate our models
for MODEL in AdaptiveMPNet SwinUNETR UMamba_concat; do
    CKPT="outputs/checkpoints/${MODEL}/best_model.pth"
    if [ -f "$CKPT" ]; then
        echo "Evaluating ${MODEL}..."
        python evaluate.py \
            --config ${CONFIG} \
            --checkpoint ${CKPT} \
            --model_name ${MODEL} \
            --output outputs/results/${MODEL}_results.json
    else
        echo "SKIP: ${MODEL} checkpoint not found at ${CKPT}"
    fi
done

# Evaluate nnU-Net v2 (needs special handling — inference + convert to our eval format)
echo "Evaluating nnU-Net v2..."
nnUNetv2_predict \
    -i ${nnUNet_raw}/Dataset100_PICAI/imagesTs \
    -o ${nnUNet_results}/predictions_fold0 \
    -d 100 -c 3d_fullres -f 0 \
    -tr nnUNetTrainerFocalDiceLoss \
    --disable_tta 2>&1 | tail -5
echo "nnU-Net v2 predictions saved."

# --- 6. nnDetection (optional, 1-GPU, longest training) ---
# echo "[6/6] Training nnDetection..."
# Uncomment when ready:
# python train_nndetection.py --fold 0

echo ""
echo "============================================================"
echo "All experiments complete. Results in outputs/results/"
echo "============================================================"
