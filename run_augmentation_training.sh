#!/bin/bash
# ============================================================================
# Train every UNet++ variant analysed in the paper.
#   6 augmentation modes  ×  2 modalities (BTFE, ssh_TSE)  =  12 trained models
# Optional: 2 COMBINED-cohort variants kept for backwards-compatibility with
# the upstream MIAGroupUT codebase (these models are not analysed in the paper).
#
# All runs use a single fold (--cv_folds 1); the train/val/test split is read
# from <dataset_root>/splits.json.
#
# Run from the repository root after the dataset is in place:
#   bash run_augmentation_training.sh
# ============================================================================

set -e

# --- paths (relative to the repo root) -------------------------------------
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
cd "$PROJECT_ROOT"

BTFE_ROOT="$PROJECT_ROOT/dataset/resized/DATASET_BTFE/"
TSE_ROOT="$PROJECT_ROOT/dataset/resized/DATASET_SSH_TSE/"
COMBINED_ROOT="$PROJECT_ROOT/dataset/resized/DATASET_COMBINED/"     # optional

# --- training hyper-parameters --------------------------------------------
NETWORK="unetplusplus"
EPOCHS=150
BATCH=8
LR=0.001
WEIGHT_DECAY=1e-5
PATIENCE=25
WORKERS=8
SEED=42

# --- helper ----------------------------------------------------------------
train() {
    local out_name=$1
    local dataset_root=$2
    shift 2
    echo "==========================================================="
    echo "training: $out_name"
    echo "==========================================================="
    python train_placenta_2d_monai_v8.py \
        --dataset_root  "$dataset_root" \
        --out_dir       "./runs/${out_name}" \
        --mode          train \
        --network       "$NETWORK" \
        --cv_folds      1 \
        --epochs        "$EPOCHS" \
        --batch_size    "$BATCH" \
        --lr            "$LR" \
        --weight_decay  "$WEIGHT_DECAY" \
        --scheduler     plateau \
        --early_stopping --patience "$PATIENCE" \
        --num_workers   "$WORKERS" \
        --amp --compile \
        --use_augmentation \
        --seed          "$SEED" \
        "$@"
}

# --- 1. Baseline (standard augmentation pipeline only) ---------------------
train "BTFE_${NETWORK}_1Fold"       "$BTFE_ROOT"
train "TSE_${NETWORK}_1Fold"        "$TSE_ROOT"

# --- 2. MixUp --------------------------------------------------------------
train "BTFE_${NETWORK}_mixup"       "$BTFE_ROOT" --use_mixup  --mixup_alpha  0.2
train "TSE_${NETWORK}_mixup"        "$TSE_ROOT"  --use_mixup  --mixup_alpha  0.2

# --- 3. CutMix -------------------------------------------------------------
train "BTFE_${NETWORK}_cutmix"      "$BTFE_ROOT" --use_cutmix --cutmix_alpha 1.0
train "TSE_${NETWORK}_cutmix"       "$TSE_ROOT"  --use_cutmix --cutmix_alpha 1.0

# --- 4. AFA ----------------------------------------------------------------
train "BTFE_${NETWORK}_afa"         "$BTFE_ROOT" --use_afa
train "TSE_${NETWORK}_afa"          "$TSE_ROOT"  --use_afa

# --- 5. MixUp + AFA --------------------------------------------------------
train "BTFE_${NETWORK}_mixup_afa"   "$BTFE_ROOT" --use_mixup  --mixup_alpha  0.2 --use_afa
train "TSE_${NETWORK}_mixup_afa"    "$TSE_ROOT"  --use_mixup  --mixup_alpha  0.2 --use_afa

# --- 6. CutMix + AFA -------------------------------------------------------
train "BTFE_${NETWORK}_cutmix_afa"  "$BTFE_ROOT" --use_cutmix --cutmix_alpha 1.0 --use_afa
train "TSE_${NETWORK}_cutmix_afa"   "$TSE_ROOT"  --use_cutmix --cutmix_alpha 1.0 --use_afa

# --- (optional) COMBINED cohort variants -----------------------------------
# Uncomment if you also want to train on the combined BTFE+ssh_TSE pool.
# These models are NOT analysed in the paper.
#
# train "COMBINED_${NETWORK}_mixup_afa"   "$COMBINED_ROOT" --use_mixup  --mixup_alpha  0.2 --use_afa
# train "COMBINED_${NETWORK}_cutmix_afa"  "$COMBINED_ROOT" --use_cutmix --cutmix_alpha 1.0 --use_afa

echo ""
echo "All training runs complete. Trained models are saved under ./runs/"
