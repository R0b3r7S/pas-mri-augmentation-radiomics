#!/bin/bash
# ============================================================================
# Same-domain + cross-domain inference for every trained UNet++ variant.
# For each (training modality, augmentation) model, run inference on the
# clean BTFE test set and the clean ssh_TSE test set. Output goes under
# ./runs/<MODEL>/test_on_<TEST_DOMAIN>/fold_0/.
#
# Run after run_augmentation_training.sh:
#   bash run_augmentation_inference.sh
# ============================================================================

set -e

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
cd "$PROJECT_ROOT"

BTFE_ROOT="$PROJECT_ROOT/dataset/resized/DATASET_BTFE/"
TSE_ROOT="$PROJECT_ROOT/dataset/resized/DATASET_SSH_TSE/"

NETWORK="unetplusplus"
WORKERS=8
SEED=42

# --- helper ----------------------------------------------------------------
# Args:  out_dir  dataset_root  checkpoint  [extra flags...]
infer() {
    local out_dir=$1
    local dataset_root=$2
    local checkpoint=$3
    shift 3
    echo "==========================================================="
    echo "inference: $out_dir"
    echo "==========================================================="
    python train_placenta_2d_monai_v8.py \
        --dataset_root  "$dataset_root" \
        --out_dir       "$out_dir" \
        --checkpoint    "$checkpoint" \
        --mode          test \
        --network       "$NETWORK" \
        --cv_folds      1 \
        --batch_size    1 \
        --num_workers   "$WORKERS" \
        --amp \
        --seed          "$SEED" \
        "$@"
}

# --- evaluate every trained model on both modalities -----------------------
# Models trained with AFA require --use_afa at test time so the dual-norm
# routing is active.

for SUFFIX in 1Fold mixup cutmix afa mixup_afa cutmix_afa; do
    EXTRA=""
    [[ "$SUFFIX" == *"afa"* ]] && EXTRA="--use_afa"

    for TRAIN in BTFE TSE; do
        ROOT_TRAIN_VAR="${TRAIN}_ROOT"
        TRAIN_ROOT="${!ROOT_TRAIN_VAR}"

        MODEL="${TRAIN}_${NETWORK}_${SUFFIX}"
        CKPT="./runs/${MODEL}/fold_0/best_model.pth"

        # Same-domain
        infer "./runs/${MODEL}/test_on_${TRAIN}" \
              "$TRAIN_ROOT" \
              "$CKPT" \
              $EXTRA

        # Cross-domain
        OTHER=$([[ "$TRAIN" == "BTFE" ]] && echo "TSE" || echo "BTFE")
        ROOT_OTHER_VAR="${OTHER}_ROOT"
        OTHER_ROOT="${!ROOT_OTHER_VAR}"

        infer "./runs/${MODEL}/test_on_${OTHER}" \
              "$OTHER_ROOT" \
              "$CKPT" \
              $EXTRA
    done
done

echo ""
echo "All inference runs complete. Per-cell outputs are under ./runs/<MODEL>/test_on_<DOMAIN>/"
