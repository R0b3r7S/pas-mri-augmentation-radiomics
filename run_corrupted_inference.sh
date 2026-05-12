#!/bin/bash
# ============================================================================
# CORRUPTED-INFERENCE GRID (Figure 1, severity 3)
# ----------------------------------------------------------------------------
# Runs every trained model against every corruption × mode combination
# produced by corrupt_test_set.py. Designed to match the existing
# run_augmentation_inference.sh conventions: same python entrypoint,
# same batch/worker settings, same output layout so the radiomics and
# metric dashboards pick up the results automatically.
#
# Output directory pattern:
#   runs/<MODEL>/test_on_<BASE_DOMAIN>_<CORR>_s3_<MODE>/fold_0/...
#
# The corrupted datasets must exist before running this:
#   conda activate monai_placenta
#   python corrupt_test_set.py         # creates dataset/resized/DATASET_*_s3_2d/ and _3d/
#
# Customise the arrays below to skip combinations; everything is idempotent
# (rerunning just re-writes the same out_dir).
# ============================================================================

set -e

# --- paths / knobs ---------------------------------------------------------
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
cd "$PROJECT_ROOT"
DATASET_BASE="$PROJECT_ROOT/dataset/resized"
NETWORK="unetplusplus"
WORKERS=8
BATCH=1
SEED=42
LOG_DIR="$PROJECT_ROOT/log"
mkdir -p "$LOG_DIR"

# --- what to run ------------------------------------------------------------
# Suffixes of the trained model directories (runs/<train>_<NET>_<SUFFIX>).
# Models marked with *_afa or *_mixup_afa / *_cutmix_afa need --use_afa at test.
BTFE_SUFFIXES=(1Fold mixup cutmix afa mixup_afa cutmix_afa)
TSE_SUFFIXES=(1Fold mixup cutmix afa mixup_afa cutmix_afa)

CORRUPTIONS=(bias_field ghosting kspace_sub rician_noise spike_noise)
CORR_MODES=(2d 3d)

# Every BTFE-trained model is evaluated on BTFE-corrupted (same-domain) and
# SSH_TSE-corrupted (cross-domain). Symmetric for TSE-trained models.
# Set e.g. CROSS_DOMAIN=0 to skip the cross-domain half during a dry-run.
CROSS_DOMAIN=${CROSS_DOMAIN:-1}

# --- helpers ---------------------------------------------------------------
needs_afa() {
    case "$1" in
        *afa*) echo "--use_afa" ;;
        *)     echo "" ;;
    esac
}

run_one() {
    # $1 = train_domain (BTFE | TSE)
    # $2 = model suffix  (e.g. mixup_afa)
    # $3 = test base domain (BTFE | TSE)
    # $4 = corruption name
    # $5 = corruption mode (2d | 3d)
    local train=$1 suffix=$2 test_domain=$3 corr=$4 corr_mode=$5

    local model="${train}_${NETWORK}_${suffix}"
    local checkpoint="./runs/${model}/fold_0/best_model.pth"
    if [ ! -f "$checkpoint" ]; then
        echo "  [skip] missing checkpoint: $checkpoint"
        return 0
    fi

    local ds_dirname
    case "$test_domain" in
        BTFE) ds_dirname="DATASET_BTFE_${corr}_s3_${corr_mode}" ;;
        TSE)  ds_dirname="DATASET_SSH_TSE_${corr}_s3_${corr_mode}" ;;
        *)    echo "  [skip] unknown test domain $test_domain"; return 0 ;;
    esac
    local dataset_root="$DATASET_BASE/$ds_dirname"
    if [ ! -d "$dataset_root" ]; then
        echo "  [skip] corrupted dataset missing: $dataset_root"
        return 0
    fi

    local out_dir="./runs/${model}/test_on_${test_domain}_${corr}_s3_${corr_mode}"
    local afa_flag
    afa_flag=$(needs_afa "$suffix")

    echo "------------------------------------------------------------------"
    echo "🧪 $model  →  $test_domain  |  $corr s3 $corr_mode"
    echo "------------------------------------------------------------------"

    python train_placenta_2d_monai_v8.py \
        --dataset_root "$dataset_root" \
        --out_dir "$out_dir" \
        --checkpoint "$checkpoint" \
        --mode test \
        --network "$NETWORK" \
        --cv_folds 1 \
        --batch_size "$BATCH" \
        --num_workers "$WORKERS" \
        --amp \
        --seed "$SEED" \
        $afa_flag
}

# --- main loops ------------------------------------------------------------
LOG_FILE="$LOG_DIR/corrupted_inference_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -i "$LOG_FILE") 2>&1

echo "=================================================================="
echo "🧪 CORRUPTED-INFERENCE GRID — severity 3"
echo "   corruptions : ${CORRUPTIONS[*]}"
echo "   modes       : ${CORR_MODES[*]}"
echo "   BTFE models : ${BTFE_SUFFIXES[*]}"
echo "   TSE  models : ${TSE_SUFFIXES[*]}"
echo "   cross-domain: $CROSS_DOMAIN"
echo "   log         : $LOG_FILE"
echo "=================================================================="

for corr in "${CORRUPTIONS[@]}"; do
    for mode in "${CORR_MODES[@]}"; do
        # --- BTFE-trained models ---
        for suf in "${BTFE_SUFFIXES[@]}"; do
            run_one "BTFE" "$suf" "BTFE" "$corr" "$mode"                # same-domain
            if [ "$CROSS_DOMAIN" = "1" ]; then
                run_one "BTFE" "$suf" "TSE" "$corr" "$mode"             # cross-domain
            fi
        done
        # --- TSE-trained models ---
        for suf in "${TSE_SUFFIXES[@]}"; do
            run_one "TSE" "$suf" "TSE" "$corr" "$mode"                  # same-domain
            if [ "$CROSS_DOMAIN" = "1" ]; then
                run_one "TSE" "$suf" "BTFE" "$corr" "$mode"             # cross-domain
            fi
        done
    done
done

echo ""
echo "✅ corrupted-inference grid complete"
