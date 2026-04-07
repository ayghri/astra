#!/usr/bin/env bash
# Train all methods for a given dataset and sparsity.
#
# Usage:
#   bash scripts/run_all.sh cifar10 0.9
#   bash scripts/run_all.sh cifar100 0.95
#   bash scripts/run_all.sh cifar10 0.8 --seed 123
#   bash scripts/run_all.sh cifar10 0.9 --device cuda:1
#   bash scripts/run_all.sh cifar10 0.9 --dry-run

set -euo pipefail

PYTHON="/misc/envs/astra/bin/python"
DATASET="${1:?Usage: $0 <dataset> <sparsity> [--seed N] [--device DEV] [--dry-run]}"
SPARSITY="${2:?Usage: $0 <dataset> <sparsity>}"
shift 2

SEED=42
DEVICE="cuda:0"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed)    SEED="$2"; shift ;;
        --device)  DEVICE="$2"; shift ;;
        --dry-run) DRY_RUN=true ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

DENSE_ALLOC=$(python3 -c "print(round(1 - $SPARSITY, 4))")

# Dataset-specific config bases
case "$DATASET" in
    cifar10)
        UNBIASED_CFG="cifar10_unbiased"
        SRIGL_CFG="cifar10_srigl"
        GMP_CFG="cifar10_gmp"
        IMP_CFG="cifar10_imp"
        ;;
    cifar100)
        UNBIASED_CFG="cifar10_unbiased"
        SRIGL_CFG="cifar10_srigl"
        GMP_CFG="cifar100_gmp"
        IMP_CFG="cifar100_imp"
        UNBIASED_EXTRA="dataset.name=cifar100 dataset.num_classes=100"
        SRIGL_EXTRA="dataset.name=cifar100 dataset.num_classes=100"
        ;;
    *)
        echo "Unknown dataset: $DATASET (supported: cifar10, cifar100)"
        exit 1
        ;;
esac

UNBIASED_EXTRA="${UNBIASED_EXTRA:-}"
SRIGL_EXTRA="${SRIGL_EXTRA:-}"

run() {
    local label="$1"; shift
    local cmd="$*"
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo " $label"
    echo " $cmd"
    echo "════════════════════════════════════════════════════════════"
    if $DRY_RUN; then
        echo "  [dry-run] skipped"
    else
        eval "$cmd"
    fi
}

# ── ASTRA (unbiased) ────────────────────────────────────────────────────────

for dist in uniform erk; do
    for stype in unstructured fanin; do
        run "ASTRA | $stype | $dist | s=$SPARSITY" \
            "$PYTHON scripts/train_unbiased_cifar.py --config-name $UNBIASED_CFG \
                sparsity=$SPARSITY sparsity_type=$stype sparsity_dist=$dist \
                training.seed=$SEED $UNBIASED_EXTRA"
    done
done

# ── SRigL ───────────────────────────────────────────────────────────────────

run "SRigL | erk | s=$SPARSITY" \
    "$PYTHON scripts/train_srigl_cifar.py --config-name $SRIGL_CFG \
        sparsity=$SPARSITY dense_alloc=$DENSE_ALLOC device=$DEVICE \
        training.seed=$SEED $SRIGL_EXTRA"

# ── RigL (no fan-in, no ablation) ──────────────────────────────────────────

run "RigL | erk | s=$SPARSITY" \
    "$PYTHON scripts/train_srigl_cifar.py --config-name $SRIGL_CFG \
        sparsity=$SPARSITY dense_alloc=$DENSE_ALLOC device=$DEVICE \
        rigl.const_fan_in=false rigl.dynamic_ablation=false \
        training.seed=$SEED $SRIGL_EXTRA"

# ── GMP ─────────────────────────────────────────────────────────────────────

for dist in uniform erk; do
    for stype in unstructured fanin; do
        run "GMP | $stype | $dist | s=$SPARSITY" \
            "$PYTHON scripts/train_magnitude_cifar.py --config-name $GMP_CFG \
                sparsity=$SPARSITY sparsity_type=$stype sparsity_dist=$dist \
                training.seed=$SEED"
    done
done

# ── IMP ─────────────────────────────────────────────────────────────────────

for dist in uniform erk; do
    for stype in unstructured fanin; do
        run "IMP | $stype | $dist | s=$SPARSITY" \
            "$PYTHON scripts/train_magnitude_cifar.py --config-name $IMP_CFG \
                sparsity=$SPARSITY sparsity_type=$stype sparsity_dist=$dist \
                training.seed=$SEED"
    done
done

echo ""
echo "All done for $DATASET s=$SPARSITY seed=$SEED"
