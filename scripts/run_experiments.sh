#!/usr/bin/env bash
# Run ASTRA vs RigL/SRigL comparison experiments.
#
# Execution order (RigL/SRigL first, ASTRA last):
#   Phase 1 — RigL & SRigL, CIFAR-10 then CIFAR-100
#               sparsities: 90%, 95%
#               sparsity types: unstructured, channel-wise (dynamic_ablation)
#   Phase 2 — ASTRA, CIFAR-10 then CIFAR-100
#               sparsities: 80%, 90%, 95%
#               sparsity types: unstructured, fanin, channel
#
# One experiment per GPU; 2 GPUs run in parallel within each phase.
#
# Usage:
#   export BASE_PATH=/path/to/condensed-sparsity-outputs
#   bash scripts/run_experiments.sh [--dry-run] [--seeds "42 123 456"]

set -euo pipefail

# ── Config ───────────────────────────────────────────────────────────────────
SEEDS="${SEEDS:-42 123 456}"
RIGL_SPARSITIES="0.9 0.95"
ASTRA_SPARSITIES="0.8 0.9 0.95"
DRY_RUN=false
PYTHON="/misc/envs/astra/bin/python"
ASTRA_SCRIPT="scripts/train_cifar.py"
RIGL_SCRIPT="condensed-sparsity/train_rigl.py"
export PYTHONPATH="condensed-sparsity/src${PYTHONPATH:+:$PYTHONPATH}"
LOG_DIR="logs"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export BASE_PATH="${BASE_PATH:-/tmp/condensed-sparsity-out}"

mkdir -p "$LOG_DIR"

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true ;;
        --seeds)   SEEDS="$2"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
    shift
done

# ── Helpers ───────────────────────────────────────────────────────────────────

cmd_to_logfile() {
    local cmd="$1"
    local slug
    if [[ "$cmd" =~ experiment\.name=([^[:space:]\"]+) ]]; then
        slug="${BASH_REMATCH[1]}"
    else
        local cfg sptype sp seed
        [[ "$cmd" =~ --config-name[[:space:]]+([^[:space:]]+) ]] && cfg="${BASH_REMATCH[1]}"
        [[ "$cmd" =~ sparsity_type=([^[:space:]]+) ]]            && sptype="${BASH_REMATCH[1]}"
        [[ "$cmd" =~ sparsity=([^[:space:]]+) ]]                 && sp="${BASH_REMATCH[1]}"
        [[ "$cmd" =~ training\.seed=([^[:space:]]+) ]]           && seed="${BASH_REMATCH[1]}"
        slug="${cfg:-job}_${sptype:-}_s${sp:-}_seed${seed:-}"
    fi
    echo "${LOG_DIR}/${slug}.log"
}

run_gpu_queue() {
    local gpu=$1; shift
    for cmd in "$@"; do
        local logfile
        logfile=$(cmd_to_logfile "$cmd")
        echo "[GPU $gpu] $(date '+%H:%M:%S') → $logfile"
        if $DRY_RUN; then
            echo "  $cmd"
        else
            CUDA_VISIBLE_DEVICES=$gpu bash -c "$cmd" \
                > "$logfile" 2>&1 \
                && echo "[GPU $gpu] DONE  $logfile" \
                || echo "[GPU $gpu] FAIL  $logfile (exit $?)"
        fi
    done
}

dispatch_two_queues() {
    local -n q0=$1
    local -n q1=$2
    run_gpu_queue 0 "${q0[@]}" &
    local pid0=$!
    run_gpu_queue 1 "${q1[@]}" &
    local pid1=$!
    wait $pid0 $pid1
}

split_jobs() {
    local -n _src=$1
    local -n _g0=$2
    local -n _g1=$3
    _g0=(); _g1=()
    for i in "${!_src[@]}"; do
        if (( i % 2 == 0 )); then _g0+=("${_src[$i]}")
        else                       _g1+=("${_src[$i]}")
        fi
    done
}

# ── Job builders ──────────────────────────────────────────────────────────────

# RigL or SRigL with a given sparsity structure.
#   sparsity_style: unstructured → const_fan_in=False, dynamic_ablation=False
#                   channel      → const_fan_in=False, dynamic_ablation=True
#                   srigl        → const_fan_in=True,  dynamic_ablation=False
#                   srigl_ch     → const_fan_in=True,  dynamic_ablation=True
rigl_jobs() {
    local dataset=$1
    local method=$2      # rigl_unstructured | rigl_channel | srigl | srigl_channel
    local training_cfg
    if [[ "$dataset" == "cifar100" ]]; then
        training_cfg="wide_resnet22_cifar100"
    else
        training_cfg="wide_resnet22"
    fi

    local const_fan ablation
    case "$method" in
        rigl_unstructured) const_fan=False; ablation=False ;;
        rigl_channel)      const_fan=False; ablation=True  ;;
        srigl)             const_fan=True;  ablation=False ;;
        srigl_channel)     const_fan=True;  ablation=True  ;;
    esac

    local -a jobs=()
    for s in $RIGL_SPARSITIES; do
        local dense
        dense=$("$PYTHON" -c "print(round(1 - $s, 2))")
        for seed in $SEEDS; do
            jobs+=("$PYTHON $RIGL_SCRIPT \
                dataset=$dataset \
                model=wide_resnet22 \
                training=$training_cfg \
                rigl.const_fan_in=$const_fan \
                rigl.dynamic_ablation=$ablation \
                rigl.dense_allocation=$dense \
                rigl.delta=100 \
                training.seed=$seed \
                compute.distributed=False \
                wandb.log_to_wandb=True \
                wandb.entity=null \
                wandb.project=astra-cifar \
                \"experiment.name=${method}_${dataset}_s${s}_seed${seed}\"")
        done
    done
    printf '%s\n' "${jobs[@]}"
}

astra_jobs() {
    local dataset=$1
    local -a jobs=()
    for type in unstructured fanin channel; do
        for s in $ASTRA_SPARSITIES; do
            for seed in $SEEDS; do
                jobs+=("$PYTHON $ASTRA_SCRIPT \
                    --config-name ${dataset}_astra \
                    sparsity=$s \
                    sparsity_type=$type \
                    training.seed=$seed \
                    wandb.group=astra_${type}_${dataset}_s${s}")
            done
        done
    done
    printf '%s\n' "${jobs[@]}"
}

# ── Phase 1: RigL & SRigL (both datasets) ────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo " Phase 1: RigL & SRigL"
echo "════════════════════════════════════════════════════════════"

for DATASET in cifar10 cifar100; do
    echo ""
    echo "  Dataset: $DATASET"

    mapfile -t ALL_JOBS < <(
        rigl_jobs "$DATASET" rigl_unstructured
        rigl_jobs "$DATASET" rigl_channel
        rigl_jobs "$DATASET" srigl
        rigl_jobs "$DATASET" srigl_channel
    )

    echo "  Total jobs: ${#ALL_JOBS[@]}"
    declare -a G0_JOBS=() G1_JOBS=()
    split_jobs ALL_JOBS G0_JOBS G1_JOBS
    echo "  GPU 0: ${#G0_JOBS[@]}  |  GPU 1: ${#G1_JOBS[@]}"
    dispatch_two_queues G0_JOBS G1_JOBS
    unset G0_JOBS G1_JOBS ALL_JOBS
    echo "  Done: $DATASET"
done

# ── Phase 2: ASTRA (both datasets) ───────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo " Phase 2: ASTRA"
echo "════════════════════════════════════════════════════════════"

for DATASET in cifar10 cifar100; do
    echo ""
    echo "  Dataset: $DATASET"

    mapfile -t ALL_JOBS < <(astra_jobs "$DATASET")

    echo "  Total jobs: ${#ALL_JOBS[@]}"
    declare -a G0_JOBS=() G1_JOBS=()
    split_jobs ALL_JOBS G0_JOBS G1_JOBS
    echo "  GPU 0: ${#G0_JOBS[@]}  |  GPU 1: ${#G1_JOBS[@]}"
    dispatch_two_queues G0_JOBS G1_JOBS
    unset G0_JOBS G1_JOBS ALL_JOBS
    echo "  Done: $DATASET"
done

echo ""
echo "All experiments complete."
