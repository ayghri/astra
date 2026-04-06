#!/usr/bin/env bash
# Run 2:4 pruning benchmark for a given model and method.
#
# Usage:
#   bash autoresearch/run_bench.sh --model Qwen/Qwen3-1.7B --method sparsegpt
#   bash autoresearch/run_bench.sh --model Qwen/Qwen3-1.7B --method admm --admm-iter 2000
#   bash autoresearch/run_bench.sh --model Qwen/Qwen3-1.7B --method admm_corr --device cuda:1
#
#   # Run all three methods:
#   bash autoresearch/run_bench.sh --model Qwen/Qwen3-1.7B --method all

set -euo pipefail

PYTHON="/misc/envs/astra/bin/python"
SCRIPT="autoresearch/bench_prune_llm.py"
OUTDIR="autoresearch/results"

# Defaults
MODEL="Qwen/Qwen3-1.7B"
METHOD=""
DEVICE="cuda:0"
ADMM_ITER=1500
NUM_SAMPLES=256
SEQ_LEN=2048
# EVAL_TASKS="wikitext,arc_easy,arc_challenge,piqa,winogrande,boolq,lambada_openai"
EVAL_TASKS="wikitext"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)       MODEL="$2"; shift ;;
        --method)      METHOD="$2"; shift ;;
        --device)      DEVICE="$2"; shift ;;
        --admm-iter)   ADMM_ITER="$2"; shift ;;
        --num-samples) NUM_SAMPLES="$2"; shift ;;
        --seq-len)     SEQ_LEN="$2"; shift ;;
        --eval-tasks)  EVAL_TASKS="$2"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

if [[ -z "$METHOD" ]]; then
    echo "Usage: $0 --model MODEL --method {sparsegpt|admm|admm_corr|all} [options]"
    exit 1
fi

mkdir -p "$OUTDIR"
MODEL_TAG="${MODEL##*/}"

run_one() {
    local m="$1"
    local out="$OUTDIR/bench_${m}_${MODEL_TAG}.json"
    local log="$OUTDIR/bench_${m}_${MODEL_TAG}.log"

    echo "════════════════════════════════════════════════════════════"
    echo " Method: $m  Model: $MODEL  Device: $DEVICE"
    echo " Output: $out"
    echo " Log:    $log"
    echo "════════════════════════════════════════════════════════════"

    $PYTHON "$SCRIPT" \
        --model "$MODEL" \
        --method "$m" \
        --device "$DEVICE" \
        --admm-iter "$ADMM_ITER" \
        --num-samples "$NUM_SAMPLES" \
        --seq-len "$SEQ_LEN" \
        --eval-tasks "$EVAL_TASKS" \
        --output "$out" \
        2>&1 | tee "$log"
}

if [[ "$METHOD" == "all" ]]; then
    for m in sparsegpt admm admm_corr; do
        run_one "$m"
    done
else
    run_one "$METHOD"
fi
