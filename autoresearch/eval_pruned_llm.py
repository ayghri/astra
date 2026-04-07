#!/usr/bin/env python
"""Evaluate a pruned LLM by loading layer checkpoints from bench_prune_llm.

Supports:
  - Loading all pruned layers from a checkpoint directory
  - Loading a specific range of layers (e.g., layers 0-15 only)
  - Loading only specific sublayer types (e.g., only o_proj, down_proj)
  - Running wikitext PPL + any lm_eval benchmarks

Usage:
    # Eval all pruned layers
    python autoresearch/eval_pruned_llm.py --model Qwen/Qwen3-1.7B \\
        --ckpt-dir autoresearch/results/admm_Qwen3-1.7B_20260406/checkpoints

    # Eval only layers 0-15
    python autoresearch/eval_pruned_llm.py --model Qwen/Qwen3-1.7B \\
        --ckpt-dir .../checkpoints --layers 0-15

    # Eval only attention projections
    python autoresearch/eval_pruned_llm.py --model Qwen/Qwen3-1.7B \\
        --ckpt-dir .../checkpoints --sublayers q_proj,k_proj,v_proj,o_proj

    # Eval with additional benchmarks
    python autoresearch/eval_pruned_llm.py --model Qwen/Qwen3-1.7B \\
        --ckpt-dir .../checkpoints --tasks wikitext,arc_easy,piqa
"""

import argparse
import json
import logging
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lm_eval.evaluator import simple_evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager

for _name in ("lm_eval", "httpx", "transformers", "datasets", "huggingface_hub"):
    logging.getLogger(_name).setLevel(logging.ERROR)

_devnull = open(os.devnull, "w")


def eval_tasks(model, tokenizer, tasks, task_mgr):
    """Run lm_eval on given tasks, return dict of results."""
    hflm = HFLM(model, tokenizer=tokenizer)
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = _devnull, _devnull
    try:
        with torch.no_grad():
            res = simple_evaluate(
                model=hflm, tasks=tasks, num_fewshot=0,
                task_manager=task_mgr, log_samples=False,
                batch_size=4, verbosity="ERROR",
            )
    finally:
        sys.stdout, sys.stderr = old_out, old_err

    out = {}
    for task in tasks:
        tr = res["results"].get(task, {})
        if task == "wikitext":
            out["word_ppl"] = tr.get("word_perplexity,none")
            out["byte_ppl"] = tr.get("byte_perplexity,none")
            out["bpb"] = tr.get("bits_per_byte,none")
        else:
            if task in ("arc_challenge", "winogrande", "hellaswag"):
                acc = tr.get("acc_norm,none")
            else:
                acc = tr.get("acc,none") or tr.get("acc_norm,none")
            out[f"{task}_acc"] = acc
    return out


def parse_layers(layers_str, num_layers):
    """Parse layer spec: '0-15', '0,3,5', 'all'."""
    if layers_str is None or layers_str == "all":
        return list(range(num_layers))
    if "-" in layers_str and "," not in layers_str:
        start, end = layers_str.split("-")
        return list(range(int(start), int(end) + 1))
    return [int(x) for x in layers_str.split(",")]


def filter_state_dict(state_dict, sublayer_filter):
    """Keep only sublayer types matching the filter (e.g. ['o_proj', 'down_proj'])."""
    if sublayer_filter is None:
        return state_dict
    filtered = {}
    for k, v in state_dict.items():
        for sub in sublayer_filter:
            if sub in k:
                filtered[k] = v
                break
    return filtered


def main():
    parser = argparse.ArgumentParser(description="Evaluate pruned LLM from checkpoints")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--ckpt-dir", required=True, help="Directory with layer_NNN.pt checkpoints")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layers", default=None,
                        help="Layer range: 'all', '0-15', '0,3,5' (default: all available)")
    parser.add_argument("--sublayers", default=None,
                        help="Comma-separated sublayer types to load: 'o_proj,down_proj' (default: all)")
    parser.add_argument("--tasks", default="wikitext",
                        help="Comma-separated eval tasks (default: wikitext)")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    device = torch.device(args.device)
    sublayer_filter = args.sublayers.split(",") if args.sublayers else None
    task_list = [t.strip() for t in args.tasks.split(",")]

    # ── Load model ────────────────────────────────────────────────────────
    print(f"Loading {args.model}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto", device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.eval()

    num_layers = len(model.model.layers)

    # ── Find available checkpoints ────────────────────────────────────────
    available = {}
    for f in sorted(os.listdir(args.ckpt_dir)):
        if f.startswith("layer_") and f.endswith(".pt"):
            idx = int(f.replace("layer_", "").replace(".pt", ""))
            available[idx] = os.path.join(args.ckpt_dir, f)

    requested = parse_layers(args.layers, num_layers)
    to_load = [i for i in requested if i in available]
    missing = [i for i in requested if i not in available]

    print(f"Model layers: {num_layers}")
    print(f"Available checkpoints: {sorted(available.keys())}")
    print(f"Loading layers: {to_load}")
    if sublayer_filter:
        print(f"Sublayer filter: {sublayer_filter}")
    if missing:
        print(f"Missing checkpoints (will stay dense): {missing}")

    # ── Load pruned weights ───────────────────────────────────────────────
    loaded_info = []
    for li in to_load:
        ckpt = torch.load(available[li], map_location=device, weights_only=True)
        filtered = filter_state_dict(ckpt, sublayer_filter)

        layer = model.model.layers[li]
        current_sd = layer.state_dict()

        # Partial load: only replace keys in filtered
        n_replaced = 0
        for k, v in filtered.items():
            if k in current_sd:
                current_sd[k] = v
                n_replaced += 1
        layer.load_state_dict(current_sd)

        # Compute per-sublayer sparsity
        sublayer_info = {}
        for k, v in filtered.items():
            if "weight" in k:
                total = v.numel()
                nnz = (v.abs() > 0).sum().item()
                sublayer_info[k] = {"sparsity": round(1 - nnz / total, 4), "nnz": nnz, "total": total}

        loaded_info.append({"layer": li, "keys_loaded": n_replaced, "sublayers": sublayer_info})
        print(f"  Layer {li}: loaded {n_replaced}/{len(ckpt)} keys"
              + (f" (filtered to {sublayer_filter})" if sublayer_filter else ""))

    # ── Evaluate ──────────────────────────────────────────────────────────
    print(f"\nEvaluating on: {task_list}", flush=True)
    task_mgr = TaskManager()
    t0 = time.time()
    eval_results = eval_tasks(model, tokenizer, task_list, task_mgr)
    dt = time.time() - t0

    print(f"\nResults (took {dt:.0f}s):")
    for k, v in eval_results.items():
        if v is not None:
            print(f"  {k:<30} {v:.4f}" if isinstance(v, float) else f"  {k:<30} {v}")

    # ── Save results ──────────────────────────────────────────────────────
    results = {
        "model": args.model,
        "ckpt_dir": args.ckpt_dir,
        "layers_loaded": to_load,
        "sublayer_filter": sublayer_filter,
        "layers_missing": missing,
        "num_model_layers": num_layers,
        "tasks": task_list,
        "eval_time_s": round(dt, 1),
        "loaded_info": loaded_info,
        **eval_results,
    }

    out_path = args.output
    if out_path is None:
        model_tag = args.model.split("/")[-1]
        layers_tag = args.layers or "all"
        sub_tag = f"_{args.sublayers}" if args.sublayers else ""
        out_path = os.path.join(args.ckpt_dir, "..",
                                f"eval_{layers_tag}{sub_tag}.json")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
