#!/usr/bin/env python
"""Evaluate a pruned LLM by loading layer checkpoints from bench_prune_llm.

Supports:
  - Loading all pruned layers from a checkpoint directory
  - Loading a specific range of layers (e.g., layers 0-15 only)
  - Loading only specific sublayer types (e.g., only o_proj, down_proj)
  - Running wikitext PPL + lm_eval benchmarks with progress

Usage:
    # Eval all pruned layers
    python autoresearch/eval_pruned_llm.py ckpt_dir=.../checkpoints

    # Only layers 0-15
    python autoresearch/eval_pruned_llm.py ckpt_dir=.../checkpoints layers=0-15

    # Only attention projections
    python autoresearch/eval_pruned_llm.py ckpt_dir=.../checkpoints sublayers=q_proj,k_proj,v_proj,o_proj

    # With MMLU
    python autoresearch/eval_pruned_llm.py ckpt_dir=.../checkpoints eval.mmlu=true
"""

import json
import logging
import os
import sys
import time

import hydra
import torch
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from lm_eval.evaluator import simple_evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager

for _name in ("lm_eval", "httpx", "transformers", "datasets", "huggingface_hub"):
    logging.getLogger(_name).setLevel(logging.ERROR)

_devnull = open(os.devnull, "w")


def eval_single_task(hflm, task, task_mgr, verbose=True):
    """Run one lm_eval task. Returns result dict."""
    verbosity = "INFO" if verbose else "ERROR"
    if not verbose:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = _devnull, _devnull
    try:
        with torch.no_grad():
            res = simple_evaluate(
                model=hflm, tasks=[task], num_fewshot=0,
                task_manager=task_mgr, log_samples=False,
                batch_size=4, verbosity=verbosity,
            )
    finally:
        if not verbose:
            sys.stdout, sys.stderr = old_out, old_err

    tr = res["results"].get(task, {})
    if task == "wikitext":
        return {
            "word_ppl": tr.get("word_perplexity,none"),
            "byte_ppl": tr.get("byte_perplexity,none"),
            "bpb": tr.get("bits_per_byte,none"),
        }
    else:
        if task in ("arc_challenge", "winogrande", "hellaswag"):
            acc = tr.get("acc_norm,none")
        else:
            acc = tr.get("acc,none") or tr.get("acc_norm,none")
        return {f"{task}_acc": acc}


def parse_layers(layers_str, num_layers):
    """Parse layer spec: '0-15', '0,3,5', 'all'."""
    if layers_str is None or layers_str == "all":
        return list(range(num_layers))
    layers_str = str(layers_str)
    if "-" in layers_str and "," not in layers_str:
        start, end = layers_str.split("-")
        return list(range(int(start), int(end) + 1))
    return [int(x) for x in layers_str.split(",")]


def filter_state_dict(state_dict, sublayer_filter):
    if sublayer_filter is None:
        return state_dict
    return {k: v for k, v in state_dict.items()
            if any(sub in k for sub in sublayer_filter)}


@hydra.main(config_path="configs", config_name="eval_llm", version_base=None)
def main(cfg: DictConfig):
    assert cfg.ckpt_dir is not None, "ckpt_dir is required"

    device = torch.device(cfg.device)
    sublayer_filter = cfg.sublayers.split(",") if cfg.sublayers else None

    # Build task list from config
    task_list = [name for name, enabled in cfg.eval.items() if enabled]

    # ── Load model ────────────────────────────────────────────────────────
    print(f"Loading {cfg.model}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(cfg.model, dtype="auto", device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    model.eval()
    num_layers = len(model.model.layers)

    # ── Find available checkpoints ────────────────────────────────────────
    available = {}
    for f in sorted(os.listdir(cfg.ckpt_dir)):
        if f.startswith("layer_") and f.endswith(".pt"):
            idx = int(f.replace("layer_", "").replace(".pt", ""))
            available[idx] = os.path.join(cfg.ckpt_dir, f)

    requested = parse_layers(cfg.layers, num_layers)
    to_load = [i for i in requested if i in available]
    missing = [i for i in requested if i not in available]

    print(f"Model: {cfg.model} ({num_layers} layers)")
    print(f"Checkpoints: {len(available)} available, loading {len(to_load)}")
    if sublayer_filter:
        print(f"Sublayer filter: {sublayer_filter}")
    if missing:
        print(f"Missing (stay dense): {missing}")
    print(f"Tasks: {task_list}")

    # ── Load pruned weights ───────────────────────────────────────────────
    loaded_info = []
    for li in to_load:
        ckpt = torch.load(available[li], map_location=device, weights_only=True)
        filtered = filter_state_dict(ckpt, sublayer_filter)

        layer = model.model.layers[li]
        current_sd = layer.state_dict()
        n_replaced = 0
        for k, v in filtered.items():
            if k in current_sd:
                current_sd[k] = v
                n_replaced += 1
        layer.load_state_dict(current_sd)

        # Per-sublayer sparsity
        sublayer_info = {}
        for k, v in filtered.items():
            if "weight" in k:
                total = v.numel()
                nnz = (v.abs() > 0).sum().item()
                sublayer_info[k] = round(1 - nnz / total, 4)

        loaded_info.append({"layer": li, "keys": n_replaced, "sparsity": sublayer_info})
        print(f"  Layer {li:3d}: {n_replaced} keys loaded", flush=True)

    # ── Evaluate each task with progress ──────────────────────────────────
    print(f"\n{'Task':<25} {'Metric':<15} {'Value':>10} {'Time':>8}", flush=True)
    print("-" * 60, flush=True)

    hflm = HFLM(model, tokenizer=tokenizer)
    task_mgr = TaskManager()
    all_results = {}
    total_eval_time = 0

    for task in task_list:
        t0 = time.time()
        task_results = eval_single_task(hflm, task, task_mgr, verbose=cfg.verbose)
        dt = time.time() - t0
        total_eval_time += dt

        for metric, value in task_results.items():
            if value is not None:
                val_str = f"{value:.4f}" if isinstance(value, float) else str(value)
                print(f"  {task:<23} {metric:<15} {val_str:>10} {dt:7.0f}s", flush=True)

        all_results.update(task_results)

    print("-" * 60)
    print(f"  Total eval time: {total_eval_time:.0f}s")

    # ── Save results ──────────────────────────────────────────────────────
    results = {
        "model": cfg.model,
        "ckpt_dir": cfg.ckpt_dir,
        "layers_loaded": to_load,
        "sublayer_filter": sublayer_filter,
        "num_model_layers": num_layers,
        "tasks": task_list,
        "eval_time_s": round(total_eval_time, 1),
        "loaded_info": loaded_info,
        **all_results,
    }

    out_path = cfg.output
    if out_path is None:
        model_tag = cfg.model.split("/")[-1]
        layers_tag = str(cfg.layers) if cfg.layers else "all"
        sub_tag = f"_{cfg.sublayers}" if cfg.sublayers else ""
        out_path = os.path.join(
            os.path.dirname(cfg.ckpt_dir),
            f"eval_{layers_tag}{sub_tag}.json",
        )

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
