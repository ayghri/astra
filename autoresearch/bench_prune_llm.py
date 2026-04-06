#!/usr/bin/env python
"""Benchmark SparseGPT vs ADMM vs ADMM-Corr vs Wanda for 2:4 LLM pruning.

Usage:
    python autoresearch/bench_prune_llm.py method=sparsegpt model=Qwen/Qwen3-1.7B
    python autoresearch/bench_prune_llm.py method=admm fp16=true
    python autoresearch/bench_prune_llm.py method=admm_corr admm_iter=2000
    # Sweep:
    python autoresearch/bench_prune_llm.py -m method=sparsegpt,admm model=Qwen/Qwen3-0.6B,Qwen/Qwen3-1.7B
"""

import json
import os
import subprocess
import sys
import time

import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from astra.data.llm import get_c4
from astra.hooks import ModuleInputCatcher
from astra.misc import transfer_to_device
from astra.pruners.admm import admm_prune as admm_prune_fp32
from astra.pruners.admm import compute_cross_H, compute_H
from astra.pruners.admm_fp16 import admm_prune as admm_prune_fp16
from astra.pruners.sparsegpt import sparsegpt_prune

torch.set_float32_matmul_precision("highest")

import logging

from lm_eval.evaluator import simple_evaluate as _lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager

# Suppress noisy loggers
for _logger_name in ("lm_eval", "httpx", "transformers", "datasets", "huggingface_hub"):
    logging.getLogger(_logger_name).setLevel(logging.ERROR)

_task_mgr = TaskManager()
_hflm_cache = {}
_devnull = open(os.devnull, "w")


def eval_wikitext(model, tokenizer):
    key = id(model)
    if key not in _hflm_cache:
        _hflm_cache[key] = HFLM(model, tokenizer=tokenizer)
    hflm = _hflm_cache[key]
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = _devnull, _devnull
    try:
        with torch.no_grad():
            res = _lm_eval(
                model=hflm, tasks=["wikitext"], num_fewshot=0,
                task_manager=_task_mgr, log_samples=False, batch_size=4,
                verbosity="ERROR",
            )
    finally:
        sys.stdout, sys.stderr = old_out, old_err
    w = res["results"]["wikitext"]
    return {
        "word_ppl": w["word_perplexity,none"],
        "byte_ppl": w["byte_perplexity,none"],
        "bpb": w["bits_per_byte,none"],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_linear_sublayers(layer):
    subs = {}
    for name, mod in layer.self_attn.named_children():
        if "_proj" in name:
            subs[f"self_attn.{name}"] = mod
    for name, mod in layer.mlp.named_children():
        if "_proj" in name:
            subs[f"mlp.{name}"] = mod
    return subs


def capture_first_layer_inputs(model, tokenized_data, device):
    catcher = ModuleInputCatcher(device=torch.device("cpu"))
    catcher.attach(model.model.layers[0], "L0", raise_error=True)
    with torch.no_grad():
        for d in tqdm(tokenized_data, desc="Capturing layer-0 inputs"):
            try:
                model(**transfer_to_device(d, device), labels=None, use_cache=False)
            except RuntimeError:
                pass
    result = catcher.inputs["L0"]
    catcher.detach("L0")
    return result


def capture_sublayer_inputs(layer, layer_inputs, sublayer_dict, device):
    catcher = ModuleInputCatcher(device=torch.device("cpu"))
    for name, mod in sublayer_dict.items():
        catcher.attach(mod, name)
    with torch.no_grad():
        for inp in layer_inputs:
            d = transfer_to_device(inp, device)
            layer(*d["args"], **d["kwargs"])
    result = dict(catcher.inputs)
    for name in sublayer_dict:
        catcher.detach(name)
    return result


def propagate_layer(layer, layer_inputs, device):
    outputs = []
    with torch.no_grad():
        for inp in layer_inputs:
            d = transfer_to_device(inp, device)
            out = layer(*d["args"], **d["kwargs"])
            hidden = out[0] if isinstance(out, tuple) else out
            new_args = (hidden,) + d["args"][1:]
            outputs.append(
                transfer_to_device(
                    {"args": new_args, "kwargs": d["kwargs"]}, torch.device("cpu")
                )
            )
    return outputs


def flatten_acts(sublayer_inputs_list):
    parts = []
    for inp in sublayer_inputs_list:
        x = inp["args"][0]
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        parts.append(x)
    return torch.cat(parts, dim=0)


def get_git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(config_path="configs", config_name="bench", version_base=None)
def main(cfg: DictConfig):
    device = torch.device(cfg.device)
    method = cfg.method
    admm_prune = admm_prune_fp16 if cfg.fp16 else admm_prune_fp32

    # Build experiment directory: output_dir/method_model_timestamp/
    model_tag = cfg.model.split("/")[-1]
    timestamp = time.strftime("%Y%m%d_%H%M")
    exp_name = f"{method}_{model_tag}_{timestamp}"
    if cfg.fp16 and "admm" in method:
        exp_name = f"{method}_fp16_{model_tag}_{timestamp}"
    exp_dir = os.path.join(cfg.output_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    out_path = os.path.join(exp_dir, "results.json")
    log_path = os.path.join(exp_dir, "run.log")
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Tee stdout to log file
    class Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()
        def flush(self):
            for s in self.streams:
                s.flush()

    _log_file = open(log_path, "w")
    sys.stdout = Tee(sys.__stdout__, _log_file)

    git_hash = get_git_hash()
    print(f"Experiment: {exp_dir}", flush=True)

    def save_results(results):
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"Model:      {cfg.model}")
    print(f"Method:     {method}")
    print(f"Device:     {device}")
    if "admm" in method:
        print(f"ADMM iters: {cfg.admm_iter}  fp16: {cfg.fp16}  percdamp: {cfg.percdamp}")
    print(f"Calibration: {cfg.num_samples} x {cfg.seq_len}  seed={cfg.seed}")
    print(f"Experiment: {exp_dir}")
    print(f"Git commit: {git_hash}")

    results = {
        "model": cfg.model,
        "method": method,
        "admm_iter": cfg.admm_iter if "admm" in method else None,
        "fp16": cfg.fp16 if "admm" in method else None,
        "percdamp": cfg.percdamp,
        "num_samples": cfg.num_samples,
        "seq_len": cfg.seq_len,
        "seed": cfg.seed,
        "git_commit": git_hash,
        "experiment_dir": exp_dir,
        "status": "loading",
        "layers_pruned": [],
    }
    save_results(results)

    # ── Load model ────────────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model, dtype="auto", device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    model.eval()

    # ── Baseline eval ─────────────────────────────────────────────────────
    print("\n=== Baseline Evaluation ===")
    baseline = eval_wikitext(model, tokenizer)
    print(f"  word_ppl={baseline['word_ppl']:.4f}  bpb={baseline['bpb']:.4f}")
    results["baseline_word_ppl"] = baseline["word_ppl"]
    results["baseline_bpb"] = baseline["bpb"]
    save_results(results)

    # ── Calibration data ──────────────────────────────────────────────────
    print("\n=== Loading Calibration Data ===")
    c4_data = get_c4(
        num_samples=cfg.num_samples,
        seq_len=cfg.seq_len,
        tokenizer=tokenizer,
        seed=cfg.seed,
    )
    tokenized = [
        {"input_ids": d[0], "attention_mask": torch.ones_like(d[0])} for d in c4_data
    ]

    # ── Capture first-layer inputs ────────────────────────────────────────
    print("\n=== Capturing Layer-0 Inputs ===")
    layer_inputs = capture_first_layer_inputs(model, tokenized, device)
    dense_inputs = list(layer_inputs) if method == "admm_corr" else None

    # ── Layer-by-layer pruning ────────────────────────────────────────────
    num_layers = len(model.model.layers)
    total_prune_time = 0.0
    results["status"] = "pruning"
    save_results(results)

    print(f"\n=== Pruning ({method}) — {num_layers} layers ===")
    for li in range(num_layers):
        layer = model.model.layers[li]
        subs = get_linear_sublayers(layer)
        sub_names = list(subs.keys())
        print(f"\n── Layer {li}/{num_layers} ({len(sub_names)} sublayers) ──")

        pruned_sub = capture_sublayer_inputs(layer, layer_inputs, subs, device)

        dense_sub = None
        if dense_inputs is not None:
            dense_sub = capture_sublayer_inputs(layer, dense_inputs, subs, device)
            dense_inputs = propagate_layer(layer, dense_inputs, device)

        layer_time = 0.0
        layer_sublayers = {}
        for name in sub_names:
            sublayer = subs[name]
            W0 = sublayer.weight.data.float().to(device)
            X_p = flatten_acts(pruned_sub[name])
            H = compute_H(X_p, device)

            t0 = time.time()

            if method == "sparsegpt":
                W_new = sparsegpt_prune(
                    W0, H, blocksize=128, sparsity=0.5,
                    prune_n=2, prune_m=4, percdamp=cfg.percdamp,
                )
            else:
                C_target = None
                if dense_sub is not None:
                    X_d = flatten_acts(dense_sub[name])
                    cross_H = compute_cross_H(X_d, X_p, device)
                    C_target = W0 @ cross_H
                W_new = admm_prune(
                    W0, H, C_target=C_target, num_iter=cfg.admm_iter,
                    percdamp=cfg.percdamp, verbose=True,
                )

            dt = time.time() - t0
            total_prune_time += dt
            layer_time += dt

            if W_new.isnan().any() or W_new.isinf().any():
                print(f"  WARNING: {name} has nan/inf! Falling back to magnitude pruning.")
                W_new = W0.clone()
                W_abs = W_new.abs().reshape(-1, 4)
                mask = torch.zeros_like(W_abs, dtype=torch.bool)
                mask.scatter_(1, W_abs.topk(2, dim=1).indices, True)
                W_new *= mask.reshape(W_new.shape)

            # Reconstruction error
            dW = W_new - W0
            rmse = ((dW @ H) * dW).sum().sqrt().item()
            ref_norm = ((W0 @ H) * W0).sum().sqrt().item()
            rel_rmse = rmse / ref_norm if ref_norm > 0 else float("inf")

            sublayer.weight.data = W_new.to(sublayer.weight.dtype).to(
                sublayer.weight.device
            )
            print(f"  {name:<25} {dt:6.1f}s  RMSE={rmse:.4f}  relRMSE={rel_rmse:.4%}")
            layer_sublayers[name] = {"time_s": round(dt, 1), "rmse": rmse, "rel_rmse": rel_rmse}

        del pruned_sub, dense_sub

        layer_inputs = propagate_layer(layer, layer_inputs, device)
        torch.cuda.empty_cache()

        # Evaluate PPL after this layer
        print(f"  Evaluating PPL after layer {li}...", flush=True)
        ppl_result = eval_wikitext(model, tokenizer)
        layer_ppl = ppl_result["word_ppl"]
        layer_bpb = ppl_result["bpb"]
        print(f"  word_ppl={layer_ppl:.4f}  bpb={layer_bpb:.4f}")

        # Save layer checkpoint
        torch.save(layer.state_dict(), os.path.join(ckpt_dir, f"layer_{li:03d}.pt"))

        results["layers_pruned"].append(
            {"layer": li, "time_s": round(layer_time, 1),
             "word_ppl": layer_ppl, "bpb": layer_bpb,
             "sublayers": layer_sublayers}
        )
        results["prune_time_s"] = round(total_prune_time, 1)
        save_results(results)

    print(f"\nTotal pruning time: {total_prune_time:.1f}s")

    # ── Final wikitext eval ───────────────────────────────────────────────
    print("\n=== Final Wikitext Evaluation ===")
    final = eval_wikitext(model, tokenizer)
    results["final_word_ppl"] = final["word_ppl"]
    results["final_bpb"] = final["bpb"]
    results["status"] = "evaluating_benchmarks"
    save_results(results)

    # ── lm_eval benchmarks ────────────────────────────────────────────────
    tasks = [t.strip() for t in cfg.eval_tasks.split(",") if t.strip() != "wikitext"]
    if tasks:
        print(f"\n=== lm_eval benchmarks: {tasks} ===")
        from lm_eval.evaluator import simple_evaluate
        from lm_eval.models import huggingface

        hf_model = huggingface.HFLM(model, tokenizer=tokenizer)
        task_manager = TaskManager()

        for task in tasks:
            print(f"  {task}...", end=" ", flush=True)
            with torch.no_grad():
                res = simple_evaluate(
                    model=hf_model, tasks=[task], num_fewshot=0,
                    task_manager=task_manager, log_samples=False,
                    batch_size=4, verbosity="ERROR",
                )
            tr = res["results"].get(task, {})
            if task in ("arc_challenge", "winogrande"):
                acc = tr.get("acc_norm,none")
            else:
                acc = tr.get("acc,none") or tr.get("acc_norm,none")
            results[f"{task}_acc"] = acc
            print(f"{acc:.4f}" if acc else "N/A")
            save_results(results)

    # ── Done ──────────────────────────────────────────────────────────────
    results["status"] = "done"
    save_results(results)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for k, v in results.items():
        if k == "layers_pruned":
            continue
        if isinstance(v, float):
            print(f"  {k:<30} {v:.4f}")
        else:
            print(f"  {k:<30} {v}")
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
