#!/usr/bin/env python
"""Benchmark SparseGPT vs ADMM vs ADMM-Corr for 2:4 LLM pruning.

Usage:
    python autoresearch/bench_prune_llm.py --model Qwen/Qwen3.5-2B --method sparsegpt
    python autoresearch/bench_prune_llm.py --model Qwen/Qwen3.5-2B --method admm
    python autoresearch/bench_prune_llm.py --model Qwen/Qwen3.5-2B --method admm_corr
"""

import argparse
import json
import os
import sys
import time

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HOME"] = "/buckets/datasets/huggingface"

from astra.data.llm import get_c4
from astra.hooks import ModuleInputCatcher
from astra.misc import transfer_to_device
from astra.pruners.admm import admm_prune, compute_cross_H, compute_H
from astra.pruners.sparsegpt import sparsegpt_prune

torch.set_float32_matmul_precision('highest')

from lm_eval.evaluator import simple_evaluate as _lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager

_task_mgr = TaskManager()
_hflm_cache = {}


_devnull = open(os.devnull, "w")


def eval_wikitext(model, tokenizer):
    """Run official lm_eval wikitext, return dict with word_ppl, byte_ppl, bpb."""
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
    """Return dict of {dotted_name: nn.Linear} for all projections in a decoder layer."""
    subs = {}
    for name, mod in layer.self_attn.named_children():
        if "_proj" in name:
            subs[f"self_attn.{name}"] = mod
    for name, mod in layer.mlp.named_children():
        if "_proj" in name:
            subs[f"mlp.{name}"] = mod
    return subs


def capture_first_layer_inputs(model, tokenized_data, device):
    """Forward calibration data through model, capture inputs to first decoder layer."""
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
    """Forward layer_inputs through layer, capture inputs to each sublayer."""
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
    """Forward layer_inputs through layer, return next-layer inputs."""
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
    """Flatten captured sublayer inputs into [N*seq, K] on CPU."""
    parts = []
    for inp in sublayer_inputs_list:
        x = inp["args"][0]
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        parts.append(x)
    return torch.cat(parts, dim=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Benchmark 2:4 pruning methods")
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument(
        "--method", choices=["sparsegpt", "admm", "admm_corr"], required=True
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--admm-iter", type=int, default=1000)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument(
        "--eval-tasks",
        default="wikitext,arc_easy,arc_challenge,piqa,winogrande,boolq,lambada_openai",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    method = args.method
    tag = f"{method}_{args.model.split('/')[-1]}"
    out_path = args.output or f"bench_{tag}.json"

    def save_results(results):
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"Model:  {args.model}")
    print(f"Method: {method}")
    print(f"Device: {device}")
    print(f"ADMM iters: {args.admm_iter}")
    print(f"Calibration: {args.num_samples} samples x {args.seq_len} tokens")
    print(f"Output: {out_path}")

    results = {
        "model": args.model,
        "method": method,
        "admm_iter": args.admm_iter if "admm" in method else None,
        "num_samples": args.num_samples,
        "status": "loading",
        "layers_pruned": [],
    }
    save_results(results)

    # ── Load model ────────────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype="auto", device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.eval()

    # ── Baseline eval ─────────────────────────────────────────────────────
    # print("\n=== Baseline Evaluation ===")
    print("\n=== Baseline Evaluation ===")
    baseline = eval_wikitext(model, tokenizer)
    print(f"  word_ppl={baseline['word_ppl']:.4f}  bpb={baseline['bpb']:.4f}")
    results["baseline_word_ppl"] = baseline["word_ppl"]
    results["baseline_bpb"] = baseline["bpb"]
    # results["status"] = "pruning"
    # save_results(results)

    # ── Calibration data ──────────────────────────────────────────────────
    print("\n=== Loading Calibration Data ===")
    c4_data = get_c4(
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        tokenizer=tokenizer,
        seed=42,
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

    print(f"\n=== Pruning ({method}) — {num_layers} layers ===")
    for li in range(num_layers):
        layer = model.model.layers[li]
        subs = get_linear_sublayers(layer)
        sub_names = list(subs.keys())
        print(f"\n── Layer {li}/{num_layers} ({len(sub_names)} sublayers) ──")

        # 1. Capture sublayer inputs (pruned stream, original weights)
        pruned_sub = capture_sublayer_inputs(layer, layer_inputs, subs, device)

        # 2. For ADMM corr: capture dense sublayer inputs + propagate dense
        dense_sub = None
        if dense_inputs is not None:
            dense_sub = capture_sublayer_inputs(layer, dense_inputs, subs, device)
            dense_inputs = propagate_layer(layer, dense_inputs, device)

        # 3. Prune each sublayer
        layer_time = 0.0
        for name in sub_names:
            sublayer = subs[name]
            W0 = sublayer.weight.data.float().to(device)
            X_p = flatten_acts(pruned_sub[name])
            H = compute_H(X_p, device)

            t0 = time.time()

            if method == "sparsegpt":
                W_new = sparsegpt_prune(
                    W0, H, blocksize=128, sparsity=0.5,
                    prune_n=2, prune_m=4, percdamp=0.01,
                )
            else:
                C_target = None
                if dense_sub is not None:
                    X_d = flatten_acts(dense_sub[name])
                    # Diagnostic: verify dense vs pruned streams diverge
                    diff_norm = (X_d - X_p).norm().item()
                    xp_norm = X_p.norm().item()
                    xd_norm = X_d.norm().item()
                    cos = (X_d * X_p).sum() / (xd_norm * xp_norm + 1e-8)
                    print(f"    {name} X_p={xp_norm:.1f} X_d={xd_norm:.1f} diff={diff_norm:.1f} cos={cos:.6f}")
                    cross_H = compute_cross_H(X_d, X_p, device)
                    C_target = W0 @ cross_H
                W_new = admm_prune(
                    W0, H, C_target=C_target, num_iter=args.admm_iter,verbose=True
                )

            dt = time.time() - t0
            total_prune_time += dt
            layer_time += dt

            sublayer.weight.data = W_new.to(sublayer.weight.dtype).to(
                sublayer.weight.device
            )
            print(f"  {name:<25} {dt:6.1f}s  shape={tuple(W0.shape)}")

        # 4. Free sublayer captures
        del pruned_sub, dense_sub

        # 5. Propagate pruned stream (through now-pruned layer)
        layer_inputs = propagate_layer(layer, layer_inputs, device)
        torch.cuda.empty_cache()

        # Evaluate PPL after this layer
        print(f"  Evaluating PPL after layer {li}...", flush=True)
        ppl_result = eval_wikitext(model, tokenizer)
        layer_ppl = ppl_result["word_ppl"]
        layer_bpb = ppl_result["bpb"]
        print(f"  word_ppl={layer_ppl:.4f}  bpb={layer_bpb:.4f}")

        # Save progress after each layer
        results["layers_pruned"].append(
            {"layer": li, "time_s": round(layer_time, 1),
             "word_ppl": layer_ppl, "bpb": layer_bpb}
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
    tasks = [t.strip() for t in args.eval_tasks.split(",") if t.strip() != "wikitext"]
    if tasks:
        print(f"\n=== lm_eval benchmarks: {tasks} ===")
        from lm_eval.evaluator import simple_evaluate
        from lm_eval.models import huggingface
        from lm_eval.tasks import TaskManager

        hf_model = huggingface.HFLM(model, tokenizer=tokenizer)
        task_manager = TaskManager()

        for task in tasks:
            print(f"  {task}...", end=" ", flush=True)
            with torch.no_grad():
                res = simple_evaluate(
                    model=hf_model,
                    tasks=[task],
                    num_fewshot=0,
                    task_manager=task_manager,
                    log_samples=False,
                    batch_size=4,
                    verbosity="ERROR",
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
