#!/usr/bin/env python
"""Prune an LLM to 2:4 structured sparsity using a given method.

Saves pruned weights to disk. No evaluation — use bench_prune_llm.py for benchmarks.

Usage:
    python autoresearch/prune_llm.py --model Qwen/Qwen3-1.7B --method sparsegpt
    python autoresearch/prune_llm.py --model Qwen/Qwen3-1.7B --method admm --admm-iter 2000
    python autoresearch/prune_llm.py --model Qwen/Qwen3-1.7B --method admm_corr
"""

import argparse
import json
import os
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prune an LLM to 2:4 sparsity")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument(
        "--method", choices=["sparsegpt", "admm", "admm_corr"], required=True
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--admm-iter", type=int, default=2000)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument(
        "--save-model", default=None,
        help="Directory to save pruned model. Default: pruned_<method>_<model>/",
    )
    parser.add_argument(
        "--output", default=None,
        help="JSON log path. Default: prune_<method>_<model>.json",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    method = args.method
    model_tag = args.model.split("/")[-1]
    tag = f"{method}_{model_tag}"
    out_path = args.output or f"prune_{tag}.json"
    save_dir = args.save_model or f"pruned_{tag}"

    def save_log(log):
        with open(out_path, "w") as f:
            json.dump(log, f, indent=2)

    print(f"Model:      {args.model}")
    print(f"Method:     {method}")
    print(f"Device:     {device}")
    if "admm" in method:
        print(f"ADMM iters: {args.admm_iter}")
    print(f"Samples:    {args.num_samples} x {args.seq_len}")
    print(f"Log:        {out_path}")
    print(f"Save to:    {save_dir}")

    log = {
        "model": args.model,
        "method": method,
        "admm_iter": args.admm_iter if "admm" in method else None,
        "num_samples": args.num_samples,
        "seq_len": args.seq_len,
        "status": "loading",
        "layers": [],
    }
    save_log(log)

    # ── Load model ────────────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype="auto", device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.eval()
    num_layers = len(model.model.layers)
    print(f"Loaded {args.model} — {num_layers} layers")

    # ── Calibration data ──────────────────────────────────────────────────
    print("\nLoading calibration data...")
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
    layer_inputs = capture_first_layer_inputs(model, tokenized, device)
    dense_inputs = list(layer_inputs) if method == "admm_corr" else None

    log["status"] = "pruning"
    save_log(log)

    # ── Layer-by-layer pruning ────────────────────────────────────────────
    total_time = 0.0
    print(f"\nPruning ({method}) — {num_layers} layers")

    for li in range(num_layers):
        layer = model.model.layers[li]
        subs = get_linear_sublayers(layer)
        sub_names = list(subs.keys())
        print(f"\n── Layer {li}/{num_layers} ──")

        pruned_sub = capture_sublayer_inputs(layer, layer_inputs, subs, device)

        dense_sub = None
        if dense_inputs is not None:
            dense_sub = capture_sublayer_inputs(layer, dense_inputs, subs, device)
            dense_inputs = propagate_layer(layer, dense_inputs, device)

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
                    cross_H = compute_cross_H(X_d, X_p, device)
                    C_target = W0 @ cross_H
                W_new = admm_prune(
                    W0, H, C_target=C_target, num_iter=args.admm_iter,
                )

            dt = time.time() - t0
            total_time += dt
            layer_time += dt

            sublayer.weight.data = W_new.to(sublayer.weight.dtype).to(
                sublayer.weight.device
            )
            print(f"  {name:<25} {dt:6.1f}s  {tuple(W0.shape)}")

        del pruned_sub, dense_sub

        layer_inputs = propagate_layer(layer, layer_inputs, device)
        torch.cuda.empty_cache()

        log["layers"].append({"layer": li, "time_s": round(layer_time, 1)})
        log["prune_time_s"] = round(total_time, 1)
        save_log(log)

    # ── Save pruned model ─────────────────────────────────────────────────
    print(f"\nSaving pruned model to {save_dir}...")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    log["status"] = "done"
    log["save_dir"] = save_dir
    save_log(log)

    print(f"\nDone. Total prune time: {total_time:.1f}s")
    print(f"Model saved to: {save_dir}")
    print(f"Log saved to:   {out_path}")


if __name__ == "__main__":
    main()
