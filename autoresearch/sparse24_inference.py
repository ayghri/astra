#!/usr/bin/env python
"""Convert pruned LLM to 2:4 semi-structured sparse format and benchmark inference.

Loads layer checkpoints from bench_prune_llm.py, converts Linear weights to
PyTorch's SparseSemiStructuredTensor, and benchmarks inference speed + evaluates.

Usage:
    # Convert + benchmark
    python autoresearch/sparse24_inference.py \
        model=Qwen/Qwen3-1.7B \
        ckpt_dir=autoresearch/results/.../checkpoints

    # Use CUTLASS backend
    python autoresearch/sparse24_inference.py \
        model=Qwen/Qwen3-1.7B \
        ckpt_dir=.../checkpoints \
        backend=cutlass

    # Benchmark only (no eval)
    python autoresearch/sparse24_inference.py \
        model=Qwen/Qwen3-1.7B \
        ckpt_dir=.../checkpoints \
        eval.wikitext=false
"""

import json
import logging
import os
import sys
import time

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from lm_eval.evaluator import simple_evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager

for _name in ("lm_eval", "httpx", "transformers", "datasets", "huggingface_hub"):
    logging.getLogger(_name).setLevel(logging.ERROR)

_devnull = open(os.devnull, "w")


def eval_single_task(hflm, task, task_mgr, verbose=True):
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
    out = {}
    for k, v in tr.items():
        if k == "alias":
            continue
        if isinstance(v, (int, float)):
            out[f"{task}/{k}"] = v
    return out


def verify_24_pattern(weight):
    """Check that weight has valid 2:4 sparsity pattern."""
    w = weight.reshape(-1, 4)
    nnz_per_group = (w != 0).sum(dim=1)
    return (nnz_per_group <= 2).all().item()


def convert_linear_to_sparse24(model, dtype=torch.float16):
    """Convert all Linear layers with 2:4 pattern to semi-structured sparse.

    Returns list of (name, converted, reason) for logging.
    """
    log = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        w = module.weight.data
        # Check dimensions: must be divisible by 4 in the inner dim
        if w.shape[1] % 4 != 0:
            log.append((name, False, f"K={w.shape[1]} not divisible by 4"))
            continue

        # Check 2:4 pattern
        if not verify_24_pattern(w):
            nnz_ratio = (w != 0).float().mean().item()
            if nnz_ratio > 0.99:
                log.append((name, False, "dense (not pruned)"))
            else:
                log.append((name, False, f"not 2:4 pattern (density={nnz_ratio:.3f})"))
            continue

        # Convert
        w_cast = w.to(dtype)
        try:
            w_sparse = to_sparse_semi_structured(w_cast)
            module.weight = nn.Parameter(w_sparse, requires_grad=False)
            log.append((name, True, f"{tuple(w.shape)} -> sparse {dtype}"))
        except Exception as e:
            log.append((name, False, str(e)))

    return log


def benchmark_prefill(model, tokenizer, device, seq_len=512, n_runs=10):
    """Benchmark prefill (time-to-first-token). Compute-bound — benefits from 2:4 sparsity."""
    prompt = "The quick brown fox " * (seq_len // 5)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=seq_len).to(device)
    actual_len = inputs["input_ids"].shape[1]

    # Warmup
    with torch.no_grad():
        model(**inputs)
    torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            model(**inputs)
        torch.cuda.synchronize()
        times.append(time.time() - t0)

    avg = sum(times) / len(times)
    return {
        "seq_len": actual_len,
        "avg_ms": round(avg * 1000, 1),
        "tok_per_sec": round(actual_len / avg, 0),
        "n_runs": n_runs,
    }


def benchmark_decode(model, tokenizer, device, max_new_tokens=64, n_runs=5):
    """Benchmark decode (token generation). Memory-bandwidth-bound."""
    inputs = tokenizer("The capital of France is", return_tensors="pt").to(device)

    # Warmup
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=8, do_sample=False)
    torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        torch.cuda.synchronize()
        n_new = out.shape[1] - inputs["input_ids"].shape[1]
        times.append((time.time() - t0, n_new))

    avg_time = sum(t for t, _ in times) / len(times)
    avg_tokens = sum(n for _, n in times) / len(times)
    return {
        "avg_tokens": int(avg_tokens),
        "avg_ms": round(avg_time * 1000, 1),
        "tok_per_sec": round(avg_tokens / avg_time, 1),
        "n_runs": n_runs,
    }


@hydra.main(config_path="configs", config_name="eval_llm", version_base=None)
def main(cfg: DictConfig):
    device = torch.device(cfg.device)
    backend = cfg.get("backend", "cusparselt")  # cusparselt or cutlass
    sparse_dtype = torch.bfloat16 if cfg.get("sparse_dtype", "bf16") == "bf16" else torch.float16

    if backend == "cutlass":
        SparseSemiStructuredTensor._FORCE_CUTLASS = True

    # ── Load model ────────────────────────────────────────────────────────
    print(f"Loading {cfg.model}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(cfg.model, dtype="auto", device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    model.eval()
    num_layers = len(model.model.layers)

    # ── Load pruned checkpoints ───────────────────────────────────────────
    if cfg.ckpt_dir is not None:
        available = {}
        for f in sorted(os.listdir(cfg.ckpt_dir)):
            if f.startswith("layer_") and f.endswith(".pt"):
                idx = int(f.replace("layer_", "").replace(".pt", ""))
                available[idx] = os.path.join(cfg.ckpt_dir, f)

        print(f"Loading {len(available)} layer checkpoints...", flush=True)
        for li, path in sorted(available.items()):
            ckpt = torch.load(path, map_location=device, weights_only=True)
            model.model.layers[li].load_state_dict(ckpt)
            print(f"  Layer {li}: loaded", flush=True)
    else:
        print("No ckpt_dir — using model weights as-is", flush=True)

    # ── Dense baseline benchmark ──────────────────────────────────────────
    print("\nDense benchmarks...", flush=True)
    dense_prefill = benchmark_prefill(model, tokenizer, device)
    dense_decode = benchmark_decode(model, tokenizer, device)
    print(f"  Prefill: {dense_prefill['tok_per_sec']:.0f} tok/s ({dense_prefill['avg_ms']:.1f}ms, {dense_prefill['seq_len']} tokens)")
    print(f"  Decode:  {dense_decode['tok_per_sec']:.1f} tok/s ({dense_decode['avg_ms']:.1f}ms, {dense_decode['avg_tokens']} tokens)")

    # ── Convert to 2:4 sparse ─────────────────────────────────────────────
    print(f"\nConverting to 2:4 sparse ({backend}, {sparse_dtype})...", flush=True)
    conversion_log = convert_linear_to_sparse24(model, dtype=sparse_dtype)

    n_converted = sum(1 for _, ok, _ in conversion_log if ok)
    n_skipped = sum(1 for _, ok, _ in conversion_log if not ok)
    print(f"  Converted: {n_converted}  Skipped: {n_skipped}")
    for name, ok, reason in conversion_log:
        status = "OK" if ok else "SKIP"
        print(f"    [{status}] {name}: {reason}", flush=True)

    # ── Sparse benchmark ──────────────────────────────────────────────────
    print("\nSparse benchmarks...", flush=True)
    sparse_prefill = benchmark_prefill(model, tokenizer, device)
    sparse_decode = benchmark_decode(model, tokenizer, device)
    prefill_speedup = sparse_prefill["tok_per_sec"] / dense_prefill["tok_per_sec"] if dense_prefill["tok_per_sec"] > 0 else 0
    decode_speedup = sparse_decode["tok_per_sec"] / dense_decode["tok_per_sec"] if dense_decode["tok_per_sec"] > 0 else 0
    print(f"  Prefill: {sparse_prefill['tok_per_sec']:.0f} tok/s ({sparse_prefill['avg_ms']:.1f}ms) — {prefill_speedup:.2f}x")
    print(f"  Decode:  {sparse_decode['tok_per_sec']:.1f} tok/s ({sparse_decode['avg_ms']:.1f}ms) — {decode_speedup:.2f}x")

    # ── Evaluation ────────────────────────────────────────────────────────
    task_list = [name for name, enabled in cfg.eval.items() if enabled]
    results = {
        "model": cfg.model,
        "ckpt_dir": cfg.ckpt_dir,
        "backend": backend,
        "sparse_dtype": str(sparse_dtype),
        "converted_layers": n_converted,
        "skipped_layers": n_skipped,
        "dense_prefill": dense_prefill,
        "dense_decode": dense_decode,
        "sparse_prefill": sparse_prefill,
        "sparse_decode": sparse_decode,
        "prefill_speedup": round(prefill_speedup, 2),
        "decode_speedup": round(decode_speedup, 2),
        "status": "evaluating",
    }

    timestamp = time.strftime("%Y%m%d_%H%M")
    out_path = cfg.output
    if out_path is None:
        parent = os.path.dirname(cfg.ckpt_dir) if cfg.ckpt_dir else "."
        out_path = os.path.join(parent, f"sparse24_{backend}_{timestamp}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    def save():
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    save()

    if task_list:
        print(f"\nEvaluating: {task_list}", flush=True)
        hflm = HFLM(model, tokenizer=tokenizer)
        task_mgr = TaskManager()

        print(f"{'Task':<25} {'Metric':<15} {'Value':>10} {'Time':>8}", flush=True)
        print("-" * 60)

        for task in task_list:
            t0 = time.time()
            task_results = eval_single_task(hflm, task, task_mgr, verbose=cfg.verbose)
            dt = time.time() - t0
            for metric, value in task_results.items():
                if value is not None:
                    val_str = f"{value:.4f}" if isinstance(value, float) else str(value)
                    print(f"  {task:<23} {metric:<15} {val_str:>10} {dt:7.0f}s", flush=True)
            results.update(task_results)
            save()

    results["status"] = "done"
    save()
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
