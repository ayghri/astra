"""Load a pruned layer checkpoint, swap it into the model, compute wikitext PPL.

Reports baseline PPL (unpruned) and pruned PPL for the same model.
"""
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from astra.evaluate import evaluate_ppl_hf


checkpoint_dir = Path("/buckets/checkpoints")
os.environ["HF_HOME"] = str(Path("~/scratch/buckets/datasets/huggingface").expanduser())

model_name = "Qwen/Qwen3-8B"
layer_idx = 0

# Match the tag built by prune_llm_fp32.py
# method_tag = "astra_fp32_la3_acc4_t5_r10_n1024_2of4"
# method_tag= "astra_fp32_la3_kthmid_t20_r10_n1024_2of4"

# method_tag = "astra_fp32_la3_kthmid_t20_r10_n5120_2of4"
method_tag = "astra_admm_la3_t20_r10_n1024_2of4"

ckpt_path = checkpoint_dir / f"{model_name}_decoder_{layer_idx}_{method_tag}.cpt"

assert ckpt_path.exists(), f"Pruned checkpoint not found: {ckpt_path}"
print(f"Loading pruned weights from: {ckpt_path}")

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name, dtype="auto", device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Baseline PPL (no pruning)
print("\n=== Baseline (unpruned) ===")
baseline_results = evaluate_ppl_hf(model, tokenizer, silent=True)
print(baseline_results)

# Swap in the pruned layer
print(f"\nReplacing layer {layer_idx} with pruned weights...")
target_layer = model.model.layers[layer_idx]
target_device = next(target_layer.parameters()).device
pruned_state = torch.load(ckpt_path, map_location=target_device)
target_layer.load_state_dict(pruned_state)

# Verify dtype + sparsity
for n, p in target_layer.named_parameters():
    assert p.dtype == torch.bfloat16, f"{n} is {p.dtype}"
    if "_proj.weight" in n:
        density = (p.data.abs() > 0).float().mean().item()
        print(f"  {n}: density={density:.4f}")

print(f"\n=== After replacing layer {layer_idx} (2:4 pruned) ===")
pruned_results = evaluate_ppl_hf(model, tokenizer, silent=True)
print(pruned_results)

# Summary
def ppl(r):
    return r["wikitext"]["word_perplexity,none"]

print("\n=== Summary ===")
print(f"Baseline PPL:  {ppl(baseline_results):.4f}")
print(f"Pruned PPL:    {ppl(pruned_results):.4f}")
print(f"Delta:         {ppl(pruned_results) - ppl(baseline_results):+.4f}")
print(f"Relative:      {(ppl(pruned_results) / ppl(baseline_results) - 1) * 100:+.2f}%")
