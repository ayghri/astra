"""Compute RMSE and relative RMSE for a pruned layer.

Reports two metrics:
  - single-layer:  pruned layer i output vs original layer i output
  - lookahead:     pruned layer i + frozen layers (i+1..i+lookahead) output
                   vs original full block output
"""

import os
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from astra.hooks import ModuleInputCatcher
from astra.misc import transfer_to_device

checkpoint_dir = Path("/buckets/checkpoints")
os.environ["HF_HOME"] = str(
    Path("~/scratch/buckets/datasets/huggingface").expanduser()
)

model_name = "Qwen/Qwen3-8B"
layer_idx = 0
lookahead = 3
seq_length = 1024
num_samples = 1024

# method_tag = "astra_fp32_la3_acc4_t5_r10_n1024_2of4"

# method_tag= "astra_fp32_la3_kthmid_t20_r10_n1024_2of4"
# method_tag = "astra_fp32_la3_kthmid_t20_r10_n5120_2of4"
method_tag = "astra_admm_la3_t20_r10_n1024_2of4"
ckpt_path = (
    checkpoint_dir / f"{model_name}_decoder_{layer_idx}_{method_tag}.cpt"
)

# ckpt_path = checkpoint_dir / "sparsegpt_Qwen3-8B_20260406_1749/checkpoints/layer_000.pt" # sparsegpt
assert ckpt_path.exists(), f"Pruned checkpoint not found: {ckpt_path}"
print(f"Loading pruned weights from: {ckpt_path}")

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name, dtype="auto", device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Build calibration data
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
wikitext = " ".join(ds["text"])
input_texts = [
    wikitext[i * seq_length : (i + 1) * seq_length] for i in range(num_samples)
]
tokenized = [
    tokenizer([t], return_tensors="pt", return_token_type_ids=False)
    for t in input_texts
]

# Capture inputs to target layer via full model forward
input_catcher = ModuleInputCatcher(device=torch.device("cpu"))
input_catcher.attach(model.model.layers[layer_idx], "tgt")

print("Capturing layer inputs...")
with torch.no_grad():
    for batch in tqdm(tokenized):
        _ = model(**batch.to(model.device), use_cache=False)
layer_inputs = input_catcher.inputs["tgt"]
input_catcher.detach("tgt")

# Capture original (teacher) outputs: single-layer and lookahead
target_layer = model.model.layers[layer_idx]
target_device = next(target_layer.parameters()).device
block_end = min(layer_idx + 1 + lookahead, len(model.model.layers))
frozen_layers = list(model.model.layers[layer_idx + 1 : block_end])
print(f"Lookahead through layers {layer_idx+1}..{block_end-1}")

print("Capturing teacher outputs (single + lookahead)...")
teacher_single = []
teacher_lookahead = []
with torch.no_grad():
    for inputs in tqdm(layer_inputs):
        inputs = transfer_to_device(inputs, target_device)
        x = inputs["args"][0]
        kwargs = inputs["kwargs"]
        y0 = target_layer(x, **kwargs)
        teacher_single.append(y0.cpu())
        ylh = y0
        for fl in frozen_layers:
            ylh = fl(ylh, **kwargs)
        teacher_lookahead.append(ylh.cpu())

# Swap in pruned layer
print(f"\nLoading pruned weights into layer {layer_idx}...")
pruned_state = torch.load(ckpt_path, map_location=target_device)
target_layer.load_state_dict(pruned_state)

# Verify density / dtype
for n, p in target_layer.named_parameters():
    assert p.dtype == torch.bfloat16, f"{n} is {p.dtype}"
    if "_proj.weight" in n:
        density = (p.data.abs() > 0).float().mean().item()
        print(f"  {n}: density={density:.4f}")

# Compute pruned outputs and accumulate squared errors
print("\nComputing pruned outputs and metrics...")
sum_se_single = 0.0
sum_target_se_single = 0.0
sum_se_lookahead = 0.0
sum_target_se_lookahead = 0.0
n_batches = 0

with torch.no_grad():
    for inputs, t_single, t_look in zip(
        tqdm(layer_inputs), teacher_single, teacher_lookahead
    ):
        inputs = transfer_to_device(inputs, target_device)
        x = inputs["args"][0]
        kwargs = inputs["kwargs"]
        y0 = target_layer(x, **kwargs)  # pruned layer
        t_single_dev = t_single.to(target_device).float()
        y0_f = y0.float()
        sum_se_single += (y0_f - t_single_dev).pow(2).mean().item()
        sum_target_se_single += t_single_dev.pow(2).mean().item()

        ylh = y0
        for fl in frozen_layers:
            ylh = fl(ylh, **kwargs)
        t_look_dev = t_look.to(target_device).float()
        ylh_f = ylh.float()
        sum_se_lookahead += (ylh_f - t_look_dev).pow(2).mean().item()
        sum_target_se_lookahead += t_look_dev.pow(2).mean().item()
        n_batches += 1

rmse_single = (sum_se_single / n_batches) ** 0.5
target_rmse_single = (sum_target_se_single / n_batches) ** 0.5
rel_rmse_single = rmse_single / target_rmse_single

rmse_look = (sum_se_lookahead / n_batches) ** 0.5
target_rmse_look = (sum_target_se_lookahead / n_batches) ** 0.5
rel_rmse_look = rmse_look / target_rmse_look

print("\n=== RMSE Results ===")
print(f"Pruned layer: {layer_idx}")
print(f"Calibration samples: {num_samples}")
print()
print(f"Single layer (layer {layer_idx} output):")
print(f"  RMSE:           {rmse_single:.6f}")
print(f"  Target RMSE:    {target_rmse_single:.6f}")
print(f"  Relative RMSE:  {rel_rmse_single:.4f}  ({rel_rmse_single*100:.2f}%)")
print()
print(f"Lookahead (after layers {layer_idx}..{block_end-1}):")
print(f"  RMSE:           {rmse_look:.6f}")
print(f"  Target RMSE:    {target_rmse_look:.6f}")
print(f"  Relative RMSE:  {rel_rmse_look:.4f}  ({rel_rmse_look*100:.2f}%)")
