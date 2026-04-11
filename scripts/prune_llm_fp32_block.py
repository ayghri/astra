"""Prune a block of `prune_size` consecutive decoder layers simultaneously,
using `lookahead` frozen layers downstream for nonlinear gradient signal.

bf16 model + fp32 grads (with hook-based fp32 accumulation) + warmup epoch.
Saves each pruned layer's state_dict to /buckets/checkpoints/.
"""

import torch
import os
from pathlib import Path

from astra.hooks import ModuleInputCatcher, ModuleOutputCatcher
from copy import deepcopy
from tqdm import tqdm
from torch.optim import Adam, AdamW
from torch import nn
import numpy as np
import torch.nn.utils.prune as prune

from astra.misc import transfer_to_device
from astra.proximals import AdamProxy

from sparsekit import BlockSpec
from sparsekit import ScopeSpec
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from datasets import load_dataset

torch.backends.cuda.enable_flash_sdp(True)


class FP32Optimizer:
    """Optimizer wrapper: bf16 model, fp32 grad accumulation/states/step."""

    def __init__(self, optimizer_cls, params, accum_steps=1, **kwargs):
        all_params = [p for p in params if p.requires_grad]
        self.bf16_params = all_params
        self.fp32_params = [
            p.detach().float().clone().requires_grad_(True) for p in all_params
        ]
        self._id_to_fp32 = {
            id(bp): fp for bp, fp in zip(self.bf16_params, self.fp32_params)
        }
        self.optimizer = optimizer_cls(self.fp32_params, **kwargs)
        self.accum_steps = accum_steps
        self._fp32_grads = [torch.zeros_like(fp) for fp in self.fp32_params]
        self._hooks = []
        for i, bp in enumerate(self.bf16_params):
            h = bp.register_post_accumulate_grad_hook(self._make_hook(i))
            self._hooks.append(h)

    def _make_hook(self, idx):
        def hook(param):
            if param.grad is not None:
                self._fp32_grads[idx].add_(param.grad.float())
                param.grad = None

        return hook

    def fp32_of(self, param):
        return self._id_to_fp32[id(param)]

    def zero_grad(self):
        for g in self._fp32_grads:
            g.zero_()
        for bp in self.bf16_params:
            bp.grad = None

    def step(self):
        scale = 1.0 / self.accum_steps
        for fp, g in zip(self.fp32_params, self._fp32_grads):
            fp.grad = g * scale
        self.optimizer.step()
        for bp, fp in zip(self.bf16_params, self.fp32_params):
            bp.data.copy_(fp.data)
        self.zero_grad()

    def sync_bf16_to_fp32(self):
        for bp, fp in zip(self.bf16_params, self.fp32_params):
            fp.data.copy_(bp.data)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


base_dir = Path("/buckets")
checkpoint_dir = base_dir / "checkpoints"
checkpoint_dir.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(Path("~/scratch/buckets/datasets/huggingface").expanduser())


print(os.environ["HF_HOME"])

seq_length = 1024
num_samples = 32
accum_steps = 4

prune_size = 3  # number of consecutive layers to prune at once
lookahead = 3  # frozen layers after the prune block (for nonlinear signal)
threshold_epochs = 5
recover_epochs = 10

model_name = "Qwen/Qwen3-8B"

method_tag = (
    f"astra_fp32_blk{prune_size}_la{lookahead}"
    f"_acc{accum_steps}_t{threshold_epochs}_r{recover_epochs}"
    f"_n{num_samples}_2of4"
)

teacher = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

student = teacher

# --- Calibration data ---
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
wikitext = " ".join(ds["text"])
input_text = [
    wikitext[i * seq_length : (i + 1) * seq_length] for i in range(num_samples)
]
tokenized_inputs = [
    tokenizer([t], return_tensors="pt", return_token_type_ids=False) for t in input_text
]

input_catcher = ModuleInputCatcher(device=torch.device("cpu"))
output_catcher = ModuleOutputCatcher(device=torch.device("cpu"))

# --- Capture inputs to layer 0 ---
input_catcher.attach(student.model.layers[0], "block_input")
print("Computing teacher inputs to layer 0...")
with torch.no_grad():
    for batch in tqdm(tokenized_inputs):
        _ = teacher(**batch.to(teacher.device), labels=None, use_cache=False)
student_inputs = input_catcher.inputs["block_input"]
input_catcher.detach("block_input")

n_layers = len(student.model.layers)

# --- Block-by-block loop ---
# Process only the first block (size = prune_size). Change to a loop over blocks
# if you want to prune the whole model.
for block_start in [0]:  # iterate by `prune_size` if you want full-model pruning
    block_end = min(block_start + prune_size, n_layers)
    target_layers = list(student.model.layers[block_start:block_end])
    target_indices = list(range(block_start, block_end))

    look_end = min(block_end + lookahead, n_layers)
    frozen_layers = list(student.model.layers[block_end:look_end])

    print(
        f"\n=== Pruning layers {target_indices} (lookahead {block_end}..{look_end-1}) ==="
    )

    target_device = next(target_layers[0].parameters()).device

    # --- Capture teacher targets: run target+frozen on inputs ---
    teacher_targets = []
    print("Capturing teacher targets...")
    with torch.no_grad():
        for inputs in tqdm(student_inputs):
            inputs = transfer_to_device(inputs, target_device)
            x = inputs["args"][0]
            kwargs = inputs["kwargs"]
            for tl in target_layers:
                x = tl(x, **kwargs)
            for fl in frozen_layers:
                x = fl(x, **kwargs)
            teacher_targets.append(x.cpu())

    # --- Free memory: drop everything except target + frozen layers ---
    # After target capture we only need to forward through target_layers and
    # frozen_layers using the cached student_inputs.
    keep = set(range(block_start, look_end))
    for i in range(n_layers):
        if i not in keep:
            student.model.layers[i] = nn.Identity()
    student.model.embed_tokens = nn.Identity()
    student.model.norm = nn.Identity()
    student.lm_head = nn.Identity()
    torch.cuda.empty_cache()
    print(
        f"Freed unused layers; kept layers {sorted(keep)} "
        f"(target={target_indices}, frozen={list(range(block_end, look_end))})"
    )

    # --- Set requires_grad: only target layers' non-norm params trainable ---
    for p in student.model.parameters():
        p.requires_grad = False
    for tl in target_layers:
        for n, p in tl.named_parameters():
            p.requires_grad = "norm" not in n

    # Save originals so we can reload after threshold phase
    original_weights = [deepcopy(tl.state_dict()) for tl in target_layers]

    criterion = nn.MSELoss()

    # --- Build prune_layers + groups across all target layers ---
    prune_layers = {}  # name -> nn.Linear
    groups = []
    for li, tl in enumerate(target_layers):
        for n, l in tl.self_attn.named_children():
            if isinstance(l, nn.Linear):
                prune_layers[f"L{li}_attn_{n}"] = l
        for n, l in tl.mlp.named_children():
            if isinstance(l, nn.Linear):
                prune_layers[f"L{li}_mlp_{n}"] = l

        for n, p in tl.self_attn.named_parameters():
            if "_proj.weight" in n and p.requires_grad:
                groups.append(
                    ScopeSpec(
                        BlockSpec(p, shape=(1, 1)), shape=(1, 4), name=f"L{li}_{n}"
                    )
                )
        for n, p in tl.mlp.named_parameters():
            if "_proj.weight" in n and p.requires_grad:
                groups.append(
                    ScopeSpec(
                        BlockSpec(p, shape=(1, 1)), shape=(1, 4), name=f"L{li}_{n}"
                    )
                )

    groups_nnz = [2] * len(groups)  # 2:4 sparsity
    print(f"Total groups (projections): {len(groups)}")

    lambds = {g: torch.zeros_like(g.kth_largest(None, 1)) for g in groups}
    beta = 0.9

    # --- Capture inputs into each linear projection (for alphas) ---
    for name, layer in prune_layers.items():
        input_catcher.attach(layer, name)

    print("Capturing projection inputs (for alpha)...")
    with torch.no_grad():
        for inputs in tqdm(student_inputs):
            inputs = transfer_to_device(inputs, target_device)
            x = inputs["args"][0]
            kwargs = inputs["kwargs"]
            for tl in target_layers:
                x = tl(x, **kwargs)

    # --- Compute alphas (per-projection input scale) ---
    # alphas = {}
    # for g_idx, (name, layer) in enumerate(prune_layers.items()):
    #     X = 0.0
    #     for ins in input_catcher.inputs[name]:
    #         batch = ins["args"][0][0]
    #         X = X + batch.to(target_device).square().mean(dim=0)
    #     X = X / len(input_catcher.inputs[name]) + 1e-12
    #     alphas[groups[g_idx].block] = X.unsqueeze(0)

    for name in list(prune_layers.keys()):
        input_catcher.detach(name)

    # for b in alphas:
    #     alphas[b] = ((alphas[b] / alphas[b].mean()) * 1e-3).clamp_(min=1e-4).to(target_device)

    # --- Eval initial loss ---
    print("\n--- Eval initial loss (block) ---")
    pbar = tqdm(range(len(student_inputs)), desc="Eval initial loss")
    total_se = 0.0
    total_target_se = 0.0
    n_b = 0
    with torch.no_grad():
        for idx in pbar:
            inputs = transfer_to_device(student_inputs[idx], target_device)
            target = transfer_to_device(teacher_targets[idx], target_device)
            x = inputs["args"][0]
            kwargs = inputs["kwargs"]
            for tl in target_layers:
                x = tl(x, **kwargs)
            for fl in frozen_layers:
                x = fl(x, **kwargs)
            total_se += criterion(x.float(), target.float()).item()
            total_target_se += target.float().pow(2).mean().item()
            n_b += 1
            pbar.set_postfix(
                rmse=f"{(total_se / n_b) ** 0.5:.6f}",
                rel_rmse=f"{(total_se / total_target_se) ** 0.5:.4f}",
                density=np.mean([g.nnz() / g.block.numblk() for g in groups]),
            )

    torch.cuda.empty_cache()

    # --- Threshold phase ---
    threshold_lr = 2e-5
    optimizer = FP32Optimizer(
        Adam,
        [p for tl in target_layers for p in tl.parameters()],
        accum_steps=accum_steps,
        lr=threshold_lr,
        weight_decay=0.0,
        betas=(0.9, 0.999),
    )
    proxy = AdamProxy(optimizer.optimizer)

    # warmup
    warmup_scale = 1e-8
    for pg in optimizer.optimizer.param_groups:
        pg["lr"] = threshold_lr * warmup_scale
    indices = np.random.permutation(len(student_inputs))
    pbar = tqdm(range(0, len(indices), accum_steps), desc="Warmup (threshold)")
    for start in pbar:
        bi = indices[start : start + accum_steps]
        n_micro = len(bi)
        optimizer.zero_grad()
        bl = 0.0
        for idx in bi:
            inputs = transfer_to_device(student_inputs[idx], target_device)
            target = transfer_to_device(teacher_targets[idx], target_device)
            x = inputs["args"][0]
            kwargs = inputs["kwargs"]
            for tl in target_layers:
                x = tl(x, **kwargs)
            for fl in frozen_layers:
                x = fl(x, **kwargs)
            loss = criterion(x.float(), target.float()) / n_micro
            loss.backward()
            bl += loss.item()
        optimizer.step()
        pbar.set_postfix(loss=f"{bl:.6f}")
    for pg in optimizer.optimizer.param_groups:
        pg["lr"] = threshold_lr

    print("\n--- Threshold training ---")
    for epoch in range(threshold_epochs):
        indices = np.random.permutation(len(student_inputs))
        pbar = tqdm(
            range(0, len(indices), accum_steps),
            desc=f"Epoch {epoch + 1}/{threshold_epochs}",
        )
        total_loss = 0.0
        n_steps = 0
        for start in pbar:
            bi = indices[start : start + accum_steps]
            n_micro = len(bi)
            optimizer.zero_grad()
            bl = 0.0
            for idx in bi:
                inputs = transfer_to_device(student_inputs[idx], target_device)
                target = transfer_to_device(teacher_targets[idx], target_device)
                x = inputs["args"][0]
                kwargs = inputs["kwargs"]
                for tl in target_layers:
                    x = tl(x, **kwargs)
                for fl in frozen_layers:
                    x = fl(x, **kwargs)
                loss = criterion(x.float(), target.float()) / n_micro
                loss.backward()
                bl += loss.item()
            optimizer.step()
            n_steps += 1
            total_loss += bl

            for g_nnz, g in zip(groups_nnz, groups):
                block = g.block
                data = block.view.param.data.clone()
                gradient, lr, conditioner = proxy.get_info(
                    optimizer.fp32_of(block.view.param)
                )
                # psi = (gradient - alphas[g.block] * block.view.param.data).abs()
                psi = gradient - conditioner * block.view.param.data
                vals = g.kth_mid({block: psi}, nnz=g_nnz)
                # vals.add_(g.kth_largest({block: psi}, nnz=g_nnz))
                # vals.mul_(0.5)
                lambds[g].mul_(beta).add_((1 - beta) * vals)
                g.soft_threshold(lambds[g] * lr, conditioners={block: conditioner})
                m = g.get_masks(nnz=g_nnz)[block].float()
                block.view.param.data.copy_(data * m + (1 - m) * block.view.param.data)

                # g.soft_threshold(lambds[g] * lr, conditioners={block: conditioner})

            optimizer.sync_bf16_to_fp32()
            pbar.set_postfix(
                loss=f"{total_loss / n_steps:.6f}",
                density=np.mean([g.nnz() / g.numblk() for g in groups]),
            )

    optimizer.remove_hooks()
    del optimizer
    torch.cuda.empty_cache()

    # --- Extract masks ---
    param_masks = {}
    for g_nnz, g in zip(groups_nnz, groups):
        for b, m in g.get_masks(nnz=g_nnz).items():
            param_masks[b.view.param] = m
    for p, m in param_masks.items():
        print(p.shape, m.sum() / m.numel())

    # --- Reload originals, apply masks, register pruning hooks ---
    for tl, ow in zip(target_layers, original_weights):
        tl.load_state_dict(deepcopy(ow))
    for p, m in param_masks.items():
        p.data.mul_(m)
    for tl in target_layers:
        for name, layer in list(tl.self_attn.named_children()) + list(
            tl.mlp.named_children()
        ):
            if isinstance(layer, nn.Linear) and layer.weight in param_masks:
                prune.custom_from_mask(layer, "weight", param_masks[layer.weight])

    # --- Recovery phase ---
    recover_lr = 4e-5
    optimizer = FP32Optimizer(
        AdamW,
        [p for tl in target_layers for p in tl.parameters()],
        accum_steps=accum_steps,
        lr=recover_lr,
        weight_decay=1e-3,
        betas=(0.9, 0.999),
    )

    # warmup
    for pg in optimizer.optimizer.param_groups:
        pg["lr"] = recover_lr * warmup_scale
    indices = np.random.permutation(len(student_inputs))
    pbar = tqdm(range(0, len(indices), accum_steps), desc="Warmup (recovery)")
    for start in pbar:
        bi = indices[start : start + accum_steps]
        n_micro = len(bi)
        optimizer.zero_grad()
        bl = 0.0
        for idx in bi:
            inputs = transfer_to_device(student_inputs[idx], target_device)
            target = transfer_to_device(teacher_targets[idx], target_device)
            x = inputs["args"][0]
            kwargs = inputs["kwargs"]
            for tl in target_layers:
                x = tl(x, **kwargs)
            for fl in frozen_layers:
                x = fl(x, **kwargs)
            loss = criterion(x.float(), target.float()) / n_micro
            loss.backward()
            bl += loss.item()
        optimizer.step()
        pbar.set_postfix(loss=f"{bl:.6f}")
    for pg in optimizer.optimizer.param_groups:
        pg["lr"] = recover_lr

    print("\n--- Recovery training ---")
    for epoch in range(recover_epochs):
        indices = np.random.permutation(len(student_inputs))
        pbar = tqdm(
            range(0, len(indices), accum_steps),
            desc=f"Epoch {epoch + 1}/{recover_epochs}",
        )
        total_loss = 0.0
        n_steps = 0
        for start in pbar:
            bi = indices[start : start + accum_steps]
            n_micro = len(bi)
            optimizer.zero_grad()
            bl = 0.0
            for idx in bi:
                inputs = transfer_to_device(student_inputs[idx], target_device)
                target = transfer_to_device(teacher_targets[idx], target_device)
                x = inputs["args"][0]
                kwargs = inputs["kwargs"]
                for tl in target_layers:
                    x = tl(x, **kwargs)
                for fl in frozen_layers:
                    x = fl(x, **kwargs)
                loss = criterion(x.float(), target.float()) / n_micro
                loss.backward()
                bl += loss.item()
            optimizer.step()
            n_steps += 1
            total_loss += bl
            pbar.set_postfix(loss=f"{total_loss / n_steps:.6f}")

    optimizer.remove_hooks()
    del optimizer
    torch.cuda.empty_cache()

    # --- Commit pruning, save each layer ---
    for tl in target_layers:
        for name, layer in list(tl.self_attn.named_children()) + list(
            tl.mlp.named_children()
        ):
            if isinstance(layer, nn.Linear):
                prune.remove(layer, "weight")
    for p, m in param_masks.items():
        p.data.mul_(m)

    for li, tl in enumerate(target_layers):
        actual_idx = block_start + li
        ckpt_path = (
            checkpoint_dir / f"{model_name}_decoder_{actual_idx}_{method_tag}.cpt"
        )
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving layer {actual_idx} weights to {ckpt_path}")
        torch.save(tl.state_dict(), ckpt_path)

    # --- Eval final loss ---
    print("\n--- Eval final loss (block) ---")
    pbar = tqdm(range(len(student_inputs)), desc="Eval final loss")
    total_se = 0.0
    total_target_se = 0.0
    n_b = 0
    with torch.no_grad():
        for idx in pbar:
            inputs = transfer_to_device(student_inputs[idx], target_device)
            target = transfer_to_device(teacher_targets[idx], target_device)
            x = inputs["args"][0]
            kwargs = inputs["kwargs"]
            for tl in target_layers:
                x = tl(x, **kwargs)
            for fl in frozen_layers:
                x = fl(x, **kwargs)
            total_se += criterion(x.float(), target.float()).item()
            total_target_se += target.float().pow(2).mean().item()
            n_b += 1
            pbar.set_postfix(
                rmse=f"{(total_se / n_b) ** 0.5:.6f}",
                rel_rmse=f"{(total_se / total_target_se) ** 0.5:.4f}",
                density=np.mean([g.nnz() / g.numblk() for g in groups]),
            )

    # --- Verify dtype ---
    for tl in target_layers:
        for n, p in tl.named_parameters():
            assert p.dtype == torch.bfloat16, f"{n} is {p.dtype}"
    print("All target layer weights confirmed bfloat16")
