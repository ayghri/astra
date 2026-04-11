"""2:4 prune the first decoder layer with notebook-style threshold training.

Pipeline:
  1. Capture inputs to target layer (full model fwd, no_grad)
  2. Capture teacher targets through target + lookahead frozen layers
  3. Threshold training: AdamW on fp32 W, kth_mid + soft_threshold proximal
     step on fp32 weights every grad_accum_steps batches
  4. Extract 2:4 mask, reload originals, apply mask, register prune hook
  5. Recovery training: AdamW + frozen mask
  6. Save bf16 state_dict
"""

import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.utils.prune as prune
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer

from astra.data.llm import get_c4
from astra.hooks import ModuleInputCatcher
from astra.misc import transfer_to_device
from astra.proximals import AdamProxy
from sparsekit import BlockSpec, ScopeSpec

torch.backends.cuda.enable_flash_sdp(True)


# ============================================================================
# Config
# ============================================================================
base_dir = Path("~/alpine/").expanduser()
checkpoint_dir = base_dir / "checkpoints"
checkpoint_dir.mkdir(parents=True, exist_ok=True)

model_name = "Qwen/Qwen3-8B"
seq_length = 1024*2
num_samples = 1024*4
lookahead = 4
threshold_epochs = 20
recover_epochs = 10

threshold_lr = 2e-4
threshold_wd = 0.02
threshold_betas = (0.98, 0.99)

recover_lr = 5e-5
recover_wd = 1e-3
recover_betas = (0.95, 0.999)

grad_accum_steps = 16
clip_norm = 1.0
warmup_scale = 1e-8
beta = 0.98
k_val_weight = 1999.0

method_tag = (
    f"astra_fp32_la{lookahead}_kthmid"
    f"_t{threshold_epochs}_r{recover_epochs}_n{num_samples}_2of4"
)


# ============================================================================
# FP32Optimizer
# ============================================================================
class FP32Optimizer:
    """bf16 model + fp32 grad accumulation + fp32 optimizer state.

    Each backward() fires a hook that upcasts the bf16 grad to fp32 and adds
    it to an internal accumulator, then clears the bf16 grad. Hosting the
    inner optimizer on fp32 copies of the trainable params keeps Adam state
    in fp32 and avoids bf16+bf16 grad accumulation rounding.
    """

    def __init__(self, optimizer_cls, params, **kwargs):
        self.bf16_params = [p for p in params if p.requires_grad]
        self.fp32_params = [
            p.detach().float().clone().requires_grad_(True)
            for p in self.bf16_params
        ]
        self._id_to_fp32 = {
            id(bp): fp for bp, fp in zip(self.bf16_params, self.fp32_params)
        }
        self.optimizer = optimizer_cls(self.fp32_params, **kwargs)
        self._fp32_grads = [torch.zeros_like(fp) for fp in self.fp32_params]
        self._hooks = [
            bp.register_post_accumulate_grad_hook(self._make_hook(i))
            for i, bp in enumerate(self.bf16_params)
        ]

    def _make_hook(self, idx):
        def hook(param):
            if param.grad is not None:
                self._fp32_grads[idx].add_(param.grad.float())
                param.grad = None

        return hook

    def zero_grad(self):
        for g in self._fp32_grads:
            g.zero_()
        for bp in self.bf16_params:
            bp.grad = None

    def step(self, copy_params=True):
        for fp, g in zip(self.fp32_params, self._fp32_grads):
            fp.grad = g
        self.optimizer.step()
        if copy_params:
            self.sync_fp32_to_bf16()
        self.zero_grad()

    def sync_fp32_to_bf16(self):
        for bp, fp in zip(self.bf16_params, self.fp32_params):
            bp.data.copy_(fp.data)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ============================================================================
# Helpers
# ============================================================================
def forward_block(layer, frozen_layers, model_inputs):
    pred = layer(model_inputs["args"][0], **model_inputs["kwargs"])
    for fl in frozen_layers:
        pred = fl(pred, **model_inputs["kwargs"])
    return pred


def eval_block(layer, frozen_layers, inputs, targets, criterion, desc):
    pbar = tqdm(range(len(inputs)), desc=desc)
    total_se, total_t_se = 0.0, 0.0
    with torch.no_grad():
        for n, idx in enumerate(pbar, 1):
            mi = transfer_to_device(inputs[idx], layer.device)
            t = transfer_to_device(targets[idx], layer.device)
            pred = forward_block(layer, frozen_layers, mi)
            total_se += criterion(pred.float(), t.float()).item()
            total_t_se += t.float().pow(2).mean().item()
            pbar.set_postfix(
                rmse=f"{(total_se / n) ** 0.5:.6f}",
                rel_rmse=f"{(total_se / total_t_se) ** 0.5:.4f}",
            )


def set_lr(optimizer, lr):
    for pg in optimizer.optimizer.param_groups:
        pg["lr"] = lr


def clip_fp32_grads_(grads, max_norm):
    """Global gradient norm clip on fp32 grad accumulators (in-place)."""
    total = torch.norm(torch.stack([g.norm() for g in grads]))
    coef = (max_norm / (total + 1e-6)).clamp(max=1.0)
    for g in grads:
        g.mul_(coef)


def threshold_proximal_step(groups, groups_nnz, lambds, proxy):
    """kth_mid + soft_threshold + mask preservation on fp32 weights."""
    for g_nnz, g in zip(groups_nnz, groups):
        block = g.block
        gradient, lr, conditioner = proxy.get_info(block.view.param)
        conditioner = conditioner + conditioner.mean()*0.1
        data = block.view.param.data.clone()
        psi = gradient - conditioner * data
        vals = g.kth_mid({block: psi}, nnz=g_nnz, k_weight=k_val_weight)
        lambds[g].mul_(beta).add_((1 - beta) * vals)
        g.soft_threshold(lambds[g] * lr, conditioners={block: conditioner})
        m = g.get_masks(nnz=g_nnz)[block].float()
        block.view.param.data.copy_(data * m + (1 - m) * block.view.param.data)


def run_epoch(
    layer,
    frozen_layers,
    inputs,
    targets,
    optimizer,
    criterion,
    desc,
    on_step=None,
):
    """One pass over `inputs`. `on_step` (if given) runs after optimizer.step
    on the still-fp32 weights, then we sync to bf16."""
    indices = np.random.permutation(len(inputs))
    pbar = tqdm(indices, desc=desc)
    total_loss, n = 0.0, 0
    optimizer.zero_grad()
    for n_b, idx in enumerate(pbar, 1):
        mi = transfer_to_device(inputs[idx], layer.device)
        t = transfer_to_device(targets[idx], layer.device)
        pred = forward_block(layer, frozen_layers, mi)
        loss = criterion(pred.float(), t.float())
        loss.backward()
        total_loss += loss.item()
        n += 1
        if n_b % grad_accum_steps != 0:
            continue
        clip_fp32_grads_(optimizer._fp32_grads, clip_norm)
        optimizer.step(copy_params=on_step is None)
        if on_step is not None:
            on_step()
            optimizer.sync_fp32_to_bf16()
        optimizer.zero_grad()
        pbar.set_postfix(loss=f"{total_loss / n:.6e}")


def warmup_epoch(
    layer, frozen_layers, inputs, targets, optimizer, criterion, base_lr, desc
):
    set_lr(optimizer, base_lr * warmup_scale)
    run_epoch(layer, frozen_layers, inputs, targets, optimizer, criterion, desc)
    set_lr(optimizer, base_lr)


def build_groups(prune_layers, optimizer):
    """ScopeSpec(1,4) over fp32 copies of each linear projection's weight."""
    groups, groups_nnz, fp32_id_to_bf16 = [], [], {}
    for name, layer in prune_layers.items():
        bf16_w = layer.weight
        fp32_w = optimizer._id_to_fp32[id(bf16_w)]
        fp32_id_to_bf16[id(fp32_w)] = bf16_w
        blk = BlockSpec(fp32_w, shape=(1, 1), name=name)
        groups.append(ScopeSpec(blk, shape=(1, 4)))
        groups_nnz.append(2)
    return groups, groups_nnz, fp32_id_to_bf16


def extract_param_masks(groups, groups_nnz, fp32_id_to_bf16):
    """Get 2:4 masks keyed by the bf16 model parameter."""
    param_masks = {}
    for g_nnz, g in zip(groups_nnz, groups):
        for b, m in g.get_masks(nnz=g_nnz).items():
            param_masks[fp32_id_to_bf16[id(b.view.param)]] = m
    return param_masks


def linear_children(layer):
    yield from (
        m
        for m in (
            list(layer.self_attn.named_children())
            + list(layer.mlp.named_children())
        )
        if isinstance(m[1], nn.Linear)
    )


# ============================================================================
# Load model + calibration data
# ============================================================================
print(os.environ["HF_HOME"])
teacher = AutoModelForCausalLM.from_pretrained(
    model_name, dtype="auto", device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
student = teacher

c4_data = get_c4(
    num_samples=num_samples, seq_len=seq_length, tokenizer=tokenizer, seed=42
)
tokenized_inputs = [{"input_ids": d[0]} for d in c4_data]


# ============================================================================
# Capture inputs to layer 0
# ============================================================================
input_catcher = ModuleInputCatcher(device=torch.device("cpu"))
input_catcher.attach(student.model.layers[0], "decoder_0")
print("Capturing layer-0 inputs...")
with torch.no_grad():
    for batch in tqdm(tokenized_inputs):
        input_ids = batch["input_ids"].to(teacher.device)
        attention_mask = torch.ones_like(input_ids)
        _ = teacher(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
            use_cache=False,
        )
layer_inputs = input_catcher.inputs["decoder_0"]
input_catcher.detach("decoder_0")


# ============================================================================
# Prune layer 0
# ============================================================================
layer_idx = 0
student_layer = student.model.layers[layer_idx]
target_device = next(student_layer.parameters()).device
student_layer.device = target_device

block_end = min(layer_idx + 1 + lookahead, len(student.model.layers))
frozen_layers = list(student.model.layers[layer_idx + 1 : block_end])
print(f"Target layer {layer_idx}, lookahead {layer_idx+1}..{block_end-1}")

# Capture teacher targets BEFORE any pruning (full block forward)
teacher_targets = []
with torch.no_grad():
    for mi in tqdm(layer_inputs, desc="Capturing teacher targets"):
        mi = transfer_to_device(mi, target_device)
        hidden = mi["args"][0]
        for layer in [student_layer] + frozen_layers:
            hidden = layer(hidden, **mi["kwargs"])
        teacher_targets.append(hidden.cpu())

# Freeze everything, then unfreeze target layer's non-norm params
for p in student.model.parameters():
    p.requires_grad = False
for n, p in student_layer.named_parameters():
    p.requires_grad = "norm" not in n

original_weights = deepcopy(student_layer.state_dict())
criterion = nn.MSELoss()

prune_layers = {
    n: m
    for n, m in student_layer.named_modules()
    if isinstance(m, nn.Linear) and m.weight.requires_grad
}


# ----------------------------------------------------------------------------
# Threshold phase
# ----------------------------------------------------------------------------
optimizer = FP32Optimizer(
    AdamW,
    student_layer.parameters(),
    lr=threshold_lr,
    weight_decay=threshold_wd,
    betas=threshold_betas,
)
proxy = AdamProxy(optimizer.optimizer)
groups, groups_nnz, fp32_id_to_bf16 = build_groups(prune_layers, optimizer)
lambds = {g: torch.zeros_like(g.kth_largest(None, 1)) for g in groups}

eval_block(
    student_layer,
    frozen_layers,
    layer_inputs,
    teacher_targets,
    criterion,
    "Eval initial",
)
torch.cuda.empty_cache()

warmup_epoch(
    student_layer,
    frozen_layers,
    layer_inputs,
    teacher_targets,
    optimizer,
    criterion,
    threshold_lr,
    "Warmup (threshold)",
)

for epoch in range(threshold_epochs):
    run_epoch(
        student_layer,
        frozen_layers,
        layer_inputs,
        teacher_targets,
        optimizer,
        criterion,
        f"Threshold {epoch + 1}/{threshold_epochs}",
        on_step=lambda: threshold_proximal_step(
            groups, groups_nnz, lambds, proxy
        ),
    )

optimizer.remove_hooks()
del optimizer
torch.cuda.empty_cache()


# ----------------------------------------------------------------------------
# Apply mask, recovery phase
# ----------------------------------------------------------------------------
param_masks = extract_param_masks(groups, groups_nnz, fp32_id_to_bf16)
student_layer.load_state_dict(deepcopy(original_weights))
for p, m in param_masks.items():
    p.data.mul_(m)
for _, layer in linear_children(student_layer):
    prune.custom_from_mask(layer, "weight", param_masks[layer.weight])

optimizer = FP32Optimizer(
    AdamW,
    student_layer.parameters(),
    lr=recover_lr,
    weight_decay=recover_wd,
    betas=recover_betas,
)
warmup_epoch(
    student_layer,
    frozen_layers,
    layer_inputs,
    teacher_targets,
    optimizer,
    criterion,
    recover_lr,
    "Warmup (recovery)",
)

for epoch in range(recover_epochs):
    run_epoch(
        student_layer,
        frozen_layers,
        layer_inputs,
        teacher_targets,
        optimizer,
        criterion,
        f"Recover {epoch + 1}/{recover_epochs}",
    )

optimizer.remove_hooks()
del optimizer
torch.cuda.empty_cache()


# ----------------------------------------------------------------------------
# Commit prune, save, eval
# ----------------------------------------------------------------------------
for _, layer in linear_children(student_layer):
    prune.remove(layer, "weight")
for p, m in param_masks.items():
    p.data.mul_(m)

ckpt_path = (
    checkpoint_dir / f"{model_name}_decoder_{layer_idx}_{method_tag}.cpt"
)
ckpt_path.parent.mkdir(parents=True, exist_ok=True)
print(f"Saving layer {layer_idx} weights to {ckpt_path}")
torch.save(student_layer.state_dict(), ckpt_path)

eval_block(
    student_layer,
    frozen_layers,
    layer_inputs,
    teacher_targets,
    criterion,
    "Eval final",
)

for name, layer in linear_children(student_layer):
    p = layer.weight
    print(f"  {name}: density={(p.data.abs() > 0).float().mean().item():.4f}")
for n, p in student_layer.named_parameters():
    assert p.dtype == torch.bfloat16, f"{n} is {p.dtype}"
print("All layer weights confirmed bfloat16")
