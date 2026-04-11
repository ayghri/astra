"""ADMM-style 2:4 pruning of an LLM decoder layer with lookahead training.

For each weight W in the target layer we maintain three fp32 tensors:

  W  -- the trainable fp32 parameter (lives inside FP32Optimizer)
  Z  -- the sparse 2:4 projection (held externally)
  U  -- the dual variable, U += W - Z

Iteration (every `grad_accum_steps` backward calls):

  1. f(W) gradient comes from the lookahead loss (forward through
     student_layer + frozen_layers, MSE against teacher target).
  2. We add the ADMM proximal gradient  rho * (W - Z + U)  to W's grad.
  3. AdamW step on the augmented gradient updates W.
  4. Z update: V = W + U, then soft_threshold(V) -> 2:4 mask, with
     "kept" positions restored to V (mask preservation).
  5. U update: U += W - Z.

Final layer weight = Z * mask (the masked sparse solution).
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

from astra.hooks import ModuleInputCatcher, ModuleOutputCatcher
from astra.misc import transfer_to_device
from astra.proximals import AdamProxy
from sparsekit import BlockSpec, ScopeSpec
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

torch.backends.cuda.enable_flash_sdp(True)


# ============================================================================
# Config
# ============================================================================
base_dir = Path("/buckets")
checkpoint_dir = base_dir / "checkpoints"
checkpoint_dir.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(
    Path("~/scratch/buckets/datasets/huggingface").expanduser()
)

model_name = "Qwen/Qwen3-8B"
seq_length = 1024
num_samples = 1024
lookahead = 3
threshold_epochs = 20
recover_epochs = 10

threshold_lr = 2e-4
threshold_wd = 0.01
threshold_betas = (0.98, 0.99)

recover_lr = 5e-5
recover_wd = 1e-3
recover_betas = (0.95, 0.999)

beta = 0.98
grad_accum_steps = 8
warmup_scale = 1e-8
admm_rho = 0.01  # weight on the ADMM proximal term
admm_max_psi = 2e-4  # clamp for psi values (per autoresearch/admm.py)
k_val_weight = 1999.0

method_tag = (
    f"astra_admm_la{lookahead}_t{threshold_epochs}_r{recover_epochs}"
    f"_n{num_samples}_2of4"
)


# ============================================================================
# FP32Optimizer
# ============================================================================
class FP32Optimizer:
    """bf16 model + fp32 grad accumulation + fp32 optimizer state."""

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
        scale = 1.0 / self.accum_steps
        for fp, g in zip(self.fp32_params, self._fp32_grads):
            fp.grad = g * scale
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
def forward_block(student_layer, frozen_layers, model_inputs):
    pred = student_layer(model_inputs["args"][0], **model_inputs["kwargs"])
    for fl in frozen_layers:
        pred = fl(pred, **model_inputs["kwargs"])
    return pred


def eval_block(student_layer, frozen_layers, inputs, targets, criterion, desc):
    pbar = tqdm(range(len(inputs)), desc=desc)
    total_se = 0.0
    total_target_se = 0.0
    n_b = 0
    with torch.no_grad():
        for idx in pbar:
            mi = transfer_to_device(inputs[idx], student_layer.device)
            t = transfer_to_device(targets[idx], student_layer.device)
            pred = forward_block(student_layer, frozen_layers, mi)
            total_se += criterion(pred.float(), t.float()).item()
            total_target_se += t.float().pow(2).mean().item()
            n_b += 1
            pbar.set_postfix(
                rmse=f"{(total_se / n_b) ** 0.5:.6f}",
                rel_rmse=f"{(total_se / total_target_se) ** 0.5:.4f}",
            )


def warmup_optimizer(
    optimizer,
    base_lr,
    student_inputs,
    teacher_targets,
    student_layer,
    frozen_layers,
    criterion,
    desc="Warmup",
):
    """Run 1 epoch at lr * warmup_scale to populate Adam state."""
    for pg in optimizer.optimizer.param_groups:
        pg["lr"] = base_lr * warmup_scale
    indices = np.random.permutation(len(student_inputs))
    optimizer.zero_grad()
    pbar = tqdm(indices, desc=desc)
    for n_b, idx in enumerate(pbar, 1):
        mi = transfer_to_device(student_inputs[idx], student_layer.device)
        t = transfer_to_device(teacher_targets[idx], student_layer.device)
        pred = forward_block(student_layer, frozen_layers, mi)
        loss = criterion(pred.float(), t.float())
        loss.backward()
        if n_b % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        pbar.set_postfix(loss=f"{loss.item():.6f}")
    for pg in optimizer.optimizer.param_groups:
        pg["lr"] = base_lr


def init_admm_state(prune_layers, optimizer):
    """For each linear projection, build (W_fp32, Z, U, g_spec, lamb) and a
    Z->bf16 mapping for final write-back."""
    W_list, Z_list, U_list, g_specs, lambs = [], [], [], [], []
    z_id_to_bf16 = {}
    for name, layer in prune_layers.items():
        bf16_w = layer.weight
        W_fp32 = optimizer._id_to_fp32[id(bf16_w)]
        Z = W_fp32.data.clone()
        U = torch.zeros_like(W_fp32.data)
        b_spec = BlockSpec(Z, shape=(1, 1), name=name)
        g_spec = ScopeSpec(b_spec, shape=(1, 4), name=name)
        W_list.append(W_fp32)
        Z_list.append(Z)
        U_list.append(U)
        g_specs.append(g_spec)
        lambs.append(torch.zeros_like(g_spec.kth_largest(None, 1)))
        z_id_to_bf16[id(Z)] = bf16_w
    return W_list, Z_list, U_list, g_specs, lambs, z_id_to_bf16


def admm_zu_update(W_list, Z_list, U_list, g_specs, lambs, proxy):
    """One ADMM round: refresh Z and U for all blocks."""
    for w, z, u, g_spec, lamb in zip(W_list, Z_list, U_list, g_specs, lambs):
        V = w.data + u  # the point we want to project to 2:4

        # Adam preconditioner from the optimizer state of W
        _, _, conditioner = proxy.get_info(w)

        # psi: |V * cond|, clamped per autoresearch/admm.py
        psi = (V * conditioner).abs().clamp(max=admm_max_psi)
        phi = g_spec.kth_mid({g_spec.block: psi}, nnz=2, k_weight=k_val_weight)
        lamb.mul_(beta).add_((1 - beta) * phi)

        # Initialize Z = V, save a clone, soft-threshold in place
        z.copy_(V)
        v_clone = z.clone()
        g_spec.soft_threshold(lamb, conditioners={g_spec.block: conditioner})

        # Mask preservation: kept positions get original V (no shrinkage),
        # pruned positions stay at the soft-thresholded value (~0).
        m = g_spec.get_masks(nnz=2)[g_spec.block].to(z.dtype)
        z.copy_((1 - m) * z + m * v_clone)

        # Dual update
        u.add_(w.data - z)


def add_admm_proximal_grad(W_list, Z_list, U_list, optimizer, rho):
    """Add  rho * (W - Z + U)  to the fp32 grad accumulator (W subproblem)."""
    if rho == 0.0:
        return
    for w, z, u, fp32_grad in zip(
        W_list, Z_list, U_list, optimizer._fp32_grads
    ):
        fp32_grad.add_(rho * (w.data - z + u))


# ============================================================================
# Load model + calibration data
# ============================================================================
print(os.environ["HF_HOME"])
teacher = AutoModelForCausalLM.from_pretrained(
    model_name, dtype="auto", device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
student = teacher

ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
wikitext = " ".join(ds["text"])
input_text = [
    wikitext[i * seq_length : (i + 1) * seq_length] for i in range(num_samples)
]
tokenized_inputs = [
    tokenizer([t], return_tensors="pt", return_token_type_ids=False)
    for t in input_text
]

# --- Capture inputs to layer 0 via full model forward ---
input_catcher = ModuleInputCatcher(device=torch.device("cpu"))
input_catcher.attach(student.model.layers[0], "decoder_0")
print("Computing layer-0 inputs from full model forward...")
with torch.no_grad():
    for batch in tqdm(tokenized_inputs):
        _ = teacher(**batch.to(teacher.device), labels=None, use_cache=False)
student_inputs = input_catcher.inputs["decoder_0"]
input_catcher.detach("decoder_0")

# ============================================================================
# Per-layer pruning loop (currently only layer 0)
# ============================================================================
all_layers = student.model.layers[:1]
criterion = nn.MSELoss()

for layer_idx in range(len(all_layers)):
    print(f"\n========= Pruning layer {layer_idx} =========")

    teacher_layer = teacher.model.layers[layer_idx]
    target_device = next(teacher_layer.parameters()).device
    block_end = min(layer_idx + 1 + lookahead, len(teacher.model.layers))
    frozen_layers = list(teacher.model.layers[layer_idx + 1 : block_end])
    print(f"Lookahead through layers {layer_idx+1}..{block_end-1}")

    # --- Capture teacher targets (layer + lookahead) BEFORE any pruning ---
    teacher_targets = []
    with torch.no_grad():
        for mi in tqdm(student_inputs, desc="Capturing teacher targets"):
            mi = transfer_to_device(mi, target_device)
            hidden = mi["args"][0]
            kwargs = mi["kwargs"]
            for l in [teacher_layer] + frozen_layers:
                hidden = l(hidden, **kwargs)
            teacher_targets.append(hidden.cpu())

    student_layer = student.model.layers[layer_idx]

    # Only target layer's non-norm params are trainable
    for p in student.model.parameters():
        p.requires_grad = False
    for n, p in student_layer.named_parameters():
        p.requires_grad = "norm" not in n

    original_weights = deepcopy(student_layer.state_dict())

    # Linear projections to prune
    prune_layers = {
        n: m
        for n, m in student_layer.named_modules()
        if isinstance(m, nn.Linear) and m.weight.requires_grad
    }

    # --- Threshold optimizer (FP32) ---
    optimizer = FP32Optimizer(
        AdamW,
        student_layer.parameters(),
        accum_steps=1,
        lr=threshold_lr,
        weight_decay=threshold_wd,
        betas=threshold_betas,
    )
    proxy = AdamProxy(optimizer.optimizer)

    # --- ADMM state: W (in optimizer), Z, U, g_specs ---
    W_list, Z_list, U_list, g_specs, lambs, z_id_to_bf16 = init_admm_state(
        prune_layers, optimizer
    )
    student_layer.device = target_device

    # --- Eval initial loss ---
    eval_block(
        student_layer,
        frozen_layers,
        student_inputs,
        teacher_targets,
        criterion,
        "Eval initial loss",
    )
    torch.cuda.empty_cache()

    # --- Threshold warmup ---
    warmup_optimizer(
        optimizer,
        threshold_lr,
        student_inputs,
        teacher_targets,
        student_layer,
        frozen_layers,
        criterion,
        desc="Warmup (threshold)",
    )

    # --- ADMM threshold training ---
    for epoch in range(threshold_epochs):
        indices = np.random.permutation(len(student_inputs))
        pbar = tqdm(indices, desc=f"Epoch {epoch + 1}/{threshold_epochs}")
        total_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()

        for idx in pbar:
            n_batches += 1
            mi = transfer_to_device(student_inputs[idx], student_layer.device)
            t = transfer_to_device(teacher_targets[idx], student_layer.device)
            pred = forward_block(student_layer, frozen_layers, mi)
            loss = criterion(pred.float(), t.float())
            loss.backward()
            total_loss += loss.item()

            if n_batches % grad_accum_steps != 0:
                continue

            # 1. Add ADMM proximal grad to fp32 buffer (W subproblem)
            add_admm_proximal_grad(W_list, Z_list, U_list, optimizer, admm_rho)

            # 2. AdamW step on W (don't sync to bf16 yet)
            optimizer.step(copy_params=False)
            optimizer.zero_grad()

            # 3. ADMM Z and U update on fp32 buffers
            admm_zu_update(W_list, Z_list, U_list, g_specs, lambs, proxy)

            # 4. Sync the (still-dense) W back to bf16 for the next forward
            optimizer.sync_fp32_to_bf16()

            pbar.set_postfix(
                loss=f"{total_loss / n_batches:.6e}",
                density=np.mean(
                    [g.nnz(eps=1e-4) / g.block.numblk() for g in g_specs]
                ),
                u_norm=f"{sum(u.norm().item() for u in U_list):.3e}",
            )

    optimizer.remove_hooks()
    del optimizer
    torch.cuda.empty_cache()

    # --- Set bf16 weights to masked Z; collect masks for recovery ---
    param_masks = {}
    for w, z, g_spec in zip(W_list, Z_list, g_specs):
        bf16_w = z_id_to_bf16[id(z)]
        m = g_spec.get_masks(nnz=2)[g_spec.block]
        m_f = m.to(z.dtype)
        bf16_w.data.copy_((z * m_f).to(bf16_w.dtype))
        param_masks[bf16_w] = m_f

    for p, m in param_masks.items():
        print(p.shape, m.sum() / m.numel())

    # Free ADMM tensors
    del W_list, Z_list, U_list, g_specs, lambs
    torch.cuda.empty_cache()

    # --- Recovery setup: mask the linear layers, build new optimizer ---
    for name, layer in student_layer.named_modules():
        if isinstance(layer, nn.Linear) and layer.weight in param_masks:
            prune.custom_from_mask(layer, "weight", param_masks[layer.weight])

    optimizer = FP32Optimizer(
        AdamW,
        student_layer.parameters(),
        accum_steps=1,
        lr=recover_lr,
        weight_decay=recover_wd,
        betas=recover_betas,
    )

    warmup_optimizer(
        optimizer,
        recover_lr,
        student_inputs,
        teacher_targets,
        student_layer,
        frozen_layers,
        criterion,
        desc="Warmup (recovery)",
    )

    for epoch in range(recover_epochs):
        indices = np.random.permutation(len(student_inputs))
        pbar = tqdm(indices, desc=f"Recover {epoch + 1}/{recover_epochs}")
        total_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()
        for idx in pbar:
            n_batches += 1
            mi = transfer_to_device(student_inputs[idx], student_layer.device)
            t = transfer_to_device(teacher_targets[idx], student_layer.device)
            pred = forward_block(student_layer, frozen_layers, mi)
            loss = criterion(pred.float(), t.float())
            loss.backward()
            total_loss += loss.item()
            if n_batches % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            pbar.set_postfix(loss=f"{total_loss / n_batches:.6e}")

    optimizer.remove_hooks()
    del optimizer
    torch.cuda.empty_cache()

    # --- Commit pruning, save, eval ---
    for name, layer in student_layer.named_modules():
        if isinstance(layer, nn.Linear) and hasattr(layer, "weight_orig"):
            prune.remove(layer, "weight")
    for p, m in param_masks.items():
        p.data.mul_(m.to(p.dtype))

    ckpt_path = (
        checkpoint_dir / f"{model_name}_decoder_{layer_idx}_{method_tag}.cpt"
    )
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving layer {layer_idx} weights to {ckpt_path}")
    torch.save(student_layer.state_dict(), ckpt_path)

    eval_block(
        student_layer,
        frozen_layers,
        student_inputs,
        teacher_targets,
        criterion,
        "Eval final loss",
    )

    for n, p in student_layer.named_parameters():
        if "_proj.weight" in n:
            print(
                f"  {n}: density={(p.data.abs() > 0).float().mean().item():.4f}"
            )
        assert p.dtype == torch.bfloat16, f"{n} is {p.dtype}"
    print("All layer weights confirmed bfloat16")
