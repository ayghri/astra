import torch
import os
from pathlib import Path

# from torch.nn.utils import clip_grad_norm_

from astra.hooks import ModuleInputCatcher, ModuleOutputCatcher
from copy import deepcopy
from tqdm import tqdm
from torch.optim import Adam, AdamW
from torch import nn
import numpy as np
import torch.nn.utils.prune as prune
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from datasets import load_dataset

# from spastra.blocks import BlockCoupling
from astra.misc import transfer_to_device
from astra.proximals import AdamProxy

# from astra.evaluate import evaluate_ppl_hf

from sparsekit import BlockSpec
from sparsekit import ScopeSpec

torch.backends.cuda.enable_flash_sdp(True)

base_dir = Path("/buckets")
checkpoint_dir = base_dir / "checkpoints"
checkpoint_dir.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(
    Path("~/scratch/buckets/datasets/huggingface").expanduser()
)


print(os.environ["HF_HOME"])


seq_length = 2048
num_samples = 1024*5

lookahead = 3  # propagate through this many frozen layers after target layer
threshold_epochs = 20

recover_epochs = 10
model_name = "Qwen/Qwen3-8B"

threshold_lr = 2e-4
threshold_wd = 0.01
threshold_betas = (0.95, 0.99)

beta = 0.98
grad_accum_steps = 8

# Encodes pruning method + settings into checkpoint filename
method_tag = (
    f"astra_fp32_la{lookahead}_kthmid"
    f"_t{threshold_epochs}_r{recover_epochs}"
    f"_n{num_samples}_2of4"
)


class FP32Optimizer:
    """Optimizer wrapper: bf16 model, fp32 grad accumulation/states/step.

    Each backward() fires a hook that upcasts the bf16 grad to fp32 and
    adds it to an internal accumulator, then clears the bf16 grad so the
    next backward starts fresh -- no lossy bf16-on-bf16 accumulation.
    """

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

        # fp32 gradient accumulators + hooks
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
        """Return the fp32 copy of a bf16 model parameter."""
        return self._id_to_fp32[id(param)]

    def zero_grad(self):
        for g in self._fp32_grads:
            g.zero_()
        for bp in self.bf16_params:
            bp.grad = None

    def step(self, copy_params=True):
        """Copy accumulated fp32 grads (averaged) to fp32 params, step, optionally sync."""
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

    def sync_bf16_to_fp32(self):
        """Call after external in-place edits to bf16 params (e.g. proximal step)."""
        for bp, fp in zip(self.bf16_params, self.fp32_params):
            fp.data.copy_(bp.data)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


teacher = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

student = teacher

print("embed_weights", teacher.model.embed_tokens.weight.numel() / 1000**2)
print(
    "attn weights",
    sum(
        p.numel()
        for n, p in teacher.model.named_parameters()
        if "self_attn" in n
    )
    / 1000**2,
)
print(
    "mlp weights",
    sum(p.numel() for n, p in teacher.model.named_parameters() if "mlp" in n)
    / 1000**2,
)
print("lm head weights", teacher.lm_head.weight.numel() / 1000**2)


ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

wikitext = " ".join(ds["text"])

len(wikitext)

input_text = []

for i in range(num_samples):
    input_text.append(wikitext[i * seq_length : (i + 1) * seq_length])

tokenized_inputs = [
    tokenizer([text], return_tensors="pt", return_token_type_ids=False)
    for text in input_text
]


input_catcher = ModuleInputCatcher(device=torch.device("cpu"))
output_catcher = ModuleOutputCatcher(device=torch.device("cpu"))

layer_idx = 0
layer_name = f"decoder_{layer_idx}"

first_layer = teacher.model.layers[layer_idx]
input_catcher.attach(first_layer, layer_name)

print("Computing teacher inputs")
with torch.no_grad():
    for model_inputs in tqdm(tokenized_inputs):
        _ = teacher(
            **model_inputs.to(teacher.device), labels=None, use_cache=False
        )

teacher_inputs = input_catcher.inputs[layer_name]
student_inputs = deepcopy(teacher_inputs)
input_catcher.detach(layer_name)


# all_layers = student.model.layers
all_layers = student.model.layers[:1]
prev_layers = list(range(0))

for layer_idx in range(len(prev_layers)):
    layer_name = f"decoder_{layer_idx}"
    print(prev_layers, layer_idx, layer_name)

    teacher_layer = teacher.model.layers[layer_idx]
    teacher_layer.device = list(teacher_layer.parameters())[0].device

    output_catcher.attach(teacher_layer, layer_name)
    for model_inputs in tqdm(teacher_inputs):
        model_inputs = transfer_to_device(model_inputs, teacher_layer.device)
        _ = teacher_layer(model_inputs["args"][0], **model_inputs["kwargs"])

    teacher_targets = output_catcher.outputs[layer_name]
    output_catcher.detach(layer_name)

    student_layer = student.model.layers[layer_idx]
    layer_ckpt_path = (
        checkpoint_dir / f"{model_name}_decoder_{layer_idx}_{method_tag}.cpt"
    )
    print("Loading:", layer_ckpt_path)
    student_layer.load_state_dict(torch.load(layer_ckpt_path))

    torch.cuda.empty_cache()

    for t_input, t_target in zip(teacher_inputs, teacher_targets):
        t_input["args"] = (t_target,)

    output_catcher.attach(student_layer, layer_name)
    for model_inputs in tqdm(student_inputs):
        model_inputs = transfer_to_device(model_inputs, student_layer.device)
        _ = student_layer(model_inputs["args"][0], **model_inputs["kwargs"])

    for s_input, s_target in zip(
        student_inputs, output_catcher.outputs[layer_name]
    ):
        s_input["args"] = (s_target,)

    output_catcher.detach(layer_name)

    torch.cuda.empty_cache()

for layer_idx in range(len(prev_layers), len(all_layers)):
    layer_name = f"decoder_{layer_idx}"
    print(prev_layers, layer_idx, layer_name)

    teacher_layer = teacher.model.layers[layer_idx]
    teacher_layer.device = list(teacher_layer.parameters())[0].device
    block_end = min(layer_idx + 1 + lookahead, len(teacher.model.layers))
    frozen_layers = list(teacher.model.layers[layer_idx + 1 : block_end])
    print(
        f"Target: layer {layer_idx}, lookahead through layers {layer_idx+1}..{block_end-1}"
    )

    teacher_targets = []
    with torch.no_grad():
        for model_inputs in tqdm(
            teacher_inputs, desc="Capturing teacher targets"
        ):
            model_inputs = transfer_to_device(
                model_inputs, teacher_layer.device
            )
            hidden = model_inputs["args"][0]
            kwargs = model_inputs["kwargs"]
            for l in [teacher_layer] + frozen_layers:
                hidden = l(hidden, **kwargs)
            teacher_targets.append(hidden.to(torch.device("cpu")))

    student_layer = student.model.layers[layer_idx]

    for p in student.model.parameters():
        p.requires_grad = False

    for n, p in student_layer.named_parameters():
        p.requires_grad = True
        if "norm" in n:
            p.requires_grad = False

    original_weights = deepcopy(student_layer.state_dict())

    criterion = nn.MSELoss()

    # Collect linear projections to prune (q/k/v/o + gate/up/down)
    prune_layers = {}
    for n, l in student_layer.named_modules():
        if isinstance(l, nn.Linear) and l.weight.requires_grad:
            prune_layers[n] = l

    # --- Build threshold optimizer FIRST so we can wrap groups around fp32 params ---
    optimizer = FP32Optimizer(
        AdamW,
        student_layer.parameters(),
        accum_steps=1,
        lr=threshold_lr,
        weight_decay=threshold_wd,
        betas=threshold_betas,
    )
    proxy = AdamProxy(optimizer.optimizer)

    # Build groups around the FP32 copies of each projection's weight.
    # The proximal step will then read/write fp32 weights directly.
    groups = []
    groups_nnz = []
    fp32_id_to_bf16 = {}  # for mapping mask keys back to bf16 model params
    for name, layer in prune_layers.items():
        bf16_w = layer.weight
        fp32_w = optimizer._id_to_fp32[id(bf16_w)]
        fp32_id_to_bf16[id(fp32_w)] = bf16_w
        blk = BlockSpec(fp32_w, shape=(1, 1), name=name)
        groups.append(ScopeSpec(blk, shape=(1, 4)))
        groups_nnz.append(2)  # 2:4 sparsity

    assert len(groups_nnz) == len(groups)

    lambds = {g: torch.zeros_like(g.kth_largest(None, 1)) for g in groups}

    student_layer.device = list(student_layer.parameters())[0].device

    pbar = tqdm(range(len(student_inputs)), desc="Eval initial loss")
    total_se = 0.0
    total_target_se = 0.0
    num_batches = 0
    with torch.no_grad():
        for idx in pbar:
            model_inputs = transfer_to_device(
                student_inputs[idx], student_layer.device
            )
            target = transfer_to_device(
                teacher_targets[idx], student_layer.device
            )
            num_batches += 1
            pred = student_layer(
                model_inputs["args"][0], **model_inputs["kwargs"]
            )
            for fl in frozen_layers:
                pred = fl(pred, **model_inputs["kwargs"])
            total_se += criterion(pred.float(), target.float()).item()
            total_target_se += target.float().pow(2).mean().item()
            pbar.set_postfix(
                rmse=f"{(total_se / num_batches) ** 0.5:.6f}",
                rel_rmse=f"{(total_se / total_target_se) ** 0.5:.4f}",
                density=np.mean([g.nnz() / g.block.numblk() for g in groups]),
            )

    torch.cuda.empty_cache()

    # Warm-up: 1 epoch at reduced lr to populate Adam state
    warmup_scale = 1e-8
    for pg in optimizer.optimizer.param_groups:
        pg["lr"] = threshold_lr * warmup_scale
    indices = np.random.permutation(len(student_inputs))
    pbar = tqdm(indices, desc="Warmup (threshold)")
    optimizer.zero_grad()
    for n_b, idx in enumerate(pbar, 1):
        model_inputs = transfer_to_device(
            student_inputs[idx], student_layer.device
        )
        target = transfer_to_device(teacher_targets[idx], student_layer.device)
        pred = student_layer(model_inputs["args"][0], **model_inputs["kwargs"])
        for fl in frozen_layers:
            pred = fl(pred, **model_inputs["kwargs"])
        loss = criterion(pred.float(), target.float())
        loss.backward()
        if n_b % grad_accum_steps == 0:
            optimizer.step(copy_params=True)
            optimizer.zero_grad()
        pbar.set_postfix(loss=f"{loss.item():.6f}")
    for pg in optimizer.optimizer.param_groups:
        pg["lr"] = threshold_lr

    # Threshold training (notebook-style: kth_mid + soft_threshold on fp32 weights)
    for epoch in range(threshold_epochs):
        indices = np.random.permutation(len(student_inputs))
        pbar = tqdm(indices, desc=f"Epoch {epoch + 1}/{threshold_epochs}")
        total_loss = 0.0
        num_batches = 0
        optimizer.zero_grad()

        for idx in pbar:
            num_batches += 1
            model_inputs = transfer_to_device(
                student_inputs[idx], student_layer.device
            )
            target = transfer_to_device(
                teacher_targets[idx], student_layer.device
            )
            pred = student_layer(
                model_inputs["args"][0], **model_inputs["kwargs"]
            )
            for fl in frozen_layers:
                pred = fl(pred, **model_inputs["kwargs"])
            loss = criterion(pred.float(), target.float())
            loss.backward()
            total_loss += loss.item()

            if num_batches % grad_accum_steps != 0:
                continue

            # Global grad-norm clip on the fp32 accumulators (in-place)
            grad_total_norm = torch.norm(
                torch.stack([g.norm() for g in optimizer._fp32_grads])
            )
            clip_coef = (1.0 / (grad_total_norm + 1e-6)).clamp(max=1.0)
            for g in optimizer._fp32_grads:
                g.mul_(clip_coef)

            optimizer.step(copy_params=False)
            optimizer.zero_grad()

            for g_nnz, g in zip(groups_nnz, groups):
                block = g.block
                gradient, lr, conditioner = proxy.get_info(block.view.param)
                data = block.view.param.data.clone()
                psi = gradient - conditioner * data
                vals = g.kth_mid({block: psi}, nnz=g_nnz, k_weight=1999.0)
                lambds[g].mul_(beta).add_((1 - beta) * vals)
                g.soft_threshold(
                    lambds[g] * lr, conditioners={block: conditioner}
                )
                m = g.get_masks(nnz=g_nnz)[block].float()
                block.view.param.data.copy_(
                    data * m + (1 - m) * block.view.param.data
                )

            optimizer.sync_fp32_to_bf16()

            pbar.set_postfix(
                loss=f"{total_loss / num_batches:.6e}",
                density=np.mean(
                    [g.nnz(eps=1e-4) / g.block.numblk() for g in groups]
                ),
                lambds=sum(lambds[g].mean().item() for g in groups)
                / len(groups),
            )

    optimizer.remove_hooks()
    del optimizer
    torch.cuda.empty_cache()

    for n, p in student_layer.named_parameters():
        if "proj" in n:
            print(n, ((p.data.abs() > 1e-12).sum() / p.numel()).item())

    # Extract masks via fp32 -> bf16 mapping (groups were built on fp32 params)
    param_masks = {}
    for g_nnz, g in zip(groups_nnz, groups):
        for b, m in g.get_masks(nnz=g_nnz).items():
            bf16_p = fp32_id_to_bf16[id(b.view.param)]
            param_masks[bf16_p] = m

    for p, m in param_masks.items():
        print(p.shape, m.sum() / m.numel())

    student_layer.load_state_dict(deepcopy(original_weights))

    for p, m in param_masks.items():
        p.data.mul_(m)

    for layer_name, layer in student_layer.self_attn.named_children():
        if isinstance(layer, nn.Linear):
            print(layer_name)
            mask = param_masks[layer.weight]
            prune.custom_from_mask(layer, "weight", mask)

    for layer_name, layer in student_layer.mlp.named_children():
        if isinstance(layer, nn.Linear):
            print(layer_name)
            mask = param_masks[layer.weight]
            prune.custom_from_mask(layer, "weight", mask)

    recover_lr = 5e-5
    optimizer = FP32Optimizer(
        AdamW,
        student_layer.parameters(),
        accum_steps=1,
        lr=recover_lr,
        weight_decay=1e-3,
        betas=(0.95, 0.999),
    )

    # Recovery warmup
    for pg in optimizer.optimizer.param_groups:
        pg["lr"] = recover_lr * warmup_scale
    indices = np.random.permutation(len(student_inputs))
    pbar = tqdm(indices, desc="Warmup (recovery)")
    optimizer.zero_grad()
    for n_b, idx in enumerate(pbar, 1):
        model_inputs = transfer_to_device(
            student_inputs[idx], student_layer.device
        )
        target = transfer_to_device(teacher_targets[idx], student_layer.device)
        pred = student_layer(model_inputs["args"][0], **model_inputs["kwargs"])
        for fl in frozen_layers:
            pred = fl(pred, **model_inputs["kwargs"])
        loss = criterion(pred.float(), target.float())
        loss.backward()
        if n_b % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        pbar.set_postfix(loss=f"{loss.item():.6f}")
    for pg in optimizer.optimizer.param_groups:
        pg["lr"] = recover_lr

    for epoch in range(recover_epochs):
        indices = np.random.permutation(len(student_inputs))
        pbar = tqdm(indices, desc=f"Epoch {epoch + 1}/{recover_epochs}")
        total_loss = 0.0
        num_batches = 0
        optimizer.zero_grad()
        for idx in pbar:
            num_batches += 1
            model_inputs = transfer_to_device(
                student_inputs[idx], student_layer.device
            )
            target = transfer_to_device(
                teacher_targets[idx], student_layer.device
            )
            pred = student_layer(
                model_inputs["args"][0], **model_inputs["kwargs"]
            )
            for fl in frozen_layers:
                pred = fl(pred, **model_inputs["kwargs"])
            loss = criterion(pred.float(), target.float())
            loss.backward()
            total_loss += loss.item()
            if num_batches % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            pbar.set_postfix(loss=f"{total_loss / num_batches:.6e}")

    optimizer.remove_hooks()

    for layer_name, layer in prune_layers.items():
        print(layer_name)
        prune.remove(layer, "weight")

    for p, m in param_masks.items():
        p.data.mul_(m)

    layer_ckpt_path = (
        checkpoint_dir / f"{model_name}_decoder_{layer_idx}_{method_tag}.cpt"
    )
    layer_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving layer {layer_idx} weights to {layer_ckpt_path}")

    torch.save(student_layer.state_dict(), layer_ckpt_path)

    pbar = tqdm(range(len(student_inputs)), desc="Eval final loss")
    total_se = 0.0
    total_target_se = 0.0
    num_batches = 0
    with torch.no_grad():
        for idx in pbar:
            model_inputs = transfer_to_device(
                student_inputs[idx], student_layer.device
            )
            target = transfer_to_device(
                teacher_targets[idx], student_layer.device
            )
            num_batches += 1
            pred = student_layer(
                model_inputs["args"][0], **model_inputs["kwargs"]
            )
            for fl in frozen_layers:
                pred = fl(pred, **model_inputs["kwargs"])
            total_se += criterion(pred.float(), target.float()).item()
            total_target_se += target.float().pow(2).mean().item()
            pbar.set_postfix(
                rmse=f"{(total_se / num_batches) ** 0.5:.6f}",
                rel_rmse=f"{(total_se / total_target_se) ** 0.5:.4f}",
                density=np.mean([g.nnz() / g.numblk() for g in groups]),
            )

    for layer_name, layer in list(
        student_layer.self_attn.named_children()
    ) + list(student_layer.mlp.named_children()):
        if isinstance(layer, nn.Linear):
            p = layer.weight
            print(layer_name, ((p.data.abs() > 1e-12).sum() / p.numel()).item())

    del optimizer
    torch.cuda.empty_cache()

    # Verify weights are still bf16
    for n, p in student_layer.named_parameters():
        assert p.dtype == torch.bfloat16, f"{n} is {p.dtype}, expected bfloat16"
    print("All layer weights confirmed bfloat16")

    for t_input, t_target in zip(teacher_inputs, teacher_targets):
        t_input["args"] = (t_target,)

    output_catcher.attach(student_layer, layer_name)

    with torch.no_grad():
        for model_inputs in tqdm(student_inputs):
            model_inputs = transfer_to_device(
                model_inputs, student_layer.device
            )
            _ = student_layer(model_inputs["args"][0], **model_inputs["kwargs"])

    for s_input, s_target in zip(
        student_inputs, output_catcher.outputs[layer_name]
    ):
        s_input["args"] = (s_target,)

    output_catcher.detach(layer_name)

    torch.cuda.empty_cache()
    prev_layers.append(layer_idx)
