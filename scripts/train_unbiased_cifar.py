#!/usr/bin/env python
"""CIFAR training with Adam + unbiased soft-thresholding.

Uses the ADMM-style mask-unthresholding trick: soft-threshold decides the
sparsity pattern, but kept weights are restored to their pre-threshold values,
eliminating the L1 shrinkage bias on surviving entries.

Three phases:
  1. Warmup   [0, T_w):   dense Adam, linear LR warmup
  2. Sparsify [T_w, T_f): Adam step + unbiased soft-threshold
  3. Frozen   [T_f, T]:   hard-threshold once, Adam on frozen support

Usage:
  python scripts/train_unbiased_cifar.py --config-name cifar10_unbiased
  python scripts/train_unbiased_cifar.py --config-name cifar10_unbiased sparsity=0.95
"""

import json
import logging
import os
from typing import Dict, List, Tuple

import hydra
import torch
import torch.nn as nn
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from sparsekit import BlockSpec, ScopeSpec, View

from astra.controllers import EMAController, LambdaController
from astra.data.datasets import get_dataloaders
from astra.models.wideresnet import WideResNet
from astra.proximals import OptimizerProxy
from astra.train.utils import evaluate_accuracy, save_checkpoint

log = logging.getLogger(__name__)


# ── Layer selection ───────────────────────────────────────────────────────────

def prunable_params(model: nn.Module) -> List[nn.Parameter]:
    """Conv2d / Linear weights — skip biases, BN, first layer, classifier."""
    candidates = [
        (n, p) for n, p in model.named_parameters()
        if p.requires_grad and p.ndim >= 2
        and not n.endswith("bias")
        and "bn" not in n and "norm" not in n
    ]
    if len(candidates) <= 2:
        return [p for _, p in candidates]
    return [p for _, p in candidates[1:-1]]


# ── Coupling builders (reused from train_cifar.py) ───────────────────────────

def _flat_view(p: nn.Parameter) -> View:
    assert p.is_contiguous()
    return View(p, shape=(p.numel(),), stride=(1,))


def _2d_view(p: nn.Parameter) -> Tuple[View, int, int]:
    C_out = p.shape[0]
    filter_size = p.numel() // C_out
    assert p.is_contiguous()
    view = View(p, shape=(C_out, filter_size), stride=(filter_size, 1))
    return view, C_out, filter_size


def erk_sparsities(params, target_sparsity, power_scale=1.0):
    """Per-layer sparsity via Erdős-Rényi Kernel. Layers needing density > 1 stay dense."""
    import numpy as np
    dense_set = set()
    is_valid = False
    while not is_valid:
        raw_probs = {}
        divisor = 0.0
        rhs = 0.0
        for i, p in enumerate(params):
            n = p.numel()
            n_ones = int(n * (1.0 - target_sparsity))
            if i in dense_set:
                rhs -= int(n * target_sparsity)
            else:
                rhs += n_ones
                raw = (sum(p.shape) / np.prod(p.shape)) ** power_scale
                raw_probs[i] = raw
                divisor += raw * n
        eps = rhs / divisor
        max_prob = max(raw_probs.values())
        if max_prob * eps > 1:
            for i, rp in raw_probs.items():
                if rp == max_prob:
                    dense_set.add(i)
                    break
        else:
            is_valid = True
    return [0.0 if i in dense_set else 1.0 - eps * raw_probs[i] for i, _ in enumerate(params)]


def _make_scope(p, sparsity_type):
    """Create a ScopeSpec for a parameter given the sparsity type."""
    if sparsity_type == "unstructured":
        view = _flat_view(p)
        block = BlockSpec(view, shape=(1,))
        return ScopeSpec(block, shape=(-1,))  # one scope, all elements compete
    elif sparsity_type == "fanin":
        view, C_out, filter_size = _2d_view(p)
        block = BlockSpec(view, shape=(1, 1))
        return ScopeSpec(block, shape=(1, filter_size))  # per-neuron scope
    elif sparsity_type == "channel":
        view, C_out, filter_size = _2d_view(p)
        block = BlockSpec(view, shape=(1, filter_size))
        return ScopeSpec(block, shape=(-1, 1))  # all filters compete
    else:
        raise ValueError(f"Unknown sparsity_type: {sparsity_type!r}")


def build_coupling(params, sparsity, sparsity_type, sparsity_dist="uniform"):
    if sparsity_dist == "erk":
        layer_sparsities = erk_sparsities(params, sparsity)
    else:
        layer_sparsities = [sparsity] * len(params)

    groups, kappas, dense_params = [], [], []
    for p, s in zip(params, layer_sparsities):
        if s <= 0:
            dense_params.append(p)
            continue
        scope = _make_scope(p, sparsity_type)
        kappa = scope.sparsity_to_nnz(s)
        groups.append(scope)
        kappas.append(kappa)
    return groups, kappas, layer_sparsities


def count_sparse_flops(model, input_size=(1, 3, 32, 32)):
    dense_macs = 0
    sparse_macs = 0
    hooks = []

    def _hook(mod, inp, out):
        nonlocal dense_macs, sparse_macs
        if isinstance(mod, nn.Conv2d):
            C_out, C_in_per_group, kH, kW = mod.weight.shape
            H_out, W_out = out.shape[-2:]
            macs = int(C_out * (C_in_per_group * kH * kW) * H_out * W_out)
        elif isinstance(mod, nn.Linear):
            batch = inp[0].shape[0] if inp[0].ndim > 1 else 1
            macs = int(mod.in_features * mod.out_features * batch)
        else:
            return
        density = (mod.weight.data.abs() > 0).float().mean().item()
        dense_macs += macs
        sparse_macs += int(macs * density)

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            hooks.append(m.register_forward_hook(_hook))

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        model(torch.randn(input_size, device=device))
    model.train(was_training)

    for h in hooks:
        h.remove()

    return dense_macs, sparse_macs


# ── Unbiased soft-threshold step ─────────────────────────────────────────────

@torch.no_grad()
def unbiased_threshold_step(
    groups: List[ScopeSpec],
    kappas: List[int],
    optimizer: torch.optim.Optimizer,
    lambda_ctrl: LambdaController,
    ema_ctrl: EMAController,
    eps: float = 1e-7,
):
    """Soft-threshold for pattern selection, then restore kept weights (no bias).

    1. Update EMA of optimizer momentum
    2. Compute scores = EMA_grad (no alpha term — Adam already handles weight decay)
    3. Update lambda via kth_largest score
    4. Clone weights
    5. Soft-threshold (shrinks everything)
    6. Get masks (top-k selection)
    7. Restore kept entries from clone (undo shrinkage on survivors)
    """
    proxy = OptimizerProxy.get_proxy(optimizer)

    # Gather EMA updates
    spec_to_info = {}
    for group_cfg in optimizer.param_groups:
        for p in group_cfg["params"]:
            for group in groups:
                for sp in group.specs():
                    if sp.view.param is p:
                        direction, step_size, conditioner = proxy.get_info(p)
                        direction = direction.reshape(sp.data.shape)
                        ema_ctrl.update_single(sp, direction)
                        spec_to_info[sp] = ema_ctrl.get(sp)

    # Per-group: compute lambda, soft-threshold, then mask-unthreshold
    for group, kappa in zip(groups, kappas):
        # Score = EMA of gradient (no alpha*W term — Adam handles decay)
        grad_bar = {sp: spec_to_info[sp] for sp in group.specs() if sp in spec_to_info}

        psi = group.kth_largest(grad_bar, nnz=kappa)
        lambda_ctrl.update_single(group, psi)
        current_lambda = lambda_ctrl.get(group).add(eps)

        # Clone before thresholding
        clones = {sp: sp.data.clone() for sp in group.specs()}

        # Soft-threshold (shrinks all entries)
        group.soft_threshold(current_lambda)

        # Get mask (which entries survived)
        masks = group.get_masks(nnz=kappa)

        # Restore kept entries from clone (undo shrinkage bias)
        for sp in group.specs():
            m = masks[sp].to(sp.data)
            sp.data.copy_((1 - m) * sp.data + m * clones[sp])


# ── Frozen support ────────────────────────────────────────────────────────────

def freeze_support(groups, kappas, optimizer):
    for group, kappa in zip(groups, kappas):
        group.hard_threshold(nnz=kappa)

    hooks = []
    for group in groups:
        for scope in group.specs():
            p = scope.view.param
            mask = p.data.abs() > 0
            def _mask_grad(param, m=mask):
                if param.grad is not None:
                    param.grad.mul_(m)
            h = p.register_post_accumulate_grad_hook(_mask_grad
            )
            hooks.append(h)

    # Zero optimizer state for pruned positions
    for g in optimizer.param_groups:
        for p in g["params"]:
            state = optimizer.state.get(p, {})
            mask = (p.data.abs() > 0).float()
            for key in ["momentum_buffer", "exp_avg", "exp_avg_sq"]:
                if key in state and state[key] is not None:
                    state[key].mul_(mask)

    return hooks


# ── Main ──────────────────────────────────────────────────────────────────────

@hydra.main(config_path="../configs", config_name="cifar10_unbiased", version_base="1.3")
def main(cfg: DictConfig) -> None:
    log.info("\n%s", OmegaConf.to_yaml(cfg))
    torch.manual_seed(cfg.training.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.get("entity", None),
        mode=cfg.wandb.get("mode", "online"),
        group=cfg.wandb.get("group", None),
        config=OmegaConf.to_container(cfg, resolve=True),
        name=(
            f"unbiased_{cfg.sparsity_type}_{cfg.dataset.name}"
            f"_s{cfg.sparsity}_seed{cfg.training.seed}"
        ),
    )

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader, val_loader = get_dataloaders(
        cfg.dataset.name,
        cfg.dataset.data_dir,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = WideResNet(
        depth=cfg.model.depth,
        widen_factor=cfg.model.widen_factor,
        num_classes=cfg.dataset.num_classes,
        drop_rate=cfg.model.drop_rate,
    ).to(device)
    log.info("Params: %d", sum(p.numel() for p in model.parameters()))

    # ── Optimizer & LR schedule ───────────────────────────────────────────────
    optimizer = Adam(
        model.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.get("weight_decay", 0.0),
    )
    lr_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg.training.epochs,
        eta_min=cfg.optimizer.get("min_lr", 0.0),
    )
    criterion = nn.CrossEntropyLoss()

    # ── Sparsity ─────────────────────────────────────────────────────────────
    params_to_prune = prunable_params(model)
    groups, kappas, layer_sparsities = build_coupling(
        params_to_prune, cfg.sparsity, cfg.sparsity_type,
        sparsity_dist=cfg.get("sparsity_dist", "uniform"),
    )
    total_prunable = sum(p.numel() for p in params_to_prune)
    log.info(
        "Sparsity type: %s | dist: %s | target: %.3f | prunable params: %d | groups: %d",
        cfg.sparsity_type, cfg.get("sparsity_dist", "uniform"),
        cfg.sparsity, total_prunable, len(groups),
    )
    # Log per-layer sparsity/kappa (all layers, including dense)
    name_map = {id(p): n for n, p in model.named_parameters()}
    scope_idx = 0
    for p, s in zip(params_to_prune, layer_sparsities):
        layer_name = name_map.get(id(p), "?")
        if s <= 0:
            log.info("  %-40s shape=%-20s DENSE", layer_name, str(tuple(p.shape)))
        else:
            scope = groups[scope_idx]
            kappa = kappas[scope_idx]
            blocks_per_scope = scope.block_numel
            log.info("  %-40s shape=%-20s kappa=%d/%d  sparsity=%.4f",
                     layer_name, str(tuple(p.shape)), kappa, blocks_per_scope, s)
            scope_idx += 1

    ema_ctrl = EMAController(rho=cfg.astra.ema_rho)
    lambda_ctrl = LambdaController(
        device=device,
        beta=cfg.astra.beta,
        cap=cfg.astra.lambda_max,
    )

    T_w = cfg.training.warmup_epochs
    T_f = cfg.training.freeze_epoch
    T = cfg.training.epochs
    freeze_hooks: List = []
    best_acc = 0.0
    cumulative_train_macs = 0
    n_train_samples = len(train_loader.dataset)

    # JSON results log
    import time as _time, hashlib
    cfg_hash = hashlib.md5(OmegaConf.to_yaml(cfg).encode()).hexdigest()[:6]
    timestamp = _time.strftime("%Y%m%d_%H%M")
    exp_dir = os.path.join(
        cfg.get("output_dir", "autoresearch/results"),
        f"{timestamp}_astra_unbiased_{cfg.dataset.name}_{cfg.sparsity_type}_s{cfg.sparsity}_seed{cfg.training.seed}_{cfg_hash}",
    )
    os.makedirs(exp_dir, exist_ok=True)
    json_path = os.path.join(exp_dir, "results.json")
    log.info("Experiment: %s", exp_dir)

    json_results = {
        "method": "astra_unbiased",
        "dataset": cfg.dataset.name,
        "sparsity": cfg.sparsity,
        "sparsity_type": cfg.sparsity_type,
        "sparsity_dist": cfg.get("sparsity_dist", "uniform"),
        "seed": cfg.training.seed,
        "epochs": T,
        "experiment_dir": exp_dir,
        "status": "running",
        "epoch_log": [],
    }

    def save_json():
        with open(json_path, "w") as f:
            json.dump(json_results, f, indent=2)

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(T):

        if epoch == T_f and not freeze_hooks:
            log.info("Epoch %d: entering frozen-support phase", epoch)
            freeze_hooks = freeze_support(groups, kappas, optimizer)

        model.train()
        total_loss, n_correct, n_total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Unbiased threshold: soft-threshold for pattern, restore kept weights
            if T_w <= epoch < T_f:
                unbiased_threshold_step(
                    groups, kappas, optimizer, lambda_ctrl, ema_ctrl,
                )

            total_loss += loss.item() * labels.size(0)
            n_correct += (outputs.argmax(1) == labels).sum().item()
            n_total += labels.size(0)

        lr_scheduler.step()

        train_acc = 100.0 * n_correct / n_total
        train_loss = total_loss / n_total
        val_acc = evaluate_accuracy(model, val_loader)

        nnz = sum(g.nnz() for g in groups)
        current_sparsity = 1.0 - nnz / total_prunable
        phase = "warmup" if epoch < T_w else ("sparsify" if epoch < T_f else "frozen")

        # Training FLOPs accounting
        dense_macs, sparse_macs = count_sparse_flops(model)
        if phase == "warmup":
            # Warmup: fully dense training → 3 * dense_macs
            epoch_train_macs = 3 * dense_macs * n_train_samples
        elif phase == "frozen":
            # Frozen: gradients masked to support → 3 * sparse_macs
            epoch_train_macs = 3 * sparse_macs * n_train_samples
        else:
            # Sparsify: ASTRA needs dense dW for scoring → 2*sparse + dense
            epoch_train_macs = (2 * sparse_macs + dense_macs) * n_train_samples
        cumulative_train_macs += epoch_train_macs

        log.info(
            "Epoch %3d/%d | %-8s | lr=%.5f | loss=%.4f | "
            "train=%.2f%% | val=%.2f%% | sparsity=%.4f | cum_macs=%.1fG",
            epoch + 1, T, phase,
            optimizer.param_groups[0]["lr"],
            train_loss, train_acc, val_acc, current_sparsity,
            cumulative_train_macs / 1e9,
        )
        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "train/acc": train_acc,
            "val/acc": val_acc,
            "sparsity": current_sparsity,
            "lr": optimizer.param_groups[0]["lr"],
            "train_macs/epoch_G": epoch_train_macs / 1e9,
            "train_macs/cumulative_G": cumulative_train_macs / 1e9,
        }, step=epoch)

        json_results["epoch_log"].append({
            "epoch": epoch + 1,
            "phase": phase,
            "lr": round(optimizer.param_groups[0]["lr"], 6),
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 2),
            "val_acc": round(val_acc, 2),
            "sparsity": round(current_sparsity, 4),
            "sparse_macs": sparse_macs,
            "cumulative_train_macs": cumulative_train_macs,
        })
        save_json()

        if val_acc > best_acc:
            best_acc = val_acc
            if cfg.training.get("checkpoint_dir"):
                save_checkpoint(
                    model,
                    f"wrn22_{cfg.dataset.name}_{cfg.sparsity_type}_s{cfg.sparsity}_unbiased",
                    epoch,
                    cfg.training.checkpoint_dir,
                    OmegaConf.to_container(cfg),
                )

    # Final test eval + checkpoint
    dense_macs, sparse_macs = count_sparse_flops(model)
    flops_ratio = sparse_macs / dense_macs if dense_macs > 0 else 0.0
    test_acc = evaluate_accuracy(model, val_loader)
    ckpt_path = os.path.join(exp_dir, "model_final.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "sparsity": current_sparsity,
        "test_acc": test_acc,
        "dense_macs": dense_macs,
        "sparse_macs": sparse_macs,
    }, ckpt_path)

    json_results["best_val_acc"] = best_acc
    json_results["test_acc"] = test_acc
    json_results["final_sparsity"] = current_sparsity
    json_results["dense_macs"] = dense_macs
    json_results["final_sparse_macs"] = sparse_macs
    json_results["inference_flops_ratio"] = flops_ratio
    json_results["total_train_macs"] = cumulative_train_macs
    json_results["checkpoint"] = ckpt_path
    json_results["status"] = "done"
    save_json()
    log.info("Test acc: %.2f%%  Inference MACs: %.1fM / %.1fM = %.4f",
             test_acc, sparse_macs/1e6, dense_macs/1e6, flops_ratio)
    log.info("Checkpoint: %s", ckpt_path)

    wandb.summary["best_val_acc"] = best_acc
    wandb.summary["test_acc"] = test_acc
    wandb.summary["final_sparsity"] = current_sparsity
    wandb.summary["total_train_macs_G"] = cumulative_train_macs / 1e9
    wandb.summary["inference_flops_ratio"] = flops_ratio
    wandb.summary["target_sparsity"] = cfg.sparsity
    wandb.summary["sparsity_type"] = cfg.sparsity_type
    wandb.summary["dataset"] = cfg.dataset.name
    wandb.summary["seed"] = cfg.training.seed
    run.finish()


if __name__ == "__main__":
    main()
