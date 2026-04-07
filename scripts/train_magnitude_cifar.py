#!/usr/bin/env python
"""Train WideResNet-22-2 with magnitude pruning (IMP or GMP).

IMP (Iterative Magnitude Pruning):
  Train dense for full schedule, prune to target, retrain. Repeat for N rounds.
  Each round prunes a fraction, rewinds to init or epoch-k weights.

GMP (Gradual Magnitude Pruning):
  Single training run. Sparsity ramps linearly from 0 to target over
  [prune_start, prune_end] epochs. Magnitude pruning applied each epoch.

Usage:
  # GMP
  python scripts/train_magnitude_cifar.py --config-name cifar10_gmp

  # IMP (3 rounds, prune 50% each round to reach ~87.5%)
  python scripts/train_magnitude_cifar.py --config-name cifar10_imp

  # Override
  python scripts/train_magnitude_cifar.py --config-name cifar10_gmp sparsity=0.95 sparsity_dist=erk
"""

import json
import logging
import time
from typing import List, Tuple

import hydra
import torch
import torch.nn as nn
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from sparsekit import BlockSpec, ScopeSpec, View

from astra.data.datasets import get_dataloaders
from astra.models.wideresnet import WideResNet
from astra.train.utils import evaluate_accuracy, save_checkpoint
import numpy as np
import os

log = logging.getLogger(__name__)


# ── Layer selection (same as ASTRA/SRigL) ────────────────────────────────────

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


# ── Scope builders ───────────────────────────────────────────────────────────

def _flat_view(p):
    assert p.is_contiguous()
    return View(p, shape=(p.numel(),), stride=(1,))


def _2d_view(p):
    C_out = p.shape[0]
    filter_size = p.numel() // C_out
    assert p.is_contiguous()
    return View(p, shape=(C_out, filter_size), stride=(filter_size, 1)), C_out, filter_size


def erk_sparsities(params, target_sparsity, power_scale=1.0):
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
    if sparsity_type == "unstructured":
        view = _flat_view(p)
        block = BlockSpec(view, shape=(1,))
        return ScopeSpec(block, shape=(-1,))
    elif sparsity_type == "fanin":
        view, C_out, filter_size = _2d_view(p)
        block = BlockSpec(view, shape=(1, 1))
        return ScopeSpec(block, shape=(1, filter_size))
    elif sparsity_type == "channel":
        view, C_out, filter_size = _2d_view(p)
        block = BlockSpec(view, shape=(1, filter_size))
        return ScopeSpec(block, shape=(-1, 1))
    else:
        raise ValueError(f"Unknown sparsity_type: {sparsity_type!r}")


def build_scopes(params, sparsity, sparsity_type, sparsity_dist="uniform"):
    if sparsity_dist == "erk":
        layer_sparsities = erk_sparsities(params, sparsity)
    else:
        layer_sparsities = [sparsity] * len(params)

    scopes, kappas = [], []
    for p, s in zip(params, layer_sparsities):
        if s <= 0:
            continue
        scope = _make_scope(p, sparsity_type)
        kappa = scope.sparsity_to_nnz(s)
        scopes.append(scope)
        kappas.append(kappa)
    return scopes, kappas, layer_sparsities


# ── FLOPs counting ───────────────────────────────────────────────────────────

def count_sparse_flops(model, input_size=(1, 3, 32, 32)):
    dense_macs = 0
    sparse_macs = 0
    hooks = []
    def _hook(mod, inp, out):
        nonlocal dense_macs, sparse_macs
        if isinstance(mod, nn.Conv2d):
            C_out, C_in, kH, kW = mod.weight.shape
            H_out, W_out = out.shape[-2:]
            macs = int(C_out * (C_in * kH * kW) * H_out * W_out)
        elif isinstance(mod, nn.Linear):
            macs = int(mod.in_features * mod.out_features)
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


def get_sparsity(model):
    total, nnz = 0, 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            total += m.weight.numel()
            nnz += (m.weight.data.abs() > 0).sum().item()
    return 1.0 - nnz / total if total > 0 else 0.0


# ── Magnitude pruning ────────────────────────────────────────────────────────

@torch.no_grad()
def magnitude_prune(scopes, kappas):
    """Hard-threshold each scope to keep kappa largest-magnitude blocks."""
    for scope, kappa in zip(scopes, kappas):
        scope.hard_threshold(nnz=kappa)


@torch.no_grad()
def magnitude_prune_at_sparsity(params, target_sparsity, sparsity_type, sparsity_dist="uniform"):
    """Build scopes and prune to target sparsity. Returns scopes, kappas, layer_sparsities."""
    scopes, kappas, layer_sparsities = build_scopes(
        params, target_sparsity, sparsity_type, sparsity_dist
    )
    magnitude_prune(scopes, kappas)
    return scopes, kappas, layer_sparsities


def freeze_support(scopes, kappas, optimizer):
    """Mask gradients to support after final pruning."""
    magnitude_prune(scopes, kappas)
    hooks = []
    for scope in scopes:
        for sp in scope.specs():
            p = sp.view.param
            mask = p.data.abs() > 0
            def _mask_grad(param, m=mask):
                if param.grad is not None:
                    param.grad.mul_(m)
            h = p.register_post_accumulate_grad_hook(_mask_grad)
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


# ── Training one epoch ───────────────────────────────────────────────────────

def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss, n_correct, n_total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        n_correct += (outputs.argmax(1) == labels).sum().item()
        n_total += labels.size(0)
    return total_loss / n_total, 100.0 * n_correct / n_total


# ── Main ─────────────────────────────────────────────────────────────────────

@hydra.main(config_path="../configs", config_name="cifar10_gmp", version_base="1.3")
def main(cfg: DictConfig):
    log.info("\n%s", OmegaConf.to_yaml(cfg))
    torch.manual_seed(cfg.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import hashlib
    method = cfg.method  # "gmp" or "imp"
    cfg_hash = hashlib.md5(OmegaConf.to_yaml(cfg).encode()).hexdigest()[:6]
    timestamp = time.strftime("%Y%m%d_%H%M")
    exp_dir = os.path.join(
        cfg.get("output_dir", "autoresearch/results"),
        f"{timestamp}_{method}_{cfg.dataset.name}_{cfg.sparsity_type}_s{cfg.sparsity}_seed{cfg.training.seed}_{cfg_hash}",
    )
    os.makedirs(exp_dir, exist_ok=True)
    json_path = os.path.join(exp_dir, "results.json")

    results = {
        "method": method,
        "dataset": cfg.dataset.name,
        "sparsity": cfg.sparsity,
        "sparsity_type": cfg.sparsity_type,
        "sparsity_dist": cfg.get("sparsity_dist", "uniform"),
        "seed": cfg.training.seed,
        "status": "running",
        "epoch_log": [],
    }
    def save():
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.get("entity", None),
        mode=cfg.wandb.get("mode", "disabled"),
        config=OmegaConf.to_container(cfg, resolve=True),
        name=f"{method}_{cfg.dataset.name}_s{cfg.sparsity}_seed{cfg.training.seed}",
    )

    # Data
    train_loader, val_loader = get_dataloaders(
        cfg.dataset.name, cfg.dataset.data_dir,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
    )

    # Model
    model = WideResNet(
        depth=cfg.model.depth, widen_factor=cfg.model.widen_factor,
        num_classes=cfg.dataset.num_classes, drop_rate=cfg.model.drop_rate,
    ).to(device)
    log.info("Params: %d", sum(p.numel() for p in model.parameters()))

    params_to_prune = prunable_params(model)
    total_prunable = sum(p.numel() for p in params_to_prune)

    # Log target per-layer sparsities
    _, _, layer_sparsities = build_scopes(
        params_to_prune, cfg.sparsity, cfg.sparsity_type,
        cfg.get("sparsity_dist", "uniform"),
    )
    name_map = {id(p): n for n, p in model.named_parameters()}
    for p, s in zip(params_to_prune, layer_sparsities):
        ln = name_map.get(id(p), "?")
        if s <= 0:
            log.info("  %-40s shape=%-20s DENSE", ln, str(tuple(p.shape)))
        else:
            log.info("  %-40s shape=%-20s target_sparsity=%.4f", ln, str(tuple(p.shape)), s)

    criterion = nn.CrossEntropyLoss()
    cumulative_train_macs = 0
    n_train = len(train_loader.dataset)
    best_acc = 0.0
    global_epoch = 0

    if method == "gmp":
        # ── GMP: Gradual Magnitude Pruning (Zhu & Gupta 2017) ────────────
        # Cubic sparsity schedule: s_t = s_f * (1 - (1 - (t-t0)/(tf-t0))^3)
        T = cfg.training.epochs
        prune_start = cfg.pruning.prune_start
        prune_end = cfg.pruning.prune_end
        prune_freq = cfg.pruning.get("prune_freq", 1)  # epochs between pruning

        optimizer = SGD(model.parameters(), lr=cfg.optimizer.lr,
                        momentum=cfg.optimizer.momentum,
                        weight_decay=cfg.optimizer.weight_decay)
        lr_sched = CosineAnnealingLR(optimizer, T_max=T)

        freeze_hooks = []
        log.info("GMP: %d epochs, prune [%d, %d] every %d epochs, cubic schedule, target %.3f",
                 T, prune_start, prune_end, prune_freq, cfg.sparsity)

        for epoch in range(T):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device)
            lr_sched.step()

            if prune_start <= epoch < prune_end and epoch % prune_freq == 0:
                # Cubic schedule (Zhu & Gupta 2017)
                frac = (epoch - prune_start) / (prune_end - prune_start)
                current_target = cfg.sparsity * (1.0 - (1.0 - frac) ** 3)
                scopes, kappas, _ = build_scopes(
                    params_to_prune, current_target, cfg.sparsity_type,
                    cfg.get("sparsity_dist", "uniform"),
                )
                magnitude_prune(scopes, kappas)
                phase = "pruning"
            elif epoch == prune_end and not freeze_hooks:
                scopes, kappas, _ = build_scopes(
                    params_to_prune, cfg.sparsity, cfg.sparsity_type,
                    cfg.get("sparsity_dist", "uniform"),
                )
                freeze_hooks = freeze_support(scopes, kappas, optimizer)
                phase = "frozen"
            elif epoch >= prune_end:
                phase = "frozen"
            else:
                phase = "dense"

            val_acc = evaluate_accuracy(model, val_loader)
            sparsity = get_sparsity(model)
            dense_macs, sparse_macs = count_sparse_flops(model)

            if phase == "dense":
                epoch_macs = 3 * dense_macs * n_train
            elif phase == "frozen":
                epoch_macs = 3 * sparse_macs * n_train
            else:
                # During pruning: dense backward (need full grads for next prune decision)
                epoch_macs = (2 * sparse_macs + dense_macs) * n_train
            cumulative_train_macs += epoch_macs

            if val_acc > best_acc:
                best_acc = val_acc

            log.info(
                "Epoch %3d/%d | %-7s | lr=%.5f | loss=%.4f | "
                "train=%.2f%% | val=%.2f%% | sparsity=%.4f | cum_macs=%.1fG",
                epoch + 1, T, phase, optimizer.param_groups[0]["lr"],
                train_loss, train_acc, val_acc, sparsity,
                cumulative_train_macs / 1e9,
            )
            results["epoch_log"].append({
                "epoch": epoch + 1, "phase": phase,
                "train_loss": round(train_loss, 4),
                "train_acc": round(train_acc, 2),
                "val_acc": round(val_acc, 2),
                "sparsity": round(sparsity, 4),
                "sparse_macs": sparse_macs,
                "cumulative_train_macs": cumulative_train_macs,
            })
            save()
            global_epoch = epoch + 1

    elif method == "imp":
        # ── IMP: Iterative Magnitude Pruning (Frankle & Carlin 2019) ─────
        # Train → prune p% of remaining → rewind to init → retrain with mask
        # Each round: density *= (1 - prune_rate), so after R rounds:
        #   final_density = (1 - prune_rate)^R
        n_rounds = cfg.pruning.rounds
        epochs_per_round = cfg.training.epochs
        prune_rate = cfg.pruning.get("prune_rate", 0.2)  # fraction of remaining to prune each round
        rewind = cfg.pruning.get("rewind", True)
        rewind_epoch = cfg.pruning.get("rewind_epoch", 0)  # 0 = rewind to init

        # Save init (or early epoch) weights for rewinding
        init_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        log.info("IMP: %d rounds x %d epochs, prune_rate=%.2f per round, rewind=%s (epoch %d), target=%.3f",
                 n_rounds, epochs_per_round, prune_rate, rewind, rewind_epoch, cfg.sparsity)

        for rnd in range(n_rounds):
            # Rewind weights to init, keep current masks
            if rnd > 0 and rewind:
                masks = {id(p): (p.data.abs() > 0).float() for p in params_to_prune}
                model.load_state_dict({k: v.to(device) for k, v in init_state.items()})
                for p in params_to_prune:
                    p.data.mul_(masks[id(p)].to(device))

            optimizer = SGD(model.parameters(), lr=cfg.optimizer.lr,
                            momentum=cfg.optimizer.momentum,
                            weight_decay=cfg.optimizer.weight_decay)
            lr_sched = CosineAnnealingLR(optimizer, T_max=epochs_per_round)

            current_sparsity = get_sparsity(model)
            log.info("IMP round %d/%d — training %d epochs (sparsity %.4f)",
                     rnd + 1, n_rounds, epochs_per_round, current_sparsity)

            for epoch in range(epochs_per_round):
                train_loss, train_acc = train_one_epoch(
                    model, train_loader, optimizer, criterion, device)
                lr_sched.step()

                # Save rewind checkpoint at specified epoch (late rewinding)
                if rnd == 0 and epoch == rewind_epoch and rewind_epoch > 0:
                    init_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    log.info("  Saved rewind checkpoint at epoch %d", rewind_epoch)

                val_acc = evaluate_accuracy(model, val_loader)
                sparsity = get_sparsity(model)
                dense_macs, sparse_macs = count_sparse_flops(model)
                # IMP: gradients are masked to support (no topology exploration)
                epoch_macs = 3 * (sparse_macs if sparsity > 0.01 else dense_macs) * n_train
                cumulative_train_macs += epoch_macs

                if val_acc > best_acc:
                    best_acc = val_acc

                global_epoch += 1
                results["epoch_log"].append({
                    "epoch": global_epoch, "round": rnd + 1,
                    "phase": "train",
                    "train_loss": round(train_loss, 4),
                    "train_acc": round(train_acc, 2),
                    "val_acc": round(val_acc, 2),
                    "sparsity": round(sparsity, 4),
                    "sparse_macs": sparse_macs,
                    "cumulative_train_macs": cumulative_train_macs,
                })
                save()

            log.info("  Round %d done — val=%.2f%% — pruning...", rnd + 1, val_acc)

            # Prune: remove prune_rate fraction of remaining weights
            # Cumulative sparsity after rnd+1 rounds: 1 - (1-prune_rate)^(rnd+1)
            target_this_round = 1.0 - (1.0 - prune_rate) ** (rnd + 1)
            target_this_round = min(target_this_round, cfg.sparsity)
            scopes, kappas, _ = build_scopes(
                params_to_prune, target_this_round, cfg.sparsity_type,
                cfg.get("sparsity_dist", "uniform"),
            )
            magnitude_prune(scopes, kappas)
            log.info("  Pruned to sparsity %.4f (target was %.4f)",
                     get_sparsity(model), target_this_round)

    else:
        raise ValueError(f"Unknown method: {method}")

    # ── Final eval + save ────────────────────────────────────────────────
    test_acc = evaluate_accuracy(model, val_loader)
    dense_macs, sparse_macs = count_sparse_flops(model)
    flops_ratio = sparse_macs / dense_macs if dense_macs > 0 else 0.0

    ckpt_path = os.path.join(exp_dir, "model_final.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "sparsity": get_sparsity(model),
        "test_acc": test_acc,
        "dense_macs": dense_macs,
        "sparse_macs": sparse_macs,
    }, ckpt_path)

    results["best_val_acc"] = best_acc
    results["test_acc"] = test_acc
    results["final_sparsity"] = get_sparsity(model)
    results["dense_macs"] = dense_macs
    results["final_sparse_macs"] = sparse_macs
    results["inference_flops_ratio"] = flops_ratio
    results["total_train_macs"] = cumulative_train_macs
    results["checkpoint"] = ckpt_path
    results["status"] = "done"
    save()

    log.info("Test acc: %.2f%%  Best val: %.2f%%", test_acc, best_acc)
    log.info("Inference MACs: %.1fM / %.1fM = %.4f",
             sparse_macs / 1e6, dense_macs / 1e6, flops_ratio)
    log.info("Total training MACs: %.1fG", cumulative_train_macs / 1e9)
    log.info("Results: %s", json_path)
    run.finish()


if __name__ == "__main__":
    main()
