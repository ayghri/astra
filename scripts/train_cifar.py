#!/usr/bin/env python
"""ASTRA CIFAR training — Algorithm 2 (three-phase SASTRA).

Three phases:
  1. Warmup   [0, T_w):   dense SGD, linear LR warmup
  2. Sparsify [T_w, T_f): SASTRA — SGD step then soft-threshold
  3. Frozen   [T_f, T]:   hard-threshold once, SGD on frozen support

Sparsity granularities (cfg.sparsity_type):
  unstructured — element-wise, single global coupling
  fanin        — constant fan-in: equal non-zero inputs per output neuron
  channel      — filter pruning: zero out entire output filters per layer

Usage:
  python scripts/train_cifar.py --config-name cifar10_astra
  python scripts/train_cifar.py --config-name cifar100_astra sparsity=0.95 sparsity_type=fanin
"""

import json
import logging
from typing import Dict, List, Tuple

import hydra
import torch
import torch.nn as nn
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from sparsekit import BlockSpec, ScopeSpec, ScopeCoupling, View

from astra.controllers import AlphaController, EMAController, LambdaController
from astra.data.datasets import get_dataloaders
from astra.models.wideresnet import WideResNet
from astra.proximals import ASTRASparsifier
from astra.train.utils import evaluate_accuracy, save_checkpoint

log = logging.getLogger(__name__)


# ── Layer selection ───────────────────────────────────────────────────────────

def prunable_params(model: nn.Module) -> List[nn.Parameter]:
    """Conv2d / Linear weights only — no biases, no BN."""
    return [
        p
        for n, p in model.named_parameters()
        if p.requires_grad
        and p.ndim >= 2
        and not n.endswith("bias")
        and "bn" not in n
        and "norm" not in n
    ]


# ── Coupling builders ─────────────────────────────────────────────────────────

def _flat_view(p: nn.Parameter) -> View:
    """1-D View of a contiguous parameter."""
    assert p.is_contiguous(), f"param must be contiguous: {p.shape}"
    return View(p, shape=(p.numel(),), stride=(1,))


def _2d_view(p: nn.Parameter) -> Tuple[View, int, int]:
    """2-D View (C_out, filter_size) for any Conv2d or Linear weight."""
    C_out = p.shape[0]
    filter_size = p.numel() // C_out
    assert p.is_contiguous()
    view = View(p, shape=(C_out, filter_size), stride=(filter_size, 1))
    return view, C_out, filter_size


def build_unstructured(
    params: List[nn.Parameter], sparsity: float
) -> Tuple[List[ScopeCoupling], List[int]]:
    """One global ScopeCoupling; each element is its own block.

    grid_shape = (1,) for every layer → all scopes are compatible.
    kappa = total elements to keep (global budget).
    """
    scopes, total = [], 0
    for p in params:
        view = _flat_view(p)
        block = BlockSpec(view, shape=(1,))
        scope = ScopeSpec(block, shape=(-1,))   # grid_shape = (1,)
        scopes.append(scope)
        total += p.numel()
    orders = [(0,)] * len(scopes)
    coupling = ScopeCoupling(scopes, orders, name="global_unstructured")
    kappa = max(1, int(round(total * (1.0 - sparsity))))
    return [coupling], [kappa]


def build_fanin(
    params: List[nn.Parameter], sparsity: float
) -> Tuple[List[ScopeCoupling], List[int]]:
    """One ScopeCoupling per layer; equal non-zero inputs per output neuron.

    Each scope = one output neuron spanning all its inputs.
    grid_shape = (C_out, 1) per layer coupling.
    kappa = number of non-zero inputs to keep per neuron.
    """
    groups, kappas = [], []
    for p in params:
        view, C_out, filter_size = _2d_view(p)
        block = BlockSpec(view, shape=(1, 1))              # element-wise
        scope = ScopeSpec(block, shape=(1, filter_size))   # grid_shape = (C_out, 1)
        coupling = ScopeCoupling([scope], [(0, 1)], name=f"fanin_{p.shape}")
        kappa = max(1, int(round(filter_size * (1.0 - sparsity))))
        groups.append(coupling)
        kappas.append(kappa)
    return groups, kappas


def build_channel(
    params: List[nn.Parameter], sparsity: float
) -> Tuple[List[ScopeCoupling], List[int]]:
    """One ScopeCoupling per layer; each block = one full output filter.

    grid_shape = (1, 1) per layer coupling — all filters compete.
    kappa = number of output filters to keep.
    """
    groups, kappas = [], []
    for p in params:
        view, C_out, filter_size = _2d_view(p)
        block = BlockSpec(view, shape=(1, filter_size))   # one block = one filter
        scope = ScopeSpec(block, shape=(-1, 1))           # grid_shape = (1, 1)
        coupling = ScopeCoupling([scope], [(0, 1)], name=f"channel_{p.shape}")
        kappa = max(1, int(round(C_out * (1.0 - sparsity))))
        groups.append(coupling)
        kappas.append(kappa)
    return groups, kappas


def build_coupling(
    params: List[nn.Parameter], sparsity: float, sparsity_type: str
) -> Tuple[List[ScopeCoupling], List[int]]:
    if sparsity_type == "unstructured":
        return build_unstructured(params, sparsity)
    elif sparsity_type == "fanin":
        return build_fanin(params, sparsity)
    elif sparsity_type == "channel":
        return build_channel(params, sparsity)
    else:
        raise ValueError(f"Unknown sparsity_type: {sparsity_type!r}")


# ── Frozen support ────────────────────────────────────────────────────────────

def freeze_support(
    groups: List[ScopeCoupling],
    kappas: List[int],
    optimizer: torch.optim.Optimizer,
) -> List:
    """Hard-threshold all groups, register gradient mask hooks."""
    for group, kappa in zip(groups, kappas):
        group.hard_threshold(nnz=kappa)

    hooks = []
    for group in groups:
        for scope in group.scopes:
            p = scope.block.view.param
            mask = p.data.abs() > 0
            def _mask_grad(param, m=mask):
                if param.grad is not None:
                    param.grad.mul_(m)
            h = p.register_post_accumulate_grad_hook(_mask_grad
            )
            hooks.append(h)

    # Zero momentum for pruned positions
    for g in optimizer.param_groups:
        for p in g["params"]:
            buf = optimizer.state.get(p, {}).get("momentum_buffer")
            if buf is not None:
                buf.mul_(p.data.abs() > 0)

    return hooks


# ── FLOPs counting ────────────────────────────────────────────────────────────

def count_sparse_flops(
    model: nn.Module, input_size: Tuple = (1, 3, 32, 32)
) -> Tuple[int, int]:
    """Count dense and effective (non-zero) MACs via forward hooks.

    Effective MACs for a layer = dense_MACs * (nnz_weights / total_weights).
    This matches the sparse FLOPs definition used in the SRigL / RigL papers.

    Returns:
        dense_macs:  total MACs if the model were dense
        sparse_macs: effective MACs given current weight sparsity
    """
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


# ── W&B logging helpers ───────────────────────────────────────────────────────

_PHASE_INT = {"warmup": 0, "sparsify": 1, "frozen": 2}


def _layer_sparsity_rows(
    groups: List[ScopeCoupling],
    model: nn.Module,
) -> List[List]:
    """Return rows of [layer_name, shape, sparsity, nnz, total] for each prunable layer."""
    # Build name map: id(param) → layer name
    name_map: Dict[int, str] = {
        id(p): n
        for n, p in model.named_parameters()
    }
    rows = []
    seen = set()
    for group in groups:
        for sp in group.specs():
            p = sp.view.param
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)
            layer_name = name_map.get(pid, "?")
            total = p.numel()
            nnz = int((p.data.abs() > 0).sum().item())
            sparsity = 1.0 - nnz / total if total > 0 else 0.0
            rows.append([layer_name, str(tuple(p.shape)), round(sparsity, 4), nnz, total])
    return rows


def log_sparsity_table(
    groups: List[ScopeCoupling],
    model: nn.Module,
    step: int,
    prefix: str = "",
):
    """Log per-layer sparsity as a W&B Table."""
    rows = _layer_sparsity_rows(groups, model)
    table = wandb.Table(
        columns=["layer", "shape", "sparsity", "nnz", "total"],
        data=rows,
    )
    key = f"{prefix}layer_sparsity" if prefix else "layer_sparsity"
    wandb.log({key: table}, step=step)


def log_lambda_metrics(
    sparsifier: ASTRASparsifier,
    groups: List[ScopeCoupling],
    step: int,
):
    """Log current lambda (threshold) values — mean and max across groups.

    Logged every epoch so the full trajectory is visible in W&B:
    0 during warmup, growing during sparsify, fixed during frozen.
    Lambda can be a tensor (one value per grid position) for fanin/channel,
    so we always aggregate with .mean() / .max().
    """
    means, maxs = [], []
    for group in groups:
        if group in sparsifier.lambdas._momentums:
            lam = sparsifier.lambdas.get(group).float()
            means.append(lam.mean().item())
            maxs.append(lam.max().item())
        else:
            means.append(0.0)
            maxs.append(0.0)
    wandb.log({
        "lambda/mean": sum(means) / len(means),
        "lambda/max": max(maxs),
        "lambda/min": min(means),
    }, step=step)


def log_weight_histograms(
    groups: List[ScopeCoupling],
    model: nn.Module,
    step: int,
    max_layers: int = 6,
):
    """Log weight magnitude histograms for up to max_layers prunable layers."""
    name_map: Dict[int, str] = {
        id(p): n for n, p in model.named_parameters()
    }
    seen, count = set(), 0
    metrics = {}
    for group in groups:
        for sp in group.specs():
            p = sp.view.param
            pid = id(p)
            if pid in seen or count >= max_layers:
                continue
            seen.add(pid)
            count += 1
            name = name_map.get(pid, f"param_{count}").replace(".", "/")
            weights = p.data.abs().flatten().cpu().float().numpy()
            metrics[f"weights/{name}"] = wandb.Histogram(weights, num_bins=64)
    if metrics:
        wandb.log(metrics, step=step)


# ── Main ──────────────────────────────────────────────────────────────────────

@hydra.main(config_path="../configs", config_name="cifar10_astra", version_base="1.3")
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
            f"astra_{cfg.sparsity_type}_{cfg.dataset.name}"
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
    optimizer = SGD(
        model.parameters(),
        lr=cfg.optimizer.lr,
        momentum=cfg.optimizer.momentum,
        weight_decay=cfg.optimizer.weight_decay,
        nesterov=cfg.optimizer.get("nesterov", False),
    )
    lr_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg.training.epochs,
        eta_min=cfg.optimizer.get("min_lr", 0.0),
    )
    criterion = nn.CrossEntropyLoss()

    # ── Sparsity ─────────────────────────────────────────────────────────────
    params_to_prune = prunable_params(model)
    groups, kappas = build_coupling(params_to_prune, cfg.sparsity, cfg.sparsity_type)
    total_prunable = sum(p.numel() for p in params_to_prune)
    log.info(
        "Sparsity type: %s | target: %.3f | prunable params: %d | groups: %d",
        cfg.sparsity_type, cfg.sparsity, total_prunable, len(groups),
    )

    ema_ctrl = EMAController(rho=cfg.astra.ema_rho)
    alpha_ctrl = AlphaController(default=cfg.astra.alpha)
    lambda_ctrl = LambdaController(
        device=device,
        beta=cfg.astra.beta,
        cap=cfg.astra.lambda_max,
    )
    sparsifier = ASTRASparsifier(
        groups=groups,
        kappas=kappas,
        lambdas=lambda_ctrl,
        ema_grad=ema_ctrl,
        alphas=alpha_ctrl,
        optimizer=optimizer,
    )

    T_w = cfg.training.warmup_epochs
    T_f = cfg.training.freeze_epoch
    T = cfg.training.epochs
    freeze_hooks: List = []
    best_acc = 0.0
    log_hist_every = max(1, T // 8)   # ~8 histogram snapshots per run
    cumulative_train_macs = 0
    n_train_samples = len(train_loader.dataset)

    # JSON results log
    json_results = {
        "method": "astra",
        "dataset": cfg.dataset.name,
        "sparsity": cfg.sparsity,
        "sparsity_type": cfg.sparsity_type,
        "seed": cfg.training.seed,
        "epochs": T,
        "status": "running",
        "epoch_log": [],
    }
    json_path = f"astra_{cfg.dataset.name}_{cfg.sparsity_type}_s{cfg.sparsity}_seed{cfg.training.seed}.json"

    def save_json():
        with open(json_path, "w") as f:
            json.dump(json_results, f, indent=2)

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(T):

        if epoch == T_f and not freeze_hooks:
            log.info("Epoch %d: entering frozen-support phase", epoch)
            freeze_hooks = freeze_support(groups, kappas, optimizer)
            # Log layer sparsity at the moment we freeze
            log_sparsity_table(groups, model, step=epoch, prefix="freeze/")

        model.train()
        total_loss, n_correct, n_total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if epoch < T_w:
                sparsifier.step(sparsify=False)
            elif epoch < T_f:
                sparsifier.step(sparsify=True)

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
            "train=%.2f%% | val=%.2f%% | sparsity=%.4f | "
            "ep_macs=%.1fG | cum_macs=%.1fG",
            epoch + 1, T, phase,
            optimizer.param_groups[0]["lr"],
            train_loss, train_acc, val_acc, current_sparsity,
            epoch_train_macs / 1e9, cumulative_train_macs / 1e9,
        )
        wandb.log({
            "epoch": epoch,
            "phase": _PHASE_INT[phase],
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

        # Lambda values — every epoch (0 in warmup, grows in sparsify, fixed in frozen)
        log_lambda_metrics(sparsifier, groups, step=epoch)

        # Periodic weight histograms
        if epoch % log_hist_every == 0 or epoch == T - 1:
            log_weight_histograms(groups, model, step=epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            if cfg.training.get("checkpoint_dir"):
                save_checkpoint(
                    model,
                    f"wrn22_{cfg.dataset.name}_{cfg.sparsity_type}_s{cfg.sparsity}",
                    epoch,
                    cfg.training.checkpoint_dir,
                    OmegaConf.to_container(cfg),
                )

    # ── FLOPs report ─────────────────────────────────────────────────────────
    dense_macs, sparse_macs = count_sparse_flops(
        model, input_size=(1, 3, 32, 32)
    )
    flops_ratio = sparse_macs / dense_macs if dense_macs > 0 else 0.0
    log.info(
        "FLOPs: dense=%.2fM  sparse=%.2fM  ratio=%.4f  total_train=%.1fG",
        dense_macs / 1e6, sparse_macs / 1e6, flops_ratio,
        cumulative_train_macs / 1e9,
    )

    # Final test eval + checkpoint
    test_acc = evaluate_accuracy(model, val_loader)
    ckpt_path = json_path.replace(".json", "_model.pt")
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
    log.info("Checkpoint: %s", ckpt_path)
    log.info("Test acc: %.2f%%", test_acc)

    # ── Final per-layer sparsity table ───────────────────────────────────────
    log_sparsity_table(groups, model, step=T, prefix="final/")

    # ── Summary metrics (appear in W&B run summary panel) ────────────────────
    wandb.summary["best_val_acc"] = best_acc
    wandb.summary["final_sparsity"] = current_sparsity
    wandb.summary["dense_macs_M"] = dense_macs / 1e6
    wandb.summary["sparse_macs_M"] = sparse_macs / 1e6
    wandb.summary["flops_ratio"] = flops_ratio
    wandb.summary["total_train_macs_G"] = cumulative_train_macs / 1e9
    wandb.summary["target_sparsity"] = cfg.sparsity
    wandb.summary["sparsity_type"] = cfg.sparsity_type
    wandb.summary["dataset"] = cfg.dataset.name
    wandb.summary["seed"] = cfg.training.seed

    wandb.log({
        "best_val_acc": best_acc,
        "dense_macs": dense_macs,
        "sparse_macs": sparse_macs,
        "flops_ratio": flops_ratio,
    }, step=T)
    run.finish()


if __name__ == "__main__":
    main()
