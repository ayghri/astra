#!/usr/bin/env python
"""Train WideResNet-22-2 on CIFAR with RigL / SRigL (constant fan-in).

Usage:
    # SRigL (default: const fan-in + dynamic ablation)
    python scripts/train_srigl_cifar.py --config-name cifar10_srigl

    # RigL (no fan-in constraint)
    python scripts/train_srigl_cifar.py --config-name cifar10_srigl rigl.const_fan_in=false rigl.dynamic_ablation=false

    # Different sparsity
    python scripts/train_srigl_cifar.py --config-name cifar10_srigl sparsity=0.95 dense_alloc=0.05

    # CIFAR-100
    python scripts/train_srigl_cifar.py --config-name cifar10_srigl dataset.name=cifar100 dataset.num_classes=100
"""

import json
import os
import sys
import time

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add condensed-sparsity to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "condensed-sparsity", "src"))
os.environ.setdefault("BASE_PATH", "/tmp/srigl")

from rigl_torch.rigl_constant_fan import RigLConstFanScheduler
from rigl_torch.rigl_scheduler import RigLScheduler

from astra.data.datasets import get_dataloaders
from astra.models.wideresnet import WideResNet
from astra.train.utils import evaluate_accuracy


# ── FLOPs counting ────────────���───────────────────────────────────────────────

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


def get_sparsity(model):
    total, nnz = 0, 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            total += m.weight.numel()
            nnz += (m.weight.data.abs() > 0).sum().item()
    return 1.0 - nnz / total if total > 0 else 0.0


# ── Main ��────────────────────────────��────────────────────────────────────────

@hydra.main(config_path="../configs", config_name="cifar10_srigl", version_base="1.3")
def main(cfg: DictConfig):
    torch.manual_seed(cfg.training.seed)
    device = torch.device(cfg.device)
    num_classes = cfg.dataset.num_classes

    # Experiment directory
    model_tag = f"{'srigl' if cfg.rigl.const_fan_in else 'rigl'}_{cfg.dataset.name}"
    timestamp = time.strftime("%Y%m%d_%H%M")
    exp_dir = os.path.join(cfg.output_dir, f"{timestamp}_{model_tag}_s{cfg.sparsity}")
    os.makedirs(exp_dir, exist_ok=True)
    json_path = os.path.join(exp_dir, "results.json")

    results = {
        "method": "srigl" if cfg.rigl.const_fan_in else "rigl",
        "dataset": cfg.dataset.name,
        "sparsity": cfg.sparsity,
        "dense_alloc": cfg.dense_alloc,
        "const_fan_in": cfg.rigl.const_fan_in,
        "dynamic_ablation": cfg.rigl.dynamic_ablation,
        "seed": cfg.training.seed,
        "epochs": cfg.training.epochs,
        "status": "running",
        "experiment_dir": exp_dir,
        "epoch_log": [],
    }

    def save():
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)

    save()

    # ── Data ────────────────────────────���─────────────────────────────────
    train_loader, val_loader = get_dataloaders(
        cfg.dataset.name, cfg.dataset.data_dir,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
    )

    # ── Model ───��────────────────────────────────��────────────────────────
    model = WideResNet(
        depth=cfg.model.depth,
        widen_factor=cfg.model.widen_factor,
        num_classes=num_classes,
        drop_rate=cfg.model.drop_rate,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: WideResNet-{cfg.model.depth}-{cfg.model.widen_factor}  Params: {total_params}")

    # ── Optimizer ─────────────────────────────────────────────────────────
    optimizer = SGD(
        model.parameters(), lr=cfg.optimizer.lr,
        momentum=cfg.optimizer.momentum,
        weight_decay=cfg.optimizer.weight_decay,
    )
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=cfg.training.epochs)
    criterion = nn.CrossEntropyLoss()

    # ── RigL Scheduler ──────��─────────────────────────────────────────────
    T_end = int(0.75 * cfg.training.epochs * len(train_loader))
    rigl_cls = RigLConstFanScheduler if cfg.rigl.const_fan_in else RigLScheduler
    pruner_kwargs = dict(
        dense_allocation=cfg.dense_alloc,
        T_end=T_end,
        delta=cfg.rigl.delta,
        sparsity_distribution=cfg.rigl.sparsity_distribution,
    )
    if cfg.rigl.const_fan_in:
        pruner_kwargs.update(
            dynamic_ablation=cfg.rigl.dynamic_ablation,
            min_salient_weights_per_neuron=cfg.rigl.min_salient_weights_per_neuron,
            use_sparse_const_fan_in_for_ablation=cfg.rigl.use_sparse_const_fan_in_for_ablation,
            use_sparse_init=cfg.rigl.use_sparse_init,
            init_method_str=cfg.rigl.init_method_str,
        )
    pruner = rigl_cls(model, optimizer, **pruner_kwargs)

    # ── Dense FLOPs baseline ──────────────────────────────────────��───────
    dense_macs, sparse_macs_init = count_sparse_flops(model)
    print(f"Dense MACs: {dense_macs/1e6:.1f}M  Sparse MACs (init): {sparse_macs_init/1e6:.1f}M")
    print(f"Experiment: {exp_dir}")
    results["dense_macs"] = dense_macs

    cumulative_train_macs = 0
    best_acc = 0.0
    n_train_samples = len(train_loader.dataset)

    # ── Training loop ─���───────────────────────��───────────────────────────
    print(f"\nTraining {cfg.dataset.name} | sparsity={cfg.sparsity} | {cfg.training.epochs} epochs")
    print(f"{'Epoch':>5} {'Phase':>8} {'LR':>8} {'Loss':>8} {'Train%':>7} {'Val%':>7} {'Sparsity':>9} {'EpMACs(G)':>10} {'CumMACs(G)':>11}")

    for epoch in range(cfg.training.epochs):
        model.train()
        total_loss, n_correct, n_total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if pruner is not None:
                pruner()

            total_loss += loss.item() * labels.size(0)
            n_correct += (outputs.argmax(1) == labels).sum().item()
            n_total += labels.size(0)

        lr_scheduler.step()

        train_acc = 100.0 * n_correct / n_total
        train_loss = total_loss / n_total
        val_acc = evaluate_accuracy(model, val_loader)

        sparsity = get_sparsity(model)
        dense_macs, sparse_macs = count_sparse_flops(model)

        # Forward uses sparse weights, but backward computes dense dW
        # (PyTorch autograd computes full gradient, mask applied after)
        epoch_train_macs = (2 * sparse_macs + dense_macs) * n_train_samples
        cumulative_train_macs += epoch_train_macs

        phase = "explore" if epoch < int(0.75 * cfg.training.epochs) else "fixed"

        print(
            f"{epoch+1:5d} {phase:>8} {optimizer.param_groups[0]['lr']:8.5f} "
            f"{train_loss:8.4f} {train_acc:6.2f}% {val_acc:6.2f}% "
            f"{sparsity:8.4f} {epoch_train_macs/1e9:10.1f} {cumulative_train_macs/1e9:10.1f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc

        results["epoch_log"].append({
            "epoch": epoch + 1,
            "phase": phase,
            "lr": round(optimizer.param_groups[0]["lr"], 6),
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 2),
            "val_acc": round(val_acc, 2),
            "sparsity": round(sparsity, 4),
            "sparse_macs": sparse_macs,
            "cumulative_train_macs": cumulative_train_macs,
        })
        save()

    # ── Final test eval ─────────────────────────────────────────────────────
    test_acc = evaluate_accuracy(model, val_loader)
    dense_macs_final, final_sparse_macs = count_sparse_flops(model)

    # ── Save checkpoint ───────────────────────────────────────────────────
    ckpt_path = os.path.join(exp_dir, "model_final.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "sparsity": get_sparsity(model),
        "test_acc": test_acc,
        "dense_macs": dense_macs_final,
        "sparse_macs": final_sparse_macs,
    }, ckpt_path)

    results["best_val_acc"] = best_acc
    results["test_acc"] = test_acc
    results["final_sparsity"] = get_sparsity(model)
    results["dense_macs"] = dense_macs_final
    results["final_sparse_macs"] = final_sparse_macs
    results["inference_flops_ratio"] = final_sparse_macs / dense_macs_final
    results["total_train_macs"] = cumulative_train_macs
    results["checkpoint"] = ckpt_path
    results["status"] = "done"
    save()

    print(f"\nDone. Test acc: {test_acc:.2f}%  Best val acc: {best_acc:.2f}%")
    print(f"Final sparsity: {results['final_sparsity']:.4f}")
    print(f"Inference MACs: {final_sparse_macs/1e6:.1f}M / {dense_macs_final/1e6:.1f}M = {results['inference_flops_ratio']:.4f}")
    print(f"Total training MACs: {cumulative_train_macs/1e9:.1f}G")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Results: {json_path}")


if __name__ == "__main__":
    main()
