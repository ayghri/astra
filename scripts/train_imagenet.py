#!/usr/bin/env python
"""ResNet-50 ImageNet-1K training — multi-GPU, bf16, DALI/FFCV/torchvision.

Launch:
    torchrun --nproc_per_node=2 scripts/train_imagenet.py

Override config:
    torchrun --nproc_per_node=2 scripts/train_imagenet.py \
        training.loader=tv wandb.mode=disabled
"""

import json
import math
import os
import time
from pathlib import Path

import hydra
import numpy as np
import timm
import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NUM_CLASSES = 1000


# ── Distributed ──────────────────────────────────────────────────────────────

def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size()


# ── Dataloaders ──────────────────────────────────────────────────────────────

def build_tv_dataloaders(cfg, rank, world_size):
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms

    data_root = Path(get_original_cwd()) / cfg.data
    aug = cfg.aug

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=aug.randaug_ops, magnitude=aug.randaug_magnitude),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=aug.erasing_prob),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_ds = datasets.ImageFolder(data_root / "train", transform=train_transform)
    val_ds = datasets.ImageFolder(data_root / "val", transform=val_transform)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)

    bs, nw = cfg.training.batch_size, cfg.workers
    train_loader = DataLoader(
        train_ds, batch_size=bs, sampler=train_sampler,
        num_workers=nw, pin_memory=True, persistent_workers=nw > 0, prefetch_factor=4,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs * 2, sampler=val_sampler,
        num_workers=nw, pin_memory=True, persistent_workers=nw > 0,
    )
    return train_loader, val_loader, train_sampler


def build_ffcv_dataloaders(cfg, rank, world_size, device):
    from ffcv.loader import Loader, OrderOption
    from ffcv.transforms import (
        ToTensor, ToDevice, Squeeze, NormalizeImage,
        RandomHorizontalFlip, ToTorchImage,
    )
    from ffcv.fields.decoders import (
        IntDecoder, RandomResizedCropRGBImageDecoder, CenterCropRGBImageDecoder,
    )

    data_root = Path(get_original_cwd()) / cfg.data
    bs, nw = cfg.training.batch_size, cfg.workers
    mean = np.array(IMAGENET_MEAN, dtype=np.float32) * 255
    std = np.array(IMAGENET_STD, dtype=np.float32) * 255

    train_pipeline = {
        "image": [
            RandomResizedCropRGBImageDecoder((224, 224), scale=(0.08, 1.0)),
            RandomHorizontalFlip(),
            ToTorchImage(channels_last=False, convert_back_int16=False),
            NormalizeImage(mean, std, np.float32),
            ToDevice(device, non_blocking=True),
        ],
        "label": [IntDecoder(), ToTensor(), Squeeze(), ToDevice(device, non_blocking=True)],
    }
    val_pipeline = {
        "image": [
            CenterCropRGBImageDecoder((224, 224), ratio=224 / 256),
            ToTorchImage(channels_last=False, convert_back_int16=False),
            NormalizeImage(mean, std, np.float32),
            ToDevice(device, non_blocking=True),
        ],
        "label": [IntDecoder(), ToTensor(), Squeeze(), ToDevice(device, non_blocking=True)],
    }

    train_loader = Loader(
        str(data_root / "train.beton"), batch_size=bs, num_workers=nw,
        order=OrderOption.RANDOM, distributed=world_size > 1,
        seed=cfg.training.seed + rank, pipelines=train_pipeline,
        drop_last=True, os_cache=True,
    )
    val_loader = Loader(
        str(data_root / "val.beton"), batch_size=bs * 2, num_workers=nw,
        order=OrderOption.SEQUENTIAL, distributed=world_size > 1,
        pipelines=val_pipeline, drop_last=False, os_cache=True,
    )
    return train_loader, val_loader, None


# ── Augmentation ─────────────────────────────────────────────────────────────

def _one_hot(targets, num_classes):
    return torch.zeros(targets.size(0), num_classes, device=targets.device).scatter_(
        1, targets.unsqueeze(1), 1.0
    )


def mixup(images, targets, alpha):
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    idx = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1 - lam) * images[idx]
    soft = lam * _one_hot(targets, NUM_CLASSES) + (1 - lam) * _one_hot(targets[idx], NUM_CLASSES)
    return mixed, soft


def cutmix(images, targets, alpha):
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    idx = torch.randperm(images.size(0), device=images.device)
    _, _, H, W = images.shape
    cut_h = int(H * (1 - lam) ** 0.5)
    cut_w = int(W * (1 - lam) ** 0.5)
    cx, cy = torch.randint(W, (1,)).item(), torch.randint(H, (1,)).item()
    x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, W)
    y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, H)
    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[idx, :, y1:y2, x1:x2]
    lam = 1.0 - (x2 - x1) * (y2 - y1) / (H * W)
    soft = lam * _one_hot(targets, NUM_CLASSES) + (1 - lam) * _one_hot(targets[idx], NUM_CLASSES)
    return mixed, soft


def apply_mix(images, targets, mixup_alpha, cutmix_alpha):
    use_mixup, use_cutmix = mixup_alpha > 0, cutmix_alpha > 0
    if not use_mixup and not use_cutmix:
        return images, _one_hot(targets, NUM_CLASSES)
    if use_mixup and use_cutmix:
        fn = mixup if torch.rand(1).item() < 0.5 else cutmix
        a = mixup_alpha if torch.rand(1).item() < 0.5 else cutmix_alpha
        return fn(images, targets, a)
    return (mixup if use_mixup else cutmix)(images, targets, mixup_alpha or cutmix_alpha)


# ── LR scheduler ────────────────────────────────────────────────────────────

def build_scheduler(optimizer, epochs, warmup_epochs, steps_per_epoch):
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-4, end_factor=1.0,
        total_iters=warmup_epochs * steps_per_epoch,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=(epochs - warmup_epochs) * steps_per_epoch, eta_min=1e-6,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine],
        milestones=[warmup_epochs * steps_per_epoch],
    )


# ── Metrics ──────────────────────────────────────────────────────────────────

class AverageMeter:
    def __init__(self):
        self.sum = self.count = 0.0

    @property
    def avg(self):
        return self.sum / self.count if self.count else 0.0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    def reset(self):
        self.sum = self.count = 0.0


def accuracy_topk(output, target, topk=(1, 5)):
    with torch.no_grad():
        maxk = max(topk)
        batch = target.size(0)
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        correct = pred.t().eq(target.view(1, -1).expand_as(pred.t()))
        return [correct[:k].reshape(-1).float().sum().item() * 100.0 / batch for k in topk]


# ── Train / Validate ────────────────────────────────────────────────────────

def train_epoch(model, loader, sampler, optimizer, scheduler, scaler, criterion,
                device, epoch, cfg, run, local_rank):
    model.train()
    if sampler is not None:
        sampler.set_epoch(epoch)

    aug = cfg.aug
    print_freq = cfg.training.print_freq
    loss_m, top1_m, top5_m = AverageMeter(), AverageMeter(), AverageMeter()
    imgs_since = 0
    t0 = time.perf_counter()

    for i, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        images, soft_targets = apply_mix(images, targets, aug.mixup_alpha, aug.cutmix_alpha)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(images)
            loss = criterion(output, soft_targets)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if local_rank == 0:
            top1, top5 = accuracy_topk(output.detach(), targets)
            n = images.size(0)
            loss_m.update(loss.item(), n)
            top1_m.update(top1, n)
            top5_m.update(top5, n)
            imgs_since += n

            step = epoch * len(loader) + i
            if run is not None:
                run.log({"train/loss": loss.item(), "train/acc1": top1,
                         "train/acc5": top5, "train/lr": scheduler.get_last_lr()[0]}, step=step)

            if (i + 1) % print_freq == 0:
                elapsed = time.perf_counter() - t0
                print(
                    f"Epoch [{epoch}][{i+1}/{len(loader)}]  "
                    f"Loss {loss_m.avg:.4f}  Acc@1 {top1_m.avg:.2f}%  Acc@5 {top5_m.avg:.2f}%  "
                    f"LR {scheduler.get_last_lr()[0]:.6f}  {imgs_since/elapsed:.0f} img/s",
                    flush=True,
                )
                imgs_since = 0
                t0 = time.perf_counter()


@torch.no_grad()
def validate(model, loader, criterion, device, epoch, run, local_rank):
    model.eval()
    loss_m, top1_m, top5_m = AverageMeter(), AverageMeter(), AverageMeter()

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(images)
            loss = criterion(output, targets)
        top1, top5 = accuracy_topk(output, targets)
        loss_m.update(loss.item(), images.size(0))
        top1_m.update(top1, images.size(0))
        top5_m.update(top5, images.size(0))

    stats = torch.tensor(
        [loss_m.sum, top1_m.sum, top5_m.sum, loss_m.count],
        device=device, dtype=torch.float64,
    )
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    loss_avg = (stats[0] / stats[3]).item()
    top1_avg = (stats[1] / stats[3]).item()
    top5_avg = (stats[2] / stats[3]).item()

    if local_rank == 0:
        print(f"  Val  Loss {loss_avg:.4f}  Acc@1 {top1_avg:.2f}%  Acc@5 {top5_avg:.2f}%", flush=True)
        if run is not None:
            run.log({"val/loss": loss_avg, "val/acc1": top1_avg, "val/acc5": top5_avg}, step=epoch)

    return top1_avg, top5_avg


# ── Main ─────────────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="../configs", config_name="train_imagenet")
def main(cfg: DictConfig):
    local_rank, rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    output_dir = Path(get_original_cwd()) / cfg.output

    torch.manual_seed(cfg.training.seed + rank)

    lr = cfg.optimizer.lr * world_size * cfg.training.batch_size / 256

    run = None
    loader_key = str(cfg.training.get("loader", "tv"))

    if rank == 0:
        print(OmegaConf.to_yaml(cfg))
        print(f"DDP x{world_size}  |  Loader: {loader_key}  |  "
              f"Per-GPU batch: {cfg.training.batch_size}  |  LR: {lr:.5f}")
        run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.get("entity", None),
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=cfg.wandb.get("mode", "disabled"),
            group=cfg.wandb.get("group", None),
        )

    # Create and compile model before FFCV (avoids numba/inductor JIT conflict)
    model = timm.create_model("resnet50", pretrained=False, num_classes=NUM_CLASSES).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    model = torch.compile(model)

    if loader_key == "ffcv":
        train_loader, val_loader, train_sampler = build_ffcv_dataloaders(cfg, rank, world_size, device)
    else:
        train_loader, val_loader, train_sampler = build_tv_dataloaders(cfg, rank, world_size)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr,
        momentum=cfg.optimizer.momentum,
        weight_decay=cfg.optimizer.weight_decay,
        nesterov=True,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    scaler = torch.amp.GradScaler()
    scheduler = build_scheduler(
        optimizer, cfg.training.epochs, cfg.training.warmup_epochs, len(train_loader),
    )

    start_epoch = 0
    best_acc1 = 0.0

    # Resume
    if cfg.training.resume:
        ckpt = torch.load(cfg.training.resume, map_location=device, weights_only=False)
        model.module.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_acc1 = ckpt.get("best_acc1", 0.0)
        if rank == 0:
            print(f"Resumed from epoch {ckpt['epoch']} (best acc1={best_acc1:.2f}%)")

    # JSON log (rank 0 only)
    json_path = output_dir / "results.json"
    results = {"model": "resnet50", "epochs": cfg.training.epochs, "status": "running", "epoch_log": []}

    def save_json():
        if rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(json_path, "w") as f:
                json.dump(results, f, indent=2)

    save_json()

    for epoch in range(start_epoch, cfg.training.epochs):
        train_epoch(
            model, train_loader, train_sampler, optimizer, scheduler,
            scaler, criterion, device, epoch, cfg, run, local_rank,
        )
        acc1, acc5 = validate(model, val_loader, criterion, device, epoch, run, local_rank)

        if rank == 0:
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

            state = {
                "epoch": epoch,
                "state_dict": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_acc1": best_acc1,
            }
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(state, output_dir / "last.pth")
            if is_best:
                torch.save(state, output_dir / "best.pth")
                print(f"  New best: {best_acc1:.2f}%")

            results["epoch_log"].append({
                "epoch": epoch + 1,
                "val_acc1": round(acc1, 2),
                "val_acc5": round(acc5, 2),
                "lr": round(scheduler.get_last_lr()[0], 6),
            })
            results["best_acc1"] = round(best_acc1, 2)
            save_json()

    if rank == 0:
        results["status"] = "done"
        save_json()

    dist.destroy_process_group()
    if rank == 0 and run is not None:
        run.finish()


if __name__ == "__main__":
    main()
