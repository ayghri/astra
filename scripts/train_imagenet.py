#!/misc/envs/bonsai/bin/python
"""
ResNet-50 ImageNet training — 2 GPUs, fp16, DALI GPU decode + MixUp/CutMix.
Config via Hydra, metrics logged to W&B.

Launch:
    torchrun --nproc_per_node=2 scripts/train_imagenet.py

    # FFCV loader (default) requires LD_PRELOAD on this machine:
    LD_PRELOAD=/misc/envs/bonsai/lib/libjpeg.so.8 \
    torchrun --nproc_per_node=2 scripts/train_imagenet.py

Override any config key on the CLI:
    torchrun --nproc_per_node=2 scripts/train_imagenet.py \
        training.loader=dali wandb.mode=disabled
"""
import math
import os
import time
from pathlib import Path

import numpy as np
import hydra
import timm
import torch
import torch.distributed as dist
import torch.nn as nn
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from astra.configs import init_wandb

# Optional DALI import
try:
    import nvidia.dali.fn as dali_fn
    import nvidia.dali.types as dali_types
    from nvidia.dali.pipeline import pipeline_def
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
    HAS_DALI = True
except ImportError:
    HAS_DALI = False

# Optional FFCV import
try:
    from ffcv.loader import Loader, OrderOption
    from ffcv.transforms import (
        ToTensor, ToDevice, Squeeze, NormalizeImage,
        RandomHorizontalFlip, ToTorchImage,
        RandomBrightness, RandomContrast, RandomSaturation,
    )
    from ffcv.fields.decoders import (
        IntDecoder, RandomResizedCropRGBImageDecoder, CenterCropRGBImageDecoder,
    )
    HAS_FFCV = True
except ImportError:
    HAS_FFCV = False


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
NUM_CLASSES   = 1000


# ---------------------------------------------------------------------------
# Distributed setup
# ---------------------------------------------------------------------------

def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size()


# ---------------------------------------------------------------------------
# DALI dataloaders
# ---------------------------------------------------------------------------

class DALILoader:
    """Thin wrapper around DALIClassificationIterator that yields (images, labels) tuples."""

    def __init__(self, dali_iter, num_batches: int):
        self._iter = dali_iter
        self._num_batches = num_batches

    def __len__(self):
        return self._num_batches

    def __iter__(self):
        for batch in self._iter:
            images = batch[0]["data"]                       # float32, already on GPU
            labels = batch[0]["label"].squeeze(-1).long()  # int64
            yield images, labels

    def reset(self):
        """Explicitly reset and start pre-filling the pipeline for the next epoch.
        Call this before validation so the pipeline warms up in the background."""
        self._iter.reset()


def build_dali_dataloaders(cfg: DictConfig, local_rank: int, world_size: int):
    if not HAS_DALI:
        raise RuntimeError("DALI not installed. Run: pip install nvidia-dali-cuda120")

    data_root = str(Path(get_original_cwd()) / cfg.data)
    bs = cfg.training.batch_size
    # Give each GPU an equal share of CPU threads for pre-processing
    num_threads = max(os.cpu_count() // world_size, 4)

    @pipeline_def(batch_size=bs, num_threads=num_threads,
                  device_id=local_rank, seed=42 + local_rank)
    def train_pipeline():
        jpegs, labels = dali_fn.readers.file(
            file_root=data_root + "/train",
            random_shuffle=True,
            shard_id=local_rank,
            num_shards=world_size,
            name="Reader",
        )
        images = dali_fn.decoders.image(jpegs, device="mixed", output_type=dali_types.RGB)
        images = dali_fn.random_resized_crop(
            images, device="gpu", size=224,
            random_area=[0.08, 1.0],
            random_aspect_ratio=[0.75, 4.0 / 3.0],
        )
        # GPU colour jitter (replaces RandAugment; keeps augmentation on-device)
        images = dali_fn.color_twist(
            images, device="gpu",
            brightness=dali_fn.random.uniform(range=[0.6, 1.4]),
            saturation=dali_fn.random.uniform(range=[0.6, 1.4]),
            contrast=dali_fn.random.uniform(range=[0.6, 1.4]),
            hue=dali_fn.random.uniform(range=[-0.1, 0.1]),
        )
        mirror = dali_fn.random.coin_flip(probability=0.5)
        images = dali_fn.crop_mirror_normalize(
            images, device="gpu",
            mean=[m * 255 for m in IMAGENET_MEAN],
            std=[s * 255 for s in IMAGENET_STD],
            mirror=mirror,
            output_layout="CHW",
            dtype=dali_types.FLOAT,
        )
        return images, labels.gpu()

    @pipeline_def(batch_size=bs * 2, num_threads=num_threads,
                  device_id=local_rank, seed=0)
    def val_pipeline():
        jpegs, labels = dali_fn.readers.file(
            file_root=data_root + "/val",
            random_shuffle=False,
            shard_id=local_rank,
            num_shards=world_size,
            name="Reader",
        )
        images = dali_fn.decoders.image(jpegs, device="mixed", output_type=dali_types.RGB)
        images = dali_fn.resize(images, device="gpu", size=256,
                                interp_type=dali_types.INTERP_LINEAR)
        images = dali_fn.crop_mirror_normalize(
            images, device="gpu",
            crop=(224, 224),
            mean=[m * 255 for m in IMAGENET_MEAN],
            std=[s * 255 for s in IMAGENET_STD],
            mirror=0,
            output_layout="CHW",
            dtype=dali_types.FLOAT,
        )
        return images, labels.gpu()

    train_pipe = train_pipeline()
    train_pipe.build()
    train_batches = math.ceil(train_pipe.epoch_size("Reader") / (bs * world_size))

    val_pipe = val_pipeline()
    val_pipe.build()
    val_batches = math.ceil(val_pipe.epoch_size("Reader") / (bs * 2 * world_size))

    train_iter = DALIClassificationIterator(
        train_pipe, reader_name="Reader",
        last_batch_policy=LastBatchPolicy.PARTIAL,
        auto_reset=True,
    )
    val_iter = DALIClassificationIterator(
        val_pipe, reader_name="Reader",
        last_batch_policy=LastBatchPolicy.PARTIAL,
        auto_reset=True,
    )

    return DALILoader(train_iter, train_batches), DALILoader(val_iter, val_batches), None


# ---------------------------------------------------------------------------
# Torchvision dataloaders (fallback)
# ---------------------------------------------------------------------------

def build_tv_dataloaders(cfg: DictConfig, rank: int, world_size: int):
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms

    aug = cfg.aug
    data_root = Path(get_original_cwd()) / cfg.data

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

    train_dataset = datasets.ImageFolder(data_root / "train", transform=train_transform)
    val_dataset   = datasets.ImageFolder(data_root / "val",   transform=val_transform)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler   = DistributedSampler(val_dataset,   num_replicas=world_size, rank=rank, shuffle=False)

    bs, nw = cfg.training.batch_size, cfg.workers
    train_loader = DataLoader(
        train_dataset, batch_size=bs, sampler=train_sampler,
        num_workers=nw, pin_memory=True, persistent_workers=nw > 0, prefetch_factor=4,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=bs * 2, sampler=val_sampler,
        num_workers=nw, pin_memory=True, persistent_workers=nw > 0,
    )
    return train_loader, val_loader, train_sampler


# ---------------------------------------------------------------------------
# FFCV dataloaders
# ---------------------------------------------------------------------------

def build_ffcv_dataloaders(cfg: DictConfig, rank: int, world_size: int, device: torch.device):
    if not HAS_FFCV:
        raise RuntimeError("ffcv not installed. Install with PKG_CONFIG_PATH set — see README.")

    data_root  = Path(get_original_cwd()) / cfg.data
    train_beton = str(data_root / "train.beton")
    val_beton   = str(data_root / "val.beton")

    bs = cfg.training.batch_size
    nw = cfg.workers

    # NormalizeImage expects float32 mean/std in [0, 255] scale
    mean = np.array(IMAGENET_MEAN, dtype=np.float32) * 255
    std  = np.array(IMAGENET_STD,  dtype=np.float32) * 255

    train_pipeline = {
        "image": [
            RandomResizedCropRGBImageDecoder((224, 224), scale=(0.08, 1.0)),
            RandomHorizontalFlip(),
            RandomBrightness(magnitude=0.4, p=0.5),
            RandomContrast(magnitude=0.4,   p=0.5),
            RandomSaturation(magnitude=0.4, p=0.5),
            ToTorchImage(channels_last=False, convert_back_int16=False),
            NormalizeImage(mean, std, np.float32),
            ToDevice(device, non_blocking=True),
        ],
        "label": [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(device, non_blocking=True),
        ],
    }

    val_pipeline = {
        "image": [
            CenterCropRGBImageDecoder((224, 224), ratio=224 / 256),
            ToTorchImage(channels_last=False, convert_back_int16=False),
            NormalizeImage(mean, std, np.float32),
            ToDevice(device, non_blocking=True),
        ],
        "label": [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(device, non_blocking=True),
        ],
    }

    train_loader = Loader(
        train_beton,
        batch_size=bs,
        num_workers=nw,
        order=OrderOption.RANDOM,
        distributed=world_size > 1,
        seed=cfg.training.seed + rank,
        pipelines=train_pipeline,
        drop_last=True,
        os_cache=True,
    )
    val_loader = Loader(
        val_beton,
        batch_size=bs * 2,
        num_workers=nw,
        order=OrderOption.SEQUENTIAL,
        distributed=world_size > 1,
        pipelines=val_pipeline,
        drop_last=False,
        os_cache=True,
    )

    return train_loader, val_loader, None   # no sampler needed (ffcv handles it)


# ---------------------------------------------------------------------------
# RandomErasing (GPU-compatible, used after DALI)
# ---------------------------------------------------------------------------

class RandomErasing:
    """Randomly erase a rectangle of a batch of CHW float tensors (already on GPU)."""

    def __init__(self, p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0.0):
        self.p, self.scale, self.ratio, self.value = p, scale, ratio, value

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return images
        _, _, H, W = images.shape
        area = H * W
        erase_area = area * (self.scale[0] + torch.rand(1).item() * (self.scale[1] - self.scale[0]))
        aspect = self.ratio[0] * (self.ratio[1] / self.ratio[0]) ** torch.rand(1).item()
        h = min(int(round((erase_area * aspect) ** 0.5)), H)
        w = min(int(round((erase_area / aspect) ** 0.5)), W)
        top  = torch.randint(0, H - h + 1, (1,)).item()
        left = torch.randint(0, W - w + 1, (1,)).item()
        images = images.clone()
        images[:, :, top:top + h, left:left + w] = self.value
        return images


# ---------------------------------------------------------------------------
# Augmentation: MixUp / CutMix  (applied on GPU after DALI or torchvision)
# ---------------------------------------------------------------------------

def _one_hot(targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    return torch.zeros(targets.size(0), num_classes, device=targets.device).scatter_(
        1, targets.unsqueeze(1), 1.0
    )


def mixup(images: torch.Tensor, targets: torch.Tensor, alpha: float):
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    idx = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1 - lam) * images[idx]
    soft  = lam * _one_hot(targets, NUM_CLASSES) + (1 - lam) * _one_hot(targets[idx], NUM_CLASSES)
    return mixed, soft


def cutmix(images: torch.Tensor, targets: torch.Tensor, alpha: float):
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


def apply_mix(images: torch.Tensor, targets: torch.Tensor, mixup_alpha: float, cutmix_alpha: float):
    use_mixup, use_cutmix = mixup_alpha > 0, cutmix_alpha > 0
    if not use_mixup and not use_cutmix:
        return images, _one_hot(targets, NUM_CLASSES)
    if use_mixup and use_cutmix:
        return (mixup if torch.rand(1).item() < 0.5 else cutmix)(
            images, targets, mixup_alpha if torch.rand(1).item() < 0.5 else cutmix_alpha
        )
    return (mixup if use_mixup else cutmix)(images, targets, mixup_alpha or cutmix_alpha)


# ---------------------------------------------------------------------------
# LR scheduler
# ---------------------------------------------------------------------------

def build_scheduler(optimizer, epochs: int, warmup_epochs: int, steps_per_epoch: int):
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


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = self.count = 0.0

    @property
    def avg(self):
        return self.sum / self.count if self.count else 0.0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n


def accuracy_topk(output: torch.Tensor, target: torch.Tensor, topk=(1, 5)):
    with torch.no_grad():
        maxk = max(topk)
        batch = target.size(0)
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        correct = pred.t().eq(target.view(1, -1).expand_as(pred.t()))
        return [correct[:k].reshape(-1).float().sum().item() * 100.0 / batch for k in topk]


# ---------------------------------------------------------------------------
# Train / validate
# ---------------------------------------------------------------------------

def train_epoch(
    model, loader, sampler, optimizer, scheduler, scaler, criterion,
    device, epoch: int, cfg: DictConfig, run, local_rank: int,
    erasing: RandomErasing | None,
):
    model.train()
    if sampler is not None:       # None when using DALI (it handles shuffling internally)
        sampler.set_epoch(epoch)

    aug = cfg.aug
    print_freq = cfg.training.print_freq
    loss_m, top1_m, top5_m = AverageMeter(), AverageMeter(), AverageMeter()
    imgs_since_print = 0
    t0 = time.perf_counter()

    for i, (images, targets) in enumerate(loader):
        # DALI outputs are already on-device; torchvision needs the transfer
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # RandomErasing (GPU tensor, works for both paths)
        if erasing is not None:
            images = erasing(images)

        images, soft_targets = apply_mix(images, targets, aug.mixup_alpha, aug.cutmix_alpha)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(images)
            loss   = criterion(output, soft_targets)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if local_rank == 0:
            top1, top5 = accuracy_topk(output.detach(), targets)
            lr = scheduler.get_last_lr()[0]
            n  = images.size(0)
            loss_m.update(loss.item(), n)
            top1_m.update(top1, n)
            top5_m.update(top5, n)
            imgs_since_print += n

            step = epoch * len(loader) + i
            if run is not None:
                run.log({"train/loss": loss.item(), "train/acc1": top1,
                         "train/acc5": top5, "train/lr": lr}, step=step)

            if (i + 1) % print_freq == 0:
                elapsed = time.perf_counter() - t0
                throughput = imgs_since_print / elapsed
                print(
                    f"Epoch [{epoch}][{i+1}/{len(loader)}]  "
                    f"Loss {loss_m.avg:.4f}  "
                    f"Acc@1 {top1_m.avg:.2f}%  Acc@5 {top5_m.avg:.2f}%  "
                    f"LR {lr:.6f}  {throughput:.0f} img/s"
                )
                imgs_since_print = 0
                t0 = time.perf_counter()


@torch.no_grad()
def validate(model, loader, criterion, device, epoch: int, run, local_rank: int):
    model.eval()
    loss_m, top1_m, top5_m = AverageMeter(), AverageMeter(), AverageMeter()

    for images, targets in loader:
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(images)
            loss   = criterion(output, targets)
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
        print(f"  Val  Loss {loss_avg:.4f}  Acc@1 {top1_avg:.2f}%  Acc@5 {top5_avg:.2f}%")
        if run is not None:
            run.log({"val/loss": loss_avg, "val/acc1": top1_avg, "val/acc5": top5_avg}, step=epoch)

    return top1_avg


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(state, output_dir: Path, filename: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, output_dir / filename)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../configs", config_name="train_imagenet")
def main(cfg: DictConfig) -> None:
    local_rank, rank, world_size = setup_distributed()
    device     = torch.device(f"cuda:{local_rank}")
    output_dir = Path(get_original_cwd()) / cfg.output

    torch.manual_seed(cfg.training.seed + rank)

    lr = cfg.optimizer.lr * world_size * cfg.training.batch_size / 256

    run = None
    # Resolve loader: prefer training.loader if set, else fall back to training.dali bool
    loader_key = str(getattr(cfg.training, "loader", "dali" if getattr(cfg.training, "dali", True) else "tv"))

    if rank == 0:
        print(OmegaConf.to_yaml(cfg))
        print(f"DDP x{world_size}  |  Loader: {loader_key}  |  "
              f"Per-GPU batch: {cfg.training.batch_size}  |  LR: {lr:.5f}")
        run = init_wandb(cfg.wandb, cfg)

    if loader_key == "dali":
        train_loader, val_loader, train_sampler = build_dali_dataloaders(cfg, local_rank, world_size)
        erasing = RandomErasing(p=cfg.aug.erasing_prob) if cfg.aug.erasing_prob > 0 else None
    elif loader_key == "ffcv":
        train_loader, val_loader, train_sampler = build_ffcv_dataloaders(cfg, rank, world_size, device)
        erasing = RandomErasing(p=cfg.aug.erasing_prob) if cfg.aug.erasing_prob > 0 else None
    else:
        train_loader, val_loader, train_sampler = build_tv_dataloaders(cfg, rank, world_size)
        erasing = None

    model = timm.create_model("resnet50", pretrained=False, num_classes=NUM_CLASSES).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    model = torch.compile(model)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr,
        momentum=cfg.optimizer.momentum,
        weight_decay=cfg.optimizer.weight_decay,
        nesterov=True,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    scaler    = torch.cuda.amp.GradScaler()
    scheduler = build_scheduler(
        optimizer, cfg.training.epochs, cfg.training.warmup_epochs, len(train_loader)
    )

    start_epoch = 0
    best_acc1   = 0.0

    if cfg.training.resume:
        ckpt = torch.load(cfg.training.resume, map_location=device)
        model.module.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_acc1   = ckpt.get("best_acc1", 0.0)
        if rank == 0:
            print(f"Resumed from epoch {ckpt['epoch']}  (best acc1={best_acc1:.2f}%)")

    for epoch in range(start_epoch, cfg.training.epochs):
        train_epoch(
            model, train_loader, train_sampler, optimizer, scheduler,
            scaler, criterion, device, epoch, cfg, run, local_rank, erasing,
        )
        # Reset train pipeline before validation so it pre-warms during the val pass
        if hasattr(train_loader, "reset"):
            train_loader.reset()
        acc1 = validate(model, val_loader, criterion, device, epoch, run, local_rank)

        if rank == 0:
            is_best   = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            state = {
                "epoch": epoch,
                "state_dict": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_acc1": best_acc1,
            }
            save_checkpoint(state, output_dir, "last.pth")
            if is_best:
                save_checkpoint(state, output_dir, "best.pth")
                print(f"  New best: {best_acc1:.2f}%")

    dist.destroy_process_group()
    if rank == 0 and run is not None:
        run.finish()


if __name__ == "__main__":
    main()
