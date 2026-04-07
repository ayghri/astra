# CV Benchmarking Plan: ASTRA vs Baselines

## Methods

| Method | Script | Description |
|--------|--------|-------------|
| ASTRA | `scripts/train_cifar.py` | SGD + soft-threshold with EMA lambda |
| ASTRA Unbiased | `scripts/train_unbiased_cifar.py` | SGD + mask-unthresholding (no L1 bias) |
| SRigL | `scripts/train_srigl_cifar.py` | RigL + constant fan-in + dynamic ablation |
| RigL | `scripts/train_srigl_cifar.py` | Vanilla RigL |
| GMP | `scripts/train_magnitude_cifar.py` | Gradual Magnitude Pruning (cubic schedule) |
| IMP | `scripts/train_magnitude_cifar.py` | Iterative Magnitude Pruning (lottery ticket) |

## Datasets & Models

| Dataset | Classes | Model | Data dir |
|---------|---------|-------|----------|
| CIFAR-10 | 10 | WideResNet-22-2 | /buckets/datasets/torchvision |
| CIFAR-100 | 100 | WideResNet-22-2 | /buckets/datasets/torchvision |
| ImageNet-100 | 100 | ResNet-50 (timm) | /buckets/datasets/torchvision/imagenet-100 |

## Sparsities

- 80%, 90%, 95%
- Types: unstructured, fanin
- Distribution: uniform, erk

## Metrics (all scripts report)

- Train/val accuracy per epoch
- Final test accuracy
- Per-layer sparsity at init
- Training FLOPs (cumulative MACs, dense dW during explore)
- Inference FLOPs (final sparse MACs)
- Model checkpoint

## Commands

### CIFAR-10

```bash
# ASTRA
python scripts/train_cifar.py --config-name cifar10_astra sparsity=0.9 sparsity_type=fanin sparsity_dist=erk wandb.mode=disabled

# ASTRA Unbiased
python scripts/train_unbiased_cifar.py --config-name cifar10_unbiased sparsity=0.9 sparsity_type=fanin sparsity_dist=erk

# SRigL
python scripts/train_srigl_cifar.py --config-name cifar10_srigl sparsity=0.9 dense_alloc=0.1

# RigL
python scripts/train_srigl_cifar.py --config-name cifar10_srigl sparsity=0.9 dense_alloc=0.1 rigl.const_fan_in=false rigl.dynamic_ablation=false

# GMP
python scripts/train_magnitude_cifar.py --config-name cifar10_gmp sparsity=0.9

# IMP
python scripts/train_magnitude_cifar.py --config-name cifar10_imp sparsity=0.9
```

### CIFAR-100

```bash
python scripts/train_cifar.py --config-name cifar100_astra sparsity=0.9 sparsity_type=fanin sparsity_dist=erk wandb.mode=disabled
python scripts/train_unbiased_cifar.py --config-name cifar10_unbiased dataset.name=cifar100 dataset.num_classes=100 sparsity=0.9 sparsity_type=fanin sparsity_dist=erk
python scripts/train_srigl_cifar.py --config-name cifar10_srigl dataset.name=cifar100 dataset.num_classes=100 sparsity=0.9 dense_alloc=0.1
python scripts/train_magnitude_cifar.py --config-name cifar100_gmp sparsity=0.9
python scripts/train_magnitude_cifar.py --config-name cifar100_imp sparsity=0.9
```

### ImageNet-100

```bash
python scripts/train_srigl_cifar.py --config-name cifar10_srigl \
    dataset.name=imagenet100 dataset.num_classes=100 \
    dataset.data_dir=/buckets/datasets/torchvision/imagenet-100 \
    model.name=resnet50 training.epochs=90 training.batch_size=128 \
    sparsity=0.9 dense_alloc=0.1

python scripts/train_magnitude_cifar.py --config-name cifar10_gmp \
    dataset.name=imagenet100 dataset.num_classes=100 \
    dataset.data_dir=/buckets/datasets/torchvision/imagenet-100 \
    model.name=resnet50 model.depth=50 training.epochs=90 training.batch_size=128 \
    sparsity=0.9
```

### Sweep (3 seeds x 3 sparsities)

```bash
for s in 0.8 0.9 0.95; do
  da=$(python -c "print(1-$s)")
  for seed in 42 123 456; do
    python scripts/train_cifar.py --config-name cifar10_astra \
      sparsity=$s sparsity_type=fanin sparsity_dist=erk training.seed=$seed wandb.mode=disabled &
    python scripts/train_srigl_cifar.py --config-name cifar10_srigl \
      sparsity=$s dense_alloc=$da training.seed=$seed &
    wait
  done
done
```
