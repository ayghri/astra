import torch
from omegaconf import DictConfig

from rigl_torch.datasets._mnist import MNISTDataStem
from rigl_torch.datasets._cifar import CIFAR10DataStem
from rigl_torch.datasets._cifar100 import CIFAR100DataStem
from rigl_torch.datasets._imagenet import ImageNetDataStem
from rigl_torch.datasets._imagenet100 import ImageNet100DataStem
from rigl_torch.datasets._vit_imagenet import VitImageNetDataStem
try:
    from rigl_torch.datasets._coco import CocoSegmentationDataStem
except ImportError:
    CocoSegmentationDataStem = None


def get_dataloaders(cfg: DictConfig) -> torch.utils.data.DataLoader:
    if cfg.dataset.name.lower() == "mnist":
        data_stem = MNISTDataStem(cfg)
    elif cfg.dataset.name.lower() == "cifar10":
        data_stem = CIFAR10DataStem(cfg)
    elif cfg.dataset.name.lower() == "cifar100":
        data_stem = CIFAR100DataStem(cfg)
    elif cfg.dataset.name.lower() == "imagenet100":
        data_stem = ImageNet100DataStem(cfg, data_path_override=cfg.dataset.root)
    elif cfg.dataset.name.lower() == "imagenet" and cfg.model.name != "vit":
        data_stem = ImageNetDataStem(cfg, data_path_override=cfg.dataset.root)
    elif cfg.dataset.name.lower() == "imagenet" and cfg.model.name == "vit":
        data_stem = VitImageNetDataStem(
            cfg, data_path_override=cfg.dataset.root
        )
    elif cfg.dataset.name.lower() == "coco":
        if CocoSegmentationDataStem is None:
            raise ImportError("pycocotools is required for COCO dataset")
        data_stem = CocoSegmentationDataStem(
            cfg, data_path_override=cfg.dataset.root
        )
    else:
        raise ValueError(
            f"{cfg.dataset.name.lower()} is not a recognized dataset name!"
        )
    train_dataloader, test_dataloader = data_stem.get_train_test_loaders()
    return train_dataloader, test_dataloader
