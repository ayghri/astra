"""ImageNet-100: the first 100 alphabetically-sorted synset classes of ImageNet.

Torchvision sorts ImageNet classes by WNID (n01440764, n01443537, …), so
class indices 0–99 are reproducible across installations.  Labels need no
remapping since they are already in [0, 99].
"""

from typing import Dict, Any, Union
import pathlib

import numpy as np
from torch.utils.data import Subset
from torchvision import transforms, datasets

from rigl_torch.datasets import _data_stem

N_CLASSES = 100


class ImageNet100DataStem(_data_stem.ABCDataStem):
    _IMAGE_HEIGHT = 224
    _IMAGE_WIDTH = 224
    _MEAN_RGB = [0.485, 0.456, 0.406]
    _STDDEV_RGB = [0.229, 0.224, 0.225]

    def __init__(
        self,
        cfg: Dict[str, Any],
        data_path_override: Union[pathlib.Path, str, None] = None,
    ):
        super().__init__(cfg, data_path_override=data_path_override)

    def _get_datasets(self):
        train_dataset = datasets.ImageNet(
            self.data_path, split="train", transform=self._get_transform()
        )
        test_dataset = datasets.ImageNet(
            self.data_path, split="val", transform=self._get_test_transform()
        )
        return (
            self._filter_to_n_classes(train_dataset),
            self._filter_to_n_classes(test_dataset),
        )

    @staticmethod
    def _filter_to_n_classes(dataset) -> Subset:
        """Return a Subset keeping only samples with label < N_CLASSES."""
        indices = np.where(np.array(dataset.targets) < N_CLASSES)[0].tolist()
        return Subset(dataset, indices)

    def _get_transform(self):
        return transforms.Compose(
            [
                transforms.RandomChoice(
                    [transforms.Resize(256), transforms.Resize(480)]
                ),
                transforms.RandomCrop(
                    size=[self._IMAGE_WIDTH, self._IMAGE_HEIGHT]
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=self._MEAN_RGB, std=self._STDDEV_RGB),
            ]
        )

    def _get_test_transform(self):
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(
                    size=[self._IMAGE_WIDTH, self._IMAGE_HEIGHT]
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=self._MEAN_RGB, std=self._STDDEV_RGB),
            ]
        )
