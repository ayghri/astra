import torchvision
import torch
from .transforms import get_transforms


def get_dataset(
    dataset_name,
    data_dir,
    train_transform,
    test_transform,
):
    trainset, testset = None, None
    if dataset_name == "mnist":
        trainset = torchvision.datasets.MNIST(
            root=data_dir, train=True, download=True, transform=train_transform
        )
        testset = torchvision.datasets.MNIST(
            root=data_dir, train=False, download=True, transform=test_transform
        )

    if dataset_name == "cifar10":
        trainset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=train_transform
        )
        testset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=test_transform
        )

    if dataset_name == "cifar100":
        trainset = torchvision.datasets.CIFAR100(
            root=data_dir, train=True, download=True, transform=train_transform
        )
        testset = torchvision.datasets.CIFAR100(
            root=data_dir, train=False, download=True, transform=test_transform
        )

    if dataset_name == "cinic-10":
        trainset = torchvision.datasets.ImageFolder(
            data_dir + "/cinic-10/trainval", transform=train_transform
        )
        testset = torchvision.datasets.ImageFolder(
            data_dir + "/cinic-10/test", transform=test_transform
        )

    if dataset_name == "tiny_imagenet":
        trainset = torchvision.datasets.ImageFolder(
            data_dir + "/tiny_imagenet/train", transform=train_transform
        )
        testset = torchvision.datasets.ImageFolder(
            data_dir + "/tiny_imagenet/val", transform=test_transform
        )

    assert trainset is not None and testset is not None, (
        "Error, no dataset %s" % dataset_name
    )
    return trainset, testset


def get_dataloaders(
    dataset_name, data_dir, batch_size, num_workers=2, test_multiplier=2
):
    train_t, test_t = get_transforms(dataset_name)
    trainset, testset = get_dataset(dataset_name, data_dir, train_t, test_t)

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        # persistent_workers=True if num_workers > 0 else False,
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size * test_multiplier,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        # persistent_workers=True if num_workers > 0 else False,
    )

    return trainloader, testloader
