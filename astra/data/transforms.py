import torchvision.transforms as transforms


def basic_rgb_crop_flip(means, stds, crop_size=32, padding=4):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(crop_size, padding=padding),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ]
    )
    return train_transform, test_transform


def get_transforms(dataset):
    if dataset == "mnist":
        train_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        return train_transform, test_transform

    if dataset == "cifar10":
        return basic_rgb_crop_flip(
            means=(0.4914, 0.4822, 0.4465),
            stds=(0.2023, 0.1994, 0.2010),
            crop_size=32,
            padding=4,
        )


    if dataset == "cifar100":
        return basic_rgb_crop_flip(
            means=(0.5071, 0.4865, 0.4409),
            stds=(0.2673, 0.2564, 0.2762),
            crop_size=32,
            padding=4,
        )

    if dataset == "cinic-10":
        return basic_rgb_crop_flip(
            means=(0.47889522, 0.47227842, 0.43047404),
            stds=(0.24205776, 0.23828046, 0.25874835),
            crop_size=32,
            padding=4,
        )

    if dataset == "tiny_imagenet":
        return basic_rgb_crop_flip(
            means=(
                0.48024578664982126,
                0.44807218089384643,
                0.3975477478649648,
            ),
            stds=(0.2769864069088257, 0.26906448510256, 0.282081906210584),
            crop_size=64,
            padding=4,
        )

    if dataset in ("imagenet", "imagenet100"):
        means = (0.485, 0.456, 0.406)
        stds = (0.229, 0.224, 0.225)
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ])
        return train_transform, test_transform

    raise NotImplementedError(f"Unknown dataset: {dataset}")
