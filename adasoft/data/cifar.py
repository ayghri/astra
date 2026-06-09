# print("Dataset loaded.")
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2023, 0.1994, 0.2010)
# --- ---

transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),  # Basic augmentation
        transforms.RandomHorizontalFlip(),  # Basic augmentation
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ]
)
transform_test = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(cifar10_mean, cifar10_std)]
)

try:
    train_dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=transform_test
    )

    num_workers = 2 if DEVICE.type == "cuda" else 0
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if DEVICE.type == "cuda" else False,
        persistent_workers=True if num_workers > 0 else False,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if DEVICE.type == "cuda" else False,
        persistent_workers=True if num_workers > 0 else False,
    )
    print("Dataset loaded.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    print(
        "Please check network connection and permissions for directory:",
        DATA_DIR,
    )
    exit()
