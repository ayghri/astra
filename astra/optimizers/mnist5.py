# --- Imports and Helper Functions remain the same ---
import torch, torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os, time
from collections import defaultdict
import tests.old.utils as utils

import optimizers

# --- Variant 2: Using Full Adam Step, PER-PARAMETER Lambda, Skip Bias & First Layer ---


# ==============================================================================
# Model Definition (Remains the same)
# ==============================================================================
# ... (MLP class) ...
class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten image
        out = self.fc1(x)
        out = self.fc2(out)
        return out


# ==============================================================================
# Evaluation and Sparsity Helpers (Remain the same)
# ==============================================================================
# ... (calculate_sparsity_per_layer, evaluate) ...


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return (correct / total) * 100.0


# ==============================================================================
# Main Training Script (Using AdamH2)
# ==============================================================================
if __name__ == "__main__":
    # --- Configuration ---
    # if torch.backends.cuda.is_available():
    # DEVICE = torch.device("cuda")
    # print("Using device: MPS (Apple Silicon GPU)")
    DEVICE = torch.device("cuda")
    # Optional: Add CUDA check if the code needs to run on NVIDIA too
    # elif torch.cuda.is_available():
    #     DEVICE = torch.device("cuda")
    #     print("Using device: CUDA (NVIDIA GPU)")
    # else:
    # DEVICE = torch.device("cpu")
    # print(
    # "Warning: MPS not available. Using CPU."
    # )  # Or print MPS not built, etc.
    print(f"Using device: {DEVICE}")
    BATCH_SIZE = 128
    EPOCHS = 100
    LEARNING_RATE = 5e-3  # This is Adam's LR (alpha) now
    BETA_LAMBDA = 0.01  # Step size for lambda updates
    ADAM_BETAS = (0.9, 0.999)
    ADAM_EPS = 1e-8
    WEIGHT_DECAY = 1e-6
    INITIAL_LAMBDA = 0.0
    TARGET_NON_ZERO_PERCENTAGE = 0.05

    DATA_DIR = "./data_mnist"
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created data directory: {DATA_DIR}")
    PRINT_FREQ = 100

    # --- Data Loading ---
    print("Loading MNIST dataset...")
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_dataset = torchvision.datasets.MNIST(
        root=DATA_DIR, train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root=DATA_DIR, train=False, download=True, transform=transform
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

    # Basic transforms for CIFAR-10 (add augmentation for better results)
    # --- Model, Optimizer, Loss ---
    print("Initializing model and optimizer...")
    model = MLP(input_size=28*28, hidden_size=512).to(DEVICE)
    # --- Use the AdamH2 Variant ---
    utils.initialize_sparse(model, TARGET_NON_ZERO_PERCENTAGE)
    optimizer = optimizers.AdamH2_ProxSparse_PerParam_SkipFirst(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=ADAM_BETAS,
        eps=ADAM_EPS,
        weight_decay=WEIGHT_DECAY,
        initial_lambda=INITIAL_LAMBDA,
    )
    # --- ---
    criterion = nn.CrossEntropyLoss()
    print("Model and optimizer initialized.")

    # --- Sparsity Target PER PARAMETER (Skip Bias & First Layer Weights) ---
    # ... (This setup section remains exactly the same as before) ...
    k_values_dict = {}
    param_info_list = []
    print(
        "\nCalculating Target k per Weight Parameter (Skipping First Layer)..."
    )
    param_id_to_name = {
        id(p): name for name, p in model.named_parameters() if p.requires_grad
    }
    # first_layer_params_set = set(model.fc1.parameters())
    first_layer_params_set = set()
    print(
        f"Identified {len(first_layer_params_set)} parameters in the first layer (fc1) to skip."
    )
    for i, group in enumerate(optimizer.param_groups):
        group_info = {"group_idx": i, "params": []}
        print(f"Processing Optimizer Group {i}")
        for p in group["params"]:
            if p.requires_grad:
                param_id = id(p)
                name = param_id_to_name.get(param_id, f"Group{i}_UnknownParam")
                numel = p.numel()
                is_bias = p.dim() == 1 and "bias" in name
                is_first_layer = p in first_layer_params_set
                skip_param = is_bias or is_first_layer
                k_p = 0
                status = ""
                if is_bias:
                    status = "N/A (Bias)"
                elif is_first_layer:
                    status = "N/A (Skipped Layer)"
                elif numel <= 0:
                    status = "N/A (Empty)"
                else:
                    target_k = max(1, int(numel * TARGET_NON_ZERO_PERCENTAGE))
                    k_values_dict[param_id] = target_k
                    k_p = target_k
                    status = f"k={k_p}"
                group_info["params"].append(
                    {"name": name, "shape": tuple(p.shape), "status": status}
                )
        param_info_list.append(group_info)
    for group_info in param_info_list:
        print(f"  Group {group_info['group_idx']}:")
        for p_info in group_info["params"]:
            print(
                f"    Param: {p_info['name']}, Shape: {p_info['shape']}, Status: {p_info['status']}"
            )
    print(f"Total parameters with target k assigned: {len(k_values_dict)}")
    print("Target k calculation complete.")

    # --- Training Loop ---
    print("\nStarting Training (using AdamH2 Variant)...")
    # alpha_values dict can be used for per-group LR schedules if needed
    alpha_values = {
        i: group["lr"] for i, group in enumerate(optimizer.param_groups)
    }
    global_step = 0
    start_time_train = time.time()

    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        steps_in_epoch = len(train_loader)

        for i_batch, (images, labels) in enumerate(train_loader):
            global_step += 1
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # --- Optimizer Step ---
            # Pass alpha_dict (or None to use default LR)
            optimizer.step(
                k_values_dict=k_values_dict,
                beta_lambda=BETA_LAMBDA,
                first_layer_params_set=first_layer_params_set,
                alpha_dict=alpha_values,
            )  # Pass alpha dict

            running_loss += loss.item()

            # --- Logging ---
            if (i_batch + 1) % PRINT_FREQ == 0 or (
                i_batch + 1
            ) == steps_in_epoch:
                log_lambdas = defaultdict(list)
                first_layer_params_ids = {id(p) for p in first_layer_params_set}
                for i_group, group in enumerate(optimizer.param_groups):
                    for p in group["params"]:
                        param_id = id(p)
                        if (
                            p.requires_grad
                            and not (p.dim() == 1)
                            and param_id not in first_layer_params_ids
                            and p in optimizer.state
                            and "lambda_val" in optimizer.state[p]
                        ):
                            lambda_p = optimizer.state[p]["lambda_val"]
                            log_lambdas[i_group].append(lambda_p)

                if log_lambdas:
                    lambda_stats_str_list = []
                    for g, vals in log_lambdas.items():
                        if vals:
                            min_l, max_l = min(vals), max(vals)
                            avg_l = sum(vals) / len(vals)
                            lambda_stats_str_list.append(
                                f"L{g}(Min:{min_l:.4f}, Avg:{avg_l:.4f}, Max:{max_l:.4f})"
                            )
                        else:
                            lambda_stats_str_list.append(
                                f"L{g}(No Sparsified Weights)"
                            )
                    lambda_stats_str = ", ".join(lambda_stats_str_list)
                else:
                    lambda_stats_str = "N/A"

                print(
                    f"Epoch [{epoch+1}/{EPOCHS}], Step [{i_batch+1}/{steps_in_epoch}], "
                    f"Loss: {running_loss / PRINT_FREQ:.4f}, Lambda Stats: [{lambda_stats_str}]"
                )
                running_loss = 0.0

        # End of Epoch Evaluation
        epoch_duration = time.time() - epoch_start_time
        test_acc = evaluate(model, test_loader, DEVICE)
        overall_weight_sparsity, layer_sparsity = (
            utils.calculate_sparsity_per_layer(model)
        )

        # Final Lambda Stats
        final_log_lambdas = defaultdict(list)
        # first_layer_params_ids = {id(p) for p in first_layer_params_set}
        for i_group, group in enumerate(optimizer.param_groups):
            for p in group["params"]:
                # param_id = id(p)
                # if (
                #     p.requires_grad
                #     and not (p.dim() == 1)
                #     and param_id not in first_layer_params_ids
                #     and p in optimizer.state
                #     and "lambda_val" in optimizer.state[p]
                # ):
                if (
                    p.requires_grad
                    and not (p.dim() == 1)
                    and p in optimizer.state
                    and "lambda_val" in optimizer.state[p]
                ):
                    lambda_p = optimizer.state[p]["lambda_val"]
                    final_log_lambdas[i_group].append(lambda_p)

        final_lambda_stats_str_list = []
        for g, vals in final_log_lambdas.items():
            if vals:
                avg_l = sum(vals) / len(vals)
                final_lambda_stats_str_list.append(f"L{g}(Avg:{avg_l:.4f})")
            else:
                final_lambda_stats_str_list.append(
                    f"L{g}(No Sparsified Weights)"
                )
        final_lambda_stats_str = ", ".join(final_lambda_stats_str_list)

        print("-" * 70)
        print(
            f"End of Epoch {epoch+1}/{EPOCHS} (Duration: {epoch_duration:.2f}s)"
        )
        print(f"  Test Accuracy: {test_acc:.2f}%")
        print(
            f"  Overall Sparsity (Weights, Excluding First Layer): {overall_weight_sparsity:.2f}%"
        )
        print("  Layer Sparsity:")
        for name, sparsity in layer_sparsity.items():
            print(f"    {name}: {sparsity:.2f}%")
        print(
            f"  Final Lambda Stats (Sparsified Weights): [{final_lambda_stats_str}]"
        )
        print("-" * 70)
        utils.print_model_sparsity(model, threshold=1e-10)

    total_training_time = time.time() - start_time_train
    print(f"\nTraining finished in {total_training_time:.2f} seconds.")
