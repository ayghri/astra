# --- Imports and Helper Functions remain the same ---
import torch, torch.nn as nn, torchvision, torchvision.transforms as transforms
from torch.utils.data import DataLoader
import warnings, os, time
from torch.optim.optimizer import Optimizer
from collections import defaultdict


# Helper function for soft-thresholding
@torch.no_grad()
def soft_threshold(x, thresh):
    thresh = torch.tensor(max(0.0, thresh), dtype=x.dtype, device=x.device)
    return torch.sign(x) * torch.relu(torch.abs(x) - thresh)


# Helper function to calculate k-th largest absolute value safely
@torch.no_grad()
def calculate_kth_value(vec, k):
    if vec is None or vec.numel() == 0:
        return 0.0
    N = vec.numel()
    k = max(1, min(N, int(k)))  # Ensure 1 <= k <= N
    kth_smallest_index = max(1, N - k + 1)
    try:
        kth_largest_abs_val = torch.kthvalue(
            vec.abs().flatten(), kth_smallest_index
        ).values
    except RuntimeError as e:
        print(
            f"Warning: kthvalue failed (N={N}, k={k}, index={kth_smallest_index}). Error: {e}. Returning 0."
        )
        return 0.0
    return kth_largest_abs_val.item()


# --- CORRECTED Variant 1: PER-PARAMETER Lambda, Skip Bias AND First Layer ---
class AdamH1_ProxSparse_PerParam_SkipFirst(Optimizer):
    """
    Heuristic Adam Variant 1 for Two-Timescale Sparse Optimization.
    Uses Adam's m_hat as the gradient estimate.
    Maintains and updates lambda PER PARAMETER tensor.
    SKIPS soft-thresholding for bias parameters (dim==1) AND
    parameters belonging to the specified first_layer_params set.

    WARNING: Heuristic approach.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        initial_lambda=0.0,
    ):
        # ... (initial checks remain the same) ...
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        warnings.warn(
            "AdamH1_ProxSparse_PerParam_SkipFirst is a heuristic optimizer.",
            UserWarning,
        )

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            initial_lambda=initial_lambda,
        )
        super(AdamH1_ProxSparse_PerParam_SkipFirst, self).__init__(
            params, defaults
        )

    @torch.no_grad()
    def step(
        self,
        k_values_dict,
        beta_lambda,
        first_layer_params_set,
        eta_dict=None,
        closure=None,
    ):
        """
        Performs step, skipping prox for bias and specified first layer params.

        Args:
            k_values_dict (dict): Maps WEIGHT param id (int) -> target k (int)
                                  (MUST exclude first layer params).
            beta_lambda (float): Robbins-Monro step size for lambda updates.
            first_layer_params_set (set): A set containing the parameter objects
                                         (p) belonging to the first layer.
            eta_dict (dict, optional): Maps group index (int) -> step size (float).
            closure (callable, optional): A closure that reevaluates the model.
        Returns:
            Loss (optional).
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        _active_device = None

        for group_idx, group in enumerate(
            self.param_groups
        ):  # Group index 'i' is group_idx
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            initial_lambda = group["initial_lambda"]
            current_eta = group["lr"]  # Default eta for the group
            if eta_dict is not None and group_idx in eta_dict:
                current_eta = eta_dict[group_idx]  # Override if provided

            for p in group["params"]:
                if p.grad is None:
                    continue
                if _active_device is None:
                    _active_device = p.grad.device

                # Identify bias parameters
                is_bias = p.dim() == 1
                # --- Check if parameter is in the first layer set ---
                is_first_layer = p in first_layer_params_set
                # --- ---
                # Determine if prox step should be skipped
                skip_prox = is_bias or is_first_layer

                grad = p.grad
                state = self.state[p]  # Use parameter object 'p' as key

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["lambda_val"] = float(initial_lambda)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                step_tensor = torch.tensor(
                    state["step"], dtype=torch.float32, device=_active_device
                )
                bias_correction1 = 1 - beta1**step_tensor
                # bias_correction2 = 1 - beta2**step_tensor

                # Weight decay (skip bias)
                if wd != 0 and not is_bias:
                    grad = grad.add(p, alpha=wd)

                # Adam moment updates
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(
                    grad, grad.conj(), value=1 - beta2
                )

                # Compute m_hat
                m_hat = exp_avg / (bias_correction1 + eps)

                # Adam-like step component
                adam_step_component = current_eta * m_hat

                # --- Heuristic Update ---
                if skip_prox:
                    # Perform standard Adam-like update (no prox)
                    p.data.sub_(adam_step_component)
                    # Ensure lambda doesn't accumulate for skipped layers/biases if not needed
                    # state['lambda_val'] = 0.0 # Optional: reset lambda for skipped
                else:
                    # For WEIGHT parameters in OTHER layers: update lambda and apply prox step
                    psi_vector_p = (p.data - m_hat).detach()
                    param_id = id(p)
                    # Target k for this param
                    k_p = k_values_dict.get(param_id, 0)

                    # Update lambda only if k_p > 0 (i.e., it's being sparsified)
                    if k_p > 0:
                        psi_val_p = calculate_kth_value(psi_vector_p, k_p)
                        current_lambda_p = state["lambda_val"]
                        updated_lambda_p = (
                            1 - beta_lambda
                        ) * current_lambda_p + beta_lambda * psi_val_p
                        state["lambda_val"] = max(
                            0.0, updated_lambda_p
                        )  # Store updated lambda
                    # If k_p is 0, lambda update is skipped, lambda remains as is

                    # Apply proximal step using the current lambda
                    y = p.data - adam_step_component
                    # Use current lambda
                    threshold = current_eta * state["lambda_val"]
                    p.data = soft_threshold(y, threshold)

        return loss


# ==============================================================================
# Model Definition (Remains the same)
# ==============================================================================
# ... (MLP class) ...
class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        # self.fc2 = nn.Linear(hidden_size, num_classes)
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
# Evaluation and Sparsity Helpers (Remains the same)
# ==============================================================================
# ... (calculate_sparsity_per_layer, evaluate) ...
@torch.no_grad()
def calculate_sparsity_per_layer(model, threshold=1e-8):
    """
    Calculates sparsity per layer, IGNORING biases AND first layer weights
    for the OVERALL calculation.
    """
    sparsity_dict = {}
    total_sparsity_num = 0
    total_sparsity_den = 0
    # Identify first layer weights by name for reporting (adjust if model changes)
    first_layer_weight_names = {"fc1.weight"}

    param_idx = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            is_bias = param.dim() == 1 and "bias" in name
            is_first_weight = name in first_layer_weight_names

            numel = param.numel()
            zero_params = (param.abs() < threshold).sum().item()
            layer_sparsity = (zero_params / numel) * 100.0 if numel > 0 else 0.0

            key_name = f"L{param_idx}_{name}"
            status_tags = []
            if is_bias:
                status_tags.append("BIAS")
            if is_first_weight:
                status_tags.append("SKIPPED")
            if status_tags:
                key_name += f" [{', '.join(status_tags)}]"

            sparsity_dict[key_name] = layer_sparsity

            if not is_bias and not is_first_weight:
                total_sparsity_num += zero_params
                total_sparsity_den += numel

            param_idx += 1

    overall_sparsity = (
        (total_sparsity_num / total_sparsity_den) * 100.0
        if total_sparsity_den > 0
        else 0.0
    )
    return overall_sparsity, sparsity_dict


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
# Main Training Script (Updated Optimizer Call and Setup)
# ==============================================================================
if __name__ == "__main__":
    # --- Configuration ---
    # ... (DEVICE, BATCH_SIZE, EPOCHS, etc.) ...
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    BATCH_SIZE = 128
    EPOCHS = 100
    LEARNING_RATE = 1e-2
    BETA_LAMBDA = 0.005
    ADAM_BETAS = (0.9, 0.999)
    ADAM_EPS = 1e-8
    WEIGHT_DECAY = 1e-7
    INITIAL_LAMBDA = 0.0
    TARGET_NON_ZERO_PERCENTAGE = 0.20  # Target for sparsified layers

    DATA_DIR = "./data_mnist"
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created data directory: {DATA_DIR}")
    PRINT_FREQ = 100

    # --- Data Loading ---
    # ... (remains the same) ...
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
    print("Dataset loaded.")

    # --- Model, Optimizer, Loss ---
    print("Initializing model and optimizer...")
    model = MLP().to(DEVICE)
    # Use the CORRECTED SkipFirst version
    optimizer = AdamH1_ProxSparse_PerParam_SkipFirst(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=ADAM_BETAS,
        eps=ADAM_EPS,
        weight_decay=WEIGHT_DECAY,
        initial_lambda=INITIAL_LAMBDA,
    )
    criterion = nn.CrossEntropyLoss()
    print("Model and optimizer initialized.")

    # --- Sparsity Target PER PARAMETER (Skip Bias & First Layer Weights) ---
    k_values_dict = {}  # Map param ID -> target k
    param_info_list = []

    print(
        "\nCalculating Target k per Weight Parameter (Skipping First Layer)..."
    )
    param_id_to_name = {
        id(p): name for name, p in model.named_parameters() if p.requires_grad
    }
    # --- Get the set of first layer parameters ---
    first_layer_params_set = set(model.fc1.parameters())
    print(
        f"Identified {len(first_layer_params_set)} parameters in the first layer (fc1) to skip."
    )
    # --- ---

    for i, group in enumerate(optimizer.param_groups):
        group_info = {"group_idx": i, "params": []}
        print(f"Processing Optimizer Group {i}")
        for p in group["params"]:
            if p.requires_grad:
                param_id = id(p)
                name = param_id_to_name.get(param_id, f"Group{i}_UnknownParam")
                numel = p.numel()

                # Check skip conditions
                is_bias = p.dim() == 1 and "bias" in name
                is_first_layer = p in first_layer_params_set  # Use the set
                skip_param = is_bias or is_first_layer

                k_p = 0
                status = ""
                if is_bias:
                    status = "N/A (Bias)"
                elif is_first_layer:
                    status = "N/A (Skipped Layer)"
                elif numel <= 0:
                    status = "N/A (Empty)"
                else:  # Weight in a layer to be sparsified
                    target_k = max(1, int(numel * TARGET_NON_ZERO_PERCENTAGE))
                    k_values_dict[param_id] = target_k  # Store k value using ID
                    k_p = target_k
                    status = f"k={k_p}"

                group_info["params"].append(
                    {"name": name, "shape": tuple(p.shape), "status": status}
                )
        param_info_list.append(group_info)

    # Print summary
    for group_info in param_info_list:
        print(f"  Group {group_info['group_idx']}:")
        for p_info in group_info["params"]:
            print(
                f"    Param: {p_info['name']}, Shape: {p_info['shape']}, Status: {p_info['status']}"
            )
    print(f"Total parameters with target k assigned: {len(k_values_dict)}")
    print("Target k calculation complete.")

    # --- Training Loop ---
    print("\nStarting Training...")
    eta_values = {
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
            # Pass the set of first layer params to the step method
            optimizer.step(
                k_values_dict=k_values_dict,
                beta_lambda=BETA_LAMBDA,
                first_layer_params_set=first_layer_params_set,  # Pass the set
                eta_dict=eta_values,
            )

            running_loss += loss.item()

            # --- Logging ---
            if (i_batch + 1) % PRINT_FREQ == 0 or (
                i_batch + 1
            ) == steps_in_epoch:
                log_lambdas = defaultdict(list)
                # Use the set for checking during logging as well
                first_layer_params_ids = {id(p) for p in first_layer_params_set}

                for i_group, group in enumerate(optimizer.param_groups):
                    for p in group["params"]:
                        param_id = id(p)
                        # Check if state exists, is weight, and NOT first layer
                        if (
                            p.requires_grad
                            and not (p.dim() == 1)
                            and param_id not in first_layer_params_ids
                            and p in optimizer.state
                        ):
                            # Check if lambda_val exists in state
                            if "lambda_val" in optimizer.state[p]:
                                lambda_p = optimizer.state[p]["lambda_val"]
                                log_lambdas[i_group].append(lambda_p)
                            # else: lambda state not initialized yet for this param?

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
                    lambda_stats_str = (
                        "N/A (No layers sparsified or no state yet)"
                    )

                print(
                    f"Epoch [{epoch+1}/{EPOCHS}], Step [{i_batch+1}/{steps_in_epoch}], "
                    f"Loss: {running_loss / PRINT_FREQ:.4f}, Lambda Stats: [{lambda_stats_str}]"
                )
                running_loss = 0.0  # Reset running loss after printing

        # End of Epoch Evaluation
        epoch_duration = time.time() - epoch_start_time
        test_acc = evaluate(model, test_loader, DEVICE)
        overall_weight_sparsity, layer_sparsity = calculate_sparsity_per_layer(
            model
        )  # Uses names now

        # Final Lambda Stats
        final_log_lambdas = defaultdict(list)
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

    total_training_time = time.time() - start_time_train
    print(f"\nTraining finished in {total_training_time:.2f} seconds.")
