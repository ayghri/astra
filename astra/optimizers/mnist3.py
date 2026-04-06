import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import math
import warnings
from torch.optim.optimizer import Optimizer
from collections import defaultdict
import os
import time  # For timing epochs

# ==============================================================================
# Helper Functions
# ==============================================================================


@torch.no_grad()
def soft_threshold(x, thresh):
    """Applies element-wise soft-thresholding."""
    # Ensure threshold is non-negative and a scalar tensor on the same device
    thresh = torch.tensor(max(0.0, thresh), dtype=x.dtype, device=x.device)
    return torch.sign(x) * torch.relu(torch.abs(x) - thresh)


@torch.no_grad()
def calculate_kth_value(vec, k):
    """
    Safely calculates the k-th largest absolute value in a flattened vector.
    k=1 means largest, k=N means smallest.
    """
    if vec is None or vec.numel() == 0:
        return 0.0  # Vector is empty

    N = vec.numel()
    # Ensure k is valid (1 <= k <= N)
    k = max(1, min(N, int(k)))

    # k-th largest absolute value is the (N - k + 1)-th smallest absolute value
    # We need the value at index k_eff when sorted descending by abs value
    # which corresponds to index (N - k) when sorted ascending by abs value
    # torch.kthvalue finds the k-th smallest (1-based index for k).
    # So we need the (N - k + 1)-th smallest absolute value.
    kth_smallest_index = max(1, N - k + 1)  # Ensure index is at least 1
    try:
        kth_largest_abs_val = torch.kthvalue(
            vec.abs().flatten(), kth_smallest_index
        ).values
    except RuntimeError as e:
        # Can happen if k is calculated incorrectly relative to numel
        print(
            f"Warning: kthvalue failed (N={N}, k={k}, index={kth_smallest_index}). Error: {e}. Returning 0."
        )
        return 0.0
    return kth_largest_abs_val.item()


# ==============================================================================
# Custom Optimizer
# ==============================================================================


class AdamH1_ProxSparse_PerParam(Optimizer):
    """
    Heuristic Adam Variant 1 for Two-Timescale Sparse Optimization.
    Uses Adam's m_hat as the gradient estimate within a Proximal SGD structure.
    Maintains and updates a lambda value PER PARAMETER tensor.
    SKIPS soft-thresholding for bias parameters (dim==1).

    WARNING: Heuristic approach, potential bias in convergence target.
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
            "AdamH1_ProxSparse_PerParam is a heuristic optimizer.", UserWarning
        )

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            initial_lambda=initial_lambda,
        )
        super(AdamH1_ProxSparse_PerParam, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, k_values_dict, beta_lambda, eta_dict=None, closure=None):
        """
        Performs a single optimization step with per-parameter lambdas.

        Args:
            k_values_dict (dict): A dictionary mapping parameter id (int) to the
                                  target k (int) for that parameter's weights.
            beta_lambda (float): The Robbins-Monro step size for lambda updates.
            eta_dict (dict, optional): A dictionary mapping group index (int)
                                       to the step size (float) for that group.
                                       If None, uses group's default lr.
            closure (callable, optional): A closure that reevaluates the model.
        Returns:
            Loss (optional): The loss if closure is provided.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        _active_device = None  # Track device from first available gradient

        for i, group in enumerate(self.param_groups):  # Group index 'i'
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            initial_lambda = group["initial_lambda"]
            current_eta = group["lr"]  # Default eta for the group
            if eta_dict is not None and i in eta_dict:
                current_eta = eta_dict[i]  # Override if provided

            for p in group["params"]:
                if p.grad is None:
                    continue
                if _active_device is None:
                    _active_device = p.grad.device  # Set device

                # Identify bias parameters (simple check: 1 dimension)
                is_bias = p.dim() == 1

                grad = p.grad
                state = self.state[p]  # Use parameter object 'p' as key

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )  # m_t
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )  # v_t
                    # Initialize lambda state for this parameter
                    state["lambda_val"] = float(
                        initial_lambda
                    )  # Store as float

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                # Use tensor for step count to avoid potential device issues in power op
                step_tensor = torch.tensor(
                    state["step"], dtype=torch.float32, device=_active_device
                )
                bias_correction1 = 1 - beta1**step_tensor
                bias_correction2 = 1 - beta2**step_tensor

                # Weight decay (skip bias)
                if wd != 0 and not is_bias:
                    grad = grad.add(p, alpha=wd)

                # Adam moment updates
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # Use addcmul_ for V update to handle potential complex grads if needed
                exp_avg_sq.mul_(beta2).addcmul_(
                    grad, grad.conj(), value=1 - beta2
                )

                # Compute m_hat
                # Add small epsilon to bias_correction1 to avoid division by zero if step=0 or beta1=1
                m_hat = exp_avg / (bias_correction1 + eps)

                # --- Adam-like step component ---
                adam_step_component = current_eta * m_hat

                # --- Heuristic Update ---
                if is_bias:
                    # For BIAS parameters: perform standard Adam-like step
                    p.data.sub_(adam_step_component)
                else:
                    # For WEIGHT parameters: update lambda and apply prox step
                    # 1. Calculate psi_vector = x_t - m_hat_t
                    # Ensure detachment to avoid tracking history for lambda update
                    psi_vector_p = (p.data - m_hat).detach()

                    # 2. Get k for this parameter using its ID
                    param_id = id(p)
                    k_p = k_values_dict.get(
                        param_id, 0
                    )  # Target sparsity k for this param

                    # 3. Calculate psi_val for this parameter (k-th largest abs val)
                    psi_val_p = calculate_kth_value(
                        psi_vector_p, k_p
                    )  # Pass flattened inside

                    # 4. Update this parameter's lambda value (stored in state[p])
                    current_lambda_p = state["lambda_val"]
                    updated_lambda_p = (
                        1 - beta_lambda
                    ) * current_lambda_p + beta_lambda * psi_val_p
                    state["lambda_val"] = max(
                        0.0, updated_lambda_p
                    )  # Ensure non-negative

                    # 5. Compute tentative update before prox: y = x_t - eta * m_hat_t
                    y = p.data - adam_step_component

                    # 6. Apply proximal operator using the UPDATED parameter lambda
                    # Retrieve lambda again in case state was modified (shouldn't happen here, but safe)
                    threshold = current_eta * state["lambda_val"]
                    p.data = soft_threshold(y, threshold)

        return loss  # Return loss if closure was used


# ==============================================================================
# Model Definition
# ==============================================================================
# class MLP(nn.Module):
#     def __init__(self, input_size=784, hidden_size=256, num_classes=10):
#         super(MLP, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_size, num_classes)
#
#     def forward(self, x):
#         x = x.view(x.size(0), -1) # Flatten image
#         out = self.fc1(x)
#         out = self.relu(out)
#         out = self.fc2(out)
#         return out


class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=512, num_classes=10):
        super(MLP, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten image
        return self.nn(x)


# ==============================================================================
# Evaluation and Sparsity Helpers
# ==============================================================================
@torch.no_grad()
def calculate_sparsity_per_layer(model, threshold=1e-8):
    """Calculates the percentage of weights close to zero per layer/param group, IGNORING biases."""
    sparsity_dict = {}
    total_sparsity_num = 0
    total_sparsity_den = 0

    param_idx = 0  # Use simple index as layer identifier
    for name, param in model.named_parameters():
        if param.requires_grad:
            # --- Check if it's a bias parameter ---
            # Check dimension AND name convention for robustness
            is_bias = param.dim() == 1 and "bias" in name
            # --- ---

            numel = param.numel()
            zero_params = (param.abs() < threshold).sum().item()
            layer_sparsity = (zero_params / numel) * 100.0 if numel > 0 else 0.0

            # Store sparsity, potentially marking biases
            key_name = f"L{param_idx}_{name}" + (" [BIAS]" if is_bias else "")
            sparsity_dict[key_name] = layer_sparsity

            # --- Accumulate overall sparsity ONLY for non-bias parameters ---
            if not is_bias:
                total_sparsity_num += zero_params
                total_sparsity_den += numel
            # --- ---
            param_idx += 1

    overall_sparsity = (
        (total_sparsity_num / total_sparsity_den) * 100.0
        if total_sparsity_den > 0
        else 0.0
    )
    return overall_sparsity, sparsity_dict


@torch.no_grad()
def evaluate(model, data_loader, device):
    """Evaluates model accuracy on the data loader."""
    model.eval()  # Set model to evaluation mode
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
# Main Training Script
# ==============================================================================
if __name__ == "__main__":
    # --- Configuration ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    BATCH_SIZE = 128
    EPOCHS = 50  # Increased epochs for better observation
    LEARNING_RATE = 1e-3  # Base LR for the optimizer group
    BETA_LAMBDA = 0.01  # Step size for lambda updates (Robbins-Monro) - Increased slightly
    ADAM_BETAS = (0.9, 0.999)
    ADAM_EPS = 1e-8
    WEIGHT_DECAY = 0.0  # Weight decay applied only to weights by the optimizer
    INITIAL_LAMBDA = 0.0  # Initial lambda value for all parameters
    TARGET_NON_ZERO_PERCENTAGE = 0.50  # Target 20% non-zero weights

    DATA_DIR = "./data_mnist"  # Changed dir name slightly
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created data directory: {DATA_DIR}")

    PRINT_FREQ = 100  # How often to print training stats

    # --- Data Loading ---
    print("Loading MNIST dataset...")
    try:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean/std
            ]
        )
        train_dataset = torchvision.datasets.MNIST(
            root=DATA_DIR, train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root=DATA_DIR, train=False, download=True, transform=transform
        )

        # Use persistent_workers=True if num_workers > 0 for efficiency
        num_workers = (
            2 if DEVICE.type == "cuda" else 0
        )  # Use workers only if cuda is available
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

    # --- Model, Optimizer, Loss ---
    print("Initializing model and optimizer...")
    model = MLP().to(DEVICE)
    optimizer = AdamH1_ProxSparse_PerParam(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=ADAM_BETAS,
        eps=ADAM_EPS,
        weight_decay=WEIGHT_DECAY,
        initial_lambda=INITIAL_LAMBDA,
    )
    criterion = nn.CrossEntropyLoss()
    print("Model and optimizer initialized.")

    # --- Sparsity Target PER PARAMETER (WEIGHTS ONLY) ---
    k_values_dict = {}  # Map param ID to target k
    param_info_list = []  # For printing info

    print("\nCalculating Target k per Weight Parameter...")
    param_id_to_name = {
        id(p): name for name, p in model.named_parameters() if p.requires_grad
    }

    # Assume one param group for simplicity unless defined otherwise
    if len(optimizer.param_groups) > 1:
        print(
            "Warning: Multiple parameter groups detected. Ensure k calculation is correct for your setup."
        )

    for i, group in enumerate(optimizer.param_groups):
        group_info = {"group_idx": i, "params": []}
        print(f"Processing Optimizer Group {i}")
        for p in group["params"]:
            if p.requires_grad:
                # --- Check if bias ---
                is_bias = p.dim() == 1 and "bias" in param_id_to_name.get(
                    id(p), ""
                )
                # --- ---
                param_id = id(p)
                name = param_id_to_name.get(param_id, f"Group{i}_UnknownParam")
                numel = p.numel()
                k_p = 0  # Default for bias or untracked

                if not is_bias and numel > 0:
                    # Calculate k: target number of non-zeros
                    target_k = max(1, int(numel * TARGET_NON_ZERO_PERCENTAGE))
                    k_values_dict[param_id] = target_k  # Store k value using ID
                    group_info["params"].append(
                        {"name": name, "shape": tuple(p.shape), "k": target_k}
                    )
                elif is_bias:
                    # Just note the bias for info
                    group_info["params"].append(
                        {
                            "name": name,
                            "shape": tuple(p.shape),
                            "k": "N/A (Bias)",
                        }
                    )
                else:
                    group_info["params"].append(
                        {
                            "name": name,
                            "shape": tuple(p.shape),
                            "k": "N/A (Empty)",
                        }
                    )

        param_info_list.append(group_info)

    # Print summary
    for group_info in param_info_list:
        print(f"  Group {group_info['group_idx']}:")
        has_weights = False
        for p_info in group_info["params"]:
            print(
                f"    Param: {p_info['name']}, Shape: {p_info['shape']}, Target k: {p_info['k']}"
            )
            if isinstance(p_info["k"], int):
                has_weights = True
        if not has_weights:
            print(
                "    (No weight parameters found in this group for k calculation)"
            )
    print("Target k calculation complete.")

    # --- Training Loop ---
    print("\nStarting Training...")
    # eta_values dict can be used for per-group LR schedules if needed
    eta_values = {
        i: group["lr"] for i, group in enumerate(optimizer.param_groups)
    }
    global_step = 0

    start_time_train = time.time()  # Time the entire training

    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        model.train()  # Set model to training mode
        running_loss = 0.0
        steps_in_epoch = len(train_loader)

        for i_batch, (images, labels) in enumerate(train_loader):
            global_step += 1
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # --- Standard Training Steps ---
            optimizer.zero_grad(set_to_none=True)  # More memory efficient
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # --- Optimizer Step ---
            # Internally updates lambdas stored in state and applies prox
            optimizer.step(
                k_values_dict=k_values_dict,
                beta_lambda=BETA_LAMBDA,
                eta_dict=eta_values,
            )  # Pass eta dict if using per-group LR

            running_loss += loss.item()

            # --- Logging ---
            if (i_batch + 1) % PRINT_FREQ == 0 or (
                i_batch + 1
            ) == steps_in_epoch:
                # Gather Lambda Stats for logging
                log_lambdas = defaultdict(list)
                for i_group, group in enumerate(optimizer.param_groups):
                    for p in group["params"]:
                        # Check if state exists using 'p' as key, and if it's a weight
                        if (
                            p.requires_grad
                            and not (p.dim() == 1)
                            and p in optimizer.state
                        ):
                            lambda_p = optimizer.state[p].get(
                                "lambda_val", -1.0
                            )  # Get lambda or default
                            log_lambdas[i_group].append(lambda_p)

                # Format stats string safely
                if log_lambdas:
                    lambda_stats_str_list = []
                    for g, vals in log_lambdas.items():
                        if vals:  # Check if list is not empty
                            # print(vals)
                            min_l, max_l = min(vals), max(vals)
                            avg_l = sum(vals) / len(vals)
                            lambda_stats_str_list.append(
                                f"L{g}(Min:{min_l:.4f}, Avg:{avg_l:.4f}, Max:{max_l:.4f})"
                            )
                        else:
                            lambda_stats_str_list.append(f"L{g}(No Weights)")
                    lambda_stats_str = ", ".join(lambda_stats_str_list)
                else:
                    lambda_stats_str = "N/A"

                # Print status
                print(
                    f"Epoch [{epoch+1}/{EPOCHS}], Step [{i_batch+1}/{steps_in_epoch}], "
                    f"Loss: {running_loss / PRINT_FREQ:.4f}, Lambda Stats: [{lambda_stats_str}]"
                )
                running_loss = 0.0  # Reset running loss after printing

        # --- End of Epoch Evaluation ---
        epoch_duration = time.time() - epoch_start_time
        test_acc = evaluate(model, test_loader, DEVICE)
        overall_weight_sparsity, layer_sparsity = calculate_sparsity_per_layer(
            model
        )

        # Gather final lambda stats
        final_log_lambdas = defaultdict(list)
        for i_group, group in enumerate(optimizer.param_groups):
            for p in group["params"]:
                if (
                    p.requires_grad
                    and not (p.dim() == 1)
                    and p in optimizer.state
                ):
                    lambda_p = optimizer.state[p].get("lambda_val", -1.0)
                    final_log_lambdas[i_group].append(lambda_p)

        final_lambda_stats_str_list = []
        for g, vals in final_log_lambdas.items():
            if vals:
                avg_l = sum(vals) / len(vals)
                final_lambda_stats_str_list.append(f"L{g}(Avg:{avg_l:.4f})")
            else:
                final_lambda_stats_str_list.append(f"L{g}(No Weights)")
        final_lambda_stats_str = ", ".join(final_lambda_stats_str_list)

        print("-" * 70)
        print(
            f"End of Epoch {epoch+1}/{EPOCHS} (Duration: {epoch_duration:.2f}s)"
        )
        print(f"  Test Accuracy: {test_acc:.2f}%")
        print(f"  Overall WEIGHT Sparsity: {overall_weight_sparsity:.2f}%")
        # print("  Layer Sparsity (includes biases marked):")
        # for name, sparsity in layer_sparsity.items():
        #     print(f"    {name}: {sparsity:.2f}%")
        print(f"  Final Lambda Stats (Weights): [{final_lambda_stats_str}]")
        print("-" * 70)

    total_training_time = time.time() - start_time_train
    print(f"\nTraining finished in {total_training_time:.2f} seconds.")
