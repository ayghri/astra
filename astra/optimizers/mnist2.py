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


# Helper function for soft-thresholding
def soft_threshold(x, thresh):
    """Applies element-wise soft-thresholding."""
    return torch.sign(x) * torch.relu(torch.abs(x) - thresh)


# --- Variant 1: Using Adam's m_hat, Per-Layer Lambda ---
class AdamH1_ProxSparse_PerLayer(Optimizer):
    """
    Heuristic Adam Variant 1 for Two-Timescale Sparse Optimization.
    Uses Adam's m_hat as the gradient estimate within a Proximal SGD structure.
    Applies and updates lambda values on a PER-PARAMETER-GROUP basis.

    WARNING: Heuristic approach, potential bias in convergence target.
    """

    def __init__(
        self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0
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
            "AdamH1_ProxSparse_PerLayer is a heuristic optimizer lacking strong theoretical guarantees.",
            UserWarning,
        )

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdamH1_ProxSparse_PerLayer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, lambda_dict, eta_dict=None, closure=None):
        """
        Performs a single optimization step with per-group lambdas.

        Args:
            lambda_dict (dict): A dictionary mapping group index (int) to the
                                current lambda value (float) for that group.
            eta_dict (dict, optional): A dictionary mapping group index (int)
                                       to the step size (float) for that group.
                                       If None, uses group's default lr.
            closure (callable, optional): A closure that reevaluates the model
                                          and returns the loss.
        Returns:
            dict: A dictionary mapping group index (int) to a flattened
                  tensor containing the vectors (x_t - m_hat_t) for that group,
                  needed for the external lambda update. Returns None if no
                  gradients were found.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        lambda_update_info = {}
        found_grads = False
        _active_device = None

        # Use group index for consistent mapping
        for i, group in enumerate(self.param_groups):
            group_params_with_grad = []
            group_grads = []
            group_exp_avgs = []
            group_exp_avg_sqs = []
            group_states = []
            group_lambda_update_vectors = []  # Store vectors for this group

            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            current_eta = group["lr"]  # Default eta
            if eta_dict is not None and i in eta_dict:
                current_eta = eta_dict[i]  # Override if provided

            # Get the specific lambda for this group
            current_lambda = lambda_dict.get(
                i, 0.0
            )  # Default to 0 if not found

            for p in group["params"]:
                if p.grad is not None:
                    found_grads = True
                    if _active_device is None:
                        _active_device = p.grad.device

                    group_params_with_grad.append(p)
                    group_grads.append(p.grad)
                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                        state["exp_avg_sq"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

                    group_exp_avgs.append(state["exp_avg"])
                    group_exp_avg_sqs.append(state["exp_avg_sq"])
                    group_states.append(state)

            if not group_params_with_grad:  # Skip group if no grads
                continue

            # === Perform Adam updates for the group ===
            for p, grad, exp_avg, exp_avg_sq, state in zip(
                group_params_with_grad,
                group_grads,
                group_exp_avgs,
                group_exp_avg_sqs,
                group_states,
            ):
                state["step"] += 1
                step_tensor = torch.tensor(
                    state["step"], dtype=torch.float32, device=_active_device
                )
                bias_correction1 = 1 - beta1**step_tensor
                bias_correction2 = 1 - beta2**step_tensor

                if wd != 0:
                    grad = grad.add(p, alpha=wd)

                # Adam moment updates
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(
                    grad, grad.conj(), value=1 - beta2
                )

                # Compute m_hat (bias-corrected first moment)
                m_hat = exp_avg / bias_correction1

                # --- Heuristic Step ---
                # 1. Calculate term for lambda update: x_t - m_hat_t
                # Store this parameter's vector
                lambda_update_vector = (p.data - m_hat).detach().flatten()
                group_lambda_update_vectors.append(lambda_update_vector)

                # 2. Compute gradient step using m_hat: y = x_t - eta * m_hat_t
                y = p.data - current_eta * m_hat

                # 3. Apply proximal operator (soft-thresholding) with GROUP lambda
                threshold = current_eta * current_lambda
                p.data = soft_threshold(y, threshold)
                # ----------------------

            # Concatenate vectors for this group
            if group_lambda_update_vectors:
                lambda_update_info[i] = torch.cat(group_lambda_update_vectors)

        if not found_grads:
            return None

        return (
            lambda_update_info  # Return dict mapping group_idx -> update_vector
        )


# --- End of Optimizer Code ---


# 2. Model Definition (Remains the same)
# class MLP(nn.Module):
#     def __init__(self, input_size=784, hidden_size=256, num_classes=10):
#         super(MLP, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.relu = nn.GELU()
#         self.fc2 = nn.Linear(hidden_size, num_classes)
#
#     def forward(self, x):
#         x = x.view(x.size(0), -1)  # Flatten image
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


# 3. Helper Functions (Updated Sparsity Calculation)
@torch.no_grad()
def calculate_sparsity_per_layer(model, threshold=1e-8):
    """Calculates the percentage of weights close to zero per layer/param group."""
    sparsity_dict = {}
    total_sparsity_num = 0
    total_sparsity_den = 0

    # Assuming parameters in model.parameters() maintain order w.r.t groups
    # A more robust way might involve inspecting optimizer.param_groups if structure differs
    layer_idx = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            numel = param.numel()
            zero_params = (param.abs() < threshold).sum().item()
            layer_sparsity = (zero_params / numel) * 100.0 if numel > 0 else 0.0
            sparsity_dict[f"{name}({layer_idx})"] = (
                layer_sparsity  # Use name and index
            )

            total_sparsity_num += zero_params
            total_sparsity_den += numel
            layer_idx += 1

    overall_sparsity = (
        (total_sparsity_num / total_sparsity_den) * 100.0
        if total_sparsity_den > 0
        else 0.0
    )
    return overall_sparsity, sparsity_dict


@torch.no_grad()
def evaluate(model, data_loader, device):
    """Evaluates model accuracy on the data loader."""
    # ... (remains the same) ...
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


def calculate_kth_value(vec, k):
    """Safely calculates the k-th largest absolute value."""
    if vec is None or vec.numel() == 0 or k <= 0 or k > vec.numel():
        # Handle invalid k or empty vector - return 0 or raise error?
        # Returning 0 means lambda won't increase much in these cases.
        return 0.0
    # k-th largest absolute value is (N - k + 1)-th smallest absolute value
    kth_largest_abs_val = torch.kthvalue(vec.abs(), vec.numel() - k + 1).values
    return kth_largest_abs_val.item()


# 4. Main Training Script (Updated Setup Section)
if __name__ == '__main__':
    # --- Configuration ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    # ... (rest of configuration) ...
    BATCH_SIZE = 128
    EPOCHS = 50
    LEARNING_RATE = 1e-3 # Default LR for AdamH1 (eta)
    BETA_LAMBDA = 0.005 # Step size for lambda update (might need per-layer?)
    ADAM_BETAS = (0.9, 0.999)
    ADAM_EPS = 1e-8
    WEIGHT_DECAY = 0

    DATA_DIR = './data'
    if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)

    # --- Data Loading ---
    # ... (remains the same) ...
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = torchvision.datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=2)

    # --- Model, Optimizer, Loss ---
    model = MLP().to(DEVICE)
    optimizer = AdamH1_ProxSparse_PerLayer(model.parameters(), lr=LEARNING_RATE,
                                           betas=ADAM_BETAS, eps=ADAM_EPS,
                                           weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    # --- Sparsity Target per Layer ---
    # Define desired *percentage* of non-zero elements per group
    target_non_zero_percentage = 0.20 # Example: 10% non-zero overall goal
    k_values = {}
    param_group_info = defaultdict(lambda: {'names': [], 'shapes': [], 'total': 0}) # Use defaultdict

    print("Parameter Groups & Target k:")
    # Map parameter object IDs to their names and info
    param_id_to_name = {id(p): name for name, p in model.named_parameters() if p.requires_grad}
    param_id_to_shape = {id(p): tuple(p.shape) for name, p in model.named_parameters() if p.requires_grad}

    # Iterate through optimizer groups and collect info
    for i, group in enumerate(optimizer.param_groups):
        group_total_params = 0
        for p in group['params']:
             if p.requires_grad:
                 param_id = id(p)
                 if param_id in param_id_to_name: # Check if it's a tracked parameter
                     numel = p.numel()
                     group_total_params += numel
                     param_group_info[i]['names'].append(param_id_to_name[param_id])
                     param_group_info[i]['shapes'].append(param_id_to_shape[param_id])
                 # else: parameter not found in model.named_parameters? Should not happen normally.

        param_group_info[i]['total'] = group_total_params
        # Calculate k for this group based on the target percentage
        k_g = max(1, int(group_total_params * target_non_zero_percentage)) # Ensure k >= 1
        k_values[i] = k_g
        print(f"  Group {i}: Names={param_group_info[i]['names']}, Total Params={group_total_params}, Target k={k_g} ({target_non_zero_percentage*100:.1f}%)")

    # --- Training Loop ---
    # Initialize lambda values per group
    lambda_values = {i: 0.0 for i in range(len(optimizer.param_groups))}
    eta_values = {i: group['lr'] for i, group in enumerate(optimizer.param_groups)}
    global_step = 0

    # ... (Rest of the training loop remains the same) ...
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for i_batch, (images, labels) in enumerate(train_loader): # Renamed 'i' to 'i_batch'
            global_step += 1
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # --- Standard Training Steps ---
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # --- Two-Timescale Update ---
            # 1. Optimizer step (pass dicts for lambda and eta)
            lambda_update_info = optimizer.step(lambda_dict=lambda_values, eta_dict=eta_values)

            # 2. Lambda update per group
            if lambda_update_info is not None:
                with torch.no_grad():
                    for group_idx, group_vec in lambda_update_info.items():
                        k_g = k_values[group_idx]
                        psi_val_g = calculate_kth_value(group_vec, k_g)

                        # Robbins-Monro update for this group's lambda
                        lambda_values[group_idx] = (1 - BETA_LAMBDA) * lambda_values[group_idx] \
                                                   + BETA_LAMBDA * psi_val_g
                        lambda_values[group_idx] = max(0.0, lambda_values[group_idx]) # Ensure lambda >= 0

            running_loss += loss.item()
            if (i_batch + 1) % 100 == 0:
                 lambda_str = ", ".join([f"L{idx}:{val:.4f}" for idx, val in lambda_values.items()])
                 print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i_batch+1}/{len(train_loader)}], "
                       f"Loss: {running_loss / 100:.4f}, Lambdas: [{lambda_str}]")
                 running_loss = 0.0

        # --- End of Epoch Evaluation ---
        test_acc = evaluate(model, test_loader, DEVICE)
        overall_sparsity, layer_sparsity = calculate_sparsity_per_layer(model)
        print("-" * 50)
        print(f"End of Epoch {epoch+1}/{EPOCHS}")
        print(f"Test Accuracy: {test_acc:.2f}%")
        print(f"Overall Model Sparsity: {overall_sparsity:.2f}%")
        print("Layer Sparsity:")
        for name, sparsity in layer_sparsity.items():
            print(f"  {name}: {sparsity:.2f}%")
        lambda_str = ", ".join([f"L{idx}:{val:.4f}" for idx, val in lambda_values.items()])
        print(f"Final Lambdas: [{lambda_str}]")
        print("-" * 50)

    print("Training finished.")
