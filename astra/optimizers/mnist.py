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
import os  # For checking directory existence


# --- Paste the Optimizer Code Here ---
# Helper function for soft-thresholding
def soft_threshold(x, thresh):
    """Applies element-wise soft-thresholding."""
    return torch.sign(x) * torch.relu(torch.abs(x) - thresh)


# --- Variant 1: Using Adam's m_hat as gradient estimate ---
class AdamH1_ProxSparse(Optimizer):
    """
    Heuristic Adam Variant 1 for Two-Timescale Sparse Optimization.

    Uses Adam's bias-corrected first moment estimate (m_hat) as the
    gradient estimate within a Proximal SGD structure.

    WARNING: This is a heuristic approach. Convergence guarantees are
    limited due to the bias in m_hat. The algorithm likely converges
    to a biased fixed point.
    """

    def __init__(
        self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0
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
            "AdamH1_ProxSparse is a heuristic optimizer lacking strong theoretical guarantees.",
            UserWarning,
        )

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdamH1_ProxSparse, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, lambda_val, eta=None, closure=None):
        """
        Performs a single optimization step.

        Args:
            lambda_val (float): The current lambda value for L1 penalty.
            eta (float, optional): The current step size for the prox-gradient
                                   step. If None, uses the optimizer's default lr.
            closure (callable, optional): A closure that reevaluates the model
                                       and returns the loss.
        Returns:
            torch.Tensor: A flattened tensor containing the vectors
                          (x_t - m_hat_t) for all parameters, needed for
                          the external lambda update calculation |x - m_hat|_{(k)}.
                          Returns None if no gradients were found.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        lambda_update_vectors = []
        found_grads = False
        _active_device = None  # Track device

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            current_eta = (
                eta if eta is not None else group["lr"]
            )  # Use provided eta or default lr

            for p in group["params"]:
                if p.grad is None:
                    continue
                found_grads = True
                if _active_device is None:
                    _active_device = p.grad.device  # Get device from first grad

                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )  # m_t
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )  # v_t

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                # Ensure step count is a tensor on the correct device for power operation
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
                )  # Use conj() for complex grads just in case

                # Compute m_hat (bias-corrected first moment)
                m_hat = exp_avg / bias_correction1

                # --- Heuristic Step ---
                # 1. Calculate term for lambda update: x_t - m_hat_t
                lambda_update_vector = (p.data - m_hat).detach().flatten()
                lambda_update_vectors.append(lambda_update_vector)

                # 2. Compute gradient step using m_hat: y = x_t - eta * m_hat_t
                y = p.data - current_eta * m_hat

                # 3. Apply proximal operator (soft-thresholding)
                threshold = current_eta * lambda_val
                p.data = soft_threshold(y, threshold)
                # ----------------------

        if not found_grads:
            return None

        # Concatenate all vectors for external processing
        full_lambda_update_vector = torch.cat(lambda_update_vectors)
        return full_lambda_update_vector  # Return the vector for lambda update


# --- End of Optimizer Code ---


# 2. Model Definition (Simple MLP)
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


# 3. Helper Functions
@torch.no_grad()
def calculate_sparsity(model, threshold=1e-8):
    """Calculates the percentage of weights close to zero."""
    total_params = 0
    zero_params = 0
    for param in model.parameters():
        if param.requires_grad:  # Only count trainable parameters
            total_params += param.numel()
            zero_params += (param.abs() < threshold).sum().item()
    if total_params == 0:
        return 0.0
    return (zero_params / total_params) * 100.0


@torch.no_grad()
def evaluate(model, data_loader, device):
    """Evaluates model accuracy on the data loader."""
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


# 4. Main Training Script
if __name__ == "__main__":
    # --- Configuration ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    BATCH_SIZE = 128
    EPOCHS = 50  # Keep short for demonstration
    LEARNING_RATE = 1e-4  # Initial LR for AdamH1 (eta)
    BETA_LAMBDA = 0.001  # Step size for lambda update (Robbins-Monro)
    # Needs careful tuning, maybe decay
    ADAM_BETAS = (0.9, 0.999)
    ADAM_EPS = 1e-8
    WEIGHT_DECAY = 1e-7  # Set to 0 if L1 is the primary regularizer

    DATA_DIR = "./data"
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # --- Data Loading ---
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

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=2,
    )

    # --- Model, Optimizer, Loss ---
    model = MLP().to(DEVICE)
    optimizer = AdamH1_ProxSparse(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=ADAM_BETAS,
        eps=ADAM_EPS,
        weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.CrossEntropyLoss()

    # --- Sparsity Target ---
    # Calculate total parameters (N)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Define desired *percentage* of non-zero elements (e.g., 10%)
    target_non_zero_percentage = 0.10
    # Calculate k (number of non-zero elements)
    k = int(total_params * target_non_zero_percentage)
    print(f"Total parameters: {total_params}")
    print(
        f"Target non-zero elements (k): {k} ({target_non_zero_percentage*100:.1f}%)"
    )
    print(
        f"This implies target sparsity: {(1-target_non_zero_percentage)*100:.1f}%"
    )

    # --- Training Loop ---
    lambda_val = 0.0  # Initialize lambda
    global_step = 0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            global_step += 1
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # --- Standard Training Steps ---
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # --- Two-Timescale Update ---
            # 1. Optimizer step (returns vector for lambda update)
            # Use current default LR from optimizer for eta
            current_eta = optimizer.param_groups[0]["lr"]
            lambda_update_vec = optimizer.step(
                lambda_val=lambda_val, eta=current_eta
            )

            # 2. Lambda update
            if lambda_update_vec is not None:
                with torch.no_grad():
                    # Calculate psi_val = |vector|_{(k)}
                    # We want the k-th largest absolute value
                    if lambda_update_vec.numel() >= k and k > 0:
                        # Ensure k is valid and vector is large enough
                        kth_largest_abs_val = torch.kthvalue(
                            lambda_update_vec.abs(),
                            lambda_update_vec.numel() - k + 1,
                        ).values
                        psi_val = (
                            kth_largest_abs_val.item()
                        )  # Use k-th largest abs value

                        # Robbins-Monro update for lambda
                        lambda_val = (
                            1 - BETA_LAMBDA
                        ) * lambda_val + BETA_LAMBDA * psi_val
                        lambda_val = max(0.0, lambda_val)  # Ensure lambda >= 0
                    elif k <= 0:
                        # If k is 0 or less, target is fully sparse, lambda should grow large?
                        # Or maybe set lambda to a large value? For simplicity, let it stay 0.
                        pass  # Keep lambda_val as is or handle as needed
                    else:
                        # If vector is smaller than k, something is wrong or k is too large
                        # Maybe set psi_val to 0 or handle appropriately
                        pass  # Keep lambda_val as is

            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], "
                    f"Loss: {running_loss / 100:.4f}, Lambda: {lambda_val:.6f}"
                )
                running_loss = 0.0

        # --- End of Epoch Evaluation ---
        test_acc = evaluate(model, test_loader, DEVICE)
        sparsity_perc = calculate_sparsity(model)
        print("-" * 50)
        print(f"End of Epoch {epoch+1}/{EPOCHS}")
        print(f"Test Accuracy: {test_acc:.2f}%")
        print(f"Model Sparsity: {sparsity_perc:.2f}%")
        print(f"Final Lambda: {lambda_val:.6f}")
        print("-" * 50)

    print("Training finished.")
