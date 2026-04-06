import torch
from torch.optim.optimizer import Optimizer
import warnings


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
        # loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        lambda_update_vectors = []
        found_grads = False

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            # eps = group["eps"]
            wd = group["weight_decay"]
            current_eta = (
                eta if eta is not None else group["lr"]
            )  # Use provided eta or default lr

            for p in group["params"]:
                if p.grad is None:
                    continue
                found_grads = True
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
                bias_correction1 = 1 - beta1 ** state["step"]
                # bias_correction2 = 1 - beta2 ** state["step"]

                if wd != 0:
                    grad = grad.add(p, alpha=wd)

                # Adam moment updates
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute m_hat (bias-corrected first moment)
                m_hat = exp_avg / bias_correction1

                # --- Heuristic Step ---
                # 1. Calculate term for lambda update: x_t - m_hat_t
                # Store original data if needed or calculate directly
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


# --- Variant 2: Using full Adam step direction Delta_t ---
class AdamH2_ProxSparse(Optimizer):
    """
    Heuristic Adam Variant 2 for Two-Timescale Sparse Optimization.

    Uses the full Adam update direction (Delta_t = m_hat / (sqrt(v_hat) + eps))
    within a Proximal step structure.

    WARNING: This is a highly heuristic approach. It deviates significantly
    from proximal gradient theory and lacks theoretical justification. Use
    with extreme caution.
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
            "AdamH2_ProxSparse is a highly heuristic optimizer lacking theoretical justification.",
            UserWarning,
        )

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdamH2_ProxSparse, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, lambda_val, alpha=None, closure=None):
        """
        Performs a single optimization step.

        Args:
            lambda_val (float): The current lambda value for L1 penalty.
            alpha (float, optional): The current step size (learning rate) for the
                                     Adam-like step. If None, uses the
                                     optimizer's default lr.
            closure (callable, optional): A closure that reevaluates the model
                                       and returns the loss.
        Returns:
            torch.Tensor: A flattened tensor containing the vectors
                          (x_t - Delta_t) for all parameters, needed for the
                          external lambda update calculation |x - Delta|_{(k)}.
                          Returns None if no gradients were found.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        lambda_update_vectors = []
        found_grads = False

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            current_alpha = (
                alpha if alpha is not None else group["lr"]
            )  # Use provided alpha or default lr

            for p in group["params"]:
                if p.grad is None:
                    continue
                found_grads = True
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
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                if wd != 0:
                    grad = grad.add(p, alpha=wd)

                # Adam moment updates
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute Adam step direction Delta_t
                m_hat = exp_avg / bias_correction1
                v_hat = exp_avg_sq / bias_correction2
                denom = v_hat.sqrt().add_(eps)
                delta_t = m_hat / denom

                # --- Heuristic Step ---
                # 1. Calculate term for lambda update: x_t - Delta_t
                lambda_update_vector = (p.data - delta_t).detach().flatten()
                lambda_update_vectors.append(lambda_update_vector)

                # 2. Compute step using Delta_t: y = x_t - alpha * Delta_t
                y = p.data - current_alpha * delta_t

                # 3. Apply proximal operator (soft-thresholding)
                threshold = (
                    current_alpha * lambda_val
                )  # Note: Threshold depends on alpha
                p.data = soft_threshold(y, threshold)
                # ----------------------

        if not found_grads:
            return None

        # Concatenate all vectors for external processing
        full_lambda_update_vector = torch.cat(lambda_update_vectors)
        return full_lambda_update_vector  # Return the vector for lambda update


# --- Example Usage (Conceptual) ---
if __name__ == "__main__":

    # Assume model, criterion, train_loader are defined
    model = torch.nn.Linear(100, 10)  # Example model
    params = model.parameters()
    k = 5  # Target sparsity

    # --- Choose Optimizer ---
    # optimizer = AdamH1_ProxSparse(params, lr=0.01) # Variant 1
    optimizer = AdamH2_ProxSparse(params, lr=0.001)  # Variant 2

    lambda_val = 0.0  # Initial lambda
    beta_t = 0.01  # Example constant beta_t, likely needs decay
    num_epochs = 10
    T = 1  # Update lambda every T steps
    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        for i, (data, target) in enumerate([]):  # Replace with your data loader
            global_step += 1
            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.functional.mse_loss(
                output, target
            )  # Replace with your criterion
            loss.backward()

            # --- Two-Timescale Update ---
            # 1. Optimizer step (x-update + prox)
            # Manage decaying eta/alpha externally if needed
            current_step_size = optimizer.defaults["lr"]  # Or decay schedule
            lambda_update_vec = optimizer.step(
                lambda_val=lambda_val, eta=current_step_size
            )  # or alpha=... for AdamH2

            # 2. Lambda update (external)
            if lambda_update_vec is not None and global_step % T == 0:
                with torch.no_grad():
                    # Compute |vector|_{(k)}
                    kth_largest_abs = torch.kthvalue(
                        lambda_update_vec.abs(),
                        max(1, lambda_update_vec.numel() - k + 1),
                    ).values  # k-th smallest ~ (N-k+1)-th largest
                    psi_val = (
                        kth_largest_abs  # Use k-th largest *absolute* value
                    )

                    # Robbins-Monro update for lambda
                    # Manage decaying beta_t externally if needed
                    lambda_val = (1 - beta_t) * lambda_val + beta_t * psi_val
                    lambda_val = max(0.0, lambda_val)  # Ensure lambda >= 0

            # --- Logging ---
            if i % 100 == 0:
                print(
                    f"Epoch {epoch}, Step {i}, Loss: {loss.item():.4f}, Lambda: {lambda_val:.4f}"
                )

                # Optional: Check sparsity
                total_params = 0
                zero_params = 0
                for p in model.parameters():
                    if p.requires_grad:
                        total_params += p.numel()
                        zero_params += (p.data == 0).sum().item()
                print(f"Sparsity: {zero_params / total_params * 100:.2f}%")
