import warnings
import torch
from torch.optim.optimizer import Optimizer
import tests.old.utils as utils


class AdamH2_ProxSparse_PerParam_SkipFirst(Optimizer):
    """
    Heuristic Adam Variant 2 for Two-Timescale Sparse Optimization.
    Uses the full Adam step direction Delta_t = m_hat / (sqrt(v_hat)+eps).
    Maintains and updates lambda PER PARAMETER tensor.
    SKIPS soft-thresholding for bias parameters (dim==1) AND
    parameters belonging to the specified first_layer_params set.

    WARNING: Highly heuristic approach, deviates significantly from theory.
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
            "AdamH2_ProxSparse_PerParam_SkipFirst is a highly heuristic optimizer lacking theoretical justification.",
            UserWarning,
        )

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            initial_lambda=initial_lambda,
        )
        super(AdamH2_ProxSparse_PerParam_SkipFirst, self).__init__(
            params, defaults
        )

    @torch.no_grad()
    def step(
        self,
        k_values_dict,
        beta_lambda,
        first_layer_params_set,
        alpha_dict=None,
        closure=None,
    ):
        """
        Performs step using full Adam update, skipping prox for bias and first layer.

        Args:
            k_values_dict (dict): Maps WEIGHT param id (int) -> target k (int)
                                  (MUST exclude first layer params).
            beta_lambda (float): Robbins-Monro step size for lambda updates.
            first_layer_params_set (set): A set containing the parameter objects
                                         (p) belonging to the first layer.
            alpha_dict (dict, optional): Maps group index (int) -> step size alpha (float).
                                        If None, uses group's default lr.
            closure (callable, optional): A closure that reevaluates the model.
        Returns:
            Loss (optional).
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        _active_device = None

        for group_idx, group in enumerate(self.param_groups):
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            initial_lambda = group["initial_lambda"]
            # Use lr as the alpha step size unless overridden
            current_alpha = group["lr"]
            if alpha_dict is not None and group_idx in alpha_dict:
                current_alpha = alpha_dict[group_idx]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if _active_device is None:
                    _active_device = p.grad.device

                is_bias = p.dim() == 1
                is_first_layer = p in first_layer_params_set
                skip_prox = is_bias or is_first_layer

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # state["exp_avg_sq"] = torch.zeros_like(
                    #     p, memory_format=torch.preserve_format
                    # )
                    state["exp_avg_sq"] = torch.tensor(0.0).to(p)
                    state["lambda_val"] = float(initial_lambda)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                step_tensor = torch.tensor(
                    state["step"], dtype=torch.float32, device=_active_device
                )
                bias_correction1 = 1 - beta1**step_tensor
                bias_correction2 = 1 - beta2**step_tensor

                if wd != 0 and not is_bias:  # Apply WD only to non-bias weights
                    grad = grad.add(p, alpha=wd)
                # grad_mean = grad.mean()

                # Adam moment updates
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).add_(
                    torch.linalg.vector_norm(grad) ** 2, alpha=1 - beta2
                )

                # --- Calculate Full Adam Step Direction Delta_t ---
                m_hat = exp_avg / (bias_correction1 + eps)
                v_hat = exp_avg_sq / (
                    bias_correction2 + eps
                )  # Add eps here too for v_hat
                denom = v_hat.sqrt().add_(eps)
                delta_t = m_hat / denom
                # --- ---

                # Full Adam step component for update
                adam_step_component = current_alpha * delta_t

                # --- Heuristic Update ---
                if skip_prox:
                    # Perform standard Adam update (using full step component)
                    p.data.sub_(adam_step_component)
                else:
                    # For WEIGHT parameters in OTHER layers: update lambda and apply prox step
                    # 1. Calculate psi_vector = x_t - Delta_t (Use full step direction)
                    psi_vector_p = (
                        p.data - delta_t
                    ).detach()  # Detach is important

                    # 2. Get k for this parameter
                    param_id = id(p)
                    k_p = k_values_dict.get(param_id, 0)

                    # 3. Update lambda only if k_p > 0
                    if k_p > 0:
                        psi_val_p = utils.calculate_kth_value(
                            psi_vector_p, k_p
                        )
                        current_lambda_p = state["lambda_val"]
                        updated_lambda_p = (
                            1 - beta_lambda
                        ) * current_lambda_p + beta_lambda * psi_val_p
                        state["lambda_val"] = max(0.0, updated_lambda_p)

                    # 4. Compute tentative update before prox: y = x_t - alpha * Delta_t
                    y = p.data - adam_step_component

                    # 5. Apply proximal operator using the UPDATED parameter lambda
                    #    Threshold depends on alpha (Adam LR) here!
                    threshold = current_alpha * state["lambda_val"]
                    p.data = utils.soft_threshold(y, threshold)

        return loss
