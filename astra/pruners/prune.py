import torch
from torch import nn


def prune_with_wanda(activations, weights: nn.Parameter, sparsity_ratio: float):
    W = weights.clone().abs()
    X = activations
    out_features, in_features = W.shape

    # Reshape to [num_tokens, in_features]
    X = X.view(-1, X.shape[-1])

    act_norm = torch.norm(X, p=2, dim=0)

    # Wanda saliencies: |Weight| * ||Activation||
    pruning_scores = W * act_norm

    num_to_prune = int(in_features * sparsity_ratio)
    if num_to_prune == 0:
        print("Sparsity is too low to prune any weights. Skipping.")
        return

    # Sort the scores (independent for each output neuron)
    _, sorted_indices = torch.sort(pruning_scores, dim=1)

    # Create a pruning mask based on the lowest scores for each output neuron.
    pruning_mask = torch.zeros_like(weights.data).bool()
    for i in range(out_features):
        pruning_mask[i, sorted_indices[i, :num_to_prune]] = True

    # Apply the pruning mask by setting the selected weights to zero.
    pruned_weight = weights.data.clone()
    pruned_weight[pruning_mask] = 0.0
    with torch.no_grad():
        weights.data = pruned_weight
    return pruned_weight


def prune_with_sparsepgt(
    activations, weights: nn.Parameter, sparsity_ratio: float
):
    W = weights.clone().abs()
    X = activations
    out_features, in_features = W.shape

    # Reshape to [num_tokens, in_features]
    X = X.view(-1, X.shape[-1])

    act_norm = torch.norm(X, p=2, dim=0)

    # Wanda saliencies: |Weight| * ||Activation||
    pruning_scores = W * act_norm

    num_to_prune = int(in_features * sparsity_ratio)
    if num_to_prune == 0:
        print("Sparsity is too low to prune any weights. Skipping.")
        return

    # Sort the scores (independent for each output neuron)
    _, sorted_indices = torch.sort(pruning_scores, dim=1)

    # Create a pruning mask based on the lowest scores for each output neuron.
    pruning_mask = torch.zeros_like(weights.data).bool()
    for i in range(out_features):
        pruning_mask[i, sorted_indices[i, :num_to_prune]] = True

    # Apply the pruning mask by setting the selected weights to zero.
    pruned_weight = weights.data.clone()
    pruned_weight[pruning_mask] = 0.0
    with torch.no_grad():
        weights.data = pruned_weight

    W = weights.clone()
    X = activations
    X = X.view(-1, X.shape[-1])
    out_features, in_features = W.shape

    hessian = X.t().mm(X)

    identity = torch.eye(in_features, device=X.device, dtype=X.dtype)
    hessian_inv = torch.linalg.inv(hessian + 1e-8 * identity)

    # Keep track of which weights are active (not pruned)
    active_weights_mask = torch.ones(
        (out_features, in_features), dtype=torch.bool, device=X.device
    )
    pruned_indices = []

    # print(f"Starting OBS. Keeping {k} out of {in_features} features.\n")
    k = int(in_features * sparsity_ratio)

    # --- Step 3-5: Iteratively prune the least salient weights ---
    # pbar = tqdm(range(n_features - k), desc="Pruning weights")
    # for i in range(n_features - k):
    for i in range(in_features - k):
        active_indices = torch.where(active_weights_mask)[0]

        saliencies = (X[active_indices] ** 2) / (
            2 * torch.diag(hessian_inv)[active_indices]
        )

        # Find the weight with the minimum saliency among the active ones
        min_saliency_idx_local = torch.argmin(saliencies)
        prune_idx = active_indices[min_saliency_idx_local]

        # --- Step 4: Update the remaining weights ---
        # The update rule is: dx = - (x_q / [H^-1]_qq) * H^-1[:, q]
        x_q = W[prune_idx].clone()
        h_inv_qq = hessian_inv[prune_idx, prune_idx]

        # Get the q-th column of the inverse Hessian
        h_inv_col_q = hessian_inv[:, prune_idx]

        # Calculate the change for ALL weights
        delta_x = -(x_q / h_inv_qq) * h_inv_col_q.view_as(X)

        # Apply the update

        X += delta_x

        # --- Enforce the prune ---
        # The update rule automatically sets the pruned weight to zero,
        # but we do it explicitly to correct for any floating point inaccuracies.
        x[prune_idx] = 0.0
        active_weights_mask[prune_idx] = False
        pruned_indices.append(prune_idx.item())

        # --- Update the inverse Hessian (optional but more accurate) ---
        # The original OBS paper includes a step to update H^-1 to reflect the removal
        # of the parameter. This avoids re-inverting the matrix at each step.
        h_inv_row_q = hessian_inv[prune_idx, :]
        hessian_inv = hessian_inv - (1.0 / h_inv_qq) * torch.outer(
            h_inv_col_q, h_inv_row_q
        )
        # Zero out the row/column for the pruned weight for cleanliness
        hessian_inv[prune_idx, :] = 0
        hessian_inv[:, prune_idx] = 0

    # return x, sorted(pruned_indices)
    return x


def get_lsqr_gradient(v, X, y):
    return X.T @ (X @ v - y)


def accelerated_iht_algorithm(
    X,
    y,
    k,
    eta=0.01,
    mu=0.9,
    max_iter=5000,
    rtol=1e-6,
    warm_up=100,
    w_0=None,
):
    """
    Accelerated Iterative Hard Thresholding (IHT) algorithm for sparse recovery.

    Args:
        X: Input data matrix (numpy.ndarray).
        y: Target vector (numpy.ndarray).
        k: Number of non-zero entries to keep in the solution.
        eta: Step size for the gradient descent step.
        mu: Momentum parameter for acceleration.
        num_iter: Number of iterations to run the algorithm.

    Returns:
        w: Estimated sparse solution vector.
    """
    if w_0 is not None:
        if w_0.shape[0] != X.shape[1]:
            raise ValueError(
                f"w_0 shape {w_0.shape} does not match X shape {X[0].shape}"
            )
        w = w_0.clone()
        w_prev = w.clone()
    else:
        w = torch.zeros_like(X[0])
        w_prev = torch.zeros_like(w)

    pbar = tqdm(range(max_iter), desc="IHT")
    res = torch.mean((X @ w - y) ** 2)

    for i in pbar:
        # acceleration point
        v = w + mu * (w - w_prev)
        # gradient at the acceleration point
        grad = get_lsqr_gradient(v, X, y)
        # gradient descent step
        w_intermediate = v - eta * grad
        # saving w
        w_prev = w
        w = hard_threshold(w_intermediate, w_intermediate.abs(), k)
        new_res = torch.mean((X @ w - y) ** 2)

        if abs(new_res - res) < rtol * res and i > warm_up:
            print("converged: ", True)
            break
        res = new_res

        pbar.set_postfix(
            {
                "residual": res,
                "norm": torch.linalg.vector_norm(w).item(),
                "nnz": torch.count_nonzero(w).item(),
            }
        )

    return w


def accelerated_astra(
    X,
    y,
    alphas,
    k,
    eta=0.001,
    beta=0.2,
    max_iter=5000,
    rtol=1e-6,
    w_0=None,
    warm_up=5000,
    x_sol=None,
    recovery_lambda=1.0,
):
    t = 1
    t_prev = 0
    lambda_reg = 0.0

    if w_0 is not None:
        if w_0.shape[0] != X.shape[1]:
            raise ValueError(
                f"w_0 shape {w_0.shape} does not match X shape {X[0].shape}"
            )
        w = w_0.clone()
        w_prev = w.clone()
    else:
        w = torch.zeros_like(X[0])
        w_prev = torch.zeros_like(w)

    res = torch.mean((X @ w - y) ** 2)
    print("Initial residual at ASTRA:", res.item())

    pbar = tqdm(range(max_iter), desc="ASTRA")
    for i in pbar:
        y_k = w + ((t_prev - 1) / t) * (w - w_prev)
        w_prev = w
        gradient = get_lsqr_gradient(y_k, X, y)
        if x_sol is not None:
            gradient += recovery_lambda * (y_k - x_sol)
        w = soft_threshold(y_k - eta * gradient, lambda_reg * eta)
        v = (gradient - alphas * w).abs()
        v_k = kth_largest(v, k + 1)
        beta_t = beta / (1 + t) ** 0.75
        lambda_reg = (1 - beta_t) * lambda_reg + beta_t * v_k
        t_prev = t
        t = (1 + (1 + 4 * t**2) ** 0.5) / 2
        new_res = torch.mean((X @ w - y) ** 2)
        nnz = torch.count_nonzero(w).item()
        if (
            abs(new_res - res) < rtol * res
            and i > warm_up
            and nnz - k <= 1
            and nnz - k >= 0
        ):
            print("converged: ", True)
            break

        res = new_res
        pbar.set_postfix(
            {
                "error": res,
                "norm": torch.linalg.vector_norm(w).item(),
                "nnz": nnz,
            }
        )
    w = hard_threshold(w, w.abs(), k)
    return w


def accelerated_wanda_algorithm(
    X,
    y,
    alphas,
    k,
    eta=0.01,
    mu=0.9,
    max_iter=5000,
    rtol=1e-6,
    warm_up=100,
    w_0=None,
):
    """
    Accelerated Iterative Hard Thresholding (IHT) algorithm for sparse recovery.

    Args:
        X: Input data matrix (numpy.ndarray).
        y: Target vector (numpy.ndarray).
        k: Number of non-zero entries to keep in the solution.
        eta: Step size for the gradient descent step.
        mu: Momentum parameter for acceleration.
        num_iter: Number of iterations to run the algorithm.

    Returns:
        w: Estimated sparse solution vector.
    """
    if w_0 is not None:
        if w_0.shape != X[0].shape:
            raise ValueError(
                f"w_0 shape {w_0.shape} does not match X shape {X[0].shape}"
            )
        w = w_0.clone()
        w_prev = w.clone()
    else:
        w = torch.zeros_like(X[0])
        w_prev = torch.zeros_like(w)

    pbar = tqdm(range(max_iter), desc="IHT")
    res = torch.mean((X @ w - y) ** 2)

    for i in pbar:
        # acceleration point
        v = w + mu * (w - w_prev)
        # gradient at the acceleration point
        grad = get_lsqr_gradient(v, X, y)
        # gradient descent step
        w_intermediate = v - eta * grad
        # saving w
        w_prev = w
        w = hard_threshold(
            w_intermediate, (grad * v - alphas * (v**2)).abs(), k
        )
        new_res = torch.mean((X @ w - y) ** 2)

        if abs(new_res - res) < rtol * res and i > warm_up:
            print("converged: ", True)
            break
        res = new_res

        pbar.set_postfix(
            {
                "residual": res,
                "norm": torch.linalg.vector_norm(w).item(),
                "nnz": torch.count_nonzero(w).item(),
            }
        )

    return w


def obs_1d_pruner(
    A: torch.Tensor, b: torch.Tensor, x_lsqr: torch.Tensor, k: int
) -> torch.Tensor:
    """
    Prunes weights from the solution of a least squares problem ||Ax - b||^2
    using the Optimal Brain Surgeon algorithm.

    Args:
        A (torch.Tensor): The design matrix of shape (n_samples, n_features).
        b (torch.Tensor): The target vector of shape (n_samples, 1).
        k (int): The number of weights (features) to keep in the solution

    Returns:
        torch.Tensor: The pruned weight vector x of shape (n_features, 1).
    """
    n_samples, n_features = A.shape
    assert k > 0 and k < n_features, "k must be between 1 and n_features - 1."

    x = x_lsqr.clone()

    hessian = A.T @ A

    identity = torch.eye(n_features, device=A.device, dtype=A.dtype)
    hessian_inv = torch.linalg.inv(hessian + 1e-8 * identity)

    # Keep track of which weights are active (not pruned)
    active_weights_mask = torch.ones(
        n_features, dtype=torch.bool, device=A.device
    )
    pruned_indices = []

    print(f"Starting OBS. Keeping {k} out of {n_features} features.\n")

    # --- Step 3-5: Iteratively prune the least salient weights ---
    pbar = tqdm(range(n_features - k), desc="Pruning weights")
    # for i in range(n_features - k):
    for i in pbar:
        active_indices = torch.where(active_weights_mask)[0]

        saliencies = (x[active_indices] ** 2) / (
            2 * torch.diag(hessian_inv)[active_indices]
        )

        # Find the weight with the minimum saliency among the active ones
        min_saliency_idx_local = torch.argmin(saliencies)
        prune_idx = active_indices[min_saliency_idx_local]

        #     f"  -> Pruning weight at index {prune_idx.item()} (saliency: {saliencies[min_saliency_idx_local].item():.4e})"
        # )
        pbar.set_postfix(
            {
                "pruned_idx": prune_idx.item(),
                "saliency": saliencies[min_saliency_idx_local].item(),
            }
        )

        # --- Step 4: Update the remaining weights ---
        # The update rule is: dx = - (x_q / [H^-1]_qq) * H^-1[:, q]
        x_q = x[prune_idx].clone()
        h_inv_qq = hessian_inv[prune_idx, prune_idx]

        # Get the q-th column of the inverse Hessian
        h_inv_col_q = hessian_inv[:, prune_idx]

        # Calculate the change for ALL weights
        delta_x = -(x_q / h_inv_qq) * h_inv_col_q.view_as(x)

        # Apply the update
        x += delta_x

        # --- Enforce the prune ---
        # The update rule automatically sets the pruned weight to zero,
        # but we do it explicitly to correct for any floating point inaccuracies.
        x[prune_idx] = 0.0
        active_weights_mask[prune_idx] = False
        pruned_indices.append(prune_idx.item())

        # --- Update the inverse Hessian (optional but more accurate) ---
        # The original OBS paper includes a step to update H^-1 to reflect the removal
        # of the parameter. This avoids re-inverting the matrix at each step.
        h_inv_row_q = hessian_inv[prune_idx, :]
        hessian_inv = hessian_inv - (1.0 / h_inv_qq) * torch.outer(
            h_inv_col_q, h_inv_row_q
        )
        # Zero out the row/column for the pruned weight for cleanliness
        hessian_inv[prune_idx, :] = 0
        hessian_inv[:, prune_idx] = 0

    # return x, sorted(pruned_indices)
    return x


def accelerated_strong_astra(
    X,
    y,
    alphas,
    k,
    eta=0.01,
    beta=5e-3,
    max_iter=5000,
    rtol=1e-6,
    w_0=None,
    warm_up=5000,
    inner_iter=200,
    mu=0.9,
):
    t = 1
    t_prev = 0
    lambda_reg = 0.0

    if w_0 is not None:
        if w_0.shape[0] != X.shape[1]:
            raise ValueError(
                f"w_0 shape {w_0.shape} does not match X shape {X[0].shape}"
            )
        w = w_0.clone()
        w_prev = w.clone()
    else:
        w = torch.zeros_like(X[0])
        w_prev = torch.zeros_like(w)

    res = torch.mean((X @ w - y) ** 2)
    print("Initial residual at ASTRA:", res.item())
    pbar = tqdm(range(max_iter), desc="ASTRA")
    for i in pbar:
        w_curr = w.clone()
        w_curr_prev = w_prev.clone()
        t_curr = t
        t_curr_prev = t_prev

        for _ in range(inner_iter):
            y_k = w + ((t_prev - 1) / t) * (w - w_prev)
            w_prev = w
            gradient = get_lsqr_gradient(y_k, X, y) + eta * (y_k - w)
            v = (gradient - alphas * w).abs()
            v_k = kth_largest(v, k + 1)
            lambda_reg = (1 - beta) * lambda_reg + beta * v_k
            w = soft_threshold(y_k - eta * gradient, lambda_reg * eta)
            t_prev = t
            t = (1 + (1 + 4 * t**2) ** 0.5) / 2

        v = w_curr + mu * (w_curr - w_curr_prev)
        grad = get_lsqr_gradient(v, X, y)
        w_intermediate = v - eta * grad
        w = w_intermediate * (w.abs() > 0)
        w_prev = w_curr
        t = t_curr
        t_prev = t_curr_prev

        new_res = torch.mean((X @ w - y) ** 2)
        nnz = torch.count_nonzero(w).item()
        if (
            abs(new_res - res) < rtol * res
            and i > warm_up
            and nnz - k <= 1
            and nnz - k >= 0
        ):
            print("converged: ", True)
            break

        res = new_res
        pbar.set_postfix(
            {
                "error": res,
                "norm": torch.linalg.vector_norm(w).item(),
                "nnz": nnz,
            }
        )
    w = hard_threshold(w, w.abs(), k)
    return w


import numpy as np
from astra.np_utils import kth_largest
from sklearn.linear_model import lars_path


def compute_psi_k(A, b, x, k):
    """Computes psi(x) = |x - grad f(x)|_{(k)}"""
    n_features = A.shape[1]
    if k <= 0 or k > n_features:
        raise ValueError(
            f"k must be between 1 and n_features ({n_features}), got {k}"
        )

    if x.ndim == 1:
        grad_f_x = A.T @ (A @ x - b)
    else:
        grad_f_x = A.T @ (A @ x - b[:, None])
    w_x = x - grad_f_x
    abs_w_x = np.abs(w_x)

    if x.ndim == 1:
        return kth_largest(abs_w_x, k=k)
    return kth_largest(abs_w_x, k=k, axis=0)


def get_lambda_kkt(A, b, x):
    """Calculate the lambda corresponding to solution x using KKT conditions."""
    if x.ndim == 1:
        return np.max(np.abs(A.T @ (A @ x - b)))
    if x.ndim == 2:
        return np.max(np.abs(A.T @ (A @ x - b[:, None])), axis=0)
    raise ValueError(f"x has shape {x.shape}, only 1 or 2 dims are supported")


def compute_kinks(A, b):
    _, _, coefs = lars_path(
        A, b, method="lasso", verbose=False, return_path=True, Gram="auto"
    )
    lambda_kkt_j = get_lambda_kkt(A, b, coefs)
    return coefs, lambda_kkt_j


def find_fixed_point_interval(A, b, k):
    x, lambdas = compute_kinks(A, b)

    # Stores tuples of (lambda_kkt, g_value, original_coef_index)
    psis = compute_psi_k(A, b, x, k=k)

    gs = psis - lambdas

    return x, lambdas, psis, gs
