"""ADMM pruning for N:M structured sparsity.

Supports two modes:
  - Standard: minimize ||(W - W0) X||^2 under sparsity
  - Error-correcting: minimize ||W X_pruned - W0 X_dense||^2 under sparsity
"""

import torch
from torch import Tensor
from sparsekit import BlockSpec, ScopeSpec, View
from tqdm import tqdm


def compute_H(X: Tensor, device: torch.device, batch_size: int = 4096) -> Tensor:
    """Compute Hessian H = X^T X / N."""
    N, K = X.shape
    H = torch.zeros(K, K, device=device, dtype=torch.float32)
    for i in range(0, N, batch_size):
        X_b = X[i : i + batch_size].to(device=device, dtype=torch.float32)
        H.addmm_(X_b.T, X_b)
    H /= N
    return H


def compute_cross_H(
    X_dense: Tensor, X_pruned: Tensor, device: torch.device, batch_size: int = 4096
) -> Tensor:
    """Compute cross-Hessian X_dense^T @ X_pruned / N."""
    N, K = X_pruned.shape
    CH = torch.zeros(K, K, device=device, dtype=torch.float32)
    for i in range(0, N, batch_size):
        Xd = X_dense[i : i + batch_size].to(device=device, dtype=torch.float32)
        Xp = X_pruned[i : i + batch_size].to(device=device, dtype=torch.float32)
        CH.addmm_(Xd.T, Xp)
    CH /= N
    return CH


def admm_prune(
    W0: Tensor,
    H: Tensor,
    C_target: Tensor | None = None,
    num_iter: int = 1000,
    kappa: int = 2,
    group_size: int = 4,
    psi_beta: float = 0.99,
    k_weight: float = 2999.0,
    percdamp: float = 0.01,
    verbose: bool = False,
) -> Tensor:
    """ADMM pruning for N:M structured sparsity.

    Diagonal matrix ops are replaced with element-wise for efficiency.
    Uses pseudo-inverse with diagonal damping for numerical stability.

    Args:
        W0: [M, K] original weight.
        H: [K, K] Hessian (X^T X / N).
        C_target: [M, K] target correlation. None = W0 @ H (standard mode).
        num_iter: ADMM iterations.
        kappa: nonzeros to keep per group.
        group_size: group size for N:M sparsity.
        psi_beta: EMA decay for adaptive lambda.
        k_weight: weight for kth_mid scoring.
        percdamp: diagonal damping as fraction of mean diagonal (0 to disable).
        verbose: show progress bar.

    Returns:
        [M, K] pruned weight (float32).
    """
    device = H.device
    M, K = W0.shape
    W0f = W0.float().to(device)
    H = H.clone()

    # Dead column handling + damping (same as SparseGPT)
    dead = H.diag() == 0
    H[dead, dead] = 1
    W0f[:, dead] = 0
    if percdamp > 0:
        H.diagonal().add_(percdamp * H.diag().mean())

    if C_target is None:
        C_target = W0f @ H
    else:
        C_target = C_target.float().to(device)

    # Precompute (fp32)
    rho_diag = H.diag()  # [K] — replaces full [K,K] diagonal matrix
    mat_A = torch.cholesky_inverse(torch.linalg.cholesky(H + rho_diag.diag()))  # [K,K]
    mat_C = C_target.mm(mat_A)  # [M, K]

    # Working variables (fp32)
    Z = View.from_existing(torch.zeros(M, K, device=device, dtype=torch.float32))
    W = Z.param.clone()
    U = torch.zeros_like(W)
    rho_cond = rho_diag.unsqueeze(0).expand(M, K)  # [M,K] — was ones @ diag_matrix

    b_spec = BlockSpec(Z, (1, 1), "BZ")
    g_spec = ScopeSpec(b_spec, (1, group_size), "GZ")

    lamb = torch.zeros_like(g_spec.block_norms(None).sum(dim=-1)).float()

    rng = range(num_iter)
    if verbose:
        rng = tqdm(rng)

    for _ in rng:
        # W update: mat_C + (Z - U) * rho_diag @ mat_A
        #   (Z-U) @ diag(rho) is element-wise, then one real matmul
        diff_rho = (Z.param - U) * rho_diag  # [M,K] element-wise
        W = mat_C + diff_rho.mm(mat_A)       # one matmul instead of two

        Z.param.copy_(W + U)

        # Gradient with diagonal removed: W @ H - C_target - W * diag(H)
        psi_pre = W.mm(H) - C_target         # one matmul
        psi_pre.sub_(W * rho_diag)            # element-wise (was W @ diag_matrix)

        phi = g_spec.kth_mid({b_spec: psi_pre}, nnz=kappa, k_weight=k_weight)
        lamb.mul_(psi_beta)
        lamb.add_((1 - psi_beta) * phi)

        z_clone = Z.param.clone()
        g_spec.soft_threshold(lamb, conditioners=rho_cond)
        m = g_spec.get_masks(nnz=kappa)[b_spec].to(Z.param)
        Z.param.copy_((1 - m) * Z.param + m * z_clone)

        U.add_(W - Z.param)

    m = g_spec.get_masks(nnz=kappa)
    return Z.param.mul(m[b_spec]).float()
