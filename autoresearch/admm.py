import torch
from tqdm import tqdm
import time

from sparsekit import BlockSpec, ScopeSpec
from sparsekit import View

_FP16_MAX = 65504.0


def progress(msg):
    print(msg, flush=True)


DEVICE = torch.device("cuda:0")
W_PATH = "/buckets/checkpoints/layer_0_W.cpt"
X_PATH = "/buckets/checkpoints/layer_0_X.cpt"

# PSI_BETA = 1.0 - 2e-2
PSI_BETA = 0.98
NUM_ITER = 600
K_VAL_WEIGHT = 9999.0
KAPPA = 2
# PERCDAMP = 0.001
PERCDAMP = 1e-5
MAX_PSI = 0.0002
# MAX_PSI = 0.0001648


def compute_H(X, batch_size=4096):
    N, K = X.shape
    H = torch.zeros(K, K, device=DEVICE, dtype=torch.float32)
    for i in range(0, N, batch_size):
        X_b = X[i : i + batch_size].to(device=DEVICE, dtype=torch.float32)
        H.addmm_(X_b.T, X_b)
    H /= N
    return H


def mm16(A, B_h):
    """Matmul: clamp A to fp16 range, cast to fp16, B_h already fp16. Returns fp32."""
    return torch.mm(A.clamp(-_FP16_MAX, _FP16_MAX).half(), B_h).float()


def compute_loss(W_quant, W0, H, N, chunk=128):
    M = W0.shape[0]
    total = 0.0
    for c0 in range(0, M, chunk):
        dW = W_quant[c0 : c0 + chunk] - W0[c0 : c0 + chunk]
        total += ((dW @ H) * dW).sum().item()
    return total * N


def main(skip_ols=True):
    W0 = torch.load(W_PATH, map_location=DEVICE, weights_only=True).float()
    X_cpu = torch.load(X_PATH, map_location="cpu", weights_only=True)
    M, K = W0.shape
    N = X_cpu.shape[0]
    progress(f"  W: {W0.shape}, X: {X_cpu.shape}")

    progress("Computing H...")
    H = compute_H(X_cpu)
    best_loss = float("inf")
    best_Z = torch.zeros_like(W0)

    # Dead column handling + damping (same as SparseGPT / astra pruner)
    dead = H.diag() == 0
    H[dead, dead] = 1
    # W0[:, dead] = 0
    if PERCDAMP > 0:
        H.diagonal().add_(PERCDAMP * H.diag().mean())

    start = time.time()
    C_target = W0 @ H  # [M,K]

    # Precompute (fp32 for accuracy, cholesky for stability)
    rho_diag = H.diag()  # [K]
    try:
        mat_A = torch.cholesky_inverse(
            torch.linalg.cholesky(H + rho_diag.diag())
        )
    except torch.linalg.LinAlgError:
        mat_A = torch.linalg.solve(
            H + rho_diag.diag(), torch.eye(K, device=DEVICE)
        )
    mat_C = C_target.mm(mat_A)  # [M,K]

    # Fused matrices for fp16 loop (avoids large intermediates):
    #   rho_A = diag(rho) @ mat_A — absorbs large rho into mat_A
    #   H_offdiag = H - diag(rho) — for gradient without diagonal
    rho_A = rho_diag.unsqueeze(1) * mat_A  # [K,K]
    H_offdiag = H - rho_diag.diag()  # [K,K]

    rho_A_h = rho_A.clamp(-_FP16_MAX, _FP16_MAX).half()
    H_offdiag_h = H_offdiag.clamp(-_FP16_MAX, _FP16_MAX).half()

    # Working variables (fp32 for stability across iterations)
    # Z = View.from_existing(torch.zeros(M, K, device=DEVICE, dtype=torch.float32))
    Z = View.from_existing(W0.clone())  # device=DEVICE, dtype=torch.float32))
    W = Z.param.clone()
    U = torch.zeros_like(W)
    rho_cond = rho_diag.unsqueeze(0).expand(M, K)

    b_spec = BlockSpec(Z, (1, 1), "BZ")
    g_spec = ScopeSpec(b_spec, (1, 4), "GZ")
    print(b_spec)
    print(g_spec)

    lamb = torch.zeros_like(g_spec.block_norms(None).sum(dim=-1)).float()
    log_step = 20

    progress("Running ADMM...")
    for i in range(NUM_ITER):
        # W update: mat_C + (Z - U) @ rho_A
        #   (Z-U) is small, rho absorbed into rho_A — no overflow
        diff = Z.param - U
        W = mat_C + mm16(diff, rho_A_h)

        Z.param.copy_(W + U)
        phi_quant = 0

        # Gradient: W @ H - C_target - W * rho_diag
        #         = W @ (H - diag(rho)) - C_target = W @ H_offdiag - C_target

        # V = W + U
        # E_val = V.abs() * torch.sqrt(rho_diag.unsqueeze(0))
        # phi = g_spec.kth_mid({b_spec: E_val}, nnz=KAPPA, k_weight=K_VAL_WEIGHT)
        # lamb.mul_(PSI_BETA)
        # lamb.add_((1 - PSI_BETA) * phi)

        # psi_pre = mm16(W, H_offdiag_h) - C_target

        psi_pre = (W + U) * rho_diag.unsqueeze(0)
        psi_pre.abs_().clamp_(max=MAX_PSI)

        phi = g_spec.kth_mid(
            {b_spec: psi_pre}, nnz=KAPPA, k_weight=K_VAL_WEIGHT
        )
        lamb.mul_(PSI_BETA)
        lamb.add_((1 - PSI_BETA) * phi)

        z_clone = Z.param.clone()
        # g_spec.soft_threshold(lamb, conditioners=rho_cond)
        # m = g_spec.get_masks(nnz=2)[b_spec].to(Z.param)
        g_spec.soft_threshold(lamb, conditioners=rho_cond)
        # g_spec.soft_threshold(lamb, conditioners=E_val)
        # m = g_spec.get_masks(values={b_spec: E_val}, nnz=KAPPA)[b_spec].to(Z.param)

        # psi_pre = mm16(W, H_offdiag_h) - C_target
        m = g_spec.get_masks(nnz=2)[b_spec].to(Z.param)
        # m = g_spec.get_masks(values={b_spec:z_clone},nnz=2)[b_spec].to(Z.param)
        # m = g_spec.get_masks(values={b_spec: (W + U) * rho_diag.unsqueeze(0)}, nnz=KAPPA)[b_spec].to(Z.param)
        Z.param.copy_((1 - m) * Z.param + m * z_clone)

        U.add_(W - Z.param)

        if (i % log_step) == 0:
            if i > 0:
                phi_quant = phi.float().quantile(0.95).item()
            vals, counts = torch.unique(
                (g_spec.block_norms(None) > 1e-4).sum(-1), return_counts=True
            )
            vals = [int(v) for v in vals.cpu().numpy()]
            counts = [int(c) for c in counts.cpu().numpy()]
            print(
                i,
                dict(zip(vals, counts)),
                ((Z.data - W0) @ H @ (Z.data - W0).T).trace().item(),
                (W - Z.data).norm().item(),
                lamb.float().quantile(0.95).item(),
                lamb.max().item(),
                phi_quant,
            )
            m = g_spec.get_masks(nnz=2)
            Z_sol = Z.param.mul(m[b_spec])
            W_star = Z_sol.float()
            dW = W_star - W0
            loss = ((dW @ H) * dW).sum().item()
            if loss < best_loss:
                best_loss = loss
                best_Z = W_star
            progress(f"Loss: {loss * N}")

    # m = g_spec.get_masks(nnz=2)
    # Z_sol = Z.param.mul(m[b_spec])

    torch.backends.cuda.preferred_linalg_library("cusolver")
    progress("Refining solution row-wise (batched)...")
    # W_star = Z_sol
    # Z_sol = best_Z

    W_star = best_Z
    m = g_spec.get_masks(
        values={b_spec: best_Z},
        nnz=2,
    )

    C = W0 @ H
    mask = m[b_spec]  # [M, K]
    batch_size = 32

    if not skip_ols:
        for j0 in tqdm(range(0, W_star.shape[0], batch_size)):
            j1 = min(j0 + batch_size, W_star.shape[0])
            B = j1 - j0

            batch_mask = mask[j0:j1]  # [B, K]
            nz = batch_mask.nonzero(as_tuple=False)  # [B * nnz_count, 2]
            nnz_count = nz.shape[0] // B
            indices = nz[:, 1].reshape(B, nnz_count)  # [B, nnz_count]

            H_batch = H[
                indices[:, :, None], indices[:, None, :]
            ]  # [B, nnz_count, nnz_count]
            rhs_batch = C[j0:j1].gather(1, indices)  # [B, nnz_count]
            sol = torch.linalg.solve(
                H_batch.float(), rhs_batch.float()
            )  # [B, nnz_count]
            W_star[j0:j1].scatter_(1, indices, sol)
    print(f"The whole thing took {time.time()-start}s")

    dW = W_star - W0
    loss = ((dW @ H) * dW).sum().item()
    progress(f"Loss: {loss * N}")


if __name__ == "__main__":
    main()
