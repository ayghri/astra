import torch
from tqdm import tqdm

from sparsekit import BlockSpec, ScopeSpec
from sparsekit import View


def progress(msg):
    print(msg, flush=True)


DEVICE = torch.device("cuda:0")
W_PATH = "/buckets/checkpoints/layer_0_W.cpt"
X_PATH = "/buckets/checkpoints/layer_0_X.cpt"

PSI_BETA = 1.0 - 1e-2
NUM_ITER = 1_600
K_VAL_WEIGHT = 2999.0
KAPPA = 2
MM_DTYPE = torch.float16  # dtype for matmul inputs (fp16 uses tensor cores)


def compute_H(X, batch_size=4096):
    N, K = X.shape
    H = torch.zeros(K, K, device=DEVICE, dtype=torch.float32)
    for i in range(0, N, batch_size):
        X_b = X[i : i + batch_size].to(device=DEVICE, dtype=torch.float32)
        H.addmm_(X_b.T, X_b)
    H /= N
    return H


def mm16(A, B):
    """Matmul with fp16 tensor cores, fp32 result."""
    return torch.mm(A.to(MM_DTYPE), B.to(MM_DTYPE)).float()


def compute_loss(W_quant, W0, H, N, chunk=128):
    M = W0.shape[0]
    total = 0.0
    for c0 in range(0, M, chunk):
        dW = W_quant[c0 : c0 + chunk] - W0[c0 : c0 + chunk]
        total += ((dW @ H) * dW).sum().item()
    return total * N


def main():
    W0 = torch.load(W_PATH, map_location=DEVICE, weights_only=True).float()
    X_cpu = torch.load(X_PATH, map_location="cpu", weights_only=True)
    M, K = W0.shape
    N = X_cpu.shape[0]
    progress(f"  W: {W0.shape}, X: {X_cpu.shape}")

    progress("Computing H...")
    H = compute_H(X_cpu)

    # Precompute (fp32 for accuracy)
    rho_diag = H.diag()  # [K] — diagonal of H
    mat_A = torch.linalg.solve(H + rho_diag.diag(), torch.eye(K).to(H))  # [K,K]
    mat_C = W0.mm(H).mm(mat_A)  # [M,K]
    # rho_A = diag(rho) @ mat_A, precomputed to fuse two ops into one matmul
    rho_A = rho_diag.unsqueeze(1) * mat_A  # [K,K]

    # fp16 copies for matmul inputs (kept in fp16 permanently)
    rho_A_h = rho_A.to(MM_DTYPE)  # [K,K]
    H_h = H.to(MM_DTYPE)          # [K,K]

    # Working variables (fp32 for stability across iterations)
    Z = View.from_existing(torch.zeros(M, K, device=DEVICE, dtype=torch.float32))
    W = Z.param.clone()
    U = torch.zeros_like(W)
    rho_cond = rho_diag.unsqueeze(0).expand(M, K)  # [M,K] — was ones @ diag_matrix

    b_spec = BlockSpec(Z, (1, 1), "BZ")
    g_spec = ScopeSpec(b_spec, (1, 4), "GZ")
    print(b_spec)
    print(g_spec)

    lamb = torch.zeros_like(g_spec.block_norms(None).sum(dim=-1)).float()
    C_target = W0 @ H  # [M,K] for gradient computation
    log_step = 20

    progress("Running ADMM...")
    for i in range(NUM_ITER):
        # W update: mat_C + (Z - U) * rho_diag @ mat_A
        #   (Z-U) @ diag(rho) is element-wise, then one matmul with mat_A
        diff_rho = (Z.param - U) * rho_diag  # [M,K] element-wise
        W = mat_C + mm16(diff_rho, mat_A)    # fp16 matmul, fp32 result

        Z.param.copy_(W + U)
        phi_quant = 0

        # Gradient: (W - W0) @ H - W * diag(H)
        #   = W @ H - C_target - W * rho_diag
        psi_pre = mm16(W, H) - C_target      # fp16 matmul, fp32 subtract
        psi_pre.sub_(W * rho_diag)            # element-wise (was W @ diag_matrix)

        phi = g_spec.kth_mid({b_spec: psi_pre}, nnz=KAPPA, k_weight=K_VAL_WEIGHT)
        lamb.mul_(PSI_BETA)
        lamb.add_((1 - PSI_BETA) * phi)

        z_clone = Z.param.clone()
        g_spec.soft_threshold(lamb, conditioners=rho_cond)
        m = g_spec.get_masks(nnz=2)[b_spec].to(Z.param)
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
                phi_quant,
            )
            m = g_spec.get_masks(nnz=2)
            Z_sol = Z.param.mul(m[b_spec])
            W_star = Z_sol.float()
            dW = W_star - W0
            loss = ((dW @ H) * dW).sum().item()
            progress(f"Loss: {loss * N}")

    m = g_spec.get_masks(nnz=2)
    Z_sol = Z.param.mul(m[b_spec])

    torch.backends.cuda.preferred_linalg_library("cusolver")
    progress("Refining solution row-wise (batched)...")
    W_star = Z_sol
    C = W0 @ H
    mask = m[b_spec]  # [M, K]
    batch_size = 32

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
        sol = torch.linalg.solve(H_batch.float(), rhs_batch.float())  # [B, nnz_count]
        W_star[j0:j1].scatter_(1, indices, sol)

    dW = W_star - W0
    loss = ((dW @ H) * dW).sum().item()
    progress(f"Loss: {loss * N}")


if __name__ == "__main__":
    main()
