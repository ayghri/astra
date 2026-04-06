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
# DTYPE = torch.float16
DTYPE = torch.float32

torch.set_float32_matmul_precision('high') 


def compute_H(X, batch_size=4096):
    N, K = X.shape
    H = torch.zeros(K, K, device=DEVICE, dtype=torch.float32)
    for i in range(0, N, batch_size):
        X_b = X[i : i + batch_size].to(device=DEVICE, dtype=torch.float32)
        H.addmm_(X_b.T, X_b)
    H /= N
    return H


def compute_loss(W_quant, W0, H, N, chunk=128):
    M = W0.shape[0]
    total = 0.0
    for c0 in range(0, M, chunk):
        dW = W_quant[c0 : c0 + chunk] - W0[c0 : c0 + chunk]
        total += ((dW @ H) * dW).sum().item()
    return total * N


def get_rho(H):
    rho = H.diag().diag()
    # return (torch.ones_like(H.diag()) * 1e-2).diag()
    return rho


def main():
    W0 = torch.load(W_PATH, map_location=DEVICE, weights_only=True).float()
    # W0 = torch.load(W_PATH, map_location=DEVICE, weights_only=True)
    X_cpu = torch.load(X_PATH, map_location="cpu", weights_only=True)
    M, K = W0.shape
    N = X_cpu.shape[0]
    progress(f"  W: {W0.shape}, X: {X_cpu.shape}")

    progress("Computing H...")
    H = compute_H(X_cpu)

    # mat_A = torch.linalg.inv(H + rho)
    rho = get_rho(H)
    mat_A = torch.linalg.solve(H + rho, torch.eye(K).to(H))
    mat_C = W0.mm(H).mm(mat_A)

    mat_A = mat_A.to(DTYPE)
    mat_C = mat_C.to(DTYPE)
    rho = get_rho(H).to(DTYPE)
    H_diag = H.diag().diag().to(DTYPE)

    # Z = View.from_existing(W0.clone())
    Z = View.from_existing(torch.zeros_like(W0).to(DTYPE))
    W = Z.param.clone()
    U = W - Z.param
    rho_cond = torch.ones_like(W) @ rho

    # g_spec = ScopeSpec(b_spec, (-1, -1), "GZ")
    # b_spec = BlockSpec(Z, (-1, 1), "BZ")
    b_spec = BlockSpec(Z, (1, 1), "BZ")
    g_spec = ScopeSpec(b_spec, (1, 4), "GZ")
    print(b_spec)
    print(g_spec)

    lamb = torch.zeros_like(g_spec.block_norms(None).sum(dim=-1)).float()
    lamb_update = 1
    # restarts = 500
    log_step = 20

    progress("Running ADMM...")
    for i in range(NUM_ITER):
        W = mat_C + (Z.param - U) @ rho @ mat_A
        Z.param.copy_(W + U)
        phi_quant = 0

        if (i + 1) % lamb_update == 0:
            # grad = (W - W0) @ H
            # grad_diag = W @ H.diag().diag()
            # psi_pre = (grad - grad_diag).abs()
            # phi = g_spec.kth_largest({b_spec: psi_pre}, nnz=2)
            psi_pre = (W - W0) @ H
            psi_pre.add_(-W @ H_diag)
            # phi += 99.0 * g_spec.kth_largest({b_spec: psi_pre}, nnz=3)
            # phi = phi / 100.0
            phi = g_spec.kth_mid({b_spec: psi_pre}, nnz=KAPPA, k_weight=K_VAL_WEIGHT)

            # phi = g_spec.kth_largest({b_spec: psi_pre}, nnz=3)
            lamb.mul_(PSI_BETA)
            lamb.add_((1 - PSI_BETA) * phi)

        z_clone = Z.param.clone()
        g_spec.soft_threshold(lamb, conditioners=rho_cond)
        # m = g_spec.get_masks(nnz=2, )[b_spec].to(Z.param)

        m = g_spec.get_masks(nnz=2, values={b_spec: psi_pre})[b_spec].to(Z.param)

        # Z.param.mul_(1 - m)
        # Z.param.addcmul_(z_clone, m)
        # Z.param.lerp_(z_clone, m)
        Z.param.copy_((1 - m) * Z.param + m * z_clone)

        U.add_(W - Z.param)

        if (i % log_step) == 0:
            if (i + 1) > lamb_update:
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
            # lamb.mul_(0.98)

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
