import torch
from tqdm import tqdm

from sparsekit import BlockSpec, ScopeSpec
from sparsekit import View


def progress(msg):
    print(msg, flush=True)


DEVICE = torch.device("cuda:1")
W_PATH = "/buckets/checkpoints/layer_0_W.cpt"
X_PATH = "/buckets/checkpoints/layer_0_X.cpt"

PSI_BETA = 1.0 - 1e-2
NUM_ITER = 1000


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
    X_cpu = torch.load(X_PATH, map_location="cpu", weights_only=True)
    M, K = W0.shape
    N = X_cpu.shape[0]
    progress(f"  W: {W0.shape}, X: {X_cpu.shape}")

    progress("Computing H...")
    H = compute_H(X_cpu)

    # mat_A = torch.linalg.inv(H + rho)
    mat_A = torch.linalg.solve(H, torch.eye(K).to(H))
    # mat_C = W0.mm(H).mm(mat_A)

    W = View.from_existing(W0.clone())
    b_spec = BlockSpec(W, (1, 1), "BZ")
    g_spec = ScopeSpec(b_spec, (1, 4), "GZ")

    log_step = 20

    progress("Running ADMM...")
    for i in range(NUM_ITER):
        # W = mat_C + (Z.param - U) @ rho @ mat_A
        # Z.param.copy_(W + U)

        grad = (W - W0) @ H
        W = W - grad
        # grad_diag = W @ H.diag().diag()
        # psi_pre = (grad - grad_diag).abs()
        # phi = g_spec.kth_largest({b_spec: psi_pre}, nnz=2)
        # phi += 99.0 * g_spec.kth_largest({b_spec: psi_pre}, nnz=3)
        # phi = phi / 100.0
        # phi = g_spec.kth_largest({b_spec: psi_pre}, nnz=3)
        # lamb.mul_(PSI_BETA)
        # lamb.add_((1 - PSI_BETA) * phi)
        # z_clone = Z.param.clone()
        # g_spec.soft_threshold(lamb, conditioners=rho_cond)
        g_spec.hard_threshold(nnz=2)
        # m = g_spec.get_masks(nnz=2)[b_spec].to(Z.param)
        # Z.param.copy_(Z.param * (1 - m) + m * z_clone)

        # U.add_(W - Z.param)

        if (i % log_step) == 0:
            vals, counts = torch.unique(
                (g_spec.block_norms(None) > 1e-4).sum(-1), return_counts=True
            )
            vals = [int(v) for v in vals.cpu().numpy()]
            counts = [int(c) for c in counts.cpu().numpy()]
            print(
                i,
                dict(zip(vals, counts)),
                ((W.param - W0) @ H @ (W.param - W0).T).trace().item(),
                # (W - Z.data).norm().item(),
                # lamb.quantile(0.95).item(),
                # phi.quantile(0.95).item(),
            )

    m = g_spec.get_masks(nnz=2)
    W_sol = W.param.mul(m[b_spec])

    progress("Refining solution row-wise...")
    W_star = W_sol
    C = W0 @ H
    for j in tqdm(range(W_star.shape[0])):
        s = m[b_spec][j]
        idx = s.nonzero(as_tuple=True)[0]
        H_ss = H[idx][:, idx]
        W_star[j][idx] = torch.linalg.solve(H_ss.float(), C[j][idx].float(), left=True)
        if j % 100 == 0:
            dW = W_star - W0
            loss = ((dW @ H) * dW).sum().item()
            print(loss)

    dW = W_star - W0
    loss = ((dW @ H) * dW).sum().item()
    progress(f"Loss: {loss * N}")


if __name__ == "__main__":
    main()
