import torch
import torch.nn as nn
from typing import Iterable


def compute_hessian(input_batches: Iterable[torch.Tensor]):
    H = None
    accum_samples = None

    for batch in input_batches:
        n_samples = batch.size(0)
        if accum_samples is None or H is None:
            H = torch.matmul(batch.T, batch) / n_samples
            accum_samples = n_samples
        else:
            tmp_accum = accum_samples + n_samples
            H = (
                H * (accum_samples / tmp_accum)
                + torch.matmul(batch.T, batch) / tmp_accum
            )
            accum_samples += tmp_accum

    assert H is not None
    return H


@torch.no_grad()
def vectorized_obs_mask(
    weight: nn.Parameter, H, target_nnz_per_row, damping=1e-6
):
    W = weight.data.clone()  # Shape: (Out_Features, In_Features)
    n_rows, n_cols = W.shape

    assert (
        target_nnz_per_row < n_cols
    ), "Target NNZ must be less than total input features."

    # 1. Compute Inverse Hessian (Shared across all rows)
    # Note: We compute H based on the input features.
    # H_inv = compute_inverse_hessian(input_data.to(device))

    L = torch.linalg.cholesky(H + damping * torch.eye(H.shape[0]).to(H))
    H_inv = torch.cholesky_inverse(L)

    # Extract diagonal for fast saliency calculation
    H_inv_diag = H_inv.diagonal()  # Shape: (In_Features,)

    # 2. Initialize Mask (0 = keep, 1 = prune)
    # We assume we start from a dense state. If already pruned, you might need to load existing mask.
    mask = torch.zeros_like(W, dtype=torch.bool)

    # Determine how many iterations we need
    # We prune 1 weight per row per iteration.
    current_nnz = n_cols
    num_iterations = current_nnz - target_nnz_per_row

    for _ in range(num_iterations):
        # --- A. Compute Saliency Scores ---
        # OBS Saliency: L_q = (w_q^2) / (2 * [H^-1]_qq)
        # We compute this for ALL weights efficiently.
        # Shape: (Rows, Cols)
        scores = (W**2) / (H_inv_diag.unsqueeze(0))

        # Set scores of already pruned weights to infinity so they aren't picked again
        scores[mask] = float("inf")

        # --- B. Select Weight to Prune per Row ---
        # Find index `q` with minimum saliency for each row
        # prune_indices shape: (Rows,)
        prune_indices = torch.argmin(scores, dim=1)

        mask[torch.arange(n_rows), prune_indices] = True

        # --- C. Vectorized Weight Update (OBS Surgery) ---
        # Formula: w_new = w_old - (w_q / Hinv_qq) * Hinv_col_q

        # 1. Gather w_q (the weight being pruned) for each row
        # Shape: (Rows,)
        w_q = W[torch.arange(n_rows), prune_indices]

        # 2. Gather Hinv_qq (diagonal element) for the selected indices
        # Shape: (Rows,)
        h_qq = H_inv_diag[prune_indices]

        # 3. Gather the corresponding columns of H_inv for the selected indices
        # H_inv is symmetric, so row access H_inv[indices, :] is equivalent to columns
        # Shape: (Rows, Cols)
        H_inv_cols = H_inv[prune_indices, :]

        # 4. Compute update vector
        # factor shape: (Rows, 1) for broadcasting
        factor = (w_q / h_qq).unsqueeze(1)

        # delta shape: (Rows, Cols)
        delta = factor * H_inv_cols

        # 5. Apply update
        W.sub_(delta)

        # --- D. Enforce Zero Constraint ---
        # The update naturally sets w_q to 0, but we enforce it strictly.
        # Also, since we use a static Hessian, previously pruned weights might get non-zero updates.
        # We must re-clamp them to zero.

        # Mark newly pruned weights in the mask

        # Hard zeroing of all pruned weights (both new and old)
        W.masked_fill_(mask, 0.0)

    return mask, W


# --- Usage Example ---
if __name__ == "__main__":
    # Setup dummy data and layer
    N, D_in, D_out = 10, 5, 10
    layer = nn.Linear(D_in, D_out, bias=False)
    inputs = torch.randn(N, D_in)

    target_nnz = 3

    print("Original Weight (First 2 rows):\n", layer.weight.data[:2, :5])

    # Run Pruning
    H = compute_hessian([inputs])
    final_mask = vectorized_obs_mask(layer.weight, H, target_nnz)

    print(
        "\nPruned Weight (First 2 rows):\n",
        layer.weight.data.mul(~final_mask)[:2, :5],
    )

    # Verify sparsity
    nnz_counts = (layer.weight.data.mul(~final_mask) != 0).sum(dim=1)
    print(f"\nNon-zeros per row: {nnz_counts.tolist()}")
