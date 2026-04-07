# Training FLOPs Accounting

## Per-sample FLOPs breakdown for sparse training

For a linear layer `y = Wx` with sparsity `s`, let:
- `dense_macs` = full dense MACs
- `sparse_macs` = dense_macs * (1 - s)

### RigL / SRigL / ASTRA (explore phase)

PyTorch's autograd computes the full dense weight gradient `dW = dy @ x^T`
regardless of sparsity masks. The mask is applied *after* the gradient is
computed. Both RigL and ASTRA need dense gradients for topology decisions:
- RigL accumulates them via `IndexMaskHook` for regrowth scoring
- ASTRA uses them every step for EMA-based soft-thresholding

The forward pass and input gradient backward use sparse weights (only nonzero
entries participate), but the weight gradient is always dense.

| Pass | Operation | FLOPs |
|------|-----------|-------|
| Forward | `y = W_sparse @ x` | sparse_macs |
| Backward (dx) | `dx = W_sparse^T @ dy` | sparse_macs |
| Backward (dW) | `dy @ x^T` (dense, full gradient computed) | dense_macs |

**Total per sample: 2 * sparse_macs + dense_macs**

This applies equally to RigL, SRigL, and ASTRA during their explore/sparsify
phases.

### Comparison at 90% sparsity (explore phase)

At 90% sparsity, sparse_macs = 0.1 * dense_macs:

- Both: 2 * 0.1 + 1.0 = **1.2x** dense training cost

### Frozen phase (both methods)

Once topology is fixed (no more exploration), gradients are masked to the
support before the weight gradient computation (via `post_accumulate_grad_hook`
or equivalent). Training cost drops to **3 * sparse_macs** for both methods.

At 90% sparsity: 3 * 0.1 = **0.3x** dense training cost.
