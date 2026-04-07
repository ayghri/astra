# ADMM vs SparseGPT for LLM Pruning

## Per-layer results (single layer, same clean X, same target)

| Method    | Loss   | Relative RMSE | Time |
|-----------|--------|---------------|------|
| ADMM      | 143008 | 11%           | 10s  |
| SparseGPT | 206961 | 14%           | 1s   |

Both targeting 2:4 structured sparsity on the same layer with identical H = X^TX/N.

## Structural difference

**ADMM** solves `min_W ||WX - Y||^2` subject to sparsity, for arbitrary target Y.
The W-update generalizes: `mat_C = Y X^T (X^T X + rho)^{-1}`, so any target works.

**SparseGPT** is locked to `||(W - W0)X||^2`. The target is always W0's behavior — it cannot decouple the optimization target from the original weight matrix.

## Why this matters: error propagation across layers

In layer-by-layer LLM pruning, layer l is pruned, then its output becomes the input to layer l+1.

**SparseGPT pipeline:**
- Layer l+1 receives corrupted input X_{l+1} (from pruned layer l)
- Minimizes `||(W_{l+1} - W_{l+1}^orig) X_{l+1}^corrupted||^2`
- Best case: faithfully replicate what W_{l+1}^orig would do on corrupted input
- Errors compound with no correction mechanism

**ADMM pipeline:**
- Layer l+1 receives the same corrupted input X_{l+1}
- Can target `||W_{l+1} X_{l+1}^corrupted - Y_{l+1}^dense||^2` where Y^dense comes from the full dense model
- Each pruned layer actively **corrects** for upstream pruning error
- Pulls the signal back toward the dense model trajectory

## Impact

The 31% per-layer loss improvement (143k vs 207k) is real but secondary. The bigger win is the pipeline: SparseGPT compounds errors across 32+ transformer blocks with no way to correct them, while ADMM with dense-model targets gives each layer a chance to compensate. Over many layers, the correction capability matters far more than the per-layer gap.
