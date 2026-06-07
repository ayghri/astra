# ASTRA — Adaptive Soft-Thresholding Algorithm

PyTorch implementation of the inverse-regularization sparse-recovery method
*Inverse Regularization for Structured Sparse Recovery with Computable
Certificates*. ASTRA selects per-scope regularization weights $\boldsymbol\lambda$
online by tracking the fixed point
$\boldsymbol\psi(\boldsymbol w(\boldsymbol\lambda)) = \boldsymbol\lambda$ of an
order-statistic gauge, rather than sweeping a $\lambda$-grid.

## Install

```bash
poetry install            # core
poetry install -E plotting -E imagenet
```

Requires Python ≥3.11, PyTorch ≥2.10, and
[`sparsekit`](https://pypi.org/project/sparsekit/) for the `BlockSpec` /
`ScopeSpec` algebra used throughout.

## Algorithm in one screen

Inner proximal-gradient step on $f(\boldsymbol w) + \sum_j \lambda_j\,\Omega_j(\boldsymbol w_{\boldsymbol s_j})$, with outer EMA tracker for $\boldsymbol\lambda$:

```
for t in 1..T:
    direction = optimizer.step()                         # SGD/Adam/AdamW grad or momentum
    g         = ema_grad.update(direction)               # per-block EMA
    psi       = kth_mid(|g - alpha * w|; kappa)          # order-statistic gauge per scope
    lambda    = (1 - beta_t) * lambda + beta_t * psi     # beta_t = beta_0 / (t + t_0)
    w         = prox_lambda(w - eta * direction)         # soft-threshold under conditioner alpha
# at convergence: OLS refit on the per-scope top-kappa support
```

Hyperparameters used in the paper: $\beta_0 = 1$, $t_0 = 100$ (effective EMA
retention $\approx 0.99$); diagonal / block-diagonal / dense conditioners.

## Package layout

```
astra/
├── controllers.py      EMAController, LambdaController, AlphaController
├── proximals.py        ASTRASparsifier (PASTRA), IHTSparsifier
├── optimizers.py       Adam/SGD/AdamW + ASTRA wrappers
├── optimizers/         SASTRA, IHT, Muon-style, dataset-specific variants
├── hess.py             per-layer Hessian accumulators
├── linalg.py           Cholesky / Newton-Schulz / batched solves
├── prune.py            layer-wise OBS / closed-form prune utilities
└── pruners/
    ├── admm.py         dense-conditioner ADMM (Algorithm 3 of the paper)
    ├── admm_fp16.py    fp16/bf16 ADMM with fp32 storage
    ├── sparsegpt.py    SparseGPT baseline
    ├── wanda.py        Wanda baseline
    └── base.py         PruningStrategy ABC + PrunableLinear

astra/data/             dataset loaders (CIFAR, C4, ImageNet, MNIST)
astra/models/           ResNet, WideResNet, sparse Linear/Conv layers
astra/train/            schedulers, sweep harness, training utils
astra/configs.py        Hydra/OmegaConf config plumbing
astra/evaluate.py       lm-eval-harness + classification eval glue
```

## Minimal usage

```python
import torch
from astra.proximals import ASTRASparsifier
from astra.controllers import EMAController, LambdaController, AlphaController
from sparsekit import BlockSpec, ScopeSpec, View

# Declare structured sparsity: N=2 nonzeros per scope of M=4 (2:4)
view  = View.from_existing(linear.weight)
block = BlockSpec(view, (1, 1), "b")
scope = ScopeSpec(block, (1, 4), "s")

# Wrap any torch.optim optimizer (Adam, SGD, AdamW)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
astra = ASTRASparsifier(
    groups=[scope],
    kappas=[2],
    lambdas=LambdaController(),
    ema_grad=EMAController(rho=0.9),
    alphas=AlphaController(default=1.0),
    optimizer=opt,
)

for batch in loader:
    loss = model(batch).loss
    loss.backward()
    opt.step()
    astra.step()           # updates λ, applies block soft-threshold
    opt.zero_grad()
```

## Layer-wise LLM pruning

`astra.pruners.admm.admm_prune` implements the dense-conditioner ADMM variant
used in the Qwen3 study: closed-form $\mathbf W$-update via Cholesky of
$\mathbf H + \rho \mathbf D$, scope-aware soft-threshold $\mathbf Z$-update,
ADMM dual ascent on $\mathbf U$, and an OLS refit on the converged top-N
support per row. An fp16/bf16 variant (Tensor-Core matmuls, fp32 master
storage) is in `admm_fp16.py`. SparseGPT and Wanda baselines are included for
comparison under the same calibration set.

## License

CC BY-NC 4.0 — non-commercial. Contact the author for commercial licensing.

<!-- ## Citation -->
<!---->
<!-- ```bibtex -->
<!-- @article{astra2026, -->
<!--   title  = {Inverse Regularization for Structured Sparse Recovery with Computable Certificates}, -->
<!--   author = {Anonymous}, -->
<!--   year   = {2026}, -->
<!--   note   = {Under review, NeurIPS 2026} -->
<!-- } -->
<!-- ``` -->
