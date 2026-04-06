import torch
from collections import defaultdict


class DiagGNEstimator:
    """
    Accumulate diagonal Gauss-Newton / Fisher for nn.Linear weights:
      diag(H_W)[i,j] ≈ E[ delta[i]^2 * x[j]^2 ].
    """

    def __init__(
        self, module, *, store_full_diag=False, dtype=torch.float32, device=None
    ):
        self.module = module
        self.store_full_diag = store_full_diag
        self.dtype = dtype
        self.device = device
        self.handles = []
        self.cache_x = {}  # id(linear) -> input activations (flattened)
        self.diagW = {}  # id(linear) -> diag estimate (out,in) if store_full_diag
        self.row_delta2 = {}  # id(linear) -> sum(delta^2) over samples (out,)
        self.col_x2 = {}  # id(linear) -> sum(x^2) over samples (in,)
        self.count = defaultdict(
            int
        )  # id(linear) -> number of samples aggregated

    def _register(self):
        for name, m in self.module.named_modules():
            if isinstance(m, torch.nn.Linear):
                # forward hook to capture x
                fh = m.register_forward_hook(self._fwd_hook(name))
                # full backward hook to capture grad wrt output (delta)
                bh = m.register_full_backward_hook(self._bwd_hook(name))
                self.handles.extend([fh, bh])

    def _fwd_hook(self, name):
        def hook(m, inputs, output):
            # inputs[0] is x: [batch, seq, in] or [N, in]
            x = inputs[0]
            x = x.reshape(-1, x.shape[-1]).detach()
            if self.device is not None:
                x = x.to(self.device)
            self.cache_x[id(m)] = x

        return hook

    def _bwd_hook(self, name):
        def hook(m, grad_input, grad_output):
            # grad_output[0] is delta wrt output y: [batch, seq, out] or [N, out]
            delta = grad_output[0]
            if delta is None:
                return
            delta = delta.reshape(-1, delta.shape[-1]).detach()
            if self.device is not None:
                delta = delta.to(self.device)

            x = self.cache_x.pop(id(m), None)
            if x is None:
                return

            # squares in fp32 for stability
            x2 = x.to(self.dtype) ** 2
            d2 = delta.to(self.dtype) ** 2

            # accumulate sufficient statistics
            # sum over samples: d2_sum[out], x2_sum[in]
            d2_sum = d2.sum(dim=0)  # [out]
            x2_sum = x2.sum(dim=0)  # [in]
            n = x2.shape[0]

            if id(m) not in self.row_delta2:
                self.row_delta2[id(m)] = d2_sum
                self.col_x2[id(m)] = x2_sum
            else:
                self.row_delta2[id(m)] += d2_sum
                self.col_x2[id(m)] += x2_sum

            self.count[id(m)] += n

            # optionally accumulate full diag (out,in): sum over samples of outer(d2, x2) elementwise
            if self.store_full_diag:
                # Outer of sums is NOT equal to sum of outers; we must sum per-sample outers.
                # Efficiently: d2^T @ x2 gives sum over samples of (d2_s[:,None] * x2_s[None,:])
                diag = d2.T @ x2  # [out, in]
                if id(m) not in self.diagW:
                    self.diagW[id(m)] = diag
                else:
                    self.diagW[id(m)] += diag

        return hook

    def start(self):
        self._register()

    def stop(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def get_diag(self):
        """
        Returns diag estimates normalized by sample count:
        - if store_full_diag: dict[name_or_id] -> [out,in]
        - else: returns factorized stats (row_delta2, col_x2, count)
        """
        if self.store_full_diag:
            out = {}
            for name, m in self.module.named_modules():
                if isinstance(m, torch.nn.Linear):
                    mid = id(m)
                    if mid in self.diagW:
                        out[name] = self.diagW[mid] / max(self.count[mid], 1)
            return out
        else:
            out = {}
            for name, m in self.module.named_modules():
                if isinstance(m, torch.nn.Linear):
                    mid = id(m)
                    if mid in self.count:
                        out[name] = {
                            "E_delta2": self.row_delta2[mid]
                            / max(self.count[mid], 1),  # [out]
                            "E_x2": self.col_x2[mid]
                            / max(self.count[mid], 1),  # [in]
                            "count": self.count[mid],
                        }
            return out
