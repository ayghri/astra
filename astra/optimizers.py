"""
Licensed under CC BY-NC 4.0 (see LICENSE or https://creativecommons.org/licenses/by-nc/4.0/)
Non-commercial use only; contact us for commercial licensing.
PyTorch optimizer with ASTRA Flavor.
"""

import math
from collections.abc import MutableMapping
from typing import Optional, Tuple, Dict, Union

import torch
from torch import Tensor
from torch.nn import Parameter
from torch.optim.optimizer import (
    Optimizer,
    ParamsT,
)
from astra.proximals import SASTRA
from torch.optim.adam import Adam
from torch.optim.optimizer import _to_scalar, _get_value, DeviceDtypeDict


# Constants from Keller Jordan's Muon post: https://kellerjordan.github.io/posts/muon/
EPS = 1e-8
NS_COEFS = (3.4445, -4.7750, 2.0315)
NS_STEPS = 5


def _zeropower_via_newtonschulz(
    grad: Tensor,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
) -> Tensor:
    a, b, c = ns_coefficients
    ortho_grad = grad.bfloat16()
    if grad.size(0) > grad.size(1):
        ortho_grad = ortho_grad.T
    # Ensure spectral norm is at most 1
    ortho_grad.div_(ortho_grad.norm().clamp(min=eps))
    # Perform the NS iterations
    for _ in range(ns_steps):
        gram_matrix = ortho_grad @ ortho_grad.T
        gram_update = torch.addmm(
            gram_matrix, gram_matrix, gram_matrix, beta=b, alpha=c
        )
        ortho_grad = torch.addmm(ortho_grad, gram_update, ortho_grad, beta=a)

    if grad.size(0) > grad.size(1):
        ortho_grad = ortho_grad.T
    return ortho_grad


def _update_scale(
    scale_method: Optional[str], param_shape: torch.Size
) -> float:
    """Default learning rate adjustment used by Muon."""
    A, B = param_shape[:2]
    scale = 1.0

    if scale_method is None or scale_method == "original":
        scale = math.sqrt(max(1, A / B))
    elif scale_method == "match_rms_adamw":
        scale = 0.2 * math.sqrt(max(A, B))
    return scale


class ASTRAMuon(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        sparsifier: SASTRA,
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        force_2d: bool = False,
        ns_coefficients: Tuple[float, float, float] = NS_COEFS,
        eps: float = EPS,
        ns_steps: int = NS_STEPS,
        scale_method: Optional[str] = None,
    ) -> None:
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= lr:
            raise ValueError(f"Learning rate should be >= 0 but is: {lr}")
        if not 0.0 <= momentum:
            raise ValueError(f"momentum should be >= 0 but is: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(
                f"weight decay should be >= 0 but is: {weight_decay}"
            )
        if scale_method is not None and scale_method not in [
            "original",
            "match_rms_adamw",
        ]:
            raise ValueError(
                f"scale method {scale_method} is not supported: original,match_rms_adamw"
            )

        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_coefficients": ns_coefficients,
            "eps": eps,
            "ns_steps": ns_steps,
            "scale_method": scale_method,
        }
        super().__init__(params, defaults)

        if not force_2d:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.ndim != 2:
                        raise ValueError(
                            f"Muon only supports 2D parameters whereas we found a parameter with size: {p.size()}"
                        )
        self.force_2d = force_2d
        self.sparsifier = sparsifier

    def _init_group(
        self,
        group: MutableMapping,
        params_with_grad: list[Tensor],
        grads: list[Tensor],
        muon_momentum_bufs: list[Tensor],
        update_scalers: list[Tensor],
    ):
        for p in group["params"]:
            if p.grad is None:
                continue

            params_with_grad.append(p)
            grads.append(p.grad)

            state = self.state[p]

            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(
                    p.grad, memory_format=torch.preserve_format
                )

            muon_momentum_bufs.append(state["momentum_buffer"])

            if "update_scaler" not in state:
                state["update_scaler"] = torch.tensor(
                    _update_scale(state["scale_method"], p.shape)
                ).to(p)
            update_scalers.append(state["update_scaler"])

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]

            params_with_grad: list[Tensor] = []
            grads: list[Tensor] = []
            muon_momentum_bufs: list[Tensor] = []
            update_scalers: list[Tensor] = []

            self._init_group(
                group,
                params_with_grad,
                grads,
                muon_momentum_bufs,
                update_scalers,
            )

            muon(
                params_with_grad,
                grads,
                muon_momentum_bufs,
                update_scalers,
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum,
                nesterov=group["nesterov"],
                force_2d=self.force_2d,
                ns_coefficients=group["ns_coefficients"],
                eps=group["eps"],
                ns_steps=group["ns_steps"],
            )
        return loss


def muon(
    params: list[Tensor],
    grads: list[Tensor],
    muon_momentum_bufs: list[Tensor],
    update_scalers: list[Tensor],
    *,
    lr: float,
    weight_decay: float,
    momentum: float,
    nesterov: bool,
    force_2d: bool,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
) -> Dict[Parameter, Tensor]:
    param_grad = {}

    for i, param in enumerate(params):
        grad = grads[i]
        update_scale = update_scalers[i]
        if grad.ndim != 2:
            raise ValueError("Param gradient must be a 2D matrix")

        buf = muon_momentum_bufs[i]
        buf.lerp_(grad, 1 - momentum)
        update = buf
        if nesterov:
            update = grad.lerp(buf, momentum)

        shape = update.size()

        reshaped = False
        if len(update.size()) > 2 and force_2d:
            update = update.reshape(shape[0], -1)
            reshaped = True
        else:
            continue

        update = _zeropower_via_newtonschulz(
            update, ns_coefficients, ns_steps, eps
        )
        if reshaped:
            update = update.reshape(shape)

        update.mul_(update_scale)
        update.add(param, alpha=weight_decay)

        param.add_(update, alpha=-lr)
        param_grad[param] = update

    return param_grad


class ProxAdam(Adam):
    def __init__(
        self,
        params: ParamsT,
        proximal_operator,
        lr: Union[float, Tensor] = 1e-3,
        betas: tuple[Union[float, Tensor], Union[float, Tensor]] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
        *,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        decoupled_weight_decay: bool = False,
    ):
        super().__init__(params, lr, betas, eps, weight_decay, maximize)
        self.proximal_op = proximal_operator

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad: list[Tensor] = []
            grads: list[Tensor] = []
            exp_avgs: list[Tensor] = []
            exp_avg_sqs: list[Tensor] = []
            max_exp_avg_sqs: list[Tensor] = []
            state_steps: list[Tensor] = []
            beta1, beta2 = group["betas"]

            _ = self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
            )

            single_tensor_proxadam(
                self.proximal_op,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=group["maximize"],
            )

        return loss


def single_tensor_proxadam(
    proximal_operator,
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    state_steps: list[Tensor],
    *,
    beta1: Union[float, Tensor],
    beta2: Union[float, Tensor],
    lr: Union[float, Tensor],
    weight_decay: float,
    eps: float,
    maximize: bool,
):
    if torch.jit.is_scripting():
        # this assert is due to JIT being dumb and not realizing that the ops below
        # have overloads to handle both float and Tensor lrs, so we just assert it's
        # a float since most people using JIT are using floats
        assert isinstance(lr, float)
        assert isinstance(beta1, float)
        assert isinstance(beta2, float)
    else:
        lr = _to_scalar(lr)
        # TODO: Support nonzero-dim Tensor betas, see #147921

    # We only shuffle around the beta when it is a Tensor, otherwise, we prefer
    # treating it as a scalar.
    # Note: ensure type declaration is under conditional check for isinstance
    # or else torchscript will get cranky about the DeviceDict type.
    if isinstance(beta1, Tensor):
        beta1_dict: Optional[DeviceDtypeDict] = {
            (beta1.device, beta1.dtype): beta1
        }
    else:
        beta1_dict = None

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        # update step
        step_t += 1

        if weight_decay != 0:
            # Nested if is necessary to bypass jitscript rules
            grad = grad.add(param, alpha=weight_decay)

        device = param.device

        if beta1_dict is not None:
            dtype = param.dtype  # type: ignore[union-attr]

            # cast to workaround https://github.com/pytorch/pytorch/issues/140601
            key = (device, dtype)
            if key not in beta1_dict:
                beta1_dict[key] = beta1.to(  # type: ignore[union-attr]
                    device=device, dtype=dtype, non_blocking=True
                )

            device_beta1: Union[float, Tensor] = beta1_dict[key]
        else:
            device_beta1 = beta1

        # Decay the first and second moment running average coefficient
        exp_avg.lerp_(grad, 1 - device_beta1)

        # Nested if is necessary to bypass jitscript rules
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)  # type: ignore[arg-type]

        step = _get_value(step_t)

        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step

        step_size = lr / bias_correction1

        bias_correction2_sqrt = bias_correction2**0.5

        denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

        param.addcdiv_(exp_avg, denom, value=-step_size)  # type: ignore[arg-type]
        proximal_operator.step_tensor(
            param,
            learning_rate=lr,
            grad_momentum=exp_avg,
            grad_sq_momentum=exp_avg_sq,
            bias_correction1=bias_correction1,
            bias_correction2_sqrt=bias_correction2_sqrt,
            eps=eps,
        )
