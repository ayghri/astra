from typing import List, Optional, Union, Dict, Callable, Any

import torch
from torch import Tensor

from torch.nn import Parameter
from torch.optim.optimizer import (
    Optimizer,
    ParamsT,
)
from tests.old.utils import kth_largest_torch
from examples.threshold import soft_threshold_inplace
from tests.old.utils import mask_post_accumulate_hook


class SASTRA(Optimizer):
    """Implements Stochastic Astra (SASTRA), a variant of SGD that promotes sparsity."""

    def __init__(
        self,
        params: ParamsT,
        params_sparsity: Dict[Parameter, int],
        lr: Union[float, Tensor] = 1e-3,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: Optional[bool] = None,
        astra_beta: Union[float, Callable[[int], float]] = 0.01,
        astra_alpha: Union[float, Callable[[int], float]] = 0.1,
        astra_relative: bool = True,
        astra_momentum: bool = True,
    ):
        """
        Args:
            params: Tensors to optimize.
            params_sparsity: Dictionary mapping tensors to their desired sparsity (k).
            lr: Learning rate.
            momentum: Momentum factor.
            dampening: Dampening for momentum.
            weight_decay: Weight decay (L2 penalty).
            nesterov: Enables Nesterov momentum.
            astra_beta: EMA coefficient for the thresholding parameter lambda.
            astra_alpha: Regularization parameter for the L1 penalty.
            astra_relative: If True, beta is scaled by the learning rate.
            astra_momentum: If True, use momentum in the lambda update.
        """
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            astra_beta=astra_beta,
            astra_alpha=astra_alpha,
            astra_relative=astra_relative,
            astra_momentum=astra_momentum,
        )
        super().__init__(params, defaults)
        self._step_count = 0
        self._thresholding_active = True
        self._params_sparsity = params_sparsity
        self._init_astra_state()

    def _init_astra_state(self) -> None:
        """Initializes the sastra state for each parameter."""
        for group in self.param_groups:
            for p in group["params"]:
                state = {}
                # astra_lambda always starts at 0 (or could be another default)
                state["lambda"] = torch.tensor(0.0).to(p)
                # astra_k: from parameter attribute or group default
                state["k"] = self._params_sparsity.get(p, 0)
                # astra_beta and astra_alpha: from group (evaluate callable if needed)
                for key in ["beta", "alpha", "relative", "momentum"]:
                    group_key = f"astra_{key}"
                    val = group.get(group_key, self.defaults[group_key])
                    if callable(val):
                        val = val(0)
                    state[key] = torch.tensor(val).to(p)
                self.state[p]["astra"] = state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("differentiable", False)
            group.setdefault("nesterov", False)

    def _init_group(
        self,
        group: Dict[str, Any],
        params: List[Tensor],
        grads: List[Tensor],
        momentum_buffer_list: List[Optional[Tensor]],
        astra_params: List[Dict],
    ) -> None:
        for p in group["params"]:
            if p.grad is not None:
                params.append(p)
                grads.append(p.grad)
                if group["momentum"] != 0:
                    state = self.state[p]
                    momentum_buffer_list.append(state.get("momentum_buffer"))
                astra_params.append(self.state[p]["astra"])

    def step(self, closure=None):  # type: ignore
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1

        for group in self.param_groups:
            params: List[Tensor] = []
            grads: List[Tensor] = []
            astra_params: List[Dict] = []
            momentum_buffer_list: List[Optional[Tensor]] = []

            self._init_group(
                group, params, grads, momentum_buffer_list, astra_params
            )

            astra_sgd2(
                params=params,
                grads=grads,
                momentum_buffer_list=momentum_buffer_list,
                astra_params=astra_params,
                lr=group["lr"],
                step_count=self._step_count,
                weight_decay=group["weight_decay"],
                momentum=group["momentum"],
                dampening=group["dampening"],
                nesterov=group["nesterov"],
                threshold=self._thresholding_active,
            )

            if group["momentum"] != 0:
                # update momentum_buffers in state
                for p, momentum_buffer in zip(params, momentum_buffer_list):
                    state = self.state[p]
                    state["momentum_buffer"] = momentum_buffer

        return loss

    def deactivate_thresholding(self) -> None:
        """Temporarily disables the soft-thresholding step."""
        self._thresholding_active = False

    def activate_thresholding(self) -> None:
        """Re-enables the soft-thresholding step."""
        self._thresholding_active = True

    def mask_gradients(self, masks: Dict[Parameter, Tensor]) -> None:
        """
        Applies a mask to the gradients and momentum for a subset of parameters.
        This is useful for freezing layers or parts of layers.
        """
        for group in self.param_groups:
            # print(f"group : {group}")
            for p in group["params"]:
                mask = masks.get(p, None)
                if mask is None:
                    continue
                state = self.state[p]
                # print(f"mask : {mask}")
                if group["momentum"] != 0 and "momentum_buffer" in state:
                    # print(f"before momentum buffer: {state['momentum_buffer']}")
                    state["momentum_buffer"].mul_(mask)
                    # print(f"after momentum buffer: {state['momentum_buffer']}")

                if state.get("param_hook", None) is not None:
                    state["param_hook"].remove()
                    del state["param_hook"]

                state["param_hook"] = p.register_post_accumulate_grad_hook(
                    lambda param, m=mask: mask_post_accumulate_hook(param, m)
                )

    def unmask_gradients(self) -> None:
        """Removes gradient masks, allowing all weights to be updated."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    state = self.state[p]
                    if "param_hook" in state:
                        state["param_hook"].remove()
                        del state["param_hook"]


@torch.no_grad()
def astra_sgd(
    params: List[Tensor],
    grads: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    astra_params: List[Dict],
    weight_decay: float,
    momentum: float,
    lr: float,
    dampening: float,
    nesterov: bool,
    step_count: int,
    threshold: bool,
) -> None:
    """
    Performs a ASTRA SGD step with optional momentum and weight decay.
    This function is called by the SASTRA optimizer.
    """
    for i, param in enumerate(params):
        param_grad = grads[i]
        sastra = astra_params[i]
        is_sastra = sastra["k"] > 0

        if weight_decay != 0:
            param_grad = param_grad.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]
            if buf is None:
                buf = torch.clone(param_grad).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(param_grad, alpha=1 - dampening)
            if nesterov:
                grad = param_grad.add(buf, alpha=momentum)
            else:
                grad = buf
        else:
            grad = param_grad

        new_lambda = None

        if is_sastra and threshold:
            alpha = sastra["alpha"]
            if callable(alpha):
                alpha: float = alpha(step_count)  # type: ignore

            beta = sastra["beta"]
            if callable(beta):
                beta: float = beta(step_count)  # type: ignore
            if sastra["relative"]:
                beta = beta * lr

            new_lambda = sastra["lambda"] * (1 - beta)
            if sastra["momentum"]:
                new_lambda.add_(
                    kth_largest_torch(
                        (grad - alpha * param.data).abs(),
                        sastra["k"],
                    ),
                    alpha=beta,
                )
            else:
                new_lambda.add_(
                    kth_largest_torch(
                        (param_grad.detach() - alpha * param.data).abs(),
                        sastra["k"],
                    ),
                    alpha=beta,
                )
        param.add_(grad, alpha=-lr)
        if is_sastra and threshold:
            soft_threshold_inplace(param.data, sastra["lambda"] * lr)
            sastra["lambda"] = new_lambda


@torch.no_grad()
def astra_sgd2(
    params: List[Tensor],
    grads: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    astra_params: List[Dict],
    weight_decay: float,
    momentum: float,
    lr: float,
    dampening: float,
    nesterov: bool,
    step_count: int,
    threshold: bool,
) -> None:
    """
    Performs a ASTRA SGD step with optional momentum and weight decay.
    This function is called by the SASTRA optimizer.
    """
    for i, param in enumerate(params):
        param_grad = grads[i]
        sastra = astra_params[i]
        is_sastra = sastra["k"] > 0

        if weight_decay != 0:
            param_grad = param_grad.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]
            if buf is None:
                buf = torch.clone(param_grad).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(param_grad, alpha=1 - dampening)
            if nesterov:
                grad = param_grad.add(buf, alpha=momentum)
            else:
                grad = buf
        else:
            grad = param_grad

        param.add_(grad, alpha=-lr)

        if is_sastra and threshold:
            alpha = sastra["alpha"]
            if callable(alpha):
                alpha: float = alpha(step_count)  # type: ignore
            beta = sastra["beta"]
            if callable(beta):
                beta: float = beta(step_count)  # type: ignore

            if sastra["relative"]:
                beta = beta * lr

            sastra["lambda"].mul_(1 - beta)

            if sastra["momentum"]:
                sastra["lambda"].add_(
                    kth_largest_torch(
                        (grad - alpha * param.data).abs(),
                        sastra["k"] + 1,
                    ),
                    alpha=beta,
                )
            else:
                sastra["lambda"].add_(
                    kth_largest_torch(
                        (param_grad.detach() - alpha * param.data).abs(),
                        sastra["k"] + 1,
                    ),
                    alpha=beta,
                )
            soft_threshold_inplace(param.data, sastra["lambda"] * lr)
