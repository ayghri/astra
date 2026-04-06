from typing import Callable, List, Optional, Union, Dict


import torch
from torch import Tensor

from torch.optim.optimizer import (
    _use_grad_for_differentiable,
    Optimizer,
    ParamsT,
)
from tests.old.utils import kth_largest_torch
from examples.threshold import soft_threshold_inplace


class DySTiLSGD(Optimizer):  # noqa: D101
    def __init__(
        self,
        params: ParamsT,
        # dystil_st: Callable,
        lr: Union[float, Tensor] = 1e-3,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        maximize: bool = False,
        differentiable: bool = False,
        nesterov: Optional[bool] = None,
    ):  # noqa: D107
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
            maximize=maximize,
            differentiable=differentiable,
            nesterov=nesterov,
            dystil_lambda=0.0,
            dystil_beta=0.005,
            dystil_k=0,
        )
        super().__init__(params, defaults)

        def process_state(state, p):
            state["dystil"] = {
                "k": getattr(p, "dystil_k", defaults["dystil_k"])
            }
            for attr in ["lambda", "beta"]:
                state["dystil"][attr] = torch.tensor(
                    getattr(p, f"dystil_{attr}", defaults[f"dystil_{attr}"])
                ).to(p)

        for group in self.param_groups:
            for p in group["params"]:
                process_state(self.state[p], p)
        # self.dystil_st = dystil_st

    def __setstate__(self, state):  # noqa: D105
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("maximize", False)
            group.setdefault("differentiable", False)
            group.setdefault("nesterov", False)

    def _init_group(
        self,
        group,
        params: List,
        grads: List,
        momentum_buffer_list: List,
        dystil_params: List,
    ):
        def init_group_dystil(opt, p):
            return opt.state[p].get("dystil")

        for p in group["params"]:
            if p.grad is not None:
                params.append(p)
                grads.append(p.grad)
                if group["momentum"] != 0:
                    state = self.state[p]
                    momentum_buffer_list.append(state.get("momentum_buffer"))
                dystil_params.append(init_group_dystil(self, p))

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params: List[Tensor] = []
            grads: List[Tensor] = []
            dystil_params: List[Dict] = []
            momentum_buffer_list: List[Optional[Tensor]] = []

            self._init_group(
                group, params, grads, momentum_buffer_list, dystil_params
            )

            sgd(
                params,
                grads,
                momentum_buffer_list,
                dystil_params,
                # self.dystil_st,
                weight_decay=group["weight_decay"],
                momentum=group["momentum"],
                lr=group["lr"],
                dampening=group["dampening"],
                maximize=group["maximize"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
                nesterov=group["nesterov"],
            )

            if group["momentum"] != 0:
                # update momentum_buffers in state
                for p, momentum_buffer in zip(params, momentum_buffer_list):
                    state = self.state[p]
                    state["momentum_buffer"] = momentum_buffer

        return loss


def sgd(
    params: List[Tensor],
    grads: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    dystil_params,
    # dystil_st,
    weight_decay: float,
    momentum: float,
    lr: float,
    dampening: float,
    maximize: bool,
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    nesterov: bool,
):
    assert grad_scale is None and found_inf is None

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        dystil = dystil_params[i]
        is_dystil = dystil["k"] > 0

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(grad).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(grad, alpha=1 - dampening)

            if nesterov:
                grad = grad.add(buf, alpha=momentum)
            else:
                grad = buf

        if is_dystil:
            # new_lambda = dystil_st(dystil, param, grad)
            new_lambda = dystil["lambda"] * (1 - dystil["beta"])
            new_lambda.add_(
                kth_largest_torch(
                    (param.data * 0.1 - grad).abs(), dystil["k"]
                ),
                alpha=dystil["beta"],
            )
            # pass

        param.add_(grad, alpha=-lr)
        # if is_dystil:
        # new_lambda = dystil["lambda"] - dystil["beta"] * (
        #     dystil["lambda"]
        #     - kth_largest_torch((param.data).abs(), dystil["k"])
        # )
        if is_dystil:
            soft_threshold_inplace(param.data, dystil["lambda"] * lr)
            dystil["lambda"] = new_lambda
            # print(dystil["lambda"], new_lambda)
