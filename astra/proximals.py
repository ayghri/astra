"""
Licensed under CC BY-NC 4.0 (see LICENSE or https://creativecommons.org/licenses/by-nc/4.0/)
Non-commercial use only; contact us for commercial licensing.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List, Union
import torch
import torch.nn as nn

from abc import abstractmethod

from sparsekit.block import SparseBlock
from sparsekit.scope import SparseScope

from astra.controllers import EMAController, LambdaController, AlphaController

Tensor = torch.Tensor
Parameter = nn.Parameter
Optimizer = torch.optim.Optimizer

# Optimizer Interface (Strategy Pattern)


class OptimizerProxy:
    """
    Abstracts the differences between SGD, Adam, etc.
    Extracts:
        - direction: The momentum/gradient used for the EMA controller.
        - learning rate
        - conditioner: when applicable
    """

    @staticmethod
    def get_proxy(optimizer: Optimizer) -> "OptimizerProxy":
        name = optimizer.__class__.__name__
        if name == "Adam":
            return AdamProxy(optimizer)
        elif name == "SGD":
            return SGDProxy(optimizer)
        else:
            raise ValueError(
                f"Optimizer {name} not explicitly supported. Assuming SGD-like behavior."
            )

    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer

    def get_info(
        self,
        param: Parameter,
    ) -> Tuple[Tensor, Union[float, Tensor], Optional[Tensor]]:
        """Returns (momentum_buffer, effective_step_size, conditioner)"""
        raise NotImplementedError


class SGDProxy(OptimizerProxy):
    @abstractmethod
    def get_info(
        self,
        param: Parameter,
    ) -> Tuple[Tensor, Union[float, Tensor], Optional[Tensor]]:
        state = self.optimizer.state.get(param, {})
        if "momentum_buffer" in state:
            momentum = state["momentum_buffer"]
        else:
            momentum = param.grad

        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if id(p) == id(param):
                    lr = group["lr"]

        return momentum, lr, None


class AdamProxy(OptimizerProxy):
    def get_info(
        self,
        param: Parameter,
    ) -> Tuple[Tensor, Union[float, Tensor], Optional[Tensor]]:
        beta1 = None
        beta2 = 0.0
        eps = 1e-12
        lr = 0.001
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if id(p) == id(param):
                    beta1, beta2 = group["betas"]
                    eps = group["eps"]
                    lr = group["lr"]
                    break
        assert beta1 is not None, f"param with id{id(param)} not found in optimizer "

        state = self.optimizer.state[param]
        step = state.get("step", 1)
        if isinstance(step, Tensor):
            step = step.item()

        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step

        momentum = state["exp_avg"] / bias_correction1
        denom = (state["exp_avg_sq"].sqrt() / (bias_correction2**0.5)).add(eps)

        return momentum, lr, denom


class AdamWProxy(OptimizerProxy):
    def get_info(
        self,
        param: Parameter,
    ) -> Tuple[Tensor, Union[float, Tensor], Optional[Tensor]]:
        beta1 = None
        beta2 = 0.0
        weight_decay = 0.0
        eps = 1e-8
        lr = 0.001
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if id(p) == id(param):
                    beta1, beta2 = group["betas"]
                    eps = group["eps"]
                    lr = group["lr"]
                    weight_decay = group["weight_decay"]
                    break
        assert beta1 is not None, f"param with id{id(param)} not found in optimizer "

        state = self.optimizer.state[param]
        step = state.get("step", 1)
        if isinstance(step, Tensor):
            step = step.item()

        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step

        momentum = state["exp_avg"] / bias_correction1
        denom = (state["exp_avg_sq"].sqrt() / (bias_correction2**0.5)).add(eps)
        if weight_decay != 0.0:
            momentum = momentum + weight_decay * param.data * denom

        return momentum, lr, denom


@dataclass
class ASTRASparsifier:
    groups: List[SparseScope]
    kappas: List[int]  # non-zero block count per group (parallel to groups)
    lambdas: LambdaController
    ema_grad: EMAController
    alphas: AlphaController
    optimizer: Optimizer
    eps: float = 1e-7

    _proxy: OptimizerProxy = field(init=False)

    def __post_init__(self):
        assert len(self.groups) == len(
            self.kappas
        ), "groups and kappas must have the same length"
        self._proxy = OptimizerProxy.get_proxy(self.optimizer)
        # Map nn.Parameter -> SparseBlock for O(1) lookup
        self._param_to_spec: Dict[Parameter, SparseBlock] = {}
        for g in self.groups:
            for s in g.specs():
                self._param_to_spec[s.view.param] = s

    @property
    def specs(self) -> List[SparseBlock]:
        return [s for g in self.groups for s in g.specs()]

    @torch.no_grad()
    def step(self, sparsify: bool = True):
        """
        Call this AFTER optimizer.step().
        1. Updates EMA of gradients (using optimizer state).
        2. Computes Gradient Bar (Score).
        3. Updates Lambda (Threshold).
        4. Applies Soft Thresholding.
        """

        # 1. Gather Data & Update EMA
        param_updates: Dict[SparseBlock, Dict] = {}

        for group_cfg in self.optimizer.param_groups:
            for p in group_cfg["params"]:
                if p not in self._param_to_spec:
                    continue

                spec = self._param_to_spec[p]

                direction, step_size, conditioner = self._proxy.get_info(p)
                # Reshape to the view layout used by spec.data (e.g. flat 1-D)
                direction = direction.reshape(spec.data.shape)

                self.ema_grad.update_single(spec, direction)

                param_updates[spec] = {
                    "step_size": step_size,
                    "ema": self.ema_grad.get(spec),
                    "conditioner": conditioner,
                }

        if not sparsify:
            return

        # Compute Scores (Grad Bar) & Update Lambdas
        for group, kappa in zip(self.groups, self.kappas):
            grad_bar_values: Dict[SparseBlock, Tensor] = {}

            for sp in group.specs():
                if sp not in param_updates:
                    continue

                data = param_updates[sp]
                alpha = self.alphas.get(sp)

                # Score = EMA_Grad - Alpha * Weights
                v = data["ema"] - alpha * sp.data
                grad_bar_values[sp] = v

            # ScopeCoupling.kth_largest(k, values)
            psi = group.kth_largest(k=kappa, values=grad_bar_values)

            self.lambdas.update_single(group, psi)

            current_lambda = self.lambdas.get(group).add(self.eps)

            # SGD: Euclidean proximal (conditioners=None)
            group.soft_threshold(current_lambda)


@dataclass
class IHTSparsifier:
    groups: List[SparseScope]
    kappas: List[int]  # non-zero block count per group (parallel to groups)
    optimizer: Optional[Optimizer]

    _masks: Dict[Parameter, Tensor] = field(default_factory=dict, init=False)
    _hooks: List = field(default_factory=list, init=False)

    def __post_init__(self):
        assert len(self.groups) == len(self.kappas)

    @torch.no_grad()
    def step(self):
        """Performs Hard Thresholding. Call after optimizer.step()."""
        for group, kappa in zip(self.groups, self.kappas):
            group.hard_threshold(nnz=kappa)

    def freeze_support(self):
        """Locks the current sparsity pattern."""
        self.step()

        self._masks.clear()

        for group in self.groups:
            for sp in group.specs():
                p = sp.view.param
                mask = (p.data.abs() > 0).float()
                self._masks[p] = mask
                p.data.mul_(mask)
                h = p.register_post_accumulate_grad_hook(
                    lambda p, m=mask: p.grad.mul_(m) if p.grad is not None else None
                )
                self._hooks.append(h)

        if self.optimizer:
            self._clean_optimizer_state()

    def _clean_optimizer_state(self):
        """Zeros out momentum buffers for pruned weights."""
        if self.optimizer is not None:
            for group in self.optimizer.param_groups:
                for p in group["params"]:
                    if p in self._masks:
                        state = self.optimizer.state.get(p, {})
                        for key in ["momentum_buffer", "exp_avg", "exp_avg_sq"]:
                            if key in state and state[key] is not None:
                                state[key].mul_(self._masks[p])

    def unfreeze(self):
        """Removes hooks to allow dense training again."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._masks.clear()
