"""
Licensed under CC BY-NC 4.0 (see LICENSE or https://creativecommons.org/licenses/by-nc/4.0/)
Non-commercial use only; contact us for commercial licensing.
"""

from typing import Optional
import torch
from torch import nn
from adasoft.misc import transfer_to_device

Tensor = torch.Tensor


def mask_post_accumulate_hook(param: Tensor, mask: Tensor) -> None:
    param.grad = param.grad * mask  # type: ignore


def mask_grad_hook(grad: Tensor, mask: Tensor) -> Tensor:
    return grad * mask


def get_input_hook(name: str, activations: dict, procees_fn=None):
    """Creates a forward hook to capture the input activations of a layer."""

    def hook(model, input, output):
        if name not in activations:
            activations[name] = []
        if procees_fn is not None:
            input = procees_fn(input)
        activations[name].append(input)

    return hook


def get_output_hook(name: str, activations: dict, device=None):
    """Creates a forward hook to capture the output activations of a layer."""

    def hook(model, input, output):
        if name not in activations:
            activations[name] = []
        if device is not None:
            output = output.to(device)
        activations[name].append(output)

    return hook


def get_forward_hook(
    name: str,
    activations: dict,
    input_process_fn=None,
    output_process_fn=None,
):
    """Creates a forward hook to capture the input and output activations of a layer."""

    def hook(model, input, output):
        # print(f"Hook triggered for layer: {model}")
        if name not in activations:
            activations[name] = {"input": [], "output": []}
        if input_process_fn is not None:
            input = input_process_fn(input)
        if output_process_fn is not None:
            output = output_process_fn(output)
        activations[name]["input"].append(input)
        activations[name]["output"].append(output)

    return hook


class InputCatcher(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.inputs = None

    def forward(self, **kwargs):
        self.inputs = kwargs
        return self.layer(**kwargs)


class LLMIOCatcher:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.hooks = {}
        self.inputs = {}
        self.outputs = {}

    @staticmethod
    def _layer_name(layer_idx):
        return f"layer_{layer_idx}"

    def attach_layer(self, layer_idx):
        name = f"layer_{layer_idx}"
        layer: nn.Module = self.model.model.layers[layer_idx]

        def input_hook(module, args, kwargs):
            self.inputs[name] = {"args": args, "kwargs": kwargs}

        def output_hook(module, args, output):
            if self.device is not None and isinstance(output, Tensor):
                self.outputs[name] = output.to(self.device)
            else:
                self.outputs[name] = output

        self.hooks[f"{name}_pre"] = layer.register_forward_pre_hook(
            input_hook, with_kwargs=True
        )
        self.hooks[f"{name}_post"] = layer.register_forward_hook(output_hook)

        # hook = get_output_hook(name, self.activations, device=self.device)
        # self.hooks[name] = layer.register_forward_hook(hook)
        # self.model.model.layers[layer_idx] = InputCatcher(layer)

    def detach_layer(self, layer_idx):
        name = self._layer_name(layer_idx)

        self.inputs.pop(name, None)
        self.outputs.pop(name, None)
        for suffix in ("_pre", "_post"):
            hook_name = f"{name}{suffix}"
            if hook_name in self.hooks:
                self.hooks[hook_name].remove()
                del self.hooks[hook_name]


class ModuleInputCatcher:
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device
        self.hooks = {}
        self.inputs = {}

    def attach(self, module: nn.Module, name: str, raise_error=False):
        assert name not in self.inputs
        assert name not in self.hooks
        self.inputs[name] = []

        def input_hook(module, args, kwargs):
            self.inputs[name].append(
                {
                    "args": transfer_to_device(args, self.device),
                    "kwargs": transfer_to_device(kwargs, self.device),
                }
            )
            if raise_error:
                raise RuntimeError("Reached target layer inputs")

        self.hooks[name] = module.register_forward_pre_hook(
            input_hook, with_kwargs=True
        )

    def detach(self, name):
        assert name in self.inputs
        assert name in self.hooks
        self.inputs.pop(name, None)
        self.hooks.pop(name).remove()


class ModuleOutputCatcher:
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device
        self.hooks = {}
        self.outputs = {}

    def attach(self, module: nn.Module, name: str, raise_error=False):
        assert name not in self.outputs
        assert name not in self.hooks
        self.outputs[name] = []

        def output_hook(module, args, output):
            self.outputs[name].append(transfer_to_device(output, self.device))
            if raise_error:
                raise RuntimeError("Reached target layer output")

        self.hooks[name] = module.register_forward_hook(output_hook)

    def detach(self, name):
        assert name in self.outputs
        assert name in self.hooks
        self.outputs.pop(name, None)
        self.hooks.pop(name).remove()
