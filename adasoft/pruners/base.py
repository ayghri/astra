from abc import ABC, abstractmethod
import torch
import torch.nn as nn

def is_prunable(module):
    # This would be expanded to include PrunableConv2d, etc.
    return isinstance(module, PrunableLinear)

class PruningStrategy(ABC):
    """
    Abstract Base Class for all pruning strategies.

    This class provides a clear and extensible interface for implementing pruning
    algorithms. It is designed to work with a PruningScheduler and prunable
    wrapper layers (e.g., PrunableLinear).

    The core logic is divided into two abstract methods that subclasses must implement:
    1. compute_saliency: Determines the importance of each weight.
    2. compute_mask: Uses the saliency scores to produce a new binary mask.
    """

    def __init__(self, sparsity_level: float):
        """
        Initializes the strategy with a target sparsity level.

        Args:
            sparsity_level (float): The target fraction of weights to be zero (e.g., 0.8 for 80% sparsity).
        """
        if not (0.0 <= sparsity_level < 1.0):
            raise ValueError("Sparsity level must be between 0.0 and 1.0")
        self.sparsity_level = sparsity_level

    @abstractmethod
    def compute_saliency(self, module: nn.Module, optimizer: torch.optim.Optimizer) -> torch.Tensor:
        """
        Computes the importance scores (saliency) for the parameters of a module.

        This method encapsulates the logic for "what" makes a weight important.
        For example, it could be weight magnitude, gradient magnitude, or a combination
        of weights and activations.

        Args:
            module (nn.Module): The prunable module (e.g., PrunableLinear) being considered.
            optimizer (torch.optim.Optimizer): The optimizer, provided in case saliency
                                              is based on gradients or optimizer states.

        Returns:
            torch.Tensor: A tensor of the same shape as the module's weight, containing
                          the importance score for each weight.
        """
        pass

    @abstractmethod
    def compute_mask(self, saliency: torch.Tensor, current_mask: torch.Tensor) -> torch.Tensor:
        """
        Computes a new mask based on saliency scores and the current mask.

        This method encapsulates the "how" of pruning. Given importance scores,
        it applies a rule (e.g., thresholding) to generate the new sparsity mask.

        Args:
            saliency (torch.Tensor): The importance scores calculated by `compute_saliency`.
            current_mask (torch.Tensor): The existing mask on the module. This is the
                                         equivalent of `default_mask` in the reference code.

        Returns:
            torch.Tensor: The new binary mask to be applied to the module.
        """
        pass

    def step(self, model: nn.Module, optimizer: torch.optim.Optimizer):
        """
        The main entry point called by the PruningScheduler.

        This method orchestrates the pruning update for all prunable modules in a model.
        It handles the boilerplate of iterating, calling the abstract methods,
        applying the new mask, and resetting optimizer states where necessary.
        """
        for module in filter(is_prunable, model.modules()):
            # 1. Calculate importance scores for the current module.
            saliency = self.compute_saliency(module, optimizer)
            if saliency is None:
                continue

            current_mask = module.weight_mask.data

            # 2. Compute the new mask using the strategy's core logic.
            new_mask = self.compute_mask(saliency, current_mask)

            # 3. Detect which weights (if any) are being "grown" (0 -> 1).
            grow_mask = (new_mask > current_mask)

            # 4. Apply the new mask to the module.
            module.weight_mask.data = new_mask.to(current_mask.dtype)

            # 5. If any weights were grown, reset their optimizer states.
            if torch.any(grow_mask):
                self._reset_optimizer_state_for_mask(optimizer, module.weight, grow_mask)

    def _reset_optimizer_state_for_mask(self, optimizer: torch.optim.Optimizer, param: nn.Parameter, mask: torch.Tensor):
        """
        A helper utility to zero out optimizer state (e.g., momentum) for weights
        that have just been unpruned.
        """
        if param not in optimizer.state:
            return

        param_state = optimizer.state[param]
        for key in param_state:
            if isinstance(param_state[key], torch.Tensor) and param_state[key].shape == mask.shape:
                param_state[key][mask] = 0.0