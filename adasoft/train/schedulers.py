"""Learning rate schedulers used in training.

Provides a simple warmup scheduler implemented as a subclass of
``torch.optim.lr_scheduler.LRScheduler`` so it plugs cleanly into
standard PyTorch training code.
"""

from typing import List

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class LinearWarmupScheduler(LRScheduler):
    """Linear LR warmup scheduler.

    For each parameter group, increases the learning rate linearly from
    ``warmup_start_lr`` to its initial optimizer learning rate over
    ``num_warmup_steps`` calls to :meth:`step`. After the warmup phase,
    the learning rate is kept constant at the initial optimizer value.

    This scheduler is step-based (not epoch-based): call
    :meth:`step` once per optimizer update.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        num_warmup_steps: int,
        warmup_start_lr: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        self.num_warmup_steps = int(num_warmup_steps)
        self.warmup_start_lr = float(warmup_start_lr)

        # Store the target ("base") LR for each param group from the optimizer
        self._base_lrs: List[float] = [
            float(group.get("lr", 0.0)) for group in optimizer.param_groups
        ]

        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self) -> List[float]:
        """Compute current learning rates for all parameter groups."""

        if self.num_warmup_steps <= 0:
            return self._base_lrs

        # ``last_epoch`` starts at 0 after the first call to ``step()``.
        step = self.last_epoch

        if step < self.num_warmup_steps:
            frac = float(step + 1) / float(self.num_warmup_steps)
            return [
                self.warmup_start_lr + frac * (base_lr - self.warmup_start_lr)
                for base_lr in self._base_lrs
            ]

        return self._base_lrs
