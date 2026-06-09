"""
Licensed under CC BY-NC 4.0 (see LICENSE or https://creativecommons.org/licenses/by-nc/4.0/)
Non-commercial use only; contact us for commercial licensing.
"""

from typing import List, Tuple

import torch
import torch.nn as nn


class StatsCollector:
    def __init__(
        self,
        named_parameters: List[Tuple[str, nn.Parameter]],
        refresh_every: int,
        zero_threshold: float = 1e-6,
        device_name="cpu",
    ):
        self.named_params = named_parameters
        self.refresh_every = refresh_every
        self.zero_threshold = zero_threshold
        self.device = torch.device(device_name)

        self.nz_bitmasks = {
            p: (p.data.abs() >= zero_threshold).detach().to(self.device)
            for n, p in named_parameters
        }
        self.p_checkpoints = {
            p: p.data.detach().clone().to(self.device)
            for n, p in named_parameters
        }

        self.t = 0

    def step(self):
        if self.t < self.refresh_every:
            self.t = self.t + 1
            return None
        stats = {}
        for n, p in self.named_params:
            nz_bitmask = (
                (p.data.abs() >= self.zero_threshold).detach().to(self.device)
            )

            stats[n + ".sparsity"] = 1 - nz_bitmask.sum() / p.numel()
            stats[n + ".bitmask_change"] = (
                self.nz_bitmasks[p] != nz_bitmask
            ).sum() / p.numel()

            stats[n + ".delta"] = torch.norm(
                self.p_checkpoints[p] - p.data.detach().clone().cpu()
            )
            stats[n + ".delta_rel"] = stats[n + ".delta"] / torch.norm(
                self.p_checkpoints[p]
            )

            self.nz_bitmasks[p] = nz_bitmask
            self.p_checkpoints[p] = p.data.detach().clone().to(self.device)

        self.t = 0
        return stats
