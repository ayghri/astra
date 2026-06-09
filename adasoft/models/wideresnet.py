
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type


class BasicBlock(nn.Module):
    """
    Wide Residual Network basic block.
    """

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        stride: int,
        dropout: float = 0.0,
    ):
        super(BasicBlock, self).__init__()
        # Pre-activation BN and ReLU
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_planes,
            out_planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.dropout = dropout
        self.in_equal_out = in_planes == out_planes

        # Shortcut connection to match input and output dimensions
        if self.in_equal_out:
            self.shortcut = None
        else:
            # The 1x1 conv is for matching dimensions.
            # Per the TF reference, this is applied to the *pre-activated* input.
            self.shortcut = nn.Conv2d(
                in_planes,  # Note: input planes, not output
                out_planes,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            )

    def forward(self, x):
        # Pre-activate the input
        preactivated_x = self.relu1(self.bn1(x))

        # Handle the shortcut connection based on the reference logic
        if self.in_equal_out:
            # Identity shortcut
            shortcut = x
            # The first conv uses the pre-activated input
            out = self.conv1(preactivated_x)
        else:
            # The shortcut projection is applied to the PRE-ACTIVATED input
            shortcut = self.shortcut(preactivated_x)
            # The main path also uses the PRE-ACTIVATED input
            out = self.conv1(preactivated_x)

        # Second part of the main path
        out = self.bn2(out)
        out = self.relu2(out)
        if self.dropout > 0:
            out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.conv2(out)

        # Add the shortcut to the main path output
        return torch.add(shortcut, out)


class NetworkBlock(nn.Module):
    """
    A block of sequential BasicBlocks.
    """

    def __init__(
        self,
        nb_layers: int,
        in_planes: int,
        out_planes: int,
        block: Type[BasicBlock],
        stride: int,
        dropRate: float = 0.0,
    ):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, dropRate
        )

    def _make_layer(
        self, block, in_planes, out_planes, nb_layers, stride, dropRate
    ):
        layers = []
        for i in range(int(nb_layers)):
            # More readable version of the input plane logic
            current_in_planes = in_planes if i == 0 else out_planes
            current_stride = stride if i == 0 else 1
            layers.append(
                block(
                    current_in_planes,
                    out_planes,
                    current_stride,
                    dropRate,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    """
    Wide Residual Network with varying depth and width.
    This implementation is corrected to match the TF reference.
    """

    def __init__(
        self,
        depth: int = 22,
        widen_factor: int = 2,
        num_classes: int = 10,
        drop_rate: float = 0.3,
        small_dense_density: float = 1.0,
    ):
        super(WideResNet, self).__init__()

        # Logic for small_dense_density is re-introduced here
        small_dense_multiplier = np.sqrt(small_dense_density)
        nChannels = [
            16,
            16 * widen_factor,
            32 * widen_factor,
            64 * widen_factor,
        ]
        nChannels = [int(c * small_dense_multiplier) for c in nChannels]
        # nChannels[0] must be at least 1
        nChannels[0] = max(nChannels[0], 1)

        # Ensure depth is valid
        assert (depth - 4) % 6 == 0, "Depth must be 6n+4"
        n = (depth - 4) // 6
        block = BasicBlock

        # 1st conv before any network block
        self.conv1 = nn.Conv2d(
            3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        # 1st block group
        self.block1 = NetworkBlock(
            n, nChannels[0], nChannels[1], block, 1, drop_rate
        )
        # 2nd block group
        self.block2 = NetworkBlock(
            n, nChannels[1], nChannels[2], block, 2, drop_rate
        )
        # 3rd block group
        self.block3 = NetworkBlock(
            n, nChannels[2], nChannels[3], block, 2, drop_rate
        )
        # Global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels_out = nChannels[3]

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He initialization
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                # Default initialization is kaiming_uniform, which is good.
                # Zeroing bias is also a common practice.
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)

        # Final pre-activation and pooling, as in TF reference
        out = self.relu(self.bn1(out))
        # The pooling size assumes a 32x32 input (e.g., CIFAR) which becomes 8x8 here
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(-1, self.nChannels_out)
        out = self.fc(out)
        return out


def get_wideresnet(
    depth: int = 22,
    widen_factor: int = 2,
    num_classes: int = 10,
    drop_rate: float = 0.3,
    small_dense_density: float = 1.0,
) -> WideResNet:
    """
    Factory function to create a WideResNet instance.
    """
    return WideResNet(
        depth=depth,
        widen_factor=widen_factor,
        num_classes=num_classes,
        drop_rate=drop_rate,
        small_dense_density=small_dense_density,
    )