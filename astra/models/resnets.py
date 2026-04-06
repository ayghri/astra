# Licensed under CC BY-NC 4.0 (see LICENSE or https://creativecommons.org/licenses/by-nc/4.0/)
# Non-commercial use only; contact us for commercial licensing.
from typing import Tuple, Union
import torch
from torch import nn
import torch.nn.functional as F
from .utils import weights_init
from .utils import get_activation


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self, in_planes, planes, stride=1, affine=True, activation_name="ReLU"
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine)

        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                conv1x1(in_planes, self.expansion * planes, stride=stride),
                nn.BatchNorm2d(self.expansion * planes, affine=affine),
            )

        # self.activation = nn.ReLU(inplace=True)
        self.activation = get_activation(activation_name)(inplace=True)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))

        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)

        out = self.activation(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self, in_planes, planes, stride=1, affine=True, activation_name="ReLU"
    ):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine)

        self.conv2 = conv3x3(planes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine)

        self.conv3 = conv1x1(planes, self.expansion * planes)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes, affine=affine)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                conv1x1(in_planes, self.expansion * planes, stride=stride),
                nn.BatchNorm2d(self.expansion * planes, affine=affine),
            )

        # self.relu = nn.ReLU(inplace=True)

        self.activation = get_activation(activation_name)(in_place=True)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.activation(self.bn2(self.conv2(out)))

        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)

        out = self.activation(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        num_blocks: Tuple[int, ...],
        in_planes: Tuple[int, ...],
        num_classes=10,
        affine=True,
        maxpool=False,
        activation_name="ReLU",
        **kwargs,
    ):
        super(ResNet, self).__init__()
        assert len(num_blocks) == len(in_planes)
        self.in_planes = in_planes
        self.num_blocks = num_blocks

        # Adapted for CIFAR: 3x3 kernel, stride 1
        self.conv1 = nn.Conv2d(
            3,
            in_planes[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_planes[0], affine=affine)
        # self.stack = self._make_stack(
        #     block, num_blocks=num_blocks, in_planes=in_planes, affine=affine
        # )
        self.maxpool = nn.Sequential()
        if maxpool:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        i = 1
        self.layers = [
            self._make_layer(
                block,
                in_planes[0],
                in_planes[0],
                num_blocks[0],
                stride=1,
                affine=affine,
                activation_name=activation_name,
            )
        ]

        previous_planes = in_planes[0] * block.expansion
        for i in range(1, len(in_planes)):
            self.layers.append(
                self._make_layer(
                    block,
                    previous_planes,
                    in_planes[i],
                    num_blocks[i],
                    stride=2,
                    affine=affine,
                    activation_name=activation_name,
                )
            )
            previous_planes = in_planes[i] * block.expansion

        for i, layer in enumerate(self.layers):
            self.register_module(f"layer{i + 1}", layer)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(in_planes[-1] * block.expansion, num_classes)

        self.apply(weights_init)

    # def _make_stack(self, block, num_blocks, in_planes, affine=True):
    #     layers = [
    #         self._make_layer(
    #             block,
    #             in_planes[0],
    #             in_planes[0],
    #             num_blocks[0],
    #             stride=1,
    #             affine=affine,
    #         )
    #     ]
    #     previous_planes = in_planes[0] * block.expansion
    #     for i in range(1, len(in_planes)):
    #         layers.append(
    #             self._make_layer(
    #                 block,
    #                 previous_planes,
    #                 in_planes[i],
    #                 num_blocks[i],
    #                 stride=2,
    #                 affine=affine,
    #             )
    #         )
    #         previous_planes = in_planes[i] * block.expansion
    #     return nn.Sequential(*layers)

    def _make_layer(
        self,
        block: Union[Bottleneck, BasicBlock],
        in_planes: int,
        planes: int,
        num_blocks: int,
        stride: int,
        affine: bool,
        activation_name,
    ):
        layers = [
            block(
                in_planes,
                planes,
                stride,
                affine=affine,
                activation_name=activation_name,
            )
        ]

        curr_in = planes * block.expansion

        for _ in range(num_blocks - 1):
            layers.append(
                block(
                    curr_in,
                    planes,
                    stride=1,
                    affine=affine,
                    activation_name=activation_name,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        for layer in self.layers:
            out = layer(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out


def get_resnet(depth, num_classes, pretrained=False, imagenet=False, **kwargs):
    if depth == 18:
        return ResNet(
            BasicBlock,
            (2, 2, 2, 2),
            in_planes=(64, 128, 256, 512),
            num_classes=num_classes,
            **kwargs,
        )
    elif depth == 32:
        return ResNet(
            BasicBlock,
            (5, 5, 5),
            in_planes=(32, 64, 128),
            num_classes=num_classes,
            **kwargs,
        )

    elif depth == 34:
        return ResNet(
            BasicBlock,
            (3, 4, 6, 3),
            in_planes=(64, 128, 256, 512),
            num_classes=num_classes,
            **kwargs,
        )

    elif depth == 50:
        return ResNet(
            Bottleneck,
            (3, 4, 6, 3),
            in_planes=(64, 128, 256, 512),
            num_classes=num_classes,
            maxpool=True,
            **kwargs,
        )
    elif depth == 101:
        return ResNet(
            Bottleneck,
            (3, 4, 23, 3),
            in_planes=(64, 128, 256, 512),
            num_classes=num_classes,
            **kwargs,
        )
    elif depth == 152:
        return ResNet(
            Bottleneck,
            (3, 8, 36, 3),
            in_planes=(64, 128, 256, 512),
            num_classes=num_classes,
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported ResNet depth: {depth}")
