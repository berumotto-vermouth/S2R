import copy
import math
import warnings
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn, Tensor
# from torchvision.ops import StochasticDepth

from .misc import Conv2dNormActivation, SqueezeExcitation
from .utils import _make_divisible


# @dataclass
# class _MBConvConfig:
#     expand_ratio: float
#     kernel: int
#     stride: int
#     input_channels: int
#     out_channels: int
#     num_layers: int
#     block: Callable[..., nn.Module]

#     @staticmethod
#     def adjust_channels(channels: int, width_mult: float, min_value: Optional[int] = None) -> int:
#         return _make_divisible(channels * width_mult, 8, min_value)


# class MBConvConfig(_MBConvConfig):
#     # Stores information listed at Table 1 of the EfficientNet paper & Table 4 of the EfficientNetV2 paper
#     def __init__(
#         self,
#         expand_ratio: 4,
#         kernel: 3,
#         stride: 1,
#         input_channels: int,
#         out_channels: int,
#         num_layers: int,
#         width_mult: float = 1.0,
#         depth_mult: float = 1.0,
#         block: Optional[Callable[..., nn.Module]] = None,
#     ) -> None:
#         input_channels = self.adjust_channels(input_channels, width_mult)
#         out_channels = self.adjust_channels(out_channels, width_mult)
#         num_layers = self.adjust_depth(num_layers, depth_mult)
#         if block is None:
#             block = MBConv
#         super().__init__(expand_ratio, kernel, stride, input_channels, out_channels, num_layers, block)

#     @staticmethod
#     def adjust_depth(num_layers: int, depth_mult: float):
#         return int(math.ceil(num_layers * depth_mult))


class MBConv(nn.Module):
    def __init__(
        self,
        input_channels,
        out_channels,
        expand_ratio,
        kernel,
        stride,
        stochastic_depth_prob: float,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
        se_layer: Callable[..., nn.Module] = SqueezeExcitation,
    ) -> None:
        super().__init__()

        if not (1 <= stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = stride == 1 and input_channels == out_channels
        self.inp_dim = input_channels

        layers: List[nn.Module] = []
        activation_layer = nn.SiLU

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # expand
        expanded_channels =input_channels * expand_ratio
        if expanded_channels != input_channels:
            layers.append(
                Conv2dNormActivation(
                    input_channels,
                    expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # depthwise
        layers.append(
            Conv2dNormActivation(
                expanded_channels,
                expanded_channels,
                kernel_size=kernel,
                stride=stride,
                groups=expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )

        # squeeze and excitation
        squeeze_channels = max(1, input_channels // 4)
        layers.append(se_layer(expanded_channels, squeeze_channels, activation=partial(nn.SiLU, inplace=True)))

        # project
        layers.append(
            Conv2dNormActivation(
                expanded_channels, out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None
            )
        )

        self.block = nn.Sequential(*layers)
        # self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = out_channels

    def forward(self, x: Tensor, x_size) -> Tensor:
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.inp_dim, x_size[0], x_size[1])  # B Ph*Pw C
        result = self.block(x)
        # if self.use_res_connect:
        #     result = self.stochastic_depth(result)
        #     result += x
        return result




















# class InvertedResidual(nn.Module):
#     def __init__(
#         self, inp: int, oup: int, stride: int, expand_ratio: int, norm_layer: Optional[Callable[..., nn.Module]] = None
#     ) -> None:
#         super().__init__()
#         self.stride = stride
#         if stride not in [1, 2]:
#             raise ValueError(f"stride should be 1 or 2 insted of {stride}")

#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d

#         hidden_dim = int(round(inp * expand_ratio))
#         self.use_res_connect = self.stride == 1 and inp == oup

#         layers: List[nn.Module] = []
#         if expand_ratio != 1:
#             # pw
#             layers.append(
#                 Conv2dNormActivation(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6)
#             )
#         layers.extend(
#             [
#                 # dw
#                 Conv2dNormActivation(
#                     hidden_dim,
#                     hidden_dim,
#                     stride=stride,
#                     groups=hidden_dim,
#                     norm_layer=norm_layer,
#                     activation_layer=nn.ReLU6,
#                 ),
#                 # pw-linear
#                 nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
#                 norm_layer(oup),
#             ]
#         )
#         self.conv = nn.Sequential(*layers)
#         self.out_channels = oup
#         self._is_cn = stride > 1

#     def forward(self, x: Tensor) -> Tensor:
#         if self.use_res_connect:
#             return x + self.conv(x)
#         else:
#             return self.conv(x)