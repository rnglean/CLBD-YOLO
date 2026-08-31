import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import CondConv2d
from ultralytics.nn.modules.conv import autopad
from ultralytics.nn.modules.block import C3k, C3k2


class DynamicConv_Single(nn.Module):
    """Single conditional dynamic convolution layer."""

    def __init__(
        self,
        in_features,
        out_features,
        kernel_size=1,
        stride=1,
        padding="",
        dilation=1,
        groups=1,
        bias=False,
        num_experts=4,
    ):
        super().__init__()

        self.routing = nn.Linear(in_features, num_experts)

        self.cond_conv = CondConv2d(
            in_features,
            out_features,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            num_experts,
        )

    def forward(self, x):
        pooled_inputs = F.adaptive_avg_pool2d(x, 1).flatten(1)
        routing_weights = torch.sigmoid(self.routing(pooled_inputs))
        x = self.cond_conv(x, routing_weights)
        return x


class DynamicConv(nn.Module):
    default_act = nn.SiLU()

    def __init__(
        self,
        c1,
        c2,
        k=1,
        s=1,
        p=None,
        g=1,
        d=1,
        act=True,
        num_experts=4,
    ):
        super().__init__()

        self.conv = nn.Sequential(
            DynamicConv_Single(
                c1,
                c2,
                kernel_size=k,
                stride=s,
                padding=autopad(k, p, d),
                dilation=d,
                groups=g,
                num_experts=num_experts,
            ),
            nn.BatchNorm2d(c2),
            self.default_act
            if act is True
            else act
            if isinstance(act, nn.Module)
            else nn.Identity(),
        )

    def forward(self, x):
        return self.conv(x)


class GhostModule(nn.Module):
    def __init__(
        self,
        inp,
        oup,
        kernel_size=1,
        ratio=2,
        dw_size=3,
        stride=1,
        act_layer=nn.SiLU,
        num_experts=4,
    ):
        super().__init__()

        self.oup = oup

        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = DynamicConv(
            inp,
            init_channels,
            kernel_size,
            stride,
            num_experts=num_experts,
        )

        self.cheap_operation = DynamicConv(
            init_channels,
            new_channels,
            dw_size,
            1,
            g=init_channels,
            num_experts=num_experts,
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)

        out = torch.cat([x1, x2], dim=1)

        return out[:, : self.oup, :, :]


class C3k_GhostDynamicConv(C3k):
    def __init__(
        self,
        c1,
        c2,
        n=1,
        shortcut=False,
        g=1,
        e=0.5,
        k=3,
    ):
        super().__init__(c1, c2, n, shortcut, g, e, k)

        c_ = int(c2 * e)

        self.m = nn.Sequential(
            *(GhostModule(c_, c_) for _ in range(n))
        )


class C3k2_GhostDynamicConv(C3k2):
    def __init__(
        self,
        c1,
        c2,
        n=1,
        c3k=False,
        e=0.5,
        g=1,
        shortcut=True,
    ):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)

        self.m = nn.ModuleList(
            C3k_GhostDynamicConv(
                self.c,
                self.c,
                2,
                shortcut,
                g,
            )
            if c3k
            else GhostModule(self.c, self.c)
            for _ in range(n)
        )
