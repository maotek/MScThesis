import torch
import torch.nn as nn


def lrelu(x: torch.Tensor) -> torch.Tensor:
    return torch.maximum(x * 0.2, x)


def identity_initializer(conv: nn.Conv2d) -> None:
    with torch.no_grad():
        conv.weight.zero_()
        cx, cy = conv.kernel_size[0] // 2, conv.kernel_size[1] // 2

        for i in range(conv.in_channels):
            conv.weight[i, i, cx, cy] = 1.0

        if conv.bias is not None:
            conv.bias.zero_()


class nm(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.w0 = nn.Parameter(torch.tensor(1.0))
        self.w1 = nn.Parameter(torch.tensor(0.0))
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w0 * x + self.w1 * self.bn(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, rate: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=rate,
            dilation=rate,
            bias=True,
        )
        identity_initializer(self.conv)
        self.normalizer = nm(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.normalizer(x)
        x = lrelu(x)
        return x


class FullyConv(nn.Module):
    def __init__(self, in_channels: int = 5, out_channels: int = 3) -> None:
        super().__init__()

        self.g_conv1 = ConvBlock(in_channels, 32, rate=1)
        self.g_conv2 = ConvBlock(32, 32, rate=2)
        self.g_conv3 = ConvBlock(32, 32, rate=4)
        self.g_conv4 = ConvBlock(32, 32, rate=8)
        self.g_conv5 = ConvBlock(32, 32, rate=16)
        self.g_conv6 = ConvBlock(32, 32, rate=32)
        self.g_conv7 = ConvBlock(32, 32, rate=64)
        self.g_conv8 = ConvBlock(32, 32, rate=128)
        self.g_conv9 = ConvBlock(32, 32, rate=1)

        self.g_conv_last = nn.Conv2d(32, out_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        net = self.g_conv1(input)
        net = self.g_conv2(net)
        net = self.g_conv3(net)
        net = self.g_conv4(net)
        net = self.g_conv5(net)
        net = self.g_conv6(net)
        net = self.g_conv7(net)
        net = self.g_conv8(net)
        net = self.g_conv9(net)
        net = self.g_conv_last(net)
        return net