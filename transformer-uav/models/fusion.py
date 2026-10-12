import torch
import torch.nn as nn
from typing import Optional


class ConvBNAct(nn.Module):
    """
    Standard Convolution + BatchNorm2d + Activation block.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: Optional[int] = None,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        act: bool = True
    ):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class RepConv(nn.Module):
    """
    RepVGG-style Convolution block used in RT-DETR and YOLO necks.
    In training: combines 3x3 conv, 1x1 conv, and identity shortcut.
    In eval: can be re-parameterized into a single 3x3 convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, act: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

        self.conv1 = ConvBNAct(in_channels, out_channels, kernel_size=3, stride=1, padding=1, act=False)
        self.conv2 = ConvBNAct(in_channels, out_channels, kernel_size=1, stride=1, padding=0, act=False)
        self.identity = nn.BatchNorm2d(out_channels) if in_channels == out_channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x) + self.conv2(x)
        if self.identity is not None:
            out = out + self.identity(x)
        return self.act(out)


class FusionBlock(nn.Module):
    """
    Fusion block as depicted in the Neck (RepC3 / CSP-style fusion).
    Fuses concatenated feature maps using two 1x1 projections and repeated RepConv layers.
    """
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int = 3, expansion: float = 1.0):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.cv1 = ConvBNAct(in_channels, hidden_channels, kernel_size=1, stride=1)
        self.cv2 = ConvBNAct(in_channels, hidden_channels, kernel_size=1, stride=1)
        self.blocks = nn.Sequential(*[
            RepConv(hidden_channels, hidden_channels) for _ in range(num_blocks)
        ])
        self.cv3 = ConvBNAct(hidden_channels, out_channels, kernel_size=1, stride=1) if hidden_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(self.blocks(self.cv1(x)) + self.cv2(x))


class DownsampleConv3x3(nn.Module):
    """
    Downsampling block: Conv 3x3 with stride=2, padding=1, followed by BN and SiLU.
    Used in the bottom-up pathway (orange pill labeled Conv3x3).
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBNAct(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpSampleBlock(nn.Module):
    """
    UpSampling block used in the top-down pathway (blue pill labeled UpSample).
    Upsamples spatial dimensions by 2x using nearest/bilinear interpolation.
    """
    def __init__(self, in_channels: Optional[int] = None, out_channels: Optional[int] = None, mode: str = "nearest"):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode=mode)
        if in_channels is not None and out_channels is not None and in_channels != out_channels:
            self.proj = ConvBNAct(in_channels, out_channels, kernel_size=1, stride=1)
        else:
            self.proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.upsample(x))

