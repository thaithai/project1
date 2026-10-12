import torch
import torch.nn as nn
from .fusion import ConvBNAct


class SPDConv(nn.Module):
    """
    Space-to-Depth Convolution (SPDConv).
    Paper: 'No More Strided Convolutions or Pooling: A New CNN Building Block 
            for Low-Resolution Images and Small Objects' (ECML PKDD 2022).
    
    Replaces strided convolutions/pooling by slicing spatial patches into channels (4x channels, 0.5x H and W)
    without dropping any pixels, preserving fine-grained spatial features of small objects (like UAVs).
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # After space-to-depth, channels become in_channels * 4
        self.conv = ConvBNAct(in_channels * 4, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x0 = x[..., 0::2, 0::2]
        x1 = x[..., 1::2, 0::2]
        x2 = x[..., 0::2, 1::2]
        x3 = x[..., 1::2, 1::2]
        # Concatenate along channel dimension: (B, 4*C, H/2, W/2)
        x_spd = torch.cat([x0, x1, x2, x3], dim=1)
        return self.conv(x_spd)

