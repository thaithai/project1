import torch
import torch.nn as nn
from typing import List, Tuple
from .fusion import ConvBNAct


class StarBlock(nn.Module):
    """
    Star Block from 'Rewrite the Stars' (CVPR 2024).
    Key innovation: Star Operation (element-wise multiplication between two linear projections).
    Provides implicit high-dimensional non-linear mapping with low computational cost.
    """
    def __init__(self, dim: int, mlp_ratio: int = 3):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3, groups=dim, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        
        self.f1 = nn.Conv2d(dim, mlp_ratio * dim, kernel_size=1, stride=1)
        self.f2 = nn.Conv2d(dim, mlp_ratio * dim, kernel_size=1, stride=1)
        self.g = nn.Conv2d(mlp_ratio * dim, dim, kernel_size=1, stride=1)
        self.act = nn.ReLU6()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.bn(self.dwconv(x))
        x1 = self.f1(x)
        x2 = self.f2(x)
        # Star operation
        x = self.act(x1) * x2
        x = self.g(x)
        return residual + x


class StarStage(nn.Module):
    """
    Stage consisting of repeated StarBlocks.
    """
    def __init__(self, in_dim: int, out_dim: int, depth: int, downsample: bool = True):
        super().__init__()
        if downsample:
            self.downsample = ConvBNAct(in_dim, out_dim, kernel_size=3, stride=2, padding=1)
        else:
            self.downsample = nn.Identity() if in_dim == out_dim else ConvBNAct(in_dim, out_dim, kernel_size=1, stride=1)
        
        self.blocks = nn.Sequential(*[
            StarBlock(out_dim) for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        return self.blocks(x)


class StarNet(nn.Module):
    """
    StarNet Backbone.
    Produces multi-scale feature maps [S1, S2, S3, S4]:
      - S1: stride 4
      - S2: stride 8
      - S3: stride 16
      - S4: stride 32
    """
    def __init__(
        self,
        in_channels: int = 3,
        base_dim: int = 32,
        depths: List[int] = [2, 2, 4, 2],
        dims: List[int] = [32, 64, 128, 256]
    ):
        super().__init__()
        # Stem downsamples input image to stride 4
        self.stem = nn.Sequential(
            ConvBNAct(in_channels, base_dim, kernel_size=3, stride=2, padding=1),  # stride 2
            ConvBNAct(base_dim, dims[0], kernel_size=3, stride=2, padding=1)       # stride 4
        )

        # Stage 1: S1 (stride 4)
        self.stage1 = StarStage(dims[0], dims[0], depth=depths[0], downsample=False)

        # Stage 2: S2 (stride 8)
        self.stage2 = StarStage(dims[0], dims[1], depth=depths[1], downsample=True)

        # Stage 3: S3 (stride 16)
        self.stage3 = StarStage(dims[1], dims[2], depth=depths[2], downsample=True)

        # Stage 4: S4 (stride 32)
        self.stage4 = StarStage(dims[2], dims[3], depth=depths[3], downsample=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        s1 = self.stage1(x)   # (B, dims[0], H/4, W/4)
        s2 = self.stage2(s1)  # (B, dims[1], H/8, W/8)
        s3 = self.stage3(s2)  # (B, dims[2], H/16, W/16)
        s4 = self.stage4(s3)  # (B, dims[3], H/32, W/32)
        return s1, s2, s3, s4

