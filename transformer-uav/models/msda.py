import torch
import torch.nn as nn
from typing import List
from .fusion import ConvBNAct


class MultiScaleDilatedAttention(nn.Module):
    """
    Multi-Scale Dilated Attention (MSDA).
    Applies parallel depthwise convolutions with multiple dilation rates
    combined with multi-head attention to capture features across varying scales.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dilation_rates: List[int] = [1, 2, 3],
        qkv_bias: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dilation_rates = dilation_rates

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        # Partition channels evenly across dilation rates
        k = len(dilation_rates)
        base_c = dim // k
        remainder = dim % k
        self.split_sizes = [base_c + 1 if i < remainder else base_c for i in range(k)]

        self.dilated_dw = nn.ModuleList([
            nn.Conv2d(
                c_i,
                c_i,
                kernel_size=3,
                stride=1,
                padding=rate,
                dilation=rate,
                groups=c_i,
                bias=False
            )
            for c_i, rate in zip(self.split_sizes, dilation_rates)
        ])


        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature map tensor of shape (B, C, H, W)
        Returns:
            Refined feature map of shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # 1. Multi-Scale Dilated Convolution Branch
        chunks = torch.split(x, self.split_sizes, dim=1)
        dilated_out = torch.cat([
            conv(chunk) for conv, chunk in zip(self.dilated_dw, chunks)
        ], dim=1)

        
        # 2. Attention Branch across flattened tokens
        tokens = dilated_out.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
        norm_tokens = self.norm(tokens)
        
        qkv = self.qkv(norm_tokens).reshape(B, H * W, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        attn_out = (attn @ v).transpose(1, 2).reshape(B, H * W, C)
        attn_out = self.proj(attn_out)

        # Residual connection and FFN
        tokens = tokens + attn_out
        tokens = tokens + self.mlp(self.norm(tokens))

        out = tokens.permute(0, 2, 1).reshape(B, C, H, W)
        return out + x
