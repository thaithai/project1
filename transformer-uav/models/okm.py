import torch
import torch.nn as nn
from typing import Optional
from .fusion import ConvBNAct


class FrequencyGatedModule(nn.Module):
    """
    Frequency-Gated Module (FGM) from OKNet (AAAI 2024).
    Applies 2D Fast Fourier Transform (FFT) to capture global frequency context.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dwconv1 = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
        self.dwconv2 = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
        self.alpha = nn.Parameter(torch.zeros(dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x1 = self.dwconv1(x)
        x2 = self.dwconv2(x)

        # 2D Real-to-Complex FFT across spatial dimensions
        x2_fft = torch.fft.fft2(x2, norm="backward")
        out = x1 * x2_fft
        out = torch.fft.ifft2(out, dim=(-2, -1), norm="backward")
        out = torch.abs(out)

        return out * self.alpha + x * self.beta


class OmniKernelModule(nn.Module):
    """
    Omni-Kernel Module (OKM) from OKNet.
    Comprises three distinct branches:
      1. Local branch: 1x1 depthwise conv for fine local details.
      2. Large-kernel branch: Multi-axis and 2D large depthwise convolutions
         (e.g., 1xK, Kx1, KxK) for multi-grained large receptive fields.
      3. Global branch: Frequency-gated attention (FGM) + Spatial-Channel Attention (SCA).
    """
    def __init__(self, dim: int, kernel_size: int = 7):
        super().__init__()
        self.dim = dim
        pad = kernel_size // 2

        self.in_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0),
            nn.GELU()
        )

        # Large-kernel depthwise branches
        self.dw_h = nn.Conv2d(dim, dim, kernel_size=(1, kernel_size), padding=(0, pad), stride=1, groups=dim)
        self.dw_w = nn.Conv2d(dim, dim, kernel_size=(kernel_size, 1), padding=(pad, 0), stride=1, groups=dim)
        self.dw_2d = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=pad, stride=1, groups=dim)

        # Local branch
        self.dw_local = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=dim)

        # Global branch: Frequency Channel Attention + Spatial Channel Attention
        self.fca_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fca_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=True)
        self.fgm = FrequencyGatedModule(dim)

        self.sca_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.sca_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=True)

        self.act = nn.ReLU(inplace=True)
        self.out_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        feat = self.in_conv(x)

        # --- Global Branch (Frequency & Channel Attention) ---
        fca_att = self.fca_conv(self.fca_pool(feat))
        feat_fft = torch.fft.fft2(feat, norm="backward")
        feat_fft = fca_att * feat_fft
        feat_fca = torch.fft.ifft2(feat_fft, dim=(-2, -1), norm="backward")
        feat_fca = torch.abs(feat_fca)

        sca_att = self.sca_conv(self.sca_pool(feat_fca))
        feat_sca = sca_att * feat_fca
        feat_global = self.fgm(feat_sca)

        # --- Fusion of Local, Large-kernel, and Global branches ---
        out = (
            residual
            + self.dw_h(feat)
            + self.dw_w(feat)
            + self.dw_2d(feat)
            + self.dw_local(feat)
            + feat_global
        )
        out = self.act(out)
        return self.out_conv(out)


class OKMFusion(nn.Module):
    """
    OKMFusion block:
    Fuses multi-scale concatenated feature maps using Cross-Stage Partial (CSP) structure
    powered by the Omni-Kernel Module (OKM).
    
    Inputs: Concatenated feature maps from:
      - UpSample (from middle Fusion, stride 8)
      - Conv1x1(S2) (stride 8)
      - SPDConv(Conv1x1(S1)) (stride 8)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_okm_blocks: int = 1,
        kernel_size: int = 7,
        expansion: float = 0.5
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        
        self.cv1 = ConvBNAct(in_channels, hidden_channels, kernel_size=1, stride=1)
        self.cv2 = ConvBNAct(in_channels, hidden_channels, kernel_size=1, stride=1)
        
        self.okm_blocks = nn.Sequential(*[
            OmniKernelModule(hidden_channels, kernel_size=kernel_size)
            for _ in range(num_okm_blocks)
        ])
        
        self.cv3 = ConvBNAct(hidden_channels * 2, out_channels, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CSP structure: split through cv1 (OKM path) and cv2 (shortcut path)
        y1 = self.okm_blocks(self.cv1(x))
        y2 = self.cv2(x)
        return self.cv3(torch.cat([y1, y2], dim=1))

