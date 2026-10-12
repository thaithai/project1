import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Union
from .fusion import ConvBNAct, FusionBlock, DownsampleConv3x3, UpSampleBlock
from .okm import OKMFusion
from .spd_conv import SPDConv


class OKMFeatureFusionNeck(nn.Module):
    """
    Feature Fusion Neck for UAV / Drone Small Object Detection.
    
    Architecture (as depicted in the diagram):
      - Top-Down Pathway:
          * feat_top (from MSDA/S4, stride 32) -> UpSample (to stride 16)
          * ConCat with S3 (stride 16) -> Middle Fusion
          * Middle Fusion -> UpSample (to stride 8)
          * ConCat with S2 (stride 8) and S1 (via SPDConv, stride 8) -> OKMFusion
      - Bottom-Up Pathway:
          * OKMFusion output (stride 8) -> Conv3x3 (stride 2, to stride 16)
          * ConCat with Middle Fusion (stride 16) -> Middle Bottom-Up Fusion
          * Middle Bottom-Up Fusion -> Conv3x3 (stride 2, to stride 32)
          * ConCat with feat_top (stride 32) -> Top Fusion
      - Output (Circle C - ConCat):
          * Aggregates multi-scale features [P3 (stride 8), P4 (stride 16), P5 (stride 32)]
          * Can flatten and concatenate along sequence dimension for Uncertainty-minimal Query Selection
    """
    def __init__(
        self,
        in_channels: List[int] = [32, 64, 128, 256],  # [S1, S2, S3, S4/MSDA]
        hidden_dim: int = 256,
        okm_kernel_size: int = 7,
        num_fusion_blocks: int = 3,
    ):
        super().__init__()
        c_s1, c_s2, c_s3, c_s4 = in_channels
        self.hidden_dim = hidden_dim

        # -----------------------------------------------------------
        # 1. Input Projections (Conv1x1 & SPDConv as shown in diagram)
        # -----------------------------------------------------------
        # S1 (stride 4): Conv1x1 -> SPDConv (stride 4 -> stride 8)
        self.conv1x1_s1 = ConvBNAct(c_s1, hidden_dim // 4, kernel_size=1, stride=1)
        self.spd_conv = SPDConv(hidden_dim // 4, hidden_dim)

        # S2 (stride 8): Conv1x1 (keeps stride 8)
        self.conv1x1_s2 = ConvBNAct(c_s2, hidden_dim, kernel_size=1, stride=1)

        # S3 (stride 16): Conv1x1 (keeps stride 16)
        self.conv1x1_s3 = ConvBNAct(c_s3, hidden_dim, kernel_size=1, stride=1)

        # S4 / MSDA (stride 32): Conv1x1 projection
        self.conv1x1_top = ConvBNAct(c_s4, hidden_dim, kernel_size=1, stride=1)

        # -----------------------------------------------------------
        # 2. Top-Down Pathway
        # -----------------------------------------------------------
        # Top UpSample: stride 32 -> stride 16
        self.upsample_top = UpSampleBlock(hidden_dim, hidden_dim)
        
        # Middle Top-Down Fusion: takes [UpSample(feat_top) + S3_proj] -> 2 * hidden_dim
        self.fusion_mid_td = FusionBlock(
            in_channels=hidden_dim * 2,
            out_channels=hidden_dim,
            num_blocks=num_fusion_blocks
        )

        # Lower UpSample: stride 16 -> stride 8
        self.upsample_mid = UpSampleBlock(hidden_dim, hidden_dim)

        # OKMFusion: takes [UpSample(Mid_Fusion) + S2_proj + SPDConv(S1)] -> 3 * hidden_dim
        self.okm_fusion = OKMFusion(
            in_channels=hidden_dim * 3,
            out_channels=hidden_dim,
            num_okm_blocks=1,
            kernel_size=okm_kernel_size
        )

        # -----------------------------------------------------------
        # 3. Bottom-Up Pathway
        # -----------------------------------------------------------
        # Bottom Conv3x3: stride 8 -> stride 16 (stride=2)
        self.conv3x3_low_bu = DownsampleConv3x3(hidden_dim, hidden_dim)

        # Middle Bottom-Up Fusion: takes [Conv3x3(OKMFusion) + Mid_Fusion_td] -> 2 * hidden_dim
        self.fusion_mid_bu = FusionBlock(
            in_channels=hidden_dim * 2,
            out_channels=hidden_dim,
            num_blocks=num_fusion_blocks
        )

        # Top Conv3x3: stride 16 -> stride 32 (stride=2)
        self.conv3x3_top_bu = DownsampleConv3x3(hidden_dim, hidden_dim)

        # Top Bottom-Up Fusion: takes [Conv3x3(Mid_Fusion_bu) + feat_top] -> 2 * hidden_dim
        self.fusion_top_bu = FusionBlock(
            in_channels=hidden_dim * 2,
            out_channels=hidden_dim,
            num_blocks=num_fusion_blocks
        )

    def forward(
        self,
        s1: torch.Tensor,
        s2: torch.Tensor,
        s3: torch.Tensor,
        s4_msda: torch.Tensor,
        return_flattened_tokens: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]]:
        """
        Forward pass through the Feature Fusion Neck.

        Args:
            s1: Feature map from StarNet stage 1 (stride 4), shape (B, C1, H, W)
            s2: Feature map from StarNet stage 2 (stride 8), shape (B, C2, H/2, W/2)
            s3: Feature map from StarNet stage 3 (stride 16), shape (B, C3, H/4, W/4)
            s4_msda: Feature map from MSDA/Stage 4 (stride 32), shape (B, C4, H/8, W/8)
            return_flattened_tokens: If True, also returns flattened & concatenated tokens for Query Selection.

        Returns:
            (p3, p4, p5): Multi-scale feature maps at stride 8, stride 16, and stride 32.
            If return_flattened_tokens is True, returns ((p3, p4, p5), tokens) where tokens has shape (B, Total_Tokens, hidden_dim).
        """
        # --- Step 1: Input Projections ---
        feat_top = self.conv1x1_top(s4_msda)               # (B, hidden_dim, H/8, W/8)   [stride 32]
        s3_proj = self.conv1x1_s3(s3)                       # (B, hidden_dim, H/4, W/4)   [stride 16]
        s2_proj = self.conv1x1_s2(s2)                       # (B, hidden_dim, H/2, W/2)   [stride 8]
        s1_spd = self.spd_conv(self.conv1x1_s1(s1))         # (B, hidden_dim, H/2, W/2)   [stride 8 via SPDConv]

        # --- Step 2: Top-Down Pathway ---
        # 2.1 Top UpSample to stride 16
        top_up = self.upsample_top(feat_top)                # (B, hidden_dim, H/4, W/4)
        
        # 2.2 First ConCat & Middle Fusion (stride 16)
        concat_mid_td = torch.cat([top_up, s3_proj], dim=1) # (B, 2*hidden_dim, H/4, W/4)
        feat_mid_td = self.fusion_mid_td(concat_mid_td)     # (B, hidden_dim, H/4, W/4)

        # 2.3 Lower UpSample to stride 8
        mid_up = self.upsample_mid(feat_mid_td)             # (B, hidden_dim, H/2, W/2)

        # 2.4 Second ConCat (3-way) & OKMFusion (stride 8)
        concat_low_td = torch.cat([mid_up, s2_proj, s1_spd], dim=1) # (B, 3*hidden_dim, H/2, W/2)
        feat_okm = self.okm_fusion(concat_low_td)           # (B, hidden_dim, H/2, W/2)  [P3 candidate]

        # --- Step 3: Bottom-Up Pathway ---
        # 3.1 Downsample from OKMFusion to stride 16
        feat_low_down = self.conv3x3_low_bu(feat_okm)       # (B, hidden_dim, H/4, W/4)

        # 3.2 Middle Bottom-Up ConCat & Fusion (stride 16)
        concat_mid_bu = torch.cat([feat_low_down, feat_mid_td], dim=1) # (B, 2*hidden_dim, H/4, W/4)
        feat_mid_bu = self.fusion_mid_bu(concat_mid_bu)     # (B, hidden_dim, H/4, W/4)  [P4 candidate]

        # 3.3 Downsample from Middle Fusion to stride 32
        feat_mid_down = self.conv3x3_top_bu(feat_mid_bu)    # (B, hidden_dim, H/8, W/8)

        # 3.4 Top Bottom-Up ConCat & Fusion (stride 32)
        concat_top_bu = torch.cat([feat_mid_down, feat_top], dim=1)    # (B, 2*hidden_dim, H/8, W/8)
        feat_top_bu = self.fusion_top_bu(concat_top_bu)     # (B, hidden_dim, H/8, W/8)  [P5 candidate]

        # --- Step 4: Multi-Scale Outputs for Circle (C) ---
        p3 = feat_okm     # stride 8  (rich in fine details and small targets)
        p4 = feat_mid_bu  # stride 16 (medium objects & contextual semantics)
        p5 = feat_top_bu  # stride 32 (large objects & high-level context)

        if not return_flattened_tokens:
            return p3, p4, p5

        # Optional: Flatten and concatenate into token sequence as shown after (C)
        # Each feature map of shape (B, C, H_i, W_i) is flattened to (B, H_i*W_i, C)
        b, c, _, _ = p3.shape
        tok_p3 = p3.flatten(2).permute(0, 2, 1)  # (B, H3*W3, C)
        tok_p4 = p4.flatten(2).permute(0, 2, 1)  # (B, H4*W4, C)
        tok_p5 = p5.flatten(2).permute(0, 2, 1)  # (B, H5*W5, C)
        concat_tokens = torch.cat([tok_p3, tok_p4, tok_p5], dim=1)  # (B, N_total, C)

        return (p3, p4, p5), concat_tokens

