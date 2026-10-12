import torch
import torch.nn as nn
from typing import Dict, List, Tuple

from .starnet import StarNet
from .msda import MultiScaleDilatedAttention
from .neck import OKMFeatureFusionNeck
from .query_decoder import UncertaintyMinimalQuerySelection, DecoderHead


class UAVTransformerDetector(nn.Module):
    """
    Complete End-to-End UAV / Drone Transformer Detector as illustrated in the architecture:
      1. StarNet Backbone: Extracts multi-scale features [S1, S2, S3, S4]
      2. MSDA: Multi-Scale Dilated Attention applied to high-level stage S4
      3. OKMFeatureFusionNeck:
           - Top-down & bottom-up CCFM pathways
           - SPDConv for low-level high-resolution feature retention
           - OKMFusion (Omni-Kernel Module Fusion) for comprehensive multi-scale feature integration
      4. Output Circle (C): Multi-scale token concatenation
      5. Uncertainty-minimal Query Selection: Selects Top-K candidate queries
      6. Decoder & Head: Transformer decoder and detection prediction heads
    """
    def __init__(
        self,
        num_classes: int = 10,
        num_queries: int = 300,
        hidden_dim: int = 256,
        backbone_dims: List[int] = [32, 64, 128, 256],
        backbone_depths: List[int] = [2, 2, 4, 2],
        num_decoder_layers: int = 3
    ):
        super().__init__()
        # 1. StarNet Backbone
        self.backbone = StarNet(
            in_channels=3,
            dims=backbone_dims,
            depths=backbone_depths
        )

        # 2. MSDA on S4
        self.msda = MultiScaleDilatedAttention(
            dim=backbone_dims[3],
            num_heads=8,
            dilation_rates=[1, 2, 3]
        )

        # 3. OKM Feature Fusion Neck
        self.neck = OKMFeatureFusionNeck(
            in_channels=backbone_dims,
            hidden_dim=hidden_dim,
            okm_kernel_size=7,
            num_fusion_blocks=3
        )

        # 4. Uncertainty-minimal Query Selection
        self.query_selection = UncertaintyMinimalQuerySelection(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_queries=num_queries
        )

        # 5. Decoder & Prediction Head
        self.decoder_head = DecoderHead(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_layers=num_decoder_layers
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Input RGB image tensor of shape (B, 3, H, W)
        Returns:
            Dict containing:
              - 'pred_logits': (B, num_queries, num_classes)
              - 'pred_boxes': (B, num_queries, 4) in normalized (cx, cy, w, h)
              - 'multi_scale_features': tuple of (P3, P4, P5) feature maps
        """
        # Step 1: Backbone extraction
        s1, s2, s3, s4 = self.backbone(x)

        # Step 2: MSDA on S4
        s4_msda = self.msda(s4)

        # Step 3: Neck Feature Fusion & token concatenation at Circle (C)
        (p3, p4, p5), concat_tokens = self.neck(
            s1=s1,
            s2=s2,
            s3=s3,
            s4_msda=s4_msda,
            return_flattened_tokens=True
        )

        # Step 4: Uncertainty-minimal Query Selection
        spatial_shapes = [(p3.shape[2], p3.shape[3]), (p4.shape[2], p4.shape[3]), (p5.shape[2], p5.shape[3])]
        queries, init_boxes, topk_indices = self.query_selection(concat_tokens, spatial_shapes)

        # Step 5: Decoder & Head
        out = self.decoder_head(
            queries=queries,
            memory=concat_tokens,
            init_boxes=init_boxes
        )
        out["multi_scale_features"] = (p3, p4, p5)
        return out

