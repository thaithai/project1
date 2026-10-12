import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict


class UncertaintyMinimalQuerySelection(nn.Module):
    """
    Uncertainty-minimal Query Selection (from RT-DETR).
    Selects Top-K queries from the multi-scale fused tokens based on joint
    classification confidence and localization uncertainty minimization.
    """
    def __init__(self, hidden_dim: int, num_classes: int = 80, num_queries: int = 300):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_queries = num_queries

        # Linear projection to compute initial classification scores and bounding box coordinates
        self.enc_cls = nn.Linear(hidden_dim, num_classes)
        self.enc_reg = nn.Linear(hidden_dim, 4)

    def forward(
        self,
        concat_tokens: torch.Tensor,
        spatial_shapes: List[Tuple[int, int]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            concat_tokens: (B, N_total, hidden_dim) multi-scale flattened tokens from circle (C).
            spatial_shapes: List of (H_i, W_i) for each feature level.

        Returns:
            selected_queries: (B, num_queries, hidden_dim) initial content queries.
            selected_boxes: (B, num_queries, 4) initial reference boxes (cx, cy, w, h).
            topk_indices: (B, num_queries) indices of the selected tokens.
        """
        B, N, _ = concat_tokens.shape
        cls_logits = self.enc_cls(concat_tokens)  # (B, N, num_classes)
        box_offsets = self.enc_reg(concat_tokens).sigmoid()  # (B, N, 4)

        # Joint confidence score: maximum class probability
        # Minimal uncertainty implies highest confidence in candidate target
        scores = cls_logits.sigmoid().max(dim=-1).values  # (B, N)

        # Select Top-K queries with highest confidence / minimal uncertainty
        k = min(self.num_queries, N)
        topk_scores, topk_indices = torch.topk(scores, k, dim=-1)

        # Gather selected queries and boxes
        batch_indices = torch.arange(B, device=concat_tokens.device).unsqueeze(1).expand(-1, k)
        selected_queries = concat_tokens[batch_indices, topk_indices]
        selected_boxes = box_offsets[batch_indices, topk_indices]

        return selected_queries, selected_boxes, topk_indices


class DecoderHead(nn.Module):
    """
    Transformer Decoder & Detection Head.
    Refines selected queries via self-attention and cross-attention over multi-scale features,
    then predicts final bounding boxes and class logits.
    """
    def __init__(
        self,
        hidden_dim: int = 256,
        num_classes: int = 80,
        num_layers: int = 3,
        num_heads: int = 8,
        dim_feedforward: int = 1024
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            activation="relu",
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Detection Heads
        self.cls_head = nn.Linear(hidden_dim, num_classes)
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)
        )

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        init_boxes: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            queries: (B, num_queries, hidden_dim) selected queries.
            memory: (B, N_total, hidden_dim) concatenated multi-scale memory tokens.
            init_boxes: (B, num_queries, 4) initial reference boxes.

        Returns:
            Dict containing 'pred_logits' and 'pred_boxes'.
        """
        # Decoder forward pass
        out = self.decoder(tgt=queries, memory=memory)  # (B, num_queries, hidden_dim)

        # Prediction heads
        pred_logits = self.cls_head(out)
        box_deltas = self.reg_head(out)
        pred_boxes = (init_boxes + box_deltas).sigmoid()

        return {
            "pred_logits": pred_logits,
            "pred_boxes": pred_boxes
        }

