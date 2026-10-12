import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple
from scipy.optimize import linear_sum_assignment


def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    """Chuyển đổi hộp từ (cx, cy, w, h) sang (x1, y1, x2, y2)."""
    cx, cy, w, h = x.unbind(-1)
    b = [(cx - 0.5 * w), (cy - 0.5 * h),
         (cx + 0.5 * w), (cy + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x: torch.Tensor) -> torch.Tensor:
    """Chuyển đổi hộp từ (x1, y1, x2, y2) sang (cx, cy, w, h)."""
    x1, y1, x2, y2 = x.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = (x2 - x1)
    h = (y2 - y1)
    return torch.stack([cx, cy, w, h], dim=-1)


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    """Tính diện tích các bounding boxes dạng (x1, y1, x2, y2)."""
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Tính IoU và Union giữa 2 bộ boxes dạng (x1, y1, x2, y2).
    boxes1: (N, 4), boxes2: (M, 4)
    Trả về: iou (N, M), union (N, M)
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # (N, M, 2)

    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    inter = wh[:, :, 0] * wh[:, :, 1]  # (N, M)

    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-7)
    return iou, union


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Generalized IoU (GIoU) giữa 2 bộ hộp (x1, y1, x2, y2).
    Trả về ma trận GIoU shape (N, M).
    """
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    enclosing_area = wh[:, :, 0] * wh[:, :, 1]

    giou = iou - (enclosing_area - union) / (enclosing_area + 1e-7)
    return giou


class HungarianMatcher(nn.Module):
    """
    Bộ ghép cặp lưỡng phân (Bipartite Matching / Hungarian Algorithm)
    ghép cặp tối ưu giữa queries dự đoán và ground truth boxes.
    """
    def __init__(self, cost_class: float = 2.0, cost_bbox: float = 5.0, cost_giou: float = 2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(
        self,
        pred_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        targets: List[Dict[str, torch.Tensor]]
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            pred_logits: Tensor shape (B, num_queries, num_classes)
            pred_boxes: Tensor shape (B, num_queries, 4) in (cx, cy, w, h)
            targets: List gồm B dicts, mỗi dict gồm:
                     - 'labels': Tensor (N_gt,)
                     - 'boxes': Tensor (N_gt, 4) in (cx, cy, w, h)

        Returns:
            List gồm B tuples (index_i, index_j), trong đó:
            - index_i: chỉ số của query được ghép cặp
            - index_j: chỉ số của target ground truth tương ứng
        """
        bs, num_queries = pred_logits.shape[:2]
        indices = []

        for b in range(bs):
            out_prob = pred_logits[b].sigmoid()  # (num_queries, num_classes)
            out_bbox = pred_boxes[b]             # (num_queries, 4)

            tgt_labels = targets[b]["labels"]     # (M,)
            tgt_boxes = targets[b]["boxes"]       # (M, 4)

            if len(tgt_labels) == 0:
                indices.append((
                    torch.as_tensor([], dtype=torch.int64, device=pred_logits.device),
                    torch.as_tensor([], dtype=torch.int64, device=pred_logits.device)
                ))
                continue

            # 1. Cost phân loại
            cost_class = -out_prob[:, tgt_labels]  # (num_queries, M)

            # 2. Cost L1 cho bounding box
            cost_bbox = torch.cdist(out_bbox, tgt_boxes, p=1)  # (num_queries, M)

            # 3. Cost GIoU
            cost_giou = -generalized_box_iou(
                box_cxcywh_to_xyxy(out_bbox),
                box_cxcywh_to_xyxy(tgt_boxes)
            )  # (num_queries, M)

            # Tổng chi phí ghép cặp
            cost_matrix = self.cost_class * cost_class + self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
            cost_matrix = cost_matrix.cpu().numpy()

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            indices.append((
                torch.as_tensor(row_ind, dtype=torch.int64, device=pred_logits.device),
                torch.as_tensor(col_ind, dtype=torch.int64, device=pred_logits.device)
            ))

        return indices


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "sum"
) -> torch.Tensor:
    """
    Focal Loss dạng Sigmoid cho bài toán phân loại đa nhãn / object detection.
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
    loss = ce_loss * ((1.0 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


class SetCriterion(nn.Module):
    """
    Hàm mất mát toàn diện (Loss Criterion) cho UAVTransformerDetector.
    Gồm 3 thành phần:
      1. Focal Classification Loss
      2. L1 Bounding Box Loss
      3. GIoU Regression Loss
    """
    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher,
        weight_cls: float = 2.0,
        weight_bbox: float = 5.0,
        weight_giou: float = 2.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_cls = weight_cls
        self.weight_bbox = weight_bbox
        self.weight_giou = weight_giou
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            outputs: Dict chứa:
              - 'pred_logits': (B, num_queries, num_classes)
              - 'pred_boxes': (B, num_queries, 4) in (cx, cy, w, h)
            targets: List B dicts chứa 'labels' và 'boxes'

        Returns:
            Dict chứa các thành phần loss và total_loss
        """
        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]
        device = pred_logits.device
        bs, num_queries = pred_logits.shape[:2]

        # 1. Ghép cặp lưỡng phân
        indices = self.matcher(pred_logits, pred_boxes, targets)

        # Đếm tổng số ground-truth boxes trong toàn batch để chuẩn hóa loss
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes_tensor = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        num_boxes_tensor = torch.clamp(num_boxes_tensor, min=1.0).item()

        # 2. Classification Loss (Sigmoid Focal Loss)
        target_classes = torch.zeros(bs, num_queries, self.num_classes, dtype=torch.float32, device=device)
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) > 0:
                tgt_labels = targets[b]["labels"][tgt_idx]
                target_classes[b, src_idx, tgt_labels] = 1.0

        loss_cls = sigmoid_focal_loss(
            pred_logits,
            target_classes,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            reduction="sum"
        ) / num_boxes_tensor

        # 3. Bounding Box Losses (L1 + GIoU)
        matched_pred_boxes = []
        matched_tgt_boxes = []
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) > 0:
                matched_pred_boxes.append(pred_boxes[b, src_idx])
                matched_tgt_boxes.append(targets[b]["boxes"][tgt_idx])

        if len(matched_pred_boxes) > 0:
            src_boxes = torch.cat(matched_pred_boxes, dim=0)
            tgt_boxes = torch.cat(matched_tgt_boxes, dim=0)

            # L1 Loss
            loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction="sum") / num_boxes_tensor

            # GIoU Loss
            giou_matrix = generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(tgt_boxes)
            )
            # Chỉ lấy các phần tử trên đường chéo (cặp đã match)
            loss_giou = (1.0 - torch.diag(giou_matrix)).sum() / num_boxes_tensor
        else:
            loss_bbox = torch.as_tensor(0.0, device=device)
            loss_giou = torch.as_tensor(0.0, device=device)

        # Tổng hợp loss có trọng số
        total_loss = (
            self.weight_cls * loss_cls +
            self.weight_bbox * loss_bbox +
            self.weight_giou * loss_giou
        )

        return {
            "loss_total": total_loss,
            "loss_cls": loss_cls,
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou
        }

