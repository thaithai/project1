import os
import sys
import time
import argparse
from typing import Dict, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from models import UAVTransformerDetector, HungarianMatcher, SetCriterion
from dataset import UAVDetectionDataset, collate_fn


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0.1,
    print_freq: int = 10
) -> Dict[str, float]:
    """
    Huấn luyện mô hình trong 1 epoch.
    """
    model.train()
    criterion.train()

    total_loss_accum = 0.0
    cls_loss_accum = 0.0
    bbox_loss_accum = 0.0
    giou_loss_accum = 0.0
    num_batches = len(data_loader)

    start_time = time.time()
    print(f"\n Epoch [{epoch + 1}] - Bắt đầu huấn luyện...")

    for step, (images, targets) in enumerate(data_loader):
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # 1. Forward pass
        outputs = model(images)

        # 2. Tính toán Loss (Hungarian Matcher + Focal + L1 + GIoU)
        loss_dict = criterion(outputs, targets)
        loss_total = loss_dict["loss_total"]

        # 3. Backward & Optimizer step
        optimizer.zero_grad()
        loss_total.backward()

        # Gradient clipping chống bùng nổ gradient trong Transformer
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()

        # Tích lũy loss
        total_loss_accum += loss_total.item()
        cls_loss_accum += loss_dict["loss_cls"].item()
        bbox_loss_accum += loss_dict["loss_bbox"].item()
        giou_loss_accum += loss_dict["loss_giou"].item()

        if (step + 1) % print_freq == 0 or (step + 1) == num_batches:
            elapsed = time.time() - start_time
            print(
                f"Epoch [{epoch + 1}] Step [{step + 1}/{num_batches}] "
                f"Total Loss: {loss_total.item():.4f} "
                f"(Cls: {loss_dict['loss_cls'].item():.4f}, "
                f"BBox: {loss_dict['loss_bbox'].item():.4f}, "
                f"GIoU: {loss_dict['loss_giou'].item():.4f}) - "
                f"Time: {elapsed:.1f}s"
            )

    avg_metrics = {
        "loss_total": total_loss_accum / max(num_batches, 1),
        "loss_cls": cls_loss_accum / max(num_batches, 1),
        "loss_bbox": bbox_loss_accum / max(num_batches, 1),
        "loss_giou": giou_loss_accum / max(num_batches, 1),
    }
    return avg_metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    criterion: nn.Module,
    data_loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """
    Đánh giá mô hình trên tập validation.
    """
    model.eval()
    criterion.eval()

    total_loss_accum = 0.0
    cls_loss_accum = 0.0
    bbox_loss_accum = 0.0
    giou_loss_accum = 0.0
    num_batches = len(data_loader)

    for images, targets in data_loader:
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(images)
        loss_dict = criterion(outputs, targets)

        total_loss_accum += loss_dict["loss_total"].item()
        cls_loss_accum += loss_dict["loss_cls"].item()
        bbox_loss_accum += loss_dict["loss_bbox"].item()
        giou_loss_accum += loss_dict["loss_giou"].item()

    avg_metrics = {
        "val_loss_total": total_loss_accum / max(num_batches, 1),
        "val_loss_cls": cls_loss_accum / max(num_batches, 1),
        "val_loss_bbox": bbox_loss_accum / max(num_batches, 1),
        "val_loss_giou": giou_loss_accum / max(num_batches, 1),
    }
    return avg_metrics


def train_model(
    train_img_dir: str,
    train_label_dir: str,
    val_img_dir: Optional[str] = None,
    val_label_dir: Optional[str] = None,
    num_classes: int = 1,
    num_queries: int = 300,
    epochs: int = 50,
    batch_size: int = 4,
    lr: float = 1e-4,
    lr_backbone: float = 1e-5,
    weight_decay: float = 1e-4,
    img_size: int = 640,
    num_workers: int = 0,
    output_dir: str = "runs/train",
    device: Optional[str] = None
):
    """
    Hàm tổng thể khởi tạo và thực hiện toàn bộ quá trình huấn luyện UAV Transformer Detector.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Thiết bị chạy
    if device is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = device
    device_obj = torch.device(device_str)
    print(f" Sử dụng thiết bị: {device_obj} (CUDA: {torch.cuda.is_available()})")

    # 2. Xây dựng Dataloader
    print(f" Đang tải tập huấn luyện từ: {train_img_dir}")
    train_dataset = UAVDetectionDataset(
        img_dir=train_img_dir,
        label_dir=train_label_dir,
        img_size=img_size,
        is_train=True,
        augment=True
    )
    print(f" Tổng số ảnh huấn luyện: {len(train_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = None
    if val_img_dir and os.path.exists(val_img_dir) and val_label_dir and os.path.exists(val_label_dir):
        print(f" Đang tải tập kiểm định từ: {val_img_dir}")
        val_dataset = UAVDetectionDataset(
            img_dir=val_img_dir,
            label_dir=val_label_dir,
            img_size=img_size,
            is_train=False,
            augment=False
        )
        print(f" Tổng số ảnh kiểm định: {len(val_dataset)}")
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available()
        )

    # 3. Khởi tạo mô hình & Loss
    print(f" Đang khởi tạo mô hình UAVTransformerDetector (num_classes={num_classes}, num_queries={num_queries})...")
    model = UAVTransformerDetector(
        num_classes=num_classes,
        num_queries=num_queries,
        hidden_dim=256
    ).to(device_obj)

    matcher = HungarianMatcher(cost_class=2.0, cost_bbox=5.0, cost_giou=2.0)
    criterion = SetCriterion(
        num_classes=num_classes,
        matcher=matcher,
        weight_cls=2.0,
        weight_bbox=5.0,
        weight_giou=2.0
    ).to(device_obj)

    # 4. Tách biệt Learning Rate cho Backbone và Neck/Head
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad], "lr": lr_backbone},
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad], "lr": lr},
    ]
    optimizer = AdamW(param_dicts, lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # 5. Vòng lặp huấn luyện các epoch
    best_val_loss = float("inf")
    print(f"\n Bắt đầu quá trình huấn luyện {epochs} epochs...")

    for epoch in range(epochs):
        train_metrics = train_one_epoch(
            model=model,
            criterion=criterion,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device_obj,
            epoch=epoch,
            max_norm=0.1
        )
        scheduler.step()

        # Đánh giá validation
        val_info = ""
        current_loss = train_metrics["loss_total"]
        if val_loader is not None and len(val_loader) > 0:
            val_metrics = evaluate(model, criterion, val_loader, device=device_obj)
            current_loss = val_metrics["val_loss_total"]
            val_info = f" | Val Loss: {current_loss:.4f} (Cls: {val_metrics['val_loss_cls']:.4f}, BBox: {val_metrics['val_loss_bbox']:.4f}, GIoU: {val_metrics['val_loss_giou']:.4f})"

        print(
            f" Epoch [{epoch + 1}/{epochs}] Hoàn thành - "
            f"Train Loss: {train_metrics['loss_total']:.4f}{val_info} - "
            f"LR: {optimizer.param_groups[1]['lr']:.6f}"
        )

        # Lưu checkpoint cuối cùng (last.pt)
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_metrics["loss_total"],
            "num_classes": num_classes,
            "num_queries": num_queries
        }
        last_ckpt_path = os.path.join(output_dir, "last.pt")
        torch.save(checkpoint, last_ckpt_path)

        # Lưu checkpoint tốt nhất (best.pt)
        if current_loss < best_val_loss:
            best_val_loss = current_loss
            best_ckpt_path = os.path.join(output_dir, "best.pt")
            torch.save(checkpoint, best_ckpt_path)
            print(f" Đã lưu mô hình tốt nhất (best.pt) với Loss: {best_val_loss:.4f}")

    print(f"\n Huấn luyện hoàn tất! Trọng số được lưu tại: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình UAVTransformerDetector")

    # Mặc định tự động kiểm tra thư mục DUT_Anti_UAV_YOLO nếu có sẵn trong Downloads
    default_dataset = r"C:\Users\PC\Downloads\DUT_Anti_UAV_YOLO"
    if os.path.exists(default_dataset):
        default_train_img = os.path.join(default_dataset, "images", "train")
        default_train_lbl = os.path.join(default_dataset, "labels", "train")
        default_val_img = os.path.join(default_dataset, "images", "val")
        default_val_lbl = os.path.join(default_dataset, "labels", "val")
    else:
        default_train_img = "data/images/train"
        default_train_lbl = "data/labels/train"
        default_val_img = "data/images/val"
        default_val_lbl = "data/labels/val"

    parser.add_argument("--train-img", type=str, default=default_train_img, help="Đường dẫn thư mục ảnh train")
    parser.add_argument("--train-label", type=str, default=default_train_lbl, help="Đường dẫn thư mục nhãn txt train")
    parser.add_argument("--val-img", type=str, default=default_val_img, help="Đường dẫn thư mục ảnh val")
    parser.add_argument("--val-label", type=str, default=default_val_lbl, help="Đường dẫn thư mục nhãn txt val")
    parser.add_argument("--num-classes", type=int, default=1, help="Số lớp vật thể (mặc định 1 cho UAV/Drone)")
    parser.add_argument("--num-queries", type=int, default=300, help="Số lượng query detector")
    parser.add_argument("--epochs", type=int, default=30, help="Số epoch huấn luyện")
    parser.add_argument("--batch-size", type=int, default=2, help="Kích thước batch")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate cho Neck & Head")
    parser.add_argument("--lr-backbone", type=float, default=1e-5, help="Learning rate cho Backbone")
    parser.add_argument("--img-size", type=int, default=640, help="Kích thước ảnh đầu vào")
    parser.add_argument("--output-dir", type=str, default="runs/train", help="Thư mục lưu weights")

    args = parser.parse_args()

    train_model(
        train_img_dir=args.train_img,
        train_label_dir=args.train_label,
        val_img_dir=args.val_img,
        val_label_dir=args.val_label,
        num_classes=args.num_classes,
        num_queries=args.num_queries,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_backbone=args.lr_backbone,
        img_size=args.img_size,
        output_dir=args.output_dir
    )

