import os
import glob
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
from typing import List, Dict, Tuple, Optional


class UAVDetectionDataset(Dataset):
    """
    Dataset hỗ trợ định dạng YOLO (images/ và labels/ với file txt chứa 'class_id cx cy w h').
    Tương thích hoàn hảo với tập dữ liệu như DUT_Anti_UAV_YOLO hoặc VisDrone YOLO.
    """
    def __init__(
        self,
        img_dir: str,
        label_dir: str,
        img_size: int = 640,
        is_train: bool = True,
        augment: bool = True
    ):
        super().__init__()
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.is_train = is_train
        self.augment = augment and is_train

        # Tìm các file ảnh (jpg, png, jpeg)
        valid_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG")
        self.img_files = []
        for ext in valid_exts:
            self.img_files.extend(glob.glob(os.path.join(img_dir, ext)))
        self.img_files = sorted(self.img_files)

        if len(self.img_files) == 0:
            print(f"[Cảnh báo] Không tìm thấy ảnh nào trong thư mục: {img_dir}")

    def __len__(self) -> int:
        return len(self.img_files)

    def _load_labels(self, img_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Đọc file nhãn txt tương ứng với ảnh."""
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(self.label_dir, f"{base_name}.txt")

        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                        # Giới hạn tọa độ trong [0, 1]
                        cx = max(0.0, min(1.0, cx))
                        cy = max(0.0, min(1.0, cy))
                        w = max(0.0, min(1.0, w))
                        h = max(0.0, min(1.0, h))

                        if w > 0 and h > 0:
                            labels.append(cls_id)
                            boxes.append([cx, cy, w, h])

        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        return labels, boxes

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_path = self.img_files[idx]
        image = Image.open(img_path).convert("RGB")
        labels, boxes = self._load_labels(img_path)

        # 1. Resize ảnh
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)

        # 2. Data Augmentation (nếu ở chế độ training)
        if self.augment:
            # Lật ngang ngẫu nhiên (Horizontal Flip)
            if random.random() > 0.5:
                image = TF.hflip(image)
                if len(boxes) > 0:
                    boxes[:, 0] = 1.0 - boxes[:, 0]  # cx = 1.0 - cx

            # Biến đổi màu nhẹ (Color Jitter)
            if random.random() > 0.5:
                brightness = random.uniform(0.8, 1.2)
                contrast = random.uniform(0.8, 1.2)
                image = TF.adjust_brightness(image, brightness)
                image = TF.adjust_contrast(image, contrast)

        # 3. Chuẩn hóa Tensor
        img_tensor = TF.to_tensor(image)  # [0.0, 1.0], shape (3, H, W)
        img_tensor = TF.normalize(
            img_tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        target = {
            "labels": labels,
            "boxes": boxes,
            "image_id": torch.tensor([idx])
        }

        return img_tensor, target


def collate_fn(batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]):
    """
    Collate function tùy chỉnh gom batch các ảnh và danh sách targets.
    """
    images = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    return images, targets

