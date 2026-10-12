# UAV Transformer Detector & OKM Feature Fusion Neck

Mô hình học sâu tối ưu hóa cho phát hiện vật thể nhỏ từ thiết bị bay không người lái (UAV / Drone Detection), được xây dựng bằng **PyTorch** chuẩn theo sơ đồ kiến trúc.

---

## 1. Cấu Trúc Tổng Thể và Chi Tiết Từng Khối

### Sơ đồ luồng dữ liệu (Dataflow)

```
[Ảnh UAV đầu vào]
       │
       ▼
[StarNet Backbone] ──────► S1 (stride 4)  ──► [Conv1x1] ──► [SPDConv] ────────┐
       │             ───► S2 (stride 8)  ──► [Conv1x1] ────────────────────┤
       │             ───► S3 (stride 16) ──► [Conv1x1] ─────────────┐      │
       │             ───► S4 (stride 32) ──► [MSDA] ─────────┐      │      │
       ▼                                                     │      │      │
┌────────────────────────────────────────────────────────────┼──────┼──────┼────────────────────────┐
│ NECK (OKM Feature Fusion Neck)                             │      │      │                        │
│                                                            ▼      ▼      ▼                        │
│ [Top-Down]                                             UpSample  ConCat UpSample                  │
│                                                            │      │      │                        │
│                                                            └─► ConCat    │                        │
│                                                                   │      │                        │
│                                                             Middle Fusion│                        │
│                                                                   │      │                        │
│                                                                   └──► ConCat (3-way)             │
│                                                                           │                       │
│                                                                      [OKMFusion] (stride 8)       │
│                                                                           │                       │
│ [Bottom-Up]                                                               ├──► Conv3x3 (stride 2) │
│                                                                           │          │            │
│                                                                           │        ConCat         │
│                                                                           │          │            │
│                                                                           │    Mid BU Fusion (s16)│
│                                                                           │          │            │
│                                                                           │          ├──► Conv3x3 │
│                                                                           │          │       │    │
│                                                                           │          │     ConCat │
│                                                                           │          │       │    │
│                                                                           │          │  Top Fusion│
│                                                                           │          │       │    │
│                                                                           ▼          ▼       ▼    │
│                                                                           └──────── (C) ──────────┘
│                                                                                      │
└──────────────────────────────────────────────────────────────────────────────────────┼────────────┘
                                                                                       │
                                                                                       ▼
                                                                  [Uncertainty-minimal Query Selection]
                                                                                       │
                                                                                       ▼
                                                                               [Decoder & Head]
                                                                                       │
                                                                                       ▼
                                                                          [BBoxes & Classification]
```

---

## 2. Các Thành Phần Trong Phần Cắt (Cropped Neck)

1. **`UpSample`**: Nội suy không gian 2x (từ stride 32 lên stride 16; từ stride 16 lên stride 8).
2. **`ConCat`**: Ghép kênh đặc trưng (Channel Concatenation) giữa các nhánh đa tỷ lệ.
3. **`Fusion` (RepC3 / CSP)**: Khối hòa trộn đặc trưng sử dụng các lớp RepConv có khả năng re-parameterization, giúp tối ưu tốc độ suy luận.
4. **`SPDConv` (Space-to-Depth Convolution)**: Biến đổi không gian thành kênh chiều sâu cho tầng $S1$, giảm độ phân giải xuống stride 8 mà **không làm mất thông tin điểm ảnh nhỏ**, cực kỳ hữu ích cho mục tiêu UAV/drone siêu nhỏ.
5. **`OKMFusion` (Omni-Kernel Module Fusion)**:
   - **Local Branch**: Depthwise Conv 1x1 nắm bắt đặc trưng cục bộ mịn.
   - **Large-Kernel Branch**: Kết hợp depthwise conv $1 \times K$, $K \times 1$ và $K \times K$ ($K=7$) tạo trường tiếp nhận đa mức cực rộng.
   - **Global Frequency Branch (FGM/SCA)**: Cơ chế biến đổi Fourier 2D (FFT) và cơ chế chú ý kênh không gian điều chế ngữ cảnh toàn cục.
6. **`Conv3x3` (stride 2)**: Tích chập giảm tỷ lệ không gian ở nhánh Bottom-Up (từ stride 8 lên 16, từ 16 lên 32).
7. **Vòng tròn `(C)` (ConCat)**: Gom 3 mức đặc trưng đa tỷ lệ [$P_3$, $P_4$, $P_5$] thành chuỗi token sequence đưa vào bộ chọn truy vấn tối thiểu hóa độ bất định (*Uncertainty-minimal Query Selection*).

---

## 3. Cách Sử Dụng Trong Code Python

### A. Chỉ sử dụng phần Neck độc lập

```python
import torch
from models.neck import OKMFeatureFusionNeck

# Khởi tạo Neck
neck = OKMFeatureFusionNeck(
    in_channels=[32, 64, 128, 256],  # Số kênh đầu vào của S1, S2, S3, S4/MSDA
    hidden_dim=256,                  # Số kênh ẩn hợp nhất
    okm_kernel_size=7,
    num_fusion_blocks=3
)

# Giả lập feature maps từ backbone (kích thước ảnh gốc 640x640)
s1 = torch.randn(2, 32, 160, 160)   # stride 4
s2 = torch.randn(2, 64, 80, 80)     # stride 8
s3 = torch.randn(2, 128, 40, 40)    # stride 16
s4_msda = torch.randn(2, 256, 20, 20) # stride 32

# Forward pass
# Trả về 3 feature maps đa mức (P3, P4, P5) và chuỗi tokens tại vòng tròn (C)
(p3, p4, p5), concat_tokens = neck(s1, s2, s3, s4_msda, return_flattened_tokens=True)

print("P3 shape (stride 8) :", p3.shape)  # torch.Size([2, 256, 80, 80])
print("P4 shape (stride 16):", p4.shape)  # torch.Size([2, 256, 40, 40])
print("P5 shape (stride 32):", p5.shape)  # torch.Size([2, 256, 20, 20])
print("Token sequence (C) :", concat_tokens.shape)  # torch.Size([2, 8400, 256])
```

### B. Sử dụng toàn bộ mô hình End-to-End Detector

```python
import torch
from models.full_model import UAVTransformerDetector

model = UAVTransformerDetector(
    num_classes=10,   # Số nhãn phát hiện
    num_queries=300,  # Số lượng queries
    hidden_dim=256
)

dummy_image = torch.randn(2, 3, 640, 640)
outputs = model(dummy_image)

print("Class Logits :", outputs["pred_logits"].shape)  # [2, 300, 10]
print("Bounding Box :", outputs["pred_boxes"].shape)   # [2, 300, 4] (cx, cy, w, h)
```

### C. Vẽ Bounding Box kèm Confidence Score và lưu ảnh vào Folder

Sử dụng hàm [`draw_bounding_boxes`](file:///c:/Users/PC/Downloads/transformer-uav/models/visualize.py):

```python
from models.visualize import draw_bounding_boxes
from PIL import Image

# 1. Đọc ảnh gốc
pil_img = Image.open('C:/Users/PC/Downloads/val/val/img/02504.jpg')

# 2. Dự đoán với model
outputs = model(input_tensor)

# 3. Vẽ Bounding Box và lưu vào thư mục
annotated_img, saved_path, detections = draw_bounding_boxes(
    image=pil_img,
    pred_logits=outputs["pred_logits"],
    pred_boxes=outputs["pred_boxes"],
    output_dir="runs/detect",           # Thư mục lưu ảnh
    output_filename="detected_02504.jpg", # Tên file kết quả
    conf_threshold=0.3,                 # Ngưỡng tự tin (confidence threshold)
    top_k=5,                            # Giới hạn số lượng boxes hiển thị
    class_names=["UAV", "Drone", "Bird", ...], # Tên các lớp
    save_image=True
)

print(f"Ảnh đã lưu tại: {saved_path}")
```


---

## 4. Kiểm Thử

Chạy lệnh kiểm thử tự động:
```bash
py test_neck.py
```

