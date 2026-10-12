import sys
import torch
import numpy as np
from PIL import Image

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from models import OKMFeatureFusionNeck, UAVTransformerDetector



def test_neck_module():
    print("=" * 70)
    print("1. KIỂM TRA PHẦN NECK (OKM FEATURE FUSION NECK)")
    print("=" * 70)

    # Giả lập 4 mức feature maps từ backbone (tương ứng S1, S2, S3, S4):
    # Kích thước ảnh gốc giả định: 640x640, batch size = 2
    # S1: stride 4  -> 160x160 (32 channels)
    # S2: stride 8  -> 80x80   (64 channels)
    # S3: stride 16 -> 40x40   (128 channels)
    # S4: stride 32 -> 20x20   (256 channels, sau MSDA)
    batch_size = 2
    s1 = torch.randn(batch_size, 32, 160, 160)
    s2 = torch.randn(batch_size, 64, 80, 80)
    s3 = torch.randn(batch_size, 128, 40, 40)
    s4_msda = torch.randn(batch_size, 256, 20, 20)

    print(f"Input S1 (stride 4)      : {list(s1.shape)}")
    print(f"Input S2 (stride 8)      : {list(s2.shape)}")
    print(f"Input S3 (stride 16)     : {list(s3.shape)}")
    print(f"Input S4/MSDA (stride 32): {list(s4_msda.shape)}")

    # Khởi tạo Neck
    neck = OKMFeatureFusionNeck(
        in_channels=[32, 64, 128, 256],
        hidden_dim=256,
        okm_kernel_size=7,
        num_fusion_blocks=3
    )

    # Forward pass trả về multi-scale feature maps và concatenated tokens tại circle (C)
    (p3, p4, p5), concat_tokens = neck(
        s1=s1,
        s2=s2,
        s3=s3,
        s4_msda=s4_msda,
        return_flattened_tokens=True
    )

    print("\n--- KẾT QUẢ ĐẦU RA CỦA NECK ---")
    print(f"Output P3 (từ OKMFusion, stride 8)        : {list(p3.shape)}")
    print(f"Output P4 (từ Middle Fusion bu, stride 16): {list(p4.shape)}")
    print(f"Output P5 (từ Top Fusion bu, stride 32)   : {list(p5.shape)}")
    print(f"Output Token Sequence tại vòng tròn (C)   : {list(concat_tokens.shape)}  (B, Total_Tokens, Channels)")

    total_params = sum(p.numel() for p in neck.parameters() if p.requires_grad)
    print(f"\nTổng tham số phần Neck: {total_params / 1e6:.2f} M params")
    print("Test phần Neck THÀNH CÔNG!\n")


def test_full_model():
    print("=" * 70)
    print("2. KIỂM TRA TOÀN BỘ MÔ HÌNH (STARNET + MSDA + NECK + HEAD)")
    print("=" * 70)

    import os
    import torchvision.transforms as T

    image_path = 'C:/Users/PC/Downloads/val/val/img/02504.jpg'
    if os.path.exists(image_path):
        # 1. Đọc ảnh thật từ ổ đĩa
        pil_img = Image.open(image_path).convert('RGB')
        print(f"Ảnh UAV gốc từ đĩa (Width, Height): {pil_img.size}")

        # 2. Tiền xử lý ảnh thành Tensor (1, 3, 640, 640)
        transform = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),  # chuẩn hóa về [0.0, 1.0] và chuyển format sang (C, H, W)
        ])
        input_tensor = transform(pil_img).unsqueeze(0)  # thêm batch_size = 1 -> (1, 3, 640, 640)
    else:
        # Dự phòng ảnh ngẫu nhiên nếu không tìm thấy file
        print(f"Không tìm thấy file {image_path}, tạo tensor giả lập...")
        input_tensor = torch.randn(1, 3, 640, 640)

    print(f"Kích thước Tensor đầu vào mô hình: {list(input_tensor.shape)} (Batch, Channel, Height, Width)")

    model = UAVTransformerDetector(
        num_classes=10,       # Ví dụ: 10 lớp vật thể (drone, UAV, chim, máy bay...)
        num_queries=300,      # 300 queries
        hidden_dim=256
    )

    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)


    pred_logits = outputs["pred_logits"]
    pred_boxes = outputs["pred_boxes"]
    p3, p4, p5 = outputs["multi_scale_features"]

    print("\n--- KẾT QUẢ DỰ ĐOÁN CỦA MÔ HÌNH ---")
    print(f"Prediction Logits: {list(pred_logits.shape)}  (Batch, Num_Queries, Num_Classes)")
    print(f"Prediction Boxes : {list(pred_boxes.shape)}   (Batch, Num_Queries, 4 [cx, cy, w, h])")
    print(f"Feature Maps     : P3={list(p3.shape)}, P4={list(p4.shape)}, P5={list(p5.shape)}")

    # 3. Vẽ Bounding Box kèm Confidence Score và lưu vào Folder
    output_folder = "runs/detect"
    output_name = "detected_02504.jpg"
    class_labels = ["UAV", "Drone", "Airplane", "Helicopter", "Bird", "Parachute", "Balloon", "Object", "Target", "Obstacle"]

    from models.visualize import draw_bounding_boxes

    # Lấy ảnh gốc để vẽ bboxes (nếu có đọc từ file, nếu không tạo ảnh trắng để vẽ)
    orig_img_source = pil_img if 'pil_img' in locals() else Image.new('RGB', (640, 640), color=(100, 100, 100))

    annotated_img, saved_path, detections = draw_bounding_boxes(
        image=orig_img_source,
        pred_logits=pred_logits,
        pred_boxes=pred_boxes,
        output_dir=output_folder,
        output_filename=output_name,
        conf_threshold=0.3,
        top_k=5,  # Hiển thị top-5 bounding boxes có confidence cao nhất
        class_names=class_labels,
        save_image=True
    )

    print("\n--- THÔNG TIN LƯU ẢNH BOUNDING BOX ---")
    print(f"Thư mục lưu ảnh       : {os.path.abspath(output_folder)}")
    print(f"File ảnh kết quả      : {os.path.abspath(saved_path)}")
    print(f"Số bounding boxes vẽ  : {len(detections)}")
    for i, det in enumerate(detections, 1):
        x1, y1, x2, y2 = det['box']
        print(f"  Box #{i}: Nhãn={det['class_name']}, Conf={det['score']:.4f}, Tọa độ=[x1={x1}, y1={y1}, x2={x2}, y2={y2}]")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTổng tham số toàn mô hình: {total_params / 1e6:.2f} M params")
    print("Test toàn mô hình và LƯU ẢNH THÀNH CÔNG!")
    print("=" * 70)



if __name__ == "__main__":
    test_neck_module()
    test_full_model()
