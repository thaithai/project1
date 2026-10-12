import os
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np
import torch
from PIL import Image


# Bảng màu sắc trực quan (BGR) cho các nhãn khác nhau
COLOR_PALETTE = [
    (147, 20, 255),  # Hồng tím (giống màu drone tím trong sơ đồ)
    (50, 205, 50),   # Xanh lá chanh (giống màu drone xanh trong sơ đồ)
    (255, 140, 0),   # Cam đậm
    (0, 215, 255),   # Vàng gold
    (255, 0, 128),   # Hồng sen
    (0, 255, 255),   # Vàng sáng
    (238, 130, 238), # Tím nhạt
    (30, 144, 255),  # Xanh dương
    (0, 255, 127),   # Xanh lục bảo
    (255, 69, 0),    # Đỏ cam
]


def draw_bounding_boxes(
    image: Union[str, Image.Image, np.ndarray],
    pred_logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    output_dir: str = "output",
    output_filename: str = "detected_result.jpg",
    conf_threshold: float = 0.3,
    top_k: int = 5,
    class_names: Optional[List[str]] = None,
    save_image: bool = True
) -> Tuple[np.ndarray, str, List[dict]]:
    """
    Vẽ bounding box kèm nhãn và confidence score lên ảnh, sau đó lưu vào thư mục chỉ định.

    Args:
        image: Đường dẫn ảnh (str), đối tượng PIL.Image, hoặc mảng numpy RGB.
        pred_logits: Tensor kích thước (B, num_queries, num_classes) hoặc (num_queries, num_classes).
        pred_boxes: Tensor kích thước (B, num_queries, 4) chứa (cx, cy, w, h) chuẩn hóa [0, 1].
        output_dir: Thư mục lưu ảnh kết quả.
        output_filename: Tên file ảnh đầu ra.
        conf_threshold: Ngưỡng tự tin (confidence score) để lọc bboxes.
        top_k: Số lượng bounding box có điểm tự tin cao nhất cần hiển thị nếu dùng mô hình demo/chưa huấn luyện.
        class_names: Danh sách tên nhãn (ví dụ: ['drone', 'uav', 'bird', ...]).
        save_image: Có lưu ảnh ra file hay không.

    Returns:
        annotated_image: Ảnh mảng numpy (BGR) đã vẽ bounding box.
        saved_path: Đường dẫn đầy đủ file ảnh đã lưu.
        detections: Danh sách các vật thể được phát hiện [{'class': ..., 'score': ..., 'box': [x1, y1, x2, y2]}].
    """
    # 1. Đọc ảnh thành mảng BGR (OpenCV)
    if isinstance(image, str):
        if not os.path.exists(image):
            raise FileNotFoundError(f"Không tìm thấy file ảnh: {image}")
        img_bgr = cv2.imread(image)
        orig_h, orig_w = img_bgr.shape[:2]
    elif isinstance(image, Image.Image):
        img_rgb = np.array(image.convert("RGB"))
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        orig_w, orig_h = image.size
    elif isinstance(image, np.ndarray):
        if image.shape[2] == 3:
            img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = image.copy()
        orig_h, orig_w = img_bgr.shape[:2]
    else:
        raise ValueError("Định dạng ảnh không hợp lệ!")

    # 2. Xử lý tensor logits và boxes
    if pred_logits.dim() == 3:
        pred_logits = pred_logits[0]  # lấy batch đầu tiên (num_queries, num_classes)
    if pred_boxes.dim() == 3:
        pred_boxes = pred_boxes[0]    # lấy batch đầu tiên (num_queries, 4)

    # Chuyển logits sang xác suất (probabilities) qua Sigmoid
    probs = pred_logits.sigmoid()
    scores, class_ids = probs.max(dim=-1)

    scores = scores.detach().cpu().numpy()
    class_ids = class_ids.detach().cpu().numpy()
    boxes = pred_boxes.detach().cpu().numpy()

    # 3. Lọc theo ngưỡng tự tin
    valid_mask = scores >= conf_threshold
    valid_indices = np.where(valid_mask)[0]

    # Sắp xếp các box theo điểm confidence giảm dần
    if len(valid_indices) > 0:
        valid_indices = valid_indices[np.argsort(scores[valid_indices])[::-1]]
    else:
        # Nếu không có box nào vượt ngưỡng, lấy top_k boxes có điểm cao nhất
        valid_indices = np.argsort(scores)[::-1]

    # Giới hạn số lượng boxes tối đa hiển thị (top_k) để ảnh trực quan, rõ ràng
    if top_k is not None and top_k > 0:
        valid_indices = valid_indices[:top_k]


    detections = []
    annotated_img = img_bgr.copy()

    # 4. Vẽ từng bounding box lên ảnh
    for idx in valid_indices:
        score = float(scores[idx])
        cls_id = int(class_ids[idx])
        cx, cy, w, h = boxes[idx]

        # Đổi tọa độ chuẩn hóa (cx, cy, w, h) sang tọa độ pixel ảnh gốc (x1, y1, x2, y2)
        x1 = int((cx - w / 2.0) * orig_w)
        y1 = int((cy - h / 2.0) * orig_h)
        x2 = int((cx + w / 2.0) * orig_w)
        y2 = int((cy + h / 2.0) * orig_h)

        # Giới hạn tọa độ trong phạm vi ảnh
        x1 = max(0, min(x1, orig_w - 1))
        y1 = max(0, min(y1, orig_h - 1))
        x2 = max(0, min(x2, orig_w - 1))
        y2 = max(0, min(y2, orig_h - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        # Chọn màu sắc cho lớp
        color = COLOR_PALETTE[cls_id % len(COLOR_PALETTE)]

        # Tên nhãn kèm confidence score
        if class_names and cls_id < len(class_names):
            class_label = class_names[cls_id]
        else:
            class_label = f"Class {cls_id}"
        label_text = f"{class_label}: {score:.2f}"

        # Vẽ hình chữ nhật Bounding Box
        line_thickness = max(2, int(min(orig_w, orig_h) * 0.003))
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, line_thickness)

        # Vẽ nền nhãn để chữ rõ ràng, dễ nhìn
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.5, min(orig_w, orig_h) * 0.0007)
        font_thickness = max(1, int(line_thickness * 0.7))
        (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)

        # Tọa độ nền nhãn
        text_bg_y1 = max(0, y1 - text_h - 6)
        text_bg_y2 = y1
        text_bg_x2 = min(orig_w, x1 + text_w + 8)

        # Vẽ nền nhãn
        cv2.rectangle(annotated_img, (x1, text_bg_y1), (text_bg_x2, text_bg_y2), color, -1)
        # Vẽ chữ nhãn màu trắng
        cv2.putText(
            annotated_img,
            label_text,
            (x1 + 4, text_bg_y2 - 4),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
            lineType=cv2.LINE_AA
        )

        detections.append({
            "class_id": cls_id,
            "class_name": class_label,
            "score": score,
            "box": [x1, y1, x2, y2]
        })

    # 5. Lưu ảnh vào folder
    saved_path = ""
    if save_image:
        os.makedirs(output_dir, exist_ok=True)
        saved_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(saved_path, annotated_img)

    return annotated_img, saved_path, detections
