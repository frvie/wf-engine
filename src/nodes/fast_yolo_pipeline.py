"""
Ultra-Fast YOLO Pipeline - No Dictionary Overhead

Pure Python functions with direct variable passing.
Eliminates ALL serialization overhead for maximum performance.
"""

import cv2
import numpy as np
from typing import Tuple


def resize_letterbox(
    image: np.ndarray,
    target_width: int = 640,
    target_height: int = 640,
    padding_color: Tuple[int, int, int] = (114, 114, 114)
) -> Tuple[np.ndarray, float, int, int]:
    """Resize with letterbox padding, return (image, scale, pad_x, pad_y)"""
    h, w = image.shape[:2]
    scale = min(target_width / w, target_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    canvas = np.full((target_height, target_width, 3), padding_color, dtype=np.uint8)
    pad_x = (target_width - new_w) // 2
    pad_y = (target_height - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    
    return canvas, scale, pad_x, pad_y


def normalize(image: np.ndarray, scale: float = 255.0) -> np.ndarray:
    """Normalize image to [0, 1] range"""
    return image.astype(np.float32) / scale


def hwc_to_chw(image: np.ndarray) -> np.ndarray:
    """Convert HWC to CHW format"""
    return np.transpose(image, (2, 0, 1))


def add_batch_dim(image: np.ndarray) -> np.ndarray:
    """Add batch dimension"""
    return np.expand_dims(image, axis=0)


def run_inference(session, input_name: str, image: np.ndarray) -> np.ndarray:
    """Run ONNX inference, return raw outputs"""
    outputs = session.run(None, {input_name: image})
    return outputs[0]


def decode_yolo_v8(outputs: np.ndarray, num_classes: int = 80) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode YOLOv8 output, return (boxes, scores, class_ids)"""
    predictions = outputs[0]
    predictions = predictions.transpose(1, 0)
    
    boxes = predictions[:, :4]
    class_scores = predictions[:, 4:]
    
    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(len(class_ids)), class_ids]
    
    return boxes, scores, class_ids


def filter_confidence(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    conf_threshold: float = 0.25
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filter by confidence threshold"""
    mask = scores >= conf_threshold
    return boxes[mask], scores[mask], class_ids[mask]


def cxcywh_to_xyxy(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert center format to corner format"""
    if len(boxes) == 0:
        return boxes, scores, class_ids
    
    x_center = boxes[:, 0]
    y_center = boxes[:, 1]
    width = boxes[:, 2]
    height = boxes[:, 3]
    
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
    return boxes_xyxy, scores, class_ids


def apply_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float = 0.7
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply Non-Maximum Suppression"""
    if len(boxes) == 0:
        return boxes, scores, class_ids
    
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes.tolist(),
        scores=scores.tolist(),
        score_threshold=0.0,
        nms_threshold=iou_threshold
    )
    
    if len(indices) == 0:
        return np.array([]), np.array([]), np.array([])
    
    indices = indices.flatten()
    return boxes[indices], scores[indices], class_ids[indices]


def scale_boxes(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    input_width: int,
    input_height: int,
    original_width: int,
    original_height: int,
    pad_x: int,
    pad_y: int,
    scale: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scale boxes back to original image dimensions"""
    if len(boxes) == 0:
        return boxes, scores, class_ids
    
    boxes_copy = boxes.copy()
    boxes_copy[:, [0, 2]] = (boxes_copy[:, [0, 2]] - pad_x) / scale
    boxes_copy[:, [1, 3]] = (boxes_copy[:, [1, 3]] - pad_y) / scale
    
    boxes_copy[:, [0, 2]] = np.clip(boxes_copy[:, [0, 2]], 0, original_width)
    boxes_copy[:, [1, 3]] = np.clip(boxes_copy[:, [1, 3]], 0, original_height)
    
    return boxes_copy, scores, class_ids


def format_detections(boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray) -> list:
    """Format as detections with COCO class names"""
    if len(boxes) == 0:
        return []
    
    # COCO class names
    COCO_CLASSES = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
        "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]
    
    detections = []
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box
        detections.append({
            "bbox": [float(x1), float(y1), float(x2), float(y2)],  # xyxy format for visualization
            "confidence": float(score),
            "category_id": int(class_id),
            "class_name": COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class_{class_id}"
        })
    
    return detections


def process_frame_yolo(
    frame: np.ndarray,
    session,
    input_name: str,
    original_width: int,
    original_height: int,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.7
) -> list:
    """
    Ultra-fast single-frame YOLO processing.
    
    All operations in one function with direct variable passing.
    Zero dictionary overhead, maximum performance.
    """
    # 1. Resize with letterbox
    image, scale, pad_x, pad_y = resize_letterbox(frame, 640, 640)
    
    # 2. Normalize
    image = normalize(image, 255.0)
    
    # 3. HWC to CHW
    image = hwc_to_chw(image)
    
    # 4. Add batch dimension
    image = add_batch_dim(image)
    
    # 5. Run inference
    outputs = run_inference(session, input_name, image)
    
    # 6. Decode YOLO output
    boxes, scores, class_ids = decode_yolo_v8(outputs, 80)
    
    # 7. Filter by confidence
    boxes, scores, class_ids = filter_confidence(boxes, scores, class_ids, conf_threshold)
    
    # 8. Convert box format
    boxes, scores, class_ids = cxcywh_to_xyxy(boxes, scores, class_ids)
    
    # 9. Apply NMS
    boxes, scores, class_ids = apply_nms(boxes, scores, class_ids, iou_threshold)
    
    # 10. Scale boxes to original size
    boxes, scores, class_ids = scale_boxes(
        boxes, scores, class_ids,
        640, 640,
        original_width, original_height,
        pad_x, pad_y, scale
    )
    
    # 11. Format detections
    detections = format_detections(boxes, scores, class_ids)
    
    return detections


