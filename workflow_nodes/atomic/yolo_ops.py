"""
Atomic YOLO Postprocessing Nodes

Small, reusable nodes for YOLO model postprocessing.
Each node handles one step of the detection pipeline.
"""

import numpy as np
from typing import List, Tuple
from workflow_decorator import workflow_node


@workflow_node("decode_yolo_v8_output", isolation_mode="none", log_level="SILENT", skip_metadata=True)
def decode_yolo_v8_output_node(
    outputs: List[np.ndarray],
    num_classes: int = 80
) -> dict:
    """
    Decode raw YOLOv8 output into boxes, scores, and class IDs.
    
    Args:
        outputs: List of output tensors from model
        num_classes: Number of object classes
        
    Returns:
        dict with boxes, scores, class_ids
    """
    output = outputs[0]  # First output tensor
    
    # YOLOv8 output shape: (1, 84, 8400) for 80 classes
    # First 4 values: box coordinates (cx, cy, w, h)
    # Next 80 values: class scores
    
    predictions = output[0]  # Remove batch dimension (8400, 84) → (84, 8400)
    predictions = predictions.T  # Transpose to (8400, 84)
    
    # Extract boxes and scores
    boxes = predictions[:, :4]  # (8400, 4)
    class_scores = predictions[:, 4:]  # (8400, 80)
    
    # Get max class score and ID for each detection
    max_scores = np.max(class_scores, axis=1)  # (8400,)
    class_ids = np.argmax(class_scores, axis=1)  # (8400,)
    
    return {
        "boxes": boxes,  # (N, 4) - cx, cy, w, h
        "scores": max_scores,  # (N,)
        "class_ids": class_ids,  # (N,)
        "num_detections": len(boxes)
    }


@workflow_node("filter_by_confidence", isolation_mode="none", log_level="SILENT", skip_metadata=True)
def filter_by_confidence_node(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    conf_threshold: float = 0.25
) -> dict:
    """
    Filter detections by confidence threshold.
    
    Args:
        boxes: Bounding boxes (N, 4)
        scores: Confidence scores (N,)
        class_ids: Class IDs (N,)
        conf_threshold: Minimum confidence
        
    Returns:
        dict with filtered boxes, scores, class_ids
    """
    # Filter by confidence
    mask = scores >= conf_threshold
    
    filtered_boxes = boxes[mask]
    filtered_scores = scores[mask]
    filtered_class_ids = class_ids[mask]
    
    return {
        "boxes": filtered_boxes,
        "scores": filtered_scores,
        "class_ids": filtered_class_ids,
        "num_detections": len(filtered_boxes),
        "num_filtered": len(boxes) - len(filtered_boxes)
    }


@workflow_node("convert_cxcywh_to_xyxy", isolation_mode="none", log_level="SILENT", skip_metadata=True)
def convert_cxcywh_to_xyxy_node(
    boxes: np.ndarray,
    scores: np.ndarray = None,
    class_ids: np.ndarray = None
) -> dict:
    """
    Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2) format.
    
    Args:
        boxes: Boxes in (cx, cy, w, h) format (N, 4)
        scores: Confidence scores (N,) - passed through
        class_ids: Class IDs (N,) - passed through
        
    Returns:
        dict with converted boxes (x1, y1, x2, y2) and passed-through scores/class_ids
    """
    converted = np.zeros_like(boxes)
    
    converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1 = cx - w/2
    converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1 = cy - h/2
    converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2 = cx + w/2
    converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2 = cy + h/2
    
    result = {
        "boxes": converted,
        "format": "xyxy"
    }
    
    # Pass through scores and class_ids if provided
    if scores is not None:
        result["scores"] = scores
    if class_ids is not None:
        result["class_ids"] = class_ids
    
    return result


@workflow_node("apply_nms", isolation_mode="none", log_level="SILENT", skip_metadata=True)
def apply_nms_node(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float = 0.45
) -> dict:
    """
    Apply Non-Maximum Suppression to remove duplicate detections.
    
    Args:
        boxes: Boxes in (x1, y1, x2, y2) format (N, 4)
        scores: Confidence scores (N,)
        class_ids: Class IDs (N,)
        iou_threshold: IoU threshold for NMS
        
    Returns:
        dict with filtered boxes, scores, class_ids after NMS
    """
    import cv2
    
    if len(boxes) == 0:
        return {
            "boxes": boxes,
            "scores": scores,
            "class_ids": class_ids,
            "num_detections": 0,
            "num_suppressed": 0
        }
    
    # Perform NMS per class
    keep_indices = []
    unique_classes = np.unique(class_ids)
    
    for class_id in unique_classes:
        class_mask = class_ids == class_id
        class_boxes = boxes[class_mask]
        class_scores = scores[class_mask]
        class_indices = np.where(class_mask)[0]
        
        # OpenCV NMS
        indices = cv2.dnn.NMSBoxes(
            class_boxes.tolist(),
            class_scores.tolist(),
            score_threshold=0.0,  # Already filtered
            nms_threshold=iou_threshold
        )
        
        if len(indices) > 0:
            keep_indices.extend(class_indices[indices.flatten()].tolist())
    
    # Apply kept indices
    kept_boxes = boxes[keep_indices]
    kept_scores = scores[keep_indices]
    kept_class_ids = class_ids[keep_indices]
    
    return {
        "boxes": kept_boxes,
        "scores": kept_scores,
        "class_ids": kept_class_ids,
        "num_detections": len(kept_boxes),
        "num_suppressed": len(boxes) - len(kept_boxes)
    }


@workflow_node("scale_boxes_to_original", isolation_mode="none", log_level="SILENT", skip_metadata=True)
def scale_boxes_to_original_node(
    boxes: np.ndarray,
    original_width: int,
    original_height: int,
    input_width: int = 640,
    input_height: int = 640,
    pad_x: int = 0,
    pad_y: int = 0,
    scale: float = 1.0,
    scores: np.ndarray = None,
    class_ids: np.ndarray = None
) -> dict:
    """
    Scale boxes from model input size to original image size.
    
    Args:
        boxes: Boxes in (x1, y1, x2, y2) format (N, 4)
        original_width: Original image width
        original_height: Original image height
        input_width: Model input width
        input_height: Model input height
        pad_x: Letterbox padding X
        pad_y: Letterbox padding Y
        scale: Letterbox scale factor
        scores: Confidence scores (N,) - passed through
        class_ids: Class IDs (N,) - passed through
        
    Returns:
        dict with scaled boxes and passed-through scores/class_ids
    """
    if len(boxes) == 0:
        result = {"boxes": boxes}
        if scores is not None:
            result["scores"] = scores
        if class_ids is not None:
            result["class_ids"] = class_ids
        return result
    
    scaled_boxes = boxes.copy()
    
    # Remove letterbox padding
    scaled_boxes[:, [0, 2]] -= pad_x
    scaled_boxes[:, [1, 3]] -= pad_y
    
    # Scale back to original size
    scaled_boxes[:, [0, 2]] /= scale
    scaled_boxes[:, [1, 3]] /= scale
    
    # Clip to image boundaries
    scaled_boxes[:, [0, 2]] = np.clip(scaled_boxes[:, [0, 2]], 0, original_width)
    scaled_boxes[:, [1, 3]] = np.clip(scaled_boxes[:, [1, 3]], 0, original_height)
    
    result = {
        "boxes": scaled_boxes
    }
    
    # Pass through scores and class_ids if provided
    if scores is not None:
        result["scores"] = scores
    if class_ids is not None:
        result["class_ids"] = class_ids
    
    return result


@workflow_node("format_detections_coco", isolation_mode="none", log_level="SILENT", skip_metadata=True)
def format_detections_coco_node(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray
) -> dict:
    """
    Format detections with COCO class names.
    
    Args:
        boxes: Boxes in (x1, y1, x2, y2) format (N, 4)
        scores: Confidence scores (N,)
        class_ids: Class IDs (N,)
        
    Returns:
        dict with formatted detections
    """
    from inference_engine import get_coco_class_name
    
    detections = []
    for i in range(len(boxes)):
        detection = {
            "bbox": boxes[i].tolist(),
            "confidence": float(scores[i]),
            "class_id": int(class_ids[i]),
            "class_name": get_coco_class_name(int(class_ids[i]))
        }
        detections.append(detection)
    
    # Sort by confidence
    detections.sort(key=lambda x: x['confidence'], reverse=True)
    
    return {
        "detections": detections,
        "num_detections": len(detections),
        "top_3": detections[:3] if len(detections) >= 3 else detections
    }


@workflow_node("create_detection_summary", isolation_mode="auto")
def create_detection_summary_node(
    detections: List[dict],
    inference_time_ms: float,
    fps: float,
    provider: str = "Unknown"
) -> dict:
    """
    Create a text summary of detection results.
    
    Args:
        detections: List of detection dicts
        inference_time_ms: Average inference time
        fps: Frames per second
        provider: Backend provider name
        
    Returns:
        dict with summary text
    """
    summary_lines = [
        f"Detection Summary ({provider})",
        f"=================={'=' * len(provider)}",
        f"",
        f"Performance:",
        f"  - Inference time: {inference_time_ms:.1f}ms",
        f"  - FPS: {fps:.1f}",
        f"",
        f"Detections: {len(detections)} objects found",
    ]
    
    if detections:
        summary_lines.append("")
        summary_lines.append("Top 3 detections:")
        for i, det in enumerate(detections[:3], 1):
            summary_lines.append(
                f"  {i}. {det['class_name']} ({det['confidence']:.2%})"
            )
    
    summary = "\n".join(summary_lines)
    
    return {
        "summary": summary,
        "num_detections": len(detections),
        "performance": {
            "inference_time_ms": inference_time_ms,
            "fps": fps,
            "provider": provider
        }
    }
