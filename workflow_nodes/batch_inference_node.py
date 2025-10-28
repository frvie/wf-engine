"""
Optimized Batch Inference Node - Uses GPU pre-warming and batch processing
"""

from workflow_decorator import workflow_node
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


@workflow_node(
    "batch_inference_node",
    dependencies=["numpy", "onnxruntime-directml"],
    isolation_mode="subprocess",
    environment="directml"
)
def batch_inference_node(
    frames: List,
    model_info: Dict[str, Any] = None,
    batch_size: int = 4,  # Increased default batch size
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    warmup_iterations: int = 3
) -> Dict[str, Any]:
    """
    Optimized batch inference with larger batches and GPU warmup.
    
    Args:
        frames: List of frames (numpy arrays)
        model_info: Model information from loader node
        batch_size: Number of frames to process at once (higher = faster)
        conf_threshold: Confidence threshold for detections
        iou_threshold: IOU threshold for NMS
        warmup_iterations: Number of warmup iterations for GPU
        
    Returns:
        Dictionary with detection results for all frames
    """
    import numpy as np
    import onnxruntime as ort
    import time
    import os
    
    if model_info is None:
        return {"error": "model_info is required"}
    
    model_path = model_info.get('model_path', 'models/yolov8s.onnx')
    
    # Handle subprocess temp directory
    if 'temp' in os.getcwd().lower() or not os.path.isabs(model_path):
        model_path = os.path.join(r'C:\dev\workflow_engine', 'models', 'yolov8s.onnx')
    
    input_shape = model_info.get('input_shape', [1, 3, 640, 640])
    
    logger.info(f"Starting optimized batch inference on {len(frames)} frames")
    logger.info(f"Model: {model_path}")
    logger.info(f"Batch size: {batch_size}")
    
    # Create session with DirectML
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.enable_mem_pattern = True
    session_options.enable_cpu_mem_arena = True
    
    providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(model_path, session_options, providers=providers)
    
    logger.info(f"ONNX Runtime providers: {session.get_providers()}")
    
    # Get input/output names
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    
    # GPU Warmup
    logger.info(f"Warming up GPU with {warmup_iterations} iterations...")
    warmup_batch = np.random.randn(batch_size, 3, 640, 640).astype(np.float32)
    for _ in range(warmup_iterations):
        session.run(output_names, {input_name: warmup_batch})
    logger.info("GPU warmup complete")
    
    # Pre-allocate arrays for better performance
    all_detections = []
    total_inference_time = 0
    frames_processed = 0
    
    # Preprocess all frames first (vectorized operations)
    logger.info("Preprocessing frames...")
    preprocessed_frames = []
    for frame in frames:
        # Convert BGR to RGB and normalize
        frame_rgb = frame[:, :, ::-1].astype(np.float32) / 255.0
        # Transpose to CHW format
        frame_chw = frame_rgb.transpose(2, 0, 1)
        preprocessed_frames.append(frame_chw)
    
    logger.info(f"Preprocessing complete, starting inference...")
    
    # Process frames in batches
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i + batch_size]
        batch_preprocessed = preprocessed_frames[i:i + batch_size]
        
        # Stack batch
        actual_batch_size = len(batch_preprocessed)
        batch_tensor = np.array(batch_preprocessed, dtype=np.float32)
        
        # Run inference
        start_time = time.time()
        outputs = session.run(output_names, {input_name: batch_tensor})
        inference_time = time.time() - start_time
        total_inference_time += inference_time
        
        # Process outputs for each frame in batch
        for j, frame in enumerate(batch_frames):
            frame_height, frame_width = frame.shape[:2]
            
            # YOLOv8 output format: [batch, 84, 8400]
            # Transpose to [batch, 8400, 84]
            predictions = outputs[0]
            if predictions.shape[1] == 84:  # [batch, 84, 8400]
                predictions = np.transpose(predictions, (0, 2, 1))
            
            # Extract detections for this frame (batch index j)
            frame_predictions = predictions[j]
            
            # Vectorized confidence filtering
            class_scores = frame_predictions[:, 4:]  # [8400, 80]
            max_scores = np.max(class_scores, axis=1)  # [8400]
            class_ids = np.argmax(class_scores, axis=1)  # [8400]
            
            # Filter by confidence threshold
            valid_mask = max_scores > conf_threshold
            valid_indices = np.where(valid_mask)[0]
            
            if len(valid_indices) == 0:
                all_detections.append({
                    "frame_index": frames_processed,
                    "boxes": [],
                    "scores": [],
                    "class_ids": [],
                    "num_detections": 0
                })
                frames_processed += 1
                continue
            
            # Extract valid detections
            valid_predictions = frame_predictions[valid_indices]
            valid_scores = max_scores[valid_indices]
            valid_class_ids = class_ids[valid_indices]
            
            # Convert bbox format (vectorized)
            x_centers = valid_predictions[:, 0]
            y_centers = valid_predictions[:, 1]
            widths = valid_predictions[:, 2]
            heights = valid_predictions[:, 3]
            
            x1s = np.clip(x_centers - widths / 2, 0, frame_width).astype(int)
            y1s = np.clip(y_centers - heights / 2, 0, frame_height).astype(int)
            x2s = np.clip(x_centers + widths / 2, 0, frame_width).astype(int)
            y2s = np.clip(y_centers + heights / 2, 0, frame_height).astype(int)
            
            boxes = np.stack([x1s, y1s, x2s, y2s], axis=1)
            
            # Apply NMS
            if len(boxes) > 0:
                boxes, scores, class_ids_filtered = apply_nms_vectorized(
                    boxes, valid_scores, valid_class_ids, iou_threshold
                )
            else:
                boxes, scores, class_ids_filtered = [], [], []
            
            all_detections.append({
                "frame_index": frames_processed,
                "boxes": boxes.tolist() if isinstance(boxes, np.ndarray) else boxes,
                "scores": scores.tolist() if isinstance(scores, np.ndarray) else scores,
                "class_ids": class_ids_filtered.tolist() if isinstance(class_ids_filtered, np.ndarray) else class_ids_filtered,
                "num_detections": len(boxes) if isinstance(boxes, np.ndarray) else len(boxes)
            })
            
            frames_processed += 1
        
        if (i + batch_size) % 30 == 0:
            avg_fps = frames_processed / total_inference_time if total_inference_time > 0 else 0
            logger.info(f"Processed {frames_processed}/{len(frames)} frames ({avg_fps:.1f} FPS)")
    
    avg_fps = frames_processed / total_inference_time if total_inference_time > 0 else 0
    
    logger.info(f"Batch inference complete: {frames_processed} frames in {total_inference_time:.2f}s ({avg_fps:.1f} FPS)")
    
    return {
        "detections": all_detections,
        "frames_processed": frames_processed,
        "total_time": total_inference_time,
        "avg_fps": avg_fps,
        "conf_threshold": conf_threshold,
        "iou_threshold": iou_threshold
    }


def apply_nms_vectorized(boxes, scores, class_ids, iou_threshold):
    """Vectorized Non-Maximum Suppression for better performance"""
    import numpy as np
    
    if len(boxes) == 0:
        return np.array([]), np.array([]), np.array([])
    
    boxes = np.array(boxes) if not isinstance(boxes, np.ndarray) else boxes
    scores = np.array(scores) if not isinstance(scores, np.ndarray) else scores
    class_ids = np.array(class_ids) if not isinstance(class_ids, np.ndarray) else class_ids
    
    # Sort by score
    order = scores.argsort()[::-1]
    
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        
        if len(order) == 1:
            break
        
        # Vectorized IoU calculation
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h
        
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_order = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        union = area_i + area_order - intersection
        
        iou = intersection / (union + 1e-6)
        
        # Keep only boxes with IoU less than threshold
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return boxes[keep], scores[keep], class_ids[keep]
