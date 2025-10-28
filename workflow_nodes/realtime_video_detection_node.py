"""
Real-Time Video Detection Node - Continuous processing with live display
"""

from workflow_decorator import workflow_node
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


@workflow_node(
    "realtime_video_detection_node",
    dependencies=["opencv-python", "numpy", "onnxruntime-directml"],
    isolation_mode="none"
)
def realtime_video_detection_node(
    source: str = "0",
    model_path: str = "models/yolov8s.onnx",
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    display_fps: bool = True,
    save_video: bool = False,
    output_path: str = "realtime_output.mp4",
    max_duration: int = 0  # 0 = infinite, otherwise seconds
) -> Dict[str, Any]:
    """
    Real-time video object detection with live display.
    
    Args:
        source: Video source (webcam index or file path)
        model_path: Path to ONNX model
        conf_threshold: Detection confidence threshold
        iou_threshold: NMS IOU threshold
        display_fps: Show FPS counter on screen
        save_video: Save output video to file
        output_path: Path for saved video
        max_duration: Maximum duration in seconds (0 for infinite)
        
    Returns:
        Dictionary with processing statistics
    """
    import cv2
    import numpy as np
    import onnxruntime as ort
    import time
    import os
    
    # COCO class names
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
        'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    # Convert source to int if it's a webcam index
    try:
        source_parsed = int(source)
    except ValueError:
        source_parsed = source
    
    logger.info(f"Starting real-time detection on source: {source}")
    logger.info(f"Model: {model_path}")
    logger.info(f"Press 'Q' to quit, 'S' to toggle save")
    
    # Open video capture
    cap = cv2.VideoCapture(source_parsed)
    if not cap.isOpened():
        logger.error(f"Failed to open video source: {source}")
        return {"error": f"Failed to open video source: {source}"}
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    logger.info(f"Video opened: {frame_width}x{frame_height} @ {fps} FPS")
    
    # Initialize ONNX Runtime with DirectML
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(model_path, session_options, providers=providers)
    
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    
    logger.info(f"Model loaded with providers: {session.get_providers()}")
    
    # GPU Warmup
    warmup_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
    for _ in range(3):
        session.run(output_names, {input_name: warmup_input})
    logger.info("GPU warmup complete")
    
    # Video writer (optional)
    video_writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Pre-generate colors
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(COCO_CLASSES), 3), dtype=np.uint8).tolist()
    
    # Statistics
    frame_count = 0
    total_detections = 0
    start_time = time.time()
    fps_start = time.time()
    fps_counter = 0
    current_fps = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of video stream")
                break
            
            # Preprocess frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(frame_rgb, (640, 640))
            input_tensor = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
            input_tensor = np.expand_dims(input_tensor, axis=0)
            
            # Run inference
            inference_start = time.time()
            outputs = session.run(output_names, {input_name: input_tensor})
            inference_time = time.time() - inference_start
            
            # Post-process (YOLOv8 format: [batch, 84, 8400])
            predictions = outputs[0]
            if predictions.shape[1] == 84:
                predictions = np.transpose(predictions, (0, 2, 1))
            
            frame_predictions = predictions[0]
            
            # Extract detections
            class_scores = frame_predictions[:, 4:]
            max_scores = np.max(class_scores, axis=1)
            class_ids = np.argmax(class_scores, axis=1)
            
            valid_mask = max_scores > conf_threshold
            valid_indices = np.where(valid_mask)[0]
            
            boxes = []
            scores = []
            det_class_ids = []
            
            if len(valid_indices) > 0:
                valid_predictions = frame_predictions[valid_indices]
                valid_scores = max_scores[valid_indices]
                valid_class_ids = class_ids[valid_indices]
                
                # Convert bbox format
                for pred, score, cls_id in zip(valid_predictions, valid_scores, valid_class_ids):
                    x_center, y_center, width, height = pred[:4]
                    
                    # Scale to original frame size
                    x_center = x_center * frame_width / 640
                    y_center = y_center * frame_height / 640
                    width = width * frame_width / 640
                    height = height * frame_height / 640
                    
                    x1 = int(max(0, x_center - width / 2))
                    y1 = int(max(0, y_center - height / 2))
                    x2 = int(min(frame_width, x_center + width / 2))
                    y2 = int(min(frame_height, y_center + height / 2))
                    
                    boxes.append([x1, y1, x2, y2])
                    scores.append(float(score))
                    det_class_ids.append(int(cls_id))
                
                # Apply NMS
                if len(boxes) > 0:
                    boxes, scores, det_class_ids = apply_nms(boxes, scores, det_class_ids, iou_threshold)
            
            # Draw detections
            for box, score, cls_id in zip(boxes, scores, det_class_ids):
                x1, y1, x2, y2 = box
                color = colors[cls_id]
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                class_name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f"C{cls_id}"
                label = f"{class_name}: {score:.2f}"
                
                (label_w, label_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                
                cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (255, 255, 255), 1)
            
            # Update statistics
            frame_count += 1
            total_detections += len(boxes)
            fps_counter += 1
            
            # Calculate FPS
            if time.time() - fps_start >= 1.0:
                current_fps = fps_counter
                fps_counter = 0
                fps_start = time.time()
            
            # Display info on frame
            if display_fps:
                info_lines = [
                    f"FPS: {current_fps}",
                    f"Inference: {inference_time*1000:.1f}ms",
                    f"Detections: {len(boxes)}",
                    f"Total: {total_detections}"
                ]
                
                y_offset = 30
                for line in info_lines:
                    cv2.putText(frame, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                               0.6, (0, 255, 0), 2)
                    y_offset += 25
            
            # Save frame
            if video_writer:
                video_writer.write(frame)
            
            # Display frame
            cv2.imshow('Real-Time Object Detection (Press Q to quit)', frame)
            
            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                logger.info("User requested quit")
                break
            elif key == ord('s') or key == ord('S'):
                if video_writer:
                    logger.info("Stopping recording")
                    video_writer.release()
                    video_writer = None
                else:
                    logger.info("Starting recording")
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(output_path, fourcc, fps, 
                                                   (frame_width, frame_height))
            
            # Check max duration
            if max_duration > 0 and (time.time() - start_time) >= max_duration:
                logger.info(f"Max duration {max_duration}s reached")
                break
                
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        # Cleanup
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
    
    elapsed_time = time.time() - start_time
    avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    
    logger.info(f"Real-time detection complete:")
    logger.info(f"  Frames processed: {frame_count}")
    logger.info(f"  Total detections: {total_detections}")
    logger.info(f"  Duration: {elapsed_time:.2f}s")
    logger.info(f"  Average FPS: {avg_fps:.1f}")
    
    return {
        "frames_processed": frame_count,
        "total_detections": total_detections,
        "duration": elapsed_time,
        "avg_fps": avg_fps,
        "video_saved": video_writer is not None,
        "output_path": output_path if save_video else None
    }


def apply_nms(boxes, scores, class_ids, iou_threshold):
    """Non-Maximum Suppression"""
    import numpy as np
    
    if len(boxes) == 0:
        return [], [], []
    
    boxes = np.array(boxes)
    scores = np.array(scores)
    class_ids = np.array(class_ids)
    
    order = scores.argsort()[::-1]
    keep = []
    
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        
        if len(order) == 1:
            break
        
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
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return boxes[keep].tolist(), scores[keep].tolist(), class_ids[keep].tolist()
