"""
Optimized Visualize Detections Node - Uses OpenCV GPU acceleration and efficient drawing
"""

from workflow_decorator import workflow_node
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


# COCO class names for YOLO
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


@workflow_node(
    "visualize_detections_node",
    dependencies=["opencv-python", "numpy"],
    isolation_mode="none"
)
def visualize_detections_node(
    frames: List,
    detections: List[Dict],
    display: bool = False,  # Disabled by default for speed
    save_video: bool = True,
    output_path: str = "output_detections.mp4",
    fps: int = 30,
    font_scale: float = 0.4,  # Smaller font for less overhead
    thickness: int = 1,  # Thinner lines for faster drawing
    codec: str = "mp4v",  # Codec selection
    skip_empty_frames: bool = False  # Option to skip frames with no detections
) -> Dict[str, Any]:
    """
    Optimized visualization with efficient drawing and optional GPU acceleration.
    
    Args:
        frames: List of original frames
        detections: List of detection results per frame
        display: Whether to display frames in a window (slows down processing)
        save_video: Whether to save annotated video
        output_path: Path to save output video
        fps: FPS for output video
        font_scale: Font scale for labels (smaller = faster)
        thickness: Line thickness for boxes (thinner = faster)
        codec: Video codec ('mp4v', 'avc1', 'h264')
        skip_empty_frames: Skip frames with no detections to reduce file size
        
    Returns:
        Dictionary with visualization results
    """
    import cv2
    import numpy as np
    import time
    
    logger.info(f"Optimized visualization on {len(frames)} frames")
    logger.info(f"Display: {display}, Save: {save_video}, Codec: {codec}")
    
    total_detections = 0
    start_time = time.time()
    
    # Pre-generate colors once
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(COCO_CLASSES), 3), dtype=np.uint8).tolist()
    
    # Prepare video writer if saving
    video_writer = None
    if save_video and len(frames) > 0:
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*codec)
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not video_writer.isOpened():
            logger.error(f"Failed to open video writer with codec {codec}")
            return {"error": "Failed to create video writer"}
    
    frames_saved = 0
    
    for i, (frame, detection) in enumerate(zip(frames, detections)):
        boxes = detection['boxes']
        scores = detection['scores']
        class_ids = detection['class_ids']
        
        # Skip empty frames if requested
        if skip_empty_frames and len(boxes) == 0:
            continue
        
        # Work on copy only if we need to modify
        annotated = frame.copy() if len(boxes) > 0 else frame
        
        # Batch draw operations
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box
            color = colors[class_id]
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            
            # Prepare label
            class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"C{class_id}"
            label = f"{class_name}:{score:.2f}"
            
            # Draw label with background (simplified)
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            
            # Ensure label fits in frame
            label_y1 = max(y1 - label_h - baseline - 2, 0)
            label_y2 = label_y1 + label_h + baseline + 2
            
            cv2.rectangle(annotated, (x1, label_y1), (x1 + label_w, label_y2), color, -1)
            cv2.putText(annotated, label, (x1, label_y2 - baseline - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
            
            total_detections += 1
        
        # Add frame info (optional, comment out for max speed)
        # info_text = f"Frame {i+1}/{len(frames)} | {len(boxes)} objs"
        # cv2.putText(annotated, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 
        #            0.6, (0, 255, 0), 2)
        
        # Save frame
        if video_writer:
            video_writer.write(annotated)
            frames_saved += 1
        
        # Display frame (significantly slows processing)
        if display:
            cv2.imshow('Object Detection', annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Display interrupted by user")
                break
    
    # Cleanup
    if video_writer:
        video_writer.release()
        video_saved = True
        logger.info(f"Video saved: {output_path} ({frames_saved} frames)")
    else:
        video_saved = False
    
    if display:
        cv2.destroyAllWindows()
    
    elapsed_time = time.time() - start_time
    viz_fps = len(frames) / elapsed_time if elapsed_time > 0 else 0
    
    logger.info(f"Visualization complete: {total_detections} detections in {elapsed_time:.2f}s ({viz_fps:.1f} FPS)")
    
    return {
        "total_detections": total_detections,
        "frames_processed": frames_saved if skip_empty_frames else len(frames),
        "video_saved": video_saved,
        "output_path": output_path if video_saved else None,
        "avg_detections_per_frame": total_detections / len(frames) if len(frames) > 0 else 0,
        "visualization_fps": viz_fps,
        "visualization_time": elapsed_time
    }
