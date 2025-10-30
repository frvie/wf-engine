"""
Granular Video Loop Node

Uses atomic nodes to process video frames in a loop.
This is a higher-level orchestrator that calls atomic nodes frame-by-frame.
"""

import cv2
import time
import logging
import numpy as np
from typing import Dict, Any
from workflow_decorator import workflow_node


@workflow_node("granular_video_loop",
               dependencies=["opencv-python", "numpy"],
               isolation_mode="none")
def granular_video_loop_node(
    source: str = None,
    width: int = None,
    height: int = None,
    fps: float = None,
    session: Any = None,
    input_name: str = None,
    model_path: str = None,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    display_fps: bool = True,
    save_video: bool = False,
    output_path: str = "output.mp4",
    max_duration: int = 0
) -> Dict[str, Any]:
    """
    Process video using atomic inference nodes in a loop.
    
    This node orchestrates the video processing loop while using
    atomic nodes for each inference operation.
    
    Args:
        source: Video source (from video_capture node)
        width, height, fps: Video properties
        session: ONNX session (from directml_session node)
        input_name: Model input name
        model_path: Path to model
        conf_threshold: Confidence threshold
        iou_threshold: NMS IoU threshold
        display_fps: Show FPS overlay
        save_video: Save output video
        output_path: Output video path
        max_duration: Maximum duration in seconds (0 = unlimited)
        
    Returns:
        dict with processing statistics
    """
    logger = logging.getLogger('workflow.video.granular_loop')
    
    # Import atomic node functions (composable with dict I/O)
    from workflow_nodes.atomic.image_ops import (
        resize_image_letterbox_node,
        normalize_image_node,
        hwc_to_chw_node,
        add_batch_dimension_node
    )
    from workflow_nodes.atomic.onnx_ops import run_onnx_inference_single_node
    from workflow_nodes.atomic.yolo_ops import (
        decode_yolo_v8_output_node,
        filter_by_confidence_node,
        convert_cxcywh_to_xyxy_node,
        apply_nms_node,
        scale_boxes_to_original_node,
        format_detections_coco_node
    )
    
    
    # Open video capture
    cap = cv2.VideoCapture(source if not isinstance(source, int) else int(source))
    
    if not cap.isOpened():
        return {"error": f"Failed to open video source: {source}"}
    
    # Get actual video properties if not provided
    if width is None:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    if height is None:
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps is None:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    logger.info(f"Processing video: {width}x{height} @ {fps:.1f} FPS")
    
    # Setup video writer if saving
    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        logger.info(f"Saving output to: {output_path}")
    
    # Processing loop
    frame_count = 0
    total_detections = 0
    start_time = time.time()
    fps_update_time = start_time
    current_fps = 0
    recording = save_video
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_start = time.time()
            original_height, original_width = frame.shape[:2]
            
            # === COMPOSABLE ATOMIC PIPELINE ===
            # Using shared context dict for efficiency while keeping composability
            ctx = {'image': frame}
            
            # 1. Resize with letterbox
            ctx.update(resize_image_letterbox_node(image=ctx['image'], target_width=640, target_height=640))
            
            # 2. Normalize
            ctx.update(normalize_image_node(image=ctx['image'], scale=255.0))
            
            # 3. HWC to CHW
            ctx.update(hwc_to_chw_node(image=ctx['image']))
            
            # 4. Add batch dimension
            ctx.update(add_batch_dimension_node(image=ctx['image']))
            
            # 5. Run inference
            ctx.update(run_onnx_inference_single_node(session=session, input_name=input_name, image=ctx['image']))
            
            # 6. Decode YOLO output
            ctx.update(decode_yolo_v8_output_node(outputs=ctx['outputs'], num_classes=80))
            
            # 7. Filter by confidence
            ctx.update(filter_by_confidence_node(boxes=ctx['boxes'], scores=ctx['scores'], class_ids=ctx['class_ids'], conf_threshold=conf_threshold))
            
            # 8. Convert box format
            ctx.update(convert_cxcywh_to_xyxy_node(boxes=ctx['boxes'], scores=ctx.get('scores'), class_ids=ctx.get('class_ids')))
            
            # 9. Apply NMS
            ctx.update(apply_nms_node(boxes=ctx['boxes'], scores=ctx['scores'], class_ids=ctx['class_ids'], iou_threshold=iou_threshold))
            
            # 10. Scale boxes to original size
            ctx.update(scale_boxes_to_original_node(
                boxes=ctx['boxes'], scores=ctx.get('scores'), class_ids=ctx.get('class_ids'),
                input_width=640, input_height=640,
                original_width=original_width, original_height=original_height,
                pad_x=ctx.get('pad_x', 0), pad_y=ctx.get('pad_y', 0), scale=ctx.get('scale', 1.0)
            ))
            
            # 11. Format detections
            ctx.update(format_detections_coco_node(boxes=ctx['boxes'], scores=ctx['scores'], class_ids=ctx['class_ids']))
            
            detections = ctx['detections']
            total_detections += len(detections)
            
            # === VISUALIZATION ===
            # Draw bounding boxes
            display_frame = frame.copy()
            for det in detections:
                bbox = det['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                conf = det['confidence']
                label = det['class_name']
                
                # Draw box
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label_text = f"{label} {conf:.2f}"
                (label_w, label_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(display_frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), (0, 255, 0), -1)
                cv2.putText(display_frame, label_text, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Calculate and display FPS
            frame_end = time.time()
            frame_time = frame_end - frame_start
            if frame_time > 0:
                instant_fps = 1.0 / frame_time
                # Smooth FPS
                if time.time() - fps_update_time > 0.5:
                    current_fps = instant_fps
                    fps_update_time = time.time()
            
            if display_fps:
                fps_text = f"FPS: {current_fps:.1f} | Detections: {len(detections)}"
                if recording:
                    fps_text += " | REC"
                cv2.putText(display_frame, fps_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Granular Video Detection', display_frame)
            
            # Save frame if recording
            if recording and writer:
                writer.write(display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                logger.info("Quit requested")
                break
            elif key == ord('s') or key == ord('S'):
                recording = not recording
                logger.info(f"Recording {'started' if recording else 'stopped'}")
                if recording and not writer:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            frame_count += 1
            
            # Check max duration
            if max_duration > 0 and (time.time() - start_time) > max_duration:
                logger.info(f"Max duration {max_duration}s reached")
                break
    
    finally:
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
    
    elapsed_time = time.time() - start_time
    avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    
    logger.info(f"Processed {frame_count} frames in {elapsed_time:.2f}s")
    logger.info(f"Average FPS: {avg_fps:.1f}")
    logger.info(f"Total detections: {total_detections}")
    
    return {
        "frames_processed": frame_count,
        "total_detections": total_detections,
        "elapsed_time": elapsed_time,
        "average_fps": avg_fps,
        "output_path": output_path if save_video else None,
        "status": "success"
    }
