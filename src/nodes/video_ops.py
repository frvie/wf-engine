"""
Atomic Video Operations

Provides granular, composable video processing nodes.
"""

import cv2
import logging
from typing import Dict, Any
from src.core.decorator import workflow_node


@workflow_node("open_video_capture", isolation_mode="auto")
def open_video_capture_node(source: str = "0") -> Dict[str, Any]:
    """
    Open video capture from webcam or file.
    
    Args:
        source: Video source ("0" for webcam, or path to video file)
        
    Returns:
        dict with video capture info
    """
    logger = logging.getLogger('workflow.video.capture')
    
    # Convert "0" to integer for webcam
    if source.isdigit():
        source = int(source)
    
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        return {
            "error": f"Failed to open video source: {source}",
            "status": "failed"
        }
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"Opened video: {width}x{height} @ {fps:.1f} FPS")
    if frame_count > 0:
        logger.info(f"Total frames: {frame_count}")
    
    # Note: We can't return the capture object itself as it's not serializable
    # Instead, return the source info so nodes can re-open it
    cap.release()
    
    return {
        "source": source,
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "is_webcam": isinstance(source, int),
        "status": "success"
    }


@workflow_node("read_video_frame", isolation_mode="auto")
def read_video_frame_node(capture: Any) -> Dict[str, Any]:
    """
    Read a single frame from video capture.
    
    Note: This is meant to be called in a loop, but cv2.VideoCapture
    objects cannot be serialized, so this is a placeholder for documentation.
    
    Args:
        capture: OpenCV VideoCapture object
        
    Returns:
        dict with frame data
    """
    ret, frame = capture.read()
    
    if not ret:
        return {
            "frame": None,
            "status": "end_of_stream",
            "error": "No frame available"
        }
    
    return {
        "frame": frame,
        "height": frame.shape[0],
        "width": frame.shape[1],
        "channels": frame.shape[2],
        "status": "success"
    }


@workflow_node("display_frame", isolation_mode="auto")
def display_frame_node(
    frame,
    window_name: str = "Video",
    wait_key: int = 1
) -> Dict[str, Any]:
    """
    Display a video frame in a window.
    
    Args:
        frame: Frame to display (numpy array)
        window_name: Name of display window
        wait_key: Milliseconds to wait for key press
        
    Returns:
        dict with key press info
    """
    cv2.imshow(window_name, frame)
    key = cv2.waitKey(wait_key) & 0xFF
    
    return {
        "key_pressed": key,
        "window_name": window_name,
        "quit_requested": key == ord('q') or key == ord('Q'),
        "status": "success"
    }


