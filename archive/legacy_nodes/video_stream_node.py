"""
Optimized Video Stream Node - Multi-threaded capture for better FPS
"""

from workflow_decorator import workflow_node
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


@workflow_node(
    "video_stream_node",
    dependencies=["opencv-python", "numpy"],
    isolation_mode="none"
)
def video_stream_node(
    source: str = "0",
    max_frames: int = 300,
    resize_width: int = 640,
    resize_height: int = 640,
    fps_limit: int = 30,
    use_threading: bool = True
) -> Dict[str, Any]:
    """
    Optimized video stream with threading for better capture performance.
    
    Args:
        source: Video source (webcam index or file path)
        max_frames: Maximum number of frames to process
        resize_width: Width to resize frames to
        resize_height: Height to resize frames to
        fps_limit: Maximum FPS to process
        use_threading: Use threaded capture for better performance
        
    Returns:
        Dictionary with video stream information
    """
    import cv2
    import numpy as np
    import time
    from threading import Thread
    from queue import Queue
    
    # Convert source to int if it's a webcam index
    try:
        source_parsed = int(source)
    except ValueError:
        source_parsed = source
    
    # Open video capture
    cap = cv2.VideoCapture(source_parsed)
    
    if not cap.isOpened():
        logger.error(f"Failed to open video source: {source}")
        return {
            "error": f"Failed to open video source: {source}",
            "frames_processed": 0
        }
    
    # Optimize capture settings
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to get latest frames
    cap.set(cv2.CAP_PROP_FPS, fps_limit)  # Set desired FPS
    
    # Get video properties
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    
    logger.info(f"Video stream opened: {original_width}x{original_height} @ {original_fps} FPS")
    logger.info(f"Will resize to: {resize_width}x{resize_height}")
    logger.info(f"Processing up to {max_frames} frames")
    
    if use_threading:
        # Threaded capture for better performance
        frame_queue = Queue(maxsize=30)
        stop_flag = [False]
        
        def capture_thread():
            while not stop_flag[0]:
                ret, frame = cap.read()
                if not ret:
                    stop_flag[0] = True
                    break
                if not frame_queue.full():
                    frame_queue.put(frame)
        
        # Start capture thread
        thread = Thread(target=capture_thread, daemon=True)
        thread.start()
        
        frames = []
        frame_count = 0
        start_time = time.time()
        frame_delay = 1.0 / fps_limit if fps_limit > 0 else 0
        last_frame_time = 0
        
        while frame_count < max_frames and not stop_flag[0]:
            if frame_queue.empty():
                time.sleep(0.001)
                continue
            
            # FPS limiting
            current_time = time.time()
            if current_time - last_frame_time < frame_delay:
                frame_queue.get()  # Discard frame
                continue
            last_frame_time = current_time
            
            frame = frame_queue.get()
            
            # Resize frame
            if (original_width, original_height) != (resize_width, resize_height):
                frame = cv2.resize(frame, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)
            
            frames.append(frame)
            frame_count += 1
            
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                logger.info(f"Captured {frame_count} frames ({fps:.1f} FPS)")
        
        stop_flag[0] = True
        thread.join(timeout=1.0)
        
    else:
        # Standard capture (fallback)
        frames = []
        frame_count = 0
        start_time = time.time()
        frame_delay = 1.0 / fps_limit if fps_limit > 0 else 0
        last_frame_time = 0
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            
            if not ret:
                logger.info("End of video stream reached")
                break
            
            # FPS limiting
            current_time = time.time()
            if current_time - last_frame_time < frame_delay:
                continue
            last_frame_time = current_time
            
            # Resize frame
            if (original_width, original_height) != (resize_width, resize_height):
                frame = cv2.resize(frame, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)
            
            frames.append(frame)
            frame_count += 1
            
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                logger.info(f"Captured {frame_count} frames ({fps:.1f} FPS)")
    
    cap.release()
    
    elapsed_time = time.time() - start_time
    avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    
    logger.info(f"Video stream capture complete: {frame_count} frames in {elapsed_time:.2f}s ({avg_fps:.1f} FPS)")
    
    return {
        "frames": frames,
        "frame_count": frame_count,
        "original_width": original_width,
        "original_height": original_height,
        "resize_width": resize_width,
        "resize_height": resize_height,
        "original_fps": original_fps,
        "actual_fps": avg_fps,
        "total_time": elapsed_time,
        "source": source
    }
