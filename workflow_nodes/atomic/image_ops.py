"""
Atomic Image Processing Nodes

Small, reusable, composable nodes for image operations.
Each node does ONE thing and does it well.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from workflow_decorator import workflow_node


@workflow_node("read_image", isolation_mode="auto")
def read_image_node(image_path: str) -> dict:
    """
    Read image from file.
    
    Args:
        image_path: Path to image file
        
    Returns:
        dict with image (numpy array), path, and dimensions
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    h, w, c = image.shape
    
    return {
        "image": image,
        "path": image_path,
        "height": h,
        "width": w,
        "channels": c
    }


@workflow_node("resize_image_letterbox", isolation_mode="none", log_level="SILENT", skip_metadata=True)
def resize_image_letterbox_node(
    image: np.ndarray,
    target_width: int = 640,
    target_height: int = 640,
    padding_color: Tuple[int, int, int] = (114, 114, 114)
) -> dict:
    """
    Resize image with letterbox padding to maintain aspect ratio.
    
    Args:
        image: Input image (H, W, C)
        target_width: Target width
        target_height: Target height
        padding_color: RGB color for padding
        
    Returns:
        dict with resized image, scale factor, and padding info
    """
    h, w = image.shape[:2]
    
    # Calculate scale to fit image in target size
    scale = min(target_width / w, target_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create padded canvas
    padded = np.full((target_height, target_width, 3), padding_color, dtype=np.uint8)
    
    # Center the resized image
    pad_x = (target_width - new_w) // 2
    pad_y = (target_height - new_h) // 2
    padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
    
    return {
        "image": padded,
        "scale": scale,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "original_width": w,
        "original_height": h,
        "resized_width": new_w,
        "resized_height": new_h
    }


@workflow_node("normalize_image", isolation_mode="none", log_level="SILENT", skip_metadata=True)
def normalize_image_node(
    image: np.ndarray,
    scale: float = 255.0,
    mean: Optional[Tuple[float, float, float]] = None,
    std: Optional[Tuple[float, float, float]] = None
) -> dict:
    """
    Normalize image pixel values.
    
    Args:
        image: Input image
        scale: Division factor (e.g., 255.0 for 0-1 normalization)
        mean: Optional mean values for each channel
        std: Optional standard deviation for each channel
        
    Returns:
        dict with normalized image
    """
    # Convert to float and scale
    normalized = image.astype(np.float32) / scale
    
    # Apply mean/std if provided
    if mean is not None:
        normalized -= np.array(mean, dtype=np.float32)
    if std is not None:
        normalized /= np.array(std, dtype=np.float32)
    
    return {
        "image": normalized,
        "normalization_scale": scale,
        "mean": mean,
        "std": std
    }


@workflow_node("hwc_to_chw", isolation_mode="none", log_level="SILENT", skip_metadata=True)
def hwc_to_chw_node(image: np.ndarray) -> dict:
    """
    Transpose image from HWC to CHW format.
    
    Args:
        image: Input image (H, W, C)
        
    Returns:
        dict with transposed image (C, H, W)
    """
    transposed = np.transpose(image, (2, 0, 1))
    
    return {
        "image": transposed,
        "original_shape": image.shape,
        "new_shape": transposed.shape
    }


@workflow_node("add_batch_dimension", isolation_mode="none", log_level="SILENT", skip_metadata=True)
def add_batch_dimension_node(image: np.ndarray) -> dict:
    """
    Add batch dimension to image.
    
    Args:
        image: Input image (C, H, W)
        
    Returns:
        dict with batched image (1, C, H, W)
    """
    batched = np.expand_dims(image, axis=0)
    
    return {
        "image": batched,
        "original_shape": image.shape,
        "batch_shape": batched.shape
    }


@workflow_node("bgr_to_rgb", isolation_mode="auto")
def bgr_to_rgb_node(image: np.ndarray) -> dict:
    """
    Convert image from BGR to RGB color space.
    
    Args:
        image: Input image in BGR format
        
    Returns:
        dict with RGB image
    """
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return {
        "image": rgb_image
    }


@workflow_node("get_image_shape", isolation_mode="auto")
def get_image_shape_node(image: np.ndarray) -> dict:
    """
    Extract image dimensions.
    
    Args:
        image: Input image
        
    Returns:
        dict with shape information
    """
    shape = image.shape
    
    result = {
        "shape": shape,
        "ndim": image.ndim
    }
    
    if image.ndim == 3:
        result["height"] = shape[0]
        result["width"] = shape[1]
        result["channels"] = shape[2]
    elif image.ndim == 4:
        result["batch"] = shape[0]
        result["channels"] = shape[1]
        result["height"] = shape[2]
        result["width"] = shape[3]
    
    return result


@workflow_node("save_image", isolation_mode="auto")
def save_image_node(image: np.ndarray, output_path: str) -> dict:
    """
    Save image to file.
    
    Args:
        image: Image to save
        output_path: Output file path
        
    Returns:
        dict with save status and path
    """
    import os
    
    # Create parent directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    success = cv2.imwrite(output_path, image)
    
    if not success:
        raise IOError(f"Failed to save image to {output_path}")
    
    return {
        "success": True,
        "path": output_path
    }
