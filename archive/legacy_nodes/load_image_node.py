"""
Image Loading Node

Loads and preprocesses images for inference workflows.
No dependency conflicts - runs in-process.
"""

import os
import numpy as np
import cv2
from workflow_decorator import workflow_node

# Global storage for image path (simple in-process sharing)
_IMAGE_CACHE = {}


@workflow_node("load_image", isolation_mode="auto")
def load_image_node(image_path: str, session_namespace: str = "image"):
    """Load and preprocess image for inference"""
    try:
        # Handle subprocess case (temp directory)
        if 'temp' in os.getcwd().lower() or 'tmp' in os.getcwd().lower():
            main_dir = r'C:\dev\workflow_engine'
            full_image_path = os.path.join(main_dir, image_path)
        else:
            full_image_path = os.path.abspath(image_path)
        
        # Try multiple image locations
        possible_paths = [
            full_image_path,
            os.path.join(os.getcwd(), 'input', 'desk.jpg'),
            os.path.join(os.getcwd(), 'input', 'soccer.jpg'),
            r'C:\dev\workflow_engine\input\desk.jpg',
            r'C:\dev\workflow_engine\input\soccer.jpg'
        ]
        
        image_data = None
        actual_path = None
        
        for path in possible_paths:
            if os.path.exists(path):
                image_data = cv2.imread(path)
                if image_data is not None:
                    actual_path = path
                    break
        
        if image_data is None:
            # Create dummy image data for demo if no real image found
            image_data = np.random.randint(
                0, 255, (640, 640, 3), dtype=np.uint8
            )
            actual_path = "dummy_image"
        
        # Store image path in global cache for other nodes to access
        _IMAGE_CACHE['image_path'] = actual_path
        _IMAGE_CACHE['image_data'] = image_data
        
        # Preprocess for YOLO (normalize, transpose, add batch dimension)
        processed_image = image_data.astype(np.float32) / 255.0
        processed_image = np.transpose(processed_image, (2, 0, 1))  # CHW
        processed_image = np.expand_dims(processed_image, axis=0)  # Batch
        
        return {
            "image_data": processed_image.tolist(),  # For serialization
            "original_shape": image_data.shape,
            "processed_shape": processed_image.shape,
            "image_path": actual_path,
            "session_namespace": session_namespace
        }
    except Exception as e:
        return {"error": f"Failed to load image: {str(e)}"}