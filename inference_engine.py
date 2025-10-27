"""
Simplified ONNX Inference Engine for Function-Based Workflow

A lightweight inference engine for real ONNX model execution.
"""

import numpy as np
import cv2
import time
import onnxruntime as ort
from typing import Dict, Any, List, Tuple
import logging


class SimpleOnnxEngine:
    """Simplified ONNX inference engine for function-based workflows"""
    
    def __init__(self, model_path: str, providers: List[str] = None):
        """
        Initialize ONNX inference engine
        
        Args:
            model_path: Path to ONNX model file
            providers: List of execution providers (e.g., ['DmlExecutionProvider', 'CPUExecutionProvider'])
        """
        if providers is None:
            providers = ['CPUExecutionProvider']
            
        self.logger = logging.getLogger('inference.onnx')
        
        try:
            # Enable graph optimizations for DirectML
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Create ONNX Runtime session
            self.session = ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=providers
            )
            
            # Get input details
            self.input_name = self.session.get_inputs()[0].name
            input_shape = self.session.get_inputs()[0].shape
            
            # Handle dynamic shapes - use 640 as default for YOLOv8
            self.input_height = 640
            self.input_width = 640
            
            if len(input_shape) > 2 and input_shape[2] is not None and isinstance(input_shape[2], int):
                self.input_height = input_shape[2]
            if len(input_shape) > 3 and input_shape[3] is not None and isinstance(input_shape[3], int):
                self.input_width = input_shape[3]
            
            # Get provider info
            self.provider = self.session.get_providers()[0]
            self.logger.info(f"ONNX Engine initialized with {self.provider}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ONNX session: {e}")
            raise
    
    def preprocess_image(self, image_path: str) -> Tuple[np.ndarray, int, int]:
        """
        Preprocess image for ONNX inference
        
        Args:
            image_path: Path to input image
            
        Returns:
            Tuple of (preprocessed_tensor, original_width, original_height)
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        original_height, original_width = image.shape[:2]
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size - ensure integers
        target_size = (int(self.input_width), int(self.input_height))
        image_resized = cv2.resize(image_rgb, target_size)
        
        # Normalize to [0, 1]
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # Transpose to CHW format and add batch dimension
        image_tensor = np.transpose(image_normalized, (2, 0, 1))
        image_tensor = np.expand_dims(image_tensor, axis=0)
        
        return image_tensor, original_width, original_height
    
    def run_inference(self, image_tensor: np.ndarray, num_iterations: int = 1) -> Dict[str, Any]:
        """
        Run inference iterations
        
        Args:
            image_tensor: Preprocessed image tensor
            num_iterations: Number of iterations to run
            
        Returns:
            Dictionary with inference results and timing
        """
        inference_times = []
        outputs = None
        
        for i in range(num_iterations):
            start_time = time.perf_counter()
            
            # Run inference
            outputs = self.session.run(
                None,
                {self.input_name: image_tensor}
            )
            
            iter_time = time.perf_counter() - start_time
            inference_times.append(iter_time)
        
        return {
            'outputs': outputs,
            'inference_times': inference_times,
            'num_iterations': num_iterations
        }
    
    def postprocess_yolo(
        self,
        outputs: List[np.ndarray],
        original_width: int,
        original_height: int,
        confidence_threshold: float = 0.25
    ) -> List[Dict[str, Any]]:
        """
        Post-process YOLOv8 outputs to extract detections
        
        Args:
            outputs: Raw model outputs
            original_width: Original image width
            original_height: Original image height
            confidence_threshold: Confidence threshold for detections
            
        Returns:
            List of detection dictionaries
        """
        if not outputs or len(outputs) == 0:
            return []
        
        # YOLOv8 output format: [batch, 84, 8400]
        predictions = outputs[0]
        
        # Transpose to [batch, 8400, 84]
        predictions = np.transpose(predictions, (0, 2, 1))
        
        detections = []
        
        # Process first batch
        for detection in predictions[0]:
            # Get bbox coordinates (first 4 values)
            x_center, y_center, width, height = detection[:4]
            
            # Get class scores (remaining 80 values for COCO classes)
            class_scores = detection[4:]
            
            # Get max confidence and class
            confidence = float(np.max(class_scores))
            
            if confidence >= confidence_threshold:
                class_id = int(np.argmax(class_scores))
                
                # Convert from normalized coordinates to pixel coordinates
                scale_x = original_width / self.input_width
                scale_y = original_height / self.input_height
                
                x_min = (x_center - width / 2) * scale_x
                y_min = (y_center - height / 2) * scale_y
                x_max = (x_center + width / 2) * scale_x
                y_max = (y_center + height / 2) * scale_y
                
                detections.append({
                    'class_id': class_id,
                    'confidence': confidence,
                    'bbox': [
                        float(max(0, x_min)),
                        float(max(0, y_min)),
                        float(min(original_width, x_max)),
                        float(min(original_height, y_max))
                    ]
                })
        
        return detections


def get_coco_class_name(class_id: int) -> str:
    """Get COCO class name from class ID"""
    coco_classes = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
        'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]
    
    if 0 <= class_id < len(coco_classes):
        return coco_classes[class_id]
    return f'class_{class_id}'