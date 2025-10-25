"""
OpenVINO NPU Inference Node

Runs YOLO inference using OpenVINO provider for NPU acceleration.
Uses same post-processing as DirectML/CPU for consistent results.
"""

import os
import time
import logging
from typing import List, Dict
from workflow_decorator import workflow_node


@workflow_node("npu_inference",
               dependencies=["openvino"],
               isolation_mode="none")
def npu_inference_node(model_session: str = None, 
                       model_info: Dict = None, 
                       confidence_threshold: float = 0.25,
                       iterations: int = 10, 
                       device: str = "NPU",
                       image_data: List = None):
    """Run inference using OpenVINO NPU/CPU/GPU"""
    try:
        # Import OpenVINO and other dependencies
        import openvino as ov
        import numpy as np
        from inference_engine import SimpleOnnxEngine
        
        # Get model path from model_info - use ONNX directly
        model_path = model_info.get('model_path', 'models/yolov8s.onnx')
        
        # Convert to absolute path if not already
        if not os.path.isabs(model_path):
            model_path = os.path.abspath(model_path)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Initialize OpenVINO
        core = ov.Core()
        
        # Check available devices
        available_devices = core.available_devices
        logger = logging.getLogger('workflow.inference.npu')
        logger.info(f"Available OpenVINO devices: {available_devices}")
        
        # Map device names  
        device_map = {"NPU": "NPU", "CPU": "CPU", "GPU": "GPU"}
        ov_device = device_map.get(device, device)
        
        # Load model directly from ONNX
        logger.info(f"Loading model for {ov_device}: {model_path}")
        model = core.read_model(model_path)
        
        # Reshape to fixed dimensions for NPU (dynamic shapes not supported)
        if ov_device == "NPU":
            logger.info("Reshaping model to fixed shape for NPU")
            # Use named input reshaping (critical for NPU)
            input_name = model.input(0).get_any_name()
            logger.info(f"Model input name: {input_name}")
            model.reshape({input_name: [1, 3, 640, 640]})
            
        # Compile model for the target device with performance optimizations
        config = {}
        if ov_device == "NPU":
            # Enable performance optimizations for NPU
            config = {
                "PERFORMANCE_HINT": "THROUGHPUT",
                "INFERENCE_PRECISION_HINT": "f16",  # NPU optimized for FP16
                "NUM_STREAMS": "AUTO"
            }
            logger.info(f"NPU config: {config}")
        
        compiled_model = core.compile_model(model, ov_device, config)
        logger.info(f"Successfully compiled model for {ov_device}")
        
        # Use SimpleOnnxEngine for consistent post-processing (always use ONNX)
        onnx_path = model_info.get('model_path', 'models/yolov8s.onnx')
        engine = SimpleOnnxEngine(onnx_path, None)
        
        # Get image from global cache (set by load_image_node)
        from workflow_nodes.load_image_node import _IMAGE_CACHE
        
        # Get image path from global cache
        test_image = _IMAGE_CACHE.get('image_path')
        if not test_image or test_image == "dummy_image":
            # Fallback for subprocess: use standard test image locations
            test_images = [
                r'C:\dev\workflow_engine\input\soccer.jpg',
                r'C:\dev\workflow_engine\input\desk.jpg',
                'input/soccer.jpg',
                'input/desk.jpg'
            ]
            test_image = None
            for path in test_images:
                if os.path.exists(path):
                    test_image = path
                    break
            if not test_image:
                raise FileNotFoundError("No test image found")
        
        # Preprocess using engine
        image_tensor, orig_w, orig_h = engine.preprocess_image(test_image)
        
        # Create infer request once (reuse for better performance)
        infer_request = compiled_model.create_infer_request()
        input_tensor = ov.Tensor(array=image_tensor)
        
        # Run inference iterations using OpenVINO
        inference_times = []
        for _ in range(iterations):
            start_time = time.perf_counter()
            
            # Set input and run inference
            infer_request.set_input_tensor(input_tensor)
            infer_request.infer()
            
            end_time = time.perf_counter()
            inference_times.append(end_time - start_time)
        
        # Get output tensor after all iterations
        output_tensor = infer_request.get_output_tensor(0)
        result_output = output_tensor.data.copy()
        
        logger.info(f"Output shape: {result_output.shape}")
        logger.info(f"Output range: [{result_output.min():.4f}, {result_output.max():.4f}]")
        
        # Post-process using OpenVINO-style direct processing
        # Output shape: (1, 84, 8400) -> take first batch -> (84, 8400)
        predictions = result_output[0]
        detections = []
        
        # Iterate over detections (transpose to get 8400 x 84)
        for pred in predictions.T:
            if len(pred) < 5:
                continue
            
            x_center, y_center, width, height = pred[:4]
            class_scores = pred[4:]
            
            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])
            
            if confidence >= confidence_threshold:
                # Scale coordinates from 640x640 to original image size
                x1 = int((x_center - width / 2) * orig_w / 640)
                y1 = int((y_center - height / 2) * orig_h / 640)
                x2 = int((x_center + width / 2) * orig_w / 640)
                y2 = int((y_center + height / 2) * orig_h / 640)
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': confidence,
                    'class_id': class_id
                })
        
        logger.info(f"Post-processing extracted {len(detections)} detections")
        
        # Calculate performance metrics
        avg_time = sum(inference_times) / len(inference_times)
        avg_inference_time = avg_time * 1000
        fps = 1000.0 / avg_inference_time if avg_inference_time > 0 else 0
        
        # Format detections with class names
        from inference_engine import get_coco_class_name
        formatted_detections = []
        for det in detections:
            formatted_detections.append({
                "bbox": det['bbox'],
                "confidence": det['confidence'],
                "class": get_coco_class_name(det['class_id']),
                "class_id": det['class_id']
            })
        
        # Log performance and results
        logger.info(f"🎯 Detected {len(formatted_detections)} objects using {ov_device}")
        logger.info(f"⚡ Performance: {avg_inference_time:.1f}ms avg, "
                   f"{fps:.1f} FPS ({iterations} iterations)")
        if formatted_detections:
            top_detections = sorted(formatted_detections,
                                  key=lambda x: x['confidence'],
                                  reverse=True)[:3]
            for i, det in enumerate(top_detections):
                logger.info(f"  {i+1}. {det['class']} "
                           f"(conf: {det['confidence']:.3f})")
        
        return {
            "detections": formatted_detections,
            "inference_time_ms": avg_inference_time,
            "fps": fps,
            "total_time_ms": sum(inference_times) * 1000,
            "iterations": iterations,
            "provider": f"OpenVINO-{ov_device}",
            "confidence_threshold": confidence_threshold,
            "image_size": (orig_w, orig_h),
            "test_image": test_image,
            "actual_device": ov_device
        }
    except Exception as e:
        return {"error": f"OpenVINO {device} inference failed: {str(e)}"}