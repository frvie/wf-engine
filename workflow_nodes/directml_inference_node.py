"""
DirectML Inference Node

Runs YOLO inference using DirectML provider.
Requires isolation due to conflict with onnxruntime-gpu.
"""

import os
import time
import logging
from typing import List, Dict
from workflow_decorator import workflow_node


@workflow_node("directml_inference",
               dependencies=["onnxruntime-directml"],
               isolation_mode="auto",
               environment="directml-env")
def directml_inference_node(model_session: str = None, 
                           model_info: Dict = None, 
                           confidence_threshold: float = 0.25,
                           iterations: int = 10,
                           image_data: List = None):
    """Run inference using DirectML (must be isolated from GPU ONNX)"""
    try:
        # Import in isolation to avoid conflicts
        from inference_engine import SimpleOnnxEngine
        from workflow_nodes.load_image_node import _IMAGE_CACHE
        
        # Get model path from model_info
        model_path = model_info.get('model_path', 'models/yolov8s.onnx')
        
        # Handle subprocess case (temp directory)
        if 'temp' in os.getcwd().lower() or 'tmp' in os.getcwd().lower():
            main_dir = r'C:\dev\workflow_engine'
            model_path = os.path.join(main_dir, 'models', 'yolov8s.onnx')
        else:
            # Convert to absolute path if not already
            if not os.path.isabs(model_path):
                model_path = os.path.abspath(model_path)
        
        # Create DirectML inference engine
        engine = SimpleOnnxEngine(
            model_path=model_path,
            providers=['DmlExecutionProvider', 'CPUExecutionProvider']
        )
        
        # Get image path from global cache (set by load_image_node)
        test_image = _IMAGE_CACHE.get('image_path')
        if not test_image or test_image == "dummy_image":
            # Fallback for subprocess: use standard test image locations
            if 'temp' in os.getcwd().lower() or 'tmp' in os.getcwd().lower():
                main_dir = r'C:\dev\workflow_engine'
                test_images = [
                    os.path.join(main_dir, 'input', 'soccer.jpg'),
                    os.path.join(main_dir, 'input', 'desk.jpg')
                ]
            else:
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
        
        # Preprocess image
        image_tensor, orig_w, orig_h = engine.preprocess_image(test_image)
        
        # Run inference iterations
        start_time = time.perf_counter()
        result = engine.run_inference(image_tensor, iterations)
        end_time = time.perf_counter()
        
        # Post-process results
        detections = engine.postprocess_yolo(
            result['outputs'], 
            orig_w, 
            orig_h, 
            confidence_threshold
        )
        
        # Convert detections to simpler format
        formatted_detections = []
        for det in detections:
            from inference_engine import get_coco_class_name
            formatted_detections.append({
                "bbox": det['bbox'],
                "confidence": det['confidence'],
                "class": get_coco_class_name(det['class_id']),
                "class_id": det['class_id']
            })
        
        avg_time = sum(result['inference_times']) / len(result['inference_times'])
        avg_inference_time = avg_time * 1000
        fps = 1000.0 / avg_inference_time if avg_inference_time > 0 else 0
        
        # Log performance and results
        logger = logging.getLogger('workflow.inference.directml')
        logger.info(f"🎯 Detected {len(formatted_detections)} objects using {engine.provider}")
        logger.info(f"⚡ Performance: {avg_inference_time:.1f}ms avg, {fps:.1f} FPS ({iterations} iterations)")
        
        # Also print to stdout for subprocess visibility
        print(f"DirectML: Detected {len(formatted_detections)} objects")
        
        if formatted_detections:
            # Log top 3 detections
            top_detections = sorted(formatted_detections, 
                                  key=lambda x: x['confidence'], reverse=True)[:3]
            for i, det in enumerate(top_detections):
                logger.info(f"  {i+1}. {det['class']} (conf: {det['confidence']:.3f})")
                print(f"  {i+1}. {det['class']} (conf: {det['confidence']:.3f})")
        
        return {
            "detections": formatted_detections,
            "inference_time_ms": avg_inference_time,
            "fps": fps,
            "total_time_ms": (end_time - start_time) * 1000,
            "iterations": iterations,
            "provider": engine.provider,
            "confidence_threshold": confidence_threshold,
            "image_size": (orig_w, orig_h),
            "test_image": test_image,
            "top_detections": [{"class": d['class'], "confidence": d['confidence']} 
                             for d in top_detections] if formatted_detections else []
        }
    except Exception as e:
        return {"error": f"DirectML inference failed: {str(e)}"}