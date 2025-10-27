"""
DirectML Inference Node

Runs YOLO inference using DirectML provider.
Requires isolation due to conflict with onnxruntime-gpu.
"""

import os
import time
import logging
from typing import Dict
from workflow_decorator import workflow_node


@workflow_node("directml_inference",
               dependencies=["onnxruntime-directml", "numpy", "opencv-python", "Pillow"],
               isolation_mode="subprocess")
def directml_inference_node(model_info: Dict = None, 
                           confidence_threshold: float = 0.25,
                           iterations: int = 10,
                           gpu_info: Dict = None):
    """Run inference using DirectML (isolated from GPU ONNX)"""
    from inference_engine import SimpleOnnxEngine, get_coco_class_name
    from workflow_nodes.load_image_node import _IMAGE_CACHE
    
    logger = logging.getLogger('workflow.inference.directml')
    
    # Get model path (handle subprocess temp directory)
    model_path = model_info.get('model_path', 'models/yolov8s.onnx')
    if 'temp' in os.getcwd().lower() or not os.path.isabs(model_path):
        model_path = os.path.join(r'C:\dev\workflow_engine', 'models', 'yolov8s.onnx')
    
    # Get device ID from GPU detection or default to 0
    device_id = gpu_info.get('directml_device_id', 0) if gpu_info else 0
    gpu_name = gpu_info.get('recommended_gpu', {}).get('name', 'Unknown') if gpu_info else 'Unknown'
    logger.info(f"Using DirectML device {device_id}: {gpu_name}")
    
    # Create DirectML engine
    engine = SimpleOnnxEngine(
        model_path=model_path,
        providers=[('DmlExecutionProvider', {'device_id': device_id}), 'CPUExecutionProvider']
    )
    logger.info(f"Using providers: {engine.session.get_providers()}")
    
    # Get test image
    test_image = _IMAGE_CACHE.get('image_path')
    if not test_image or test_image == "dummy_image":
        test_images = [r'C:\dev\workflow_engine\input\soccer.jpg', 'input/soccer.jpg']
        test_image = next((p for p in test_images if os.path.exists(p)), None)
        if not test_image:
            return {"error": "No test image found"}
    
    # Preprocess and warmup
    image_tensor, orig_w, orig_h = engine.preprocess_image(test_image)
    logger.info("Running warmup...")
    engine.session.run(None, {engine.input_name: image_tensor})
    
    # Run inference
    start_time = time.perf_counter()
    result = engine.run_inference(image_tensor, iterations)
    total_time = (time.perf_counter() - start_time) * 1000
    
    # Post-process
    detections = engine.postprocess_yolo(result['outputs'], orig_w, orig_h, confidence_threshold)
    formatted = [
        {
            "bbox": d['bbox'],
            "confidence": d['confidence'],
            "class": get_coco_class_name(d['class_id']),
            "class_id": d['class_id']
        }
        for d in detections
    ]
    
    # Calculate metrics
    avg_time_ms = sum(result['inference_times']) / len(result['inference_times']) * 1000
    fps = 1000.0 / avg_time_ms if avg_time_ms > 0 else 0
    
    # Log results
    logger.info(f"Detected {len(formatted)} objects")
    logger.info(f"Performance: {avg_time_ms:.1f}ms avg, {fps:.1f} FPS ({iterations} iterations)")
    
    top_3 = sorted(formatted, key=lambda x: x['confidence'], reverse=True)[:3]
    for i, det in enumerate(top_3, 1):
        logger.info(f"  {i}. {det['class']} (conf: {det['confidence']:.3f})")
        print(f"  {i}. {det['class']} (conf: {det['confidence']:.3f})")
    
    return {
        "detections": formatted,
        "inference_time_ms": avg_time_ms,
        "fps": fps,
        "total_time_ms": total_time,
        "iterations": iterations,
        "provider": engine.provider,
        "confidence_threshold": confidence_threshold,
        "image_size": (orig_w, orig_h),
        "test_image": test_image,
        "top_detections": [{"class": d['class'], "confidence": d['confidence']} for d in top_3]
    }
