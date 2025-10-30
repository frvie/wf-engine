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
                           model_path: str = None,
                           confidence_threshold: float = 0.25,
                           iou_threshold: float = 0.45,
                           iterations: int = 10,
                           gpu_info: Dict = None):
    """Run inference using DirectML (isolated from GPU ONNX)"""
    from inference_engine import SimpleOnnxEngine, get_coco_class_name
    from workflow_nodes.load_image_node import _IMAGE_CACHE
    
    logger = logging.getLogger('workflow.inference.directml')
    
    # Get model path (handle both model_info dict and direct model_path)
    if model_info:
        final_model_path = model_info.get('model_path', 'models/yolov8s.onnx')
    elif model_path:
        final_model_path = model_path
    else:
        final_model_path = 'models/yolov8s.onnx'
    
    if 'temp' in os.getcwd().lower() or not os.path.isabs(final_model_path):
        # Get the project root directory (where this script is located)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        final_model_path = os.path.join(project_root, 'models', 'yolov8s.onnx')
    
    # Detect DirectML GPU availability
    try:
        import onnxruntime as ort
        available_providers = ort.get_available_providers()
        if 'DmlExecutionProvider' not in available_providers:
            logger.warning("DirectML GPU not available. Skipping inference.")
            return {"error": "DirectML GPU not available.", "skipped": True, "available_providers": available_providers}
    except Exception as e:
        logger.warning(f"Could not check DirectML provider: {e}")
        return {"error": f"Could not check DirectML provider: {e}", "skipped": True}

    device_id = gpu_info.get('directml_device_id', 0) if gpu_info else 0
    gpu_name = gpu_info.get('recommended_gpu', {}).get('name', 'Unknown') if gpu_info else 'Unknown'
    logger.info(f"Using DirectML device {device_id}: {gpu_name}")

    # Create DirectML engine
    engine = SimpleOnnxEngine(
        model_path=final_model_path,
        providers=[('DmlExecutionProvider', {'device_id': device_id}), 'CPUExecutionProvider']
    )
    logger.info(f"Using providers: {engine.session.get_providers()}")
    
    # Get test image
    test_image = _IMAGE_CACHE.get('image_path')
    if not test_image or test_image == "dummy_image":
        # Get the project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        test_images = [
            os.path.join(project_root, 'input', 'soccer.jpg'),
            'input/soccer.jpg'
        ]
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
    
    # Apply NMS to remove duplicate detections
    if detections:
        boxes = [d['bbox'] for d in detections]
        confidences = [d['confidence'] for d in detections]
        class_ids = [d['class_id'] for d in detections]
        
        boxes, confidences, class_ids = apply_nms(boxes, confidences, class_ids, iou_threshold)
        
        # Rebuild detections after NMS
        detections = []
        for bbox, conf, cls_id in zip(boxes, confidences, class_ids):
            detections.append({
                'bbox': bbox,
                'confidence': conf,
                'class_id': cls_id
            })
    
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
        "iou_threshold": iou_threshold,
        "image_size": (orig_w, orig_h),
        "test_image": test_image,
        "top_detections": [{"class": d['class'], "confidence": d['confidence']} for d in top_3]
    }


def apply_nms(boxes, scores, class_ids, iou_threshold):
    """Non-Maximum Suppression to remove duplicate detections"""
    import numpy as np
    
    if len(boxes) == 0:
        return [], [], []
    
    boxes = np.array(boxes)
    scores = np.array(scores)
    class_ids = np.array(class_ids)
    
    order = scores.argsort()[::-1]
    keep = []
    
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        
        if len(order) == 1:
            break
        
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h
        
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_order = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        union = area_i + area_order - intersection
        
        iou = intersection / (union + 1e-6)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return boxes[keep].tolist(), scores[keep].tolist(), class_ids[keep].tolist()

