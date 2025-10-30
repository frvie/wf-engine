"""
OpenVINO Model Loader Node

Loads ONNX models with OpenVINO provider for NPU acceleration.
Supports Intel NPU, CPU, and GPU devices.
"""

import os
from src.core.decorator import workflow_node


@workflow_node("load_openvino_model",
               dependencies=["openvino", "numpy"],
               isolation_mode="none")
def load_openvino_model_node(model_path: str, device: str = "NPU",
                             session_namespace: str = "npu"):
    """Load ONNX model with OpenVINO provider (NPU/CPU/GPU support)"""
    try:
        import openvino as ov
        import logging
        logger = logging.getLogger('workflow.model_loader.openvino')
        
        # Convert to absolute path if not already
        if not os.path.isabs(model_path):
            model_path = os.path.abspath(model_path)
        
        if not os.path.exists(model_path):
            logger.warning(f"Model not found: {model_path}")
            return {"error": f"Model not found: {model_path}", "skipped": True}
        
        # Initialize OpenVINO core
        core = ov.Core()
        
        # Check if device is available
        available_devices = core.available_devices
        logger.info(f"Available OpenVINO devices: {available_devices}")
        
        if device not in available_devices:
            logger.warning(f"Requested device '{device}' not available. Skipping.")
            return {"error": f"Device '{device}' not available.", "skipped": True, "available_devices": available_devices}
        
        # Read model
        model = core.read_model(model=model_path)
        
        # Set static shape for NPU (required - NPU doesn't support dynamic shapes)
        if device == 'NPU':
            model.reshape({"images": [1, 3, 640, 640]})
        
        # Compile model for target device
        try:
            compiled_model = core.compile_model(model=model, device_name=device)
        except Exception as compile_error:
            logger.warning(f"Failed to compile model for {device}: {compile_error}")
            return {"error": f"Failed to compile model for {device}: {str(compile_error)}", "skipped": True}
        
        # Get input/output info
        input_layer = compiled_model.input(0)
        output_layer = compiled_model.output(0)
        
        # Cache the compiled model (cannot be pickled, must use global cache)
        from src.core.decorator import _GLOBAL_CACHE
        cache_key = f"openvino_compiled_model_{device.lower()}"
        _GLOBAL_CACHE[cache_key] = compiled_model
        
        model_info = {
            "model_path": model_path,
            "provider": "OpenVINO",
            "device": device,
            "input_shape": list(input_layer.shape),
            "output_names": ["output0"],
            "loaded": True,
            "session_namespace": session_namespace,
            "input_layer": str(input_layer),
            "output_layer": str(output_layer),
            "cache_key": cache_key  # For retrieving from cache
        }
        
        return {
            "model_session": f"openvino_session_{device.lower()}",
            "model_info": model_info,
            "provider_type": "OpenVINO",
            "cache_key": cache_key  # Pass cache key instead of compiled_model
        }
    except Exception as e:
        logger = logging.getLogger('workflow.model_loader.openvino')
        logger.warning(f"Failed to load OpenVINO model: {e}")
        return {"error": f"Failed to load OpenVINO model: {str(e)}", "skipped": True}


@workflow_node("run_openvino_inference",
               dependencies=["openvino", "numpy"],
               isolation_mode="none")
def run_openvino_inference_node(
    cache_key: str,
    input_tensor,
    iterations: int = 5,
    warmup_iterations: int = 1
):
    """
    Run OpenVINO inference benchmark
    
    Args:
        cache_key: Cache key for retrieving the compiled model from global cache
        input_tensor: Input tensor (numpy array) in NCHW format
        iterations: Number of iterations to benchmark
        warmup_iterations: Number of warmup iterations
        
    Returns:
        dict with inference results and timing
    """
    import numpy as np
    import time
    import logging
    from src.core.decorator import _GLOBAL_CACHE
    
    logger = logging.getLogger("workflow.inference.openvino_benchmark")
    
    # Retrieve compiled model from cache
    compiled_model = _GLOBAL_CACHE.get(cache_key)
    if compiled_model is None:
        error_msg = f"Compiled model not found in cache with key: {cache_key}"
        logger.error(error_msg)
        return {"error": error_msg, "skipped": True}
    
    logger.debug(f"Retrieved OpenVINO compiled model from cache: {cache_key}")
    
    # Warmup
    for _ in range(warmup_iterations):
        result = compiled_model([input_tensor])
    
    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = compiled_model([input_tensor])
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # Convert to ms
    
    # Get output
    output = result[0]
    
    # Calculate statistics
    avg_time_ms = np.mean(times)
    min_time_ms = np.min(times)
    max_time_ms = np.max(times)
    fps = 1000.0 / avg_time_ms if avg_time_ms > 0 else 0
    
    # Quick detection count from output (YOLOv8 format: [1, 84, 8400])
    # Each detection: [x, y, w, h, class_scores...]
    # Filter by confidence > 0.25 to get detection count
    try:
        if output.ndim == 3 and output.shape[0] == 1:
            # YOLOv8 output format: transpose to [8400, 84]
            predictions = output[0].T  # Shape: [8400, 84]
            # Extract box confidence (max class score from columns 4:84)
            if predictions.shape[1] >= 84:
                class_scores = predictions[:, 4:]  # Shape: [8400, 80]
                max_scores = np.max(class_scores, axis=1)  # Shape: [8400]
                num_detections = np.sum(max_scores > 0.25)  # Count detections above threshold
            else:
                num_detections = 0
        else:
            num_detections = 0
    except Exception as e:
        logger.debug(f"Could not extract detection count: {e}")
        num_detections = 0
    
    logger.info(f"OpenVINO Benchmark: {avg_time_ms:.1f}ms avg ({fps:.1f} FPS), {num_detections} detections")
    
    return {
        "outputs": [output],  # Wrap in list to match YOLO decode_yolo_v8_output_node expectation
        "inference_time_ms": avg_time_ms,
        "min_time_ms": min_time_ms,
        "max_time_ms": max_time_ms,
        "total_time_ms": sum(times),
        "fps": fps,
        "iterations": iterations,
        "times_ms": times,
        "detections": num_detections  # Add detection count
    }


