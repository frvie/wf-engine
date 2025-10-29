"""
OpenVINO Model Loader Node

Loads ONNX models with OpenVINO provider for NPU acceleration.
Supports Intel NPU, CPU, and GPU devices.
"""

import os
from typing import Dict
from workflow_decorator import workflow_node


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
        
        model_info = {
            "model_path": model_path,
            "provider": "OpenVINO",
            "device": device,
            "input_shape": list(input_layer.shape),
            "output_names": ["output0"],
            "loaded": True,
            "session_namespace": session_namespace,
            "input_layer": str(input_layer),
            "output_layer": str(output_layer)
        }
        
        return {
            "model_session": f"openvino_session_{device.lower()}",
            "model_info": model_info,
            "provider_type": "OpenVINO",
            "compiled_model": compiled_model  # For direct use in inference
        }
    except Exception as e:
        logger = logging.getLogger('workflow.model_loader.openvino')
        logger.warning(f"Failed to load OpenVINO model: {e}")
        return {"error": f"Failed to load OpenVINO model: {str(e)}", "skipped": True}