"""
OpenVINO Model Loader Node

Loads ONNX models with OpenVINO provider for NPU acceleration.
Supports Intel NPU, CPU, and GPU devices.
"""

import os
from typing import Dict
from workflow_decorator import workflow_node


@workflow_node("load_openvino_model",
               dependencies=["openvino"],
               isolation_mode="in_process")
def load_openvino_model_node(model_path: str, device: str = "NPU",
                             session_namespace: str = "npu"):
    """Load ONNX model with OpenVINO provider (NPU/CPU/GPU support)"""
    try:
        import openvino as ov
        
        # Convert to absolute path if not already
        if not os.path.isabs(model_path):
            model_path = os.path.abspath(model_path)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Initialize OpenVINO core
        core = ov.Core()
        
        # Read model
        model = core.read_model(model=model_path)
        
        # Set static shape for NPU (required - NPU doesn't support dynamic shapes)
        if device == 'NPU':
            model.reshape({"images": [1, 3, 640, 640]})
        
        # Compile model for target device
        compiled_model = core.compile_model(model=model, device_name=device)
        
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
        return {"error": f"Failed to load OpenVINO model: {str(e)}"}