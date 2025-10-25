"""
DirectML Model Loader Node

Loads ONNX models with DirectML provider.
Requires isolation due to conflict with onnxruntime-gpu.
"""

from typing import Dict
from workflow_decorator import workflow_node


@workflow_node("load_directml_model", 
               dependencies=["onnxruntime-directml"], 
               isolation_mode="auto",
               environment="directml-env")
def load_directml_model_node(model_path: str, device_id: int = 0, 
                            session_namespace: str = "directml"):
    """Load ONNX model with DirectML provider (conflicts with onnxruntime-gpu)"""
    try:
        # Simulate DirectML model loading
        # In real implementation:
        # import onnxruntime as ort
        # providers = [('DmlExecutionProvider', {'device_id': device_id})]
        # session = ort.InferenceSession(model_path, providers=providers)
        
        model_info = {
            "model_path": model_path,
            "provider": "DirectML",
            "device_id": device_id,
            "input_shape": [1, 3, 640, 640],
            "output_names": ["output0"],
            "loaded": True,
            "session_namespace": session_namespace
        }
        
        return {
            "model_session": f"directml_session_{device_id}",
            "model_info": model_info,
            "provider_type": "DirectML"
        }
    except Exception as e:
        return {"error": f"Failed to load DirectML model: {str(e)}"}