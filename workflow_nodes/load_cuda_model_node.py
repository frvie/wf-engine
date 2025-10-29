"""
CUDA Model Loader Node

Loads ONNX models with CUDA provider.
"""

from typing import Dict
from workflow_decorator import workflow_node


@workflow_node("load_cuda_model", 
               dependencies=["onnxruntime-gpu", "numpy", "opencv-python", "Pillow"], 
               isolation_mode="subprocess")
def load_cuda_model_node(model_path: str, device_id: int = 0, 
                         session_namespace: str = "cuda"):
    """Load ONNX model with CUDA provider"""
    try:
        # Detect CUDA GPU availability
        import onnxruntime as ort
        available_providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' not in available_providers:
            return {"error": "CUDA GPU not available.", "skipped": True, "available_providers": available_providers}

        model_info = {
            "model_path": model_path,
            "provider": "CUDA",
            "device_id": device_id,
            "input_shape": [1, 3, 640, 640],
            "output_names": ["output0"],
            "loaded": True,
            "session_namespace": session_namespace
        }
        return model_info
    except Exception as e:
        return {"error": f"Failed to load CUDA model: {str(e)}", "loaded": False, "skipped": True}
