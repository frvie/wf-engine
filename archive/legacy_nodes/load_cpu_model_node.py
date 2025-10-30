"""
CPU Model Loader Node

Loads ONNX models with CPU provider.
No dependency conflicts - runs in-process.
"""

from workflow_decorator import workflow_node


@workflow_node("load_cpu_model", isolation_mode="auto")
def load_cpu_model_node(model_path: str, session_namespace: str = "cpu"):
    """Load ONNX model with CPU provider (no conflicts)"""
    try:
        # Simulate CPU ONNX model loading
        # In real implementation:
        # import onnxruntime as ort
        # session = ort.InferenceSession(model_path, 
        #                               providers=['CPUExecutionProvider'])
        
        model_info = {
            "model_path": model_path,
            "provider": "CPU",
            "input_shape": [1, 3, 640, 640],
            "output_names": ["output0"],
            "loaded": True,
            "session_namespace": session_namespace
        }
        
        return {
            "model_session": "cpu_session",
            "model_info": model_info,
            "provider_type": "CPU"
        }
    except Exception as e:
        return {"error": f"Failed to load CPU model: {str(e)}"}