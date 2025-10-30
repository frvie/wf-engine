"""
Atomic ONNX Runtime Nodes

Small, reusable nodes for ONNX model operations.
Backend-specific but focused on single responsibilities.
"""

import time
import numpy as np
from typing import Dict, List, Optional, Any
from workflow_decorator import workflow_node


@workflow_node("create_onnx_cpu_session", isolation_mode="auto")
def create_onnx_cpu_session_node(model_path: str) -> dict:
    """
    Create ONNX Runtime session with CPU provider.
    
    Args:
        model_path: Path to ONNX model file
        
    Returns:
        dict with session info
    """
    import onnxruntime as ort
    
    session = ort.InferenceSession(
        model_path,
        providers=['CPUExecutionProvider']
    )
    
    # Get input/output metadata
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    output_names = [output.name for output in session.get_outputs()]
    
    return {
        "session": session,
        "model_path": model_path,
        "provider": "CPU",
        "input_name": input_name,
        "input_shape": input_shape,
        "output_names": output_names
    }


@workflow_node("create_onnx_directml_session", 
               dependencies=["onnxruntime-directml", "opencv-python", "numpy"],
               isolation_mode="none")
def create_onnx_directml_session_node(
    model_path: str,
    device_id: int = 0
) -> dict:
    """
    Create ONNX Runtime session with DirectML provider.
    
    Args:
        model_path: Path to ONNX model file
        device_id: DirectML device ID
        
    Returns:
        dict with session info
    """
    import onnxruntime as ort
    import os
    
    # Handle subprocess path resolution
    if not os.path.isabs(model_path):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_path = os.path.join(project_root, model_path)
    
    session = ort.InferenceSession(
        model_path,
        providers=[
            ('DmlExecutionProvider', {'device_id': device_id}),
            'CPUExecutionProvider'
        ]
    )
    
    # Get input/output metadata
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    output_names = [output.name for output in session.get_outputs()]
    
    return {
        "session": session,
        "model_path": model_path,
        "provider": "DirectML",
        "device_id": device_id,
        "input_name": input_name,
        "input_shape": input_shape,
        "output_names": output_names,
        "actual_providers": session.get_providers()
    }


@workflow_node("create_onnx_cuda_session", isolation_mode="auto")
def create_onnx_cuda_session_node(
    model_path: str,
    device_id: int = 0
) -> dict:
    """
    Create ONNX Runtime session with CUDA provider.
    
    Args:
        model_path: Path to ONNX model file
        device_id: CUDA device ID
        
    Returns:
        dict with session info
    """
    import onnxruntime as ort
    
    session = ort.InferenceSession(
        model_path,
        providers=[
            ('CUDAExecutionProvider', {'device_id': device_id}),
            'CPUExecutionProvider'
        ]
    )
    
    # Get input/output metadata
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    output_names = [output.name for output in session.get_outputs()]
    
    return {
        "session": session,
        "model_path": model_path,
        "provider": "CUDA",
        "device_id": device_id,
        "input_name": input_name,
        "input_shape": input_shape,
        "output_names": output_names
    }


@workflow_node("run_onnx_inference_single", isolation_mode="none", log_level="SILENT", skip_metadata=True)
def run_onnx_inference_single_node(
    session: Any,
    input_name: str,
    image: np.ndarray
) -> dict:
    """
    Run single inference on ONNX session.
    
    Args:
        session: ONNX Runtime session
        input_name: Name of input tensor
        image: Input tensor
        
    Returns:
        dict with raw outputs
    """
    outputs = session.run(None, {input_name: image})
    
    return {
        "outputs": outputs,
        "num_outputs": len(outputs)
    }


@workflow_node("run_onnx_inference_benchmark", isolation_mode="auto")
def run_onnx_inference_benchmark_node(
    session: Any,
    input_name: str,
    image: np.ndarray,
    iterations: int = 100,
    warmup_iterations: int = 3
) -> dict:
    """
    Benchmark ONNX inference with multiple iterations.
    
    Args:
        session: ONNX Runtime session
        input_name: Name of input tensor
        image: Input tensor
        iterations: Number of benchmark iterations
        warmup_iterations: Number of warmup iterations
        
    Returns:
        dict with outputs, timing stats, and FPS
    """
    # Warmup
    for _ in range(warmup_iterations):
        session.run(None, {input_name: image})
    
    # Benchmark
    inference_times = []
    for _ in range(iterations):
        start_time = time.perf_counter()
        outputs = session.run(None, {input_name: image})
        end_time = time.perf_counter()
        inference_times.append(end_time - start_time)
    
    # Calculate stats
    avg_time_s = sum(inference_times) / len(inference_times)
    avg_time_ms = avg_time_s * 1000
    min_time_ms = min(inference_times) * 1000
    max_time_ms = max(inference_times) * 1000
    fps = 1.0 / avg_time_s if avg_time_s > 0 else 0
    
    return {
        "outputs": outputs,  # Last inference output
        "iterations": iterations,
        "avg_time_ms": avg_time_ms,
        "inference_time_ms": avg_time_ms,  # Alias for compatibility with summary nodes
        "min_time_ms": min_time_ms,
        "max_time_ms": max_time_ms,
        "fps": fps,
        "all_times_ms": [t * 1000 for t in inference_times]
    }


@workflow_node("get_onnx_model_info", isolation_mode="auto")
def get_onnx_model_info_node(session: Any) -> dict:
    """
    Extract metadata from ONNX session.
    
    Args:
        session: ONNX Runtime session
        
    Returns:
        dict with model metadata
    """
    inputs_info = []
    for inp in session.get_inputs():
        inputs_info.append({
            "name": inp.name,
            "shape": inp.shape,
            "type": str(inp.type)
        })
    
    outputs_info = []
    for out in session.get_outputs():
        outputs_info.append({
            "name": out.name,
            "shape": out.shape,
            "type": str(out.type)
        })
    
    return {
        "providers": session.get_providers(),
        "inputs": inputs_info,
        "outputs": outputs_info,
        "num_inputs": len(inputs_info),
        "num_outputs": len(outputs_info)
    }
