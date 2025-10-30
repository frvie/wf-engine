"""
GPU Detection Node

Detects discrete GPUs in the system and determines optimal device IDs for DirectML/CUDA.
Uses py3nvml for simple, reliable NVIDIA GPU detection.
"""

import logging
from typing import Dict
from src.core.decorator import workflow_node


@workflow_node("detect_gpus",
               dependencies=["py3nvml"],
               isolation_mode="none")
def detect_gpus_node() -> Dict:
    """
    Detect discrete GPUs and return optimal device configuration.
    
    Uses py3nvml for NVIDIA GPU detection.
    
    Returns:
        Dictionary containing:
        - all_gpus: List of all detected GPUs
        - discrete_gpus: List of discrete GPU info (filtered)
        - directml_device_id: Recommended device ID for DirectML
        - cuda_device_id: Recommended device ID for CUDA
        - cuda_available: Whether NVIDIA CUDA is available
        - gpu_count: Total number of GPUs detected
    """
    logger = logging.getLogger('workflow.gpu_detection')
    
    try:
        from py3nvml import py3nvml
        
        # Initialize NVML
        py3nvml.nvmlInit()
        device_count = py3nvml.nvmlDeviceGetCount()
        
        if device_count == 0:
            logger.warning("No NVIDIA GPUs detected")
            py3nvml.nvmlShutdown()
            return {
                'gpu_count': 0,
                'all_gpus': [],
                'discrete_gpus': [],
                'directml_device_id': 0,
                'cuda_device_id': 0,
                'cuda_available': False,
                'recommended_gpu': None,
                'has_openvino_npu': False,
                'npu_devices': [],
                'has_directml': False,
                'has_cuda': False
            }
        
        logger.info(f"Detected {device_count} NVIDIA GPU(s)")
        
        all_gpus = []
        discrete_gpus = []
        
        for i in range(device_count):
            handle = py3nvml.nvmlDeviceGetHandleByIndex(i)
            name = py3nvml.nvmlDeviceGetName(handle)
            mem_info = py3nvml.nvmlDeviceGetMemoryInfo(handle)
            
            # Convert bytes to GB
            memory_total_gb = mem_info.total / (1024**3)
            memory_free_gb = mem_info.free / (1024**3)
            memory_used_gb = mem_info.used / (1024**3)
            
            try:
                # Get additional info
                utilization = py3nvml.nvmlDeviceGetUtilizationRates(handle)
                temperature = py3nvml.nvmlDeviceGetTemperature(handle, py3nvml.NVML_TEMPERATURE_GPU)
            except Exception:
                utilization = None
                temperature = None
            
            gpu_info = {
                'id': i,
                'name': name,
                'memory_total_gb': round(memory_total_gb, 2),
                'memory_free_gb': round(memory_free_gb, 2),
                'memory_used_gb': round(memory_used_gb, 2),
                'gpu_utilization': utilization.gpu if utilization else None,
                'memory_utilization': utilization.memory if utilization else None,
                'temperature': temperature
            }
            
            # Consider discrete if it has > 2GB VRAM
            is_discrete = gpu_info['memory_total_gb'] > 2.0
            gpu_info['is_discrete'] = is_discrete
            
            all_gpus.append(gpu_info)
            if is_discrete:
                discrete_gpus.append(gpu_info)
            
            logger.info(f"  GPU {i}: {name} ({gpu_info['memory_total_gb']} GB) - "
                       f"{'Discrete' if is_discrete else 'Integrated'}")
        
        py3nvml.nvmlShutdown()
        
        # Sort discrete GPUs by memory (descending) to get most powerful
        if discrete_gpus:
            discrete_gpus.sort(key=lambda x: x['memory_total_gb'], reverse=True)
            best_gpu = discrete_gpus[0]
            cuda_device_id = best_gpu['id']
            
            # DirectML device mapping (Windows-specific behavior)
            # DirectML enumerates ALL GPUs (including integrated), while CUDA/NVML only shows NVIDIA GPUs
            # On systems with integrated + discrete NVIDIA GPUs:
            #   CUDA/NVML: Device 0 = RTX 5090 (discrete), Device 1 = (not visible, Intel iGPU)
            #   DirectML:  Device 0 = Intel iGPU, Device 1 = RTX 5090 (discrete)
            # So we add +1 offset for DirectML to account for the integrated GPU at device 0
            directml_device_id = cuda_device_id + 1
            
            logger.info(f"Selected GPU: {best_gpu['name']} ({best_gpu['memory_total_gb']} GB)")
            logger.info(f"Device mapping: CUDA device {cuda_device_id} -> DirectML device {directml_device_id}")
        else:
            # No discrete GPU, use first available
            cuda_device_id = 0
            directml_device_id = 0
            logger.warning("No discrete GPU detected, using device 0")
        
        # Check for OpenVINO NPU
        has_npu = False
        npu_devices = []
        try:
            import openvino as ov
            core = ov.Core()
            available_devices = core.available_devices
            
            for device in available_devices:
                if 'NPU' in device:
                    has_npu = True
                    npu_devices.append(device)
                    logger.info(f"  NPU detected: {device}")
        except ImportError:
            logger.debug("OpenVINO not available, NPU detection skipped")
        except Exception as e:
            logger.debug(f"NPU detection failed: {e}")
        
        return {
            'gpu_count': len(all_gpus),
            'all_gpus': all_gpus,
            'discrete_gpus': discrete_gpus,
            'directml_device_id': directml_device_id,
            'cuda_device_id': cuda_device_id,
            'cuda_available': device_count > 0,
            'recommended_gpu': discrete_gpus[0] if discrete_gpus else all_gpus[0] if all_gpus else None,
            'has_openvino_npu': has_npu,
            'npu_devices': npu_devices,
            'has_directml': len(all_gpus) > 0,
            'has_cuda': device_count > 0
        }
        
    except ImportError as e:
        logger.warning(f"py3nvml not available: {e}, using default device IDs")
        return {
            'gpu_count': 0,
            'all_gpus': [],
            'discrete_gpus': [],
            'directml_device_id': 1,  # Default to 1 (common for discrete GPU)
            'cuda_device_id': 0,
            'cuda_available': False,
            'recommended_gpu': None,
            'has_openvino_npu': False,
            'npu_devices': [],
            'has_directml': False,
            'has_cuda': False,
            'error': f'py3nvml not available: {str(e)}'
        }
    except Exception as e:
        logger.error(f"GPU detection failed: {e}", exc_info=True)
        return {
            'gpu_count': 0,
            'all_gpus': [],
            'discrete_gpus': [],
            'directml_device_id': 1,  # Default to 1 (common for discrete GPU)
            'cuda_device_id': 0,
            'cuda_available': False,
            'recommended_gpu': None,
            'has_openvino_npu': False,
            'npu_devices': [],
            'has_directml': False,
            'has_cuda': False,
            'error': str(e)
        }



