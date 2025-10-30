"""
Atomic Workflow Nodes

Granular, composable nodes for building workflows.
Each node does ONE thing and does it well.

Import all atomic nodes for easy discovery.
"""

# Image operations
from .image_ops import (
    read_image_node,
    resize_image_letterbox_node,
    normalize_image_node,
    hwc_to_chw_node,
    add_batch_dimension_node,
    bgr_to_rgb_node,
    get_image_shape_node,
    save_image_node
)

# ONNX operations
from .onnx_ops import (
    create_onnx_cpu_session_node,
    create_onnx_directml_session_node,
    create_onnx_cuda_session_node,
    run_onnx_inference_single_node,
    run_onnx_inference_benchmark_node,
    get_onnx_model_info_node
)

# YOLO postprocessing
from .yolo_ops import (
    decode_yolo_v8_output_node,
    filter_by_confidence_node,
    convert_cxcywh_to_xyxy_node,
    apply_nms_node,
    scale_boxes_to_original_node,
    format_detections_coco_node,
    create_detection_summary_node
)

# Video operations
from .video_ops import (
    open_video_capture_node,
    read_video_frame_node,
    display_frame_node
)

__all__ = [
    # Image operations
    'read_image_node',
    'resize_image_letterbox_node',
    'normalize_image_node',
    'hwc_to_chw_node',
    'add_batch_dimension_node',
    'bgr_to_rgb_node',
    'get_image_shape_node',
    'save_image_node',
    
    # ONNX operations
    'create_onnx_cpu_session_node',
    'create_onnx_directml_session_node',
    'create_onnx_cuda_session_node',
    'run_onnx_inference_single_node',
    'run_onnx_inference_benchmark_node',
    'get_onnx_model_info_node',
    
    # YOLO postprocessing
    'decode_yolo_v8_output_node',
    'filter_by_confidence_node',
    'convert_cxcywh_to_xyxy_node',
    'apply_nms_node',
    'scale_boxes_to_original_node',
    'format_detections_coco_node',
    'create_detection_summary_node',
    
    # Video operations
    'open_video_capture_node',
    'read_video_frame_node',
    'display_frame_node',
]
