# Legacy Archive

This directory contains legacy workflows and nodes that have been superseded by the modern atomic/granular node system but are kept for reference.

## Archive Structure

### `legacy_workflows/`
Legacy monolithic workflows that have been replaced by granular workflows:

- **`video_detection*.json`** - Monolithic video detection workflows
  - Replaced by: `granular_video_detection.json` and `granular_video_detection_mp4.json`
  - Uses: Monolithic `realtime_video_detection_node`
  
- **`parallel_yolov8.json`** - Legacy multi-backend benchmark
  - Replaced by: `granular_parallel_inference.json`
  - Uses: Monolithic inference nodes

- **`test_granular_image_pipeline.json`** - Early granular testing workflow
  - Replaced by: Production granular workflows

### `legacy_nodes/`
Legacy monolithic nodes replaced by atomic nodes:

#### Inference Nodes (Monolithic)
- **`cpu_inference_node.py`** - Monolithic CPU inference
  - Replaced by: Atomic nodes in `workflow_nodes/atomic/onnx_ops.py` and `yolo_ops.py`
  
- **`directml_inference_node.py`** - Monolithic DirectML inference  
  - Replaced by: Atomic nodes with `create_onnx_directml_session_node` + pipeline
  
- **`npu_inference_node.py`** - Monolithic NPU inference
  - Note: Still has a copy for compatibility, but prefer atomic approach
  
- **`cuda_inference_node.py`** - Monolithic CUDA inference
  - Replaced by: Atomic CUDA session + pipeline nodes
  
- **`batch_inference_node.py`** - Batch processing node
  - Replaced by: Atomic pipeline approach

#### Video Nodes (Legacy)
- **`realtime_video_detection_node.py`** - Monolithic video detection
  - Replaced by: `granular_video_loop_node.py` using atomic pipeline
  
- **`video_stream_node.py`** - Legacy video streaming
  - Replaced by: Atomic `open_video_capture_node` and related nodes

#### Model Loaders (Legacy)
- **`load_cpu_model_node.py`** → Replaced by `atomic/onnx_ops.create_onnx_cpu_session_node`
- **`load_directml_model_node.py`** → Replaced by `atomic/onnx_ops.create_onnx_directml_session_node`
- **`load_cuda_model_node.py`** → Replaced by atomic CUDA nodes
- **`load_image_node.py`** → Replaced by `atomic/image_ops.read_image_node`

#### Other
- **`custom_node_template.py`** - Template examples (still useful reference)

## Why Archive?

The atomic/granular node system provides:

1. **Composability** - Mix and match operations
2. **Reusability** - Use same preprocessing/postprocessing across backends
3. **Testability** - Test individual operations in isolation
4. **Performance** - SILENT mode for high-frequency operations (18.7 FPS vs 14 FPS)
5. **Maintainability** - Single source of truth for each operation

## Migration Guide

### From Legacy Video Detection → Granular Video Detection

**Before (Monolithic):**
```json
{
  "id": "detection",
  "function": "workflow_nodes.realtime_video_detection_node.realtime_video_detection_node",
  "inputs": {
    "source": "video.mp4",
    "model_path": "models/yolov8s.onnx",
    "conf_threshold": 0.25
  }
}
```

**After (Granular):**
```json
{
  "id": "video_capture",
  "function": "workflow_nodes.atomic.video_ops.open_video_capture_node",
  "inputs": {"source": "video.mp4"}
},
{
  "id": "session",
  "function": "workflow_nodes.atomic.onnx_ops.create_onnx_directml_session_node",
  "inputs": {"model_path": "models/yolov8s.onnx"}
},
{
  "id": "video_loop",
  "function": "workflow_nodes.granular_video_loop_node.granular_video_loop_node",
  "inputs": {"conf_threshold": 0.25},
  "dependencies": ["video_capture", "session"]
}
```

### From Legacy Inference → Atomic Pipeline

**Before (Monolithic):**
- Single node does: load → preprocess → infer → postprocess → format

**After (Atomic):**
1. `read_image_node` - Load image
2. `resize_image_letterbox_node` - Resize with letterbox
3. `normalize_image_node` - Normalize pixels
4. `hwc_to_chw_node` - Transpose dimensions
5. `add_batch_dimension_node` - Add batch dim
6. `run_onnx_inference_single_node` - Run inference (SILENT)
7. `decode_yolo_v8_output_node` - Decode output (SILENT)
8. `filter_by_confidence_node` - Filter boxes (SILENT)
9. `convert_cxcywh_to_xyxy_node` - Convert format (SILENT)
10. `apply_nms_node` - Non-max suppression (SILENT)
11. `scale_boxes_to_original_node` - Scale to original (SILENT)
12. `format_detections_coco_node` - Format output (SILENT)

Note: SILENT mode eliminates logging overhead for 18.7 FPS vs 14 FPS baseline.

## When to Use Legacy Nodes

Generally, **prefer the atomic/granular system**. However, legacy nodes may be useful for:

- Quick prototyping (less verbose)
- Backwards compatibility with existing workflows
- Reference implementation for new nodes
- Comparing performance (monolithic baseline)

## See Also

- `../workflows/` - Current granular workflows
- `../workflow_nodes/atomic/` - Atomic node system
- `../GRANULAR_NODES_DESIGN.md` - Design documentation
- `../GRANULAR_NODES_IMPLEMENTATION.md` - Implementation details
