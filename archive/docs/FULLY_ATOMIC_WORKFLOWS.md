# Fully Atomic Granular Workflows 🧩

## Overview

The workflow agent can now generate **TRUE atomic workflows** where each node performs exactly **ONE atomic operation**, matching the design pattern of `granular_parallel_inference.json`.

## Atomic Design Principles

1. **Atomicity**: Each node does exactly ONE operation (read, resize, normalize, etc.)
2. **Composability**: Nodes can be reused across different workflows
3. **Reusability**: Same nodes work for different pipelines
4. **Minimal Overhead**: Workflow engine handles shared memory communication

## Atomic vs Regular Workflows

### Regular Mode (4 nodes)
```
detect_hardware → download_model → process_video → performance_stats
```
Uses wrapper nodes that hide internal operations.

### Atomic Mode (15-17 nodes for images, 6 for videos)

**IMAGE Processing (16 atomic nodes):**
```
Infrastructure (2):
  detect_hardware → download_model

Preprocessing (5):
  read_img → resize → normalize → transpose → add_batch

Session (1):
  create_session (DirectML or CPU)

Inference (1):
  inference (benchmark with timing)

Post-processing (6):
  decode → filter → convert_boxes → nms → scale_boxes → format

Summary (1):
  summary (detection statistics)
```

**VIDEO Processing (6 nodes):**
```
detect_hardware → download_model → video_capture → 
create_session → video_loop → performance_stats
```
Video uses `granular_video_loop_node` which internally orchestrates atomic operations per frame (efficient for frame iteration).

## Usage

### Trigger Atomic Mode

Set `quality_over_speed=True` in your WorkflowGoal:

```python
from workflow_agent import AgenticWorkflowSystem, WorkflowGoal

system = AgenticWorkflowSystem()

goal = WorkflowGoal(
    task='object_detection',
    input_type='input/test.jpg',  # IMAGE input
    output_type='display',
    performance_target=20.0,
    hardware_preference='auto',
    quality_over_speed=True  # ← Triggers TRUE atomic mode
)

workflow = system.create_workflow_from_goal(goal)
```

### Generated Workflow Structure

The generated workflow will have:

```json
{
  "workflow": {
    "name": "Atomic Image: object_detection",
    "strategy": "fully_atomic",
    "node_breakdown": {
      "preprocessing": 5,
      "session": 1,
      "inference": 1,
      "postprocessing": 6,
      "summary": 1,
      "infrastructure": 2,
      "total": 16
    }
  },
  "nodes": [
    {
      "id": "read_img",
      "function": "workflow_nodes.atomic.image_ops.read_image_node",
      "inputs": {"image_path": "input/test.jpg"},
      "dependencies": []
    },
    {
      "id": "resize",
      "function": "workflow_nodes.atomic.image_ops.resize_image_letterbox_node",
      "inputs": {"target_width": 640, "target_height": 640},
      "dependencies": ["read_img"]
    },
    // ... 14 more atomic nodes
  ]
}
```

## Comparison with Reference

### Reference: `granular_parallel_inference.json` (20 nodes)

```
Pattern:
  1 download_model
  5 preprocessing (read, resize, normalize, transpose, batch)
  3 sessions (cpu, npu, detect_gpu)
  3 inference (cpu, npu, directml) - parallel execution
  6 post-processing (decode, filter, convert, nms, scale, format)
  1 summary
  1 compare
```

### Agent-Generated Image Workflow (16 nodes)

```
Pattern:
  2 infrastructure (download, detect)
  5 preprocessing (read, resize, normalize, transpose, batch)
  1 session creation (DirectML or CPU)
  1 inference (benchmark with timing)
  6 post-processing (decode, filter, convert, nms, scale, format)
  1 summary
```

**Key Differences:**
- Reference: Multi-backend parallel execution (3 sessions + 3 inference)
- Agent: Single-backend optimized (1 session + 1 inference)
- Reference: Cross-backend comparison (compare_performance node)
- Agent: Single-backend summary (detection statistics)

**Shared Pattern:**
- ✅ Same atomic preprocessing pipeline (5 nodes)
- ✅ Same atomic post-processing pipeline (6 nodes)
- ✅ Same node functions from `workflow_nodes.atomic.*`
- ✅ Each node performs ONE operation
- ✅ Maximum composability and reuse

## Atomic Node Categories

### Image Operations (`workflow_nodes.atomic.image_ops`)
- `read_image_node` - Load image from file
- `resize_image_letterbox_node` - Resize with aspect ratio preservation
- `normalize_image_node` - Normalize pixel values (0-1 range)
- `hwc_to_chw_node` - Transpose from HWC to CHW format
- `add_batch_dimension_node` - Add batch dimension for inference

### ONNX Operations (`workflow_nodes.atomic.onnx_ops`)
- `create_onnx_cpu_session_node` - Create CPU inference session
- `create_onnx_directml_session_node` - Create DirectML GPU session
- `run_onnx_inference_benchmark_node` - Run inference with timing

### YOLO Operations (`workflow_nodes.atomic.yolo_ops`)
- `decode_yolo_v8_output_node` - Decode raw model output
- `filter_by_confidence_node` - Filter low-confidence detections
- `convert_cxcywh_to_xyxy_node` - Convert box format (center → corners)
- `apply_nms_node` - Non-maximum suppression
- `scale_boxes_to_original_node` - Scale boxes to original image size
- `format_detections_coco_node` - Format as COCO-style detections
- `create_detection_summary_node` - Generate summary statistics

## Benefits of Atomic Workflows

### 1. **Maximum Flexibility**
Each operation is a separate node, enabling fine-grained customization:
```python
# Want different normalization? Just swap the normalize node
# Want custom NMS threshold? Just modify the nms node
```

### 2. **Easy Debugging**
Each node can be tested independently:
```python
# Test just the preprocessing pipeline
nodes = ["read_img", "resize", "normalize", "transpose", "add_batch"]

# Test just the post-processing pipeline
nodes = ["decode", "filter", "convert_boxes", "nms", "scale_boxes", "format"]
```

### 3. **Composability**
Reuse nodes across different workflows:
```python
# Same preprocessing for detection, classification, segmentation
preprocessing = ["read_img", "resize", "normalize", "transpose", "add_batch"]

# Different backends, same interface
session_cpu = "create_onnx_cpu_session_node"
session_gpu = "create_onnx_directml_session_node"
session_npu = "create_openvino_npu_session_node"
```

### 4. **Parallel Execution**
Workflow engine can execute independent nodes in parallel:
```python
# These nodes can run in parallel:
# - cpu_session, npu_session, detect_gpu (session creation)
# - cpu_inference, npu_inference, directml_inference (parallel backends)
```

## Testing

Run the test script:

```bash
python test_true_atomic.py
```

Expected output:
```
=== Testing TRUE Atomic Granular Workflow Generation ===

1. IMAGE WORKFLOW (Full Atomic Breakdown)
Strategy: fully_atomic
Total Nodes: 16

Node Breakdown:
  Preprocessing: 5 nodes
  Session: 1 nodes
  Inference: 1 nodes
  Postprocessing: 6 nodes
  Summary: 1 nodes
  Infrastructure: 2 nodes

✅ ATOMICITY ACHIEVED!
```

## Generated Workflows

The agent generates two types of atomic workflows:

### Image Workflows
- **File**: `workflows/agent_atomic_image.json`
- **Nodes**: 16 (fully atomic breakdown)
- **Pattern**: Infrastructure → Preprocess → Session → Inference → Postprocess → Summary
- **Use Case**: Single image detection with maximum flexibility

### Video Workflows
- **File**: `workflows/agent_atomic_video.json`
- **Nodes**: 6 (atomic setup + efficient loop)
- **Pattern**: Infrastructure → VideoCapture → Session → Loop → Stats
- **Use Case**: Video detection with frame-by-frame atomic processing

## Implementation Details

### Code Location
- **Main Composer**: `workflow_agent.py` - `WorkflowComposer` class
- **Atomic Image Method**: `_compose_atomic_image_workflow()`
- **Atomic Video Method**: `_compose_atomic_video_workflow()`
- **Router**: `_compose_image_detection()` and `_compose_video_detection()`

### Trigger Logic
```python
if goal.quality_over_speed:
    # Use fully atomic mode
    return self._compose_atomic_image_workflow(goal, base_nodes)
else:
    # Use regular mode (wrapper nodes)
    return self._compose_regular_image_workflow(goal)
```

## Future Enhancements

1. **Multi-Backend Atomic Workflows**
   - Generate parallel sessions for CPU, GPU, NPU
   - Compare performance across backends (like granular_parallel_inference.json)

2. **MCP Tool Exposure**
   - Expose `quality_over_speed` parameter in `create_workflow_from_nl` MCP tool
   - Allow natural language to trigger atomic mode: "create atomic workflow", "breakdown steps"

3. **Per-Frame Video Atomic**
   - Currently video uses `granular_video_loop_node` (efficient)
   - Could create per-frame atomic breakdown (maximally atomic, less efficient)

4. **Custom Atomic Pipelines**
   - Allow users to specify which nodes to include
   - Mix and match preprocessing/postprocessing nodes
   - Support custom atomic node modules

## Conclusion

The TRUE atomic workflow generation matches the design pattern of `granular_parallel_inference.json`:
- ✅ Each node performs exactly ONE operation
- ✅ Maximum composability and reusability
- ✅ Minimal communication overhead (workflow engine manages shared memory)
- ✅ 15-17 atomic nodes for complete image detection pipeline
- ✅ Same atomic node functions as reference workflows

This enables maximum flexibility while maintaining the proven atomic design pattern.
