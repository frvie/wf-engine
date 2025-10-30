# Granular Node Architecture Design

## Problem
Current nodes are **monolithic** - they do too much in a single function:
- `directml_inference_node`: 120+ lines doing model loading, preprocessing, inference, postprocessing, formatting
- `cpu_inference_node`: Similar - all-in-one approach
- Hard for AI to compose correctly
- Not reusable across different tasks

## Solution: Atomic, Composable Nodes

Break down into **single-responsibility nodes** (5-20 lines each):

---

## Node Categories

### 1. **Image Operations** (Pure functions, no ML)
```
├── read_image              # Load image from file → numpy array
├── resize_image            # Resize to target dimensions
├── normalize_image         # Normalize to 0-1 or -1 to 1
├── letterbox_image         # Add padding to maintain aspect ratio
├── bgr_to_rgb              # Color space conversion
├── hwc_to_chw              # Transpose dimensions (H,W,C → C,H,W)
├── add_batch_dimension     # Add batch dimension [C,H,W] → [1,C,H,W]
└── image_to_tensor         # Convert numpy → tensor format
```

### 2. **Model Session Management** (Backend-specific)
```
├── create_onnx_session     # Create ONNX Runtime session (CPU)
├── create_directml_session # Create ONNX Runtime session (DirectML)
├── create_cuda_session     # Create ONNX Runtime session (CUDA)
├── create_openvino_session # Create OpenVINO compiled model
├── get_model_info          # Extract input/output shapes, names
└── warmup_model            # Run warmup inference
```

### 3. **Inference Execution** (Backend-agnostic)
```
├── run_onnx_inference      # Execute ONNX session.run()
├── run_openvino_inference  # Execute OpenVINO compiled model
├── benchmark_inference     # Run N iterations, measure time
└── batch_inference         # Run inference on multiple inputs
```

### 4. **Postprocessing** (Model-specific)
```
├── decode_yolo_output      # Parse YOLO raw output → boxes + scores
├── apply_nms               # Non-maximum suppression
├── filter_by_confidence    # Filter detections by threshold
├── scale_boxes             # Scale boxes to original image size
└── clip_boxes              # Clip boxes to image bounds
```

### 5. **Formatting & Visualization**
```
├── format_detections       # Convert to standard format (bbox, class, conf)
├── get_class_names         # Map class IDs → names (COCO, ImageNet, etc.)
├── draw_boxes              # Draw bounding boxes on image
├── draw_labels             # Draw class labels + confidence
└── create_detection_summary # Generate text summary of detections
```

### 6. **Performance & Metrics**
```
├── calculate_fps           # Convert ms → FPS
├── calculate_latency_stats # Min, max, avg, p95, p99
├── compare_backends        # Side-by-side performance comparison
└── generate_performance_report # Create detailed report
```

### 7. **Utilities**
```
├── validate_image_path     # Check if file exists
├── get_image_dimensions    # Extract width, height
├── save_image              # Write image to disk
├── log_metrics             # Log performance metrics
└── cache_data              # Store data in shared cache
```

---

## Example: Granular DirectML Inference Workflow

**Before (Monolithic):**
```json
{
  "nodes": [
    {
      "id": "inference",
      "function": "workflow_nodes.directml_inference_node.directml_inference_node",
      "inputs": {
        "model_info": {...},
        "confidence_threshold": 0.25,
        "iterations": 100
      }
    }
  ]
}
```

**After (Granular & Composable):**
```json
{
  "nodes": [
    // 1. Image Loading & Preprocessing
    {
      "id": "read_img",
      "function": "workflow_nodes.image.read_image",
      "inputs": {"path": "input/soccer.jpg"},
      "dependencies": []
    },
    {
      "id": "resize_img",
      "function": "workflow_nodes.image.letterbox_image",
      "inputs": {"target_size": [640, 640]},
      "dependencies": ["read_img"]
    },
    {
      "id": "normalize",
      "function": "workflow_nodes.image.normalize_image",
      "inputs": {"range": [0, 1]},
      "dependencies": ["resize_img"]
    },
    {
      "id": "to_tensor",
      "function": "workflow_nodes.image.hwc_to_chw_batch",
      "inputs": {},
      "dependencies": ["normalize"]
    },
    
    // 2. Model Loading
    {
      "id": "create_session",
      "function": "workflow_nodes.onnx.create_directml_session",
      "inputs": {
        "model_path": "models/yolov8s.onnx",
        "device_id": 0
      },
      "dependencies": []
    },
    {
      "id": "warmup",
      "function": "workflow_nodes.onnx.warmup_model",
      "inputs": {"iterations": 3},
      "dependencies": ["create_session", "to_tensor"]
    },
    
    // 3. Inference
    {
      "id": "run_inference",
      "function": "workflow_nodes.onnx.benchmark_inference",
      "inputs": {"iterations": 100},
      "dependencies": ["warmup"]
    },
    
    // 4. Postprocessing
    {
      "id": "decode_yolo",
      "function": "workflow_nodes.yolo.decode_yolo_output",
      "inputs": {},
      "dependencies": ["run_inference"]
    },
    {
      "id": "apply_nms",
      "function": "workflow_nodes.yolo.apply_nms",
      "inputs": {"iou_threshold": 0.45},
      "dependencies": ["decode_yolo"]
    },
    {
      "id": "filter_conf",
      "function": "workflow_nodes.yolo.filter_by_confidence",
      "inputs": {"conf_threshold": 0.25},
      "dependencies": ["apply_nms"]
    },
    {
      "id": "scale_boxes",
      "function": "workflow_nodes.yolo.scale_boxes",
      "inputs": {},
      "dependencies": ["filter_conf", "read_img"]
    },
    
    // 5. Formatting
    {
      "id": "format",
      "function": "workflow_nodes.format.format_detections",
      "inputs": {"dataset": "coco"},
      "dependencies": ["scale_boxes"]
    },
    {
      "id": "summary",
      "function": "workflow_nodes.format.create_detection_summary",
      "inputs": {},
      "dependencies": ["format", "run_inference"]
    }
  ]
}
```

---

## Benefits

### 1. **Composability**
- ✅ Mix and match nodes for different tasks
- ✅ Reuse preprocessing across all backends
- ✅ Swap backends without changing workflow logic

### 2. **Testability**
- ✅ Each node is 5-20 lines → easy to test
- ✅ Single responsibility → predictable behavior
- ✅ Pure functions → no side effects

### 3. **AI-Friendly**
- ✅ Small, focused nodes are easier for AI to understand
- ✅ Clear inputs/outputs make composition obvious
- ✅ Node names describe exact functionality

### 4. **Maintainability**
- ✅ Bug fixes are localized to small nodes
- ✅ Adding features = adding new nodes
- ✅ Easy to understand what each node does

### 5. **Performance**
- ✅ Can optimize individual nodes without affecting others
- ✅ Parallel execution still works (wave-based)
- ✅ Can cache intermediate results easily

---

## Implementation Plan

### Phase 1: Core Image Operations (5 nodes)
- `read_image`
- `resize_image`
- `normalize_image`
- `hwc_to_chw`
- `add_batch_dimension`

### Phase 2: ONNX Session Management (4 nodes)
- `create_onnx_session`
- `create_directml_session`
- `get_model_info`
- `warmup_model`

### Phase 3: Inference Execution (2 nodes)
- `run_onnx_inference`
- `benchmark_inference`

### Phase 4: YOLO Postprocessing (5 nodes)
- `decode_yolo_output`
- `apply_nms`
- `filter_by_confidence`
- `scale_boxes`
- `format_detections`

### Phase 5: Update Workflow Builder
- Update `AVAILABLE_NODES` in `workflow_builder.py`
- Create templates using granular nodes
- Add composition patterns to system prompt

---

## Node Design Principles

### ✅ DO:
- Keep nodes 5-20 lines of code
- Single responsibility (one thing well)
- Pure functions when possible (input → output, no side effects)
- Clear, descriptive names
- Type hints for inputs/outputs
- Minimal dependencies

### ❌ DON'T:
- Combine multiple operations in one node
- Add business logic to utility nodes
- Use global state (except intentional caching)
- Make nodes backend-specific unless necessary
- Duplicate code across nodes

---

## Example Node Template

```python
"""
Resize Image Node

Resizes image to target dimensions using letterbox padding.
"""

import cv2
import numpy as np
from typing import Tuple
from workflow_decorator import workflow_node


@workflow_node("resize_image", isolation_mode="auto")
def resize_image_node(
    image: np.ndarray,
    target_size: Tuple[int, int] = (640, 640),
    padding_color: Tuple[int, int, int] = (114, 114, 114)
) -> dict:
    """
    Resize image with letterbox padding to maintain aspect ratio.
    
    Args:
        image: Input image (H, W, C)
        target_size: Target size (width, height)
        padding_color: RGB color for padding
        
    Returns:
        dict with resized_image, scale, padding
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # Calculate scale to fit image in target size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Add letterbox padding
    padded = np.full((target_h, target_w, 3), padding_color, dtype=np.uint8)
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
    
    return {
        "image": padded,
        "scale": scale,
        "padding": (pad_x, pad_y),
        "original_size": (w, h),
        "new_size": (new_w, new_h)
    }
```

---

## Migration Strategy

1. ✅ Create granular nodes in parallel with existing ones
2. ✅ Keep old nodes for backward compatibility
3. ✅ Update workflow builder to use new nodes
4. ✅ Create example workflows demonstrating composition
5. ✅ Gradually deprecate monolithic nodes

---

**Next Steps:**
1. Implement Phase 1 (Image Operations)
2. Test with simple workflow
3. Implement Phase 2-4 (Model, Inference, Postprocessing)
4. Update workflow builder
5. Create comprehensive examples
