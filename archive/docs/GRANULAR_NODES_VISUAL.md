# Granular Node Composition - Visual Guide

## Complete Inference Pipeline (14 Nodes)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    IMAGE PREPROCESSING (5 nodes)                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐           │
│  │ read_image  │───▶│    resize    │───▶│  normalize   │           │
│  └─────────────┘    │  letterbox   │    │   (0-255→    │           │
│                     └──────────────┘    │    0-1)      │           │
│                                         └──────────────┘           │
│                                               │                      │
│                                               ▼                      │
│                      ┌──────────────┐    ┌──────────────┐           │
│                      │  add_batch   │◀───│ hwc_to_chw   │           │
│                      │  dimension   │    │  transpose   │           │
│                      └──────────────┘    └──────────────┘           │
│                            │                                         │
└────────────────────────────┼─────────────────────────────────────────┘
                             │
                             │ Preprocessed Image (1, 3, 640, 640)
                             │
┌────────────────────────────┼─────────────────────────────────────────┐
│                MODEL & INFERENCE (2 nodes)                           │
├────────────────────────────┼─────────────────────────────────────────┤
│                            │                                         │
│  ┌──────────────────┐      │      ┌──────────────────┐              │
│  │  create_session  │      └─────▶│   benchmark      │              │
│  │                  │             │   inference      │              │
│  │  - CPU           │             │   (100 iters)    │              │
│  │  - DirectML      │             └──────────────────┘              │
│  │  - CUDA          │                    │                          │
│  └──────────────────┘                    │ Raw Output (1, 84, 8400) │
│                                           │                          │
└───────────────────────────────────────────┼──────────────────────────┘
                                           │
                                           │
┌───────────────────────────────────────────┼──────────────────────────┐
│                 YOLO POSTPROCESSING (7 nodes)                        │
├───────────────────────────────────────────┼──────────────────────────┤
│                                           │                          │
│  ┌──────────────┐    ┌──────────────┐    │   ┌──────────────┐       │
│  │ decode_yolo  │◀───│ Raw Output   │◀───┘   │filter_by_conf│       │
│  │ (84,8400)    │    │              │        │  (conf>0.25) │       │
│  └──────────────┘    └──────────────┘        └──────────────┘       │
│        │                                            │                │
│        │ Boxes(N,4) Scores(N) ClassIDs(N)          │                │
│        └────────────────────────────────────────────┘                │
│                            │                                         │
│                            ▼                                         │
│                   ┌──────────────────┐                               │
│                   │ convert_cxcywh   │                               │
│                   │   _to_xyxy       │                               │
│                   └──────────────────┘                               │
│                            │                                         │
│                            ▼                                         │
│                   ┌──────────────────┐                               │
│                   │   apply_nms      │                               │
│                   │  (iou<0.45)      │                               │
│                   └──────────────────┘                               │
│                            │                                         │
│                            ▼                                         │
│                   ┌──────────────────┐                               │
│                   │  scale_boxes_to  │                               │
│                   │    _original     │◀────── read_img (orig size)   │
│                   └──────────────────┘                               │
│                            │                                         │
│                            ▼                                         │
│  ┌──────────────────┐  ┌─────────────────┐  ┌──────────────────┐   │
│  │create_detection  │◀─│format_detections│◀─│  Filtered boxes  │   │
│  │    _summary      │  │     _coco       │  │  scores, classes │   │
│  └──────────────────┘  └─────────────────┘  └──────────────────┘   │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Reusability Across Backends

```
┌─────────────────────────────────────────────────────────────────┐
│              SHARED IMAGE PREPROCESSING (5 nodes)               │
│   read_image → resize → normalize → transpose → add_batch      │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          │ Same preprocessed image for all
                          │
         ┌────────────────┼────────────────┬──────────────────┐
         │                │                │                  │
         ▼                ▼                ▼                  ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ ┌──────────────────┐
│  CPU Session    │ │DirectML Session │ │ CUDA Session    │ │OpenVINO Session  │
│                 │ │  (subprocess)   │ │                 │ │                  │
│  ┌───────────┐  │ │  ┌───────────┐  │ │  ┌───────────┐  │ │  ┌────────────┐  │
│  │Benchmark  │  │ │  │Benchmark  │  │ │  │Benchmark  │  │ │  │Benchmark   │  │
│  │Inference  │  │ │  │Inference  │  │ │  │Inference  │  │ │  │Inference   │  │
│  └───────────┘  │ └  └───────────┘  │ │  └───────────┘  │ │  └────────────┘  │
└─────────────────┘ └─────────────────┘ └─────────────────┘ └──────────────────┘
         │                │                │                  │
         └────────────────┼────────────────┴──────────────────┘
                          │ All outputs
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│          SHARED YOLO POSTPROCESSING (7 nodes)                   │
│  decode → filter → convert → nms → scale → format → summary     │
└──────────────────────────────────────────────────────────────────┘
```

---

## Node Composition Examples

### Example 1: Simple Image Resize
```
Input: image_path="input/soccer.jpg"
       target_width=640, target_height=640

Pipeline:
  read_image ────▶ resize_letterbox ────▶ Output: resized image
  
Result: 
  - Original image loaded
  - Resized to 640x640 with padding
  - Aspect ratio preserved
  - Padding info returned for later use
```

### Example 2: Image Preprocessing for ONNX
```
Input: image_path="input/soccer.jpg"

Pipeline:
  read_image ─▶ resize ─▶ normalize ─▶ transpose ─▶ add_batch
  
Result:
  Shape: (640, 640, 3) → (1, 3, 640, 640)
  Values: 0-255 → 0-1
  Format: HWC → CHW with batch
```

### Example 3: YOLO Detection from Raw Output
```
Input: raw_output shape=(1, 84, 8400)
       confidence_threshold=0.25
       iou_threshold=0.45

Pipeline:
  decode_yolo ─▶ filter_conf ─▶ convert_boxes ─▶ apply_nms ─▶ 
  scale_boxes ─▶ format_coco ─▶ summary

Result:
  - 2,360 raw predictions
  - 87 above confidence threshold
  - 45 after NMS
  - Boxes scaled to original image size
  - Class names added (COCO dataset)
  - Text summary generated
```

---

## Data Flow Example

```
Node: read_image
Inputs:  {"image_path": "input/soccer.jpg"}
Outputs: {"image": [...], "width": 1280, "height": 720}
         │
         ▼
Node: resize_letterbox
Inputs:  Receives "image" automatically from read_image
         {"target_width": 640, "target_height": 640}
Outputs: {"image": [...], "scale": 0.5, "pad_x": 0, "pad_y": 40}
         │
         ▼
Node: normalize_image
Inputs:  Receives "image" automatically from resize_letterbox
         {"scale": 255.0}
Outputs: {"image": [...]}  # Now 0-1 range
         │
         ▼
Node: hwc_to_chw
Inputs:  Receives "image" automatically
Outputs: {"image": [...]}  # Now (C, H, W) format
         │
         ▼
Node: add_batch_dimension
Inputs:  Receives "image" automatically
Outputs: {"image": [...]}  # Now (1, C, H, W) format
```

**Key Point:** Dependencies handle auto-injection of outputs → inputs!

---

## Comparison: Single File vs Composition

### Monolithic Node (120+ lines):
```python
def directml_inference_node(...):
    # Load image (20 lines)
    image = cv2.imread(path)
    resized = letterbox_resize(image, 640, 640)
    normalized = resized / 255.0
    transposed = np.transpose(normalized, (2, 0, 1))
    batched = np.expand_dims(transposed, 0)
    
    # Create session (15 lines)
    session = ort.InferenceSession(model_path, providers=[...])
    
    # Run inference (20 lines)
    for i in range(iterations):
        outputs = session.run(None, {input_name: batched})
    
    # Decode YOLO (25 lines)
    predictions = outputs[0][0].T
    boxes = predictions[:, :4]
    scores = np.max(predictions[:, 4:], axis=1)
    # ... 20 more lines ...
    
    # Apply NMS (20 lines)
    # ... OpenCV NMS code ...
    
    # Format results (20 lines)
    # ... Format detections ...
    
    return {...}  # 30+ keys
```

### Granular Composition (14 nodes × 5-20 lines):
```python
# workflow_nodes/atomic/image_ops.py
def read_image_node(image_path):        # 8 lines
    return {"image": cv2.imread(image_path), ...}

def resize_image_letterbox_node(...):   # 15 lines
    return {"image": resized, "scale": scale, ...}

# workflow_nodes/atomic/onnx_ops.py
def create_onnx_cpu_session_node(...):  # 12 lines
    return {"session": session, ...}

def run_onnx_inference_benchmark_node(...):  # 18 lines
    return {"outputs": outputs, "fps": fps, ...}

# workflow_nodes/atomic/yolo_ops.py
def decode_yolo_v8_output_node(...):    # 14 lines
    return {"boxes": boxes, "scores": scores, ...}

def apply_nms_node(...):                # 20 lines
    return {"boxes": kept_boxes, ...}

# ... 8 more focused nodes ...
```

**Result:** Same functionality, but:
- ✅ Each node testable in isolation
- ✅ Reusable across backends
- ✅ AI can understand and compose
- ✅ Easy to debug individual steps

---

## Node Categories at a Glance

```
📦 workflow_nodes/atomic/
├── 🖼️  image_ops.py (8 nodes)
│   ├── Basic I/O: read, save
│   ├── Transforms: resize, normalize, transpose
│   └── Utilities: get_shape, bgr_to_rgb
│
├── 🔧 onnx_ops.py (6 nodes)
│   ├── Sessions: cpu, directml, cuda
│   ├── Inference: single, benchmark
│   └── Utilities: get_model_info
│
└── 🎯 yolo_ops.py (7 nodes)
    ├── Decoding: parse output
    ├── Filtering: confidence, NMS
    ├── Transforms: box conversion, scaling
    └── Formatting: COCO names, summaries
```

---

This granular architecture enables **infinite composition possibilities** while maintaining **simplicity and clarity**! 🚀
