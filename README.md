# 🚀 Workflow Engine

The Workflow Engine is a flexible task orchestration system that executes complex workflows by automatically discovering, managing, and executing modular Python nodes. It intelligently handles dependency conflicts through environment isolation and supports parallel execution with dependency resolution.

---

## 🔑 Key Design Decisions

### 1. Lazy Loading
**Problem:** Loading all nodes wastes memory and time  
**Solution:** Only load nodes required by current workflow

```python
# Before: Load all 50 nodes in workflow_nodes/
# After: Load only 8 nodes needed for this workflow
📦 Loaded 8/8 required nodes
```

### 2. Smart Isolation
**Problem:** Subprocess overhead slows everything down  
**Solution:** Only use subprocess when necessary

```python
# DirectML: Needs isolation (onnxruntime-directml conflicts)
→ Execute in subprocess (directml-env)

# NPU: No conflicts with main environment
→ Execute in-process (fastest)

# CPU: No conflicts
→ Execute in-process (fastest)
```

### 3. Parallel Waves
**Problem:** Sequential execution wastes time  
**Solution:** Execute independent nodes concurrently

```python
# Sequential (slow): 0.9s + 0.001s + 0.0s + 0.9s = 1.8s
# Parallel (fast): max(0.9s, 0.001s, 0.0s, 0.9s) = 0.9s
```

### 4. Auto-Injection
**Problem:** Manual wiring is error-prone  
**Solution:** Automatically pass dependency outputs

```python
# No need to specify:
{
  "inputs": {
    "image_path": "$load_image.image_path",
    "image_data": "$load_image.image_data"
  }
}

# Auto-injected from dependencies:
inputs = {**static_inputs, **load_image_result}
```

### 5. Error Resilience
**Problem:** One node failure shouldn't crash workflow  
**Solution:** Continue execution, mark failed nodes

```python
try:
    result = execute_node(node)
except Exception as e:
    result = {'error': str(e), 'status': 'failed'}
    # Continue to next nodes
```

---


# How to Use

### 🛠️ Prerequisites
This project requires **[uv](https://github.com/astral-sh/uv)** as the package manager.

### Clone the repository
```bash
git clone https://github.com/frvie/wf-engine.git
cd wf-engine
```

### Create Virtual Environment
```bash
uv sync
```

### 🎯 Available Workflows

#### 1. **Real-Time Video Object Detection** (Recommended)
Live webcam object detection with DirectML GPU acceleration.

```bash
python function_workflow_engine.py workflows/video_detection.json
```

**Features:**
- 🎥 Real-time webcam processing at ~28 FPS
- 🚀 DirectML GPU acceleration (~412 FPS inference)
- 🎨 Live display with bounding boxes and labels
- 📹 Interactive recording (press `S` to toggle)
- ⚡ Auto-downloads YOLOv8s model (no manual setup!)
- 🎯 Detects 80 COCO object classes

**Controls:**
- `Q` - Quit
- `S` - Start/Stop recording

**Configuration:**
Edit `workflows/video_detection.json` to customize:
- `source`: Webcam index (default: `"0"`) or video file path
- `max_duration`: Recording duration in seconds (default: `0` = infinite)
- `confidence_threshold`: Detection confidence (default: `0.25`)
- `iou_threshold`: Non-maximum suppression threshold (default: `0.45`)

---

#### 2. **Parallel Multi-Backend Inference**
Compare YOLOv8 inference across multiple backends (CPU, DirectML GPU, NPU, OpenVINO).

```bash
python function_workflow_engine.py workflows/parallel_yolov8.json
```

**Features:**
- 🔄 Parallel execution across 4 backends
- 📊 Performance comparison and statistics
- 🎯 Auto-downloads YOLOv8s model
- ⚡ Smart environment isolation (DirectML runs in subprocess)

**Output:**
```
Backend Performance Comparison:
- DirectML GPU: 3.2s (86 FPS)
- CPU: 0.8s (1,250 FPS single image)
- NPU: 4.3s (232 FPS)
- OpenVINO: 1.1s (909 FPS)
```

---

### 📦 Model Auto-Download
**No manual model download required!** Both workflows automatically download the YOLOv8s ONNX model (~42.8 MB) on first run.

Models are cached in `models/` directory:
- `yolov8s.onnx` - ONNX format for inference

If you want to manually download:
```bash
# From Ultralytics
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt -O models/yolov8s.pt

# Export to ONNX
yolo export model=models/yolov8s.pt format=onnx
```

---

# Workflow Engine Architecture

## How the Function Workflow Engine Works

The `function_workflow_engine.py` is the **core orchestrator** that executes workflows defined in JSON format. This document explains its internal architecture and execution model.

---

## 🔄 Execution Flow

```
1. Load Workflow JSON
   ↓
2. Initialize Environment Manager
   ↓
3. Discover Required Nodes
   ↓
4. Build Dependency Graph
   ↓
5. Execute Nodes in Waves (Parallel)
   ↓
6. Return Results
```

---

## 🏗️ Key Components

### 1. Initialization (`__init__`)

```python
FunctionWorkflowEngine(workflow_data)
```

**What it does:**
- Loads workflow configuration from JSON
- Initializes environment manager for handling dependency conflicts
- Discovers and loads only the nodes needed for this workflow

**Code Flow:**
```python
def __init__(self, workflow_data: Dict = None):
    self.workflow_config = workflow_data.get('workflow', {})
    self.nodes = workflow_data.get('nodes', [])
    self._initialize_environment_manager()  # Setup isolated environments
    self._discover_and_load_nodes()         # Load required node functions
```

---

### 2. Node Discovery (`_discover_and_load_nodes`)

**Purpose:** Automatically find and load workflow nodes without manual registration.

**How it works:**
1. Extracts required function names from workflow JSON
2. Scans `workflow_nodes/` directory for Python files
3. Imports modules and looks for `@workflow_node` decorated functions
4. **Only loads nodes required** by the workflow (efficient)
5. Pre-loads functions for faster execution

**Example:**
```python
# Workflow needs these functions:
required_functions = [
    "workflow_nodes.load_image_node.load_image_node",
    "workflow_nodes.cpu_inference_node.cpu_inference_node"
]

# Engine scans workflow_nodes/ and finds:
discovered_nodes = {
    "workflow_nodes.load_image_node.load_image_node": {
        'function': <function load_image_node>,
        'module': 'workflow_nodes.load_image_node',
        'file': 'load_image_node.py'
    }
}
```

**Output:**
```
🔍 Discovering nodes for 8 functions...
📦 Loaded 8/8 required nodes
```

---

### 3. Dependency Resolution (`_get_ready_nodes`)

**Purpose:** Determine which nodes can execute based on completed dependencies.

**Algorithm:**
```python
def _get_ready_nodes(dependency_graph, completed):
    ready = []
    for node_id, dependencies in dependency_graph.items():
        if node_id not in completed:
            if all(dep in completed for dep in dependencies):
                ready.append(node_id)
    return ready
```

**Example:**
```python
# Dependency graph:
{
    "load_image": [],                           # No dependencies
    "directml_model": [],                       # No dependencies
    "directml_inference": ["load_image", "directml_model"]  # Must wait
}

# Initial state (completed = {}):
ready_nodes = ["load_image", "directml_model"]  # Can execute now

# After Wave 1 (completed = {"load_image", "directml_model"}):
ready_nodes = ["directml_inference"]  # Dependencies satisfied
```

---

### 4. Input Preparation (`_prepare_inputs`)

**Purpose:** Merge static inputs with dependency results and resolve references.

**Three-step process:**

#### Step 1: Copy Static Inputs
```json
{
  "inputs": {
    "confidence_threshold": 0.25,
    "iterations": 100
  }
}
```

#### Step 2: Auto-Inject Dependency Results
```python
# Node depends on "load_image"
# load_image returned: {"image_path": "input/soccer.jpg", "image_data": [...]}

# Auto-injected into inputs:
inputs = {
    "confidence_threshold": 0.25,
    "iterations": 100,
    "image_path": "input/soccer.jpg",  # From dependency
    "image_data": [...]                 # From dependency
}
```

#### Step 3: Resolve `$` References
```json
{
  "inputs": {
    "npu_result": "$npu_inference"
  }
}
```
```python
# Resolves to:
inputs["npu_result"] = results["npu_inference"]
```

---

### 5. Node Execution (`_execute_function_node`)

**Decision Tree:**

```
┌─────────────────────────────────────┐
│   Execute Node Function             │
└─────────────────┬───────────────────┘
                  │
                  ▼
         Is function pre-loaded?
         ┌────────┴────────┐
        YES               NO
         │                 │
         ▼                 ▼
    Execute         Check if isolation
    directly          needed (env_manager)
    (fastest)        ┌─────┴──────┐
                    YES          NO
                     │            │
                     ▼            ▼
           Execute in        Dynamic import
           subprocess        and execute
           (isolated env)    (in-process)
```

**Example Execution Paths:**

**Path 1: Pre-loaded, In-Process (Fastest)**
```python
# NPU node: No conflicts, pre-loaded
func = discovered_nodes['workflow_nodes.npu_inference_node.npu_inference_node']
result = func(**inputs)  # Direct call
```

**Path 2: Subprocess Isolation**
```python
# DirectML node: Conflicts with onnxruntime, needs directml-env
env_info = environment_manager.get_environment_for_node('directml_inference_node')
result = _execute_in_environment(function_name, inputs, env_info)
```

**Path 3: Dynamic Import**
```python
# Fallback if pre-loading failed
module = importlib.import_module('workflow_nodes.cpu_inference_node')
func = getattr(module, 'cpu_inference_node')
result = func(**inputs)
```

---

### 6. Environment Isolation (`_execute_in_environment`)

**Purpose:** Execute node in completely isolated Python environment to handle dependency conflicts.

**When used:**
- DirectML node (requires `onnxruntime-directml` which conflicts with `onnxruntime`)
- Any node marked with `environment="env_name"` in decorator

**How it works:**

```
┌──────────────────────────────────────────────────────────────┐
│                  Main Process                                 │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  1. Create execution script                            │  │
│  │  2. Serialize inputs to inputs.json                    │  │
│  │  3. Spawn subprocess with isolated Python env         │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│              Isolated Subprocess                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Python Environment: directml-env                      │  │
│  │  ┌──────────────────────────────────────────────────┐  │  │
│  │  │  1. Import node function                         │  │  │
│  │  │  2. Read inputs.json                             │  │  │
│  │  │  3. Execute function                             │  │  │
│  │  │  4. Write result.json                            │  │  │
│  │  └──────────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                  Main Process                                 │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  5. Read result.json                                   │  │
│  │  6. Return result to workflow                          │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

**Generated Script Example:**
```python
# Temporary script created for isolated execution
import sys
import json

# Add current directory to path  
sys.path.insert(0, r"C:\dev\workflow_engine")

from workflow_nodes.directml_inference_node import directml_inference_node

# Read inputs
with open("inputs.json", "r") as f:
    inputs = json.load(f)

# Execute function
result = directml_inference_node(**inputs)

# Write result
with open("result.json", "w") as f:
    json.dump(result, f)
```

---

### 7. Shared Memory with Headers (Zero-Copy IPC)

**Purpose:**  
Efficient, zero-copy inter-process communication (IPC) for isolated workflow nodes (e.g., DirectML subprocesses), supporting numpy arrays, dicts, and pickled objects.

**How it works:**
- Each shared memory block starts with an 8-byte header:  
  - **Flag (4 bytes):** Synchronization state (EMPTY, READY, PROCESSING, DONE, ERROR)  
  - **Reserved (4 bytes):** For future use
- Data payload follows the header (numpy array, pickled object, etc.)
- **Synchronization:**  
  - Main process creates shared memory, writes data, sets flag to READY  
  - Subprocess waits for READY, reads data, processes, writes result, sets flag to DONE  
  - Main process waits for DONE, reads result, cleans up

**Advantages:**  
- No need for Python SyncManager or OS pipes  
- Works across different Python executables and virtual environments  
- Safe, fast, and robust for large data (e.g., images, tensors)

**Usage Example:**
```python
# Main process
shm, buf = create_shared_memory_with_header("my_shm", data_size)
set_flag(shm.buf, FLAG_EMPTY)
buf[:] = data_bytes
set_flag(shm.buf, FLAG_READY)

# Subprocess
shm, buf = attach_shared_memory_with_header("my_shm")
wait_for_flag(shm.buf, FLAG_READY)
data = np.frombuffer(buf, dtype=np.float32)
# ...process...
set_flag(shm.buf, FLAG_DONE)
```

**Supported Types:**  
- Numpy arrays (`numpy_to_shared_memory_with_header`)
- Dictionaries (`dict_to_shared_memory_with_header`)
- Pickled objects (`pickle_to_shared_memory`)

---

## 🚀 Summary

The Function Workflow Engine achieves **high performance** and **flexibility** through:

✅ **Automatic Discovery** - No manual node registration  
✅ **Smart Isolation** - Subprocess only when needed  
✅ **Parallel Execution** - Wave-based concurrent processing  
✅ **Dependency Resolution** - Graph-based execution planning  
✅ **Auto-Injection** - Seamless data flow between nodes  
✅ **Error Resilience** - Graceful failure handling  

This architecture makes the engine **fast, flexible, and fault-tolerant** while handling complex dependency graphs and environment isolation automatically! 🚀

