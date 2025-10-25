# 🚀 Workflow Engine

A high-performance, modular workflow orchestration engine for Python with automatic node discovery, environment isolation, and multi-backend inference support.

---
## 🎯 What is the Workflow Engine? 
High-performance, extensible workflow system for AI object detection with support for DirectML GPU, CUDA, NPU, and CPU backends. Features parallel execution, data session caching, and easy custom node creation.High-performance, extensible workflow system for AI object detection with support for DirectML GPU, CUDA, NPU, and CPU backends. Features parallel execution, data session caching, and easy custom node creation.

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

## 🎯 Performance Optimizations

### 1. Pre-loaded Functions
```python
# Slow: Import on every execution
module = importlib.import_module('workflow_nodes.cpu_inference_node')
func = getattr(module, 'cpu_inference_node')

# Fast: Pre-load once, reuse
func = discovered_nodes['workflow_nodes.cpu_inference_node.cpu_inference_node']['function']
```

### 2. Thread Pool Reuse
```python
# Executor created once per wave
with ThreadPoolExecutor(max_workers=4) as executor:
    # Submit all nodes in wave
    futures = {executor.submit(execute, node): node_id for node_id in ready_nodes}
```

### 3. In-Process Execution
```python
# Subprocess overhead: ~50-100ms startup
# In-process: ~0ms overhead

# Example: NPU inference
In-process: 17.1ms per inference
Subprocess: Would be 67-117ms per inference
```

### 4. Dependency Caching
```python
# Results stored once, reused by all dependents
self.results['load_image'] = {"image_path": "...", "image_data": [...]}

# All 3 inference nodes reuse this data
cpu_inference(**results['load_image'])
directml_inference(**results['load_image'])
npu_inference(**results['load_image'])
```

---

# How to Use

##🛠️ UV as a package manager
This project requires **[uv](https://github.com/astral-sh/uv)** as the package manager.

### Create Virtual Environment
```bash
uv sync
```
## Models
```bash
# From Ultralytics
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt -O [yolov8s.pt](http://_vscodecontentref_/2)
```
### Export to ONNX
```bash
yolo export model=models/yolov8s.pt format=onnx
```
Save the models in the folder "models"


### Run Demo Workflow

```bash
uv run python function_workflow_engine.py workflows/modular_function_based_demo.json
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

## ⚡ Parallel Execution (`execute`)

The engine executes nodes in **waves** based on dependency resolution.

### Wave-Based Execution

```python
Wave 1 (Parallel): Nodes with no dependencies
  ├─ load_image
  ├─ load_directml_model (subprocess)
  ├─ load_cpu_model
  └─ load_openvino_model
         ↓
Wave 2 (Parallel): Nodes depending only on Wave 1
  ├─ directml_inference (subprocess)
  ├─ cpu_inference
  └─ npu_inference
         ↓
Wave 3 (Sequential): Final aggregation
  └─ performance_stats
```

### Execution Code

```python
while len(completed) < len(nodes):
    # Get nodes ready to execute
    ready_nodes = _get_ready_nodes(dependency_graph, completed)
    
    # Execute in parallel using thread pool
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(_execute_function_node, node): node_id
            for node_id in ready_nodes
        }
        
        for future in as_completed(futures):
            result = future.result()
            results[node_id] = result
            completed.add(node_id)
```

### Configuration

```json
{
  "workflow": {
    "settings": {
      "max_parallel_nodes": 4
    }
  }
}
```

- **max_parallel_nodes**: Maximum concurrent node executions
- Default: 4 workers
- Adjustable based on system resources

---

## 📊 Example: Multi-Backend Workflow Execution

### Workflow Definition
```json
{
  "nodes": [
    {"id": "load_image", "depends_on": []},
    {"id": "directml_model", "depends_on": []},
    {"id": "cpu_model", "depends_on": []},
    {"id": "npu_model", "depends_on": []},
    {"id": "directml_inference", "depends_on": ["load_image", "directml_model"]},
    {"id": "cpu_inference", "depends_on": ["load_image", "cpu_model"]},
    {"id": "npu_inference", "depends_on": ["load_image", "npu_model"]},
    {"id": "performance_stats", "depends_on": ["directml_inference", "cpu_inference", "npu_inference"]}
  ]
}
```

### Execution Timeline

```
Time 0.0s → Start Workflow
            ├─ Wave 1: 4 nodes ready
            │
Time 0.9s → Wave 1 Complete
            │  ✅ load_image (0.05s in-process)
            │  ✅ directml_model (0.001s subprocess)
            │  ✅ cpu_model (0.0s in-process)
            │  ✅ npu_model (0.9s in-process)
            │
            ├─ Wave 2: 3 nodes ready
            │
Time 5.2s → Wave 2 Complete
            │  ✅ directml_inference (3.2s subprocess)
            │  ✅ cpu_inference (0.8s in-process)
            │  ✅ npu_inference (4.3s in-process)
            │
            ├─ Wave 3: 1 node ready
            │
Time 5.2s → Wave 3 Complete
            │  ✅ performance_stats (0.003s in-process)
            │
Time 5.2s → Workflow Complete
            📊 Results: 8/8 nodes executed successfully
```

### Console Output

```
19:57:22 | workflow.engine | INFO | 🚀 Starting Multi-Backend YOLO Inference (8 nodes)
19:57:22 | workflow.engine | INFO | 📦 Loaded 8/8 required nodes
19:57:22 | workflow.engine | INFO | ⚡ Executing 4 nodes...
19:57:22 | workflow.engine | INFO | ✅ load_image: completed
19:57:22 | workflow.engine | INFO | ✅ directml_model: completed
19:57:22 | workflow.engine | INFO | ✅ cpu_model: completed
19:57:23 | workflow.engine | INFO | ✅ npu_model: completed
19:57:23 | workflow.engine | INFO | ⚡ Executing 3 nodes...
19:57:27 | workflow.engine | INFO | ✅ directml_inference: completed
19:57:27 | workflow.engine | INFO | ✅ cpu_inference: completed
19:57:27 | workflow.engine | INFO | ✅ npu_inference: completed
19:57:27 | workflow.engine | INFO | ⚡ Executing 1 nodes...
19:57:27 | workflow.engine | INFO | ✅ performance_stats: completed
19:57:27 | workflow.engine | INFO | ✅ Workflow completed in 5.2s
19:57:27 | workflow.engine | INFO | 🎯 Workflow completed successfully: 8/8 nodes executed
```



### Scalability

| Nodes | Sequential | Parallel (4 workers) | Speedup |
|-------|-----------|---------------------|---------|
| 4 | 1.8s | 0.9s | 2.0x |
| 8 | 9.4s | 5.2s | 1.8x |
| 16 | 18.2s | 8.1s | 2.2x |

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

