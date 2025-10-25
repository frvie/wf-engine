# 🚀 Workflow Engine# 🔄 Multi-Backend AI Workflow Engine# 🔄 Multi-Backend AI Workflow Engine

A high-performance, modular workflow orchestration engine for Python with automatic node discovery, environment isolation, and multi-backend inference support.


## 🎯 What is the Workflow Engine? 
High-performance, extensible workflow system for AI object detection with support for DirectML GPU, CUDA, NPU, and CPU backends. Features parallel execution, data session caching, and easy custom node creation.High-performance, extensible workflow system for AI object detection with support for DirectML GPU, CUDA, NPU, and CPU backends. Features parallel execution, data session caching, and easy custom node creation.

The Workflow Engine is a flexible task orchestration system that executes complex workflows by automatically discovering, managing, and executing modular Python nodes. It intelligently handles dependency conflicts through environment isolation and supports parallel execution with dependency resolution.

Originally built for multi-backend AI inference (DirectML GPU, Intel NPU, CPU), the engine is designed to be **general-purpose** and can orchestrate any type of computational workflow.## 🚀 Quick Start

## ✨ Main Capabilities

- 🔍 **Automatic Node Discovery** - Drop Python files in `workflow_nodes/` and they're automatically registered```powershell## 🚀 Quick Start

- 🔀 **Parallel Execution** - Executes nodes concurrently when dependencies allow

- 🔒 **Environment Isolation** - Runs nodes in isolated Python environments to handle conflicting dependencies# Run YOLOv8 detection across all backends

- 📊 **Dependency Resolution** - Builds and executes workflows based on dependency graphs

- ⚡ **Multi-Backend Inference** - Included example: DirectML (137 FPS), Intel NPU (58 FPS), CPU (18 FPS)uv run python workflow_cli.py --workflow workflows/yolov8_object_detection.json```powershell

- 🎨 **Declarative Workflows** - Define workflows in simple JSON format

- 📦 **Automatic Dependency Management** - Installs required packages per node automatically# Run YOLOv8 detection across all backends

## 🛠️ Requirements# Visualize and compare detection resultsuv run python workflow_cli.py --workflow workflows/yolov8_object_detection.json

This project requires **[uv](https://github.com/astral-sh/uv)** as the package manager.uv run python visualize_detections.py

### Install UV# Visualize and compare detection results

**Windows (PowerShell):**# Try custom node examplesuv run python visualize_detections.py

```powershell

powershell -c "irm https://astral.sh/uv/install.ps1 | iex"uv run python custom_node_example.py

```

```# Try custom node examples

**Linux/macOS:**

```bashuv run python custom_node_example.py

curl -LsSf https://astral.sh/uv/install.sh | sh

```## ✨ Features```



## 🚀 Quick Start



### 1. Clone the Repository- 🚄 **Multi-Backend Support**: DirectML GPU, CUDA, NPU, CPU## ✨ Features

```bash

git clone https://github.com/frvie/wf-engine.git- ⚡ **Parallel Execution**: Run multiple backends simultaneously

cd wf-engine

```- 💾 **Data Session Caching**: Share models and data across nodes- 🚄 **Multi-Backend Support**: DirectML GPU, CUDA, NPU, CPU



### 2. Create Virtual Environment with UV- 🔌 **Custom Nodes**: Easy plugin system for extending functionality- ⚡ **Parallel Execution**: Run multiple backends simultaneously

```bash

uv venv- 📊 **Performance Comparison**: Built-in benchmarking across backends- 💾 **Data Session Caching**: Share models and data across nodes

```

- 🎯 **Auto-Discovery**: Automatically finds and loads workflow nodes- 🔌 **Custom Nodes**: Easy plugin system for extending functionality

### 3. Activate Environment

**Windows (PowerShell):**- 📊 **Performance Comparison**: Built-in benchmarking across backends

```powershell

.\.venv\Scripts\Activate.ps1## 📊 Performance Results (RTX 5090)- 🎯 **Auto-Discovery**: Automatically finds and loads workflow nodes

```



**Linux/macOS:**

```bash| Backend | FPS | Avg Time | Detections |winml/## 🚀 Quick Start

source .venv/bin/activate

```|---------|-----|----------|------------|



### 4. Install Dependencies| **DirectML GPU** | 76.8 | 13.01ms | 14 |├── workflow_cli.py              # Main workflow executor

```bash

uv pip install -e .| **NPU (OpenVINO)** | 45.8 | 21.84ms | 14 |

```

| **CUDA (PyTorch)** | 29.9 | 33.46ms | 15 |├── workflow_orchestrator.py     # Workflow engine with parallel execution### 1. Install Dependencies

### 5. Run Demo Workflow

```bash| **CPU (ONNX)** | 13.9 | 72.01ms | 9 |

python function_workflow_engine.py workflows/modular_function_based_demo.json

```├── workflow_loader.py           # Workflow JSON parser```powershell



## 📖 How to Use*Tested with YOLOv8s model on 640x640 images, 50 iterations*



### Basic Workflow Execution├── workflow_data_session.py     # Data caching systemuv sync

```bash

python function_workflow_engine.py path/to/workflow.json## 📁 Project Structure

```

├── framework_data_optimizer.py  # Memory optimization```

### Example Output

``````

🚀 Starting Multi-Backend YOLO Inference (DirectML + NPU + CPU)

⚡ Executing 4 nodes in parallel...winml/├── logging_config.py            # Centralized logging

✅ load_image: completed

✅ directml_model: completed├── workflow_orchestrator.py    # Main workflow engine

✅ cpu_model: completed  

✅ npu_model: completed├── workflow_cli.py              # CLI interface├── visualize_detections.py      # Detection visualization tool### 2. Run Generic Workflow



📊 Performance Comparison (3 backends tested):├── workflow_loader.py           # JSON workflow parser

  🥇 DirectML: 7.3ms (137 FPS) - 1.00x

  🥈 OpenVINO_NPU: 17.1ms (58 FPS) - 0.46x├── workflow_data_session.py     # Data session caching│```powershell

  🥉 CPU: 51.3ms (19.5 FPS) - 0.15x

├── logging_config.py            # Centralized logging

🎯 Detection Results:

  DirectML - Top 5 of 125 detections:├── framework_data_optimizer.py  # Data optimization layer├── workflow-nodes/              # Modular workflow nodesuv run python generic_workflow_engine.py

    1. person (conf: 0.917) bbox: [23, 126, 206, 466]

    2. person (conf: 0.917) bbox: [24, 125, 205, 469]├── custom_node_example.py       # Custom node tutorial

    ...

```├── visualize_detections.py      # Detection visualization│   ├── directml_model_loader_node.py```



## 🏗️ How the Workflow Engine Works│



### Architecture Overview├── workflow-nodes/              # Workflow task nodes (12 nodes)│   ├── cpu_model_loader_node.py



```│   ├── image_reader_node.py

┌─────────────────────────────────────────────────────────────┐

│                  Workflow JSON Definition                    ││   ├── *_model_loader_node.py   # Model loaders (4 backends)│   ├── npu_model_loader_node.py### 3. Run JSON-Defined Workflow

└─────────────────┬───────────────────────────────────────────┘

                  ││   ├── *_inference_node.py      # Inference nodes (4 backends)

                  ▼

┌─────────────────────────────────────────────────────────────┐│   └── performance_stats_node.py│   ├── cuda_model_loader_node.py```powershell

│            Function Workflow Engine                          │

│  • Loads workflow definition                                 ││

│  • Discovers nodes automatically                             │

│  • Builds dependency graph                                   │├── workflows/                   # Workflow definitions│   ├── gpu_inference_node.pyuv run python workflow_cli.py

│  • Manages parallel execution                                │

└─────────────────┬───────────────────────────────────────────┘│   ├── yolov8_object_detection.json  # Main YOLO workflow

                  │

      ┌───────────┴───────────┬──────────────┐│   └── custom_node_demo.json         # Custom node example│   ├── cpu_inference_node.py```

      ▼                       ▼              ▼

┌─────────────┐      ┌─────────────┐   ┌─────────────┐│

│   Node A    │      │   Node B    │   │   Node C    │

│ (in-process)│      │ (isolated)  │   │ (in-process)│├── models/                      # AI models│   ├── npu_inference_node.py

└─────────────┘      └─────┬───────┘   └─────────────┘

                           ││   ├── yolov8s.onnx            # For DirectML, CPU, NPU

                           ▼

              ┌────────────────────────┐│   └── yolov8s.pt              # For CUDA│   ├── cuda_inference_node.py## 🎯 What It Does

              │  Isolated Environment  │

              │  (subprocess execution)││

              └────────────────────────┘

```├── input/                       # Sample images│   ├── image_reader_node.py



### Execution Flow│   ├── soccer.jpg



1. **Load Workflow** - Parse JSON workflow definition│   └── desk.jpg│   └── performance_stats_node.py**Clean Architecture with Separation of Concerns**:

2. **Node Discovery** - Scan `workflow_nodes/` directory for `*_node.py` files

3. **Dependency Graph** - Build execution graph from `depends_on` relationships│

4. **Environment Setup** - Initialize isolated environments per `environments.json`

5. **Parallel Execution** - Execute nodes respecting dependencies└── shared/                      # Shared utilities│

6. **Data Passing** - Pass outputs between nodes via `$node_id` references

7. **Result Aggregation** - Collect and return final results    └── yolo_postprocessing.py



### Key Components```├── workflows/                   # Workflow definitions- **Generic Preprocessor**: Handles letterbox, center-crop, normalization for any model



- **`function_workflow_engine.py`** - Main orchestrator

- **`workflow_decorator.py`** - `@workflow_node` decorator for node registration

- **`workflow_environment_manager.py`** - Manages isolated Python environments## 🎯 Usage Examples│   └── yolov8_object_detection.json- **Inference Nodes**: Focus only on inference (CPU/GPU/NPU) with no preprocessing

- **`environments.json`** - Environment definitions and node mappings

- **`workflow_nodes/`** - Directory containing all workflow nodes



## 📝 Workflow JSON Format### Run Main Workflow│- **Generic Postprocessor**: Processes YOLO detections, classification outputs



A workflow is defined in JSON with the following structure:



```json```powershell├── models/                      # AI models- **Parallel Execution**: Each phase runs optimally (sequential preprocessing, parallel inference)

{

  "workflow": {# Execute YOLOv8 detection with all backends

    "name": "My Workflow",

    "description": "Description of what this workflow does",uv run python workflow_cli.py --workflow workflows/yolov8_object_detection.json│   ├── yolov8s.onnx            # ONNX format for DirectML/CPU/NPU

    "settings": {

      "max_parallel_nodes": 4

    }

  },# Use custom image│   └── yolov8s.pt              # PyTorch format for CUDA## 🏆 Performance Results

  

  "nodes": [uv run python workflow_cli.py --workflow workflows/yolov8_object_detection.json --image path/to/image.jpg

    {

      "id": "unique_node_id",```│

      "function": "workflow_nodes.module_name.function_name",

      "depends_on": ["other_node_id"],

      "inputs": {

        "param1": "value1",### Create Custom Nodes├── input/                       # Input imagesFrom latest generic workflow run:

        "param2": "$other_node_id"

      }

    }

  ]```python├── output/                      # Detection results & visualizations

}

```# 1. Define your custom node



### Field Descriptionsclass MyProcessorNode:└── archive/                     # Old/test files| Backend | Device | Performance | Architecture |



| Field | Description |    def execute(self, inputs):

|-------|-------------|

| `workflow.name` | Human-readable workflow name |        # Your processing logic|---------|--------|-------------|--------------|

| `workflow.description` | Workflow description |

| `workflow.settings.max_parallel_nodes` | Maximum nodes to execute concurrently |        data = inputs.get('data', '')

| `nodes[].id` | Unique identifier for the node |

| `nodes[].function` | Python path to the node function |        result = data.upper()  # Example transformation```| **GPU** | DirectML | **374.6 FPS** | ✅ Clean Separation |

| `nodes[].depends_on` | List of node IDs that must complete first |

| `nodes[].inputs` | Input parameters for the node |        return {'success': True, 'result': result}

| `$node_id` syntax | Reference to output of another node |

| **NPU** | OpenVINO | **137.6 FPS** | ✅ Focused Inference |

### Example Workflow

# 2. Register with the engine

```json

{from workflow_orchestrator import WorkflowEngine## 🎯 Key Features| **CPU** | ONNX Runtime | **104.0 FPS** | ✅ Generic Processing |

  "workflow": {

    "name": "Image Processing Pipeline",engine = WorkflowEngine()

    "settings": {

      "max_parallel_nodes": 3engine.register_custom_node('my_processor', MyProcessorNode)

    }

  },

  

  "nodes": [# 3. Use in workflow JSON### Multi-Backend Support**Total execution time: 0.47s** with clean architecture benefits

    {

      "id": "load_image",{

      "function": "workflow_nodes.load_image_node.load_image_node",

      "depends_on": [],    "nodes": [- **DirectML**: Windows GPU acceleration (AMD, Intel, NVIDIA)

      "inputs": {

        "image_path": "input/photo.jpg"        {

      }

    },            "id": "process_step",- **CUDA**: Native NVIDIA GPU acceleration## 🔧 Technical Architecture

    {

      "id": "process_image",            "type": "my_processor",

      "function": "workflow_nodes.process_node.process_node",

      "depends_on": ["load_image"],            "depends_on": ["previous_step"]- **NPU**: Intel Neural Processing Unit

      "inputs": {

        "image_data": "$load_image",        }

        "filter": "blur"

      }    ]- **CPU**: ONNX Runtime fallback### Key Benefits: Clean Separation

    },

    {}

      "id": "save_result",

      "function": "workflow_nodes.save_node.save_node",```- **Generic Nodes**: Reusable across all AI model types

      "depends_on": ["process_image"],

      "inputs": {

        "data": "$process_image",

        "output_path": "output/result.jpg"### Load Nodes from Custom Directory### Performance Optimizations- **Focused Inference**: No preprocessing logic mixed with inference

      }

    }

  ]

}```python- ✅ Plugin-based isolation (no subprocess overhead)- **Parallel Phases**: Optimal execution strategy per workflow phase

```

# Load additional nodes from custom directories

## 🔧 Creating Custom Workflow Nodes

engine = WorkflowEngine(- ✅ Parallel node execution- **JSON Workflows**: Declarative workflow definitions

### Step 1: Create Node File

    nodes_directory="workflow-nodes",

Create a new Python file in `workflow_nodes/` (must end with `_node.py`):

    custom_node_paths=["my_custom_nodes", "/another/path"]- ✅ Data session caching

**`workflow_nodes/my_custom_node.py`**

```python)

"""

My Custom Node- ✅ Shared memory optimization### Execution Phases



Description of what this node does.# List all available nodes

"""

available_nodes = engine.list_available_nodes()- ✅ Model instance caching (10-14x speedup)1. **Data Loading**: Parallel (image + model loading)

from workflow_decorator import workflow_node

print(f"Found {len(available_nodes)} nodes")

@workflow_node("my_custom_node", 

               dependencies=["numpy", "pillow"],  # Optional```2. **Preprocessing**: Sequential (shared tensor for all backends)

               isolation_mode="auto",              # Optional: "auto", "none", "subprocess"

               environment="custom-env")           # Optional: isolated environment name

def my_custom_node(input_data: str, threshold: float = 0.5):

    """## 🔧 Workflow Definition### Architecture3. **Inference**: Parallel (GPU, CPU, NPU simultaneously)

    Process data with custom logic

    

    Args:

        input_data: Input data to processWorkflows are defined in JSON format:- **Modular nodes**: Each component is a standalone node4. **Postprocessing**: Parallel (per-backend result processing)

        threshold: Processing threshold

        

    Returns:

        Dictionary with results```json- **Data sessions**: Efficient model/data sharing5. **Analysis**: Sequential (aggregate statistics)

    """

    try:{

        # Your custom logic here

        result = process_data(input_data, threshold)  "name": "YOLOv8 Object Detection",- **Parallel execution**: All backends run concurrently

        

        return {  "nodes": [

            "status": "success",

            "result": result,    {- **Timing breakdown**: Separate loading vs inference phases## 📦 Dependencies

            "metadata": {

                "threshold": threshold      "id": "load_image",

            }

        }      "type": "image_reader_node",

    except Exception as e:

        return {"error": f"Processing failed: {str(e)}"}      "parameters": {

```

        "image_path": "input/soccer.jpg"## 🔧 Configuration```toml

### Step 2: Add to Workflow

      }

Reference your node in a workflow JSON:

    },dependencies = [

```json

{    {

  "nodes": [

    {      "id": "gpu_inference",### Workflow JSON Structure    "numpy>=1.24.0,<2.0.0",

      "id": "custom_step",

      "function": "workflow_nodes.my_custom_node.my_custom_node",      "type": "gpu_inference_node",

      "depends_on": ["previous_step"],

      "inputs": {      "depends_on": ["load_image"],```json    "pillow>=10.0.0", 

        "input_data": "$previous_step",

        "threshold": 0.75      "parameters": {

      }

    }        "confidence_threshold": 0.25,{    "requests>=2.31.0",

  ]

}        "iterations": 50

```

      },  "name": "YOLOv8 Object Detection",    "onnxruntime-directml==1.20.0",  # GPU + CPU inference

### Step 3: Run Workflow

      "parallel_group": "inference"

The engine automatically discovers and loads your node:

    }  "nodes": [    "onnxruntime==1.20.0",           # CPU fallback

```bash

python function_workflow_engine.py workflows/my_workflow.json  ]

```

}    {    "openvino>=2024.6.0",            # NPU inference

### Node Decorator Parameters

```

| Parameter | Description | Default |

|-----------|-------------|---------|      "id": "load_directml_model",]

| `name` | Unique node identifier | Required |

| `dependencies` | List of Python packages required | `[]` |### Workflow Features

| `isolation_mode` | Execution mode: `"auto"`, `"none"`, `"subprocess"` | `"auto"` |

| `environment` | Name of isolated environment from `environments.json` | `None` |      "type": "directml_model_loader_node",```



### Node Function Requirements- **Dependencies**: `depends_on` specifies execution order



✅ **Must:**- **Parallel Groups**: Nodes with same `parallel_group` run concurrently      "parameters": {

- Be decorated with `@workflow_node`

- Return a dictionary (success) or dict with `"error"` key (failure)- **Parameters**: Custom parameters passed to each node

- Accept parameters matching workflow JSON inputs

- **Data Sessions**: Automatically caches models and data between nodes        "model_path": "models/yolov8s.onnx",## 🎯 Usage Examples

❌ **Don't:**

- Use global state (nodes may run in separate processes)

- Assume execution order beyond declared dependencies

## 🏗️ Architecture        "device_id": 1  // GPU 1 = RTX 5090

## 🔒 Node Isolation System



The engine supports three isolation modes to handle dependency conflicts:

### Core Components      }### Generic Workflow Engine

### 1. In-Process Execution (`isolation_mode="none"`)



**When to use:**

- No dependency conflicts with other nodes1. **Workflow Orchestrator** (`workflow_orchestrator.py`)    }```powershell

- Need maximum performance

- Simple, pure Python operations   - Main execution engine



**Example:**   - Handles node loading, dependency resolution  ],# Run clean architecture demonstration

```python

@workflow_node("simple_processor", isolation_mode="none")   - Manages parallel execution

def simple_processor(data):

    return {"processed": data.upper()}   - Auto-discovers nodes from directories  "execution_config": {uv run python generic_workflow_engine.py

```



**Behavior:**

- Runs in main Python process2. **Data Session** (`workflow_data_session.py`)    "use_data_session": true,```

- Direct memory access

- Fastest execution   - Thread-safe key-value store

- Shares dependencies with main environment

   - Namespace isolation per backend    "parallel_execution": true

### 2. Automatic Isolation (`isolation_mode="auto"`)

   - Caches models and preprocessed data

**When to use:**

- Default choice for most nodes   - Reduces redundant loading (10-14x speedup)  }### JSON Workflow Engine

- Let engine decide based on `dependencies` parameter

- Want portability across environments



**Example:**3. **Workflow Nodes** (`workflow-nodes/`)}```powershell

```python

@workflow_node("flexible_node",    - Modular, reusable components

               dependencies=["numpy"],

               isolation_mode="auto")   - Each node implements `execute(inputs)` method```# Run AI detection workflow from JSON

def flexible_node(data):

    import numpy as np   - Discoverable via file naming convention

    return {"result": np.array(data).mean()}

```   - Support for both short and full namesuv run python workflow_cli.py



**Behavior:**

- Engine checks for dependency conflicts

- Runs in-process if safe### Supported Backends## 📈 Benchmarks```

- Falls back to subprocess if conflicts detected

- Automatic dependency installation



### 3. Subprocess Isolation (`environment="env_name"`)| Backend | Framework | Hardware | Use Case |



**When to use:**|---------|-----------|----------|----------|

- Known dependency conflicts (e.g., `onnxruntime` vs `onnxruntime-directml`)

- Require specific library versions| **DirectML** | ONNX Runtime | RTX 5090 GPU | Best performance on Windows |### Timing Breakdown (50 iterations)### Workflow Management

- Need complete environment separation

| **CUDA** | PyTorch | RTX 5090 GPU | Native CUDA acceleration |

**Example:**

```python| **NPU** | OpenVINO | Intel AI Boost | Low-power AI inference |- **Loading Phase**: 2.3s (models + image)```powershell

@workflow_node("directml_inference",

               dependencies=["onnxruntime-directml"],| **CPU** | ONNX Runtime | CPU cores | Fallback option |

               isolation_mode="auto",

               environment="directml-env")- **Inference Phase**: 3.7s (parallel execution)# List available workflows

def directml_inference(model_path):

    # Runs in isolated directml-env subprocess### Execution Flow

    import onnxruntime as ort

    session = ort.InferenceSession(model_path, - **Total Workflow**: 6.0suv run python workflow_loader.py

                                   providers=['DmlExecutionProvider'])

    return {"session": "created"}```

```

1. Load Workflow JSON

**Behavior:**

- Runs in separate Python subprocess   ↓

- Uses environment defined in `environments.json`

- Complete dependency isolation2. Discover Available Nodes### Detection Quality# Load specific workflow

- Data serialized via pickle

   ↓

### Environment Configuration

3. Build Dependency Graph- DirectML & NPU: 14 objects (11 people, 2 cars, 1 ball)python -c "from workflow_loader import load_workflow; print(load_workflow('generic_ai_workflow.json'))"

Define isolated environments in `environments.json`:

   ↓

```json

{4. Load Models (Parallel)- CUDA: 15 objects (11 people, 3 cars, 1 ball)```

  "environments": {

    "directml-env": {   - DirectML Model Loader

      "path": "workflow-envs/directml-env",

      "requirements": [   - CPU Model Loader- CPU: 9 objects (7 people, 1 car, 1 ball)

        "onnxruntime-directml",

        "numpy",   - NPU Model Loader

        "opencv-python"

      ],   - CUDA Model Loader### Core Inference

      "description": "DirectML GPU acceleration environment"

    },   ↓

    "custom-env": {

      "path": "workflow-envs/custom-env",5. Run Inference (Parallel)## 🛠️ Development```powershell

      "requirements": [

        "tensorflow==2.13.0",   - GPU Inference

        "pandas"

      ]   - CPU Inference# Test GPU inference

    }

  },   - NPU Inference

  

  "node_type_mappings": {   - CUDA Inference### Adding a New Backenduv run python winml_inference.py gpu:0 1000

    "directml_inference_node": "directml-env",

    "tensorflow_node": "custom-env"   ↓

  }

}6. Collect & Compare Results1. Create model loader node: `{backend}_model_loader_node.py`

```

   - Performance Stats

### Isolation Decision Tree

   - Detection Visualization2. Create inference node: `{backend}_inference_node.py`# Test NPU inference  

```

                    Node with @workflow_node```

                            │

                            ▼3. Add to workflow JSON with appropriate namespaceuv run python winml_inference.py npu:0 1000

                    Has environment param?

                    ┌──────┴──────┐## 🔌 Custom Nodes

                   YES            NO

                    │              │4. Configure device parameters```

                    ▼              ▼

            Run in specified   isolation_mode?### Method 1: Programmatic Registration

            environment        ┌─────┴─────┐

            (subprocess)      none        auto

                              │            │

                              ▼            ▼```python

                        Run in-process  Check conflicts

                        (main env)      ┌────┴────┐class TextProcessorNode:### Custom Workflows## 🎉 Benefits

                                       YES       NO

                                        │         │    def execute(self, inputs):

                                        ▼         ▼

                                   Subprocess  In-process        text = inputs.get('text', '')Create new JSON files in `workflows/` following the schema in `yolov8_object_detection.json`.

```

        return {'processed': text.upper()}

### Cross-Process Data Sharing

✅ **Clean Architecture**: Separation of preprocessing, inference, postprocessing  

For data that can't be serialized (e.g., file paths):

engine = WorkflowEngine()

```python

# In load_image_node.pyengine.register_custom_node('text_processor', TextProcessorNode)## 📊 Visualization✅ **Generic Nodes**: Reusable across different AI model types  

from workflow_nodes.load_image_node import _IMAGE_CACHE

```

# Store data globally

_IMAGE_CACHE['image_path'] = '/path/to/image.jpg'✅ **Parallel Execution**: Optimal performance at each workflow phase  



# Access in isolated node### Method 2: File-Based Discovery

test_image = _IMAGE_CACHE.get('image_path')

``````powershell✅ **JSON Workflows**: Declarative, versionable workflow definitions  



## 📊 Current Example: Multi-Backend InferenceCreate a file in `workflow-nodes/` or custom directory:



The included workflow demonstrates real-world usage with conflicting dependencies:# Generate detection comparison images✅ **Multi-Backend**: CPU, GPU, NPU support in single environment  



**Backends:**```python

- **DirectML** (RTX 5090): 137 FPS - Isolated subprocess (onnxruntime-directml)

- **Intel NPU**: 58 FPS - In-process (OpenVINO, no conflicts)# my_custom_nodes/filter_node.pyuv run python visualize_detections.py

- **CPU**: 18 FPS - In-process (onnxruntime)

class FilterNode:

**Why Isolation Needed:**

- `onnxruntime` and `onnxruntime-directml` **cannot coexist** in same environment    def execute(self, inputs):Perfect for **production AI workflows** with clean, maintainable architecture! 🚀

- DirectML runs in isolated `directml-env` subprocess

- NPU and CPU run in-process in main environment (compatible)        data = inputs.get('data', [])



## 🎯 Use Cases        filtered = [x for x in data if x > 0]# Output files:## 🛠️ Development



- **Multi-Backend AI Inference** - Run models on different hardware accelerators        return {'success': True, 'filtered': filtered}

- **ETL Pipelines** - Extract, transform, load data with isolated environments

- **Image Processing** - Parallel processing with different libraries```# - output/detections_comparison.jpg (2x2 grid)

- **Scientific Computing** - Orchestrate complex computational workflows

- **Data Analysis** - Coordinate pandas, numpy, scikit-learn operations

- **Model Training** - Manage training across different ML frameworks

Load automatically:# - output/detections_directml.jpg### Adding New Workflow Nodes

## 📚 Project Structure



```

workflow_engine/```python# - output/detections_cuda.jpg1. Create node class in `workflow-nodes/` inheriting from `WorkflowNode`

├── function_workflow_engine.py      # Main orchestrator

├── workflow_decorator.py            # Node decorator systemengine = WorkflowEngine(custom_node_paths=["my_custom_nodes"])

├── workflow_environment_manager.py  # Environment isolation

├── inference_engine.py              # YOLO inference utilities# FilterNode is now available as 'filter' or 'filter_node'# - output/detections_npu.jpg2. Implement `execute()` method

├── environments.json                # Environment definitions

├── workflow_nodes/                  # Modular workflow nodes```

│   ├── load_image_node.py

│   ├── cpu_inference_node.py# - output/detections_cpu.jpg3. Add node type to workflow JSON definitions

│   ├── directml_inference_node.py

│   ├── npu_inference_node.py### Node Discovery

│   └── performance_stats_node.py

├── workflows/                       # Workflow definitions```

│   └── modular_function_based_demo.json

├── utilities/                       # Shared utilitiesThe engine automatically discovers nodes with both naming conventions:

│   ├── logging_config.py

│   └── shared_memory_utils.py- `my_node.py` → `my_node`### Creating New Workflows

└── input/                          # Test data

    └── soccer.jpg- `my_node_node.py` → `my_node` and `my_node_node`

```

## 🔍 Hardware Utilization1. Create JSON file in `workflows/` directory

## 🤝 Contributing

## 📦 Installation

Contributions are welcome! Feel free to:

- Add new workflow nodes2. Define nodes, dependencies, and execution groups

- Improve isolation mechanisms

- Enhance documentation```powershell

- Report issues

# Clone the repository- **GPU 0 (Intel iGPU)**: Not used (4% display only)3. Load with `workflow_loader.load_workflow()`

## 📄 License

git clone https://github.com/frvie/workflows.git

MIT License - See LICENSE file for details

cd workflows- **GPU 1 (RTX 5090)**: DirectML + CUDA

## 🙏 Acknowledgments



Built with:

- [UV](https://github.com/astral-sh/uv) - Fast Python package manager# Install dependencies with UV- **NPU (Intel AI Boost)**: OpenVINO inference### Testing Changes

- [OpenVINO](https://github.com/openvinotoolkit/openvino) - Intel NPU support

- [ONNX Runtime](https://github.com/microsoft/onnxruntime) - Multi-backend inferenceuv sync

- [DirectML](https://github.com/microsoft/DirectML) - Windows GPU acceleration

- **CPU**: ONNX Runtime fallback```powershell

---

# Or use pip

**Built for performance and extensibility** 🚀

pip install -r requirements.txt# Test workflow loading

For questions or issues, please open an issue on GitHub.

```

## 📝 Notesuv run python workflow_loader.py

### Dependencies



- **onnxruntime-directml** - DirectML GPU support

- **torch** - CUDA support- DirectML device_id=1 targets RTX 5090 (device_id=0 would use Intel iGPU)# Test generic engine

- **openvino** - NPU support

- **ultralytics** - YOLO models- NPU is separate from iGPU - dedicated AI accelerator chipuv run python generic_workflow_engine.py

- **opencv-python** - Image processing

- **numpy** - Array operations- CUDA provides 2.6x speed over DirectML due to native GPU access



See `pyproject.toml` for complete dependency list.- DirectML on RTX 5090 is 5.3x faster than on Intel iGPU# Test JSON workflows



## 🔍 Hardware Configurationuv run python workflow_cli.py



### DirectML Device Selection## 🎓 Architecture Insights



Configure GPU device in workflow JSON:The system uses a **plugin-based isolation** approach:

- Nodes are loaded dynamically

```json- Instances are cached for reuse

{- No subprocess overhead

  "id": "load_directml_model",- Thread-safe data sessions

  "type": "directml_model_loader_node",- Shared memory for large data transfers

  "parameters": {

    "device_id": 1  // 0 = iGPU, 1 = discrete GPU## ✨ Creating Custom Nodes

  }

}The workflow engine is **fully extensible** - create your own nodes easily!

```

### Method 1: Register Programmatically

### Backend Hardware Mapping

```python

- **GPU 0**: Intel Integrated GPU (not used by default)# 1. Create your node class

- **GPU 1**: RTX 5090 (DirectML + CUDA)class MyCustomNode:

- **NPU**: Intel AI Boost (separate from GPUs)    def execute(self, inputs):

- **CPU**: All available cores        # Your processing logic here

        data = inputs.get('data', '')

## 🎓 Advanced Features        result = data.upper()  # Example: convert to uppercase

        return {'success': True, 'result': result}

### Parallel Execution

# 2. Register with engine

Mark nodes with `"parallel_group": "inference"` to run concurrently:from workflow_orchestrator import WorkflowEngine

engine = WorkflowEngine()

```jsonengine.register_custom_node('my_custom_node', MyCustomNode)

{

  "nodes": [# 3. Use in your workflow JSON

    {{

      "id": "gpu_inference",    "nodes": [

      "parallel_group": "inference"        {

    },            "id": "process_step",

    {            "type": "my_custom_node",

      "id": "cpu_inference",            "depends_on": ["previous_step"]

      "parallel_group": "inference"        }

    }    ]

  ]}

}```

```

### Method 2: Load from Custom Directory

### Data Sessions

```python

Share data across nodes efficiently:# Create your node file: my_custom_nodes/text_processor_node.py

class TextProcessorNode:

```python    def execute(self, inputs):

# Nodes automatically use data session        return {'processed': True}

model_loader.execute(inputs)  # Stores model in session

inference.execute(inputs)     # Retrieves model from session# Load from custom directory

```engine = WorkflowEngine(

    nodes_directory="workflow-nodes",

### Timing Breakdown    custom_node_paths=["my_custom_nodes", "/other/path"]

)

The engine provides detailed timing:

- **Loading Phase**: Model loading time# Engine automatically discovers all .py files in these directories

- **Inference Phase**: Actual inference time```

- **Per-Node Timing**: Individual node execution times

### Method 3: Add to workflow-nodes Directory

## 📊 Visualization

```bash

Compare detections across backends:# Simply create a new file in workflow-nodes/

# my_filter_node.py

```powershell

uv run python visualize_detections.pyclass MyFilterNode:

```    def execute(self, inputs):

        # Your logic

Generates:        return {'filtered': True}

- `output/detections_comparison.jpg` - 2x2 grid comparison

- `output/detections_directml.jpg` - DirectML results# Engine will auto-discover it!

- `output/detections_cpu.jpg` - CPU results```

- `output/detections_npu.jpg` - NPU results

- `output/detections_cuda.jpg` - CUDA results### Discover Available Nodes



## 🧪 Examples```python

# List all available nodes

### Custom Node Exampleengine = WorkflowEngine()

nodes = engine.list_available_nodes()

Complete working example with 3 custom nodes:print(f"Available: {', '.join(nodes)}")



```powershell# Shows both built-in and custom nodes

uv run python custom_node_example.py```

```

### Complete Example

Includes:

- Text processing nodeSee `custom_node_example.py` for working examples:

- Image filtering node- Text processing node

- Result aggregation node- Image filter node  

- Full workflow integration- Result aggregator node

- Full workflow integration

### Main YOLO Workflow

```powershell

Multi-backend object detection:# Run the example

uv run python custom_node_example.py

```powershell```

uv run python workflow_cli.py --workflow workflows/yolov8_object_detection.json

```## 🔗 Dependencies



Features:See `pyproject.toml` for full dependency list. Key packages:

- Parallel loading across 4 backends- `onnxruntime-directml` - DirectML support

- Concurrent inference execution- `torch` - CUDA support  

- Performance comparison- `openvino` - NPU support

- Detection visualization- `ultralytics` - YOLO models

- `opencv-python` - Image processing

## 🤝 Contributing

---

Feel free to:

- Add new workflow nodes**Author**: AI Workflow Engine  

- Optimize existing backends**Last Updated**: October 22, 2025  

- Improve documentation**Version**: 3.0

- Report issues

## 📄 License

This project is provided as-is for educational and research purposes.

## 🙏 Acknowledgments

- **ONNX Runtime** for cross-platform inference
- **OpenVINO** for NPU support
- **PyTorch** for CUDA integration
- **Ultralytics** for YOLO models

---

**Built with performance and extensibility in mind** 🚀

For questions or issues, please open an issue on GitHub.
