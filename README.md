# 🔄 Multi-Backend AI Workflow Engine# 🔄 Multi-Backend AI Workflow Engine



High-performance, extensible workflow system for AI object detection with support for DirectML GPU, CUDA, NPU, and CPU backends. Features parallel execution, data session caching, and easy custom node creation.High-performance, extensible workflow system for AI object detection with support for DirectML GPU, CUDA, NPU, and CPU backends. Features parallel execution, data session caching, and easy custom node creation.



## 🚀 Quick Start



```powershell## 🚀 Quick Start

# Run YOLOv8 detection across all backends

uv run python workflow_cli.py --workflow workflows/yolov8_object_detection.json```powershell

# Run YOLOv8 detection across all backends

# Visualize and compare detection resultsuv run python workflow_cli.py --workflow workflows/yolov8_object_detection.json

uv run python visualize_detections.py

# Visualize and compare detection results

# Try custom node examplesuv run python visualize_detections.py

uv run python custom_node_example.py

```# Try custom node examples

uv run python custom_node_example.py

## ✨ Features```



- 🚄 **Multi-Backend Support**: DirectML GPU, CUDA, NPU, CPU## ✨ Features

- ⚡ **Parallel Execution**: Run multiple backends simultaneously

- 💾 **Data Session Caching**: Share models and data across nodes- 🚄 **Multi-Backend Support**: DirectML GPU, CUDA, NPU, CPU

- 🔌 **Custom Nodes**: Easy plugin system for extending functionality- ⚡ **Parallel Execution**: Run multiple backends simultaneously

- 📊 **Performance Comparison**: Built-in benchmarking across backends- 💾 **Data Session Caching**: Share models and data across nodes

- 🎯 **Auto-Discovery**: Automatically finds and loads workflow nodes- 🔌 **Custom Nodes**: Easy plugin system for extending functionality

- 📊 **Performance Comparison**: Built-in benchmarking across backends

## 📊 Performance Results (RTX 5090)- 🎯 **Auto-Discovery**: Automatically finds and loads workflow nodes



| Backend | FPS | Avg Time | Detections |winml/## 🚀 Quick Start

|---------|-----|----------|------------|

| **DirectML GPU** | 76.8 | 13.01ms | 14 |├── workflow_cli.py              # Main workflow executor

| **NPU (OpenVINO)** | 45.8 | 21.84ms | 14 |

| **CUDA (PyTorch)** | 29.9 | 33.46ms | 15 |├── workflow_orchestrator.py     # Workflow engine with parallel execution### 1. Install Dependencies

| **CPU (ONNX)** | 13.9 | 72.01ms | 9 |

├── workflow_loader.py           # Workflow JSON parser```powershell

*Tested with YOLOv8s model on 640x640 images, 50 iterations*

├── workflow_data_session.py     # Data caching systemuv sync

## 📁 Project Structure

├── framework_data_optimizer.py  # Memory optimization```

```

winml/├── logging_config.py            # Centralized logging

├── workflow_orchestrator.py    # Main workflow engine

├── workflow_cli.py              # CLI interface├── visualize_detections.py      # Detection visualization tool### 2. Run Generic Workflow

├── workflow_loader.py           # JSON workflow parser

├── workflow_data_session.py     # Data session caching│```powershell

├── logging_config.py            # Centralized logging

├── framework_data_optimizer.py  # Data optimization layer├── workflow-nodes/              # Modular workflow nodesuv run python generic_workflow_engine.py

├── custom_node_example.py       # Custom node tutorial

├── visualize_detections.py      # Detection visualization│   ├── directml_model_loader_node.py```

│

├── workflow-nodes/              # Workflow task nodes (12 nodes)│   ├── cpu_model_loader_node.py

│   ├── image_reader_node.py

│   ├── *_model_loader_node.py   # Model loaders (4 backends)│   ├── npu_model_loader_node.py### 3. Run JSON-Defined Workflow

│   ├── *_inference_node.py      # Inference nodes (4 backends)

│   └── performance_stats_node.py│   ├── cuda_model_loader_node.py```powershell

│

├── workflows/                   # Workflow definitions│   ├── gpu_inference_node.pyuv run python workflow_cli.py

│   ├── yolov8_object_detection.json  # Main YOLO workflow

│   └── custom_node_demo.json         # Custom node example│   ├── cpu_inference_node.py```

│

├── models/                      # AI models│   ├── npu_inference_node.py

│   ├── yolov8s.onnx            # For DirectML, CPU, NPU

│   └── yolov8s.pt              # For CUDA│   ├── cuda_inference_node.py## 🎯 What It Does

│

├── input/                       # Sample images│   ├── image_reader_node.py

│   ├── soccer.jpg

│   └── desk.jpg│   └── performance_stats_node.py**Clean Architecture with Separation of Concerns**:

│

└── shared/                      # Shared utilities│

    └── yolo_postprocessing.py

```├── workflows/                   # Workflow definitions- **Generic Preprocessor**: Handles letterbox, center-crop, normalization for any model



## 🎯 Usage Examples│   └── yolov8_object_detection.json- **Inference Nodes**: Focus only on inference (CPU/GPU/NPU) with no preprocessing



### Run Main Workflow│- **Generic Postprocessor**: Processes YOLO detections, classification outputs



```powershell├── models/                      # AI models- **Parallel Execution**: Each phase runs optimally (sequential preprocessing, parallel inference)

# Execute YOLOv8 detection with all backends

uv run python workflow_cli.py --workflow workflows/yolov8_object_detection.json│   ├── yolov8s.onnx            # ONNX format for DirectML/CPU/NPU



# Use custom image│   └── yolov8s.pt              # PyTorch format for CUDA## 🏆 Performance Results

uv run python workflow_cli.py --workflow workflows/yolov8_object_detection.json --image path/to/image.jpg

```│



### Create Custom Nodes├── input/                       # Input imagesFrom latest generic workflow run:



```python├── output/                      # Detection results & visualizations

# 1. Define your custom node

class MyProcessorNode:└── archive/                     # Old/test files| Backend | Device | Performance | Architecture |

    def execute(self, inputs):

        # Your processing logic|---------|--------|-------------|--------------|

        data = inputs.get('data', '')

        result = data.upper()  # Example transformation```| **GPU** | DirectML | **374.6 FPS** | ✅ Clean Separation |

        return {'success': True, 'result': result}

| **NPU** | OpenVINO | **137.6 FPS** | ✅ Focused Inference |

# 2. Register with the engine

from workflow_orchestrator import WorkflowEngine## 🎯 Key Features| **CPU** | ONNX Runtime | **104.0 FPS** | ✅ Generic Processing |

engine = WorkflowEngine()

engine.register_custom_node('my_processor', MyProcessorNode)



# 3. Use in workflow JSON### Multi-Backend Support**Total execution time: 0.47s** with clean architecture benefits

{

    "nodes": [- **DirectML**: Windows GPU acceleration (AMD, Intel, NVIDIA)

        {

            "id": "process_step",- **CUDA**: Native NVIDIA GPU acceleration## 🔧 Technical Architecture

            "type": "my_processor",

            "depends_on": ["previous_step"]- **NPU**: Intel Neural Processing Unit

        }

    ]- **CPU**: ONNX Runtime fallback### Key Benefits: Clean Separation

}

```- **Generic Nodes**: Reusable across all AI model types



### Load Nodes from Custom Directory### Performance Optimizations- **Focused Inference**: No preprocessing logic mixed with inference



```python- ✅ Plugin-based isolation (no subprocess overhead)- **Parallel Phases**: Optimal execution strategy per workflow phase

# Load additional nodes from custom directories

engine = WorkflowEngine(- ✅ Parallel node execution- **JSON Workflows**: Declarative workflow definitions

    nodes_directory="workflow-nodes",

    custom_node_paths=["my_custom_nodes", "/another/path"]- ✅ Data session caching

)

- ✅ Shared memory optimization### Execution Phases

# List all available nodes

available_nodes = engine.list_available_nodes()- ✅ Model instance caching (10-14x speedup)1. **Data Loading**: Parallel (image + model loading)

print(f"Found {len(available_nodes)} nodes")

```2. **Preprocessing**: Sequential (shared tensor for all backends)



## 🔧 Workflow Definition### Architecture3. **Inference**: Parallel (GPU, CPU, NPU simultaneously)



Workflows are defined in JSON format:- **Modular nodes**: Each component is a standalone node4. **Postprocessing**: Parallel (per-backend result processing)



```json- **Data sessions**: Efficient model/data sharing5. **Analysis**: Sequential (aggregate statistics)

{

  "name": "YOLOv8 Object Detection",- **Parallel execution**: All backends run concurrently

  "nodes": [

    {- **Timing breakdown**: Separate loading vs inference phases## 📦 Dependencies

      "id": "load_image",

      "type": "image_reader_node",

      "parameters": {

        "image_path": "input/soccer.jpg"## 🔧 Configuration```toml

      }

    },dependencies = [

    {

      "id": "gpu_inference",### Workflow JSON Structure    "numpy>=1.24.0,<2.0.0",

      "type": "gpu_inference_node",

      "depends_on": ["load_image"],```json    "pillow>=10.0.0", 

      "parameters": {

        "confidence_threshold": 0.25,{    "requests>=2.31.0",

        "iterations": 50

      },  "name": "YOLOv8 Object Detection",    "onnxruntime-directml==1.20.0",  # GPU + CPU inference

      "parallel_group": "inference"

    }  "nodes": [    "onnxruntime==1.20.0",           # CPU fallback

  ]

}    {    "openvino>=2024.6.0",            # NPU inference

```

      "id": "load_directml_model",]

### Workflow Features

      "type": "directml_model_loader_node",```

- **Dependencies**: `depends_on` specifies execution order

- **Parallel Groups**: Nodes with same `parallel_group` run concurrently      "parameters": {

- **Parameters**: Custom parameters passed to each node

- **Data Sessions**: Automatically caches models and data between nodes        "model_path": "models/yolov8s.onnx",## 🎯 Usage Examples



## 🏗️ Architecture        "device_id": 1  // GPU 1 = RTX 5090



### Core Components      }### Generic Workflow Engine



1. **Workflow Orchestrator** (`workflow_orchestrator.py`)    }```powershell

   - Main execution engine

   - Handles node loading, dependency resolution  ],# Run clean architecture demonstration

   - Manages parallel execution

   - Auto-discovers nodes from directories  "execution_config": {uv run python generic_workflow_engine.py



2. **Data Session** (`workflow_data_session.py`)    "use_data_session": true,```

   - Thread-safe key-value store

   - Namespace isolation per backend    "parallel_execution": true

   - Caches models and preprocessed data

   - Reduces redundant loading (10-14x speedup)  }### JSON Workflow Engine



3. **Workflow Nodes** (`workflow-nodes/`)}```powershell

   - Modular, reusable components

   - Each node implements `execute(inputs)` method```# Run AI detection workflow from JSON

   - Discoverable via file naming convention

   - Support for both short and full namesuv run python workflow_cli.py



### Supported Backends## 📈 Benchmarks```



| Backend | Framework | Hardware | Use Case |

|---------|-----------|----------|----------|

| **DirectML** | ONNX Runtime | RTX 5090 GPU | Best performance on Windows |### Timing Breakdown (50 iterations)### Workflow Management

| **CUDA** | PyTorch | RTX 5090 GPU | Native CUDA acceleration |

| **NPU** | OpenVINO | Intel AI Boost | Low-power AI inference |- **Loading Phase**: 2.3s (models + image)```powershell

| **CPU** | ONNX Runtime | CPU cores | Fallback option |

- **Inference Phase**: 3.7s (parallel execution)# List available workflows

### Execution Flow

- **Total Workflow**: 6.0suv run python workflow_loader.py

```

1. Load Workflow JSON

   ↓

2. Discover Available Nodes### Detection Quality# Load specific workflow

   ↓

3. Build Dependency Graph- DirectML & NPU: 14 objects (11 people, 2 cars, 1 ball)python -c "from workflow_loader import load_workflow; print(load_workflow('generic_ai_workflow.json'))"

   ↓

4. Load Models (Parallel)- CUDA: 15 objects (11 people, 3 cars, 1 ball)```

   - DirectML Model Loader

   - CPU Model Loader- CPU: 9 objects (7 people, 1 car, 1 ball)

   - NPU Model Loader

   - CUDA Model Loader### Core Inference

   ↓

5. Run Inference (Parallel)## 🛠️ Development```powershell

   - GPU Inference

   - CPU Inference# Test GPU inference

   - NPU Inference

   - CUDA Inference### Adding a New Backenduv run python winml_inference.py gpu:0 1000

   ↓

6. Collect & Compare Results1. Create model loader node: `{backend}_model_loader_node.py`

   - Performance Stats

   - Detection Visualization2. Create inference node: `{backend}_inference_node.py`# Test NPU inference  

```

3. Add to workflow JSON with appropriate namespaceuv run python winml_inference.py npu:0 1000

## 🔌 Custom Nodes

4. Configure device parameters```

### Method 1: Programmatic Registration



```python

class TextProcessorNode:### Custom Workflows## 🎉 Benefits

    def execute(self, inputs):

        text = inputs.get('text', '')Create new JSON files in `workflows/` following the schema in `yolov8_object_detection.json`.

        return {'processed': text.upper()}

✅ **Clean Architecture**: Separation of preprocessing, inference, postprocessing  

engine = WorkflowEngine()

engine.register_custom_node('text_processor', TextProcessorNode)## 📊 Visualization✅ **Generic Nodes**: Reusable across different AI model types  

```

✅ **Parallel Execution**: Optimal performance at each workflow phase  

### Method 2: File-Based Discovery

```powershell✅ **JSON Workflows**: Declarative, versionable workflow definitions  

Create a file in `workflow-nodes/` or custom directory:

# Generate detection comparison images✅ **Multi-Backend**: CPU, GPU, NPU support in single environment  

```python

# my_custom_nodes/filter_node.pyuv run python visualize_detections.py

class FilterNode:

    def execute(self, inputs):Perfect for **production AI workflows** with clean, maintainable architecture! 🚀

        data = inputs.get('data', [])

        filtered = [x for x in data if x > 0]# Output files:## 🛠️ Development

        return {'success': True, 'filtered': filtered}

```# - output/detections_comparison.jpg (2x2 grid)



Load automatically:# - output/detections_directml.jpg### Adding New Workflow Nodes



```python# - output/detections_cuda.jpg1. Create node class in `workflow-nodes/` inheriting from `WorkflowNode`

engine = WorkflowEngine(custom_node_paths=["my_custom_nodes"])

# FilterNode is now available as 'filter' or 'filter_node'# - output/detections_npu.jpg2. Implement `execute()` method

```

# - output/detections_cpu.jpg3. Add node type to workflow JSON definitions

### Node Discovery

```

The engine automatically discovers nodes with both naming conventions:

- `my_node.py` → `my_node`### Creating New Workflows

- `my_node_node.py` → `my_node` and `my_node_node`

## 🔍 Hardware Utilization1. Create JSON file in `workflows/` directory

## 📦 Installation

2. Define nodes, dependencies, and execution groups

```powershell

# Clone the repository- **GPU 0 (Intel iGPU)**: Not used (4% display only)3. Load with `workflow_loader.load_workflow()`

git clone https://github.com/frvie/workflows.git

cd workflows- **GPU 1 (RTX 5090)**: DirectML + CUDA



# Install dependencies with UV- **NPU (Intel AI Boost)**: OpenVINO inference### Testing Changes

uv sync

- **CPU**: ONNX Runtime fallback```powershell

# Or use pip

pip install -r requirements.txt# Test workflow loading

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
