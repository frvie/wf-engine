# Workflow Engine Usage Guide

A high-performance, granular AI inference workflow platform with atomic node composition and AI-powered agentic modes.

## Quick Start

### 1. Environment Setup

```powershell
# Clone and setup
git clone <repository-url>
cd workflow_engine

# Install dependencies
uv sync

# Activate environment
.\.venv\Scripts\activate
```

### 2. Run a Pre-built Workflow

```powershell
# Execute granular parallel inference (recommended - shows full platform capabilities)
uv run python workflow_orchestrator_minimal.py workflows/granular_parallel_inference.json

# Execute simpler multi-backend workflow  
uv run python workflow_orchestrator_minimal.py workflows/multi_backend_separated_v8.json

# Or with custom parameters
uv run python workflow_orchestrator_minimal.py workflows/granular_parallel_inference.json --max-workers 8
```

---

## 🤖 AI-Powered Usage (Claude Desktop Integration)

### Prerequisites

1. **Install Claude Desktop** (version 0.14.10+)
2. **Configure MCP Server**:

```json
// %APPDATA%\Claude\claude_desktop_config.json
{
  "mcpServers": {
    "workflow-engine": {
      "command": "c:\\dev\\workflow_engine\\.venv\\Scripts\\python.exe",
      "args": ["c:\\dev\\workflow_engine\\start_mcp.py"],
      "env": {}
    }
  }
}
```

### Available AI Commands

Once Claude Desktop is configured, you can use natural language commands:

#### 🎯 **Create Workflows from Description**
```
"Create a real-time object detection workflow for my webcam"
"Set up batch image processing for a folder of images"  
"Build a video analysis pipeline with GPU acceleration"
```

#### ⚡ **Execute Existing Workflows**
```
"Run the granular parallel inference workflow"
"Execute workflows/granular_parallel_inference.json with performance benchmarking"
"Run the multi-backend detection workflow"  
"Process my video file using the fastest available backend"
```

#### 📊 **Performance Analysis**
```
"Analyze performance of my last workflow run"
"Show me optimization suggestions for better FPS"
"Compare CPU vs GPU performance on my system"
```

#### 🔧 **System Information**
```
"What AI backends are available on my system?"
"Check if my hardware supports DirectML acceleration"
"List all available workflow nodes"
```

#### 📈 **Advanced Optimization**
```
"Optimize my workflow for real-time processing"
"Suggest the best backend for my hardware"
"Create a performance-optimized detection pipeline"
```

---

## 💻 Command Line Interface (CLI)

### Core Commands

#### Execute Workflows
```powershell
# Basic execution (granular workflow - recommended)
uv run python workflow_orchestrator_minimal.py workflows/granular_parallel_inference.json

# With custom parameters for high throughput
uv run python workflow_orchestrator_minimal.py workflows/granular_parallel_inference.json --max-workers 8

# Background execution with logging
uv run python workflow_orchestrator_minimal.py workflows/granular_parallel_inference.json --log-level INFO > logs/execution.log 2>&1

# Quick simple workflow for testing
uv run python workflow_orchestrator_minimal.py workflows/multi_backend_separated_v8.json
```

#### Environment Management
```powershell
# Configure Python environment
uv run python -c "from utilities.shared_memory_utils import configure_python_environment; configure_python_environment()"

# Check environment details
uv run python -c "from utilities.shared_memory_utils import get_python_environment_details; print(get_python_environment_details())"

# Install additional packages
uv add <package_name>
```

#### Performance Testing
```powershell
# Run comprehensive performance benchmarks (granular workflow)
uv run python -c "
from src.core.engine import FunctionWorkflowEngine
engine = FunctionWorkflowEngine()
result = engine.execute_workflow('workflows/granular_parallel_inference.json')
print(f'Performance: {result.get(\"performance\", {})}')
"

# Quick performance test (simpler workflow)
uv run python -c "
from src.core.engine import FunctionWorkflowEngine
engine = FunctionWorkflowEngine()
result = engine.execute_workflow('workflows/multi_backend_separated_v8.json')
print(f'Performance: {result.get(\"performance\", {})}')
"
```

### Advanced CLI Operations

#### Workflow Validation
```powershell
# Validate granular workflow JSON (complex example)
uv run python -c "
import json
with open('workflows/granular_parallel_inference.json', 'r') as f:
    workflow = json.load(f)
    nodes = workflow.get('nodes', [])
    print(f'Granular workflow validation: {len(nodes)} nodes found')
    print('Sample nodes:', [node['id'] for node in nodes[:5]])
"

# Validate simple workflow JSON
uv run python -c "
import json  
with open('workflows/multi_backend_separated_v8.json', 'r') as f:
    workflow = json.load(f)
    print('Simple workflow validation:', 'nodes' in workflow and 'connections' in workflow)
"
```

#### Node Discovery
```powershell
# List available nodes
uv run python -c "
from src.agentic.agent import WorkflowComposer
composer = WorkflowComposer()
composer._ensure_nodes_discovered()
print('Available nodes:', list(composer.available_nodes.keys()))
"
```

#### Backend Testing
```powershell
# Test DirectML availability
uv run python -c "
try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    print('DirectML available:', 'DmlExecutionProvider' in providers)
except ImportError:
    print('ONNXRuntime not installed')
"

# Test OpenVINO availability  
uv run python -c "
try:
    import openvino as ov
    core = ov.Core()
    print('OpenVINO devices:', core.available_devices)
except ImportError:
    print('OpenVINO not installed')
"
```

#### Memory and Performance Monitoring
```powershell
# Monitor system resources during execution
uv run python -c "
import psutil
print(f'CPU usage: {psutil.cpu_percent()}%')
print(f'Memory usage: {psutil.virtual_memory().percent}%')
print(f'Available cores: {psutil.cpu_count()}')
"
```

---

## 📁 Available Workflows

### Pre-built Workflows

| Workflow | Description | Backends | Use Case |
|----------|-------------|-----------|----------|
| `granular_parallel_inference.json` | **Granular atomic node pipeline** | CPU, GPU, NPU | **Recommended - Full platform showcase** |
| `multi_backend_separated_v8.json` | Simple multi-backend detection | CPU, GPU, NPU | Quick testing and demos |

### Granular Workflow Highlights

The **`granular_parallel_inference.json`** demonstrates advanced workflow orchestration:

- **33+ atomic nodes** - Each processing step is a separate, reusable node
- **Parallel backend execution** - CPU, DirectML GPU, and NPU run simultaneously  
- **Complex dependency chains** - Automatic orchestration of 100+ dependencies
- **Built-in benchmarking** - Performance comparison across all backends
- **Complete YOLO pipeline** - From image loading to final detection results

**Example nodes**: `download_model` → `read_img` → `resize` → `normalize` → `cpu_session` + `npu_session` + `directml_session` → parallel inference → `compare_performance`

### Workflow Structure
```json
{
  "workflow": {
    "name": "Granular Multi-Backend YOLO Inference",
    "settings": {
      "max_parallel_nodes": 8
    }
  },
  "nodes": [
    {
      "id": "read_img", 
      "function": "src.nodes.image_ops.read_image_node",
      "inputs": {"image_path": "input/soccer.jpg"},
      "dependencies": []
    },
    {
      "id": "cpu_session",
      "function": "src.nodes.onnx_ops.create_onnx_cpu_session_node", 
      "dependencies": ["download_model"]
    }
    // ... 30+ more nodes with complex dependencies
  ]
}
```

---

## 🔧 Configuration

### Environment Variables
```powershell
# Set logging level
$env:WORKFLOW_LOG_LEVEL = "INFO"

# Set model directory
$env:WORKFLOW_MODEL_DIR = "models"

# Enable performance profiling
$env:WORKFLOW_PROFILE = "true"
```

### Hardware Optimization

#### For Intel CPUs
```powershell
# Enable Intel optimizations
$env:OMP_NUM_THREADS = "4"
$env:INTEL_TBB_THREAD_COUNT = "4"
```

#### For NVIDIA GPUs
```powershell
# CUDA memory management
$env:CUDA_VISIBLE_DEVICES = "0"
$env:CUDA_DEVICE_ORDER = "PCI_BUS_ID"
```

#### For DirectML (Windows)
```powershell
# DirectML adapter selection (optional)
$env:DIRECTML_DEVICE = "0"  # Use first GPU
```

---

## 📊 Performance Monitoring

### Real-time Performance Tracking
```powershell
# Monitor granular workflow execution (comprehensive benchmarking)
uv run python -c "
from src.core.engine import FunctionWorkflowEngine
from src.utilities.logging_config import setup_logging
import time

setup_logging()
engine = FunctionWorkflowEngine()

start_time = time.time()
result = engine.execute_workflow('workflows/granular_parallel_inference.json')
duration = time.time() - start_time

print(f'Execution time: {duration:.2f}s')
print(f'Performance data: {result.get(\"performance\", \"No data\")}')
print(f'Backend comparison available in result')
"

# Quick performance check (simple workflow)
uv run python -c "
from src.core.engine import FunctionWorkflowEngine
import time

engine = FunctionWorkflowEngine()
start_time = time.time()
result = engine.execute_workflow('workflows/multi_backend_separated_v8.json')
duration = time.time() - start_time
print(f'Quick test: {duration:.2f}s')
"
```

### Log Analysis
```powershell
# View recent logs
Get-Content logs/workflow_engine.log -Tail 50

# Filter performance metrics
Select-String "FPS|performance|duration" logs/workflow_engine.log
```

---

## 🚀 Advanced Usage

### Custom Node Development
```python
# Example custom node (src/nodes/custom_node.py)
from src.nodes.base_node import BaseNode

class CustomProcessingNode(BaseNode):
    def __init__(self, config):
        super().__init__(config)
        self.processing_type = config.get('processing_type', 'default')
    
    def process(self, input_data):
        # Custom processing logic
        return {'processed_data': input_data, 'node_type': 'custom'}
```

### Batch Processing
```powershell
# Process multiple files
$files = Get-ChildItem "input_folder" -Filter "*.jpg"
foreach ($file in $files) {
    uv run python workflow_orchestrator_minimal.py workflows/single_image_detection.json --input $file.FullName
}
```

### Integration with External Systems
```python
# REST API integration example
import requests
from src.core.engine import FunctionWorkflowEngine

def process_via_api(image_url):
    engine = FunctionWorkflowEngine()
    workflow_config = {
        'input_source': image_url,
        'output_format': 'json'
    }
    
    result = engine.execute_workflow('workflows/api_detection.json', workflow_config)
    return result
```

---

## 🔍 Troubleshooting

### Common Issues

#### MCP Integration Not Working
```powershell
# Check Claude Desktop configuration
Get-Content "$env:APPDATA\Claude\claude_desktop_config.json"

# Test MCP server manually
.\.venv\Scripts\python.exe start_mcp.py

# Check logs
Get-Content "$env:APPDATA\Claude\logs\mcp*.log" -Tail 20
```

#### Performance Issues
```powershell
# Check available backends
uv run python -c "
import onnxruntime as ort
print('Available providers:', ort.get_available_providers())
"

# Memory usage check
uv run python -c "
import psutil
print(f'Available RAM: {psutil.virtual_memory().available / 1024**3:.1f} GB')
"
```

#### Import Errors
```powershell
# Reinstall dependencies
uv sync --reinstall

# Check Python environment
uv run python --version
uv run python -c "import sys; print(sys.path)"
```

### Debug Mode
```powershell
# Enable verbose logging (granular workflow - shows detailed node execution)
uv run python workflow_orchestrator_minimal.py workflows/granular_parallel_inference.json --log-level DEBUG

# Profile performance (comprehensive analysis)
uv run python -m cProfile -o profile.stats workflow_orchestrator_minimal.py workflows/granular_parallel_inference.json

# Quick debug (simple workflow)
uv run python workflow_orchestrator_minimal.py workflows/multi_backend_separated_v8.json --log-level DEBUG
```

---

## 📞 Support

### Performance Benchmarks
- **CPU (Intel i7)**: ~33 FPS object detection
- **GPU (DirectML)**: ~258 FPS object detection  
- **NPU (Intel)**: ~35 FPS object detection

### System Requirements
- **Python**: 3.11+ 
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 2GB for models and dependencies
- **GPU**: Optional but recommended for high performance

### Getting Help
1. Check logs in `logs/` directory
2. Review error messages in terminal output
3. Verify hardware compatibility
4. Ensure all dependencies are installed with `uv sync`

---

*For the latest updates and advanced features, see the project README and documentation.*