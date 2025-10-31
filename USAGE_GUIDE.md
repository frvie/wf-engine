# Granular Workflow Engine Usage Guide

A high-performance, granular AI inference workflow platform with atomic node composition and AI-powered agentic modes.

---

## 1. 🚀 Environment Setup

### Prerequisites
- **Python**: 3.11+ 
- **RAM**: 16GB recommended (granular workflows are memory-intensive)
- **Storage**: 4GB for models and dependencies
- **GPU**: Optional but recommended for DirectML acceleration
- **NPU**: Intel NPU for hardware-accelerated inference

### Installation Steps

```powershell
# Clone and setup
git clone <repository-url>
cd workflow_engine

# Install dependencies
uv sync

# Activate environment (optional - uv run handles this automatically)
.\.venv\Scripts\activate
```

### Verify Installation

```powershell
# Test basic functionality
uv run python workflow_orchestrator_minimal.py --help

# Check hardware detection
uv run python -c "
from src.nodes.detect_gpus import detect_gpus_node
result = detect_gpus_node()
print('GPU Detection:', result)
"

# Verify dependencies
uv run python -c "
try:
    import onnxruntime as ort
    print('ONNX providers:', ort.get_available_providers())
except ImportError:
    print('ONNXRuntime not installed')

try:
    import openvino as ov
    print('OpenVINO devices:', ov.Core().available_devices)
except ImportError:
    print('OpenVINO not installed')
"
```

---

## 2. 🔧 Direct Workflow Engine Usage (Non-AI Mode)

### Core Workflow Orchestrator (`workflow_orchestrator_minimal.py`)

The workflow orchestrator is the foundation engine that executes JSON workflow files without AI assistance.

#### Basic Execution
```powershell
# Execute a workflow file directly via engine
uv run python src/core/engine.py <workflow_file.json>

# Execute granular parallel inference
uv run python src/core/engine.py workflows/granular_parallel_inference.json

# Execute as Python module
uv run python -m src.core.engine workflows/granular_parallel_inference.json

# Execute via WFE CLI (recommended)
uv run python wfe.py run workflows/granular_parallel_inference.json
```

#### Command Line Options
```powershell
# Direct engine execution
uv run python src/core/engine.py <workflow_file.json>

# WFE CLI options (recommended)
uv run python wfe.py run <workflow_file.json> [OPTIONS]
  --learn              # Enable agentic learning mode
  --log-level LEVEL    # Set logging level (DEBUG, INFO, WARNING, ERROR)
  --profile           # Enable performance profiling

# Python module execution
uv run python -m src.core.engine <workflow_file.json>
```

#### Workflow File Structure
```json
{
  "workflow": {
    "name": "My Custom Workflow",
    "description": "Description of what this workflow does",
    "settings": {
      "max_parallel_nodes": 8,
      "timeout_seconds": 300
    }
  },
  "nodes": [
    {
      "id": "unique_node_id",
      "function": "src.nodes.module.function_name",
      "inputs": {
        "parameter_name": "value_or_reference"
      },
      "dependencies": ["other_node_id"]
    }
  ]
}
```

#### Available Workflows 
```powershell
# List all available workflows
ls workflows/*.json

# Available workflows:
# • granular_parallel_inference.json - Multi-backend YOLO inference (CPU + NPU + DirectML)
# • granular_video_detection.json - Video processing pipeline
```

#### Performance Monitoring
```powershell
# Enable detailed execution logging
uv run python src/core/engine.py workflows/granular_parallel_inference.json --log-level INFO

# Profile workflow performance
uv run python -m cProfile -o workflow_profile.stats src/core/engine.py workflows/granular_parallel_inference.json

# Analyze profile results
uv run python -c "
import pstats
stats = pstats.Stats('workflow_profile.stats')
stats.sort_stats('cumulative').print_stats(20)
"

# Monitor memory usage
uv run python -c "
import psutil
print(f'Available RAM: {psutil.virtual_memory().available / 1024**3:.1f} GB')
print('Recommended: 16GB+ for smooth granular workflow execution')
"
```

#### Workflow Validation
```powershell
# Validate workflow structure without execution
uv run python src/core/engine.py workflows/granular_parallel_inference.json --validate-only

# Check workflow dependencies
uv run python -c "
import json
with open('workflows/granular_parallel_inference.json', 'r') as f:
    workflow = json.load(f)
    nodes = workflow.get('nodes', [])
    print(f'Workflow: {len(nodes)} nodes')
    for node in nodes[:5]:
        print(f'  {node[\"id\"]}: {node.get(\"dependencies\", [])}')
"
```

### Environment Management
```powershell
# Check available environments
ls workflow-envs/

# Create custom environment for specific nodes
uv venv workflow-envs/my-custom-env
workflow-envs/my-custom-env/Scripts/pip install specific-package

# Use custom environment in workflow
uv run python workflow_orchestrator_minimal.py workflows/my_workflow.json --env my-custom-env
```

---

## 3. 🤖 Agentic Mode Usage (Python API)

The agentic mode provides AI-powered workflow creation, optimization, and learning capabilities through Python APIs.

### Workflow Data Session (`workflow_data_session.py`)

The data session manages persistent learning and optimization data.

#### Basic Usage
```python
from workflow_data_session import WorkflowDataSession

# Initialize session
session = WorkflowDataSession()

# Record workflow execution
session.record_execution(
    workflow_name="granular_parallel_inference",
    execution_time=2.34,
    fps=452.1,
    backend="DirectML",
    metadata={"nodes": 35, "parallel_nodes": 8}
)

# Get performance trends
trends = session.get_performance_trends("granular_parallel_inference")
print(f"Average FPS: {trends['avg_fps']}")
print(f"Best FPS: {trends['best_fps']}")

# Get optimization suggestions
suggestions = session.get_optimization_suggestions("granular_parallel_inference", target_fps=60)
for suggestion in suggestions:
    print(f"💡 {suggestion}")
```

#### Advanced Session Management
```python
# Get all recorded sessions
all_sessions = session.get_all_sessions()

# Filter sessions by criteria
gpu_sessions = session.filter_sessions(backend="DirectML")
recent_sessions = session.get_recent_sessions(days=7)

# Export/import session data
session.export_data("learning_data.json")
session.import_data("previous_learning_data.json")

# Clear old data
session.cleanup_old_sessions(days=30)
```

### Workflow Environment Manager (`workflow_environment_manager.py`)

Manages isolated environments for different workflow components.

#### Environment Operations
```python
from workflow_environment_manager import WorkflowEnvironmentManager

# Initialize manager
env_manager = WorkflowEnvironmentManager()

# Create environment for DirectML
env_manager.create_environment(
    name="directml-inference",
    requirements=["onnxruntime-directml>=1.19.0", "numpy>=1.24.0"]
)

# List all environments
environments = env_manager.list_environments()
for env_name, env_info in environments.items():
    print(f"Environment: {env_name}")
    print(f"  Path: {env_info['path']}")
    print(f"  Packages: {len(env_info['packages'])}")

# Execute in isolated environment
result = env_manager.execute_in_environment(
    env_name="directml-inference",
    function_name="run_directml_inference",
    args={"model_path": "models/yolov8s.onnx"}
)

# Cleanup unused environments
env_manager.cleanup_unused_environments()
```

#### Environment Configuration
```python
# Configure environment with specific Python version
env_manager.create_environment(
    name="cpu-optimized",
    python_version="3.11",
    requirements=["onnxruntime>=1.19.0", "opencv-python>=4.8.0"]
)

# Install additional packages
env_manager.install_packages(
    env_name="cpu-optimized",
    packages=["scikit-image", "matplotlib"]
)

# Environment health check
health = env_manager.check_environment_health("directml-inference")
print(f"Environment healthy: {health['is_healthy']}")
if not health['is_healthy']:
    print(f"Issues: {health['issues']}")
```

### Agentic Workflow Creation
```python
from src.agentic.agent import WorkflowAgent

# Initialize AI agent
agent = WorkflowAgent(model_name="qwen2.5-coder:7b")

# Create workflow from natural language
workflow_spec = {
    "description": "Real-time object detection using NPU and DirectML with performance comparison",
    "target_fps": 60,
    "backends": ["NPU", "DirectML", "CPU"],
    "input_format": "image",
    "output_format": "detections"
}

workflow = await agent.create_workflow(workflow_spec)
print(f"Generated workflow: {workflow['name']}")
print(f"Nodes: {len(workflow['nodes'])}")

# Save generated workflow
with open("workflows/ai_generated_workflow.json", "w") as f:
    json.dump(workflow, f, indent=2)
```

### Learning and Optimization
```python
# Enable learning mode for executions
agent.enable_learning_mode()

# Execute workflow with learning
result = agent.execute_with_learning(
    workflow_path="workflows/granular_parallel_inference.json",
    learn_from_execution=True
)

# Get learned insights
insights = agent.get_learning_insights()
print(f"📚 Learning Summary:")
print(f"  Executions analyzed: {insights['total_executions']}")
print(f"  Workflow types learned: {len(insights['workflow_types'])}")

# Apply learned optimizations
optimized_workflow = agent.optimize_workflow(
    workflow_path="workflows/granular_parallel_inference.json",
    target_metric="fps",
    target_value=100
)

# Save optimized workflow
with open("workflows/optimized_workflow.json", "w") as f:
    json.dump(optimized_workflow, f, indent=2)
```

### Node Generation API
```python
from src.nodes.node_generator import NodeGenerator, NodeSpec

# Initialize node generator
generator = NodeGenerator(model_name="qwen2.5-coder:7b")

# Create node specification
spec = NodeSpec(
    goal="Apply Gaussian blur to image with configurable kernel size",
    inputs=["image", "kernel_size"],
    outputs=["blurred_image"],
    category="atomic",
    constraints=["use opencv", "handle edge cases", "optimize for performance"]
)

# Generate node
result = await generator.generate_node(spec)

if result['success']:
    print(f"✅ Generated node: {result['node_name']}")
    print(f"📁 Saved to: {result['file_path']}")
    print(f"🔧 Function: {result['function_name']}")
else:
    print(f"❌ Generation failed: {result['error']}")
```

### Advanced Agent Configuration
```python
# Configure agent with custom settings
agent = WorkflowAgent(
    model_name="deepseek-coder:6.7b-instruct",
    temperature=0.1,
    max_tokens=2000,
    learning_rate=0.01,
    optimization_strategy="performance_first"
)

# Set custom prompts
agent.set_system_prompt("You are an expert in high-performance inference workflows...")
agent.set_optimization_prompt("Focus on maximizing FPS while maintaining accuracy...")

# Configure learning parameters
agent.configure_learning(
    min_executions_for_insights=5,
    confidence_threshold=0.8,
    optimization_aggressiveness="moderate"
)
```

---

## 4. 🎛️ WFE CLI Usage

The `wfe` command-line interface provides a user-friendly wrapper around the workflow engine with AI-powered features.

### Core Commands

#### 📋 **List Available Workflows**
```powershell
# List all available workflows
wfe workflows

# Output shows:
# • granular_parallel_inference.json
#   Strategy: unknown, Nodes: 35
# • multi_backend_separated_v8.json  
#   Strategy: unknown, Nodes: 15
```

#### ⚡ **Execute Workflows**
```powershell
# Basic workflow execution
wfe run workflows/granular_parallel_inference.json

# Run with agentic learning enabled
wfe run workflows/granular_parallel_inference.json --learn

# Run with specific options
wfe run workflows/granular_parallel_inference.json --log-level DEBUG --profile
```

#### 💻 **Hardware Detection**
```powershell
# List available inference devices
wfe devices

# Output shows:
# GPU Devices: NVIDIA RTX 5090 (31.84 GB)
# ✅ DirectML: Available (Device ID: 1)
# ✅ OpenVINO NPU: Available (Intel NPU)
# ✅ CUDA: Available (Device ID: 0)
```

#### 🔧 **Node Management**
```powershell
# List all available workflow nodes
wfe nodes

# Shows all atomic functions available for workflows
# Organized by category: image_ops, onnx_ops, yolo_ops, etc.
```

#### 🧠 **AI-Powered Workflow Creation**
```powershell
# Create workflow from natural language description
wfe create "real-time object detection using NPU and DirectML"

# Create and run immediately
wfe create "fast video processing for dashcam footage" --run

# Save to specific location
wfe create "batch image analysis pipeline" -o workflows/custom_batch.json

# Create with specific target performance
wfe create "optimized detection for Intel NPU with 60 FPS target"
```

#### 🤖 **AI Node Generation**
```powershell
# Generate new nodes from description
wfe generate "apply gaussian blur to image" -i image -o blurred_image -c atomic

# Generate with specific model
wfe generate "extract SIFT features from grayscale image" -i image -o features -c utils --model qwen2.5-coder:14b

# Generate with constraints
wfe generate "convert video to frames" -i video_path -o frame_list --constraints "use opencv,efficient memory usage"

# Show which model is being used
wfe generate "blur image" --model qwen2.5-coder:7b  # Shows: Model: qwen2.5-coder:7b
```

#### � **Performance Analysis & Optimization**
```powershell
# Get optimization suggestions
wfe optimize workflows/granular_parallel_inference.json --target 60

# Check agent learning status and history
wfe status

# Example output:
# 📚 Knowledge Base:
#    Total executions: 12
#    Workflow types learned: 3
#    • granular_parallel: Avg FPS: 452.1, Best FPS: 500.0
#    • multi_backend: Avg FPS: 45.1, Best FPS: 67.3
```


#### 🔌 **MCP Server Management**
```powershell
# Start MCP server for AI integration
wfe mcp

# Start MCP server with specific options
wfe mcp --port 8000 --host localhost --debug
```

### WFE CLI Advanced Options

#### Command Options and Parameters
```powershell
# wfe run options
wfe run <workflow> [OPTIONS]
  --learn              # Enable agentic learning mode
  --log-level LEVEL    # Set logging level (DEBUG, INFO, WARNING, ERROR)
  --profile           # Enable performance profiling
  --timeout SECONDS   # Set execution timeout

# wfe create options
wfe create "<description>" [OPTIONS]
  --run               # Execute immediately after creation
  -o, --output PATH   # Output file path
  --model MODEL       # Ollama model to use (default: qwen2.5-coder:7b)
  --target-fps FPS    # Target performance requirement

# wfe generate options
wfe generate "<description>" [OPTIONS]
  -i, --inputs LIST   # Comma-separated input names
  -o, --outputs LIST  # Comma-separated output names  
  -c, --category CAT  # Node category (atomic, custom, utils, etc.)
  -m, --model MODEL   # Ollama model to use
  --constraints LIST  # Implementation constraints
  --show-code        # Display generated code

# wfe optimize options
wfe optimize <workflow> [OPTIONS]
  --target FPS        # Target FPS performance
  --metric METRIC     # Optimization metric (fps, latency, memory)
  --strategy STRAT    # Optimization strategy (performance_first, balanced)
```

#### Workflow Creation Examples
```powershell
# Simple detection workflow
wfe create "real-time object detection using YOLO"

# Multi-backend comparison
wfe create "compare CPU vs NPU vs DirectML performance for image classification"

# Video processing pipeline
wfe create "analyze security camera footage, detect people and vehicles, save alerts"

# Batch processing workflow
wfe create "process folder of medical images, apply preprocessing, run inference"
```

#### Node Generation Examples
```powershell
# Image processing nodes
wfe generate "resize image maintaining aspect ratio" -i image,target_width,target_height -o resized_image

# ML preprocessing nodes
wfe generate "normalize tensor values to 0-1 range" -i tensor -o normalized_tensor -c atomic

# Utility nodes with constraints
wfe generate "save detection results to JSON with timestamp" -i detections,filename -o success --constraints "handle errors,create directories"

# Computer vision nodes
wfe generate "apply CLAHE histogram equalization" -i image -o enhanced_image -c atomic --constraints "use opencv"
```

#### Verification and Troubleshooting
```powershell
# Check if Ollama model is available
ollama list | grep qwen2.5-coder

# Test Ollama connection
ollama run qwen2.5-coder:7b "Hello, test connection"

# Verify WFE installation
wfe --help

# Check node generation capability
wfe generate "test node" --show-code

# Validate workflow before execution
uv run python workflow_orchestrator_minimal.py workflows/test.json --validate-only
```

---

## 5. 🔌 MCP Server Usage

The Model Context Protocol (MCP) Server enables AI assistants to interact with the workflow engine programmatically.

### Starting the MCP Server

#### Basic Server Startup
```powershell
# Start MCP server with default settings
wfe mcp

# Start with custom configuration
wfe mcp --port 8000 --host 0.0.0.0 --debug

# Start with specific model
wfe mcp --model qwen2.5-coder:14b --temperature 0.1
```

#### Direct Python Server Startup
```powershell
# Start MCP server directly
uv run python start_mcp.py

# Start with custom parameters
uv run python start_mcp.py --port 3001 --debug --log-level INFO
```

### MCP Server Capabilities

#### Available Tools
The MCP server exposes these tools to AI clients:

```json
{
  "tools": [
    {
      "name": "execute_workflow",
      "description": "Execute a workflow file",
      "parameters": {
        "workflow_path": "Path to workflow JSON file",
        "learn": "Enable learning mode (optional)",
        "options": "Additional execution options (optional)"
      }
    },
    {
      "name": "create_workflow", 
      "description": "Create workflow from natural language",
      "parameters": {
        "description": "Natural language description of workflow",
        "save_path": "Where to save the workflow (optional)",
        "execute": "Execute immediately after creation (optional)"
      }
    },
    {
      "name": "generate_node",
      "description": "Generate new workflow node from description", 
      "parameters": {
        "description": "What the node should do",
        "inputs": "Input parameters (optional)",
        "outputs": "Output parameters (optional)",
        "category": "Node category (optional)"
      }
    },
    {
      "name": "optimize_workflow",
      "description": "Get optimization suggestions for workflow",
      "parameters": {
        "workflow_path": "Path to workflow to optimize",
        "target_fps": "Target FPS performance (optional)",
        "metric": "Optimization metric (optional)"
      }
    },
    {
      "name": "list_workflows",
      "description": "List available workflow templates",
      "parameters": {}
    },
    {
      "name": "list_nodes", 
      "description": "List available workflow nodes",
      "parameters": {}
    },
    {
      "name": "detect_hardware",
      "description": "Detect available hardware capabilities",
      "parameters": {}
    },
    {
      "name": "get_performance_data",
      "description": "Get performance and learning data",
      "parameters": {
        "workflow_name": "Specific workflow name (optional)"
      }
    }
  ]
}
```

#### Server Configuration Options
```python
# MCP server configuration
{
    "server": {
        "name": "workflow-engine",
        "version": "1.0.0"
    },
    "capabilities": {
        "tools": True,
        "resources": True,
        "prompts": True
    },
    "settings": {
        "port": 3000,
        "host": "localhost", 
        "debug": False,
        "log_level": "INFO",
        "model_name": "qwen2.5-coder:7b",
        "temperature": 0.1,
        "max_tokens": 2000
    }
}
```

### MCP Server API Usage

#### Direct HTTP API (when running as HTTP server)
```powershell
# Execute workflow via HTTP
curl -X POST http://localhost:3000/execute \
  -H "Content-Type: application/json" \
  -d '{"workflow_path": "workflows/granular_parallel_inference.json", "learn": true}'

# Create workflow via HTTP  
curl -X POST http://localhost:3000/create \
  -H "Content-Type: application/json" \
  -d '{"description": "real-time object detection using NPU", "execute": false}'

# Get hardware info
curl http://localhost:3000/hardware

# List available workflows
curl http://localhost:3000/workflows
```

#### Server Management
```powershell
# Check server status
curl http://localhost:3000/status

# Get server capabilities
curl http://localhost:3000/capabilities  

# Health check
curl http://localhost:3000/health

# Shutdown server gracefully
curl -X POST http://localhost:3000/shutdown
```

### MCP Server Logging and Monitoring

#### Log Configuration
```powershell
# Start with detailed logging
wfe mcp --log-level DEBUG --log-file mcp_server.log

# Monitor server logs
tail -f mcp_server.log

# Check server performance
wfe mcp --profile --metrics-port 9090
```

#### Performance Monitoring
```python
# Monitor MCP server metrics
from src.mcp.server_metrics import MCPServerMetrics

metrics = MCPServerMetrics()
print(f"Requests handled: {metrics.get_request_count()}")
print(f"Average response time: {metrics.get_avg_response_time()}ms") 
print(f"Active connections: {metrics.get_active_connections()}")
print(f"Error rate: {metrics.get_error_rate()}%")
```

---

## 6. 🖥️ Claude Desktop Integration

### Prerequisites and Setup

#### Claude Desktop Installation
1. **Install Claude Desktop** (version 0.14.10 or later)
2. **Locate Configuration File**: 
   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`
   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Linux: `~/.config/claude/claude_desktop_config.json`

#### Configuration File Setup
```json
// claude_desktop_config.json
{
  "mcpServers": {
    "workflow-engine": {
      "command": "c:\\dev\\workflow_engine\\.venv\\Scripts\\python.exe",
      "args": ["c:\\dev\\workflow_engine\\start_mcp.py"],
      "env": {
        "PYTHONPATH": "c:\\dev\\workflow_engine",
        "LOG_LEVEL": "INFO"
      }
    }
  },
  "anthropic": {
    "apiKey": "your-api-key-here"
  }
}
```

#### Advanced Configuration Options
```json
{
  "mcpServers": {
    "workflow-engine": {
      "command": "c:\\dev\\workflow_engine\\.venv\\Scripts\\python.exe",
      "args": [
        "c:\\dev\\workflow_engine\\start_mcp.py",
        "--port", "3001",
        "--model", "qwen2.5-coder:14b",
        "--debug"
      ],
      "env": {
        "PYTHONPATH": "c:\\dev\\workflow_engine",
        "LOG_LEVEL": "DEBUG",
        "WORKFLOW_ENV_PATH": "c:\\dev\\workflow_engine\\workflow-envs",
        "MODEL_CACHE_PATH": "c:\\dev\\workflow_engine\\models"
      }
    }
  }
}
```

### Starting the Integration

#### Step-by-Step Setup
```powershell
# 1. Ensure MCP server works standalone
wfe mcp --debug

# 2. Test server connectivity
curl http://localhost:3000/status

# 3. Verify Claude Desktop config
cat "%APPDATA%\\Claude\\claude_desktop_config.json"

# 4. Restart Claude Desktop application

# 5. Check connection in Claude Desktop
# Look for "workflow-engine" in the server list
```

#### Verification Commands in Claude Desktop
Once connected, you can test these commands in Claude Desktop:

```
"List all available workflows"
"Show me the hardware capabilities of this system"
"Execute the granular parallel inference workflow"
"Create a workflow for real-time video analysis"
"Generate a node that applies gaussian blur to images"
"Optimize the granular workflow for better performance"
```

### Claude Desktop Usage Examples

#### Workflow Creation Commands
```
"Create a granular atomic node workflow for real-time object detection using NPU and DirectML with performance benchmarking"

"Build a video processing pipeline that reads MP4 files, extracts frames, runs YOLO detection, and saves results"

"Generate a batch processing workflow for medical image analysis using CPU and GPU backends"
```

#### Analysis and Optimization Commands  
```
"Analyze the performance of the granular parallel inference workflow and suggest optimizations"

"Compare the performance between CPU, NPU, and DirectML backends"

"Show me the learning data and performance trends from recent workflow executions"

"Optimize the workflow for maximum FPS on my hardware configuration"
```

#### Node Generation Commands
```
"Generate an atomic node that resizes images while maintaining aspect ratio"

"Create a utility node that saves detection results to a JSON file with timestamps"

"Build a preprocessing node that normalizes image pixel values and converts color spaces"
```

### Troubleshooting Claude Integration

#### Common Issues and Solutions

**1. MCP Server Not Found**
```powershell
# Check if server is running
ps aux | grep start_mcp.py

# Verify path in config
where python
# Update config with correct Python path
```

**2. Connection Errors**
```powershell
# Check server logs
wfe mcp --log-level DEBUG --log-file debug.log
tail -f debug.log

# Test direct connection
curl http://localhost:3000/status
```

**3. Performance Issues**
```powershell
# Start server with optimized settings
wfe mcp --model qwen2.5-coder:7b --temperature 0.1 --cache-size 1000

# Monitor resource usage
wfe mcp --profile --metrics-port 9090
```

**4. Model Issues**
```powershell
# Verify Ollama model availability
ollama list | grep qwen2.5-coder

# Pull model if missing
ollama pull qwen2.5-coder:7b

# Test model directly
ollama run qwen2.5-coder:7b "Hello test"
```

#### Debug Mode Setup
```json
// Enhanced debug configuration
{
  "mcpServers": {
    "workflow-engine": {
      "command": "c:\\dev\\workflow_engine\\.venv\\Scripts\\python.exe",
      "args": [
        "c:\\dev\\workflow_engine\\start_mcp.py", 
        "--debug",
        "--log-level", "DEBUG",
        "--log-file", "mcp_debug.log",
        "--trace-requests"
      ],
      "env": {
        "PYTHONPATH": "c:\\dev\\workflow_engine",
        "DEBUG": "1",
        "TRACE_EXECUTION": "1"
      }
    }
  }
}
```

---

## 7. 🎯 Workflow Examples and Templates

### Granular Parallel Inference Workflow

#### What Makes This Workflow Special

The **`granular_parallel_inference.json`** demonstrates the platform's core philosophy:

- **33+ Atomic Nodes** - Each operation is a separate, reusable component
- **Complex Dependency Orchestration** - 100+ automatic dependency resolutions
- **True Parallel Execution** - CPU, DirectML GPU, and NPU backends run simultaneously
- **Built-in Performance Benchmarking** - Comprehensive speed comparisons
- **Complete YOLO Pipeline** - From raw image to final detection results

### Granular Workflow Structure

```json
{
  "workflow": {
    "name": "Granular Multi-Backend YOLO Inference (CPU + NPU + DirectML)",
    "description": "Parallel inference comparison using atomic, composable nodes",
    "settings": {
      "max_parallel_nodes": 8
    }
  },
  "nodes": [
    // Image Processing Chain
    {
      "id": "download_model",
      "function": "src.nodes.download_model.download_model_node",
      "dependencies": [],
      "inputs": {"model_name": "yolov8s.onnx", "models_dir": "models"}
    },
    {
      "id": "read_img",
      "function": "src.nodes.image_ops.read_image_node", 
      "inputs": {"image_path": "input/soccer.jpg"},
      "dependencies": []
    },
    {
      "id": "resize",
      "function": "src.nodes.image_ops.resize_image_letterbox_node",
      "inputs": {"target_width": 640, "target_height": 640},
      "dependencies": ["read_img"]
    },
    // ... 30+ more nodes with complex interdependencies
  ]
}
```

### Granular Node Categories

#### **Image Operations**
- `read_img` → `resize` → `normalize` → `transpose` → `add_batch`
- Atomic preprocessing pipeline for YOLO input

#### **Backend Sessions** 
- `cpu_session` - ONNX Runtime CPU provider
- `directml_session` - DirectML GPU acceleration  
- `npu_session` - Intel OpenVINO NPU execution

#### **Parallel Benchmarking**
- `cpu_benchmark` - CPU inference with timing
- `directml_benchmark` - GPU inference with timing
- `npu_benchmark` - NPU inference with timing

#### **YOLO Post-Processing** (per backend)
- `decode` → `filter` → `convert` → `nms` → `scale` → `format` → `summary`
- Complete detection pipeline from raw outputs to formatted results

#### **Performance Comparison**
- `compare_performance` - Aggregates results from all backends
- Provides FPS comparison and recommendation

### Execution Flow Example

```
1. download_model ────┐
                      ├─→ cpu_session ────→ cpu_benchmark ────┐
2. read_img ────┐     ├─→ npu_session ────→ npu_benchmark ────┤
            │   │     └─→ directml_session → directml_benchmark ┘
            ├─→ resize                                        │
            │   │                                             │
            │   ├─→ normalize                                 │
            │   │   │                                         │
            │   │   ├─→ transpose                             │
            │   │   │   │                                     ├─→ parallel post-processing
            │   │   │   └─→ add_batch ─────────────────────┘     │
            │   │                                                 ├─→ compare_performance
            └─→ (used for scaling) ──────────────────────────────┘
```

---

## 🚀 Advanced Usage Patterns

### Granular Workflow Execution with Learning

```powershell
# Execute with agentic learning
wfe run workflows/granular_parallel_inference.json --learn

# Check what was learned
wfe status

# Get optimization suggestions based on learning
wfe optimize workflows/granular_parallel_inference.json --target 60
```

### Custom Granular Workflow Creation

```powershell
# Create granular video processing workflow
wfe create "frame-by-frame video analysis with atomic operations for real-time processing"

# Create granular batch processing workflow  
wfe create "atomic image preprocessing pipeline for training data preparation"

# Create granular multi-model workflow
wfe create "ensemble detection using multiple YOLO models with atomic voting"
```

### Node-Level Performance Analysis

```powershell
# Run with verbose logging to see individual node performance
uv run python workflow_orchestrator_minimal.py workflows/granular_parallel_inference.json --log-level DEBUG

# Profile individual node execution
uv run python -m cProfile -o profile.stats workflow_orchestrator_minimal.py workflows/granular_parallel_inference.json
```

### Hardware-Specific Optimization

```powershell
# Check available hardware first
wfe devices

# Create optimized workflow for detected hardware
wfe create "optimized detection for Intel NPU and DirectML GPU with load balancing"

# Get hardware-specific optimization suggestions
wfe optimize workflows/granular_parallel_inference.json --target 30
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

3. **Start MCP Server**:
```powershell
wfe mcp
```

### AI Commands for Granular Workflows

#### 🎯 **Granular Workflow Creation**
```
"Create a granular atomic node workflow for real-time object detection"
"Build a granular parallel inference pipeline using all available backends"
"Generate atomic preprocessing nodes for YOLO input preparation"
```

#### ⚡ **Execute and Analyze**
```
"Run the granular parallel inference workflow and analyze performance"
"Execute workflows/granular_parallel_inference.json with detailed benchmarking"
"Compare CPU vs GPU vs NPU performance using the granular workflow"
```

#### 📊 **Performance Optimization**
```
"Optimize the granular workflow for maximum FPS on my hardware"
"Suggest atomic node improvements for better parallel execution"
"Analyze bottlenecks in the granular inference pipeline"
```

---

## 📊 Performance Monitoring

### Granular Workflow Benchmarks

**Latest Performance Results (granular_parallel_inference.json)**:
- **🥇 DirectML GPU**: 452.1 FPS (2.2ms avg) - 34.73x faster than CPU
- **🥈 Intel NPU**: 44.2 FPS (22.6ms avg) - 3.39x faster than CPU
- **🥉 CPU**: 13.0 FPS (76.8ms avg) - Baseline performance

### Real-time Performance Tracking

```powershell
# Monitor granular workflow execution
wfe run workflows/granular_parallel_inference.json --learn

# View detailed performance breakdown
wfe status

# Get node-level timing analysis
uv run python workflow_orchestrator_minimal.py workflows/granular_parallel_inference.json --log-level INFO
```

### Performance Analysis Output

```
📊 Granular Workflow Performance:
   Total execution time: 2.34s
   Node execution breakdown:
   • Image preprocessing: 0.12s (5 nodes)
   • Model loading: 0.45s (3 nodes)  
   • Parallel inference: 0.89s (3 nodes)
   • Post-processing: 0.76s (15 nodes)
   • Performance comparison: 0.12s (1 node)

💡 Optimization suggestions:
   • GPU preprocessing: +25% speed improvement
   • Batch processing: +40% throughput for multiple images
   • Model caching: -60% startup time
```

---

## 🔧 Granular Node Development

### Generate Custom Atomic Nodes

```powershell
# Generate image processing node
wfe generate "convert RGB image to grayscale using weighted average" \
  -i rgb_image -o gray_image -c atomic

# Generate ML preprocessing node
wfe generate "apply CLAHE histogram equalization to enhance contrast" \
  -i image -o enhanced_image -c atomic --constraints "use opencv"

# Generate utility node
wfe generate "save detection results to JSON file with timestamp" \
  -i detections,filename -o success -c utils
```

### Custom Granular Workflow Template

```json
{
  "workflow": {
    "name": "Custom Granular Pipeline",
    "settings": {"max_parallel_nodes": 6}
  },
  "nodes": [
    {
      "id": "input_node",
      "function": "src.nodes.custom.my_input_node",
      "dependencies": []
    },
    {
      "id": "process_a", 
      "function": "src.nodes.custom.atomic_process_a",
      "dependencies": ["input_node"]
    },
    {
      "id": "process_b",
      "function": "src.nodes.custom.atomic_process_b", 
      "dependencies": ["input_node"]
    },
    {
      "id": "combine",
      "function": "src.nodes.custom.combine_results",
      "dependencies": ["process_a", "process_b"]
    }
  ]
}
```

---

## 🔍 Troubleshooting

### Common Granular Workflow Issues

#### Node Dependency Resolution
```powershell
# Check node dependencies
wfe nodes

# Validate workflow structure
uv run python -c "
import json
with open('workflows/granular_parallel_inference.json', 'r') as f:
    workflow = json.load(f)
    nodes = workflow.get('nodes', [])
    print(f'Granular workflow: {len(nodes)} nodes')
    print('Sample node IDs:', [node['id'] for node in nodes[:10]])
"
```

#### Performance Issues
```powershell
# Check hardware availability
wfe devices

# Verify backend support
uv run python -c "
try:
    import onnxruntime as ort
    print('ONNX providers:', ort.get_available_providers())
except ImportError:
    print('ONNXRuntime not installed')

try:
    import openvino as ov
    print('OpenVINO devices:', ov.Core().available_devices)
except ImportError:
    print('OpenVINO not installed')
"
```

#### Memory Issues with Granular Workflows
```powershell
# Monitor memory usage during execution
uv run python -c "
import psutil
print(f'Available RAM: {psutil.virtual_memory().available / 1024**3:.1f} GB')
print('Recommended: 16GB+ for smooth granular workflow execution')
"
```

### Debug Mode

```powershell
# Enable detailed logging for node execution
uv run python workflow_orchestrator_minimal.py workflows/granular_parallel_inference.json --log-level DEBUG

# Profile granular workflow performance
uv run python -m cProfile -o granular_profile.stats workflow_orchestrator_minimal.py workflows/granular_parallel_inference.json

# Analyze profile results
uv run python -c "
import pstats
stats = pstats.Stats('granular_profile.stats')
stats.sort_stats('cumulative').print_stats(20)
"
```

---

## 📞 Support & System Requirements

### Granular Workflow System Requirements
- **Python**: 3.11+ 
- **RAM**: 16GB recommended (granular workflows are memory-intensive)
- **Storage**: 4GB for models and dependencies
- **GPU**: Optional but recommended for DirectML acceleration
- **NPU**: Intel NPU for hardware-accelerated inference

### Performance Expectations
- **Granular workflows**: Higher overhead but maximum flexibility and parallelism
- **Atomic nodes**: Individual profiling and optimization possible
- **Dependency resolution**: Automatic parallel execution optimization
- **Benchmarking**: Built-in performance comparison across backends

---

## 8. 📞 Troubleshooting and Support

### Common Issues and Solutions

#### Environment Setup Issues
```powershell
# Python version check
python --version  # Should be 3.11+

# UV installation check
uv --version

# Dependencies check
uv sync
uv run python -c "import onnxruntime; print('ONNX OK')"
uv run python -c "import openvino; print('OpenVINO OK')" 
```

#### Hardware Detection Issues
```powershell
# GPU detection
wfe devices

# DirectML check
uv run python -c "
import onnxruntime as ort
print('Available providers:', ort.get_available_providers())
print('DirectML available:', 'DmlExecutionProvider' in ort.get_available_providers())
"

# NPU check  
uv run python -c "
import openvino as ov
core = ov.Core()
print('Available devices:', core.available_devices)
"
```

#### Workflow Execution Issues
```powershell
# Validate workflow structure
uv run python workflow_orchestrator_minimal.py workflows/test.json --validate-only

# Debug execution
uv run python workflow_orchestrator_minimal.py workflows/test.json --log-level DEBUG

# Check node dependencies
wfe nodes
```

#### MCP Server Issues
```powershell
# Test Ollama connection
ollama list
ollama run qwen2.5-coder:7b "test"

# MCP server debug
wfe mcp --debug --log-level DEBUG

# Claude Desktop config check
cat "%APPDATA%\\Claude\\claude_desktop_config.json"
```

### Performance Optimization Tips

1. **Use appropriate isolation**: DirectML needs subprocess isolation
2. **Monitor memory usage**: 16GB+ RAM recommended for granular workflows
3. **Enable learning mode**: Use `--learn` flag to improve performance over time
4. **Hardware-specific optimization**: Use `wfe optimize` for targeted improvements
5. **Batch processing**: Process multiple items together when possible

---

## 9. 📋 Complete Command Reference

### Core Engine Commands
| Command | Purpose | Example |
|---------|---------|---------|
| `uv run python workflow_orchestrator_minimal.py <file>` | Execute workflow directly | Basic workflow execution |
| `uv run python workflow_data_session.py` | Manage learning data | Performance tracking |
| `uv run python workflow_environment_manager.py` | Manage environments | Environment isolation |

### WFE CLI Commands  
| Command | Purpose | Example |
|---------|---------|---------|
| `wfe workflows` | List available workflows | Shows all workflow templates |
| `wfe run <workflow>` | Execute workflow | `wfe run workflows/granular_parallel_inference.json` |
| `wfe create "<description>"` | Generate from description | `wfe create "real-time NPU detection"` |
| `wfe optimize <workflow>` | Get performance suggestions | `wfe optimize <workflow> --target 60` |
| `wfe status` | Show agent learning | Displays performance trends |
| `wfe devices` | List hardware | Shows GPU, NPU, CPU availability |
| `wfe nodes` | List available nodes | Shows all atomic node functions |
| `wfe generate "<description>"` | Create custom nodes | `wfe generate "blur image" -c atomic` |
| `wfe mcp` | Start MCP server | For AI integration |
| `wfe demo` | Run interactive demos | Shows platform capabilities |

### Python API Classes
| Class | Purpose | Usage |
|-------|---------|-------|
| `WorkflowDataSession` | Learning and performance tracking | Session management and optimization |
| `WorkflowEnvironmentManager` | Environment isolation management | Create and manage isolated environments |
| `WorkflowAgent` | AI-powered workflow operations | Agentic workflow creation and optimization |
| `NodeGenerator` | AI-powered node generation | Generate custom nodes from descriptions |
| `MCPServer` | Model Context Protocol server | AI assistant integration |

### Getting Help
1. Use `wfe status` to check system state
2. Use `wfe devices` to verify hardware detection
3. Use `wfe nodes` to confirm node availability
4. Check logs in `logs/` directory for detailed execution traces
5. Enable debug logging for granular node-level analysis

---

*The granular workflow approach maximizes the platform's potential for complex, high-performance AI inference pipelines with full observability and optimization capabilities.*
