# Granular Workflow Engine Usage Guide

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

### 2. Run the Granular Workflow

```powershell
# Execute granular parallel inference (recommended)
uv run python wf.py run workflows/granular_parallel_inference.json

# Or use the workflow orchestrator directly
uv run python workflow_orchestrator_minimal.py workflows/granular_parallel_inference.json
```

---

## 🔧 Workflow CLI (`wf.py`)

The `wf.py` command-line interface provides simplified access to all workflow operations:

### Core Commands

#### 📋 **List Available Templates**
```powershell
# See all available workflows
wf templates

# Output shows:
# • granular_parallel_inference.json
#   Strategy: parallel, Nodes: 33
# • multi_backend_separated_v8.json  
#   Strategy: parallel, Nodes: 15
```

#### ⚡ **Execute Workflows**
```powershell
# Run the main granular workflow
wf run workflows/granular_parallel_inference.json

# Run with agentic learning enabled
wf run workflows/granular_parallel_inference.json --learn

# Quick performance test with simpler workflow
wf run workflows/multi_backend_separated_v8.json
```

#### 🧠 **Create Workflows from Natural Language**
```powershell
# Create and save workflow
wf create "real-time object detection using NPU and DirectML"

# Create and run immediately
wf create "fast video processing for dashcam footage" --run

# Save to specific location
wf create "batch image analysis pipeline" -o workflows/custom_batch.json
```

#### 📊 **Performance Analysis**
```powershell
# Get optimization suggestions
wf optimize workflows/granular_parallel_inference.json --target 30

# Check agent status and learning history
wf status

# View performance trends and recommendations
```

#### 💻 **Hardware Detection**
```powershell
# List available inference devices
wf devices

# Output shows:
# GPU Devices: NVIDIA RTX 4080 (16 GB)
# ✅ DirectML: Available (Device ID: 0)
# ✅ OpenVINO NPU: Available (Intel NPU)
# ✅ CUDA: Available (Device ID: 0)
```

#### 🔧 **Node Management**
```powershell
# List all available workflow nodes
wf nodes

# Generate new nodes from description
wf generate "apply gaussian blur to image" -i image -o blurred_image -c atomic

# Generate complex processing node
wf generate "extract SIFT features from grayscale image" -i image -o features -c utils
```

#### 🤖 **AI Agent Integration**
```powershell
# Start MCP server for Claude Desktop
wf mcp

# Run interactive demos
wf demo
wf demo --llm
```

### Advanced CLI Usage

#### Custom Workflow Creation
```powershell
# Create workflow with specific requirements
wf create "real-time detection optimized for Intel NPU with 60 FPS target"

# Create batch processing workflow
wf create "process folder of security camera images, detect people and vehicles"

# Create video analysis pipeline
wf create "analyze dashcam video, detect traffic signs and lane markings"
```

#### Performance Optimization
```powershell
# Analyze specific workflow performance
wf optimize workflows/granular_parallel_inference.json --target 45

# Check what the agent has learned
wf status

# Example output:
# 📚 Knowledge Base:
#    Total executions: 12
#    Workflow types learned: 3
#    • granular_parallel: Avg FPS: 89.2, Best FPS: 258.0
#    • multi_backend: Avg FPS: 45.1, Best FPS: 67.3
```

#### Node Generation Examples
```powershell
# Generate image processing nodes
wf generate "resize image maintaining aspect ratio" -i image,target_size -o resized_image

# Generate ML preprocessing node  
wf generate "normalize tensor values to 0-1 range" -i tensor -o normalized_tensor -c atomic

# Generate utility node with constraints
wf generate "convert video to frames" -i video_path -o frame_list --constraints "use opencv,efficient memory usage"
```

---

## 🎯 Granular Workflow Deep Dive

### What Makes Granular Workflows Special

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
wf run workflows/granular_parallel_inference.json --learn

# Check what was learned
wf status

# Get optimization suggestions based on learning
wf optimize workflows/granular_parallel_inference.json --target 60
```

### Custom Granular Workflow Creation

```powershell
# Create granular video processing workflow
wf create "frame-by-frame video analysis with atomic operations for real-time processing"

# Create granular batch processing workflow  
wf create "atomic image preprocessing pipeline for training data preparation"

# Create granular multi-model workflow
wf create "ensemble detection using multiple YOLO models with atomic voting"
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
wf devices

# Create optimized workflow for detected hardware
wf create "optimized detection for Intel NPU and DirectML GPU with load balancing"

# Get hardware-specific optimization suggestions
wf optimize workflows/granular_parallel_inference.json --target 30
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
wf mcp
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

**Typical Performance (granular_parallel_inference.json)**:
- **DirectML GPU**: ~258 FPS
- **Intel NPU**: ~35 FPS  
- **CPU**: ~33 FPS

### Real-time Performance Tracking

```powershell
# Monitor granular workflow execution
wf run workflows/granular_parallel_inference.json --learn

# View detailed performance breakdown
wf status

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
wf generate "convert RGB image to grayscale using weighted average" \
  -i rgb_image -o gray_image -c atomic

# Generate ML preprocessing node
wf generate "apply CLAHE histogram equalization to enhance contrast" \
  -i image -o enhanced_image -c atomic --constraints "use opencv"

# Generate utility node
wf generate "save detection results to JSON file with timestamp" \
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
wf nodes

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
wf devices

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

### CLI Command Reference

| Command | Purpose | Example |
|---------|---------|---------|
| `wf templates` | List available workflows | Shows granular and simple workflows |
| `wf run` | Execute workflow | `wf run workflows/granular_parallel_inference.json` |
| `wf create` | Generate from description | `wf create "real-time NPU detection"` |
| `wf optimize` | Get performance suggestions | `wf optimize <workflow> --target 30` |
| `wf status` | Show agent learning | Displays performance trends |
| `wf devices` | List hardware | Shows GPU, NPU, CPU availability |
| `wf nodes` | List available nodes | Shows all atomic node functions |
| `wf generate` | Create custom nodes | `wf generate "blur image" -c atomic` |
| `wf mcp` | Start AI agent server | For Claude Desktop integration |
| `wf demo` | Run interactive demos | Shows platform capabilities |

### Getting Help
1. Use `wf status` to check system state
2. Use `wf devices` to verify hardware detection
3. Use `wf nodes` to confirm node availability
4. Check logs in `logs/` directory for detailed execution traces
5. Enable debug logging for granular node-level analysis

---

*The granular workflow approach maximizes the platform's potential for complex, high-performance AI inference pipelines with full observability and optimization capabilities.*