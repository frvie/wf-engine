# Workflow Engine - Agentic POC

Simple proof-of-concept demonstrating **AutoGen + Ollama + MCP** integration with the workflow engine.

## Architecture

```
User Request
     ↓
AutoGen Agents (Planning + Execution)
     ↓ (uses Ollama locally)
MCP Server Tools
     ↓
Workflow Engine
     ↓
DirectML/NPU/CUDA Inference
```

## Components

### 1. **MCP Server** (`mcp_server.py`)
Exposes workflow engine as standardized MCP tools:
- `execute_workflow` - Run workflows from JSON
- `detect_devices` - Check available hardware (DirectML, NPU, CUDA)
- `list_workflow_nodes` - List all available nodes
- `validate_workflow` - Validate workflow JSON
- `get_workflow_templates` - Get available templates

### 2. **AutoGen Agent** (`workflow_agent_poc.py`)
Multi-agent system for workflow orchestration:
- **Planner Agent** - Analyzes tasks, recommends workflows
- **Executor Agent** - Runs workflows, reports results
- **User Proxy** - Represents human user

### 3. **Ollama Integration**
All LLM inference runs locally:
- Model: `qwen2.5-coder:7b` (excellent for JSON generation)
- No cloud API calls
- Zero per-token costs
- Complete privacy

## Prerequisites

### 1. Install Ollama
```bash
# Download from: https://ollama.com
# Or on Windows with winget:
winget install Ollama.Ollama
```

### 2. Pull LLM Model
```bash
ollama pull qwen2.5-coder:7b
```

### 3. Install Python Dependencies
```bash
# Install MCP and AutoGen
uv pip install mcp pyautogen

# Or add to pyproject.toml:
# mcp>=1.0.0
# pyautogen>=0.2.0
```

## Quick Start

### Option 1: Simple Agent (No MCP Server)

```bash
# Run the AutoGen agent POC
uv run workflow_agent_poc.py
```

**Example:**
```python
from workflow_agent_poc import run_workflow_agent

# Ask agent to process video
run_workflow_agent("I want to detect objects in my webcam feed")
```

### Option 2: MCP Server (For Integration)

```bash
# Start MCP server
uv run mcp_server.py
```

The server exposes workflow tools via stdio for integration with:
- Claude Desktop
- VSCode extensions
- Custom MCP clients

## Example Usage

### 1. Webcam Object Detection
```python
from workflow_agent_poc import run_workflow_agent

run_workflow_agent("Start real-time object detection on my webcam")
```

**Agent Flow:**
1. Detects available devices (DirectML GPU found)
2. Recommends `video_detection.json` workflow
3. Executes workflow with DirectML acceleration
4. Reports results (FPS, detections, etc.)

### 2. Process MP4 Video
```python
run_workflow_agent("Process my dashcam video and detect vehicles")
```

**Agent Flow:**
1. Detects available devices
2. Recommends `video_detection_mp4.json`
3. Adjusts confidence threshold for vehicle detection
4. Executes and reports results

### 3. Multi-Backend Comparison
```python
run_workflow_agent("Compare inference performance across all backends")
```

**Agent Flow:**
1. Detects available devices (DirectML, NPU, CPU)
2. Recommends `parallel_yolov8.json`
3. Runs parallel inference
4. Reports performance comparison

## Agent Capabilities

### What the Agent Can Do:
✅ Understand natural language requests  
✅ Detect available hardware (GPU, NPU)  
✅ Select optimal workflow for task  
✅ Generate workflow JSON from description  
✅ Execute workflows automatically  
✅ Handle errors and suggest alternatives  
✅ Report results in human-friendly format  

### What's Still Manual (POC Limitations):
⚠️ Workflow generation uses templates (not full LLM generation)  
⚠️ No multi-turn conversation (single request)  
⚠️ Basic error handling  
⚠️ No workflow optimization/tuning  

## Testing the POC

### 1. Test Device Detection
```python
from workflow_agent_poc import detect_devices

devices = detect_devices()
print(devices)
# {'directml': True, 'cuda': False, 'npu': False, 'cpu': True}
```

### 2. Test Workflow Execution
```python
from workflow_agent_poc import execute_workflow

result = execute_workflow("workflows/video_detection.json")
print(result)
```

### 3. Test MCP Tools
```bash
# Start MCP server
uv run mcp_server.py

# In another terminal, test with MCP client
# (or integrate with Claude Desktop)
```

## MCP Server Configuration

### For Claude Desktop
Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "workflow-engine": {
      "command": "uv",
      "args": ["run", "mcp_server.py"],
      "cwd": "C:\\dev\\workflow_engine"
    }
  }
}
```

Now Claude Desktop can directly:
- Execute workflows
- Detect devices
- Validate configurations
- List available nodes

## Performance

### LLM Inference (Ollama - Local)
- Model: qwen2.5-coder:7b
- Speed: ~40 tokens/second
- Latency: ~1-2 seconds for workflow generation
- Memory: ~4GB VRAM

### Workflow Execution (Same as Before)
- DirectML: ~412 FPS inference
- NPU: ~232 FPS
- CPU: ~1,250 FPS (single image)

### Total E2E Latency
1. User request → Agent planning: **~2 seconds** (Ollama)
2. Workflow execution: **depends on task** (video processing, etc.)
3. Result reporting: **<1 second**

## Next Steps (Production)

### Phase 1: Enhanced Agent
- [ ] Full workflow JSON generation (not templates)
- [ ] Multi-turn conversation support
- [ ] Workflow parameter optimization
- [ ] Error recovery and retries

### Phase 2: Advanced MCP Integration
- [ ] Streaming results via MCP
- [ ] Workflow status monitoring
- [ ] Real-time performance metrics
- [ ] Resource usage tracking

### Phase 3: Multi-Agent Collaboration
- [ ] Separate agents for planning/execution/monitoring
- [ ] Agent-to-agent communication
- [ ] Consensus-based decision making
- [ ] Self-optimization

## Troubleshooting

### Ollama Not Running
```bash
# Start Ollama service
ollama serve

# In another terminal, verify:
ollama list
```

### Model Not Found
```bash
# Pull the model
ollama pull qwen2.5-coder:7b

# Verify it's available
ollama list
```

### Import Errors
```bash
# Install missing dependencies
uv pip install mcp pyautogen

# Or update pyproject.toml and sync
uv sync
```

## Architecture Benefits

### ✅ Privacy
- All AI reasoning happens locally (Ollama)
- No data sent to cloud
- Safe for sensitive videos

### ✅ Cost
- Zero API costs
- Unlimited usage
- One-time hardware investment

### ✅ Performance
- LLM inference on same GPU as workflows
- Low latency (local)
- Can run offline

### ✅ Integration
- MCP = standard protocol
- Works with Claude Desktop, VSCode, custom apps
- Easy to extend with new tools

### ✅ Flexibility
- Swap LLM models easily (Ollama)
- Add/remove agents
- Customize agent behavior

## License

Same as main workflow engine (see parent README.md)
