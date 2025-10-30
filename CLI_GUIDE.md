# Workflow Engine CLI - Quick Reference

## 🚀 Quick Start

```bash
# Simple usage
python wf.py <command> [options]

# Or use the batch file (Windows)
wf <command> [options]
```

---

## 📋 Available Commands

### **1. Run Workflows**
```bash
# Execute a workflow
python wf.py run workflows/granular_parallel_inference.json

# Execute with learning enabled (agent learns from execution)
python wf.py run workflows/my_workflow.json --learn
```

### **2. Create Workflows from Natural Language**
```bash
# Rule-based creation (fast, no LLM needed)
python wf.py create "fast video detection with NPU"

# LLM-powered creation (requires Ollama)
python wf.py create "detect objects in dashcam footage" --llm

# Create and run immediately
python wf.py create "real-time object tracking" --llm --run

# Save to specific path
python wf.py create "video analysis pipeline" -o workflows/custom.json
```

### **3. Optimize Workflows**
```bash
# Get optimization suggestions
python wf.py optimize workflows/my_workflow.json

# Target specific FPS
python wf.py optimize workflows/my_workflow.json --target 30
```

### **4. Check System Status**
```bash
# View agent knowledge base and performance history
python wf.py status
```

**Shows:**
- Total executions
- Learned workflow types
- Average FPS per strategy
- Success rates
- Active recommendations
- LLM availability (Ollama status)

### **5. List Available Hardware**
```bash
# Detect GPUs, NPU, and other devices
python wf.py devices
```

**Shows:**
- NVIDIA GPUs (RTX 5090, etc.)
- DirectML device IDs
- CUDA device IDs
- OpenVINO NPU availability
- CPU (always available)

### **6. List Available Nodes**
```bash
# Show all workflow nodes
python wf.py nodes
```

**Categories:**
- Infrastructure (GPU detection, model download)
- Model Loaders (OpenVINO, ONNX)
- Inference (DirectML, NPU, CPU)
- Atomic Operations (YOLO post-processing)
- Utilities (performance stats, logging)

### **7. List Templates**
```bash
# Show available workflow templates
python wf.py templates
```

### **8. Start MCP Server**
```bash
# Start MCP server for AI agent integration
python wf.py mcp
```

**Use with:**
- Claude Desktop
- Cline
- Any MCP-compatible AI client

### **9. Run Demos**
```bash
# Run basic agentic system demo
python wf.py demo

# Run LLM + MCP integration demo
python wf.py demo --llm
```

---

## 💡 Usage Examples

### **Example 1: Quick Workflow Execution**
```bash
# Just run it!
python wf.py run workflows/granular_parallel_inference.json
```

### **Example 2: Create & Optimize Workflow**
```bash
# 1. Create workflow
python wf.py create "fast NPU-based detection" -o workflows/npu_fast.json

# 2. Run with learning
python wf.py run workflows/npu_fast.json --learn

# 3. Get optimization suggestions
python wf.py optimize workflows/npu_fast.json --target 40

# 4. Check what the agent learned
python wf.py status
```

### **Example 3: AI-Powered Workflow Design**
```bash
# Use LLM for complex requirements
python wf.py create "Create a flexible pipeline for dashcam footage analysis with good performance and the ability to customize detection parameters" --llm --run
```

### **Example 4: Hardware-Aware Development**
```bash
# 1. Check available hardware
python wf.py devices

# 2. Create workflow targeting specific hardware
python wf.py create "video detection using DirectML on RTX 5090" --llm

# 3. Optimize for your GPU
python wf.py optimize workflows/my_workflow.json --target 200
```

### **Example 5: Iterative Improvement**
```bash
# Run multiple times to build knowledge base
python wf.py run workflows/test1.json --learn
python wf.py run workflows/test2.json --learn
python wf.py run workflows/test3.json --learn

# Agent learns optimal parameters
python wf.py status

# Get learned suggestions
python wf.py optimize workflows/test1.json
```

---

## 🔥 Current Performance

Your system achieves:

```
🥇 DirectML (RTX 5090): 178-240 FPS  (10x faster than CPU)
🥈 OpenVINO NPU:        39-40 FPS    (2.2x faster than CPU)
🥉 CPU (ONNX Runtime):  18-30 FPS    (baseline)
```

---

## 🧠 Agentic Features

The system **learns automatically** when you use `--learn`:

**What it learns:**
- Which strategies work best for your hardware
- Optimal parameters for target FPS
- Success rates per workflow type
- Performance trends over time

**What it provides:**
- Optimization suggestions
- Performance predictions
- Hardware-aware recommendations
- Adaptive parameter tuning

---

## 🎯 Quick Command Reference

| Command | What it Does | Example |
|---------|-------------|---------|
| `run` | Execute workflow | `wf run workflows/my.json` |
| `create` | Generate from description | `wf create "fast detection" --llm` |
| `optimize` | Get suggestions | `wf optimize workflows/my.json` |
| `status` | Show agent knowledge | `wf status` |
| `devices` | List hardware | `wf devices` |
| `nodes` | List available nodes | `wf nodes` |
| `templates` | List workflow templates | `wf templates` |
| `mcp` | Start MCP server | `wf mcp` |
| `demo` | Run interactive demo | `wf demo --llm` |

---

## 🚨 Tips & Tricks

### **Tip 1: Always check status after runs**
```bash
python wf.py run my_workflow.json --learn
python wf.py status  # See what changed!
```

### **Tip 2: Use LLM for complex requirements**
```bash
# LLM understands nuance
python wf.py create "I need a pipeline that prioritizes accuracy over speed, uses NPU if available, otherwise falls back to GPU, and can process dashcam videos at moderate quality" --llm
```

### **Tip 3: Target specific FPS**
```bash
python wf.py optimize workflows/my.json --target 60
```

### **Tip 4: Check hardware before creating workflows**
```bash
python wf.py devices
# Then design workflow for your hardware
```

### **Tip 5: Use batch file for convenience (Windows)**
```bash
# Instead of 'python wf.py'
wf status
wf devices
wf run workflows/test.json
```

---

## 🔌 MCP Integration

To use with **Claude Desktop**:

1. Start MCP server:
   ```bash
   python wf.py mcp
   ```

2. Add to `claude_desktop_config.json`:
   ```json
   {
     "mcpServers": {
       "workflow-engine": {
         "command": "python",
         "args": ["c:/dev/workflow_engine/wf.py", "mcp"]
       }
     }
   }
   ```

3. Ask Claude:
   - "What hardware do I have?"
   - "Create a fast video detection workflow"
   - "Optimize my workflow for 30 FPS"
   - "Analyze my performance history"

---

## 📦 Requirements

- **Python 3.8+**
- **Ollama** (optional, for LLM features): `ollama serve`
- **Model** (optional): `ollama pull qwen2.5-coder:7b`

**Without Ollama:**
- Rule-based composition still works
- All other features fully functional
- Just no LLM-powered natural language understanding

---

## ✨ Next Steps

1. **Run your first workflow:**
   ```bash
   python wf.py run workflows/granular_parallel_inference.json --learn
   ```

2. **Check what the agent knows:**
   ```bash
   python wf.py status
   ```

3. **Create a custom workflow:**
   ```bash
   python wf.py create "your requirements here" --llm --run
   ```

4. **Keep improving:**
   ```bash
   # The more you run, the smarter it gets!
   python wf.py optimize workflows/my.json
   ```

---

**Happy workflow building! 🚀**
