# Workflow Creation Guide

This guide explains how to create workflows and nodes using the workflow engine's various interfaces.

---

## Creating Workflows

### 1. CLI - Simple Creation (`wf create`)

The simplest way to create a workflow from natural language:

```bash
# Natural language prompt
wf create "detect objects in soccer.jpg using YOLOv8 on GPU"

# The system will:
# - Parse your goal
# - Choose the best strategy (image/video, backend)
# - Compose the workflow
# - Save to workflows/
# - Optionally execute immediately
```

### 2. CLI - With Options

```bash
# Specify output file
wf create "process video.mp4 with YOLO" -o my_workflow.json

# Execute immediately after creation
wf create "blur image.jpg" --execute

# Use specific strategy
wf create "detect faces" --strategy directml
```

### 3. MCP Tool - For AI Agents

When using AI agents like Claude Desktop or Cline:

```javascript
// The AI agent calls:
{
  "tool": "create_workflow_from_nl",
  "arguments": {
    "description": "detect objects in soccer.jpg using YOLOv8",
    "output_path": "workflows/my_workflow.json",
    "execute": false
  }
}
```

### 4. Python API - Programmatic

```python
from agentic_integration import AgenticWorkflowEngine

# Initialize engine
engine = AgenticWorkflowEngine()

# Create from natural language
result = engine.create_workflow_from_nl(
    description="apply gaussian blur to image.jpg",
    output_path="workflows/blur.json"
)

# Access the workflow
workflow_json = result['workflow']
```

---

## Creating Custom Nodes

### Node Generator (`wf generate`)

Create new workflow nodes on-demand:

```bash
# Generate a new node
wf generate "apply gaussian blur" -i image -o blurred

# With multiple inputs/outputs
wf generate "apply median filter" -i image -i kernel_size -o filtered

# Specify category
wf generate "detect faces" \
    -i image \
    -o faces \
    -c atomic

# Show generated code without saving
wf generate "enhance contrast" --show-code
```

**Generated nodes are:**
- ✅ Automatically discovered by the engine
- ✅ Immediately available in workflows
- ✅ Saved to `workflow_nodes/custom/`
- ✅ Follow atomic design principles

---

## Prompt Guidelines

### For Workflow Creation (`wf create`)

Your prompt should include:

- ✅ **Task**: What to do ("detect objects", "blur image", "process video")
- ✅ **Input**: File path or type
- ✅ **Backend** (optional): "using GPU", "on NPU", "with DirectML"
- ✅ **Model** (optional): "using YOLOv8"

**Good Examples:**
```bash
wf create "detect objects in input/soccer.jpg using YOLOv8 on CPU"
wf create "process video.mp4 with object detection"
wf create "apply gaussian blur to test.jpg"
wf create "run YOLO inference on webcam.mp4 using DirectML"
```

**What the system understands:**
- Input types: `.jpg`, `.png`, `.mp4`, `.avi`
- Backends: `CPU`, `GPU`, `NPU`, `DirectML`, `CUDA`
- Operations: `detect`, `blur`, `process`, `analyze`, `inference`
- Models: `YOLOv8`, `YOLO`, `object detection`

### For Node Generation (`wf generate`)

Your prompt should describe the operation:

- ✅ **Operation**: What the node does ("apply median filter", "enhance contrast")
- ✅ **Inputs**: `-i image -i threshold`
- ✅ **Outputs**: `-o filtered -o mask`
- ✅ **Category** (optional): `-c atomic` or `-c custom`

**Good Examples:**
```bash
wf generate "apply median filter" -i image -o filtered
wf generate "enhance contrast using CLAHE" -i image -i clip_limit -o enhanced
wf generate "convert to grayscale" -i image -o gray
wf generate "detect edges with Canny" -i image -i threshold1 -i threshold2 -o edges
```

---

## Command Comparison

| Command | Purpose | Output | When to Use |
|---------|---------|--------|-------------|
| `wf create "<prompt>"` | Create complete **workflow** | JSON workflow file | Need end-to-end processing pipeline |
| `wf generate "<prompt>"` | Create new **node** | Python node file | Need a specific operation not in library |
| `wf run <workflow.json>` | Execute workflow | Processing results | Have workflow, want to run it |
| `wf optimize <workflow.json>` | Tune parameters | Optimized workflow | Want better performance |

---

## Workflow JSON Format

If you want to manually create or edit workflows, here's the format:

```json
{
  "workflow": {
    "name": "my_workflow",
    "description": "Description of what it does",
    "strategy": "custom"
  },
  "nodes": [
    {
      "id": "read",
      "function": "workflow_nodes.atomic.image_ops.read_image_node",
      "inputs": {
        "image_path": "input/image.jpg"
      },
      "dependencies": []
    },
    {
      "id": "process",
      "function": "workflow_nodes.custom.my_custom_node.my_custom_node",
      "inputs": {
        "param1": 5
      },
      "dependencies": ["read"]
    },
    {
      "id": "save",
      "function": "workflow_nodes.atomic.image_ops.save_image_node",
      "inputs": {
        "output_path": "output/result.jpg"
      },
      "dependencies": ["process"]
    }
  ]
}
```

**Key fields:**
- `id`: Unique identifier for the node
- `function`: Full module path to the node function
- `inputs`: Explicit input parameters
- `dependencies`: Array of node IDs that must run before this node

**Data flow:**
- Use `dependencies` array for automatic parameter passing
- Engine auto-injects outputs from dependent nodes as inputs
- Example: `process` receives `image` from `read` automatically

---

## Examples by Use Case

### Image Processing
```bash
# Blur an image
wf create "apply gaussian blur to input/photo.jpg"

# Detect objects
wf create "detect objects in soccer.jpg using YOLOv8"

# Create custom filter
wf generate "apply sepia filter" -i image -o sepia
```

### Video Processing
```bash
# Process video with object detection
wf create "detect objects in video.mp4 using YOLOv8 on DirectML"

# Use optimized backend
wf create "process webcam.mp4 with YOLO on NPU"
```

### Custom Operations
```bash
# Create a new image processing node
wf generate "apply bilateral filter" -i image -i sigma -o filtered

# Create a detection post-processor
wf generate "filter small detections" -i detections -i min_size -o filtered

# Create a visualization node
wf generate "draw bounding boxes" -i image -i boxes -o annotated
```

---

## Tips & Best Practices

### Workflow Creation
1. **Be specific** about input files and backends
2. **Use full paths** or put files in `input/` directory
3. **Test with `--execute`** to validate immediately
4. **Check output** in `workflows/` directory

### Node Generation
1. **Follow atomic principles** - one operation per node
2. **Return dicts** with descriptive keys (e.g., `{"image": result}`)
3. **Use type hints** in function signatures
4. **Test generated nodes** with `wf run` before production use

### Performance
1. **Use `wf devices`** to see available backends
2. **Optimize workflows** with `wf optimize <workflow.json>`
3. **Monitor execution** with performance stats nodes
4. **Choose backends wisely**: NPU for efficiency, GPU for speed, CPU for compatibility

---

## Troubleshooting

### "Node not found"
- Check node exists: `wf nodes | grep <node_name>`
- Verify correct function path in workflow JSON
- Regenerate node if needed: `wf generate "<description>"`

### "Workflow failed to create"
- Make prompt more specific
- Ensure Ollama is running (for LLM features)
- Check available nodes: `wf nodes`

### "Execution failed"
- Verify input file exists
- Check workflow JSON format
- Review logs for specific errors
- Test with simpler workflow first

---

## Next Steps

1. **List available nodes**: `wf nodes`
2. **See example workflows**: `wf templates`
3. **Check system capabilities**: `wf devices`
4. **Create your first workflow**: `wf create "your task here"`
5. **Generate custom nodes**: `wf generate "your operation"`

For more details, see:
- `CLI_GUIDE.md` - Complete CLI reference
- `NODE_GENERATOR.md` - Node generator documentation
- `ARCHITECTURE.md` - System architecture overview
