# Custom Workflow Nodes

This directory contains automatically generated workflow nodes.

## Node Generation

Nodes in this directory are created using the **Node Generator** system, which uses LLM-powered code generation to create new workflow nodes from natural language descriptions.

### Generating New Nodes

#### Using CLI

```bash
# Basic generation
python wf.py generate "apply median filter to image" -i image -o filtered

# Specify inputs and outputs
python wf.py generate "resize image to target size" -i image,target_width,target_height -o resized_image

# Choose category
python wf.py generate "detect faces in image" -i image -o faces -c atomic

# Add constraints
python wf.py generate "enhance image contrast" --constraints "use CLAHE,preserve colors,handle grayscale"

# Show generated code
python wf.py generate "apply sepia tone filter" --show-code
```

#### Using Python API

```python
from workflow_nodes.generator.node_generator import NodeGenerator, NodeSpec
import asyncio

async def create_custom_node():
    generator = NodeGenerator(model_name="qwen2.5-coder:7b")
    
    spec = NodeSpec(
        goal="Apply artistic oil painting effect",
        inputs=["image"],
        outputs=["artistic_image"],
        category="custom",
        description="Transform image to look like an oil painting",
        constraints=["Use bilateral filter", "Adjust color saturation"]
    )
    
    result = await generator.generate_node(spec)
    
    if result['success']:
        print(f"Generated: {result['node_name']}")
        print(f"Location: {result['file_path']}")

asyncio.run(create_custom_node())
```

## Features

All generated nodes automatically:
- ✅ Use `@workflow_node` decorator
- ✅ Follow atomic/granular design principles
- ✅ Include comprehensive docstrings
- ✅ Handle errors gracefully
- ✅ Are immediately discoverable by the workflow engine
- ✅ Can be used in workflows right away

## Node Categories

Generated nodes can be saved to different categories:

- **custom** (default) - Custom nodes in `workflow_nodes/custom/`
- **atomic** - Atomic operations in `workflow_nodes/atomic/`
- **infrastructure** - Infrastructure nodes
- **utils** - Utility nodes
- **video** - Video processing nodes

## Examples

### Example 1: Image Filter Node

```bash
python wf.py generate "sharpen image using unsharp mask" \
    -i image,amount,radius \
    -o sharpened_image \
    -c atomic
```

### Example 2: Data Transformation Node

```bash
python wf.py generate "normalize bounding boxes to 0-1 range" \
    -i boxes,image_width,image_height \
    -o normalized_boxes \
    -c custom
```

### Example 3: Custom Analytics Node

```bash
python wf.py generate "calculate image quality metrics" \
    -i image \
    -o brightness,contrast,sharpness \
    --constraints "use numpy,return multiple metrics"
```

## Integration with Workflows

Generated nodes are automatically discovered and can be used immediately:

```json
{
  "workflow": {
    "name": "custom_pipeline",
    "strategy": "custom"
  },
  "nodes": [
    {
      "id": "filter",
      "type": "apply_gaussian_blur",
      "params": {
        "image": "$input.image",
        "kernel_size": 5
      }
    },
    {
      "id": "detect",
      "type": "detect_edges_using_canny",
      "params": {
        "image": "$filter.blurred_image"
      }
    }
  ]
}
```

## Validation

All generated nodes are validated for:
- Python syntax correctness
- Proper decorator usage
- Correct import statements
- Function signature matches specification

## LLM Models

Supported Ollama models for generation:
- `qwen2.5-coder:7b` (recommended)
- `deepseek-r1:7b`
- `deepseek-r1:14b`
- `llama3.3:70b`

## See Also

- [Node Generator Documentation](../docs/NODE_GENERATOR.md)
- [CLI Guide](../../CLI_GUIDE.md)
- [Architecture](../../ARCHITECTURE.md)
