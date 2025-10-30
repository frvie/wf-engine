# Workflow Node Generator

## Overview

The **Node Generator** system enables automatic creation of workflow nodes using LLM-powered code generation. It allows you to describe what you want a node to do in natural language, and it generates production-ready Python code that follows all workflow engine conventions.

## Architecture

```
workflow_nodes/
├── generator/
│   ├── __init__.py
│   └── node_generator.py          # Core generator logic
├── custom/                         # Generated custom nodes
│   ├── README.md
│   └── <generated_nodes>.py
└── atomic/                         # Can generate here too
    └── <generated_atomic_nodes>.py
```

## Features

### ✅ LLM-Powered Generation
- Uses Ollama models (qwen2.5-coder:7b recommended)
- Generates clean, idiomatic Python code
- Follows atomic design principles automatically

### ✅ Automatic Validation
- Syntax checking with AST parsing
- Decorator verification
- Import statement validation
- Type signature validation

### ✅ Granular Design Enforcement
- Single responsibility principle
- No side effects
- Pure functions when possible
- Proper error handling

### ✅ Immediate Integration
- Auto-discovered by workflow engine
- Available in workflows immediately
- No manual registration needed

## Usage

### CLI Interface

#### Basic Generation
```bash
python wf.py generate "apply gaussian blur to image" \
    -i image \
    -o blurred_image
```

#### Specify Category
```bash
python wf.py generate "detect edges with Canny" \
    -i image \
    -o edges \
    -c atomic
```

#### Multiple Inputs/Outputs
```bash
python wf.py generate "blend two images" \
    -i image1,image2,alpha \
    -o blended_image \
    -c custom
```

#### Add Constraints
```bash
python wf.py generate "enhance image" \
    --constraints "use CLAHE,preserve colors,handle grayscale" \
    --show-code
```

#### Choose LLM Model
```bash
python wf.py generate "complex transformation" \
    -m deepseek-r1:14b
```

### Python API

#### Simple Generation
```python
from workflow_nodes.generator.node_generator import NodeGenerator, NodeSpec
import asyncio

async def main():
    generator = NodeGenerator()
    
    spec = NodeSpec(
        goal="Apply median filter to remove noise",
        inputs=["image", "kernel_size"],
        outputs=["filtered_image"],
        category="atomic"
    )
    
    result = await generator.generate_node(spec)
    
    if result['success']:
        print(f"✅ Generated: {result['node_name']}")
        print(f"   File: {result['file_path']}")
    else:
        print(f"❌ Error: {result['error']}")

asyncio.run(main())
```

#### Advanced Generation with Constraints
```python
spec = NodeSpec(
    goal="Create artistic filter combining multiple effects",
    inputs=["image"],
    outputs=["artistic_image"],
    category="custom",
    description="Apply oil painting effect with edge enhancement",
    constraints=[
        "Use bilateral filter for base effect",
        "Detect edges separately",
        "Blend using weighted combination",
        "Preserve image dimensions",
        "Handle both RGB and grayscale"
    ],
    examples=[
        {
            "input": "RGB image (1920x1080)",
            "output": "Artistic filtered image (1920x1080)"
        }
    ]
)

result = await generator.generate_node(spec)
```

## Generated Node Structure

### Example Output

Input specification:
```python
NodeSpec(
    goal="Apply Gaussian blur to smooth image",
    inputs=["image", "kernel_size"],
    outputs=["blurred_image"],
    category="custom"
)
```

Generated code:
```python
from workflow_decorator import workflow_node
import numpy as np
import cv2

@workflow_node(
    name="apply_gaussian_blur",
    outputs=["blurred_image"]
)
def apply_gaussian_blur_node(image, kernel_size, **kwargs):
    """
    Apply Gaussian blur to smooth an image.
    
    This is an atomic workflow node following granular design principles.
    Single responsibility, no side effects, composable.
    
    Args:
        image: Input image (numpy array, BGR or grayscale)
        kernel_size: Size of Gaussian kernel (must be odd, e.g., 3, 5, 7)
        **kwargs: Additional optional parameters
    
    Returns:
        blurred_image: Smoothed image (numpy array, same shape as input)
    """
    
    try:
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
        return blurred
        
    except Exception as e:
        logger.error(f"Gaussian blur failed: {e}")
        # Return original image on error
        return image
```

## Integration with Workflow System

### 1. Automatic Discovery

Once generated, nodes are immediately available:

```bash
# List all nodes (includes newly generated)
python wf.py nodes
```

### 2. Use in Workflows

```json
{
  "workflow": {
    "name": "custom_image_pipeline",
    "strategy": "custom"
  },
  "nodes": [
    {
      "id": "blur",
      "type": "apply_gaussian_blur",
      "params": {
        "image": "$input.image",
        "kernel_size": 7
      }
    },
    {
      "id": "edges",
      "type": "detect_edges_with_canny",
      "params": {
        "image": "$blur.blurred_image"
      }
    }
  ]
}
```

### 3. Compose with WorkflowComposer

The agentic system automatically discovers and can use generated nodes:

```python
from workflow_agent import WorkflowComposer

composer = WorkflowComposer()
composer._discover_nodes()  # Finds ALL nodes including generated ones

# Now composer can use your custom nodes in workflows!
```

## Validation & Quality Assurance

### Automatic Checks

1. **Syntax Validation**
   - AST parsing for syntax errors
   - Python version compatibility

2. **Decorator Verification**
   - Ensures `@workflow_node` is present
   - Validates decorator parameters

3. **Import Validation**
   - Checks for required imports
   - Adds missing `workflow_decorator` import

4. **Signature Validation**
   - Matches function signature to spec
   - Verifies input/output parameters

### Manual Review

Generated code can be reviewed:

```bash
# Show code before saving
python wf.py generate "my node" --show-code

# Review saved file
cat workflow_nodes/custom/my_node.py
```

## Best Practices

### 1. Clear Goal Descriptions
```bash
# ✅ Good
python wf.py generate "detect faces using Haar cascades and return bounding boxes"

# ❌ Too vague
python wf.py generate "process image"
```

### 2. Specify Input/Output Types in Description
```bash
python wf.py generate "resize image to target dimensions" \
    -i "image:numpy_array,width:int,height:int" \
    -o "resized:numpy_array"
```

### 3. Use Constraints for Complex Nodes
```bash
python wf.py generate "advanced color correction" \
    --constraints "preserve alpha channel,handle 16-bit,use LAB color space"
```

### 4. Provide Examples for Complex Logic
```python
spec = NodeSpec(
    goal="Complex transformation",
    inputs=["data"],
    outputs=["result"],
    examples=[
        {"input": "[1,2,3]", "output": "[2,4,6]"},
        {"input": "[4,5,6]", "output": "[8,10,12]"}
    ]
)
```

## LLM Models

### Recommended Models

1. **qwen2.5-coder:7b** (default)
   - Best for code generation
   - Fast and accurate
   - Good Python knowledge

2. **deepseek-r1:14b**
   - More advanced reasoning
   - Better for complex nodes
   - Slower but more thorough

3. **llama3.3:70b**
   - Highest quality
   - Best for production code
   - Requires more resources

### Model Selection

```bash
# Use different model
python wf.py generate "complex operation" -m deepseek-r1:14b
```

## Troubleshooting

### Ollama Not Running

**Error:** `Connection refused to localhost:11434`

**Solution:**
```bash
# Start Ollama
ollama serve

# In another terminal, verify
ollama list
```

### Model Not Found

**Error:** `Model 'qwen2.5-coder:7b' not found`

**Solution:**
```bash
# Pull the model
ollama pull qwen2.5-coder:7b
```

### Generated Code Has Errors

**Solution:**
1. Check generated code: `--show-code`
2. Try different model: `-m deepseek-r1:14b`
3. Add more constraints: `--constraints "specific requirements"`
4. Manually edit: `workflow_nodes/custom/<node>.py`

### Node Not Discovered

**Solution:**
```bash
# Verify file exists
ls workflow_nodes/custom/

# Check syntax
python -m py_compile workflow_nodes/custom/<node>.py

# Restart workflow engine (clears cache)
```

## Examples

### Example 1: Image Processing Node

```bash
python wf.py generate \
    "apply bilateral filter to preserve edges while smoothing" \
    -i image,d,sigma_color,sigma_space \
    -o filtered_image \
    -c atomic \
    --show-code
```

### Example 2: Data Transformation Node

```bash
python wf.py generate \
    "convert YOLO detections to COCO format" \
    -i detections,image_width,image_height \
    -o coco_annotations \
    -c custom \
    --constraints "normalize coordinates,add category ids,generate timestamps"
```

### Example 3: Utility Node

```bash
python wf.py generate \
    "calculate IoU between two bounding boxes" \
    -i box1,box2 \
    -o iou_score \
    -c utils
```

### Example 4: Video Processing Node

```bash
python wf.py generate \
    "extract keyframes from video based on scene changes" \
    -i video_path,threshold \
    -o keyframes,timestamps \
    -c video \
    --constraints "use histogram difference,detect scene changes,return frame indices"
```

## Advanced Usage

### Generate Multiple Related Nodes

```python
async def generate_pipeline_nodes():
    generator = NodeGenerator()
    
    # Generate preprocessing node
    spec1 = NodeSpec(
        goal="Preprocess image for model inference",
        inputs=["image"],
        outputs=["preprocessed"],
        category="atomic"
    )
    
    # Generate postprocessing node
    spec2 = NodeSpec(
        goal="Postprocess model outputs to final results",
        inputs=["raw_outputs"],
        outputs=["results"],
        category="atomic"
    )
    
    result1 = await generator.generate_node(spec1)
    result2 = await generator.generate_node(spec2)
    
    # Now create workflow using both nodes
```

### Custom Template

For consistent style across generated nodes:

```python
# Extend NodeGenerator
class CustomNodeGenerator(NodeGenerator):
    def _build_generation_prompt(self, spec):
        # Add your custom template requirements
        base_prompt = super()._build_generation_prompt(spec)
        return base_prompt + "\nADDITIONAL REQUIREMENTS: <your style guide>"
```

## See Also

- [Architecture Documentation](../../ARCHITECTURE.md)
- [CLI Guide](../../CLI_GUIDE.md)
- [Workflow Agent Documentation](../../AGENTIC_SYSTEM.md)
- [Custom Nodes README](../custom/README.md)
