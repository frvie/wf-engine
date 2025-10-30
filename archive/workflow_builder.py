"""
Workflow Builder: Natural Language to JSON Workflow Generation

This module uses AI to convert natural language descriptions into executable
workflow JSON files that can be run by the Function Workflow Engine.

Features:
- Natural language input → JSON workflow output
- Uses Ollama (qwen2.5-coder:7b) for code generation
- Validates generated workflows
- Provides workflow templates for common tasks
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient


# ============================================================================
# Workflow Knowledge Base
# ============================================================================

AVAILABLE_NODES = {
    # ========== ATOMIC IMAGE OPERATIONS ==========
    "read_image": {
        "function": "workflow_nodes.atomic.image_ops.read_image_node",
        "description": "Read image from file → numpy array",
        "inputs": ["image_path"],
        "outputs": ["image", "path", "height", "width", "channels"],
        "category": "image"
    },
    "resize_image_letterbox": {
        "function": "workflow_nodes.atomic.image_ops.resize_image_letterbox_node",
        "description": "Resize image with letterbox padding (maintains aspect ratio)",
        "inputs": ["image", "target_width", "target_height", "padding_color"],
        "outputs": ["image", "scale", "pad_x", "pad_y", "original_width", "original_height"],
        "category": "image"
    },
    "normalize_image": {
        "function": "workflow_nodes.atomic.image_ops.normalize_image_node",
        "description": "Normalize pixel values (e.g., 0-255 → 0-1)",
        "inputs": ["image", "scale", "mean", "std"],
        "outputs": ["image"],
        "category": "image"
    },
    "hwc_to_chw": {
        "function": "workflow_nodes.atomic.image_ops.hwc_to_chw_node",
        "description": "Transpose image from HWC to CHW format",
        "inputs": ["image"],
        "outputs": ["image"],
        "category": "image"
    },
    "add_batch_dimension": {
        "function": "workflow_nodes.atomic.image_ops.add_batch_dimension_node",
        "description": "Add batch dimension (C,H,W) → (1,C,H,W)",
        "inputs": ["image"],
        "outputs": ["image"],
        "category": "image"
    },
    "bgr_to_rgb": {
        "function": "workflow_nodes.atomic.image_ops.bgr_to_rgb_node",
        "description": "Convert BGR to RGB color space",
        "inputs": ["image"],
        "outputs": ["image"],
        "category": "image"
    },
    "save_image": {
        "function": "workflow_nodes.atomic.image_ops.save_image_node",
        "description": "Save image to file",
        "inputs": ["image", "output_path"],
        "outputs": ["success", "path"],
        "category": "image"
    },
    
    # ========== ATOMIC ONNX OPERATIONS ==========
    "create_onnx_cpu_session": {
        "function": "workflow_nodes.atomic.onnx_ops.create_onnx_cpu_session_node",
        "description": "Create ONNX Runtime session with CPU provider",
        "inputs": ["model_path"],
        "outputs": ["session", "provider", "input_name", "input_shape", "output_names"],
        "category": "onnx",
        "isolation": "auto"
    },
    "create_onnx_directml_session": {
        "function": "workflow_nodes.atomic.onnx_ops.create_onnx_directml_session_node",
        "description": "Create ONNX Runtime session with DirectML GPU provider",
        "inputs": ["model_path", "device_id"],
        "outputs": ["session", "provider", "input_name", "input_shape", "output_names"],
        "dependencies": ["onnxruntime-directml"],
        "category": "onnx",
        "isolation": "subprocess"
    },
    "create_onnx_cuda_session": {
        "function": "workflow_nodes.atomic.onnx_ops.create_onnx_cuda_session_node",
        "description": "Create ONNX Runtime session with CUDA provider",
        "inputs": ["model_path", "device_id"],
        "outputs": ["session", "provider", "input_name", "input_shape", "output_names"],
        "category": "onnx",
        "isolation": "auto"
    },
    "run_onnx_inference_benchmark": {
        "function": "workflow_nodes.atomic.onnx_ops.run_onnx_inference_benchmark_node",
        "description": "Run ONNX inference with benchmarking (multiple iterations)",
        "inputs": ["session", "input_name", "image", "iterations", "warmup_iterations"],
        "outputs": ["outputs", "avg_time_ms", "fps", "min_time_ms", "max_time_ms"],
        "category": "onnx"
    },
    "run_onnx_inference_single": {
        "function": "workflow_nodes.atomic.onnx_ops.run_onnx_inference_single_node",
        "description": "Run single ONNX inference",
        "inputs": ["session", "input_name", "image"],
        "outputs": ["outputs"],
        "category": "onnx"
    },
    
    # ========== ATOMIC YOLO POSTPROCESSING ==========
    "decode_yolo_v8_output": {
        "function": "workflow_nodes.atomic.yolo_ops.decode_yolo_v8_output_node",
        "description": "Decode raw YOLOv8 output → boxes, scores, class IDs",
        "inputs": ["outputs", "num_classes"],
        "outputs": ["boxes", "scores", "class_ids", "num_detections"],
        "category": "yolo"
    },
    "filter_by_confidence": {
        "function": "workflow_nodes.atomic.yolo_ops.filter_by_confidence_node",
        "description": "Filter detections by confidence threshold",
        "inputs": ["boxes", "scores", "class_ids", "confidence_threshold"],
        "outputs": ["boxes", "scores", "class_ids", "num_detections"],
        "category": "yolo"
    },
    "convert_cxcywh_to_xyxy": {
        "function": "workflow_nodes.atomic.yolo_ops.convert_cxcywh_to_xyxy_node",
        "description": "Convert boxes from (cx,cy,w,h) to (x1,y1,x2,y2)",
        "inputs": ["boxes"],
        "outputs": ["boxes"],
        "category": "yolo"
    },
    "apply_nms": {
        "function": "workflow_nodes.atomic.yolo_ops.apply_nms_node",
        "description": "Apply Non-Maximum Suppression to remove duplicates",
        "inputs": ["boxes", "scores", "class_ids", "iou_threshold"],
        "outputs": ["boxes", "scores", "class_ids", "num_detections"],
        "category": "yolo"
    },
    "scale_boxes_to_original": {
        "function": "workflow_nodes.atomic.yolo_ops.scale_boxes_to_original_node",
        "description": "Scale boxes from model input size to original image size",
        "inputs": ["boxes", "original_width", "original_height", "input_width", "input_height", "pad_x", "pad_y", "scale"],
        "outputs": ["boxes"],
        "category": "yolo"
    },
    "format_detections_coco": {
        "function": "workflow_nodes.atomic.yolo_ops.format_detections_coco_node",
        "description": "Format detections with COCO class names",
        "inputs": ["boxes", "scores", "class_ids"],
        "outputs": ["detections", "num_detections", "top_3"],
        "category": "format"
    },
    "create_detection_summary": {
        "function": "workflow_nodes.atomic.yolo_ops.create_detection_summary_node",
        "description": "Create text summary of detection results",
        "inputs": ["detections", "inference_time_ms", "fps", "provider"],
        "outputs": ["summary", "num_detections", "performance"],
        "category": "format"
    }
}

WORKFLOW_TEMPLATES = {
    "granular_cpu_inference": {
        "name": "Granular CPU Inference",
        "description": "YOLOv8 inference using atomic, composable nodes",
        "example": {
            "workflow": {
                "name": "Granular CPU Inference",
                "description": "Atomic node composition for CPU inference"
            },
            "nodes": [
                # Image preprocessing pipeline
                {"id": "read_img", "function": "workflow_nodes.atomic.image_ops.read_image_node",
                 "inputs": {"image_path": "input/soccer.jpg"}, "dependencies": []},
                {"id": "resize", "function": "workflow_nodes.atomic.image_ops.resize_image_letterbox_node",
                 "inputs": {"target_width": 640, "target_height": 640}, "dependencies": ["read_img"]},
                {"id": "normalize", "function": "workflow_nodes.atomic.image_ops.normalize_image_node",
                 "inputs": {"scale": 255.0}, "dependencies": ["resize"]},
                {"id": "transpose", "function": "workflow_nodes.atomic.image_ops.hwc_to_chw_node",
                 "inputs": {}, "dependencies": ["normalize"]},
                {"id": "add_batch", "function": "workflow_nodes.atomic.image_ops.add_batch_dimension_node",
                 "inputs": {}, "dependencies": ["transpose"]},
                
                # Model and inference
                {"id": "create_session", "function": "workflow_nodes.atomic.onnx_ops.create_onnx_cpu_session_node",
                 "inputs": {"model_path": "models/yolov8s.onnx"}, "dependencies": []},
                {"id": "benchmark", "function": "workflow_nodes.atomic.onnx_ops.run_onnx_inference_benchmark_node",
                 "inputs": {"iterations": 100}, "dependencies": ["create_session", "add_batch"]},
                
                # Postprocessing pipeline
                {"id": "decode", "function": "workflow_nodes.atomic.yolo_ops.decode_yolo_v8_output_node",
                 "inputs": {}, "dependencies": ["benchmark"]},
                {"id": "filter", "function": "workflow_nodes.atomic.yolo_ops.filter_by_confidence_node",
                 "inputs": {"confidence_threshold": 0.25}, "dependencies": ["decode"]},
                {"id": "convert", "function": "workflow_nodes.atomic.yolo_ops.convert_cxcywh_to_xyxy_node",
                 "inputs": {}, "dependencies": ["filter"]},
                {"id": "nms", "function": "workflow_nodes.atomic.yolo_ops.apply_nms_node",
                 "inputs": {"iou_threshold": 0.45}, "dependencies": ["convert"]},
                {"id": "scale", "function": "workflow_nodes.atomic.yolo_ops.scale_boxes_to_original_node",
                 "inputs": {}, "dependencies": ["nms", "read_img", "resize"]},
                {"id": "format", "function": "workflow_nodes.atomic.yolo_ops.format_detections_coco_node",
                 "inputs": {}, "dependencies": ["scale"]},
                {"id": "summary", "function": "workflow_nodes.atomic.yolo_ops.create_detection_summary_node",
                 "inputs": {"provider": "CPU"}, "dependencies": ["format", "benchmark"]}
            ]
        }
    },
    "granular_directml_inference": {
        "name": "Granular DirectML GPU Inference",
        "description": "YOLOv8 inference on DirectML GPU using atomic nodes",
        "example": {
            "workflow": {
                "name": "Granular DirectML Inference",
                "description": "Atomic node composition for DirectML GPU"
            },
            "nodes": [
                # Same preprocessing as CPU
                {"id": "read_img", "function": "workflow_nodes.atomic.image_ops.read_image_node",
                 "inputs": {"image_path": "input/soccer.jpg"}, "dependencies": []},
                {"id": "resize", "function": "workflow_nodes.atomic.image_ops.resize_image_letterbox_node",
                 "inputs": {"target_width": 640, "target_height": 640}, "dependencies": ["read_img"]},
                {"id": "normalize", "function": "workflow_nodes.atomic.image_ops.normalize_image_node",
                 "inputs": {"scale": 255.0}, "dependencies": ["resize"]},
                {"id": "transpose", "function": "workflow_nodes.atomic.image_ops.hwc_to_chw_node",
                 "inputs": {}, "dependencies": ["normalize"]},
                {"id": "add_batch", "function": "workflow_nodes.atomic.image_ops.add_batch_dimension_node",
                 "inputs": {}, "dependencies": ["transpose"]},
                
                # DirectML session (subprocess isolation)
                {"id": "create_session", "function": "workflow_nodes.atomic.onnx_ops.create_onnx_directml_session_node",
                 "inputs": {"model_path": "models/yolov8s.onnx", "device_id": 0}, "dependencies": []},
                {"id": "benchmark", "function": "workflow_nodes.atomic.onnx_ops.run_onnx_inference_benchmark_node",
                 "inputs": {"iterations": 100}, "dependencies": ["create_session", "add_batch"]},
                
                # Same postprocessing as CPU
                {"id": "decode", "function": "workflow_nodes.atomic.yolo_ops.decode_yolo_v8_output_node",
                 "inputs": {}, "dependencies": ["benchmark"]},
                {"id": "filter", "function": "workflow_nodes.atomic.yolo_ops.filter_by_confidence_node",
                 "inputs": {"confidence_threshold": 0.25}, "dependencies": ["decode"]},
                {"id": "convert", "function": "workflow_nodes.atomic.yolo_ops.convert_cxcywh_to_xyxy_node",
                 "inputs": {}, "dependencies": ["filter"]},
                {"id": "nms", "function": "workflow_nodes.atomic.yolo_ops.apply_nms_node",
                 "inputs": {"iou_threshold": 0.45}, "dependencies": ["convert"]},
                {"id": "scale", "function": "workflow_nodes.atomic.yolo_ops.scale_boxes_to_original_node",
                 "inputs": {}, "dependencies": ["nms", "read_img", "resize"]},
                {"id": "format", "function": "workflow_nodes.atomic.yolo_ops.format_detections_coco_node",
                 "inputs": {}, "dependencies": ["scale"]},
                {"id": "summary", "function": "workflow_nodes.atomic.yolo_ops.create_detection_summary_node",
                 "inputs": {"provider": "DirectML"}, "dependencies": ["format", "benchmark"]}
            ]
        }
    }
}


# ============================================================================
# Workflow Builder Tools
# ============================================================================

async def list_workflow_nodes() -> str:
    """
    List all available workflow nodes with their descriptions.
    
    Returns:
        JSON string with node information
    """
    return json.dumps(AVAILABLE_NODES, indent=2)


async def get_workflow_templates() -> str:
    """
    Get available workflow templates.
    
    Returns:
        JSON string with template information
    """
    templates = {
        name: {
            "name": template["name"],
            "description": template["description"]
        }
        for name, template in WORKFLOW_TEMPLATES.items()
    }
    return json.dumps(templates, indent=2)


async def get_template_example(template_name: str) -> str:
    """
    Get a specific template example.
    
    Args:
        template_name: Name of the template
        
    Returns:
        JSON string with template example
    """
    if template_name not in WORKFLOW_TEMPLATES:
        return json.dumps({"error": f"Template '{template_name}' not found"})
    
    return json.dumps(WORKFLOW_TEMPLATES[template_name]["example"], indent=2)


async def validate_workflow_json(workflow_json: str) -> str:
    """
    Validate a workflow JSON structure.
    
    Args:
        workflow_json: JSON string to validate
        
    Returns:
        JSON string with validation results
    """
    try:
        workflow = json.loads(workflow_json)
        
        errors = []
        warnings = []
        
        # Check required top-level keys
        if "workflow" not in workflow:
            errors.append("Missing 'workflow' key")
        if "nodes" not in workflow:
            errors.append("Missing 'nodes' key")
        
        # Validate nodes
        if "nodes" in workflow:
            node_ids = set()
            for i, node in enumerate(workflow["nodes"]):
                # Check required node fields
                if "id" not in node:
                    errors.append(f"Node {i}: Missing 'id'")
                else:
                    if node["id"] in node_ids:
                        errors.append(f"Duplicate node id: {node['id']}")
                    node_ids.add(node["id"])
                
                if "function" not in node:
                    errors.append(f"Node {node.get('id', i)}: Missing 'function'")
                
                if "dependencies" not in node:
                    warnings.append(f"Node {node.get('id', i)}: Missing 'dependencies' (will default to [])")
                
                # Validate dependencies exist
                if "dependencies" in node:
                    for dep in node["dependencies"]:
                        if dep not in node_ids:
                            errors.append(f"Node {node.get('id', i)}: Dependency '{dep}' not found")
        
        result = {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
        
        return json.dumps(result, indent=2)
        
    except json.JSONDecodeError as e:
        return json.dumps({
            "valid": False,
            "errors": [f"Invalid JSON: {str(e)}"],
            "warnings": []
        })


async def save_workflow(workflow_json: str, filename: str) -> str:
    """
    Save a workflow to a JSON file.
    
    Args:
        workflow_json: Workflow JSON string
        filename: Name of the file to save (without path)
        
    Returns:
        JSON string with save status
    """
    try:
        workflow = json.loads(workflow_json)
        
        # Ensure workflows directory exists
        workflows_dir = Path("workflows")
        workflows_dir.mkdir(exist_ok=True)
        
        # Add .json extension if missing
        if not filename.endswith(".json"):
            filename = f"{filename}.json"
        
        filepath = workflows_dir / filename
        
        # Save with pretty formatting
        with open(filepath, 'w') as f:
            json.dump(workflow, f, indent=2)
        
        return json.dumps({
            "status": "success",
            "path": str(filepath),
            "message": f"Workflow saved to {filepath}"
        })
        
    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": str(e)
        })


# ============================================================================
# Workflow Builder Agent
# ============================================================================

async def create_workflow_builder():
    """Create the workflow builder agent."""
    
    # Create Ollama client
    model_client = OpenAIChatCompletionClient(
        model="qwen2.5-coder:7b",
        api_key="ollama",
        base_url="http://localhost:11434/v1",
        model_capabilities={
            "vision": False,
            "function_calling": True,
            "json_output": True,
        },
        temperature=0.2,  # Low temperature for consistent JSON generation
        max_tokens=4000,  # Higher limit for workflow generation
    )
    
    # Create builder agent
    builder = AssistantAgent(
        name="WorkflowBuilder",
        model_client=model_client,
        tools=[
            list_workflow_nodes,
            get_workflow_templates,
            get_template_example,
            validate_workflow_json,
            save_workflow,
        ],
        system_message="""You are an expert workflow builder that creates JSON workflow configurations using ATOMIC, COMPOSABLE nodes.

**YOUR ROLE:**
Generate valid workflow JSON files from natural language by composing small, focused nodes.

**PHILOSOPHY - Granular Composition:**
- Each node does ONE thing (5-20 lines of code)
- Compose multiple small nodes into complex workflows
- Reuse preprocessing/postprocessing across backends
- Clear data flow through dependencies

**AVAILABLE TOOLS:**
1. `list_workflow_nodes()` - Get all 23+ atomic nodes organized by category
2. `get_workflow_templates()` - See template patterns
3. `get_template_example(template_name)` - Get full example to modify
4. `validate_workflow_json(json)` - Validate generated workflow
5. `save_workflow(json, filename)` - Save workflow to file

**NODE CATEGORIES:**

1. **Image Operations** (7 nodes):
   - read_image: Load from file
   - resize_image_letterbox: Resize with padding
   - normalize_image: Scale pixel values
   - hwc_to_chw: Transpose dimensions
   - add_batch_dimension: Add batch dim
   - bgr_to_rgb: Color conversion
   - save_image: Write to file

2. **ONNX Operations** (5 nodes):
   - create_onnx_cpu_session: CPU provider
   - create_onnx_directml_session: DirectML GPU (subprocess)
   - create_onnx_cuda_session: CUDA provider
   - run_onnx_inference_benchmark: Timed inference
   - run_onnx_inference_single: Single run

3. **YOLO Postprocessing** (7 nodes):
   - decode_yolo_v8_output: Parse raw output
   - filter_by_confidence: Threshold filtering
   - convert_cxcywh_to_xyxy: Box format conversion
   - apply_nms: Non-maximum suppression
   - scale_boxes_to_original: Resize boxes
   - format_detections_coco: Add class names
   - create_detection_summary: Text summary

**COMMON PATTERNS:**

Pattern 1: Image Preprocessing (5 nodes)
```
read_image → resize_image_letterbox → normalize_image → hwc_to_chw → add_batch_dimension
```

Pattern 2: YOLO Postprocessing (7 nodes)
```
decode_yolo_v8_output → filter_by_confidence → convert_cxcywh_to_xyxy → 
apply_nms → scale_boxes_to_original → format_detections_coco → create_detection_summary
```

Pattern 3: Complete Inference (CPU)
```
[Image Preprocessing] → create_onnx_cpu_session → run_onnx_inference_benchmark → [YOLO Postprocessing]
```

Pattern 4: Complete Inference (DirectML GPU)
```
[Image Preprocessing] → create_onnx_directml_session → run_onnx_inference_benchmark → [YOLO Postprocessing]
```

**PROCESS:**
1. When asked to create a workflow:
   - Identify the task (e.g., "CPU inference", "GPU detection")
   - Select appropriate session node (cpu/directml/cuda)
   - Add image preprocessing pipeline (read → resize → normalize → transpose → batch)
   - Add inference node (benchmark or single)
   - Add YOLO postprocessing pipeline (decode → filter → convert → nms → scale → format → summary)
   - Call `validate_workflow_json()` to check
   - Call `save_workflow()` if valid

2. Key rules:
   - Use atomic nodes - don't create monolithic steps
   - Preprocessing nodes take image from previous node
   - Inference needs both session and preprocessed image
   - Postprocessing needs inference outputs and original image info
   - Dependencies ensure correct execution order

3. Example node composition:
   ```json
   {
     "id": "resize",
     "function": "workflow_nodes.atomic.image_ops.resize_image_letterbox_node",
     "inputs": {"target_width": 640, "target_height": 640},
     "dependencies": ["read_img"]
   }
   ```

**WORKFLOW JSON STRUCTURE:**
```json
{
  "workflow": {
    "name": "Descriptive Name",
    "description": "What this does"
  },
  "nodes": [
    {
      "id": "unique_id",
      "function": "workflow_nodes.atomic.category.node_function",
      "inputs": {"param": "value"},
      "dependencies": ["dependency_id"]
    }
  ]
}
```

**BEST PRACTICES:**
- Use short, clear node IDs (read_img, resize, normalize, etc.)
- Compose pipelines from atomic nodes
- Reuse preprocessing across backends (CPU vs GPU)
- Let dependencies handle data flow (auto-injection)
- Validate before saving

Be helpful and explain your composition strategy!""",
    )
    
    return builder


# ============================================================================
# Interactive Workflow Builder
# ============================================================================

async def build_workflow_interactive(description: str) -> Dict[str, Any]:
    """
    Build a workflow from natural language description.
    
    Args:
        description: Natural language workflow description
        
    Returns:
        Dict with workflow generation results
    """
    print(f"\n🏗️  Building workflow from: '{description}'")
    print("=" * 60)
    
    # Create builder agent
    builder = await create_workflow_builder()
    
    # Create termination conditions
    termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(15)
    
    # Create team
    team = RoundRobinGroupChat([builder], termination_condition=termination)
    
    # Generate workflow
    task = f"""Create a workflow for: {description}

Steps:
1. List available nodes to understand options
2. Generate appropriate workflow JSON
3. Validate the workflow
4. Save it with a descriptive filename
5. Explain what you created"""
    
    print(f"\n📝 Task: {task}\n")
    
    # Run the builder
    from autogen_agentchat.ui import Console
    await Console(team.run_stream(task=task))
    
    print("\n" + "=" * 60)
    print("✅ Workflow generation complete!")
    
    return {"status": "success"}


# ============================================================================
# CLI Interface
# ============================================================================

async def main():
    """Main CLI interface for workflow builder."""
    import sys
    
    print("🏗️  Workflow Builder - Natural Language to JSON")
    print("=" * 60)
    
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python workflow_builder.py \"<description>\"")
        print("\nExamples:")
        print("  python workflow_builder.py \"Run YOLOv8 inference on CPU\"")
        print("  python workflow_builder.py \"Compare DirectML and OpenVINO performance\"")
        print("  python workflow_builder.py \"Process an image with DirectML GPU acceleration\"")
        return
    
    description = " ".join(sys.argv[1:])
    
    try:
        await build_workflow_interactive(description)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure Ollama is running: ollama serve")
        print("2. Check model availability: ollama list")
        print("3. Pull if needed: ollama pull qwen2.5-coder:7b")


if __name__ == "__main__":
    asyncio.run(main())
