"""
Simple Workflow Agent using AutoGen + Ollama

This POC demonstrates:
1. AutoGen agents for workflow planning and execution
2. Ollama for local LLM inference
3. MCP server integration (via function calls)
"""

import json
import os
import autogen

# Configure Ollama as the LLM provider
config_list = [
    {
        "model": "codellama:latest",  # Using available model
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",  # Dummy key (Ollama doesn't need it)
    }
]

llm_config = {
    "config_list": config_list,
    "temperature": 0.7,
    "timeout": 120,
}

# Import workflow execution function
from function_workflow_engine import run_function_workflow

# Define workflow execution tools for agents
def execute_workflow(workflow_path: str) -> dict:
    """Execute a workflow from JSON file"""
    try:
        results = run_function_workflow(workflow_path)
        return {"success": True, "results": results}
    except Exception as e:
        return {"success": False, "error": str(e)}

def detect_devices() -> dict:
    """Detect available inference devices"""
    devices = {
        "directml": False,
        "cuda": False,
        "npu": False,
        "cpu": True
    }
    
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        devices["directml"] = 'DmlExecutionProvider' in providers
        devices["cuda"] = 'CUDAExecutionProvider' in providers
    except Exception:
        pass
    
    try:
        import openvino as ov
        core = ov.Core()
        devices["npu"] = 'NPU' in core.available_devices
    except Exception:
        pass
    
    return devices

def list_available_workflows() -> list:
    """List available workflow templates"""
    workflows = []
    workflows_dir = "workflows"
    
    if os.path.exists(workflows_dir):
        for file in os.listdir(workflows_dir):
            if file.endswith('.json'):
                workflows.append({
                    "name": file,
                    "path": os.path.join(workflows_dir, file)
                })
    
    return workflows

def generate_workflow_json(task_description: str, available_devices: dict) -> str:
    """Generate workflow JSON based on task description"""
    
    # Simple template-based generation for POC
    # In production, the LLM would generate this
    
    if "video" in task_description.lower() or "mp4" in task_description.lower():
        # Use video detection workflow
        workflow = {
            "workflow": {
                "name": "Generated Video Detection",
                "description": f"Auto-generated workflow for: {task_description}",
                "version": "1.0"
            },
            "nodes": [
                {
                    "id": "download_model",
                    "function": "workflow_nodes.download_model_node.download_model_node",
                    "inputs": {
                        "model_name": "yolov8s.onnx",
                        "models_dir": "models"
                    }
                },
                {
                    "id": "realtime_detection",
                    "function": "workflow_nodes.realtime_video_detection_node.realtime_video_detection_node",
                    "depends_on": ["download_model"],
                    "inputs": {
                        "source": "0",  # Webcam by default
                        "model_path": "$download_model.model_path",
                        "conf_threshold": 0.25,
                        "iou_threshold": 0.45,
                        "display_fps": True,
                        "save_video": False,
                        "max_duration": 0
                    }
                }
            ]
        }
    else:
        # Default to parallel inference workflow
        workflow = {
            "workflow": {
                "name": "Generated Multi-Backend Inference",
                "description": f"Auto-generated workflow for: {task_description}",
                "version": "1.0"
            },
            "nodes": [
                {
                    "id": "download_model",
                    "function": "workflow_nodes.download_model_node.download_model_node",
                    "inputs": {
                        "model_name": "yolov8s.onnx",
                        "models_dir": "models"
                    }
                }
            ]
        }
    
    # Save to temp file
    temp_path = "generated_workflow.json"
    with open(temp_path, 'w') as f:
        json.dump(workflow, f, indent=2)
    
    return temp_path

# Create AutoGen agents

# User proxy (represents the human user)
user_proxy = autogen.UserProxyAgent(
    name="User",
    system_message="A human user who wants to run computer vision workflows.",
    human_input_mode="NEVER",  # For POC, no human input required
    max_consecutive_auto_reply=0,
    code_execution_config=False,
)

# Planning agent (analyzes task and plans workflow)
planner = autogen.AssistantAgent(
    name="Planner",
    system_message="""You are a workflow planning agent. Your job is to:
1. Analyze user requests for computer vision tasks
2. Determine which workflow to use (video detection, image inference, etc.)
3. Check available hardware (DirectML GPU, NPU, CUDA, CPU)
4. Generate or select appropriate workflow configuration

Available workflows:
- video_detection.json: Real-time webcam object detection
- video_detection_mp4.json: Process MP4 video files
- parallel_yolov8.json: Multi-backend inference comparison

When a user asks to process video, recommend video_detection_mp4.json.
When a user asks for webcam/live detection, recommend video_detection.json.
When a user asks to compare performance, recommend parallel_yolov8.json.

Be concise and direct.""",
    llm_config=llm_config,
)

# Execution agent (runs workflows)
executor = autogen.AssistantAgent(
    name="Executor",
    system_message="""You are a workflow execution agent. Your job is to:
1. Execute workflows using the execute_workflow function
2. Monitor execution and report results
3. Handle errors and suggest alternatives if needed

Be concise and report results clearly.""",
    llm_config=llm_config,
)

# Register functions with agents
autogen.register_function(
    execute_workflow,
    caller=executor,
    executor=user_proxy,
    name="execute_workflow",
    description="Execute a workflow from JSON file path"
)

autogen.register_function(
    detect_devices,
    caller=planner,
    executor=user_proxy,
    name="detect_devices",
    description="Detect available inference devices (DirectML, CUDA, NPU, CPU)"
)

autogen.register_function(
    list_available_workflows,
    caller=planner,
    executor=user_proxy,
    name="list_available_workflows",
    description="List all available workflow templates"
)

def run_workflow_agent(user_request: str):
    """Main function to run the workflow agent"""
    
    print(f"\n{'='*60}")
    print("🤖 Workflow Agent POC (AutoGen + Ollama + MCP)")
    print(f"{'='*60}\n")
    print(f"User Request: {user_request}\n")
    
    # Start the conversation
    user_proxy.initiate_chat(
        planner,
        message=f"""User request: {user_request}

Please:
1. Detect available devices
2. List available workflows
3. Recommend the best workflow for this task
4. Then ask the Executor to run it
"""
    )

if __name__ == "__main__":
    # Example usage
    user_request = "I want to detect objects in my webcam feed"
    run_workflow_agent(user_request)
