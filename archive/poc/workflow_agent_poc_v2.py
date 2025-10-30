"""
Proof of Concept: Agentic Workflow Engine with AutoGen + Ollama (Latest API)

This POC demonstrates how to use AutoGen agents with Ollama (local LLM)
to provide a natural language interface to the workflow engine.

Architecture:
- Assistant Agent: Analyzes requests, detects devices, executes workflows
- Uses Ollama for local AI reasoning (no external API calls)

All AI reasoning happens locally via Ollama.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from function_workflow_engine import FunctionWorkflowEngine


# ============================================================================
# Function Tools for Agent
# ============================================================================

async def execute_workflow(workflow_path: str) -> str:
    """
    Execute a workflow from a JSON file.
    
    Args:
        workflow_path: Path to the workflow JSON file
        
    Returns:
        JSON string with execution results
    """
    try:
        with open(workflow_path, 'r') as f:
            workflow_data = json.load(f)
        
        engine = FunctionWorkflowEngine(workflow_data)
        results = engine.execute()
        
        return json.dumps({
            "status": "success",
            "workflow": workflow_path,
            "results": str(results)  # Simplified for POC
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "status": "error",
            "workflow": workflow_path,
            "error": str(e)
        }, indent=2)


async def detect_devices() -> str:
    """
    Detect available hardware acceleration devices.
    
    Returns:
        JSON string with device availability
    """
    devices = {
        "directml": False,
        "cuda": False,
        "openvino": False,
        "cpu": True
    }
    
    try:
        # Try to import and check ONNX Runtime providers
        import onnxruntime as ort
        if hasattr(ort, 'get_available_providers'):
            available = ort.get_available_providers()
            devices["directml"] = "DmlExecutionProvider" in available
            devices["cuda"] = "CUDAExecutionProvider" in available
            devices["openvino"] = "OpenVINOExecutionProvider" in available
    except Exception as e:
        devices["error"] = str(e)
    
    return json.dumps(devices, indent=2)


async def list_available_workflows() -> str:
    """
    List all available workflow files.
    
    Returns:
        JSON string with list of workflow paths
    """
    workflows_dir = Path("workflows")
    if not workflows_dir.exists():
        return json.dumps({"workflows": []})
    
    workflows = [str(f) for f in workflows_dir.glob("*.json")]
    return json.dumps({"workflows": workflows}, indent=2)


async def get_workflow_info(workflow_path: str) -> str:
    """
    Get information about a workflow.
    
    Args:
        workflow_path: Path to the workflow JSON file
        
    Returns:
        JSON string with workflow information
    """
    try:
        with open(workflow_path, 'r') as f:
            workflow_data = json.load(f)
        
        info = {
            "path": workflow_path,
            "name": workflow_data.get("workflow", {}).get("name", "Unknown"),
            "description": workflow_data.get("workflow", {}).get("description", "No description"),
            "nodes": len(workflow_data.get("nodes", [])),
            "node_types": [node.get("function", "unknown") for node in workflow_data.get("nodes", [])]
        }
        
        return json.dumps(info, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ============================================================================
# Agent Setup
# ============================================================================

async def create_workflow_assistant():
    """Create the workflow assistant agent with Ollama."""
    
    # Create Ollama client (OpenAI-compatible API)
    # Using qwen2.5-coder:7b - excellent for code/JSON generation and supports function calling
    model_client = OpenAIChatCompletionClient(
        model="qwen2.5-coder:7b",
        api_key="ollama",  # Dummy key
        base_url="http://localhost:11434/v1",
        model_capabilities={
            "vision": False,
            "function_calling": True,
            "json_output": True,
        },
        # Add explicit parameters for better tool execution
        temperature=0.1,  # Lower temperature for more deterministic responses
        max_tokens=2000,  # Ensure sufficient token budget for responses
    )
    
    # Create assistant agent with tools
    assistant = AssistantAgent(
        name="WorkflowAssistant",
        model_client=model_client,
        tools=[
            execute_workflow,
            detect_devices,
            list_available_workflows,
            get_workflow_info,
        ],
        system_message="""You are a helpful workflow assistant specialized in managing AI/ML workflows.

**YOUR ROLE:**
- You MUST use the provided tools to answer questions
- ALWAYS call the appropriate function when asked about devices, workflows, or execution
- After calling a function, wait for its result and then provide a clear summary to the user

**AVAILABLE TOOLS:**
1. `detect_devices()` - Check available hardware (DirectML GPU, CUDA, OpenVINO, CPU)
2. `list_available_workflows()` - List all workflow JSON files
3. `get_workflow_info(workflow_path)` - Get details about a specific workflow
4. `execute_workflow(workflow_path)` - Run a workflow and return results

**WORKFLOWS AVAILABLE:**
- workflows/video_detection.json: Real-time webcam object detection
- workflows/video_detection_mp4.json: Process pre-recorded video files  
- workflows/video_detection_directml.json: High-performance DirectML GPU pipeline
- workflows/parallel_yolov8.json: Compare inference across multiple backends

**INSTRUCTIONS:**
1. When asked about hardware: Call detect_devices() first
2. When asked about workflows: Call list_available_workflows() and/or get_workflow_info()
3. When asked to run something: Call execute_workflow() with the appropriate path
4. ALWAYS summarize tool results in plain English after receiving them
5. Be concise but informative

Remember: USE THE TOOLS. Don't guess or make up information.""",
    )
    
    return assistant


# ============================================================================
# Main POC Demo
# ============================================================================

async def run_demo():
    """Run the agentic workflow POC."""
    print("🤖 Agentic Workflow Engine POC")
    print("=" * 60)
    print("Using Ollama (qwen2.5-coder:7b) for local AI reasoning\n")
    
    # Verify Ollama is running
    try:
        import httpx
        response = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
        if response.status_code == 200:
            print("✅ Ollama server is running")
        else:
            print(f"⚠️  Ollama responded with status {response.status_code}")
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        print("\nPlease ensure Ollama is running:")
        print("  ollama serve")
        return
    
    # Create assistant
    print("🔧 Initializing workflow assistant...")
    assistant = await create_workflow_assistant()
    print("✅ Assistant ready\n")
    
    # Create termination conditions
    termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(10)
    
    # Create a simple single-agent chat
    team = RoundRobinGroupChat([assistant], termination_condition=termination)
    
    # Example 1: Device detection
    print("📋 Example 1: Detect available devices")
    print("-" * 60)
    print("Task: 'Please detect what hardware devices are available on this system.'\n")
    
    task1 = "Please detect what hardware devices are available on this system. Use the detect_devices function."
    
    await Console(team.run_stream(task=task1))
    
    print("\n" + "=" * 60 + "\n")
    
    # Reset team for next example
    team = RoundRobinGroupChat([assistant], termination_condition=termination)
    
    # Example 2: List workflows
    print("📋 Example 2: List available workflows")
    print("-" * 60)
    print("Task: 'What workflows are available? Give me a brief description of each.'\n")
    
    task2 = "What workflows are available? Use the list_available_workflows and get_workflow_info functions to tell me about each workflow."
    
    await Console(team.run_stream(task=task2))
    
    print("\n" + "=" * 60)
    print("✅ POC Demo Complete!")
    print("\n💡 Next steps:")
    print("- Test with: 'Execute the video detection workflow'")
    print("- Try custom queries about workflows and devices")
    print("- Integrate with your own workflows")


def main():
    """Entry point for the POC."""
    try:
        asyncio.run(run_demo())
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure Ollama is running: ollama serve")
        print("2. Check if qwen2.5-coder:7b is available: ollama list")
        print("3. Pull the model if needed: ollama pull qwen2.5-coder:7b")


if __name__ == "__main__":
    main()
