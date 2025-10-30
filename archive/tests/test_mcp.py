"""Quick test script for workflow engine functionality."""
import json
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from function_workflow_engine import FunctionWorkflowEngine

print("Testing Workflow Engine for MCP Integration...\n")

# Test 1: Initialize engine
print("1. Initializing workflow engine:")
engine = FunctionWorkflowEngine()
print("✅ Engine initialized successfully")
print()

# Test 2: Detect available devices (via ONNX runtime)
print("2. Testing device detection:")
try:
    import onnxruntime as ort
    available = ort.get_available_providers()
    print(f"Available execution providers: {available}")
    devices = {
        "directml": "DmlExecutionProvider" in available,
        "cuda": "CUDAExecutionProvider" in available,
        "cpu": "CPUExecutionProvider" in available
    }
    print(f"Device status: {json.dumps(devices, indent=2)}")
except Exception as e:
    print(f"⚠️ Could not detect devices: {e}")
print()

# Test 3: List available workflow files
print("3. Checking available workflows:")
workflow_dir = "workflows"
if os.path.exists(workflow_dir):
    workflows = [f for f in os.listdir(workflow_dir) if f.endswith('.json')]
    print(f"Found {len(workflows)} workflows:")
    for wf in workflows:
        print(f"  - {wf}")
else:
    print("⚠️ Workflows directory not found")
print()

# Test 4: Validate a workflow
print("4. Testing workflow validation:")
test_workflow = "workflows/video_detection.json"
if os.path.exists(test_workflow):
    try:
        with open(test_workflow, 'r') as f:
            workflow_data = json.load(f)
        print(f"✅ Workflow '{test_workflow}' is valid JSON")
        print(f"   Contains {len(workflow_data.get('nodes', []))} nodes")
    except Exception as e:
        print(f"❌ Workflow validation failed: {e}")
else:
    print(f"⚠️ Test workflow not found: {test_workflow}")
print()

print("✅ All basic tests completed!")
