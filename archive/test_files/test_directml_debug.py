import loggingimport logging

import sysimport sys

sys.path.insert(0, ".")sys.path.insert(0, ".")



# Set up detailed logging# Set up detailed logging

logging.basicConfig(logging.basicConfig(

    level=logging.DEBUG,    level=logging.DEBUG,

    format="%(levelname)s:%(name)s:%(funcName)s: %(message)s"    format="%(levelname)s:%(name)s:%(funcName)s: %(message)s"

))



from src.core.engine import FunctionWorkflowEnginefrom src.core.engine import FunctionWorkflowEngine



# Test DirectML node# Test DirectML node

engine = FunctionWorkflowEngine()engine = FunctionWorkflowEngine()

node = {node = {

    "id": "test_directml",    "id": "test_directml",

    "function": "src.nodes.onnx_ops.create_onnx_directml_session_node",    "function": "src.nodes.onnx_ops.create_onnx_directml_session_node",

    "inputs": {"model_path": "models/yolov8s.onnx", "device_id": 1}    "inputs": {"model_path": "models/yolov8s.onnx", "device_id": 1}

}}



print("=== DirectML Node Test ===")print("=== DirectML Node Test ===")

result = engine._execute_function_node(node)result = engine._execute_function_node(node)

has_error = "error" not in resultprint(f"Success: {\"error\" not in result}")

print(f"Success: {has_error}")
