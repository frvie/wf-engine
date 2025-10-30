"""
MCP Server for Workflow Engine

Exposes workflow engine capabilities as MCP tools for agent integration.

Enhanced with:
- LLM-powered workflow composition (via Ollama)
- Agentic learning and optimization
- Performance analysis and suggestions
"""

import json
import logging
from pathlib import Path

# MCP Server imports
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Workflow engine imports
from function_workflow_engine import run_function_workflow
from agentic_integration import AgenticWorkflowEngine
from workflow_agent import AgenticWorkflowSystem, WorkflowGoal
from workflow_agent_llm import LLMWorkflowComposer, verify_ollama_connection
import importlib.util
import os

# Initialize server and components
server = Server("workflow-engine")
logger = logging.getLogger("mcp.workflow")

# Initialize agentic system
agentic_system = AgenticWorkflowSystem()
llm_composer = None  # Lazy init when Ollama is needed

@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available MCP tools for workflow operations"""
    return [
        Tool(
            name="execute_workflow",
            description="Execute a workflow from JSON file path or JSON string",
            inputSchema={
                "type": "object",
                "properties": {
                    "workflow": {
                        "type": "string",
                        "description": "Path to workflow JSON file or JSON string"
                    }
                },
                "required": ["workflow"]
            }
        ),
        Tool(
            name="detect_devices",
            description="Detect available inference devices (NPU, DirectML, CUDA, CPU)",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="list_workflow_nodes",
            description="List all available workflow nodes with their capabilities",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="validate_workflow",
            description="Validate a workflow JSON configuration",
            inputSchema={
                "type": "object",
                "properties": {
                    "workflow": {
                        "type": "string",
                        "description": "Path to workflow JSON file or JSON string"
                    }
                },
                "required": ["workflow"]
            }
        ),
        Tool(
            name="get_workflow_templates",
            description="Get available workflow templates for common tasks",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="create_workflow_from_nl",
            description="Create a workflow from natural language description using LLM (requires Ollama) or rule-based composition",
            inputSchema={
                "type": "object",
                "properties": {
                    "requirement": {
                        "type": "string",
                        "description": "Natural language description of workflow requirements"
                    },
                    "use_llm": {
                        "type": "boolean",
                        "description": "Use LLM for advanced reasoning (requires Ollama running)",
                        "default": False
                    }
                },
                "required": ["requirement"]
            }
        ),
        Tool(
            name="optimize_workflow",
            description="Get optimization suggestions for a workflow based on execution history",
            inputSchema={
                "type": "object",
                "properties": {
                    "workflow_type": {
                        "type": "string",
                        "description": "Type of workflow (granular, monolithic, fast_pipeline)"
                    },
                    "target_fps": {
                        "type": "number",
                        "description": "Target FPS to achieve",
                        "default": 20.0
                    }
                },
                "required": ["workflow_type"]
            }
        ),
        Tool(
            name="analyze_performance",
            description="Analyze performance trends and execution history",
            inputSchema={
                "type": "object",
                "properties": {
                    "workflow_type": {
                        "type": "string",
                        "description": "Filter by workflow type (optional)"
                    }
                }
            }
        ),
        Tool(
            name="execute_workflow_with_learning",
            description="Execute a workflow with agentic learning enabled (records metrics for future optimization)",
            inputSchema={
                "type": "object",
                "properties": {
                    "workflow": {
                        "type": "string",
                        "description": "Path to workflow JSON file or JSON string"
                    },
                    "enable_learning": {
                        "type": "boolean",
                        "description": "Enable agentic learning from execution",
                        "default": True
                    }
                },
                "required": ["workflow"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls from MCP clients"""
    
    try:
        if name == "execute_workflow":
            workflow_input = arguments["workflow"]
            
            # Check if it's a file path or JSON string
            if os.path.exists(workflow_input):
                results = run_function_workflow(workflow_input)
            else:
                # Try to parse as JSON string
                workflow_data = json.loads(workflow_input)
                # Save to temp file
                temp_path = "temp_workflow.json"
                with open(temp_path, 'w') as f:
                    json.dump(workflow_data, f, indent=2)
                results = run_function_workflow(temp_path)
            
            return [TextContent(
                type="text",
                text=json.dumps(results, indent=2, default=str)
            )]
        
        elif name == "detect_devices":
            devices = _detect_available_devices()
            return [TextContent(
                type="text",
                text=json.dumps(devices, indent=2)
            )]
        
        elif name == "list_workflow_nodes":
            nodes = _list_available_nodes()
            return [TextContent(
                type="text",
                text=json.dumps(nodes, indent=2)
            )]
        
        elif name == "validate_workflow":
            workflow_input = arguments["workflow"]
            
            if os.path.exists(workflow_input):
                with open(workflow_input, 'r') as f:
                    workflow_data = json.load(f)
            else:
                workflow_data = json.loads(workflow_input)
            
            validation = _validate_workflow(workflow_data)
            return [TextContent(
                type="text",
                text=json.dumps(validation, indent=2)
            )]
        
        elif name == "get_workflow_templates":
            templates = _get_workflow_templates()
            return [TextContent(
                type="text",
                text=json.dumps(templates, indent=2)
            )]
        
        elif name == "create_workflow_from_nl":
            requirement = arguments["requirement"]
            use_llm = arguments.get("use_llm", False)
            result = _create_workflow_from_nl(requirement, use_llm)
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
        
        elif name == "optimize_workflow":
            workflow_type = arguments["workflow_type"]
            target_fps = arguments.get("target_fps", 20.0)
            result = _optimize_workflow(workflow_type, target_fps)
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
        
        elif name == "analyze_performance":
            workflow_type = arguments.get("workflow_type")
            result = _analyze_performance(workflow_type)
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
        
        elif name == "execute_workflow_with_learning":
            workflow_input = arguments["workflow"]
            enable_learning = arguments.get("enable_learning", True)
            result = _execute_workflow_with_learning(workflow_input, enable_learning)
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2, default=str)
            )]
        
        else:
            return [TextContent(
                type="text",
                text=json.dumps({"error": f"Unknown tool: {name}"})
            )]
    
    except Exception as e:
        logger.error(f"Error executing tool {name}: {e}")
        return [TextContent(
            type="text",
            text=json.dumps({"error": str(e)})
        )]

def _detect_available_devices() -> dict:
    """Detect available inference devices"""
    devices = {
        "directml": False,
        "cuda": False,
        "npu": False,
        "cpu": True  # Always available
    }
    
    # Check DirectML
    try:
        import onnxruntime as ort
        if 'DmlExecutionProvider' in ort.get_available_providers():
            devices["directml"] = True
    except Exception:
        pass
    
    # Check CUDA
    try:
        import onnxruntime as ort
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            devices["cuda"] = True
    except Exception:
        pass
    
    # Check NPU
    try:
        import openvino as ov
        core = ov.Core()
        if 'NPU' in core.available_devices:
            devices["npu"] = True
    except Exception:
        pass
    
    return devices

def _list_available_nodes() -> dict:
    """List all available workflow nodes"""
    nodes_dir = Path("workflow_nodes")
    nodes = {}
    
    for py_file in nodes_dir.glob("*_node.py"):
        module_name = py_file.stem
        try:
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Find decorated functions
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if callable(attr) and hasattr(attr, '__name__'):
                        nodes[attr.__name__] = {
                            "module": module_name,
                            "file": str(py_file),
                            "dependencies": getattr(attr, 'dependencies', []),
                            "isolation_mode": getattr(attr, 'isolation_mode', 'auto')
                        }
        except Exception as e:
            logger.warning(f"Could not load node {module_name}: {e}")
    
    return nodes

def _validate_workflow(workflow_data: dict) -> dict:
    """Validate workflow JSON structure"""
    errors = []
    warnings = []
    
    # Check required fields
    if "workflow" not in workflow_data:
        errors.append("Missing 'workflow' metadata")
    
    if "nodes" not in workflow_data:
        errors.append("Missing 'nodes' array")
    else:
        nodes = workflow_data["nodes"]
        node_ids = set()
        
        for i, node in enumerate(nodes):
            # Check required node fields
            if "id" not in node:
                errors.append(f"Node {i}: Missing 'id'")
            else:
                if node["id"] in node_ids:
                    errors.append(f"Duplicate node id: {node['id']}")
                node_ids.add(node["id"])
            
            if "function" not in node:
                errors.append(f"Node {node.get('id', i)}: Missing 'function'")
            
            # Check dependencies exist
            if "depends_on" in node:
                for dep in node["depends_on"]:
                    if dep not in node_ids and i > 0:
                        warnings.append(f"Node {node.get('id', i)}: Dependency '{dep}' not found")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }

def _get_workflow_templates() -> dict:
    """Get available workflow templates"""
    templates_dir = Path("workflows")
    templates = {}
    
    for json_file in templates_dir.glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                workflow_data = json.load(f)
                templates[json_file.stem] = {
                    "name": workflow_data.get("workflow", {}).get("name", json_file.stem),
                    "description": workflow_data.get("workflow", {}).get("description", ""),
                    "path": str(json_file),
                    "nodes_count": len(workflow_data.get("nodes", []))
                }
        except Exception as e:
            logger.warning(f"Could not load template {json_file}: {e}")
    
    return templates

def _create_workflow_from_nl(requirement: str, use_llm: bool = False) -> dict:
    """Create workflow from natural language using LLM or rule-based composition."""
    global llm_composer
    
    logger.info(f"Creating workflow from NL: {requirement} (LLM={use_llm})")
    
    if use_llm:
        # Initialize LLM composer if needed
        if llm_composer is None:
            from workflow_agent_llm import verify_ollama_connection
            if not verify_ollama_connection():
                return {
                    "error": "Ollama not available",
                    "suggestion": "Start Ollama: ollama serve && ollama pull qwen2.5-coder:7b",
                    "fallback": "Using rule-based composition",
                    "workflow": _create_workflow_rule_based(requirement)
                }
            llm_composer = LLMWorkflowComposer()
        
        try:
            workflow = llm_composer.compose_workflow(requirement)
        except Exception as e:
            logger.error(f"LLM composition failed: {e}")
            workflow = _create_workflow_rule_based(requirement)
    else:
        workflow = _create_workflow_rule_based(requirement)
    
    # Save workflow
    workflows_dir = Path("workflows")
    workflows_dir.mkdir(exist_ok=True)
    filename = f"mcp_nl_generated_{len(list(workflows_dir.glob('mcp_nl_*.json')))}.json"
    output_path = workflows_dir / filename
    
    with open(output_path, 'w') as f:
        json.dump(workflow, f, indent=2)
    
    return {
        "workflow": workflow,
        "saved_to": str(output_path),
        "method": "llm" if use_llm and llm_composer else "rule-based",
        "nodes": len(workflow.get("nodes", [])),
        "strategy": workflow.get("workflow", {}).get("strategy", "unknown")
    }

def _create_workflow_rule_based(requirement: str) -> dict:
    """Create workflow using rule-based composition (fallback)."""
    from workflow_agent import WorkflowGoal
    
    req_lower = requirement.lower()
    
    # Parse requirement
    task = "object_detection" if "detect" in req_lower else "object_detection"
    input_type = "video" if "video" in req_lower else "image"
    
    if "fast" in req_lower or "performance" in req_lower or "maximum" in req_lower:
        performance_target = 25.0
    elif "balanced" in req_lower or "good" in req_lower:
        performance_target = 20.0
    else:
        performance_target = 20.0
    
    flexibility = "flexible" in req_lower or "custom" in req_lower or "experiment" in req_lower
    
    goal = WorkflowGoal(
        task=task,
        input_type=input_type,
        output_type="display",
        performance_target=performance_target,
        flexibility_needed=flexibility,
        hardware_preference="auto"
    )
    
    return agentic_system.create_workflow_from_goal(goal)

def _optimize_workflow(workflow_type: str, target_fps: float = 20.0) -> dict:
    """Get optimization suggestions for a workflow."""
    logger.info(f"Optimizing {workflow_type} for {target_fps} FPS")
    
    # Get suggested parameters
    params = agentic_system.optimizer.suggest_parameters(workflow_type, target_fps=target_fps)
    
    # Get performance trend
    trend = agentic_system.optimizer.analyze_performance_trend(workflow_type)
    
    # Get strategy recommendation
    strategy, config = agentic_system.selector.select_strategy(
        fps_target=target_fps,
        flexibility_needed=True
    )
    
    return {
        "workflow_type": workflow_type,
        "target_fps": target_fps,
        "suggested_parameters": params,
        "performance_trend": trend,
        "recommended_strategy": strategy,
        "expected_fps": config["profile"]["typical_fps"],
        "strategy_rationale": config["rationale"]
    }

def _analyze_performance(workflow_type: str = None) -> dict:
    """Analyze performance history and trends."""
    logger.info(f"Analyzing performance (filter={workflow_type})")
    
    # Get insights from learner
    insights = agentic_system.learner.get_insights()
    
    # Add filtered view if requested
    if workflow_type:
        kb = agentic_system.learner.knowledge_base.get(workflow_type, {})
        insights["filtered"] = {
            "workflow_type": workflow_type,
            "knowledge": kb
        }
        
        # Add suggestions
        if kb and "avg_fps" in kb:
            suggestions = agentic_system.learner.suggest_optimizations(
                kb["avg_fps"],
                workflow_type
            )
            insights["filtered"]["suggestions"] = suggestions
    
    return insights

def _execute_workflow_with_learning(workflow_input: str, enable_learning: bool = True) -> dict:
    """Execute workflow with agentic learning enabled."""
    from agentic_integration import AgenticWorkflowEngine
    
    logger.info(f"Executing workflow with learning={enable_learning}")
    
    # Load workflow
    if os.path.exists(workflow_input):
        with open(workflow_input, 'r') as f:
            workflow_data = json.load(f)
    else:
        workflow_data = json.loads(workflow_input)
    
    # Execute with agentic engine
    engine = AgenticWorkflowEngine(workflow_data, enable_learning=enable_learning)
    results = engine.execute()
    
    # Extract performance metrics
    performance = {}
    for node_id, node_result in results.items():
        if isinstance(node_result, dict):
            if "fps" in node_result:
                performance["fps"] = node_result["fps"]
            elif "average_fps" in node_result:
                performance["fps"] = node_result["average_fps"]
            if "latency_ms" in node_result:
                performance["latency_ms"] = node_result["latency_ms"]
    
    return {
        "status": "completed",
        "performance": performance,
        "node_count": len(results),
        "learning_enabled": enable_learning,
        "workflow_name": workflow_data.get("workflow", {}).get("name", "Unknown")
    }

def _get_workflow_templates() -> dict:
    """Get available workflow templates"""
    templates_dir = Path("workflows")
    templates = {}
    
    for json_file in templates_dir.glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                workflow_data = json.load(f)
                templates[json_file.stem] = {
                    "name": workflow_data.get("workflow", {}).get("name", json_file.stem),
                    "description": workflow_data.get("workflow", {}).get("description", ""),
                    "path": str(json_file),
                    "nodes_count": len(workflow_data.get("nodes", []))
                }
        except Exception as e:
            logger.warning(f"Could not load template {json_file}: {e}")
    
    return templates

async def main():
    """Run MCP server"""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

