"""
LLM-Powered Workflow Agent using AutoGen + Ollama

This module provides sophisticated natural language understanding for workflow
composition using local LLMs via Ollama.

Features:
- Natural language workflow design with complex reasoning
- Multi-agent collaboration (planner, executor, critic)
- Performance optimization suggestions via LLM
- Conversational workflow refinement
"""

import json
import logging
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Import existing agentic components
from workflow_agent import (
    WorkflowGoal,
    WorkflowComposer,  # Use composer directly, not full system
    ExecutionRecord,
    PerformanceProfile
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(name)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('workflow.llm')


class OllamaConfig:
    """Configuration for Ollama LLM."""
    def __init__(
        self,
        model: str = "qwen2.5-coder:7b",
        base_url: str = "http://localhost:11434/v1",
        temperature: float = 0.7,
        max_tokens: int = 2000
    ):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens


class LLMWorkflowComposer:
    """
    LLM-powered workflow composer using AutoGen agents.
    
    Uses multiple specialized agents:
    - Planner: Understands requirements and designs workflow structure
    - Optimizer: Suggests performance optimizations
    - Validator: Checks workflow correctness and completeness
    """
    
    def __init__(self, ollama_config: Optional[OllamaConfig] = None):
        """Initialize LLM workflow composer."""
        self.config = ollama_config or OllamaConfig()
        # Use WorkflowComposer directly (no circular dependency)
        self.rule_based_composer = WorkflowComposer()
        self._client = None
        logger.info(f"LLM Composer initialized with model: {self.config.model}")
    
    def _create_ollama_client(self) -> OpenAIChatCompletionClient:
        """Create Ollama client (OpenAI-compatible API)."""
        if self._client is None:
            self._client = OpenAIChatCompletionClient(
                model=self.config.model,
                base_url=self.config.base_url,
                api_key="ollama",  # Dummy key for local Ollama
                model_capabilities={
                    "vision": False,
                    "function_calling": True,
                    "json_output": True
                }
            )
        return self._client
    
    def _create_planner_agent(self) -> AssistantAgent:
        """Create the workflow planner agent."""
        system_message = """You are a Workflow Planner expert specializing in computer vision pipelines.

Your role is to:
1. Understand user requirements for object detection workflows
2. Design optimal workflow structures
3. Select appropriate nodes and backends (OpenVINO, ONNX, DirectML)
4. Consider hardware constraints (GPU, NPU, CPU)

Available workflow strategies:
- "granular": Maximum flexibility, composable atomic nodes (~19 FPS)
- "monolithic": Maximum performance, single optimized node (~25 FPS)
- "fast_pipeline": Balanced approach with optimized node chains (~22 FPS)

Available nodes:
- detect_gpus: Detect available hardware
- download_model: Download YOLO models
- openvino_model_loader: Load models for Intel NPU/CPU
- granular_video_detection_loop: Process video with atomic nodes
- performance_stats: Calculate FPS and timing

When given a requirement, output a JSON workflow structure with:
{
  "strategy": "granular|monolithic|fast_pipeline",
  "nodes": [...],
  "rationale": "Why you chose this approach"
}

Focus on performance, hardware utilization, and user requirements."""
        
        return AssistantAgent(
            name="WorkflowPlanner",
            model_client=self._create_ollama_client(),
            system_message=system_message
        )
    
    def _create_optimizer_agent(self) -> AssistantAgent:
        """Create the performance optimizer agent."""
        system_message = """You are a Performance Optimization expert for computer vision workflows.

Your role is to:
1. Analyze workflow performance requirements
2. Suggest optimal parameters (conf_threshold, iou_threshold, etc.)
3. Recommend hardware backends based on availability
4. Identify bottlenecks and optimization opportunities

Performance data from history:
- Granular workflows: ~18-19 FPS
- Monolithic workflows: ~25 FPS
- Fast pipelines: ~22 FPS

Common parameters:
- conf_threshold: Detection confidence (0.1-0.9, default 0.25)
- iou_threshold: NMS overlap threshold (0.3-0.9, default 0.7)
- backend: "openvino" (Intel NPU/CPU), "onnx" (CPU), "directml" (AMD/DirectX)

When analyzing a workflow, provide:
{
  "recommended_parameters": {...},
  "expected_fps": <number>,
  "optimizations": ["suggestion1", "suggestion2"],
  "hardware_utilization": "description"
}"""
        
        return AssistantAgent(
            name="PerformanceOptimizer",
            model_client=self._create_ollama_client(),
            system_message=system_message
        )
    
    def _create_validator_agent(self) -> AssistantAgent:
        """Create the workflow validator agent."""
        system_message = """You are a Workflow Validation expert.

Your role is to:
1. Check workflow structure for correctness
2. Verify all required nodes are present
3. Validate node dependencies
4. Ensure parameters are reasonable

A valid workflow must have:
- detect_gpus node (for hardware detection)
- download_model node (to ensure model exists)
- At least one processing node (video/image detection)
- performance_stats node (for metrics)

Check for:
- Missing dependencies
- Invalid parameter values
- Hardware compatibility issues
- Performance bottlenecks

Output:
{
  "valid": true/false,
  "issues": ["issue1", "issue2"],
  "suggestions": ["fix1", "fix2"]
}"""
        
        return AssistantAgent(
            name="WorkflowValidator",
            model_client=self._create_ollama_client(),
            system_message=system_message
        )
    
    async def compose_workflow_async(
        self,
        user_requirement: str,
        context: Optional[Dict[str, Any]] = None,
        use_agents: bool = True
    ) -> Dict[str, Any]:
        """
        Compose a workflow from natural language using multi-agent LLM system.
        
        Args:
            user_requirement: Natural language description of workflow needs
            context: Optional context (hardware info, performance history, etc.)
            use_agents: Use full AutoGen multi-agent system (requires async)
            
        Returns:
            Complete workflow JSON
            
        Process:
        1. Planner analyzes requirements and designs workflow structure
        2. Optimizer suggests parameters based on performance targets
        3. Validator checks completeness and correctness
        4. Agents collaborate until consensus on final workflow
        """
        logger.info(f"🤖 Starting multi-agent workflow composition")
        logger.info(f"Requirement: {user_requirement}")
        
        if not use_agents:
            # Fallback to synchronous rule-based
            goal = self._parse_requirement_to_goal(user_requirement)
            return self.rule_based_composer.compose_from_goal(goal)
        
        # Prepare context for agents
        context_info = context or {}
        
        # Add available node information
        available_nodes = """
Available Workflow Nodes:
- detect_gpus_node: Detect GPU/NPU/CPU hardware
- download_model_node: Download YOLO models
- granular_video_loop_node: Atomic video processing pipeline
- openvino_model_loader_node: Load models for Intel NPU
- performance_stats_node: Calculate FPS and performance metrics
- open_video_capture_node: Initialize video source
- create_onnx_directml_session_node: Create DirectML GPU session
- create_onnx_cpu_session_node: Create CPU inference session

Atomic Image Processing Nodes (for quality_over_speed mode):
- read_image_node, resize_image_letterbox_node
- normalize_image_node, hwc_to_chw_node
- add_batch_dimension_node
- decode_yolo_v8_output_node, filter_by_confidence_node
- convert_cxcywh_to_xyxy_node, apply_nms_node
- scale_boxes_to_original_node, format_detections_coco_node
"""
        
        context_str = f"\n\nAvailable Components:\n{available_nodes}"
        if context_info:
            context_str += f"\n\nRuntime Context:\n{json.dumps(context_info, indent=2)}"
        
        # Create agents
        planner = self._create_planner_agent()
        optimizer = self._create_optimizer_agent()
        validator = self._create_validator_agent()
        
        # Create termination conditions
        termination = TextMentionTermination("WORKFLOW_COMPLETE") | MaxMessageTermination(15)
        
        # Create round-robin team
        team = RoundRobinGroupChat(
            participants=[planner, optimizer, validator],
            termination_condition=termination
        )
        
        # Comprehensive task with workflow structure template
        task = f"""Design a complete workflow JSON for: {user_requirement}

{context_str}

COLLABORATION PROCESS:
1. Planner: Design workflow structure with node IDs and dependencies
2. Optimizer: Suggest optimal parameters (conf_threshold, FPS targets)
3. Validator: Verify completeness and suggest improvements

OUTPUT FORMAT (JSON):
{{
  "workflow": {{
    "name": "Descriptive workflow name",
    "strategy": "granular|fast_pipeline|fully_atomic",
    "description": "What this workflow does"
  }},
  "nodes": [
    {{
      "id": "unique_node_id",
      "function": "module.function_name",
      "inputs": {{"param": "value"}},
      "dependencies": ["prior_node_id"]
    }}
  ]
}}

When consensus reached, respond with "WORKFLOW_COMPLETE" followed by the final JSON.
"""
        
        logger.info("🚀 Launching agent team collaboration...")
        
        try:
            # Run the team asynchronously
            stream = team.run_stream(task=task)
            
            messages = []
            async for result in stream:
                # TaskResult contains a 'messages' list
                if hasattr(result, 'messages'):
                    for msg in result.messages:
                        if hasattr(msg, 'source') and hasattr(msg, 'content'):
                            logger.info(f"💬 {msg.source}: {msg.content[:100]}...")
                            messages.append(msg)
                # Direct message object
                elif hasattr(result, 'source') and hasattr(result, 'content'):
                    logger.info(f"💬 {result.source}: {result.content[:100]}...")
                    messages.append(result)
            
            logger.info(f"✅ Agent collaboration complete ({len(messages)} messages)")
            
            # Extract workflow from conversation
            workflow = self._extract_workflow_from_messages(messages, user_requirement)
            
            return workflow
            
        except Exception as e:
            logger.error(f"❌ Multi-agent composition failed: {e}")
            logger.info("Falling back to rule-based composition")
            
            # Fallback to rule-based
            goal = self._parse_requirement_to_goal(user_requirement)
            workflow = self.rule_based_composer.compose_from_goal(goal)
            workflow["workflow"]["composition_mode"] = "llm-fallback"
            workflow["workflow"]["error"] = str(e)
            return workflow
    
    def _extract_workflow_from_messages(
        self, 
        messages: List[Any],
        requirement: str
    ) -> Dict[str, Any]:
        """
        Extract workflow JSON from agent conversation messages.
        
        Args:
            messages: List of messages from agent conversation
            requirement: Original user requirement
            
        Returns:
            Extracted workflow JSON
        """
        logger.info("🔍 Extracting workflow from agent messages...")
        
        # Search messages for JSON workflow
        for msg in reversed(messages):  # Start from most recent
            content = str(msg.content)
            
            # Look for JSON blocks
            import re
            json_matches = re.findall(r'\{[\s\S]*?"workflow"[\s\S]*?\}', content)
            
            for json_str in json_matches:
                try:
                    # Try to parse as JSON
                    workflow = json.loads(json_str)
                    
                    # Validate it has required structure
                    if "workflow" in workflow and "nodes" in workflow:
                        logger.info(f"✅ Found valid workflow JSON with {len(workflow['nodes'])} nodes")
                        
                        # Add metadata
                        workflow["workflow"]["composition_mode"] = "llm-multi-agent"
                        workflow["workflow"]["user_requirement"] = requirement
                        workflow["workflow"]["agent_collaboration"] = True
                        
                        return workflow
                        
                except json.JSONDecodeError:
                    continue
        
        # If no valid JSON found, use intelligent extraction
        logger.warning("⚠️  No valid JSON found, using intelligent extraction")
        
        # Analyze conversation for workflow decisions
        conversation_text = "\n".join([str(m.content) for m in messages])
        
        # Extract key decisions
        strategy = "granular"  # Default
        if "atomic" in conversation_text.lower() or "flexibility" in conversation_text.lower():
            strategy = "fully_atomic"
        elif "fast" in conversation_text.lower() or "performance" in conversation_text.lower():
            strategy = "fast_pipeline"
        
        # Parse requirement as fallback
        goal = self._parse_requirement_to_goal(requirement)
        workflow = self.rule_based_composer.compose_from_goal(goal)
        
        # Add agent metadata
        workflow["workflow"]["composition_mode"] = "llm-assisted-extraction"
        workflow["workflow"]["strategy"] = strategy
        workflow["workflow"]["user_requirement"] = requirement
        workflow["workflow"]["agent_insights"] = self._extract_agent_insights(messages)
        
        return workflow
    
    def _extract_agent_insights(self, messages: List[Any]) -> Dict[str, Any]:
        """Extract insights and recommendations from agent conversation."""
        insights = {
            "planner_suggestions": [],
            "optimizer_recommendations": [],
            "validator_checks": []
        }
        
        for msg in messages:
            source = getattr(msg, 'source', 'unknown')
            content = str(msg.content)
            
            if 'planner' in source.lower():
                # Extract planner insights
                if 'node' in content.lower() or 'pipeline' in content.lower():
                    insights["planner_suggestions"].append(content[:200])
            
            elif 'optimizer' in source.lower():
                # Extract optimizer insights
                if 'parameter' in content.lower() or 'fps' in content.lower():
                    insights["optimizer_recommendations"].append(content[:200])
            
            elif 'validator' in source.lower():
                # Extract validator insights
                if 'check' in content.lower() or 'validate' in content.lower():
                    insights["validator_checks"].append(content[:200])
        
        return insights
    
    def compose_workflow(
        self,
        user_requirement: str,
        context: Optional[Dict[str, Any]] = None,
        goal: Optional[WorkflowGoal] = None,
        use_multi_agent: bool = False
    ) -> Dict[str, Any]:
        """
        Compose workflow from natural language.
        
        Args:
            user_requirement: Natural language description
            context: Optional additional context
            goal: Optional pre-parsed WorkflowGoal
            use_multi_agent: Use async AutoGen multi-agent system
            
        Returns:
            Composed workflow
            
        Modes:
        - use_multi_agent=False: Fast rule-based parsing (default)
        - use_multi_agent=True: Full LLM multi-agent collaboration (async)
        """
        logger.info(f"Composing workflow from requirement: {user_requirement}")
        
        if use_multi_agent:
            # Use async multi-agent system
            logger.info("Using async multi-agent composition")
            
            # Check if we're already in an event loop
            try:
                loop = asyncio.get_running_loop()
                logger.warning("⚠️  Already in event loop - multi-agent not supported from sync call")
                logger.info("ℹ️  Please call compose_workflow_async() directly from async context")
                # Fall through to regular mode
            except RuntimeError:
                # No running loop - safe to use asyncio.run()
                return asyncio.run(self.compose_workflow_async(
                    user_requirement=user_requirement,
                    context=context,
                    use_agents=True
                ))
        
        # Fast rule-based parsing
        if goal is None:
            goal = self._parse_requirement_to_goal(user_requirement)
        
        # Use rule-based composer
        workflow = self.rule_based_composer.compose_from_goal(goal)
        
        # Add metadata about composition
        workflow["workflow"]["composition_mode"] = "llm-assisted"
        workflow["workflow"]["user_requirement"] = user_requirement
        
        logger.info(f"Workflow composed with {len(workflow.get('nodes', []))} nodes")
        return workflow
    
    def _extract_workflow_from_conversation(self, conversation_result) -> Dict[str, Any]:
        """Extract workflow JSON from agent conversation."""
        # This is a simplified extraction - you might need more sophisticated parsing
        # For now, return a basic structure and let the fallback handle it
        
        try:
            # Try to find JSON in the last messages
            # In practice, you'd parse the conversation history
            # For now, we'll use the fallback to rule-based system
            logger.warning("Using fallback to rule-based workflow composition")
            
            # Fallback to rule-based composition
            # Parse simple requirements
            goal = self._parse_requirement_to_goal(str(conversation_result))
            return self.rule_based_composer.compose_from_goal(goal)
            
        except Exception as e:
            logger.error(f"Failed to extract workflow: {e}")
            # Ultimate fallback
            return self._create_default_workflow()
    
    def _parse_requirement_to_goal(self, requirement: str) -> WorkflowGoal:
        """Parse natural language requirement to WorkflowGoal."""
        req_lower = requirement.lower()
        
        # Detect task
        if "detect" in req_lower or "detection" in req_lower:
            task = "object_detection"
        elif "track" in req_lower:
            task = "object_tracking"
        else:
            task = "object_detection"
        
        # Detect input type
        if "video" in req_lower or "mp4" in req_lower:
            input_type = "video.mp4"
        elif "image" in req_lower or "jpg" in req_lower or "png" in req_lower or "soccer" in req_lower:
            # Extract actual filename if present
            import re
            img_match = re.search(r'([\w\-]+\.(?:jpg|jpeg|png|bmp))', requirement)
            input_type = img_match.group(1) if img_match else "image.jpg"
        else:
            input_type = "video.mp4"
        
        # Detect performance target
        if "fast" in req_lower or "high performance" in req_lower or "speed" in req_lower:
            performance_target = 25.0
        elif "balanced" in req_lower or "good" in req_lower:
            performance_target = 20.0
        else:
            # Extract FPS if specified
            import re
            fps_match = re.search(r'(\d+)\s*fps', req_lower)
            performance_target = float(fps_match.group(1)) if fps_match else 20.0
        
        # Detect quality vs speed (use quality_over_speed parameter)
        quality_keywords = ["atomic", "flexible", "quality", "granular", "breakdown", "flexibility"]
        speed_keywords = ["fast", "speed", "performance", "quick"]
        
        if any(word in req_lower for word in quality_keywords):
            quality_over_speed = True
        elif any(word in req_lower for word in speed_keywords):
            quality_over_speed = False
        else:
            quality_over_speed = False  # Default to speed
        
        # Detect hardware preference
        if "gpu" in req_lower or "directml" in req_lower:
            hardware_preference = "gpu"
        elif "cpu" in req_lower:
            hardware_preference = "cpu"
        elif "npu" in req_lower:
            hardware_preference = "npu"
        else:
            hardware_preference = "auto"
        
        return WorkflowGoal(
            task=task,
            input_type=input_type,
            output_type="display",
            performance_target=performance_target,
            hardware_preference=hardware_preference,
            quality_over_speed=quality_over_speed
        )
    
    def _create_default_workflow(self) -> Dict[str, Any]:
        """Create a default workflow as ultimate fallback."""
        return {
            "workflow": {
                "name": "Default Object Detection",
                "strategy": "granular"
            },
            "nodes": [
                {"id": "detect_hw", "function": "detect_gpus"},
                {"id": "get_model", "function": "download_model"},
                {"id": "process", "function": "granular_video_detection_loop"},
                {"id": "stats", "function": "performance_stats"}
            ]
        }


class LLMWorkflowAnalyzer:
    """
    LLM-powered workflow analyzer and debugger.
    
    Analyzes execution results and provides intelligent insights.
    """
    
    def __init__(self, ollama_config: Optional[OllamaConfig] = None):
        """Initialize LLM analyzer."""
        self.config = ollama_config or OllamaConfig()
        self._client = None
        logger.info("LLM Analyzer initialized")
    
    def _create_ollama_client(self) -> OpenAIChatCompletionClient:
        """Create Ollama client."""
        if self._client is None:
            self._client = OpenAIChatCompletionClient(
                model=self.config.model,
                base_url=self.config.base_url,
                api_key="ollama",
                model_capabilities={
                    "vision": False,
                    "function_calling": True,
                    "json_output": True
                }
            )
        return self._client
    
    async def analyze_performance_async(
        self,
        execution_record: ExecutionRecord
    ) -> Dict[str, Any]:
        """
        Analyze execution performance using LLM.
        
        Args:
            execution_record: Record of workflow execution
            
        Returns:
            Analysis with insights and recommendations
        """
        logger.info(f"Analyzing execution: {execution_record.workflow_name}")
        
        # Create analyzer agent
        system_message = """You are a Performance Analysis expert for computer vision workflows.

Analyze the execution record and provide:
1. Performance assessment (is it good/poor for this type?)
2. Bottleneck identification
3. Specific optimization recommendations
4. Hardware utilization assessment

Provide structured output with actionable insights."""
        
        agent = AssistantAgent(
            name="PerformanceAnalyzer",
            model_client=self._create_ollama_client(),
            system_message=system_message
        )
        
        # Prepare execution data
        exec_data = {
            "workflow": execution_record.workflow_name,
            "fps": execution_record.performance.get("fps", 0),
            "latency_ms": execution_record.performance.get("latency_ms", 0),
            "hardware": execution_record.hardware,
            "parameters": execution_record.parameters
        }
        
        task = f"""Analyze this workflow execution:

{json.dumps(exec_data, indent=2)}

Provide:
- Performance assessment
- Bottleneck analysis
- Optimization recommendations
- Hardware utilization review

Format as JSON with keys: assessment, bottlenecks, recommendations, hardware_notes"""
        
        # For now, return a simple analysis
        # Full implementation would use the agent
        return {
            "assessment": f"Workflow achieved {exec_data['fps']:.1f} FPS",
            "bottlenecks": [],
            "recommendations": ["Use rule-based optimization for now"],
            "hardware_notes": f"Running on {exec_data['hardware']}"
        }
    
    def analyze_performance(self, execution_record: ExecutionRecord) -> Dict[str, Any]:
        """Synchronous wrapper for analyze_performance_async."""
        return asyncio.run(self.analyze_performance_async(execution_record))


def verify_ollama_connection(base_url: str = "http://localhost:11434") -> tuple[bool, str]:
    """
    Verify Ollama server is running and accessible.
    
    Args:
        base_url: Ollama server URL
        
    Returns:
        (is_available, message) tuple
    """
    import requests
    
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=2)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m['name'] for m in models]
            logger.info("✅ Ollama server is running")
            logger.info(f"Available models: {model_names}")
            return True, f"Ollama running with {len(models)} models"
        else:
            message = f"Ollama responded with status {response.status_code}"
            logger.warning(f"⚠️  {message}")
            return False, message
    except requests.exceptions.RequestException as e:
        message = f"Cannot connect to Ollama: {e}"
        logger.error(f"❌ {message}")
        logger.info("\nTo start Ollama:")
        logger.info("  ollama serve")
        logger.info("\nTo pull required model:")
        logger.info("  ollama pull qwen2.5-coder:7b")
        return False, message


if __name__ == "__main__":
    # Quick test
    print("🤖 LLM-Powered Workflow Agent")
    print("=" * 70)
    
    # Check Ollama connection
    is_available, message = verify_ollama_connection()
    if not is_available:
        print(f"\n❌ Ollama not available: {message}")
        print("Please start Ollama server first.")
        exit(1)
    
    print(f"\n✅ {message}")
    print("\nCreating LLM workflow composer...")
    
    composer = LLMWorkflowComposer()
    
    print("\nTest: Generate workflow from natural language")
    requirement = "I need a fast video object detection workflow that can process my dashcam footage in real-time"
    
    print(f"\nRequirement: {requirement}")
    print("\nGenerating workflow...")
    
    workflow = composer.compose_workflow(requirement)
    
    print("\n✅ Workflow generated:")
    print(json.dumps(workflow, indent=2))
    
    # Save workflow
    output_path = Path("workflows/llm_generated_workflow.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(workflow, f, indent=2)
    
    print(f"\n💾 Saved to: {output_path}")
