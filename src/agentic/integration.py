"""
Integration module for Agentic Workflow System with Function Workflow Engine.

Adds autonomous capabilities to existing workflows:
- Monitors performance in real-time
- Suggests optimizations
- Records execution history
- Learns from past runs
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any
from src.agentic.agent import (
    AgenticWorkflowSystem,
    WorkflowGoal,
    ExecutionRecord
)
from src.core.engine import FunctionWorkflowEngine


class AgenticWorkflowEngine(FunctionWorkflowEngine):
    """
    Enhanced workflow engine with agentic capabilities.
    
    Extends FunctionWorkflowEngine with:
    - Auto-optimization
    - Performance learning
    - Adaptive strategy selection
    """
    
    def __init__(self, workflow_data: Dict = None, enable_learning: bool = True):
        super().__init__(workflow_data)
        
        self.enable_learning = enable_learning
        if enable_learning:
            self.agent_system = AgenticWorkflowSystem()
            self.logger.info("✨ Agentic capabilities enabled")
        else:
            self.agent_system = None
    
    def execute(self) -> Dict[str, Any]:
        """Execute workflow with agentic monitoring."""
        import time
        
        # Track start time
        start_time = time.time()
        
        # Execute workflow normally
        results = super().execute()
        
        # Record and learn from execution if enabled
        if self.enable_learning and results:
            execution_time = time.time() - start_time
            self._record_and_learn(results, execution_time)
        
        return results
    
    def _record_and_learn(self, results: Dict[str, Any], execution_time: float):
        """Record execution and get learning insights."""
        # Extract performance metrics
        performance = {
            "execution_time": execution_time,
            "fps": 0,
            "latency_ms": execution_time * 1000
        }
        
        # Try to extract FPS from results
        for node_id, node_result in results.items():
            if isinstance(node_result, dict):
                if "fps" in node_result:
                    performance["fps"] = node_result["fps"]
                elif "average_fps" in node_result:
                    performance["fps"] = node_result["average_fps"]
        
        # Extract workflow type from config
        workflow_type = self.workflow_config.get("strategy", "unknown")
        workflow_name = self.workflow_config.get("name", "Unknown Workflow")
        
        # Get parameters from first processing node
        parameters = {}
        for node in self.nodes:
            if "video" in node.get("function", "").lower():
                parameters = node.get("inputs", {})
                break
        
        # Detect available hardware
        hardware = {
            "gpu": False,
            "npu": False,
            "cpu": True
        }
        
        try:
            import onnxruntime as ort
            available_providers = ort.get_available_providers()
            hardware["gpu"] = "DmlExecutionProvider" in available_providers or "CUDAExecutionProvider" in available_providers
            hardware["npu"] = "OpenVINOExecutionProvider" in available_providers
        except:
            pass
        
        # Record execution
        analysis = self.agent_system.analyze_execution(
            workflow_name=workflow_name,
            workflow_type=workflow_type,
            performance=performance,
            parameters=parameters,
            hardware=hardware
        )
        
        # Log insights
        if analysis.get("suggestions"):
            self.logger.info("\n" + "="*70)
            self.logger.info("🤖 AGENTIC SYSTEM INSIGHTS")
            self.logger.info("="*70)
            for suggestion in analysis["suggestions"]:
                self.logger.info(f"  💡 {suggestion}")
            self.logger.info("="*70 + "\n")


def create_workflow_from_natural_language(description: str) -> Dict[str, Any]:
    """
    Create workflow from natural language description.
    
    Examples:
        "Detect objects in my video with good performance"
        "Process webcam for real-time detection"
        "Analyze video file for objects, prioritize accuracy"
    
    Args:
        description: Natural language description
        
    Returns:
        Generated workflow JSON
    """
    agent = AgenticWorkflowSystem()
    
    # Parse description (simple keyword matching for now)
    description_lower = description.lower()
    
    # Determine input type
    if "webcam" in description_lower or "real-time" in description_lower or "camera" in description_lower:
        input_type = "webcam"
    elif "video" in description_lower or "mp4" in description_lower or "file" in description_lower:
        input_type = "videos/istockphoto-1585137173-640_adpp_is.mp4"  # Use existing sample video
    elif "image" in description_lower or "photo" in description_lower or "picture" in description_lower:
        input_type = "image"
    else:
        input_type = "videos/istockphoto-1585137173-640_adpp_is.mp4"  # default to sample video
    
    # Determine output type
    if "save" in description_lower or "file" in description_lower:
        output_type = "file"
    else:
        output_type = "display"
    
    # Determine performance vs quality
    quality_over_speed = (
        "quality" in description_lower or
        "accuracy" in description_lower or
        "accurate" in description_lower
    )
    
    performance_target = None
    if "fast" in description_lower or "performance" in description_lower:
        performance_target = 20.0
    elif "slow" in description_lower or quality_over_speed:
        performance_target = 15.0
    
    # Create goal
    goal = WorkflowGoal(
        task="object_detection",
        input_type=input_type,
        output_type=output_type,
        performance_target=performance_target,
        hardware_preference="auto",
        quality_over_speed=quality_over_speed
    )
    
    # Generate workflow
    workflow = agent.create_workflow_from_goal(goal)
    
    # Add natural language context
    workflow["workflow"]["original_request"] = description
    
    return workflow


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    """Interactive CLI for agentic workflow system."""
    import sys
    
    print("🤖 Agentic Workflow Engine")
    print("=" * 70)
    
    if len(sys.argv) > 1:
        # Check if it's a natural language description
        if sys.argv[1].endswith('.json'):
            # Execute existing workflow with learning
            workflow_path = sys.argv[1]
            print(f"📋 Executing workflow: {workflow_path}")
            print(f"✨ Agentic learning: ENABLED\n")
            
            with open(workflow_path, 'r') as f:
                workflow_data = json.load(f)
            
            engine = AgenticWorkflowEngine(workflow_data, enable_learning=True)
            results = engine.execute()
            
            print("\n✅ Workflow completed with agentic insights!")
            
        else:
            # Treat as natural language description
            description = " ".join(sys.argv[1:])
            print(f"📝 Natural language request:")
            print(f"   \"{description}\"\n")
            
            workflow = create_workflow_from_natural_language(description)
            
            # Save workflow
            output_path = Path("workflows/nl_generated_workflow.json")
            with open(output_path, 'w') as f:
                json.dump(workflow, f, indent=2)
            
            print(f"\n✅ Generated workflow saved to: {output_path}")
            print(f"   Strategy: {workflow['workflow']['strategy']}")
            print(f"   Nodes: {len(workflow['nodes'])}")
            print(f"\n🚀 Run it with:")
            print(f"   uv run python agentic_integration.py {output_path}")
    
    else:
        print("Usage:")
        print("  1. Execute existing workflow with learning:")
        print("     uv run python agentic_integration.py workflows/granular_video_detection_mp4.json")
        print("")
        print("  2. Generate workflow from natural language:")
        print("     uv run python agentic_integration.py \"Detect objects in video with good performance\"")
        print("")
        print("Examples:")
        print('  uv run python agentic_integration.py "Process webcam for real-time detection"')
        print('  uv run python agentic_integration.py "Analyze video, prioritize accuracy over speed"')


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    main()


