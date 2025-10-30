"""
Demo: LLM-Powered Workflow Engine with AutoGen + Ollama + MCP

This demo shows the complete integration of:
1. LLM-powered workflow composition (AutoGen + Ollama)
2. MCP server exposing tools
3. Agentic learning and optimization

Prerequisites:
- Ollama running: ollama serve
- Model downloaded: ollama pull qwen2.5-coder:7b
"""

import json
import asyncio
from pathlib import Path

print("🤖 LLM + AutoGen + MCP Integration Demo")
print("=" * 80)

# =============================================================================
# Step 1: Verify Ollama Connection
# =============================================================================

print("\n📡 Step 1: Verifying Ollama Connection")
print("-" * 80)

from src.agentic.agent_llm import verify_ollama_connection

ollama_available = verify_ollama_connection()

if not ollama_available:
    print("\n⚠️  Ollama is not running. LLM features will use fallback.")
    print("\n  To enable LLM features:")
    print("    1. Start Ollama: ollama serve")
    print("    2. Pull model: ollama pull qwen2.5-coder:7b")
    print("\n  Continuing with rule-based composition...")
    use_llm = False
else:
    print("\n✅ Ollama is ready for LLM-powered workflows!")
    use_llm = True

# =============================================================================
# Step 2: Test Rule-Based Workflow Composition (Baseline)
# =============================================================================

print("\n\n📝 Step 2: Rule-Based Workflow Composition (Baseline)")
print("-" * 80)

from workflow_agent import AgenticWorkflowSystem

system = AgenticWorkflowSystem()

requirement1 = "I need fast video object detection for real-time processing"

print(f"\nRequirement: \"{requirement1}\"")
print("\nUsing: Rule-based composition")

from workflow_agent import WorkflowGoal

goal = WorkflowGoal(
    task="object_detection",
    input_type="sample.mp4",  # Video file extension
    output_type="display",
    performance_target=25.0,
    hardware_preference="auto",
    quality_over_speed=False
)

workflow1 = system.create_workflow_from_goal(goal)

print("\n✅ Generated workflow:")
print(f"  - Strategy: {workflow1['workflow']['strategy']}")
print(f"  - Nodes: {len(workflow1['nodes'])}")
print("  - Expected FPS: ~25")

# Save workflow
path1 = Path("workflows/demo_rule_based.json")
with open(path1, 'w') as f:
    json.dump(workflow1, f, indent=2)
print(f"  - Saved to: {path1}")

# =============================================================================
# Step 3: Test LLM-Powered Workflow Composition (if available)
# =============================================================================

if use_llm:
    print("\n\n🧠 Step 3: LLM-Powered Workflow Composition")
    print("-" * 80)
    
    from workflow_agent_llm import LLMWorkflowComposer
    
    composer = LLMWorkflowComposer()
    
    requirement2 = "Create a flexible pipeline for dashcam footage analysis with good performance and the ability to customize detection parameters"
    
    print(f"\nRequirement: \"{requirement2}\"")
    print("\nUsing: LLM (AutoGen + Ollama qwen2.5-coder:7b)")
    print("\nGenerating workflow...")
    
    workflow2 = composer.compose_workflow(requirement2)
    
    print(f"\n✅ LLM generated workflow:")
    print(f"  - Strategy: {workflow2.get('workflow', {}).get('strategy', 'unknown')}")
    print(f"  - Nodes: {len(workflow2.get('nodes', []))}")
    
    # Save workflow
    path2 = Path("workflows/demo_llm_generated.json")
    with open(path2, 'w') as f:
        json.dump(workflow2, f, indent=2)
    print(f"  - Saved to: {path2}")
else:
    print("\n\n⏭️  Step 3: Skipped (Ollama not available)")

# =============================================================================
# Step 4: Test MCP Server Tools (Simulated)
# =============================================================================

print("\n\n🔧 Step 4: MCP Server Tool Capabilities")
print("-" * 80)

print("\nMCP Server exposes 9 tools:")
print("\n  1. execute_workflow")
print("     Execute workflows from JSON file or string")
print("\n  2. detect_devices")
print("     Detect available hardware (NPU, DirectML, CUDA, CPU)")
print("\n  3. list_workflow_nodes")
print("     List all available workflow nodes")
print("\n  4. validate_workflow")
print("     Validate workflow JSON structure")
print("\n  5. get_workflow_templates")
print("     Get available workflow templates")
print("\n  6. create_workflow_from_nl [NEW]")
print("     Create workflow from natural language (with optional LLM)")
print("\n  7. optimize_workflow [NEW]")
print("     Get optimization suggestions from execution history")
print("\n  8. analyze_performance [NEW]")
print("     Analyze performance trends")
print("\n  9. execute_workflow_with_learning [NEW]")
print("     Execute with agentic learning enabled")

print("\n\nTo use MCP server with Claude Desktop:")
print("\n1. Add to Claude Desktop config (claude_desktop_config.json):")
print("""
{
  "mcpServers": {
    "workflow-engine": {
      "command": "uv",
      "args": ["run", "python", "mcp_server.py"],
      "cwd": "C:/dev/workflow_engine"
    }
  }
}
""")

print("\n2. Restart Claude Desktop")
print("\n3. Use tools like:")
print('   "Create a fast video detection workflow"')
print('   "Optimize my granular workflow for 20 FPS"')
print('   "Analyze performance trends"')

# =============================================================================
# Step 5: Test Agentic Optimization
# =============================================================================

print("\n\n⚙️  Step 5: Agentic Optimization & Learning")
print("-" * 80)

from workflow_agent import AgenticWorkflowSystem

system = AgenticWorkflowSystem()

print("\nCurrent Knowledge Base:")
insights = system.learner.get_insights()

print(f"\n  Total Executions: {insights['total_executions']}")
print(f"  Workflow Types Learned: {len(insights['workflow_types'])}")

for wf_type in insights['workflow_types']:
    kb = system.learner.knowledge_base.get(wf_type, {})
    print(f"\n  {wf_type}:")
    print(f"    - Avg FPS: {kb.get('avg_fps', 0):.1f}")
    print(f"    - Best FPS: {kb.get('best_fps', 0):.1f}")
    print(f"    - Success Rate: {kb.get('success_rate', 0):.1%}")

# Get optimization suggestions
if insights['total_executions'] > 0:
    print("\n\nOptimization Suggestions for 20 FPS target:")
    params = system.optimizer.suggest_parameters('granular', target_fps=20.0)
    
    for key, value in params.items():
        print(f"  - {key}: {value}")

# =============================================================================
# Step 6: Demonstrate Natural Language Workflow Creation
# =============================================================================

print("\n\n💬 Step 6: Natural Language Workflow Creation")
print("-" * 80)

test_prompts = [
    "Detect objects in video with maximum performance",
    "Flexible detection pipeline for experiments",
    "Process dashcam footage using Intel NPU if available",
    "Real-time object tracking with good accuracy"
]

from agentic_integration import create_workflow_from_natural_language

for i, prompt in enumerate(test_prompts, 1):
    print(f"\n{i}. \"{prompt}\"")
    
    try:
        workflow = create_workflow_from_natural_language(prompt)
        strategy = workflow.get('workflow', {}).get('strategy', 'unknown')
        node_count = len(workflow.get('nodes', []))
        print(f"   → Strategy: {strategy}, Nodes: {node_count}")
    except Exception as e:
        print(f"   → Error: {e}")

# =============================================================================
# Step 7: Performance Comparison
# =============================================================================

print("\n\n📊 Step 7: Performance Comparison")
print("-" * 80)

print("\n┌─────────────────────┬───────────┬──────────────┬────────────┐")
print("│ Workflow Type       │ Avg FPS   │ Flexibility  │ Use Case   │")
print("├─────────────────────┼───────────┼──────────────┼────────────┤")
print("│ Monolithic          │ ~25 FPS   │ Low          │ Max Speed  │")
print("│ Fast Pipeline       │ ~22 FPS   │ Medium       │ Balanced   │")
print("│ Granular (Atomic)   │ ~19 FPS   │ High         │ Custom     │")
print("└─────────────────────┴───────────┴──────────────┴────────────┘")

print("\n\nLLM Enhancement Benefits:")
print("  ✅ Natural language understanding (complex requirements)")
print("  ✅ Multi-agent reasoning (planner, optimizer, validator)")
print("  ✅ Context-aware suggestions (considers hardware, history)")
print("  ✅ Conversational refinement (iterative workflow design)")

# =============================================================================
# Summary
# =============================================================================

print("\n\n" + "=" * 80)
print("📋 SUMMARY: What We Built")
print("=" * 80)

print("\n✅ LLM-Powered Workflow Composer (workflow_agent_llm.py)")
print("   - AutoGen multi-agent system")
print("   - Ollama for local AI reasoning")
print("   - Planner, Optimizer, Validator agents")

print("\n✅ Enhanced MCP Server (mcp_server.py)")
print("   - 9 tools exposed to LLM clients")
print("   - Natural language workflow creation")
print("   - Performance optimization suggestions")
print("   - Execution history analysis")

print("\n✅ Agentic System Preserved (workflow_agent.py)")
print("   - Rule-based composition (fallback)")
print("   - Performance tracking & learning")
print("   - Adaptive strategy selection")

print("\n✅ Integration Layer (agentic_integration.py)")
print("   - AgenticWorkflowEngine with learning")
print("   - Natural language parser")
print("   - CLI interface")

print("\n\n🚀 How to Use:")
print("\n1. Basic (Rule-Based):")
print('   python agentic_integration.py workflows/granular_video_detection_mp4.json')

print("\n2. With LLM (Natural Language):")
print('   python -c "from agentic_integration import *; create_workflow_from_natural_language(\'fast video detection\')"')

print("\n3. MCP Server (for Claude Desktop):")
print('   uv run python mcp_server.py')

print("\n4. Direct LLM Composition:")
print('   python workflow_agent_llm.py')

print("\n\n✨ Next Steps:")
print("  • Start Ollama for full LLM capabilities")
print("  • Configure Claude Desktop to use MCP server")
print("  • Run more workflows to build knowledge base")
print("  • Experiment with complex natural language requirements")

print("\n" + "=" * 80)
print("Demo complete! 🎉")
print("=" * 80)
