"""
Test Full AutoGen Multi-Agent System
====================================

This demonstrates the async multi-agent workflow composition where:
- Planner designs the workflow structure
- Optimizer suggests optimal parameters
- Validator checks completeness and correctness

The agents collaborate in a round-robin fashion to create the best workflow.
"""
import asyncio
from workflow_agent import AgenticWorkflowSystem, WorkflowGoal
from workflow_agent_llm import LLMWorkflowComposer
import json
from pathlib import Path

async def test_multi_agent_async():
    """Test async multi-agent composition."""
    print("=" * 70)
    print("🤖 MULTI-AGENT WORKFLOW COMPOSITION TEST")
    print("=" * 70)
    
    # Initialize system
    system = AgenticWorkflowSystem(enable_llm=True)
    
    if not system.llm_available:
        print("\n❌ LLM not available - Ollama must be running")
        print("Start Ollama: ollama serve")
        return
    
    print(f"\n✅ LLM Available: Multi-agent system ready")
    print(f"   Agents: Planner, Optimizer, Validator")
    print(f"   Mode: AutoGen RoundRobinGroupChat")
    
    # Test 1: Simple video detection (async direct)
    print("\n" + "=" * 70)
    print("Test 1: Simple Video Detection (Multi-Agent - Async)")
    print("=" * 70)
    
    requirement1 = "Create a fast video object detection workflow for dashcam footage at 25 FPS"
    
    print(f"\n📝 Requirement: {requirement1}")
    print("\n🚀 Launching multi-agent collaboration...")
    print("   (This may take 10-30 seconds as agents discuss)")
    
    # Use async API directly
    if system.llm_composer:
        workflow1 = await system.llm_composer.compose_workflow_async(
            user_requirement=requirement1,
            context={"performance_target": 25.0, "priority": "speed"},
            use_agents=True
        )
    
    print(f"\n✅ Multi-agent composition complete!")
    print(f"   Mode: {workflow1['workflow'].get('composition_mode', 'N/A')}")
    print(f"   Name: {workflow1['workflow']['name']}")
    print(f"   Nodes: {len(workflow1['nodes'])}")
    print(f"   Strategy: {workflow1['workflow'].get('strategy', 'N/A')}")
    
    if 'agent_insights' in workflow1['workflow']:
        insights = workflow1['workflow']['agent_insights']
        print(f"\n💡 Agent Insights:")
        if insights.get('planner_suggestions'):
            print(f"   Planner: {len(insights['planner_suggestions'])} suggestions")
        if insights.get('optimizer_recommendations'):
            print(f"   Optimizer: {len(insights['optimizer_recommendations'])} recommendations")
        if insights.get('validator_checks'):
            print(f"   Validator: {len(insights['validator_checks'])} checks")
    
    # Save
    output1 = Path('workflows/multi_agent_video.json')
    with open(output1, 'w') as f:
        json.dump(workflow1, f, indent=2)
    print(f"\n💾 Saved to: {output1}")
    
    # Test 2: Atomic image detection (async direct)
    print("\n" + "=" * 70)
    print("Test 2: Atomic Image Detection (Multi-Agent - Async)")
    print("=" * 70)
    
    requirement2 = "Process soccer.jpg with maximum flexibility using atomic nodes for detailed control"
    
    print(f"\n📝 Requirement: {requirement2}")
    print("\n🚀 Launching multi-agent collaboration...")
    
    if system.llm_composer:
        workflow2 = await system.llm_composer.compose_workflow_async(
            user_requirement=requirement2,
            context={"input": "soccer.jpg", "quality_over_speed": True},
            use_agents=True
        )
    
    print(f"\n✅ Multi-agent composition complete!")
    print(f"   Mode: {workflow2['workflow'].get('composition_mode', 'N/A')}")
    print(f"   Name: {workflow2['workflow']['name']}")
    print(f"   Nodes: {len(workflow2['nodes'])}")
    
    if 'node_breakdown' in workflow2['workflow']:
        breakdown = workflow2['workflow']['node_breakdown']
        print(f"\n📊 Node Breakdown:")
        for category, count in breakdown.items():
            if category != 'total':
                print(f"      {category.title()}: {count} nodes")
    
    # Save
    output2 = Path('workflows/multi_agent_atomic.json')
    with open(output2, 'w') as f:
        json.dump(workflow2, f, indent=2)
    print(f"\n💾 Saved to: {output2}")
    
    # Test 3: Direct async API
    print("\n" + "=" * 70)
    print("Test 3: Direct Async API (LLMWorkflowComposer)")
    print("=" * 70)
    
    if system.llm_composer:
        requirement3 = "High-performance real-time detection for security camera at 30 FPS on GPU"
        
        print(f"\n📝 Requirement: {requirement3}")
        print("\n🚀 Calling async compose_workflow_async directly...")
        
        workflow3 = await system.llm_composer.compose_workflow_async(
            user_requirement=requirement3,
            context={"hardware": "GPU available", "priority": "speed"},
            use_agents=True
        )
        
        print(f"\n✅ Async composition complete!")
        print(f"   Mode: {workflow3['workflow'].get('composition_mode', 'N/A')}")
        print(f"   Name: {workflow3['workflow']['name']}")
        print(f"   Nodes: {len(workflow3['nodes'])}")
        
        # Save
        output3 = Path('workflows/multi_agent_async_direct.json')
        with open(output3, 'w') as f:
            json.dump(workflow3, f, indent=2)
        print(f"\n💾 Saved to: {output3}")
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 MULTI-AGENT SYSTEM SUMMARY")
    print("=" * 70)
    
    print(f"""
✅ Tests Complete!

Multi-Agent Composition Features:
  🤖 3 Specialized Agents (Planner, Optimizer, Validator)
  💬 Round-robin collaborative discussion
  🎯 Intelligent workflow design from natural language
  📋 Agent insights and recommendations captured
  ⚡ Async execution for complex reasoning
  🔄 Automatic fallback to rule-based if needed

Generated Workflows:
  1. Video Detection: {workflow1['workflow']['name']}
     - Nodes: {len(workflow1['nodes'])}
     - Target: 25 FPS
     - Mode: {workflow1['workflow'].get('composition_mode', 'N/A')}
  
  2. Atomic Image: {workflow2['workflow']['name']}
     - Nodes: {len(workflow2['nodes'])}
     - Quality: Atomic breakdown
     - Mode: {workflow2['workflow'].get('composition_mode', 'N/A')}
  
  3. Security Camera: {workflow3['workflow']['name']}
     - Nodes: {len(workflow3['nodes'])}
     - Target: 30 FPS
     - Mode: {workflow3['workflow'].get('composition_mode', 'N/A')}

Next Steps:
  - Integrate with MCP server for Claude Desktop
  - Add real-time parameter tuning
  - Enable workflow refinement through conversation
""")

def test_multi_agent():
    """Synchronous wrapper for async tests."""
    asyncio.run(test_multi_agent_async())

if __name__ == "__main__":
    import sys
    
    print("🚀 Starting Multi-Agent Workflow Composition Tests\n")
    
    try:
        test_multi_agent()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
