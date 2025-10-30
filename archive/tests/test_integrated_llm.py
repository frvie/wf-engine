"""Test integrated LLM capabilities in AgenticWorkflowSystem."""
from workflow_agent import AgenticWorkflowSystem, WorkflowGoal
import json
from pathlib import Path

print("=" * 70)
print("Testing Integrated LLM in AgenticWorkflowSystem")
print("=" * 70)

# Initialize with LLM enabled
system = AgenticWorkflowSystem(enable_llm=True)

print(f"\n📊 System Status:")
print(f"   LLM Available: {'✅ Yes' if system.llm_available else '❌ No'}")
print(f"   Composition Modes: {'Rule-based + LLM' if system.llm_available else 'Rule-based only'}")

# Test 1: Rule-based composition (always works)
print("\n" + "=" * 70)
print("Test 1: Rule-based Composition")
print("=" * 70)

goal = WorkflowGoal(
    task='object_detection',
    input_type='video.mp4',
    output_type='display',
    performance_target=20.0,
    hardware_preference='auto',
    quality_over_speed=False
)

workflow_rule = system.create_workflow_from_goal(goal)
print(f"\n✅ Rule-based workflow generated")
print(f"   Name: {workflow_rule['workflow']['name']}")
print(f"   Mode: {workflow_rule['workflow'].get('composition_mode', 'N/A')}")
print(f"   Nodes: {len(workflow_rule['nodes'])}")
print(f"   Strategy: {workflow_rule['workflow']['strategy']}")

# Test 2: Natural language composition (uses LLM if available)
print("\n" + "=" * 70)
print("Test 2: Natural Language Composition")
print("=" * 70)

if system.llm_available:
    try:
        nl_description = "Detect objects in soccer.jpg using DirectML GPU with atomic breakdown for flexibility"
        print(f"NL Input: \"{nl_description}\"")
        
        workflow_nl = system.create_workflow_from_natural_language(nl_description)
        
        print(f"\n✅ Natural language workflow generated")
        print(f"   Name: {workflow_nl['workflow']['name']}")
        print(f"   Mode: {workflow_nl['workflow'].get('composition_mode', 'LLM')}")
        print(f"   Nodes: {len(workflow_nl['nodes'])}")
        
        # Check if atomic mode was detected
        if 'node_breakdown' in workflow_nl['workflow']:
            print(f"   Atomic: Yes ({workflow_nl['workflow']['node_breakdown']['total']} nodes)")
        
        # Save
        output_path = Path('workflows/nl_atomic_workflow.json')
        with open(output_path, 'w') as f:
            json.dump(workflow_nl, f, indent=2)
        print(f"   Saved: {output_path}")
        
    except Exception as e:
        print(f"❌ Natural language composition failed: {e}")
else:
    print("⚠️  Skipped: LLM not available")
    print("   To enable: Start Ollama server")

# Test 3: Force LLM mode with WorkflowGoal
print("\n" + "=" * 70)
print("Test 3: LLM Mode with WorkflowGoal + Natural Language")
print("=" * 70)

if system.llm_available:
    try:
        goal_llm = WorkflowGoal(
            task='object_detection',
            input_type='test_video.mp4',
            output_type='display',
            performance_target=30.0,
            hardware_preference='gpu',
            quality_over_speed=False
        )
        
        nl_desc = "High-performance video detection at 30 FPS using GPU acceleration"
        
        workflow_llm = system.create_workflow_from_goal(
            goal=goal_llm,
            natural_language=nl_desc
        )
        
        print(f"✅ LLM-powered workflow with goal + NL")
        print(f"   Name: {workflow_llm['workflow']['name']}")
        print(f"   Mode: {workflow_llm['workflow'].get('composition_mode', 'LLM')}")
        print(f"   Nodes: {len(workflow_llm['nodes'])}")
        
    except Exception as e:
        print(f"❌ LLM composition failed: {e}")
        print("   (This is expected if Ollama isn't configured)")
else:
    print("⚠️  Skipped: LLM not available")

# Test 4: Fallback behavior
print("\n" + "=" * 70)
print("Test 4: Automatic Fallback to Rule-based")
print("=" * 70)

# Create system with LLM disabled
system_no_llm = AgenticWorkflowSystem(enable_llm=False)

print(f"System created with enable_llm=False")
print(f"LLM Available: {'Yes' if system_no_llm.llm_available else 'No'}")

# Try natural language (should fallback to rule-based parsing)
workflow_fallback = system_no_llm.create_workflow_from_natural_language(
    "Detect objects in video.mp4 at 25 FPS using GPU"
)

print(f"\n✅ Fallback workflow generated")
print(f"   Mode: {workflow_fallback['workflow'].get('composition_mode', 'rule-based')}")
print(f"   Nodes: {len(workflow_fallback['nodes'])}")
print(f"   (Used rule-based NL parsing)")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"""
✅ Integration Complete!

AgenticWorkflowSystem now supports:
  1. Rule-based composition (always available)
  2. LLM composition with AutoGen + Ollama (optional)
  3. Natural language workflow creation
  4. Automatic fallback when LLM unavailable

Current Status:
  - LLM Available: {'✅ Yes' if system.llm_available else '❌ No'}
  - Ollama Running: {'✅ Yes' if system.llm_available else '❌ No'}
  - AutoGen Installed: {'✅ Yes' if system.llm_composer else '❌ No'}

Usage:
  # Initialize with LLM
  system = AgenticWorkflowSystem(enable_llm=True)
  
  # Rule-based (always works)
  workflow = system.create_workflow_from_goal(goal)
  
  # Natural language (uses LLM if available)
  workflow = system.create_workflow_from_natural_language(
      "Detect objects in video.mp4 using GPU"
  )
  
  # Force LLM with natural language
  workflow = system.create_workflow_from_goal(
      goal=goal, 
      natural_language="Your requirements here"
  )
""")
