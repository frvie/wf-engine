#!/usr/bin/env python3
"""
Workflow Engine Command-Line Interface
========================================

Simple CLI for creating and executing workflows.

Usage:
    python workflow_cli.py create --task object_detection --input video.mp4 --fps 30
    python workflow_cli.py create --nl "Detect objects in dashcam footage at 25 FPS"
    python workflow_cli.py execute workflows/my_workflow.json
    python workflow_cli.py list
"""

import argparse
import json
import sys
from pathlib import Path
import logging

from workflow_agent import AgenticWorkflowSystem, WorkflowGoal

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)


def create_workflow(args):
    """Create a workflow from command-line arguments."""
    print("🔨 Creating workflow...")
    print("=" * 70)
    
    # Initialize system
    system = AgenticWorkflowSystem(enable_llm=args.llm)
    
    if args.natural_language:
        # Natural language composition
        if not system.llm_available:
            print("❌ LLM not available. Start Ollama: ollama serve")
            print("⚠️  Falling back to rule-based composition")
        
        print(f"📝 Natural Language: {args.natural_language}")
        workflow = system.create_workflow_from_natural_language(args.natural_language)
        
    else:
        # Rule-based composition
        goal = WorkflowGoal(
            task=args.task,
            input_type=args.input,
            output_type=args.output,
            performance_target=args.fps,
            hardware_preference=args.hardware,
            quality_over_speed=args.quality
        )
        
        print(f"🎯 Task: {args.task}")
        print(f"📥 Input: {args.input}")
        print(f"⚡ Target FPS: {args.fps}")
        print(f"🖥️  Hardware: {args.hardware}")
        print(f"🎨 Quality Mode: {args.quality}")
        
        workflow = system.create_workflow_from_goal(goal)
    
    # Display results
    print("\n✅ Workflow Created!")
    print("-" * 70)
    print(f"Name: {workflow['workflow']['name']}")
    print(f"Mode: {workflow['workflow'].get('composition_mode', 'rule-based')}")
    print(f"Strategy: {workflow['workflow']['strategy']}")
    print(f"Nodes: {len(workflow['nodes'])}")
    
    # List nodes
    print("\n📦 Nodes:")
    for i, node in enumerate(workflow['nodes'], 1):
        deps = f" (depends on: {', '.join(node.get('dependencies', []))})" if node.get('dependencies') else ""
        print(f"  {i}. {node['id']}: {node['function']}{deps}")
    
    # Save workflow
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(workflow, f, indent=2)
    
    print(f"\n💾 Saved to: {output_file}")
    print("\nNext steps:")
    print(f"  Execute: python workflow_cli.py execute {output_file}")
    
    return workflow


def execute_workflow(args):
    """Execute a workflow from JSON file."""
    print("🚀 Executing workflow...")
    print("=" * 70)
    
    workflow_path = Path(args.workflow_file)
    
    if not workflow_path.exists():
        print(f"❌ Workflow file not found: {workflow_path}")
        sys.exit(1)
    
    print(f"📂 Loading: {workflow_path}")
    
    # Load workflow
    with open(workflow_path) as f:
        workflow = json.load(f)
    
    print(f"📋 Workflow: {workflow['workflow']['name']}")
    print(f"📦 Nodes: {len(workflow['nodes'])}")
    
    # Display workflow nodes
    print("\n🔗 Workflow Pipeline:")
    for i, node in enumerate(workflow['nodes'], 1):
        deps = f" (after: {', '.join(node.get('dependencies', []))})" if node.get('dependencies') else ""
        print(f"  {i}. {node['id']}{deps}")
    
    # Try to execute using function_workflow_engine
    try:
        from function_workflow_engine import run_function_workflow
        
        print(f"\n⚡ Executing workflow...")
        print("-" * 70)
        
        result = run_function_workflow(str(workflow_path))
        
        print("\n✅ Execution Complete!")
        
        if isinstance(result, dict):
            if 'status' in result:
                print(f"Status: {result['status']}")
            
            if 'results' in result:
                print("\n📊 Results:")
                for key, value in result['results'].items():
                    if isinstance(value, (int, float, str)):
                        print(f"  {key}: {value}")
                    else:
                        print(f"  {key}: {type(value).__name__}")
        else:
            print(f"Result: {result}")
    
    except ImportError as e:
        print(f"\n⚠️  Function workflow engine not available: {e}")
        print("\nTo execute workflows, ensure function_workflow_engine.py is available")
        print("\nAlternative: Run directly with:")
        print(f"  python function_workflow_engine.py {workflow_path}")
    
    except Exception as e:
        print(f"\n❌ Execution failed: {e}")
        import traceback
        if args.verbose:
            traceback.print_exc()
        else:
            print("Run with --verbose for full traceback")


def list_workflows(args):
    """List available workflows."""
    print("📚 Available Workflows")
    print("=" * 70)
    
    workflows_dir = Path("workflows")
    
    if not workflows_dir.exists():
        print("No workflows directory found")
        return
    
    workflow_files = list(workflows_dir.glob("*.json"))
    
    if not workflow_files:
        print("No workflows found")
        return
    
    for i, wf_file in enumerate(workflow_files, 1):
        try:
            with open(wf_file) as f:
                wf = json.load(f)
            
            print(f"\n{i}. {wf_file.name}")
            print(f"   Name: {wf['workflow']['name']}")
            print(f"   Nodes: {len(wf['nodes'])}")
            print(f"   Strategy: {wf['workflow'].get('strategy', 'N/A')}")
            print(f"   Mode: {wf['workflow'].get('composition_mode', 'N/A')}")
        
        except Exception as e:
            print(f"\n{i}. {wf_file.name} (Error reading: {e})")


def main():
    parser = argparse.ArgumentParser(
        description="Workflow Engine CLI - Create and execute computer vision workflows",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create rule-based workflow
  python workflow_cli.py create --task object_detection --input video.mp4 --fps 30

  # Create with natural language (requires LLM)
  python workflow_cli.py create --nl "Detect objects in dashcam at 25 FPS"

  # Create high-quality image workflow
  python workflow_cli.py create --task object_detection --input image.jpg --quality

  # Execute a workflow
  python workflow_cli.py execute workflows/my_workflow.json

  # List all workflows
  python workflow_cli.py list
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Create command
    create_parser = subparsers.add_parser('create', help='Create a new workflow')
    create_parser.add_argument('--task', default='object_detection', 
                              help='Task type (default: object_detection)')
    create_parser.add_argument('--input', default='video.mp4',
                              help='Input file/source (default: video.mp4)')
    create_parser.add_argument('--output', default='display',
                              help='Output type (default: display)')
    create_parser.add_argument('--fps', type=float, default=25.0,
                              help='Target FPS (default: 25.0)')
    create_parser.add_argument('--hardware', default='auto',
                              choices=['auto', 'gpu', 'cpu', 'npu'],
                              help='Hardware preference (default: auto)')
    create_parser.add_argument('--quality', action='store_true',
                              help='Use quality mode (atomic nodes)')
    create_parser.add_argument('--nl', '--natural-language', dest='natural_language',
                              help='Create from natural language description')
    create_parser.add_argument('--llm', action='store_true', default=True,
                              help='Enable LLM support (default: True)')
    create_parser.add_argument('--output-file', default='workflows/generated_workflow.json',
                              help='Output file path (default: workflows/generated_workflow.json)')
    
    # Execute command
    execute_parser = subparsers.add_parser('execute', help='Execute a workflow')
    execute_parser.add_argument('workflow_file', help='Path to workflow JSON file')
    execute_parser.add_argument('--verbose', '-v', action='store_true',
                               help='Show detailed error messages')
    
    # List command
    subparsers.add_parser('list', help='List available workflows')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command == 'create':
        create_workflow(args)
    elif args.command == 'execute':
        execute_workflow(args)
    elif args.command == 'list':
        list_workflows(args)


if __name__ == "__main__":
    main()
