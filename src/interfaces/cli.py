#!/usr/bin/env python
"""
Workflow Engine CLI - Simplified Interface

Quick commands for all workflow operations:
- wf run <workflow>              Execute a workflow
- wf create "<description>"      Create workflow from natural language
- wf optimize <workflow>         Get optimization suggestions
- wf status                      Show agent knowledge & performance
- wf devices                     List available hardware
- wf nodes                       List available workflow nodes
- wf mcp                         Start MCP server for AI agents
"""

import sys
import json
import argparse
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('wf.cli')


def cmd_run(args):
    """Execute a workflow with optional learning."""
    from src.core.engine import run_function_workflow
    from src.agentic.integration import AgenticWorkflowEngine
    
    workflow_path = Path(args.workflow)
    
    if not workflow_path.exists():
        logger.error(f"Workflow not found: {workflow_path}")
        return 1
    
    print(f"\n🚀 Executing workflow: {workflow_path.name}")
    print("=" * 80)
    
    if args.learn:
        # Use agentic engine with learning
        with open(workflow_path) as f:
            workflow_data = json.load(f)
        
        engine = AgenticWorkflowEngine(workflow_data, enable_learning=True)
        results = engine.execute()
        
        print("\n✅ Execution complete with learning enabled")
        print("\n📊 Use 'wf status' to see what the agent learned")
    else:
        # Standard execution
        results = run_function_workflow(str(workflow_path))
        print("\n✅ Execution complete")
    
    return 0


def cmd_create(args):
    """Create workflow from natural language description."""
    import asyncio
    
    description = args.description
    output_path = args.output
    
    print(f"\n🤖 Creating complete workflow from description:")
    print(f"   \"{description}\"")
    print("=" * 80)
    
    try:
        if args.agentic:
            # Use new agentic system with node generation
            from src.agentic.node_id_demo import AgenticWorkflowGenerator
            
            generator = AgenticWorkflowGenerator()
            result = asyncio.run(generator.create_complete_workflow(description))
            
            if result['success']:
                workflow_file = Path(result['workflow_file'])
                
                print(f"\n✅ Complete workflow created successfully!")
                print(f"   📄 Workflow: {workflow_file}")
                
                gen_result = result['generation_result']
                print(f"   🔧 Generated nodes: {len(gen_result['generated_nodes'])}")
                
                if gen_result['generated_nodes']:
                    print(f"\n📝 Created node files:")
                    for node in gen_result['generated_nodes']:
                        print(f"      • {node['node_id']} → {node['file_path']}")
                
                test_result = result['test_result']
                if test_result['success']:
                    print(f"\n� Workflow test: PASSED")
                else:
                    print(f"\n⚠️  Workflow test: FAILED")
                    if test_result.get('error'):
                        print(f"    Error: {test_result['error']}")
                
                if args.run:
                    print(f"\n▶️  Executing complete workflow...")
                    from src.core.engine import run_function_workflow
                    run_function_workflow(str(workflow_file))
                else:
                    print(f"\n💡 Run with: wfe run {workflow_file}")
                
                return 0
            else:
                print(f"\n❌ Agentic workflow creation failed")
                if 'generation_result' in result:
                    failed = result['generation_result'].get('failed_nodes', [])
                    if failed:
                        print(f"   Failed nodes:")
                        for node in failed:
                            print(f"      • {node['node_id']}: {node['error']}")
                return 1
        
        else:
            # Use existing integration system (workflow template only)
            from src.agentic.integration import create_workflow_from_natural_language
            
            workflow = create_workflow_from_natural_language(description)
            
            # Save workflow
            if output_path:
                save_path = Path(output_path)
            else:
                # Auto-generate filename
                safe_name = "".join(c if c.isalnum() or c in (' ', '_') else '_' for c in description[:50])
                safe_name = safe_name.replace(' ', '_').lower()
                save_path = Path(f"workflows/generated_{safe_name}.json")
            
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'w') as f:
                json.dump(workflow, f, indent=2)
            
            print(f"\n✅ Workflow template created successfully!")
            print(f"   Strategy: {workflow.get('workflow', {}).get('strategy', 'unknown')}")
            print(f"   Nodes: {len(workflow.get('nodes', []))}")
            print(f"   Saved to: {save_path}")
            print(f"\n⚠️  Note: You may need to implement missing nodes")
            print(f"   Use --agentic flag for automatic node generation")
            
            if args.run:
                print(f"\n▶️  Executing workflow...")
                from src.core.engine import run_function_workflow
                run_function_workflow(str(save_path))
            else:
                print(f"\n💡 Run with: wfe run {save_path}")
            
            return 0
        
    except Exception as e:
        logger.error(f"Failed to create workflow: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        return 1


def cmd_optimize(args):
    """Get optimization suggestions for a workflow."""
    from src.agentic.agent import AgenticWorkflowSystem
    
    workflow_path = Path(args.workflow)
    
    if not workflow_path.exists():
        logger.error(f"Workflow not found: {workflow_path}")
        return 1
    
    print(f"\n⚙️  Analyzing workflow: {workflow_path.name}")
    print("=" * 80)
    
    system = AgenticWorkflowSystem()
    
    # Load workflow to get type
    with open(workflow_path) as f:
        workflow_data = json.load(f)
    
    workflow_type = workflow_data.get('workflow', {}).get('strategy', 'unknown')
    
    # Get performance trend
    trend = system.optimizer.analyze_performance_trend(workflow_type)
    
    print(f"\n📈 Performance Trend ({workflow_type}):")
    print(f"   Status: {trend.get('trend', 'unknown')}")
    print(f"   Average FPS: {trend.get('avg_fps', 0):.1f}")
    print(f"   Range: {trend.get('min_fps', 0):.1f} - {trend.get('max_fps', 0):.1f} FPS")
    
    # Get suggestions
    target_fps = args.target if args.target else 25.0
    params = system.optimizer.suggest_parameters(workflow_type, target_fps)
    
    print(f"\n💡 Optimization Suggestions (target: {target_fps} FPS):")
    for key, value in params.items():
        print(f"   {key}: {value}")
    
    # Get insights
    insights = system.learner.get_insights()
    
    if insights['recommendations']:
        print("\n🎯 Recommendations:")
        for rec in insights['recommendations']:
            print(f"   • {rec['type']}: {rec['suggestion']}")
    
    return 0


def cmd_status(args):
    """Show agent status, knowledge base, and performance history."""
    from src.agentic.agent import AgenticWorkflowSystem
    
    print("\n📊 WORKFLOW ENGINE STATUS")
    print("=" * 80)
    
    system = AgenticWorkflowSystem()
    insights = system.learner.get_insights()
    
    print(f"\n📚 Knowledge Base:")
    print(f"   Total executions: {insights['total_executions']}")
    print(f"   Workflow types learned: {len(insights['workflow_types'])}")
    
    if insights['workflow_types']:
        print(f"\n   Learned workflows:")
        for wf_type in insights['workflow_types']:
            kb = system.learner.knowledge_base.get(wf_type, {})
            print(f"\n   • {wf_type}:")
            print(f"     - Avg FPS: {kb.get('avg_fps', 0):.1f}")
            print(f"     - Best FPS: {kb.get('best_fps', 0):.1f}")
            print(f"     - Success Rate: {kb.get('success_rate', 0):.1%}")
            print(f"     - Executions: {kb.get('execution_count', 0)}")
    
    if insights['recommendations']:
        print(f"\n💡 Active Recommendations ({len(insights['recommendations'])}):")
        for rec in insights['recommendations']:
            print(f"   • {rec['type']}: {rec['suggestion']}")
    
    # Check Ollama status
    print(f"\n🧠 LLM Capabilities:")
    from src.agentic.agent_llm import verify_ollama_connection
    if verify_ollama_connection():
        print("   ✅ Ollama running (LLM-powered composition available)")
    else:
        print("   ⚠️  Ollama not running (using rule-based composition)")
    
    return 0


def cmd_devices(args):
    """Detect and list available inference devices."""
    from src.nodes.detect_gpus import detect_gpus_node
    
    print("\n💻 AVAILABLE HARDWARE")
    print("=" * 80)
    
    devices = detect_gpus_node()
    
    print("\n🎮 GPU Devices:")
    if devices.get('recommended_gpu'):
        gpu_info = devices['recommended_gpu']
        print(f"   • {gpu_info.get('name', 'Unknown')} ({gpu_info.get('memory_total_gb', 'N/A')} GB)")
        print(f"     DirectML Device ID: {devices.get('directml_device_id', 0)}")
        print(f"     CUDA Device ID: {devices.get('cuda_device_id', 0)}")
        print(f"     Type: {'Discrete' if gpu_info.get('is_discrete') else 'Integrated'} GPU")
    else:
        print("   No GPU detected")
    
    if devices.get('has_directml'):
        print("\n✅ DirectML: Available")
        print(f"   Device ID: {devices.get('directml_device_id', 0)}")
    
    if devices.get('has_openvino_npu'):
        npu_devices = devices.get('npu_devices', [])
        print("\n✅ OpenVINO NPU: Available")
        for npu in npu_devices:
            print(f"   • {npu}")
    
    if devices.get('has_cuda'):
        print("\n✅ CUDA: Available")
        print(f"   Device ID: {devices.get('cuda_device_id', 0)}")
    
    print(f"\n💡 CPU: Always available (fallback)")
    
    return 0


def cmd_nodes(args):
    """List all available workflow nodes."""
    import importlib
    import inspect
    import sys
    from pathlib import Path
    
    # Add current directory to Python path for imports
    current_dir = Path.cwd()
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    print("\n🔧 AVAILABLE WORKFLOW NODES")
    print("=" * 80)
    
    nodes_dir = Path("src/nodes")
    discovered_nodes = {}
    
    # Scan recursively for all Python files
    for py_file in nodes_dir.rglob("*.py"):
        if py_file.name == '__init__.py' or py_file.name.startswith('_'):
            continue
        
        try:
            # Convert path to module name (src.nodes.xxx)
            # py_file is already relative to the project root
            module_name = str(py_file.with_suffix('')).replace('\\', '.').replace('/', '.')
            
            module = importlib.import_module(module_name)
            
            # Look for functions with workflow_node decorator
            for name, obj in inspect.getmembers(module):
                if inspect.isfunction(obj) and hasattr(obj, 'node_id'):
                    node_id = obj.node_id
                    # Use the file name as category
                    category = py_file.stem
                    
                    if category not in discovered_nodes:
                        discovered_nodes[category] = []
                    
                    discovered_nodes[category].append({
                        'id': node_id,
                        'function': name,
                        'module': module_name,
                        'dependencies': getattr(obj, 'dependencies', []),
                        'isolation': getattr(obj, 'isolation_mode', 'none')
                    })
        
        except Exception:
            # Silently skip files that can't be imported
            pass
    
    # Display by category
    category_names = {
        'infrastructure': 'INFRASTRUCTURE',
        'model_loaders': 'MODEL LOADERS',
        'atomic': 'ATOMIC OPERATIONS',
        'utils': 'UTILITIES',
        'video': 'VIDEO PROCESSING'
    }
    
    total_nodes = 0
    for category in sorted(discovered_nodes.keys()):
        nodes_list = discovered_nodes[category]
        if nodes_list:
            print(f"\n{category_names.get(category, category.upper())}:")
            for node in sorted(nodes_list, key=lambda x: x['id']):
                deps = f" (deps: {', '.join(node['dependencies'])})" if node['dependencies'] else ""
                iso = f" [isolated]" if node['isolation'] == 'subprocess' else ""
                print(f"   • {node['id']}{deps}{iso}")
                total_nodes += 1
    
    print(f"\n📝 Total nodes discovered: {total_nodes}")
    print(f"   Scanned directory: {nodes_dir.absolute()}")
    
    return 0


def cmd_mcp(args):
    """Start MCP server for AI agent integration."""
    print("\n🔌 STARTING MCP SERVER")
    print("=" * 80)
    print("\nMCP server will expose workflow engine tools to AI agents")
    print("Connect from Claude Desktop, Cline, or any MCP-compatible client")
    print("\nPress Ctrl+C to stop\n")
    
    import subprocess
    try:
        subprocess.run([sys.executable, "start_mcp.py"])
    except KeyboardInterrupt:
        print("\n\n🛑 MCP server stopped")
    
    return 0


def cmd_workflows(args):
    """List available workflows."""
    workflows_dir = Path("workflows")
    
    print("\n📋 AVAILABLE WORKFLOWS")
    print("=" * 80)
    
    workflows = list(workflows_dir.glob("*.json"))
    
    for workflow in sorted(workflows):
        try:
            with open(workflow) as f:
                data = json.load(f)
            
            strategy = data.get('workflow', {}).get('strategy', 'unknown')
            nodes = len(data.get('nodes', []))
            
            print(f"\n• {workflow.name}")
            print(f"  Strategy: {strategy}, Nodes: {nodes}")
            
        except Exception as e:
            print(f"\n• {workflow.name} (invalid: {e})")
    
    print(f"\n💡 Use with: wfe run workflows/<workflow>.json")
    
    return 0


def cmd_generate(args):
    """Generate a new workflow node from description."""
    from src.nodes.node_generator import NodeGenerator, NodeSpec
    import asyncio
    
    print(f"\n🔨 GENERATING WORKFLOW NODE")
    print("=" * 80)
    print(f"Goal: {args.description}")
    
    # Parse inputs and outputs
    inputs = args.inputs.split(',') if args.inputs else ['input']
    outputs = args.outputs.split(',') if args.outputs else ['output']
    
    print(f"Inputs: {', '.join(inputs)}")
    print(f"Outputs: {', '.join(outputs)}")
    print(f"Category: {args.category}")
    print(f"Model: {args.model}")
    
    # Create specification
    spec = NodeSpec(
        goal=args.description,
        inputs=inputs,
        outputs=outputs,
        category=args.category,
        description=args.detailed_description,
        constraints=args.constraints.split(',') if args.constraints else None
    )
    
    # Generate the node
    generator = NodeGenerator(model_name=args.model)
    
    try:
        result = asyncio.run(generator.generate_node(spec))
        
        if result['success']:
            print(f"\n✅ Node generated successfully!")
            print(f"   Name: {result['node_name']}")
            print(f"   File: {result['file_path']}")
            print(f"\n💡 The node is immediately available for use in workflows.")
            print(f"   Run 'wfe nodes' to see all available nodes.")
            
            if args.show_code:
                print(f"\n📄 Generated Code:")
                print("=" * 80)
                print(result['code'])
            
            return 0
        else:
            print(f"\n❌ Failed to generate node: {result['error']}")
            return 1
            
    except Exception as e:
        logger.error(f"Node generation failed: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        return 1


def cmd_analyze(args):
    """Analyze workflow and generate missing nodes."""
    from src.agentic.node_id_demo import WorkflowNodeAnalyzer
    import asyncio
    
    workflow_path = Path(args.workflow)
    
    if not workflow_path.exists():
        logger.error(f"Workflow not found: {workflow_path}")
        return 1
    
    print(f"\n🔍 ANALYZING WORKFLOW FOR MISSING NODES")
    print("=" * 80)
    print(f"Workflow: {workflow_path}")
    
    analyzer = WorkflowNodeAnalyzer(args.model)
    
    try:
        result = asyncio.run(
            analyzer.generate_missing_nodes_from_workflow(workflow_path)
        )
        
        analysis = result['analysis']
        print(f"\n📊 Analysis Results:")
        print(f"   Workflow: {analysis['workflow_name']}")
        print(f"   Total nodes: {analysis['total_nodes']}")
        print(f"   Existing: {len(analysis['existing_nodes'])}")
        print(f"   Missing: {len(analysis['missing_nodes'])}")
        
        if result['generated_nodes']:
            print(f"\n✅ Generated Nodes:")
            for node in result['generated_nodes']:
                print(f"   • {node['node_id']} → {node['file_path']}")
        
        if result['failed_nodes']:
            print(f"\n❌ Failed to Generate:")
            for node in result['failed_nodes']:
                print(f"   • {node['node_id']}: {node['error']}")
        
        if result['success']:
            print(f"\n🎉 All missing nodes generated successfully!")
            print(f"   The workflow is now ready to execute.")
            
            if args.test:
                print(f"\n🧪 Testing complete workflow...")
                from src.core.engine import run_function_workflow
                test_result = run_function_workflow(str(workflow_path))
                if test_result:
                    print(f"✅ Workflow test passed!")
                else:
                    print(f"❌ Workflow test failed!")
            else:
                print(f"\n💡 Test with: wfe run {workflow_path}")
        
        return 0 if result['success'] else 1
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        return 1


def cmd_interactive(args):
    """Start interactive workflow creation mode."""
    import asyncio
    from src.agentic.node_id_demo import AgenticWorkflowGenerator
    
    print("\n🤖 INTERACTIVE AGENTIC WORKFLOW GENERATOR")
    print("=" * 80)
    print("Enter natural language descriptions to create complete workflows!")
    print("Type 'quit', 'exit', or 'q' to stop.\n")
    
    generator = AgenticWorkflowGenerator(args.model)
    
    async def interactive_loop():
        while True:
            try:
                description = input("📝 Describe your workflow: ").strip()
                
                if description.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not description:
                    print("Please enter a workflow description.\n")
                    continue
                
                print(f"\n🔄 Creating workflow: '{description}'...")
                
                result = await generator.create_complete_workflow(description)
                
                if result['success']:
                    print("✅ Workflow created successfully!")
                    print(f"   📄 File: {result['workflow_file']}")
                    
                    gen_result = result['generation_result']
                    print(f"   🔧 Generated: {len(gen_result['generated_nodes'])} nodes")
                    
                    if args.auto_run:
                        print("   🏃 Auto-running workflow...")
                        from src.core.engine import run_function_workflow
                        run_function_workflow(result['workflow_file'])
                    
                    print()
                else:
                    print("❌ Workflow creation failed.\n")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}\n")
    
    try:
        asyncio.run(interactive_loop())
        return 0
    except Exception as e:
        logger.error(f"Interactive mode failed: {e}", exc_info=True)
        return 1


def main():
    parser = argparse.ArgumentParser(
        description='Workflow Engine CLI - Simplified Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Execute workflows
  wfe run workflows/granular_parallel_inference.json
  
  # Create workflows (template only)
  wfe create "fast video detection with NPU"
  
  # Create complete workflows (with node generation)  
  wfe create "web scraping pipeline" --agentic --run
  wfe create "text analysis system" --agentic -o my_workflow.json
  
  # Analyze existing workflows
  wfe analyze workflows/incomplete_workflow.json --test
  
  # Interactive workflow creation
  wfe interactive --auto-run
  
  # Generate individual nodes
  wfe generate "apply gaussian blur" -i image -o blurred_image
  
  # System information
  wfe status
  wfe devices
  wfe nodes
  wfe workflows
  wfe mcp
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Execute a workflow')
    run_parser.add_argument('workflow', help='Path to workflow JSON file')
    run_parser.add_argument('--learn', action='store_true', help='Enable agentic learning')
    run_parser.set_defaults(func=cmd_run)
    
    # Create command
    create_parser = subparsers.add_parser('create', help='Create workflow from description')
    create_parser.add_argument('description', help='Natural language description')
    create_parser.add_argument('-o', '--output', help='Output path for workflow JSON')
    create_parser.add_argument('--run', action='store_true', help='Execute immediately after creation')
    create_parser.add_argument('--agentic', action='store_true',
                               help='Use agentic system with node generation')
    create_parser.set_defaults(func=cmd_create)
    
    # Optimize command
    optimize_parser = subparsers.add_parser('optimize', help='Get optimization suggestions')
    optimize_parser.add_argument('workflow', help='Path to workflow JSON file')
    optimize_parser.add_argument('--target', type=float, help='Target FPS')
    optimize_parser.set_defaults(func=cmd_optimize)
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show agent status and knowledge')
    status_parser.set_defaults(func=cmd_status)
    
    # Devices command
    devices_parser = subparsers.add_parser('devices', help='List available hardware')
    devices_parser.set_defaults(func=cmd_devices)
    
    # Nodes command
    nodes_parser = subparsers.add_parser('nodes', help='List available workflow nodes')
    nodes_parser.set_defaults(func=cmd_nodes)
    
    # MCP command
    mcp_parser = subparsers.add_parser('mcp', help='Start MCP server')
    mcp_parser.set_defaults(func=cmd_mcp)
    
    # Workflows command
    workflows_parser = subparsers.add_parser('workflows', help='List available workflows')
    workflows_parser.set_defaults(func=cmd_workflows)
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate a new workflow node')
    generate_parser.add_argument('description', help='What the node should do')
    generate_parser.add_argument('-i', '--inputs', default='input', help='Comma-separated input names (default: input)')
    generate_parser.add_argument('-o', '--outputs', default='output', help='Comma-separated output names (default: output)')
    generate_parser.add_argument('-c', '--category', default='custom', 
                                 choices=['custom', 'atomic', 'infrastructure', 'utils', 'video'],
                                 help='Node category (default: custom)')
    generate_parser.add_argument('-d', '--detailed-description', help='Detailed description')
    generate_parser.add_argument('--constraints', help='Comma-separated implementation constraints')
    generate_parser.add_argument('-m', '--model', default='qwen2.5-coder:7b', 
                                 help='Ollama model to use (default: qwen2.5-coder:7b)')
    generate_parser.add_argument('--show-code', action='store_true', help='Display generated code')
    generate_parser.set_defaults(func=cmd_generate)
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze workflow and generate missing nodes')
    analyze_parser.add_argument('workflow', help='Path to workflow JSON file')
    analyze_parser.add_argument('-m', '--model', default='qwen2.5-coder:7b',
                                help='Ollama model to use')
    analyze_parser.add_argument('--test', action='store_true',
                               help='Test workflow after generation')
    analyze_parser.set_defaults(func=cmd_analyze)
    
    # Interactive command
    interactive_parser = subparsers.add_parser('interactive', help='Interactive workflow creation')
    interactive_parser.add_argument('-m', '--model', default='qwen2.5-coder:7b',
                                    help='Ollama model to use')
    interactive_parser.add_argument('--auto-run', action='store_true',
                                   help='Automatically run created workflows')
    interactive_parser.set_defaults(func=cmd_interactive)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())


