"""
Minimal Function-Based Workflow Orchestrator

A lightweight orchestrator for function-based workflow nodes using the unified decorator.
Supports parallel execution, dependency resolution, and automatic conflict isolation.
"""

import json
import time
import os
import concurrent.futures
import importlib
import inspect
from typing import Dict, List, Any
from pathlib import Path

from utilities.logging_config import get_workflow_logger


class FunctionWorkflowEngine:
    """Minimal workflow engine for function-based nodes with unified decorator"""
    
    def __init__(self, workflow_data: Dict = None):
        self.logger = get_workflow_logger()
        self.results = {}
        self.environment_manager = None
        
        if workflow_data:
            self.workflow_config = workflow_data.get('workflow', {})
            self.nodes = workflow_data.get('nodes', [])
        else:
            self.workflow_config = {}
            self.nodes = []
        
        # Initialize environment manager
        self._initialize_environment_manager()
        
        # Discover and load required nodes
        self._discover_and_load_nodes()
    
    def _initialize_environment_manager(self):
        """Initialize environment manager for handling dependencies"""
        try:
            from workflow_environment_manager import WorkflowEnvironmentManager
            from pathlib import Path
            
            # Check if environments are configured in workflow
            env_config = self.workflow_config.get('environments', {})
            env_file = env_config.get('file', 'environments.json')
            
            if os.path.exists(env_file):
                self.logger.info(f"📦 Loading environments from: {env_file}")
                self.environment_manager = WorkflowEnvironmentManager(
                    environments_file=Path(env_file)
                )
                self.logger.info("📦 Environment manager initialized")
            else:
                self.logger.debug(f"No environment file found: {env_file}")
                
        except Exception as e:
            self.logger.warning(f"Environment manager not available: {e}")
    
    def _discover_and_load_nodes(self):
        """Discover and dynamically load only the nodes needed for this workflow"""
        if not self.nodes:
            return
        
        # Extract required function names from workflow
        required_functions = set()
        for node in self.nodes:
            function_name = node.get('function', '')
            if function_name:
                required_functions.add(function_name)
        
        self.logger.info(f"🔍 Discovering nodes for {len(required_functions)} functions...")
        
        # Discover available nodes in workflow_nodes/
        nodes_dir = Path("workflow_nodes")
        if not nodes_dir.exists():
            self.logger.warning(f"Node directory not found: {nodes_dir}")
            return
        
        discovered_nodes = {}
        loaded_count = 0
        
        # Scan for Python files (excluding __init__.py)
        for py_file in nodes_dir.glob("*.py"):
            if py_file.name == '__init__.py':
                continue
                
            try:
                # Convert file path to module name
                module_name = f"workflow_nodes.{py_file.stem}"
                
                # Import the module
                module = importlib.import_module(module_name)
                
                # Find functions with @workflow_node decorator
                for name, obj in inspect.getmembers(module):
                    if (inspect.isfunction(obj) and 
                        hasattr(obj, '_workflow_node_info')):
                        
                        # Check if this function is needed for the workflow
                        full_function_name = f"{module_name}.{name}"
                        if full_function_name in required_functions:
                            discovered_nodes[full_function_name] = {
                                'function': obj,
                                'module': module_name,
                                'file': py_file.name,
                                'node_info': obj._workflow_node_info
                            }
                            loaded_count += 1
                            self.logger.debug(f"✅ Loaded: {name} from {py_file.name}")
                        
            except Exception as e:
                self.logger.warning(f"Failed to load {py_file.name}: {e}")
        
        self.logger.info(f"📦 Loaded {loaded_count}/{len(required_functions)} required nodes")
        
        # Log any missing nodes
        missing_nodes = required_functions - set(discovered_nodes.keys())
        if missing_nodes:
            self.logger.warning(f"❌ Missing nodes: {list(missing_nodes)}")
        
        self.discovered_nodes = discovered_nodes
    
    def _get_ready_nodes(self, dependency_graph: Dict, completed: set) -> List[str]:
        """Find nodes ready to execute (all dependencies completed)"""
        ready = []
        for node_id, dependencies in dependency_graph.items():
            if node_id not in completed:
                if all(dep in completed for dep in dependencies):
                    ready.append(node_id)
        return ready
    
    def _prepare_inputs(self, node: Dict) -> Dict:
        """Prepare node inputs by merging static inputs with dependency results"""
        inputs = node.get('inputs', {}).copy()
        
        # Auto-inject dependency results
        for dep_id in node.get('depends_on', []):
            if dep_id in self.results:
                dep_result = self.results[dep_id]
                # Merge dependency result into inputs
                if isinstance(dep_result, dict):
                    for key, value in dep_result.items():
                        if key not in inputs and not key.startswith('_'):
                            inputs[key] = value
        
        # Resolve $ references
        for key, value in list(inputs.items()):
            if isinstance(value, str) and value.startswith('$'):
                ref = value[1:]
                if '.' in ref:
                    node_id, output_key = ref.split('.', 1)
                    if node_id in self.results:
                        inputs[key] = self.results[node_id].get(output_key)
                else:
                    if ref in self.results:
                        inputs[key] = self.results[ref]
        
        return inputs
    
    def _execute_function_node(self, node: Dict) -> Dict:
        """Execute a function-based node, using environment manager if needed"""
        function_name = node['function']
        inputs = self._prepare_inputs(node)
        
        # Try to use pre-loaded function first
        if hasattr(self, 'discovered_nodes') and function_name in self.discovered_nodes:
            try:
                func = self.discovered_nodes[function_name]['function']
                result = func(**inputs)
                return result
            except Exception as e:
                self.logger.warning(f"Pre-loaded function failed, trying dynamic import: {e}")
        
        # Check if this node needs an isolated environment
        if self.environment_manager:
            # Create a fake node_type from function name for environment lookup
            node_type = function_name.split('.')[-1].replace('_node', '_node')
            
            env_info = self.environment_manager.get_environment_for_node(
                node_type=node_type,
                node_config=node,
                workflow_config=self.workflow_config
            )
            
            # If environment is isolated, execute in that environment
            if env_info and env_info.get('is_isolated'):
                self.logger.info(f"🔧 Executing {node['id']} in {env_info['env_name']}")
                
                try:
                    result = self._execute_in_environment(
                        function_name, inputs, env_info, node['id']
                    )
                    return result
                except Exception as e:
                    self.logger.warning(f"Isolated execution failed: {e}")
                    # Fall through to direct execution
        
        # Direct execution (no isolation needed or available)
        module_name, func_name = function_name.rsplit('.', 1)
        
        try:
            # Dynamic import
            import importlib
            module = importlib.import_module(module_name)
            func = getattr(module, func_name)
            
            # Call function with prepared inputs
            result = func(inputs)
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to execute {function_name}: {e}")
            return {'error': str(e), 'status': 'failed'}
    
    def _execute_in_environment(self, function_name: str, inputs: Dict,
                               env_info: Dict, node_id: str) -> Dict:
        """Execute function in isolated environment using subprocess"""
        import subprocess
        import tempfile
        import json
        
        module_name, func_name = function_name.rsplit('.', 1)
        python_exe = env_info['python_executable']
        
        # Create simple execution script that imports and calls the function
        script_content = f'''
import sys
import os
import json

# Add current directory to path  
sys.path.insert(0, r"{os.getcwd()}")

try:
    from {module_name} import {func_name}
    
    # Read inputs
    with open("inputs.json", "r") as f:
        inputs = json.load(f)
    
    # Execute function
    result = {func_name}(inputs)
    
    # Write result
    with open("result.json", "w") as f:
        json.dump(result, f)
        
except Exception as e:
    import traceback
    with open("error.txt", "w") as f:
        f.write(str(e) + "\\n")
        f.write(traceback.format_exc())
    sys.exit(1)
'''
        
        try:
            # Create temporary directory for communication
            with tempfile.TemporaryDirectory() as temp_dir:
                script_path = os.path.join(temp_dir, "execute.py")
                inputs_path = os.path.join(temp_dir, "inputs.json")
                result_path = os.path.join(temp_dir, "result.json")
                error_path = os.path.join(temp_dir, "error.txt")
                
                # Write script and inputs
                with open(script_path, 'w') as f:
                    f.write(script_content)
                
                with open(inputs_path, 'w') as f:
                    json.dump(inputs, f)
                
                # Execute in isolated environment
                result = subprocess.run(
                    [str(python_exe), script_path],
                    cwd=temp_dir,
                    capture_output=True, 
                    text=True, 
                    timeout=300,
                    encoding='utf-8',
                    errors='replace'  # Replace invalid characters instead of failing
                )
                
                if result.returncode == 0 and os.path.exists(result_path):
                    # Read result
                    with open(result_path, 'r') as f:
                        return json.load(f)
                else:
                    # Read error
                    error_msg = "Unknown error"
                    if os.path.exists(error_path):
                        with open(error_path, 'r') as f:
                            error_msg = f.read()
                    elif result.stderr:
                        error_msg = result.stderr
                    
                    raise RuntimeError(f"Process failed: {error_msg}")
                    
        except Exception as e:
            self.logger.error(f"Environment execution failed: {e}")
            raise
                
        return {'error': 'Execution failed', 'status': 'failed'}
    
    def execute(self) -> Dict[str, Any]:
        """Execute workflow with parallel processing"""
        if not self.nodes:
            return {}
        
        workflow_name = self.workflow_config.get('name', 'Function Workflow')
        self.logger.info(f"🚀 Starting {workflow_name} ({len(self.nodes)} nodes)")
        
        start_time = time.time()
        
        # Build dependency graph
        dependency_graph = {
            node['id']: node.get('depends_on', []) 
            for node in self.nodes
        }
        nodes_by_id = {node['id']: node for node in self.nodes}
        completed = set()
        
        # Execute in waves
        max_workers = self.workflow_config.get('settings', {}).get('max_parallel_nodes', 4)
        
        while len(completed) < len(self.nodes):
            ready_nodes = self._get_ready_nodes(dependency_graph, completed)
            
            if not ready_nodes:
                remaining = set(nodes_by_id.keys()) - completed
                self.logger.error(f"Workflow blocked. Remaining: {remaining}")
                break
            
            self.logger.info(f"⚡ Executing {len(ready_nodes)} nodes...")
            
            # Execute ready nodes in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._execute_function_node, nodes_by_id[node_id]): node_id
                    for node_id in ready_nodes
                }
                
                for future in concurrent.futures.as_completed(futures):
                    node_id = futures[future]
                    try:
                        result = future.result()
                        self.results[node_id] = result
                        completed.add(node_id)
                        
                        status = "✅" if not result.get('error') else "❌"
                        self.logger.info(f"{status} {node_id}: completed")
                        
                    except Exception as e:
                        self.logger.error(f"❌ {node_id}: {e}")
                        completed.add(node_id)
                        self.results[node_id] = {'error': str(e), 'status': 'failed'}
        
        total_time = time.time() - start_time
        self.logger.info(f"✅ Workflow completed in {total_time:.2f}s")
        
        return self.results


def run_function_workflow(workflow_file: str) -> Dict:
    """Load and execute a function-based workflow"""
    with open(workflow_file, 'r') as f:
        workflow_data = json.load(f)
    
    engine = FunctionWorkflowEngine(workflow_data)
    return engine.execute()


if __name__ == "__main__":
    import sys
    import logging
    
    if len(sys.argv) < 2:
        print("Usage: python function_workflow_engine.py <workflow.json>")
        sys.exit(1)
    
    results = run_function_workflow(sys.argv[1])
    
    # Log workflow completion summary
    logger = logging.getLogger('workflow.engine')
    successful_nodes = sum(1 for result in results.values() if result.get('status', 'success') == 'success')
    failed_nodes = len(results) - successful_nodes
    
    if failed_nodes == 0:
        logger.info(f"🎯 Workflow completed successfully: {successful_nodes}/{len(results)} nodes executed")
    else:
        logger.warning(f"⚠️ Workflow completed with issues: {successful_nodes}/{len(results)} nodes successful, {failed_nodes} failed")
    
    # Log individual node timings in a compact format
    node_timings = []
    for node_id, result in results.items():
        if 'execution_time' in result:
            timing_ms = result['execution_time'] * 1000
            node_timings.append(f"{node_id}({timing_ms:.0f}ms)")
    
    if node_timings:
        logger.info(f"📈 Node timings: {', '.join(node_timings)}")