"""
Function-Based Workflow Orchestrator

A lightweight orchestrator for function-based workflow nodes using a workflow decorator.
Supports parallel execution, dependency resolution, and automatic conflict isolation.
"""

import json
import time
import os
import concurrent.futures
import importlib
from typing import Dict, List, Any
from pathlib import Path
from utilities.logging_config import get_workflow_logger


class FunctionWorkflowEngine:
    """Minimal workflow engine for function-based nodes with unified decorator"""
    
    def __init__(self, workflow_data: Dict = None):
        self.logger = get_workflow_logger()
        self.results = {}
        self.environment_manager = None
        self.discovered_nodes = {}
        
        if workflow_data:
            self.workflow_config = workflow_data.get('workflow', {})
            self.nodes = workflow_data.get('nodes', [])
        else:
            self.workflow_config = {}
            self.nodes = []
        
        # Ensure core dependencies are installed before initializing
        self._ensure_core_dependencies()
        
        self._initialize_environment_manager()
        self._discover_and_load_nodes()
    
    def _ensure_core_dependencies(self):
        """Ensure core workflow dependencies are installed"""
        core_deps = ["numpy", "opencv-python", "requests", "onnx"]
        
        try:
            import subprocess
            
            # Check which core packages are missing
            missing = []
            for dep in core_deps:
                pkg_name = dep.replace('-', '_')  # opencv-python -> opencv_python -> cv2
                if pkg_name == 'opencv_python':
                    pkg_name = 'cv2'
                try:
                    __import__(pkg_name)
                except ImportError:
                    missing.append(dep)
            
            if missing:
                self.logger.info(f"Installing {len(missing)} core dependencies: {', '.join(missing)}")
                subprocess.run(
                    ["uv", "pip", "install"] + missing,
                    check=True,
                    capture_output=True
                )
                self.logger.info("Core dependencies installed successfully")
        except Exception as e:
            self.logger.warning(f"Could not auto-install core dependencies: {e}")
            self.logger.warning("Please run: uv pip install numpy opencv-python requests onnx")
    
    def _initialize_environment_manager(self):
        """Initialize environment manager for handling dependencies"""
        try:
            from workflow_environment_manager import WorkflowEnvironmentManager
            
            env_file = self.workflow_config.get('environments', {}).get('file', 'environments.json')
            
            if os.path.exists(env_file):
                self.logger.info(f"Loading environments from: {env_file}")
                self.environment_manager = WorkflowEnvironmentManager(environments_file=Path(env_file))
            else:
                self.logger.debug("No environments.json - using auto-generated environments")
                self.environment_manager = WorkflowEnvironmentManager()
                
            self.logger.info("Environment manager initialized")
                
        except Exception as e:
            self.logger.warning(f"Environment manager not available: {e}")
    
    def _discover_and_load_nodes(self):
        """Discover and dynamically load only the nodes needed for this workflow"""
        if not self.nodes:
            return
        
        required_functions = {node.get('function', '') for node in self.nodes if node.get('function')}
        self.logger.info(f"Discovering nodes for {len(required_functions)} functions...")
        
        nodes_dir = Path("workflow_nodes")
        if not nodes_dir.exists():
            self.logger.warning(f"Node directory not found: {nodes_dir}")
            return
        
        import inspect
        loaded_count = 0
        
        for py_file in nodes_dir.glob("*.py"):
            if py_file.name == '__init__.py':
                continue
                
            try:
                module_name = f"workflow_nodes.{py_file.stem}"
                module = importlib.import_module(module_name)
                
                for name, obj in inspect.getmembers(module):
                    # Check for workflow_node decorator by looking for node_id attribute
                    if inspect.isfunction(obj) and hasattr(obj, 'node_id'):
                        full_function_name = f"{module_name}.{name}"
                        if full_function_name in required_functions:
                            self.discovered_nodes[full_function_name] = {
                                'function': obj,
                                'node_id': obj.node_id,
                                'dependencies': getattr(obj, 'dependencies', []),
                                'environment': getattr(obj, 'environment', None)
                            }
                            loaded_count += 1
                            
            except Exception as e:
                self.logger.warning(f"Failed to load {py_file.name}: {e}")
        
        self.logger.info(f"Loaded {loaded_count}/{len(required_functions)} required nodes")
    
    def _get_ready_nodes(self, dependency_graph: Dict, completed: set) -> List[str]:
        """Get nodes that are ready to execute (all dependencies completed)"""
        return [
            node_id for node_id, deps in dependency_graph.items()
            if node_id not in completed and all(dep in completed for dep in deps)
        ]
    
    def _prepare_inputs(self, node: Dict) -> Dict:
        """Prepare node inputs by merging static inputs with dependency results"""
        inputs = node.get('inputs', {}).copy()
        
        # ONLY auto-inject model_info (common pattern for model loaders → inference nodes)
        AUTO_INJECT_KEYS = {'model_info'}
        
        for dep_id in node.get('depends_on', []):
            if dep_id in self.results and isinstance(self.results[dep_id], dict):
                for key, value in self.results[dep_id].items():
                    if key in AUTO_INJECT_KEYS and key not in inputs:
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
    
    def _ensure_main_env_dependencies(self, dependencies: List[str]):
        """Install missing dependencies in the main environment using uv"""
        if not dependencies:
            return
            
        try:
            import subprocess
            
            # Check which packages are missing
            missing = []
            for dep in dependencies:
                pkg_name = dep.split('>=')[0].split('==')[0].split('<')[0].split('>')[0].strip()
                try:
                    __import__(pkg_name.replace('-', '_'))
                except ImportError:
                    missing.append(dep)
            
            if missing:
                self.logger.info(f"Installing {len(missing)} missing dependencies: {', '.join(missing)}")
                subprocess.run(
                    ["uv", "pip", "install"] + missing,
                    check=True,
                    capture_output=True
                )
                self.logger.info("Dependencies installed successfully")
        except Exception as e:
            self.logger.warning(f"Failed to auto-install dependencies: {e}")
    
    def _execute_function_node(self, node: Dict) -> Dict:
        """Execute a function-based node, using environment manager if needed"""
        function_name = node['function']
        inputs = self._prepare_inputs(node)
        
        # Debug: Log the prepared inputs
        self.logger.debug(f"Prepared inputs for {node['id']}: {inputs}")
        
        # Get function metadata if available
        node_metadata = None
        if function_name in self.discovered_nodes:
            func = self.discovered_nodes[function_name]['function']
            # Extract metadata from decorated function
            node_metadata = {
                'dependencies': getattr(func, 'dependencies', []),
                'environment': getattr(func, 'environment', None),
                'isolation_mode': getattr(func, 'isolation_mode', 'auto')
            }
            
            # Auto-install missing dependencies for in-process execution
            if node_metadata['dependencies'] and node_metadata['isolation_mode'] != 'subprocess':
                self._ensure_main_env_dependencies(node_metadata['dependencies'])
        
        # Check if isolation needed
        if self.environment_manager and node_metadata:
            node_type = function_name.split('.')[-1]
            env_info = self.environment_manager.get_environment_for_node(
                node_type, 
                node, 
                workflow_config=self.workflow_config,
                node_metadata=node_metadata
            )
            
            if env_info and env_info.get('is_isolated'):
                self.logger.info(f"Executing {node['id']} in {env_info['env_name']}")
                
                try:
                    return self._execute_in_environment(function_name, inputs, env_info, node['id'])
                except Exception as e:
                    self.logger.warning(f"Isolated execution failed: {e}")
        
        # Try pre-loaded function first
        if function_name in self.discovered_nodes:
            try:
                func = self.discovered_nodes[function_name]['function']
                return func(**inputs)
            except Exception as e:
                self.logger.warning(f"Pre-loaded function failed, trying dynamic import: {e}")
        
        # Direct execution fallback
        module_name, func_name = function_name.rsplit('.', 1)
        
        try:
            module = importlib.import_module(module_name)
            func = getattr(module, func_name)
            return func(**inputs)
        except Exception as e:
            self.logger.error(f"Failed to execute {function_name}: {e}")
            return {'error': str(e), 'status': 'failed'}
    
    def _generate_subprocess_script(self, module_name: str, func_name: str, cwd: str) -> str:
        """Generate the subprocess execution script"""
        return f'''
import sys
import os
import json

sys.path.insert(0, r"{cwd}")

# Monkey-patch SharedMemory to suppress BufferError in __del__
from multiprocessing.shared_memory import SharedMemory
_original_close = SharedMemory.close

def _close_no_error(self):
    try:
        _original_close(self)
    except BufferError:
        pass  # Suppress harmless BufferError during cleanup

SharedMemory.close = _close_no_error

try:
    from utilities.shared_memory_utils import (
        dict_from_shared_memory_with_header,
        attach_shared_memory_with_header,
        set_flag,
        FLAG_READY
    )
    import pickle
    from {module_name} import {func_name}
    
    with open("metadata.json", "r") as f:
        metadata = json.load(f)
    
    # Read inputs from shared memory
    shm_in, inputs = dict_from_shared_memory_with_header(
        metadata["input"], wait_for_ready=True, timeout=30.0
    )
    
    # Execute function
    result = {func_name}(**inputs)
    
    # Attach to output shared memory and write result
    shm_out, output_buf = attach_shared_memory_with_header(metadata["output"]["shm_name"])
    
    result_bytes = pickle.dumps(result)
    result_size = len(result_bytes)
    
    if result_size > len(output_buf):
        raise ValueError(f"Result too large: {{result_size}} bytes > {{len(output_buf)}} bytes")
    
    # Copy result to shared memory
    output_buf[:result_size] = result_bytes
    
    # Get shm_out.buf to set flag, then release it immediately
    shm_out_buf = shm_out.buf
    set_flag(shm_out_buf, FLAG_READY)
    shm_out_buf.release()
    del shm_out_buf
    
    with open("result_size.txt", "w") as f:
        f.write(str(result_size))
    
    # Write success marker first
    with open("success.txt", "w") as f:
        f.write("OK")
    
    # Critical: Release all memoryview references before subprocess exits
    # This prevents BufferError during garbage collection
    output_buf.release()
    del output_buf
    del result
    del result_bytes
    del inputs
    
    # Close and cleanup shared memory now that all memoryviews are released
    shm_in.close()
    shm_out.close()
        
except Exception as e:
    import traceback
    with open("error.txt", "w") as f:
        f.write(str(e) + "\\n")
        f.write(traceback.format_exc())
    sys.exit(1)
'''
    
    def _execute_in_environment(self, function_name: str, inputs: Dict, 
                                env_info: Dict, node_id: str) -> Dict:
        """Execute function in isolated environment using subprocess with shared memory"""
        import subprocess
        import tempfile
        import pickle
        
        from utilities.shared_memory_utils import (
            dict_to_shared_memory_with_header,
            cleanup_shared_memory,
            create_shared_memory_with_header,
            wait_for_flag,
            FLAG_READY
        )
        
        module_name, func_name = function_name.rsplit('.', 1)
        python_exe = env_info['python_executable']
        
        # Create unique shared memory names
        timestamp = int(time.time() * 1000000)
        shm_input_name = f"wf_input_{node_id}_{timestamp}"
        shm_output_name = f"wf_output_{node_id}_{timestamp}"
        
        # Parent creates input shared memory with data
        shm_inputs, metadata_inputs = dict_to_shared_memory_with_header(inputs, shm_input_name)
        self.logger.debug(f"[{node_id}] Created input shared memory '{shm_input_name}'")
        
        # Parent pre-creates output shared memory (10MB buffer)
        shm_outputs, _ = create_shared_memory_with_header(shm_output_name, 10 * 1024 * 1024)
        self.logger.debug(f"[{node_id}] Pre-created output shared memory '{shm_output_name}'")
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Write subprocess script and metadata
                script_path = os.path.join(temp_dir, "execute.py")
                with open(script_path, 'w') as f:
                    f.write(self._generate_subprocess_script(module_name, func_name, os.getcwd()))
                
                metadata_path = os.path.join(temp_dir, "metadata.json")
                with open(metadata_path, 'w') as f:
                    json.dump({
                        "input": metadata_inputs,
                        "output": {"shm_name": shm_output_name}
                    }, f)
                
                # Execute subprocess
                self.logger.debug(f"[{node_id}] Launching subprocess...")
                result = subprocess.run(
                    [str(python_exe), script_path],
                    cwd=temp_dir,
                    capture_output=True,
                    text=True,
                    timeout=300,
                    encoding='utf-8',
                    errors='replace'
                )
                
                # Log subprocess output
                if result.stdout:
                    for line in result.stdout.strip().split('\n'):
                        if line.strip():
                            self.logger.info(f"[{node_id}] {line}")
                
                # Log stderr but filter out harmless BufferError from shared memory cleanup
                if result.stderr:
                    stderr_lines = result.stderr.strip().split('\n')
                    # Filter out BufferError traceback (harmless GC issue on subprocess exit)
                    filtered_stderr = []
                    skip_next = 0
                    for i, line in enumerate(stderr_lines):
                        if skip_next > 0:
                            skip_next -= 1
                            continue
                        if 'BufferError: cannot close exported pointers exist' in line:
                            # Skip this line and previous 5 lines (traceback)
                            filtered_stderr = filtered_stderr[:-min(5, len(filtered_stderr))]
                            skip_next = 0  # Already past the error
                        else:
                            filtered_stderr.append(line)
                    
                    # Log remaining errors
                    for line in filtered_stderr:
                        if line.strip():
                            self.logger.warning(f"[{node_id}] stderr: {line}")
                
                # Check for success
                success_path = os.path.join(temp_dir, "success.txt")
                if result.returncode == 0 and os.path.exists(success_path):
                    # Wait for subprocess to set FLAG_READY on output
                    self.logger.debug(f"[{node_id}] Waiting for FLAG_READY on output...")
                    if not wait_for_flag(shm_outputs.buf, FLAG_READY, timeout=30.0):
                        raise TimeoutError("Subprocess did not complete within timeout")
                    
                    # Read result size and data
                    result_size_path = os.path.join(temp_dir, "result_size.txt")
                    with open(result_size_path, 'r') as f:
                        result_size = int(f.read().strip())
                    
                    # Copy data WITHOUT creating memoryview references
                    # Use tobytes() which creates independent copy
                    result_bytes = shm_outputs.buf[8:8+result_size].tobytes()
                    
                    # Unpickle data
                    output_data = pickle.loads(result_bytes)
                    del result_bytes
                    
                    # Deep copy numpy arrays if present to ensure no shared memory references
                    # This is critical - unpickled arrays may still reference the shared buffer
                    if isinstance(output_data, dict):
                        import numpy as np
                        output_data_copy = {}
                        for key, value in output_data.items():
                            if isinstance(value, np.ndarray):
                                # Force independent copy - breaks link to shared memory
                                output_data_copy[key] = np.array(value, copy=True, order='C')
                            else:
                                output_data_copy[key] = value
                        # Delete original to release shared memory references
                        del output_data
                    else:
                        output_data_copy = output_data
                    
                    # Cleanup shared memory - catch BufferError (harmless, subprocess still has references)
                    try:
                        shm_inputs.close()
                        shm_outputs.close()
                        cleanup_shared_memory(shm_input_name)
                        cleanup_shared_memory(shm_output_name)
                    except BufferError as e:
                        # Harmless - subprocess memoryviews still exist during GC
                        self.logger.debug(f"[{node_id}] BufferError during cleanup (expected): {e}")
                        # Try to at least unlink the memory
                        try:
                            cleanup_shared_memory(shm_input_name)
                            cleanup_shared_memory(shm_output_name)
                        except Exception:
                            pass
                    
                    self.logger.debug(f"[{node_id}] Successfully read result from shared memory")
                    return output_data_copy
                else:
                    # Handle error
                    error_path = os.path.join(temp_dir, "error.txt")
                    error_msg = "Unknown error"
                    if os.path.exists(error_path):
                        with open(error_path, 'r') as f:
                            error_msg = f.read()
                    elif result.stderr:
                        error_msg = result.stderr
                    
                    shm_inputs.close()
                    cleanup_shared_memory(shm_input_name)
                    cleanup_shared_memory(shm_output_name)
                    
                    raise RuntimeError(f"Process failed: {error_msg}")
                    
        except Exception as e:
            self.logger.error(f"Environment execution failed: {e}")
            try:
                cleanup_shared_memory(shm_input_name)
                cleanup_shared_memory(shm_output_name)
            except Exception:
                pass
            raise
                
        return {'error': 'Execution failed', 'status': 'failed'}
    
    def execute(self) -> Dict[str, Any]:
        """Execute workflow with parallel processing"""
        if not self.nodes:
            return {}
        
        workflow_name = self.workflow_config.get('name', 'Function Workflow')
        self.logger.info(f"Starting {workflow_name} ({len(self.nodes)} nodes)")
        
        # Build dependency graph
        dependency_graph = {node['id']: node.get('depends_on', []) for node in self.nodes}
        max_parallel = self.workflow_config.get('settings', {}).get('max_parallel_nodes', 4)
        
        completed = set()
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
            while len(completed) < len(self.nodes):
                ready_nodes = self._get_ready_nodes(dependency_graph, completed)
                
                if not ready_nodes:
                    if len(completed) < len(self.nodes):
                        self.logger.error("Workflow deadlock: circular dependencies detected")
                    break
                
                self.logger.info(f"Executing {len(ready_nodes)} nodes...")
                
                futures = {}
                for node_id in ready_nodes:
                    node = next(n for n in self.nodes if n['id'] == node_id)
                    futures[executor.submit(self._execute_function_node, node)] = node_id
                
                for future in concurrent.futures.as_completed(futures):
                    node_id = futures[future]
                    
                    try:
                        result = future.result()
                        self.results[node_id] = result
                        completed.add(node_id)
                        
                        status = "COMPLETED" if not result.get('error') else "FAILED"
                        self.logger.info(f"{status}: {node_id}")
                        
                    except Exception as e:
                        self.logger.error(f"FAILED: {node_id}: {e}")
                        completed.add(node_id)
                        self.results[node_id] = {'error': str(e), 'status': 'failed'}
        
        total_time = time.time() - start_time
        self.logger.info(f"Workflow completed in {total_time:.2f}s")
        
        return self.results


def run_function_workflow(workflow_file: str) -> Dict:
    """Load and execute a function-based workflow"""
    with open(workflow_file, 'r') as f:
        workflow_data = json.load(f)
    
    # Debug: Log the loaded workflow data
    import logging
    logger = logging.getLogger('workflow.engine')
    logger.debug(f"Loaded workflow from {workflow_file}")
    for node in workflow_data.get('nodes', []):
        logger.debug(f"  Node {node['id']}: inputs = {node.get('inputs', {})}")
    
    engine = FunctionWorkflowEngine(workflow_data)
    return engine.execute()


if __name__ == "__main__":
    import sys
    import logging
    
    if len(sys.argv) < 2:
        print("Usage: python function_workflow_engine.py <workflow.json>")
        sys.exit(1)
    
    results = run_function_workflow(sys.argv[1])
    
    # Log completion summary
    logger = logging.getLogger('workflow.engine')
    successful_nodes = sum(1 for result in results.values() 
                          if isinstance(result, dict) and result.get('status') != 'failed' and 'error' not in result)
    failed_nodes = len(results) - successful_nodes
    
    if failed_nodes > 0:
        for node_id, result in results.items():
            if isinstance(result, dict) and (result.get('status') == 'failed' or 'error' in result):
                logger.info(f"Node '{node_id}' marked as failed - status: {result.get('status')}, error: {result.get('error')}")
    
    if failed_nodes == 0:
        logger.info(f"Workflow completed successfully: {successful_nodes}/{len(results)} nodes executed")
    else:
        logger.warning(f"Workflow completed with issues: {successful_nodes}/{len(results)} nodes successful, {failed_nodes} failed")
