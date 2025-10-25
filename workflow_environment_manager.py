#!/usr/bin/env python3
"""
Workflow Environment Manager

Manages multiple isolated virtual environments for workflows and nodes.
Separates engine infrastructure from workflow-specific dependencies.
"""

import os
import sys
import json
import subprocess
import multiprocessing
import multiprocessing.shared_memory as shared_memory
import pickle
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import logging


class WorkflowEnvironmentManager:
    """
    Manages virtual environments for workflows and nodes.
    
    Architecture:
    - Engine environment: Core workflow engine and common dependencies
    - Workflow environments: Isolated per-workflow dependencies
    - Node environments: Isolated per-node for conflicting dependencies
    """
    
    def __init__(self, 
                 engine_env: Optional[Path] = None,
                 environments_dir: Path = Path("workflow-envs"),
                 environments_file: Optional[Path] = None):
        """
        Initialize environment manager
        
        Architecture:
        - workflow_engine: Platform environment for base nodes (default)
        - User environments: Custom environments for user-defined nodes
        
        Args:
            engine_env: Path to platform's virtual environment (default: current)
            environments_dir: Directory to store user environments
            environments_file: Path to external environments definition file
        """
        self.logger = logging.getLogger('workflow.env_manager')
        
        # Platform environment (workflow_engine) - used by base nodes
        self.engine_env = engine_env or Path(sys.prefix)
        self.engine_python = Path(sys.executable)
        
        # Directory for user-defined workflow environments
        self.environments_dir = Path(environments_dir)
        self.environments_dir.mkdir(exist_ok=True)
        
        # Cache of environment info
        self._env_cache: Dict[str, Dict[str, Any]] = {}
        
        # Load external environments file if provided
        self._external_envs: Dict[str, Any] = {}
        self._external_mappings: Dict[str, str] = {}
        if environments_file:
            self._load_environments_file(environments_file)
        
        self.logger.info("📦 Environment Manager initialized")
        self.logger.info(f"   Platform env (workflow_engine): {self.engine_env}")
        self.logger.info(f"   User environments: {self.environments_dir}")
        if self._external_envs:
            self.logger.info(
                f"   Loaded definitions: {len(self._external_envs)} environments")
    
    def _load_environments_file(self, env_file: Path):
        """Load environment definitions from external file"""
        try:
            with open(env_file, 'r') as f:
                env_config = json.load(f)
            
            self._external_envs = env_config.get('environments', {})
            self._external_mappings = env_config.get('node_type_mappings', {})
            
            # Filter out 'description' key from mappings
            self._external_mappings = {
                k: v for k, v in self._external_mappings.items()
                if k != 'description'
            }
            
            self.logger.debug(
                f"Loaded {len(self._external_envs)} environment definitions")
            self.logger.debug(
                f"Loaded {len(self._external_mappings)} node type mappings")
            
        except Exception as e:
            self.logger.error(f"Failed to load environments file: {e}")
            raise
    
    def get_environment_for_node(self,
                                 node_type: str,
                                 node_config: Dict[str, Any],
                                 workflow_config: Optional[Dict[str, Any]] = None
                                 ) -> Dict[str, Any]:
        """
        Get environment configuration for a specific node
        
        Resolution order:
        1. Node-level 'environment' field (explicit override)
        2. Workflow 'node_mappings' (workflow-specific)
        3. External 'node_type_mappings' (shared definitions)
        4. Platform environment 'workflow_engine' (DEFAULT for all base nodes)
        
        The platform environment is used by default unless explicitly configured.
        This means base workflow nodes automatically use the platform environment,
        and only user-defined nodes with special dependencies need custom environments.
        
        Returns:
            Dict with:
                - python_executable: Path to Python
                - env_path: Path to virtual environment
                - env_name: Name of environment
                - is_isolated: Whether this is an isolated environment
        """
        # Store workflow config for environment resolution
        self._current_workflow_config = workflow_config
        
        # Check node-specific environment (highest priority)
        node_env_spec = node_config.get('environment')
        if node_env_spec:
            return self._resolve_environment(
                node_env_spec, f"node:{node_type}", workflow_config)
        
        # Check workflow-level node environment mapping
        if workflow_config:
            env_config = workflow_config.get('environments', {})
            node_mappings = env_config.get('node_mappings', {})
            
            if node_type in node_mappings:
                env_name = node_mappings[node_type]
                return self._resolve_environment(
                    env_name, f"workflow_mapping:{node_type}", workflow_config)
        
        # Check external node type mappings
        if node_type in self._external_mappings:
            env_name = self._external_mappings[node_type]
            self.logger.debug(
                f"Using external mapping: {node_type} -> {env_name}")
            return self._resolve_environment(
                env_name, f"external_mapping:{node_type}", workflow_config)
        
        # Default to platform environment (workflow_engine)
        # This is used by all base nodes automatically
        return {
            'python_executable': self.engine_python,
            'env_path': self.engine_env,
            'env_name': 'workflow_engine',
            'is_isolated': False
        }
    
    def _resolve_environment(self,
                             env_spec: Any,
                             context: str,
                             workflow_config: Optional[Dict[str, Any]] = None
                             ) -> Dict[str, Any]:
        """
        Resolve environment specification to actual environment
        
        Env spec can be:
        - String: environment name (looks up in workflow/external definitions)
        - Dict: {'name': 'env-name', 'requirements': [...], 'path': '...'}
        
        Resolution order:
        1. Workflow definitions
        2. External definitions (from environments.json)
        3. Treat as path/name and create if needed
        
        Automatically creates environment if it doesn't exist.
        """
        # If string, try to look up in definitions
        if isinstance(env_spec, str):
            env_name = env_spec
            
            # 1. Look up in workflow definitions first
            if workflow_config:
                env_defs = workflow_config.get('environments', {}).get(
                    'definitions', {})
                
                if env_name in env_defs:
                    # Found in workflow - use it
                    env_def = env_defs[env_name]
                    self.logger.debug(
                        f"Using workflow definition for: {env_name}")
                    return self._resolve_environment(
                        env_def, context, workflow_config)
            
            # 2. Look up in external definitions
            if env_name in self._external_envs:
                env_def = self._external_envs[env_name]
                self.logger.debug(
                    f"Using external definition for: {env_name}")
                return self._resolve_environment(
                    env_def, context, workflow_config)
            
            # 3. No definition found - treat as path/name
            env_path = Path(env_name)
            if not env_path.is_absolute():
                env_path = self.environments_dir / env_name
            
            # Check if exists, create if not
            if not env_path.exists():
                self.logger.warning(
                    f"Environment '{env_name}' not found at {env_path}")
                self.logger.info(
                    f"Creating environment with no requirements...")
                self._create_environment(env_path, [])
            
            return self._get_env_info(env_path, env_name)
        
        elif isinstance(env_spec, dict):
            # Dict specification with more details
            env_name = env_spec.get('name')
            env_path_str = env_spec.get('path')
            requirements = env_spec.get('requirements', [])
            
            if env_path_str:
                env_path = Path(env_path_str)
            elif env_name:
                env_path = self.environments_dir / env_name
            else:
                raise ValueError(
                    f"Environment spec must have 'name' or 'path': {env_spec}")
            
            # Check if environment exists
            if not env_path.exists():
                self.logger.warning(
                    f"Environment not found: {env_path} (context: {context})")
                self.logger.info(f"🔧 Creating new virtual environment: {env_path.name}")
                self.logger.info(f"   Requirements: {requirements}")
                self._create_environment(env_path, requirements)
            else:
                # Environment exists - verify requirements are installed
                self._ensure_requirements(env_path, requirements)
            
            return self._get_env_info(env_path, env_name or env_path.name)
        
        else:
            raise ValueError(f"Invalid environment spec type: {type(env_spec)}")
    
    def _get_env_info(self, env_path: Path, env_name: str) -> Dict[str, Any]:
        """Get information about an environment"""
        # Check cache
        cache_key = str(env_path)
        if cache_key in self._env_cache:
            return self._env_cache[cache_key]
        
        # Find Python executable
        if os.name == 'nt':  # Windows
            python_exe = env_path / "Scripts" / "python.exe"
        else:  # Unix/Linux/Mac
            python_exe = env_path / "bin" / "python"
        
        if not python_exe.exists():
            # Environment doesn't exist - create it
            self.logger.info(
                f"Environment '{env_name}' not fully initialized")
            self.logger.info(f"Creating basic environment at: {env_path}")
            self._create_environment(env_path, [])
            
            # Check again after creation
            if not python_exe.exists():
                raise FileNotFoundError(
                    f"Failed to create Python executable: {python_exe}")
        
        env_info = {
            'python_executable': python_exe,
            'env_path': env_path,
            'env_name': env_name,
            'is_isolated': True
        }
        
        # Cache it
        self._env_cache[cache_key] = env_info
        return env_info
    
    def _create_environment(self, env_path: Path, requirements: List[str]):
        """
        Create a new virtual environment with specified requirements using uv
        
        Args:
            env_path: Path where environment should be created
            requirements: List of package requirements (e.g., ['torch>=2.0', 'numpy'])
        """
        self.logger.info(f"📦 Creating virtual environment: {env_path.name}")
        self.logger.info(f"   Location: {env_path}")
        
        try:
            # Create venv using uv
            self.logger.info("🔧 Running: uv venv...")
            subprocess.run(
                ["uv", "venv", str(env_path)],
                check=True,
                capture_output=True
            )
            self.logger.info("✅ Virtual environment created successfully")
            
            # Install requirements using uv
            if requirements:
                self.logger.info(f"📦 Installing {len(requirements)} packages with uv...")
                self.logger.info(f"   Packages: {', '.join(requirements)}")
                
                # Use uv pip install with the environment's Python
                if os.name == 'nt':
                    python_exe = env_path / "Scripts" / "python.exe"
                else:
                    python_exe = env_path / "bin" / "python"
                
                subprocess.run(
                    ["uv", "pip", "install", "--python", str(python_exe)] + requirements,
                    check=True,
                    capture_output=True
                )
                self.logger.info("✅ All packages installed successfully")
            else:
                self.logger.info("ℹ️ No requirements to install")
            
            self.logger.info(f"🎉 Environment {env_path.name} ready for use")
            
        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to create environment: {e.stderr.decode() if e.stderr else str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _ensure_requirements(self, env_path: Path, requirements: List[str]):
        """
        Verify and install missing requirements in an existing environment using uv
        
        Args:
            env_path: Path to the virtual environment
            requirements: List of required packages
        """
        if not requirements:
            return  # No requirements to check
            
        # Get Python executable from environment
        if os.name == 'nt':
            python_exe = env_path / "Scripts" / "python.exe"
        else:
            python_exe = env_path / "bin" / "python"
            
        if not python_exe.exists():
            self.logger.warning(f"Python not found in {env_path}, skipping requirement check")
            return
            
        try:
            # Get list of installed packages using uv pip list
            result = subprocess.run(
                ["uv", "pip", "list", "--python", str(python_exe), "--format=freeze"],
                check=True,
                capture_output=True,
                text=True
            )
            
            installed_packages = {}
            for line in result.stdout.strip().split('\n'):
                if '==' in line:
                    name, version = line.split('==', 1)
                    installed_packages[name.lower()] = version
            
            # Check which requirements are missing
            missing_requirements = []
            for req in requirements:
                # Simple package name extraction (handles "package>=1.0" -> "package")
                pkg_name = req.split('>=')[0].split('==')[0].split('<')[0].split('>')[0].strip()
                if pkg_name.lower() not in installed_packages:
                    missing_requirements.append(req)
                    
            if missing_requirements:
                self.logger.info(f"📦 Installing {len(missing_requirements)} missing packages...")
                
                # Install missing requirements using uv
                subprocess.run(
                    ["uv", "pip", "install", "--python", str(python_exe)] + missing_requirements,
                    check=True,
                    capture_output=True
                )
                self.logger.info("✅ Missing packages installed successfully")
                
        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to check/install requirements: {e.stderr.decode() if e.stderr else str(e)}"
            self.logger.warning(error_msg)
    
    def execute_in_environment(self,
                               env_info: Dict[str, Any],
                               script_path: Path,
                               args: List[str],
                               timeout: int = 300) -> Dict[str, Any]:
        """
        Execute a Python script in a specific environment
        
        Args:
            env_info: Environment info from get_environment_for_node()
            script_path: Path to Python script to execute
            args: Command-line arguments
            timeout: Timeout in seconds
        
        Returns:
            Dict with execution results
        """
        python_exe = env_info['python_executable']
        env_name = env_info['env_name']
        
        cmd = [str(python_exe), str(script_path)] + args
        
        self.logger.debug(f"Executing in {env_name}: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            return {
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'success': result.returncode == 0,
                'environment': env_name
            }
            
        except subprocess.TimeoutExpired:
            return {
                'returncode': -1,
                'stdout': '',
                'stderr': f'Execution timed out after {timeout}s',
                'success': False,
                'environment': env_name
            }
        except Exception as e:
            return {
                'returncode': -1,
                'stdout': '',
                'stderr': str(e),
                'success': False,
                'environment': env_name
            }
    
    def execute_node_multiprocess(self,
                                   env_info: Dict[str, Any],
                                   node_type: str,
                                   node_class_module: str,
                                   node_class_name: str,
                                   inputs: Dict[str, Any],
                                   nodes_directory: str,
                                   timeout: int = 300) -> Dict[str, Any]:
        """
        Execute a node in isolated environment using multiprocessing
        
        Uses multiprocessing.Process with pickle for serialization.
        Better than subprocess as it can handle complex Python objects.
        
        Args:
            env_info: Environment info from get_environment_for_node()
            node_type: Type of node being executed
            node_class_module: Module name (e.g., 'directml_model_loader_node')
            node_class_name: Class name (e.g., 'DirectmlModelLoaderNode')
            inputs: Input dictionary for node execution
            nodes_directory: Path to nodes directory
            timeout: Timeout in seconds
        
        Returns:
            Dict with execution results
        """
        import multiprocessing as mp
        import tempfile
        import pickle
        
        env_name = env_info['env_name']
        python_exe = str(env_info['python_executable'])
        
        self.logger.debug(
            f"Executing {node_type} in isolated process: {env_name}"
        )
        
        # Filter out unpicklable objects from inputs
        # For isolated execution, we don't pass session objects
        filtered_inputs = {}
        for key, value in inputs.items():
            if key == 'session':
                # Don't pass session object to isolated process
                # Nodes in isolated env will need to create their own session
                continue
            try:
                # Test if value is picklable
                pickle.dumps(value)
                filtered_inputs[key] = value
            except (TypeError, pickle.PicklingError) as e:
                self.logger.debug(
                    f"Skipping unpicklable input '{key}': {type(value)}"
                )
        
        # Create temporary files for input/output
        with tempfile.NamedTemporaryFile(
            mode='wb', delete=False, suffix='.pkl'
        ) as input_file:
            input_path = input_file.name
            pickle.dump(filtered_inputs, input_file)
        
        output_path = input_path.replace('.pkl', '_output.pkl')
        
        # Create execution script
        exec_script = f'''
import sys
import os
import pickle

# Add nodes directory to path
sys.path.insert(0, r"{nodes_directory}")
sys.path.insert(0, os.path.dirname(r"{nodes_directory}"))

# Load inputs
with open(r"{input_path}", "rb") as f:
    inputs = pickle.load(f)

try:
    # Import and execute node
    module = __import__("{node_class_module}")
    node_class = getattr(module, "{node_class_name}")
    node_instance = node_class()
    
    # Execute
    result = node_instance.execute(inputs)
    
    # Save result
    with open(r"{output_path}", "wb") as f:
        pickle.dump({{"status": "success", "result": result}}, f)
        
except Exception as e:
    import traceback
    error_info = {{
        "status": "error",
        "error": str(e),
        "traceback": traceback.format_exc()
    }}
    with open(r"{output_path}", "wb") as f:
        pickle.dump(error_info, f)
'''
        
        # Write script to temporary file
        with tempfile.NamedTemporaryFile(
            mode='w', delete=False, suffix='.py', encoding='utf-8'
        ) as script_file:
            script_path = script_file.name
            script_file.write(exec_script)
        
        try:
            # Execute in isolated environment
            result = subprocess.run(
                [python_exe, script_path],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            # Load result
            if os.path.exists(output_path):
                with open(output_path, 'rb') as f:
                    output_data = pickle.load(f)
                
                if output_data['status'] == 'success':
                    return output_data['result']
                else:
                    error_msg = output_data.get('error', 'Unknown error')
                    self.logger.error(
                        f"Node execution failed: {error_msg}"
                    )
                    if 'traceback' in output_data:
                        self.logger.debug(output_data['traceback'])
                    return {
                        'status': 'failed',
                        'error': error_msg
                    }
            else:
                # No output file - check stderr
                error_msg = result.stderr if result.stderr else (
                    'No output produced'
                )
                self.logger.error(f"Execution failed: {error_msg}")
                return {
                    'status': 'failed',
                    'error': error_msg
                }
                
        except subprocess.TimeoutExpired:
            return {
                'status': 'failed',
                'error': f'Execution timed out after {timeout}s'
            }
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }
        finally:
            # Cleanup temporary files
            for temp_file in [input_path, output_path, script_path]:
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                except:
                    pass
    
    def execute_node_shared_memory(self,
                                    env_info: Dict[str, Any],
                                    node_type: str,
                                    node_class_module: str,
                                    node_class_name: str,
                                    inputs: Dict[str, Any],
                                    nodes_directory: str,
                                    data_session=None,
                                    timeout: int = 300) -> Dict[str, Any]:
        """
        Execute node in isolated environment with shared memory
        
        Uses multiprocessing.shared_memory for efficient data passing.
        Supports WorkflowDataSession via shared memory buffer.
        
        Args:
            env_info: Environment info from get_environment_for_node()
            node_type: Type of node being executed
            node_class_module: Module name
            node_class_name: Class name
            inputs: Input dictionary for node execution
            nodes_directory: Path to nodes directory
            data_session: WorkflowDataSession instance (optional)
            timeout: Timeout in seconds
        
        Returns:
            Dict with execution results
        """
        import multiprocessing.shared_memory as shm
        import pickle
        import tempfile
        
        env_name = env_info['env_name']
        python_exe = str(env_info['python_executable'])
        
        self.logger.debug(
            f"Executing {node_type} in isolated process "
            f"with shared memory: {env_name}"
        )
        
        # Prepare inputs - filter unpicklable objects but keep references
        filtered_inputs = {}
        shared_mem_info = {}
        
        for key, value in inputs.items():
            if key == 'session' and data_session is not None:
                # Pass session namespace instead of session object
                filtered_inputs['session_namespace'] = inputs.get(
                    'session_namespace', 'default'
                )
                # Signal that session should be created in subprocess
                filtered_inputs['_create_local_session'] = True
                continue
                
            try:
                # Test if value is picklable
                pickle.dumps(value)
                filtered_inputs[key] = value
            except (TypeError, pickle.PicklingError):
                self.logger.debug(
                    f"Skipping unpicklable input '{key}': {type(value)}"
                )
        
        # Create temp files for communication
        with tempfile.NamedTemporaryFile(
            mode='wb', delete=False, suffix='.pkl'
        ) as input_file:
            input_path = input_file.name
            pickle.dump(filtered_inputs, input_file)
        
        output_path = input_path.replace('.pkl', '_output.pkl')
        
        # Create execution script with shared memory support
        exec_script = f'''
import sys
import os
import pickle

# Add nodes directory to path
sys.path.insert(0, r"{nodes_directory}")
sys.path.insert(0, os.path.dirname(r"{nodes_directory}"))

# Load inputs
with open(r"{input_path}", "rb") as f:
    inputs = pickle.load(f)

# Create local session if needed
if inputs.get('_create_local_session'):
    try:
        sys.path.insert(0, r"{os.path.dirname(nodes_directory)}")
        from workflow_data_session import WorkflowDataSession
        session = WorkflowDataSession()
        inputs['session'] = session
        inputs.pop('_create_local_session', None)
    except Exception as e:
        print(f"Warning: Could not create session: {{e}}")

try:
    # Import and execute node
    module = __import__("{node_class_module}")
    node_class = getattr(module, "{node_class_name}")
    node_instance = node_class()
    
    # Execute
    result = node_instance.execute(inputs)
    
    # Save result
    with open(r"{output_path}", "wb") as f:
        pickle.dump({{"status": "success", "result": result}}, f)
        
except Exception as e:
    import traceback
    error_info = {{
        "status": "error",
        "error": str(e),
        "traceback": traceback.format_exc()
    }}
    with open(r"{output_path}", "wb") as f:
        pickle.dump(error_info, f)
'''
        
        # Write script to temp file
        with tempfile.NamedTemporaryFile(
            mode='w', delete=False, suffix='.py', encoding='utf-8'
        ) as script_file:
            script_path = script_file.name
            script_file.write(exec_script)
        
        try:
            # Execute in isolated environment
            result = subprocess.run(
                [python_exe, script_path],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            # Load result
            if os.path.exists(output_path):
                with open(output_path, 'rb') as f:
                    output_data = pickle.load(f)
                
                if output_data['status'] == 'success':
                    return output_data['result']
                else:
                    error_msg = output_data.get('error', 'Unknown error')
                    self.logger.error(
                        f"Node execution failed: {error_msg}"
                    )
                    if 'traceback' in output_data:
                        self.logger.debug(output_data['traceback'])
                    return {
                        'status': 'failed',
                        'error': error_msg
                    }
            else:
                # No output file - check stderr
                error_msg = result.stderr if result.stderr else (
                    'No output produced'
                )
                self.logger.error(f"Execution failed: {error_msg}")
                return {
                    'status': 'failed',
                    'error': error_msg
                }
                
        except subprocess.TimeoutExpired:
            return {
                'status': 'failed',
                'error': f'Execution timed out after {timeout}s'
            }
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }
        finally:
            # Cleanup temporary files
            for temp_file in [input_path, output_path, script_path]:
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                except:
                    pass
    
    def list_environments(self) -> List[Dict[str, Any]]:
        """List all workflow environments"""
        envs = []
        
        # Add engine environment
        envs.append({
            'name': 'engine',
            'path': self.engine_env,
            'python': self.engine_python,
            'type': 'engine'
        })
        
        # List workflow environments
        if self.environments_dir.exists():
            for env_dir in self.environments_dir.iterdir():
                if env_dir.is_dir():
                    python_path = (env_dir / "Scripts" / "python.exe" 
                                  if os.name == 'nt' 
                                  else env_dir / "bin" / "python")
                    
                    if python_path.exists():
                        envs.append({
                            'name': env_dir.name,
                            'path': env_dir,
                            'python': python_path,
                            'type': 'workflow'
                        })
        
        return envs
    
    def get_node_dependencies(self, node_type: str) -> Optional[List[str]]:
        """
        Get dependencies for a specific node type
        
        This can be extended to read from node metadata or config files
        """
        # Node dependency registry
        # Can be loaded from external config or node metadata
        node_deps = {
            'onnx_cuda_subprocess_node': [
                'onnxruntime-gpu',
                'numpy',
                'opencv-python',
                'Pillow'
            ],
            'cuda_inference_node': [
                'torch',
                'torchvision',
                'ultralytics'
            ],
            'npu_inference_node': [
                'openvino',
                'opencv-python'
            ],
            # Add more node types as needed
        }
        
        return node_deps.get(node_type)
    
    def execute_node_with_true_shared_memory(self,
                                             env_info: Dict[str, Any],
                                             node_type: str,
                                             node_class_module: str,
                                             node_class_name: str,
                                             inputs: Dict[str, Any],
                                             nodes_directory: str,
                                             data_session=None,
                                             timeout: int = 300) -> Dict[str, Any]:
        """
        Execute node in isolated environment using TRUE shared memory.
        
        WORKFLOW ENGINE CONTROLS EVERYTHING:
        - Main process creates input and output shared memory blocks
        - Subprocess reads from input, writes to output
        - Main process reads output and cleans up
        - No subprocess waiting or lifetime management needed
        
        Args:
            env_info: Environment info from get_environment_for_node()
            node_type: Type of node being executed
            node_class_module: Module name
            node_class_name: Class name
            inputs: Input dictionary for node execution
            nodes_directory: Path to nodes directory
            data_session: WorkflowDataSession instance (optional)
            timeout: Timeout in seconds
        
        Returns:
            Dict with execution results
        """
        import uuid
        import subprocess
        import tempfile
        import pickle
        from multiprocessing.shared_memory import SharedMemory
        from utilities.shared_memory_utils import (
            HEADER_SIZE, set_flag, get_flag, wait_for_flag,
            FLAG_EMPTY, FLAG_READY, FLAG_DONE, FLAG_ERROR
        )
        
        env_name = env_info['env_name']
        python_exe = str(env_info['python_executable'])
        
        self.logger.info(
            f"🚀 Executing {node_type} with TRUE shared memory "
            f"(workflow-controlled) in isolated env: {env_name}"
        )
        
        # Generate unique shared memory names
        shm_id = str(uuid.uuid4())[:8]
        input_shm_name = f"wf_input_{shm_id}"
        output_shm_name = f"wf_output_{shm_id}"
        
        # Prepare inputs - filter unpicklable objects
        filtered_inputs = {}
        
        for key, value in inputs.items():
            if key == 'session' and data_session is not None:
                # Pass session namespace instead of session object
                filtered_inputs['session_namespace'] = inputs.get(
                    'session_namespace', 'default'
                )
                filtered_inputs['_create_local_session'] = True
                continue
                
            try:
                # Test if value is picklable
                pickle.dumps(value)
                filtered_inputs[key] = value
            except (TypeError, pickle.PicklingError):
                self.logger.debug(
                    f"Skipping unpicklable input '{key}': {type(value)}"
                )
        
        # MAIN PROCESS CREATES BOTH SHARED MEMORY BLOCKS
        # Input: Pickle inputs and store
        input_data_bytes = pickle.dumps(filtered_inputs)
        input_size = HEADER_SIZE + len(input_data_bytes)
        input_shm = SharedMemory(name=input_shm_name, create=True, size=input_size)
        set_flag(input_shm.buf, FLAG_EMPTY)
        input_shm.buf[HEADER_SIZE:HEADER_SIZE+len(input_data_bytes)] = input_data_bytes
        set_flag(input_shm.buf, FLAG_READY)  # Signal ready
        
        self.logger.debug(
            f"Created input shared memory: {input_shm_name} ({len(input_data_bytes)} bytes)"
        )
        
        # Output: Pre-allocate large enough buffer for results (10MB should be plenty)
        output_size = HEADER_SIZE + 10 * 1024 * 1024  # 10MB
        output_shm = SharedMemory(name=output_shm_name, create=True, size=output_size)
        set_flag(output_shm.buf, FLAG_EMPTY)  # Not ready yet
        
        self.logger.debug(
            f"Created output shared memory: {output_shm_name} ({output_size} bytes pre-allocated)"
        )
        
        # Create execution script for subprocess
        # SUBPROCESS SIMPLY READS INPUT AND WRITES OUTPUT - NO CREATION/CLEANUP
        exec_script = f'''
import sys
import os
import pickle

# Add paths for imports
sys.path.insert(0, r"{nodes_directory}")
sys.path.insert(0, os.path.dirname(r"{nodes_directory}"))
sys.path.insert(0, r"{os.path.dirname(nodes_directory)}")

from multiprocessing.shared_memory import SharedMemory
from utilities.shared_memory_utils import (
    HEADER_SIZE, set_flag, get_flag, wait_for_flag,
    FLAG_READY, FLAG_DONE, FLAG_ERROR
)

try:
    # Attach to input shared memory (already created by main process)
    input_shm = SharedMemory(name="{input_shm_name}")
    
    # Wait for ready flag
    if not wait_for_flag(input_shm.buf, FLAG_READY, timeout=30.0):
        raise TimeoutError("Input shared memory not ready")
    
    # Read input data - COPY the bytes to avoid holding reference
    input_data_bytes = bytes(input_shm.buf[HEADER_SIZE:])
    
    # Close input immediately after reading to release reference
    input_shm.close()
    
    # Now unpickle from the copied bytes
    inputs = pickle.loads(input_data_bytes)
    
    # Create local session if needed
    if inputs.get('_create_local_session'):
        try:
            from workflow_data_session import WorkflowDataSession
            session = WorkflowDataSession()
            inputs['session'] = session
            inputs.pop('_create_local_session', None)
        except Exception as e:
            print(f"⚠️  Could not create session: {{e}}", flush=True)
    
    # Import and execute node
    module = __import__("{node_class_module}")
    node_class = getattr(module, "{node_class_name}")
    node_instance = node_class()
    
    # Execute
    result = node_instance.execute(inputs)
    
    # Attach to output shared memory (already created by main process)
    output_shm = SharedMemory(name="{output_shm_name}")
    
    # Prepare result
    output_data = {{"status": "success", "result": result}}
    output_bytes = pickle.dumps(output_data)
    
    # Write to output shared memory - copy directly without holding reference
    output_shm.buf[HEADER_SIZE:HEADER_SIZE+len(output_bytes)] = output_bytes
    
    # Signal completion
    set_flag(output_shm.buf, FLAG_DONE)
    
    # Close output (don't unlink - main process owns it)
    output_shm.close()
    
except Exception as e:
    import traceback
    print(f"❌ Error: {{e}}", flush=True)
    print(traceback.format_exc(), flush=True)
    
    # Try to signal error
    try:
        output_shm = SharedMemory(name="{output_shm_name}")
        error_data = {{
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }}
        error_bytes = pickle.dumps(error_data)
        output_shm.buf[HEADER_SIZE:HEADER_SIZE+len(error_bytes)] = error_bytes
        set_flag(output_shm.buf, FLAG_ERROR)
        output_shm.close()
    except Exception as cleanup_err:
        print(f"Failed to signal error: {{cleanup_err}}", flush=True)
    
    sys.exit(1)
'''
        
        # Write script to temp file
        with tempfile.NamedTemporaryFile(
            mode='w', delete=False, suffix='.py', encoding='utf-8'
        ) as script_file:
            script_path = script_file.name
            script_file.write(exec_script)
        
        try:
            # Execute in isolated environment
            self.logger.debug(f"Spawning subprocess: {python_exe}")
            
            process = subprocess.Popen(
                [python_exe, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for subprocess to complete (it will set FLAG_DONE or FLAG_ERROR)
            # The subprocess writes to our pre-created output shared memory
            self.logger.debug("Waiting for subprocess to complete...")
            
            # Wait for FLAG_DONE or FLAG_ERROR on output_shm
            if not wait_for_flag(output_shm.buf, FLAG_DONE, timeout=timeout):
                # Check if error flag was set
                flag = get_flag(output_shm.buf)
                if flag == FLAG_ERROR:
                    self.logger.error("Subprocess signaled error")
                else:
                    # Process might still be running, terminate it
                    process.kill()
                    raise TimeoutError(
                        f"Subprocess execution timeout after {timeout}s. Flag={flag}"
                    )
            
            # Read subprocess output for logging
            try:
                stdout, stderr = process.communicate(timeout=5.0)
                if stdout:
                    self.logger.info(f"Subprocess stdout:\n{stdout}")
                if stderr:
                    self.logger.warning(f"Subprocess stderr:\n{stderr}")
            except subprocess.TimeoutExpired:
                self.logger.warning("Subprocess stdout/stderr read timeout")
            
            # Read result from output shared memory - COPY bytes to avoid holding reference
            import pickle
            output_data_bytes = bytes(output_shm.buf[HEADER_SIZE:])
            
            # Get flag before closing
            flag = get_flag(output_shm.buf)
            
            # Now we can unpickle from the copied bytes
            try:
                output_data = pickle.loads(output_data_bytes)
            except Exception as e:
                self.logger.error(f"Failed to unpickle output data: {e}")
                raise
            
            # Check status
            if flag == FLAG_DONE and output_data.get('status') == 'success':
                self.logger.info("✅ Subprocess completed successfully")
                result = output_data['result']
                
            elif flag == FLAG_ERROR or output_data.get('status') == 'error':
                self.logger.error("❌ Subprocess execution failed")
                error_msg = output_data.get('error', 'Unknown error')
                traceback_msg = output_data.get('traceback', '')
                raise RuntimeError(
                    f"Node execution failed: {error_msg}\n{traceback_msg}"
                )
            else:
                raise RuntimeError(
                    f"Unexpected result status. Flag={flag}, Status={output_data.get('status')}"
                )
            
            # Cleanup shared memory (main process owns both)
            input_shm.close()
            input_shm.unlink()
            output_shm.close()
            output_shm.unlink()
            
            self.logger.info(
                f"🎉 Node {node_type} completed with true shared memory"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"True shared memory execution failed: {e}")
            
            # Try to cleanup shared memory on error
            try:
                input_shm.close()
                input_shm.unlink()
            except:
                pass
                
            try:
                output_shm.close()
                output_shm.unlink()
            except:
                pass
            
            raise
            
        finally:
            # Cleanup script file
            try:
                os.unlink(script_path)
            except:
                pass


if __name__ == "__main__":
    # Test the environment manager
    print("🧪 Testing Workflow Environment Manager")
    print("=" * 60)
    
    manager = WorkflowEnvironmentManager()
    
    print("\n📋 Available Environments:")
    for env in manager.list_environments():
        print(f"  {env['type']:10} | {env['name']:20} | {env['path']}")
    
    print("\n✅ Environment Manager ready!")
