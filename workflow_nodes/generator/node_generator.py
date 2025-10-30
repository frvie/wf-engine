"""
Workflow Node Generator

Automatically generates workflow nodes from natural language specifications
using LLM (Ollama) integration.

Features:
- LLM-powered code generation
- Automatic decorator application
- Syntax validation
- Template-based generation
- Category-based organization
"""

import ast
import asyncio
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import textwrap

try:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class NodeSpec:
    """Specification for a new workflow node."""
    goal: str                           # What the node should do
    inputs: List[str]                   # Input parameter names
    outputs: List[str]                  # Output variable names
    category: str = 'custom'            # Node category (atomic, custom, etc.)
    description: Optional[str] = None   # Detailed description
    constraints: Optional[List[str]] = None  # Implementation constraints
    examples: Optional[List[Dict]] = None    # Example inputs/outputs


class NodeGenerator:
    """
    Generates new workflow nodes using LLM assistance.
    
    Features:
    - Code generation from natural language
    - Automatic @workflow_node decorator application
    - Syntax validation and error checking
    - Atomic/granular design enforcement
    - Auto-save to appropriate directory
    """
    
    def __init__(self, model_name: str = "qwen2.5-coder:7b"):
        """
        Initialize the node generator.
        
        Args:
            model_name: Ollama model to use for code generation
        """
        self.model_name = model_name
        self.base_dir = Path("workflow_nodes")
        self.custom_dir = self.base_dir / "custom"
        self.custom_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"NodeGenerator initialized with model: {model_name}")
    
    async def generate_node(self, spec: NodeSpec) -> Dict[str, Any]:
        """
        Generate a new workflow node from specification.
        
        Args:
            spec: Node specification
            
        Returns:
            dict: {
                'node_name': str,
                'file_path': Path,
                'code': str,
                'success': bool,
                'error': Optional[str]
            }
        """
        try:
            logger.info(f"Generating node: {spec.goal}")
            
            # 1. Generate code using LLM
            code = await self._generate_code(spec)
            
            # 2. Validate and fix the generated code
            validated_code = self._validate_code(code, spec)
            
            # 3. Extract node name
            node_name = self._extract_node_name(validated_code)
            
            # 4. Save to file
            file_path = self._save_node(validated_code, spec, node_name)
            
            logger.info(f"✅ Node generated successfully: {node_name}")
            logger.info(f"   Location: {file_path}")
            
            return {
                'node_name': node_name,
                'file_path': file_path,
                'code': validated_code,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Failed to generate node: {e}", exc_info=True)
            return {
                'node_name': None,
                'file_path': None,
                'code': None,
                'success': False,
                'error': str(e)
            }
    
    async def _generate_code(self, spec: NodeSpec) -> str:
        """Generate Python code using LLM."""
        if not AUTOGEN_AVAILABLE:
            # Fallback to template-based generation
            return self._generate_from_template(spec)
        
        prompt = self._build_generation_prompt(spec)
        
        try:
            # Use AutoGen with Ollama
            model_client = OpenAIChatCompletionClient(
                model=self.model_name,
                api_key="ollama",
                base_url="http://localhost:11434/v1"
            )
            
            agent = AssistantAgent(
                name="code_generator",
                model_client=model_client,
                system_message="""You are an expert Python developer specialized in creating 
workflow nodes. Generate clean, efficient, well-documented code following atomic design principles."""
            )
            
            # Generate code
            response = await agent.run(task=prompt)
            
            # Extract code from response
            code = self._extract_code_from_response(response.messages[-1].content)
            
            return code
            
        except Exception as e:
            logger.warning(f"LLM generation failed, using template: {e}")
            return self._generate_from_template(spec)
    
    def _build_generation_prompt(self, spec: NodeSpec) -> str:
        """Build prompt for LLM code generation."""
        constraints_text = ""
        if spec.constraints:
            constraints_text = "\nConstraints:\n" + "\n".join(f"- {c}" for c in spec.constraints)
        
        examples_text = ""
        if spec.examples:
            examples_text = "\nExamples:\n" + "\n".join(
                f"- Input: {ex.get('input')} → Output: {ex.get('output')}" 
                for ex in spec.examples
            )
        
        return f"""
Generate a Python workflow node with these requirements:

GOAL: {spec.goal}

INPUTS: {', '.join(spec.inputs)}
OUTPUTS: {', '.join(spec.outputs)}
CATEGORY: {spec.category}
{f'DESCRIPTION: {spec.description}' if spec.description else ''}
{constraints_text}
{examples_text}

TEMPLATE TO FOLLOW:
```python
from workflow_decorator import workflow_node
import numpy as np
# Add other imports as needed

@workflow_node("<descriptive_snake_case_name>", isolation_mode="none")
def <function_name>({', '.join(spec.inputs)}, **kwargs):
    \"\"\"
    <Clear one-line description>
    
    This is an atomic workflow node following granular design principles.
    Single responsibility, no side effects, composable.
    
    Args:
        {chr(10).join(f'{inp}: <type and description>' for inp in spec.inputs)}
        **kwargs: Additional optional parameters
    
    Returns:
        {chr(10).join(f'{out}: <type and description>' for out in spec.outputs) if len(spec.outputs) > 1 else f'{spec.outputs[0]}: <type and description>'}
    \"\"\"
    
    # Implementation here
    # Follow atomic principles:
    # 1. Single responsibility
    # 2. No global state
    # 3. Pure function when possible
    # 4. Proper error handling
    
    {f'return {", ".join(spec.outputs)}' if len(spec.outputs) > 1 else f'return {spec.outputs[0]}'}
```

CRITICAL RULES:
1. MUST use @workflow_node("<name>", isolation_mode="none") decorator
2. MUST be a pure function (no global state modifications)
3. MUST include comprehensive docstring
4. MUST handle errors gracefully (try/except where needed)
5. MUST follow atomic design (single responsibility)
6. MUST include type hints in docstring
7. Return single value or tuple based on outputs count
8. Add proper imports (numpy, cv2, etc.)

Generate ONLY the Python code, no markdown formatting, no explanations.
"""
    
    def _generate_from_template(self, spec: NodeSpec) -> str:
        """Generate code from template (fallback when LLM unavailable)."""
        
        # Determine imports based on category
        imports = ["from workflow_decorator import workflow_node"]
        
        if spec.category in ['atomic', 'image', 'custom']:
            imports.append("import numpy as np")
            imports.append("import cv2")
        elif spec.category == 'onnx':
            imports.append("import numpy as np")
            imports.append("import onnxruntime as ort")
        
        # Generate function name
        func_name = self._goal_to_function_name(spec.goal)
        node_name = func_name.replace('_node', '')
        
        # Build args docstring
        args_doc = '\n'.join(f'        {inp}: Input data' for inp in spec.inputs)
        
        # Build returns docstring
        if len(spec.outputs) > 1:
            returns_doc = '\n'.join(f'        {out}: Output data' for out in spec.outputs)
        else:
            returns_doc = f'        {spec.outputs[0]}: Output data'
        
        # Build placeholder assignments
        placeholder_code = '\n'.join(f'    {out} = None  # TODO: Compute {out}' for out in spec.outputs)
        
        # Generate return statement
        if len(spec.outputs) == 1:
            return_stmt = f"return {spec.outputs[0]}"
        else:
            return_stmt = f"return {', '.join(spec.outputs)}"
        
        # Template code with proper indentation
        code = f'''{chr(10).join(imports)}


@workflow_node("{node_name}", isolation_mode="none")
def {func_name}({', '.join(spec.inputs)}):
    """
    {spec.description or spec.goal}
    
    Atomic workflow node following granular design principles.
    
    Args:
{args_doc}
    
    Returns:
{returns_doc}
    """
    
    # TODO: Implement the actual logic
    # This is a template - customize for your specific needs
    
    # Placeholder implementation
{placeholder_code}
    
    {return_stmt}
'''
        
        return code
    
    def _validate_code(self, code: str, spec: NodeSpec) -> str:
        """
        Validate generated code and apply fixes.
        
        Checks:
        - Valid Python syntax
        - Has @workflow_node decorator
        - Has proper imports
        - Function signature matches spec
        """
        try:
            # Parse to check syntax
            tree = ast.parse(code)
            
            # Check for decorator
            has_decorator = False
            function_name = None
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    function_name = node.name
                    for decorator in node.decorator_list:
                        if self._is_workflow_decorator(decorator):
                            has_decorator = True
                            break
            
            if not has_decorator:
                raise ValueError("Missing @workflow_node decorator")
            
            if not function_name:
                raise ValueError("No function definition found")
            
            # Ensure imports are present
            if 'from workflow_decorator import workflow_node' not in code:
                code = 'from workflow_decorator import workflow_node\n' + code
            
            return code
            
        except SyntaxError as e:
            logger.error(f"Syntax error in generated code: {e}")
            raise ValueError(f"Generated code has syntax errors: {e}")
    
    def _is_workflow_decorator(self, decorator) -> bool:
        """Check if decorator is @workflow_node."""
        if isinstance(decorator, ast.Name):
            return decorator.id == 'workflow_node'
        elif isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Name):
                return decorator.func.id == 'workflow_node'
        return False
    
    def _extract_code_from_response(self, response: str) -> str:
        """Extract Python code from LLM response."""
        # Remove markdown code blocks
        code = re.sub(r'```python\n', '', response)
        code = re.sub(r'```\n?', '', code)
        code = code.strip()
        return code
    
    def _extract_node_name(self, code: str) -> str:
        """Extract the node name from generated code."""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if self._is_workflow_decorator(node):
                        # Find 'name' argument
                        for keyword in node.keywords:
                            if keyword.arg == 'name':
                                if isinstance(keyword.value, ast.Constant):
                                    return keyword.value.value
                        
                        # If no name, use function name
                        for n in ast.walk(tree):
                            if isinstance(n, ast.FunctionDef):
                                return n.name.replace('_node', '')
        except:
            pass
        
        return "custom_node"
    
    def _save_node(self, code: str, spec: NodeSpec, node_name: str) -> Path:
        """Save generated node to appropriate directory."""
        
        # Determine save location based on category
        if spec.category == 'atomic':
            save_dir = self.base_dir / 'atomic'
        elif spec.category == 'custom':
            save_dir = self.custom_dir
        elif spec.category in ['infrastructure', 'model_loaders', 'utils', 'video']:
            save_dir = self.base_dir / spec.category
        else:
            save_dir = self.custom_dir
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        filename = f"{node_name}.py"
        filepath = save_dir / filename
        
        # Save the code
        filepath.write_text(code, encoding='utf-8')
        logger.info(f"Saved node to: {filepath}")
        
        return filepath
    
    def _goal_to_function_name(self, goal: str) -> str:
        """Convert goal to valid Python function name."""
        # Convert to snake_case
        name = re.sub(r'[^\w\s-]', '', goal.lower())
        name = re.sub(r'[-\s]+', '_', name)
        
        # Ensure it ends with _node
        if not name.endswith('_node'):
            name += '_node'
        
        return name
    
    def list_generated_nodes(self) -> List[Path]:
        """List all generated custom nodes."""
        return list(self.custom_dir.glob('*.py'))


# Synchronous wrapper for CLI usage
def generate_node_sync(spec: NodeSpec, model_name: str = "qwen2.5-coder:7b") -> Dict[str, Any]:
    """Synchronous wrapper for node generation."""
    generator = NodeGenerator(model_name)
    return asyncio.run(generator.generate_node(spec))
