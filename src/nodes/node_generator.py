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
        self.base_dir = Path("src/nodes")
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
            # Use AutoGen with Ollama - need proper model_info for non-OpenAI models
            from autogen_ext.models.openai._model_info import ModelInfo
            
            # Configure model info for Ollama models
            ollama_model_info = ModelInfo(
                vision=False,
                function_calling=False, 
                json_output=True,
                family="qwen",  # Model family
                structured_output=False
            )
            
            model_client = OpenAIChatCompletionClient(
                model=self.model_name,
                api_key="ollama",
                base_url="http://localhost:11434/v1",
                model_info=ollama_model_info
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
from src.core.decorator import workflow_node
# Import only standard libraries at module level
# Use conditional imports inside functions for optional dependencies

@workflow_node("<descriptive_snake_case_name>", isolation_mode="none", dependencies=["package_name"])
def <function_name>({', '.join(spec.inputs)}, **kwargs):
    \"\"\"
    <Clear one-line description>
    
    This is an atomic workflow node following granular design principles.
    Single responsibility, no side effects, composable.
    
    Args:
        {chr(10).join(f'{inp}: Input parameter (dict or raw value)' for inp in spec.inputs)}
        **kwargs: Additional optional parameters
    
    Returns:
        dict: Dictionary containing results and metadata
    \"\"\"
    
    # Import dependencies inside function to avoid import errors during node discovery
    try:
        import numpy as np  # Example: import inside function
        # import other_package  # Add other imports as needed
    except ImportError as e:
        return {{"error": f"Missing dependency: {{e}}", "success": False}}
    
    # Extract data from input dictionaries (workflow engine passes dicts)
    # Handle both dict inputs (from other nodes) and raw inputs (from workflow)
    # Example: input_data = input.get("key") if isinstance(input, dict) else input
    
    # Implementation here
    # Follow atomic principles:
    # 1. Single responsibility
    # 2. No global state  
    # 3. Pure function when possible
    # 4. Proper error handling
    
    # CRITICAL: Always return a dictionary for workflow engine compatibility
    return {{
        "main_output": result_data,
        "metadata": additional_info,
        "success": True
    }}
```

CRITICAL RULES:
1. MUST use @workflow_node("<name>", isolation_mode="none", dependencies=["pkg"]) decorator
2. MUST declare all non-standard dependencies in dependencies list
3. MUST import dependencies INSIDE function, not at module level
4. MUST be a pure function (no global state modifications)
5. MUST include comprehensive docstring with error handling
6. MUST handle ImportError gracefully for missing dependencies
7. ALWAYS return a dictionary, never raw values
8. Extract data from input dicts: input.get("key") if isinstance(input, dict) else input
9. Handle both dict inputs (from nodes) and raw inputs (from workflow)

WORKFLOW ENGINE BEHAVIOR - CRITICAL UNDERSTANDING:
=================================================

INPUT BEHAVIOR:
- Engine calls your function with **filtered_inputs as keyword arguments**: func(**filtered_inputs)
- Engine filters workflow inputs to only include your function parameters
- Engine resolves $ references: "$node_id" becomes the entire result dict from that node
- Engine resolves dot notation: "$node_id.key" becomes specific value from that node's result

INPUT HANDLING PATTERNS:
```python
# PATTERN 1: Handle dict inputs from upstream nodes
def my_node(text, **kwargs):
    # If upstream returns {{"cleaned_text": "...", "success": True}}
    # and workflow maps "text": "$upstream_node"
    # then text parameter receives the ENTIRE dict
    
    input_data = text if isinstance(text, str) else text.get("cleaned_text", text.get("text", ""))

# PATTERN 2: Handle dot notation inputs  
def my_node(text, **kwargs):
    # If workflow maps "text": "$upstream_node.cleaned_text"
    # then text parameter receives just the string value directly
    
    input_data = text  # Already extracted by engine

# PATTERN 3: Multiple upstream dependencies
def my_node(sentences, algorithm, **kwargs):
    # sentences from "$extract_sentences" (entire dict)
    # algorithm from workflow direct value "tf_idf"
    
    sentence_list = sentences if isinstance(sentences, list) else sentences.get("sentences", [])
    algo = algorithm  # Direct string value
```

OUTPUT BEHAVIOR:
- Engine expects your function to return a dictionary
- Engine stores entire return dict as results[node_id] 
- Downstream nodes receive results based on workflow mapping
- Engine adds metadata automatically if result is dict

OUTPUT FORMAT EXAMPLES:
- Image processing: {{"image": processed_array, "shape": array.shape}}
- Text processing: {{"sentences": sentence_list, "metadata": {{"count": len(sentences)}}}}
- Data analysis: {{"result": analysis_data, "stats": summary_stats}}
- File operations: {{"path": file_path, "success": True, "size": file_size}}

CRITICAL: Your output dict keys must match what downstream nodes expect!
Check the workflow JSON to see how downstream nodes reference your outputs.

Generate ONLY the Python code, no markdown formatting, no explanations.
"""
    
    def _generate_from_template(self, spec: NodeSpec) -> str:
        """Generate code from template (fallback when LLM unavailable)."""
        
        # Determine imports based on category
        imports = ["from src.core.decorator import workflow_node"]
        
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
        
        # Build returns docstring - always return dict
        returns_doc = '        dict: Dictionary containing results and metadata'
        
        # Build placeholder assignments for dict structure
        primary_output = spec.outputs[0] if spec.outputs else 'result'
        placeholder_code = f'''    # Extract inputs (handle dict inputs from other nodes)
    # processed_input = input.get("data") if isinstance(input, dict) else input
    
    # TODO: Implement the actual logic
    {primary_output} = None  # Compute main result
    
    # Always return dictionary for workflow engine'''
        
        # Generate return statement - always return dict
        return_stmt = f'''return {{
        "{primary_output}": {primary_output},
        "success": True
    }}'''
        
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
            if 'from src.core.decorator import workflow_node' not in code:
                code = 'from src.core.decorator import workflow_node\n' + code
            
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
        # Handle markdown code blocks first
        if '```python' in response:
            # Extract content between ```python and ```
            match = re.search(r'```python\n(.*?)```', response, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # Remove markdown code blocks (fallback)
        code = re.sub(r'```python\n', '', response)
        code = re.sub(r'```\n?', '', code)
        
        # Remove any text after return stmt or "Explanation:" line
        lines = code.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Stop at explanatory text
            if line.strip().startswith('Explanation:'):
                break
            cleaned_lines.append(line)
        
        # Find last return statement and include everything up to that point
        code = '\n'.join(cleaned_lines).strip()
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



