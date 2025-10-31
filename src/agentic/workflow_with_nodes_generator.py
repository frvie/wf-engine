"""
Complete Workflow Generator with Automatic Node Generation

This module combines template-based workflow generation with automatic node
code generation. It uses node IDs from workflows as references to generate
missing node implementations following the @workflow_node decorator pattern.

Features:
- Template-based workflow generation using existing workflows as examples
- Automatic detection of missing nodes from generated workflows
- Node ID-based code generation using the decorator pattern
- Integration with the existing NodeGenerator system
- Complete end-to-end workflow creation
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Import our existing components
from src.nodes.node_generator import NodeGenerator, NodeSpec
from src.agentic.template_generator import (
    TemplateBasedWorkflowGenerator, WorkflowTemplateLibrary
)

logger = logging.getLogger(__name__)


@dataclass
class WorkflowNodeRef:
    """Reference to a node in a workflow."""
    id: str                     # Node ID in workflow
    function_path: str          # Full function path (e.g., "src.nodes.custom.clean_text.clean_text_node")
    dependencies: List[str]     # Node dependencies
    inputs: Dict[str, Any]      # Node inputs
    description: str = ""       # Inferred description from ID and inputs


class CompleteWorkflowGenerator:
    """
    Generates complete workflows with automatic node implementation.
    
    This class combines:
    1. Template-based workflow generation (JSON structure)
    2. Node ID analysis and missing node detection  
    3. Automatic node code generation using decorators
    4. Integration testing of generated workflow
    """
    
    def __init__(self, 
                 llm_model: str = "qwen2.5-coder:7b",
                 workflows_dir: str = "workflows",
                 nodes_dir: str = "src/nodes"):
        """
        Initialize the complete workflow generator.
        
        Args:
            llm_model: Ollama model for generation
            workflows_dir: Directory containing workflow templates and outputs
            nodes_dir: Directory containing node implementations
        """
        self.llm_model = llm_model
        self.workflows_dir = Path(workflows_dir)
        self.nodes_dir = Path(nodes_dir)
        
        # Initialize sub-generators
        self.workflow_generator = TemplateBasedWorkflowGenerator(
            template_library=WorkflowTemplateLibrary(workflows_dir),
            model_name=llm_model
        )
        self.node_generator = NodeGenerator(llm_model)
        
        logger.info(f"CompleteWorkflowGenerator initialized")
        logger.info(f"  LLM Model: {llm_model}")
        logger.info(f"  Workflows Dir: {workflows_dir}")
        logger.info(f"  Nodes Dir: {nodes_dir}")
    
    async def generate_complete_workflow(self, 
                                       user_request: str,
                                       workflow_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a complete workflow with all required nodes implemented.
        
        Args:
            user_request: Natural language description of desired workflow
            workflow_name: Optional custom workflow name
            
        Returns:
            dict: {
                'workflow_file': Path,
                'workflow_data': dict,
                'generated_nodes': List[Dict],
                'missing_nodes': List[str],
                'success': bool,
                'error': Optional[str]
            }
        """
        try:
            logger.info(f"🚀 Starting complete workflow generation")
            logger.info(f"   Request: {user_request}")
            
            # Step 1: Generate workflow structure using templates
            logger.info("📋 Step 1: Generating workflow structure...")
            workflow_result = await self.workflow_generator.generate_workflow(user_request)
            
            if not workflow_result['success']:
                return {
                    'workflow_file': None,
                    'workflow_data': None,
                    'generated_nodes': [],
                    'missing_nodes': [],
                    'success': False,
                    'error': f"Workflow generation failed: {workflow_result.get('error')}"
                }
            
            workflow_data = workflow_result['workflow_data']
            workflow_file = workflow_result['workflow_file']
            
            # Step 2: Analyze nodes and detect missing implementations
            logger.info("🔍 Step 2: Analyzing nodes and detecting missing implementations...")
            node_analysis = self._analyze_workflow_nodes(workflow_data)
            missing_nodes = node_analysis['missing_nodes']
            existing_nodes = node_analysis['existing_nodes']
            
            logger.info(f"   Found {len(existing_nodes)} existing nodes")
            logger.info(f"   Need to generate {len(missing_nodes)} missing nodes")
            
            # Step 3: Generate missing node implementations
            generated_nodes = []
            if missing_nodes:
                logger.info("⚙️ Step 3: Generating missing node implementations...")
                for node_ref in missing_nodes:
                    logger.info(f"   Generating node: {node_ref.id}")
                    
                    node_spec = self._create_node_spec_from_ref(node_ref, workflow_data)
                    generation_result = await self.node_generator.generate_node(node_spec)
                    
                    if generation_result['success']:
                        generated_nodes.append({
                            'node_id': node_ref.id,
                            'function_path': node_ref.function_path,
                            'file_path': generation_result['file_path'],
                            'node_name': generation_result['node_name']
                        })
                        logger.info(f"     ✅ Generated: {generation_result['node_name']}")
                    else:
                        logger.error(f"     ❌ Failed: {generation_result['error']}")
                        # Continue with other nodes even if one fails
            
            # Step 4: Validate and return results
            success = len(generated_nodes) == len(missing_nodes)
            
            result = {
                'workflow_file': workflow_file,
                'workflow_data': workflow_data,
                'generated_nodes': generated_nodes,
                'missing_nodes': [node.id for node in missing_nodes],
                'existing_nodes': [node.id for node in existing_nodes],
                'success': success,
                'error': None if success else f"Generated {len(generated_nodes)}/{len(missing_nodes)} nodes"
            }
            
            # Log final results
            logger.info("🎉 Complete workflow generation finished!")
            logger.info(f"   Workflow: {workflow_file}")
            logger.info(f"   Generated nodes: {len(generated_nodes)}")
            logger.info(f"   Success: {success}")
            
            return result
            
        except Exception as e:
            logger.error(f"Complete workflow generation failed: {e}", exc_info=True)
            return {
                'workflow_file': None,
                'workflow_data': None,
                'generated_nodes': [],
                'missing_nodes': [],
                'success': False,
                'error': str(e)
            }
    
    def _analyze_workflow_nodes(self, workflow_data: Dict) -> Dict[str, List[WorkflowNodeRef]]:
        """
        Analyze workflow nodes to determine which exist and which need generation.
        
        Args:
            workflow_data: Parsed workflow JSON
            
        Returns:
            dict: {
                'existing_nodes': List[WorkflowNodeRef],
                'missing_nodes': List[WorkflowNodeRef]  
            }
        """
        existing_nodes = []
        missing_nodes = []
        
        nodes = workflow_data.get('nodes', [])
        
        for node in nodes:
            node_id = node.get('id', '')
            function_path = node.get('function', '')
            dependencies = node.get('dependencies', [])
            inputs = node.get('inputs', {})
            
            # Create node reference
            node_ref = WorkflowNodeRef(
                id=node_id,
                function_path=function_path,
                dependencies=dependencies,
                inputs=inputs,
                description=self._infer_node_description(node_id, inputs)
            )
            
            # Check if node implementation exists
            if self._node_implementation_exists(function_path):
                existing_nodes.append(node_ref)
            else:
                missing_nodes.append(node_ref)
        
        return {
            'existing_nodes': existing_nodes,
            'missing_nodes': missing_nodes
        }
    
    def _node_implementation_exists(self, function_path: str) -> bool:
        """
        Check if a node implementation file exists.
        
        Args:
            function_path: Full function path (e.g., "src.nodes.custom.clean_text.clean_text_node")
            
        Returns:
            bool: True if implementation exists
        """
        try:
            # Convert function path to file path
            # "src.nodes.custom.clean_text.clean_text_node" -> "src/nodes/custom/clean_text.py"
            parts = function_path.split('.')
            if len(parts) < 2:
                return False
            
            # Remove function name, keep module path
            module_parts = parts[:-1]  # Remove function name
            module_path = '/'.join(module_parts) + '.py'
            
            file_path = Path(module_path)
            exists = file_path.exists()
            
            logger.debug(f"Checking node implementation: {function_path}")
            logger.debug(f"  File path: {file_path}")
            logger.debug(f"  Exists: {exists}")
            
            return exists
            
        except Exception as e:
            logger.warning(f"Error checking node implementation {function_path}: {e}")
            return False
    
    def _infer_node_description(self, node_id: str, inputs: Dict[str, Any]) -> str:
        """
        Infer node description from ID and inputs.
        
        Args:
            node_id: Node identifier
            inputs: Node input specification
            
        Returns:
            str: Inferred description
        """
        # Convert snake_case or kebab-case to readable description
        description = node_id.replace('_', ' ').replace('-', ' ').title()
        
        # Add context from inputs
        input_names = list(inputs.keys())
        if input_names:
            description += f" (inputs: {', '.join(input_names)})"
        
        return description
    
    def _create_node_spec_from_ref(self, node_ref: WorkflowNodeRef, workflow_data: Dict) -> NodeSpec:
        """
        Create a NodeSpec from a WorkflowNodeRef for code generation.
        
        Args:
            node_ref: Reference to node in workflow
            workflow_data: Complete workflow data for context
            
        Returns:
            NodeSpec: Specification for node generation
        """
        # Extract function name from path
        function_name = node_ref.function_path.split('.')[-1]
        
        # Determine inputs from node inputs
        inputs = list(node_ref.inputs.keys())
        if not inputs:
            inputs = ['input']  # Default input name
        
        # Infer outputs based on node type
        outputs = self._infer_outputs_from_node_id(node_ref.id)
        
        # Determine category from function path
        path_parts = node_ref.function_path.split('.')
        if 'custom' in path_parts:
            category = 'custom'
        elif 'atomic' in path_parts:
            category = 'atomic'
        else:
            category = 'custom'
        
        # Create enhanced description with context
        description = self._create_enhanced_description(node_ref, workflow_data)
        
        # Generate constraints based on workflow context
        constraints = self._generate_constraints_from_context(node_ref, workflow_data)
        
        return NodeSpec(
            goal=description,
            inputs=inputs,
            outputs=outputs,
            category=category,
            description=description,
            constraints=constraints,
            examples=None  # Could be generated from similar nodes
        )
    
    def _infer_outputs_from_node_id(self, node_id: str) -> List[str]:
        """Infer output names from node ID."""
        
        # Common patterns for outputs based on node type
        output_patterns = {
            'fetch': ['content'],
            'parse': ['text'],
            'clean': ['cleaned_text'],
            'split': ['chunks'],
            'chunk': ['chunks'],
            'summarize': ['summaries'],
            'combine': ['result'],
            'analyze': ['analysis'],
            'detect': ['detections'],
            'process': ['processed_data'],
            'extract': ['extracted_data'],
            'transform': ['transformed_data'],
            'filter': ['filtered_data'],
            'validate': ['validation_result'],
            'format': ['formatted_data'],
            'save': ['success'],
            'load': ['data'],
            'convert': ['converted_data']
        }
        
        # Check if node_id contains any pattern keywords
        for pattern, outputs in output_patterns.items():
            if pattern in node_id.lower():
                return outputs
        
        # Default output
        return ['output']
    
    def _create_enhanced_description(self, node_ref: WorkflowNodeRef, workflow_data: Dict) -> str:
        """Create enhanced description with workflow context."""
        
        base_desc = node_ref.description
        workflow_name = workflow_data.get('workflow', {}).get('name', 'Unknown Workflow')
        
        # Add workflow context
        context_desc = f"{base_desc} as part of {workflow_name}"
        
        # Add dependency context if available
        if node_ref.dependencies:
            dep_context = f" (depends on: {', '.join(node_ref.dependencies)})"
            context_desc += dep_context
        
        return context_desc
    
    def _generate_constraints_from_context(self, node_ref: WorkflowNodeRef, workflow_data: Dict) -> List[str]:
        """Generate implementation constraints based on workflow context."""
        
        constraints = [
            "Follow atomic design principles (single responsibility)",
            "Use @workflow_node decorator with appropriate isolation_mode",
            "Handle errors gracefully with proper exception handling",
            "Include comprehensive docstring with type information",
            "Return data in format compatible with dependent nodes"
        ]
        
        # Add type-specific constraints based on node ID
        node_type = node_ref.id.lower()
        
        if 'web' in node_type or 'fetch' in node_type:
            constraints.extend([
                "Use requests library for HTTP operations",
                "Include proper timeout handling",
                "Validate URL format before making requests"
            ])
        
        if 'text' in node_type or 'parse' in node_type:
            constraints.extend([
                "Handle different text encodings",
                "Strip unnecessary whitespace and formatting",
                "Preserve important text structure"
            ])
        
        if 'chunk' in node_type or 'split' in node_type:
            constraints.extend([
                "Implement configurable chunk size and overlap",
                "Maintain text coherence across chunk boundaries",
                "Return chunks as list or generator as appropriate"
            ])
        
        return constraints

    async def generate_workflow_from_template(self, 
                                            template_name: str,
                                            customizations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate workflow from existing template with customizations.
        
        Args:
            template_name: Name of template workflow to use as base
            customizations: Dictionary of customizations to apply
            
        Returns:
            dict: Generation result similar to generate_complete_workflow
        """
        # This could extend the system to use specific templates
        # and apply targeted customizations
        pass


# Convenience function for CLI usage
async def generate_complete_workflow_cli(user_request: str, 
                                       model_name: str = "qwen2.5-coder:7b") -> Dict[str, Any]:
    """
    CLI convenience function for complete workflow generation.
    
    Args:
        user_request: Natural language workflow description
        model_name: Ollama model to use
        
    Returns:
        dict: Complete generation results
    """
    generator = CompleteWorkflowGenerator(llm_model=model_name)
    return await generator.generate_complete_workflow(user_request)


# Synchronous wrapper
def generate_complete_workflow_sync(user_request: str, 
                                  model_name: str = "qwen2.5-coder:7b") -> Dict[str, Any]:
    """Synchronous wrapper for complete workflow generation."""
    return asyncio.run(generate_complete_workflow_cli(user_request, model_name))


if __name__ == "__main__":
    # Demo: Generate a complete web summarization workflow
    import sys
    
    if len(sys.argv) > 1:
        request = sys.argv[1]
    else:
        request = "Create a workflow to summarize the content of a web page"
    
    print(f"🚀 Generating complete workflow for: {request}")
    print("=" * 60)
    
    result = generate_complete_workflow_sync(request)
    
    if result['success']:
        print("✅ Complete workflow generation successful!")
        print(f"   Workflow file: {result['workflow_file']}")
        print(f"   Generated {len(result['generated_nodes'])} new nodes")
        print(f"   Found {len(result['existing_nodes'])} existing nodes")
        
        if result['generated_nodes']:
            print("\n📝 Generated Nodes:")
            for node in result['generated_nodes']:
                print(f"     • {node['node_id']} → {node['file_path']}")
    else:
        print(f"❌ Generation failed: {result['error']}")