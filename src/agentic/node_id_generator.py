"""
Node ID-Based Code Generation Demo

This demonstrates how to use workflow node IDs as references for automatic
code generation, integrating with the existing @workflow_node decorator pattern.

Key Features:
1. Parse workflow JSON to extract node references
2. Detect missing node implementations  
3. Generate nodes using IDs as specifications
4. Follow existing decorator patterns
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass

from src.nodes.node_generator import NodeGenerator, NodeSpec

logger = logging.getLogger(__name__)


@dataclass
class NodeReference:
    """Reference to a node from workflow JSON."""
    id: str
    function_path: str
    dependencies: List[str]
    inputs: Dict[str, Any]


class NodeIDBasedGenerator:
    """Generate missing nodes based on workflow node IDs and function paths."""
    
    def __init__(self, model_name: str = "qwen2.5-coder:7b"):
        self.model_name = model_name
        self.node_generator = NodeGenerator(model_name)
    
    def parse_workflow_nodes(self, workflow_file: str) -> List[NodeReference]:
        """Parse workflow JSON and extract node references."""
        try:
            with open(workflow_file, 'r') as f:
                workflow_data = json.load(f)
            
            node_refs = []
            for node in workflow_data.get('nodes', []):
                ref = NodeReference(
                    id=node.get('id', ''),
                    function_path=node.get('function', ''),
                    dependencies=node.get('dependencies', []),
                    inputs=node.get('inputs', {})
                )
                node_refs.append(ref)
            
            return node_refs
        
        except Exception as e:
            logger.error(f"Error parsing workflow: {e}")
            return []
    
    def find_missing_nodes(self, node_refs: List[NodeReference]) -> List[NodeReference]:
        """Find nodes that need to be implemented."""
        missing = []
        
        for ref in node_refs:
            # Convert function path to file path
            # "src.nodes.custom.clean_text.clean_text_node" -> "src/nodes/custom/clean_text.py"
            parts = ref.function_path.split('.')
            if len(parts) >= 2:
                file_path = '/'.join(parts[:-1]) + '.py'
                
                if not Path(file_path).exists():
                    missing.append(ref)
                    logger.info(f"Missing node: {ref.id} -> {file_path}")
        
        return missing
    
    def create_node_spec_from_id(self, ref: NodeReference) -> NodeSpec:
        """Create NodeSpec from node ID and reference info."""
        
        # Use node ID to infer purpose and constraints
        node_id = ref.id.lower()
        
        # Determine inputs from workflow inputs
        inputs = list(ref.inputs.keys()) if ref.inputs else ['input']
        
        # Infer outputs based on node ID patterns
        if 'fetch' in node_id:
            outputs = ['content']
        elif 'parse' in node_id:
            outputs = ['text']  
        elif 'clean' in node_id:
            outputs = ['cleaned_text']
        elif 'split' in node_id or 'chunk' in node_id:
            outputs = ['chunks']
        elif 'summarize' in node_id:
            outputs = ['summaries']
        elif 'combine' in node_id:
            outputs = ['result']
        else:
            outputs = ['output']
        
        # Create description from ID
        description = ref.id.replace('_', ' ').title()
        
        # Add constraints based on ID patterns
        constraints = [
            "Use @workflow_node decorator with isolation_mode='none'",
            "Follow atomic design principles",
            "Include proper error handling",
            "Add comprehensive docstring"
        ]
        
        if 'web' in node_id or 'fetch' in node_id:
            constraints.append("Use requests library for web operations")
        
        if 'text' in node_id:
            constraints.append("Handle text encoding and cleanup properly")
        
        return NodeSpec(
            goal=description,
            inputs=inputs,
            outputs=outputs,
            category='custom',
            description=f"Atomic node for {description}",
            constraints=constraints
        )
    
    async def generate_missing_nodes(self, workflow_file: str) -> Dict[str, Any]:
        """Generate all missing nodes from a workflow file."""
        logger.info(f"Processing workflow: {workflow_file}")
        
        # Parse workflow
        node_refs = self.parse_workflow_nodes(workflow_file)
        logger.info(f"Found {len(node_refs)} total nodes")
        
        # Find missing nodes
        missing_nodes = self.find_missing_nodes(node_refs)
        logger.info(f"Need to generate {len(missing_nodes)} missing nodes")
        
        # Generate each missing node
        results = []
        for ref in missing_nodes:
            logger.info(f"Generating node: {ref.id}")
            
            # Create spec from node ID
            spec = self.create_node_spec_from_id(ref)
            
            # Generate the node
            result = await self.node_generator.generate_node(spec)
            
            if result['success']:
                logger.info(f"  ✅ Generated: {result['file_path']}")
            else:
                logger.error(f"  ❌ Failed: {result['error']}")
            
            results.append({
                'node_id': ref.id,
                'function_path': ref.function_path,
                'success': result['success'],
                'file_path': result.get('file_path'),
                'error': result.get('error')
            })
        
        return {
            'total_nodes': len(node_refs),
            'missing_nodes': len(missing_nodes),
            'generated_nodes': [r for r in results if r['success']],
            'failed_nodes': [r for r in results if not r['success']],
            'success': all(r['success'] for r in results)
        }


async def demo_node_generation():
    """Demo: Generate missing nodes from the web summarization workflow."""
    
    generator = NodeIDBasedGenerator()
    
    # Use the workflow we created earlier
    workflow_file = "workflows/granular_web_summarization.json"
    
    print(f"🚀 Demo: Generating nodes from workflow IDs")
    print(f"   Workflow: {workflow_file}")
    print("=" * 60)
    
    result = await generator.generate_missing_nodes(workflow_file)
    
    print(f"📊 Results:")
    print(f"   Total nodes: {result['total_nodes']}")
    print(f"   Missing nodes: {result['missing_nodes']}")
    print(f"   Generated: {len(result['generated_nodes'])}")
    print(f"   Failed: {len(result['failed_nodes'])}")
    
    if result['generated_nodes']:
        print(f"\n✅ Successfully generated nodes:")
        for node in result['generated_nodes']:
            print(f"     • {node['node_id']} -> {node['file_path']}")
    
    if result['failed_nodes']:
        print(f"\n❌ Failed to generate:")
        for node in result['failed_nodes']:
            print(f"     • {node['node_id']}: {node['error']}")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(demo_node_generation())