#!/usr/bin/env python3
"""
Workflow-Aware Node Generation System

This system analyzes complete workflow data flow and generates nodes that are
perfectly compatible with each other, solving data format mismatches.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
import re
import inspect

logging.basicConfig(level=logging.INFO, format='%(name)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

class WorkflowDataFlowAnalyzer:
    """
    Analyzes workflow structure to understand complete data flow between nodes.
    Provides context for generating compatible nodes.
    """
    
    def __init__(self, workflow_path: str):
        self.workflow_path = Path(workflow_path)
        self.workflow_data = None
        self.nodes = {}
        self.connections = {}
        self.data_contracts = {}
        
        self._load_workflow()
        self._analyze_structure()
    
    def _load_workflow(self):
        """Load and parse the workflow JSON."""
        with open(self.workflow_path, 'r') as f:
            self.workflow_data = json.load(f)
        
        logger.info(f"Loaded workflow: {self.workflow_data['workflow']['name']}")
    
    def _analyze_structure(self):
        """Analyze workflow structure to understand node connections and data flow."""
        for node in self.workflow_data['nodes']:
            node_id = node['id']
            self.nodes[node_id] = {
                'id': node_id,
                'function': node['function'],
                'dependencies': node.get('dependencies', []),
                'inputs': node.get('inputs', {}),
                'upstream': [],
                'downstream': [],
                'input_mappings': {},
                'expected_outputs': set(),
                'required_inputs': set()
            }
        
        # Build connection graph
        for node_id, node_info in self.nodes.items():
            for dep in node_info['dependencies']:
                if dep in self.nodes:
                    self.nodes[dep]['downstream'].append(node_id)
                    self.nodes[node_id]['upstream'].append(dep)
            
            # Analyze input mappings
            for input_key, input_value in node_info['inputs'].items():
                if isinstance(input_value, str) and input_value.startswith('$'):
                    ref = input_value[1:]
                    if '.' in ref:
                        upstream_node, output_key = ref.split('.', 1)
                        self.nodes[node_id]['input_mappings'][input_key] = {
                            'type': 'node_output_key',
                            'source_node': upstream_node,
                            'source_key': output_key
                        }
                        # Track what outputs are expected
                        if upstream_node in self.nodes:
                            self.nodes[upstream_node]['expected_outputs'].add(output_key)
                    else:
                        upstream_node = ref
                        self.nodes[node_id]['input_mappings'][input_key] = {
                            'type': 'full_node_output',
                            'source_node': upstream_node
                        }
                else:
                    self.nodes[node_id]['input_mappings'][input_key] = {
                        'type': 'literal',
                        'value': input_value
                    }
                
                # Track required inputs for this node
                self.nodes[node_id]['required_inputs'].add(input_key)
    
    def _inspect_existing_node(self, function_path: str) -> Dict[str, Any]:
        """Inspect existing node implementation to understand its data contract."""
        try:
            # Convert function path to file path
            parts = function_path.split('.')
            if len(parts) >= 4:
                file_path = Path("src") / "nodes" / parts[2] / f"{parts[3]}.py"
                
                if file_path.exists():
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    # Extract return statement patterns
                    return_patterns = self._extract_return_patterns(content)
                    
                    # Extract function signature
                    func_signature = self._extract_function_signature(content)
                    
                    return {
                        'file_exists': True,
                        'return_patterns': return_patterns,
                        'function_signature': func_signature,
                        'content': content
                    }
        except Exception as e:
            logger.debug(f"Could not inspect {function_path}: {e}")
        
        return {'file_exists': False}
    
    def _extract_return_patterns(self, content: str) -> List[Dict[str, Any]]:
        """Extract return statement patterns from node code."""
        patterns = []
        
        # Find return statements with dict literals
        return_matches = re.finditer(r'return\s*{([^}]+)}', content, re.DOTALL)
        
        for match in return_matches:
            dict_content = match.group(1)
            
            # Extract key-value pairs
            keys = re.findall(r'"([^"]+)":', dict_content)
            patterns.append({
                'type': 'dict_return',
                'keys': keys,
                'raw': match.group(0)
            })
        
        return patterns
    
    def _extract_function_signature(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract function signature and parameters."""
        # Find function definition
        func_match = re.search(r'def\s+(\w+)\s*\(([^)]+)\)', content)
        if func_match:
            func_name = func_match.group(1)
            params_str = func_match.group(2)
            
            # Parse parameters
            params = []
            for param in params_str.split(','):
                param = param.strip()
                if param and param != '**kwargs':
                    param_name = param.split(':')[0].strip()
                    params.append(param_name)
            
            return {
                'name': func_name,
                'parameters': params
            }
        
        return None
    
    def analyze_node_context(self, target_node_id: str) -> Dict[str, Any]:
        """
        Analyze complete context for a specific node including:
        - What inputs it receives and their exact formats
        - What outputs downstream nodes expect
        - Existing node implementations for reference
        """
        if target_node_id not in self.nodes:
            return {}
        
        node_info = self.nodes[target_node_id]
        context = {
            'node_id': target_node_id,
            'function_path': node_info['function'],
            'required_inputs': list(node_info['required_inputs']),
            'input_sources': {},
            'expected_outputs': list(node_info['expected_outputs']),
            'downstream_expectations': {},
            'upstream_contracts': {},
            'existing_implementation': None
        }
        
        # Analyze input sources - what format will each input have?
        for input_key, mapping in node_info['input_mappings'].items():
            if mapping['type'] == 'full_node_output':
                source_node = mapping['source_node']
                if source_node in self.nodes:
                    # Inspect the source node to see what it returns
                    source_func_path = self.nodes[source_node]['function']
                    source_inspection = self._inspect_existing_node(source_func_path)
                    
                    context['input_sources'][input_key] = {
                        'type': 'full_node_output',
                        'source_node': source_node,
                        'source_inspection': source_inspection
                    }
            
            elif mapping['type'] == 'node_output_key':
                source_node = mapping['source_node']
                source_key = mapping['source_key']
                if source_node in self.nodes:
                    source_func_path = self.nodes[source_node]['function']
                    source_inspection = self._inspect_existing_node(source_func_path)
                    
                    context['input_sources'][input_key] = {
                        'type': 'node_output_key',
                        'source_node': source_node,
                        'source_key': source_key,
                        'source_inspection': source_inspection
                    }
            
            else:  # literal
                context['input_sources'][input_key] = {
                    'type': 'literal',
                    'value': mapping['value']
                }
        
        # Analyze downstream expectations
        for downstream_node in node_info['downstream']:
            downstream_info = self.nodes[downstream_node]
            downstream_func_path = downstream_info['function']
            downstream_inspection = self._inspect_existing_node(downstream_func_path)
            
            # Find how downstream node uses our output
            downstream_input_needs = {}
            for down_input_key, down_mapping in downstream_info['input_mappings'].items():
                if (down_mapping.get('source_node') == target_node_id):
                    downstream_input_needs[down_input_key] = {
                        'mapping_type': down_mapping['type'],
                        'expected_key': down_mapping.get('source_key'),
                        'downstream_inspection': downstream_inspection
                    }
            
            if downstream_input_needs:
                context['downstream_expectations'][downstream_node] = downstream_input_needs
        
        # Check if target node already exists
        context['existing_implementation'] = self._inspect_existing_node(node_info['function'])
        
        return context
    
    def generate_compatible_nodes_spec(self) -> List[Dict[str, Any]]:
        """
        Generate specifications for all missing nodes that are compatible with
        the complete workflow data flow.
        """
        missing_nodes = []
        
        for node_id, node_info in self.nodes.items():
            existing = self._inspect_existing_node(node_info['function'])
            
            if not existing['file_exists']:
                context = self.analyze_node_context(node_id)
                
                # Generate smart specification based on context
                spec = self._create_node_specification(node_id, context)
                missing_nodes.append(spec)
        
        return missing_nodes
    
    def _create_node_specification(self, node_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create a detailed specification for generating a compatible node."""
        
        # Analyze input handling requirements
        input_handling_code = self._generate_input_handling_patterns(context)
        
        # Analyze output format requirements  
        output_format_code = self._generate_output_format_patterns(context)
        
        # Generate comprehensive constraints
        constraints = self._generate_workflow_constraints(context)
        
        return {
            'node_id': node_id,
            'function_path': context['function_path'],
            'required_inputs': context['required_inputs'],
            'expected_outputs': context['expected_outputs'],
            'input_handling_code': input_handling_code,
            'output_format_code': output_format_code,
            'constraints': constraints,
            'context': context
        }
    
    def _generate_input_handling_patterns(self, context: Dict[str, Any]) -> str:
        """Generate specific input handling code based on upstream node analysis."""
        patterns = []
        
        for input_key, source_info in context['input_sources'].items():
            if source_info['type'] == 'full_node_output':
                # Analyze what the upstream node actually returns
                source_inspection = source_info.get('source_inspection', {})
                if source_inspection.get('return_patterns'):
                    return_keys = source_inspection['return_patterns'][0].get('keys', [])
                    if return_keys:
                        primary_key = return_keys[0]  # Use first key as primary
                        patterns.append(f"""    # {input_key} from ${source_info['source_node']} (full output)
    # Upstream returns keys: {return_keys}
    {input_key}_data = {input_key} if isinstance({input_key}, (str, list)) else {input_key}.get("{primary_key}", {input_key})""")
                    else:
                        patterns.append(f"""    # {input_key} from ${source_info['source_node']} (full output - unknown format)
    {input_key}_data = {input_key}""")
                else:
                    patterns.append(f"""    # {input_key} from ${source_info['source_node']} (full output - not analyzed)
    {input_key}_data = {input_key}""")
            
            elif source_info['type'] == 'node_output_key':
                patterns.append(f"""    # {input_key} from ${source_info['source_node']}.{source_info['source_key']} (specific key)
    {input_key}_data = {input_key}  # Already extracted by workflow engine""")
            
            else:  # literal
                patterns.append(f"""    # {input_key} is literal value: {source_info['value']}
    {input_key}_data = {input_key}""")
        
        return '\n'.join(patterns) if patterns else "    # No input processing needed"
    
    def _generate_output_format_patterns(self, context: Dict[str, Any]) -> str:
        """Generate specific output format code based on downstream requirements."""
        if not context['downstream_expectations']:
            return """    return {
        "result": processed_data,
        "success": True
    }"""
        
        # Build output dict based on downstream needs
        output_keys = set()
        
        for downstream_node, expectations in context['downstream_expectations'].items():
            for input_key, expectation in expectations.items():
                if expectation['mapping_type'] == 'node_output_key':
                    expected_key = expectation['expected_key']
                    output_keys.add(expected_key)
                else:
                    # Downstream expects full output - need to provide comprehensive dict
                    output_keys.update(['result', 'data', 'output'])
        
        if not output_keys:
            output_keys = {'result'}
        
        # Generate return statement
        lines = ["    return {"]
        for key in sorted(output_keys):
            lines.append(f'        "{key}": processed_data,')
        lines.append('        "success": True')
        lines.append("    }")
        
        return '\n'.join(lines)
    
    def _generate_workflow_constraints(self, context: Dict[str, Any]) -> List[str]:
        """Generate specific constraints based on workflow analysis."""
        constraints = []
        
        # Input format constraints
        for input_key, source_info in context['input_sources'].items():
            if source_info['type'] == 'full_node_output':
                source_node = source_info['source_node']
                constraints.append(f"Input '{input_key}' receives full output dict from {source_node}")
                
                source_inspection = source_info.get('source_inspection', {})
                if source_inspection.get('return_patterns'):
                    keys = source_inspection['return_patterns'][0].get('keys', [])
                    if keys:
                        constraints.append(f"Expected input keys from {source_node}: {keys}")
        
        # Output format constraints  
        for downstream_node, expectations in context['downstream_expectations'].items():
            for input_key, expectation in expectations.items():
                if expectation['mapping_type'] == 'node_output_key':
                    expected_key = expectation['expected_key']
                    constraints.append(f"Must provide output key '{expected_key}' for {downstream_node}")
                else:
                    constraints.append(f"Must provide complete output dict for {downstream_node}")
        
        return constraints

def demo_workflow_aware_generation():
    """Demonstrate complete workflow-aware node generation."""
    print("🎯 WORKFLOW-AWARE NODE GENERATION SYSTEM")
    print("=" * 60)
    
    # Analyze the text summarization workflow
    analyzer = WorkflowDataFlowAnalyzer("workflows/text_summarization_pipeline.json")
    
    print(f"\n📊 Workflow Analysis Complete:")
    print(f"   Total nodes: {len(analyzer.nodes)}")
    print(f"   Connections mapped: {sum(len(n['downstream']) for n in analyzer.nodes.values())}")
    
    # Analyze each node's context
    print(f"\n🔍 Node Context Analysis:")
    for node_id in analyzer.nodes:
        context = analyzer.analyze_node_context(node_id)
        existing = context['existing_implementation']['file_exists']
        status = "✅ EXISTS" if existing else "❌ MISSING"
        print(f"   {status} {node_id}")
        
        if not existing:
            print(f"      Inputs: {context['required_inputs']}")
            print(f"      Expected outputs: {context['expected_outputs']}")
            print(f"      Downstream nodes: {list(context['downstream_expectations'].keys())}")
    
    # Generate specifications for missing nodes
    missing_specs = analyzer.generate_compatible_nodes_spec()
    
    print(f"\n📝 Generated {len(missing_specs)} Compatible Node Specifications:")
    for spec in missing_specs:
        print(f"\n   NODE: {spec['node_id']}")
        print(f"   Constraints: {len(spec['constraints'])}")
        for constraint in spec['constraints']:
            print(f"      - {constraint}")
    
    return analyzer, missing_specs

if __name__ == "__main__":
    analyzer, specs = demo_workflow_aware_generation()