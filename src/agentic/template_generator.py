"""
Template-Based Workflow Generator using In-Context Learning

This approach is much simpler than complex domain composers. Instead, we:
1. Collect existing workflow patterns as templates
2. Teach the LLM the workflow engine patterns through examples
3. Use in-context learning to generate new workflows
4. Let the LLM leverage its domain knowledge + our patterns

Key Benefits:
- Simpler architecture (no complex domain composers)
- Uses LLM's existing domain knowledge 
- Learns from your actual workflow patterns
- Easy to extend with new examples
- More flexible and adaptive
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.openai._model_info import ModelInfo

logger = logging.getLogger('workflow.template_generator')


@dataclass
class WorkflowTemplate:
    """A workflow template used for in-context learning."""
    name: str
    description: str
    domain: str                    # e.g., "computer_vision", "web_scraping"
    workflow_json: Dict[str, Any]
    example_usage: str
    key_patterns: List[str]        # Important patterns this template demonstrates


class WorkflowTemplateLibrary:
    """
    Library of workflow templates for in-context learning.
    
    Automatically discovers existing workflows and extracts patterns
    that can be used to teach the LLM how to create new workflows.
    """
    
    def __init__(self, workflows_dir: Path = None):
        self.workflows_dir = workflows_dir or Path("workflows")
        self.templates: Dict[str, WorkflowTemplate] = {}
        self.node_patterns: Dict[str, List[str]] = {}  # Common node usage patterns
        self.logger = logging.getLogger('workflow.templates')
        
    def discover_templates(self):
        """Discover and analyze existing workflow files as templates."""
        if not self.workflows_dir.exists():
            self.logger.warning(f"Workflows directory not found: {self.workflows_dir}")
            return
            
        for workflow_file in self.workflows_dir.glob("*.json"):
            try:
                with open(workflow_file, 'r') as f:
                    workflow_data = json.load(f)
                
                template = self._analyze_workflow_as_template(workflow_file.stem, workflow_data)
                if template:
                    self.templates[template.name] = template
                    
            except Exception as e:
                self.logger.debug(f"Could not analyze {workflow_file}: {e}")
        
        self.logger.info(f"Discovered {len(self.templates)} workflow templates")
        self._extract_node_patterns()
    
    def _analyze_workflow_as_template(self, filename: str, workflow_data: Dict) -> Optional[WorkflowTemplate]:
        """Analyze a workflow file to create a template."""
        workflow_info = workflow_data.get("workflow", {})
        nodes = workflow_data.get("nodes", [])
        
        if not nodes:
            return None
        
        # Determine domain from workflow content
        domain = self._infer_domain_from_workflow(workflow_info, nodes)
        
        # Extract key patterns
        patterns = self._extract_patterns_from_nodes(nodes)
        
        # Create example usage description
        example_usage = self._generate_example_usage(workflow_info, nodes)
        
        return WorkflowTemplate(
            name=filename,
            description=workflow_info.get("description", "Workflow template"),
            domain=domain,
            workflow_json=workflow_data,
            example_usage=example_usage,
            key_patterns=patterns
        )
    
    def _infer_domain_from_workflow(self, workflow_info: Dict, nodes: List[Dict]) -> str:
        """Infer domain from workflow content."""
        # Check workflow metadata
        description = (workflow_info.get("description", "") + " " + 
                      workflow_info.get("name", "")).lower()
        
        # Check node functions
        node_functions = [node.get("function", "").lower() for node in nodes]
        all_functions = " ".join(node_functions)
        
        # Domain detection patterns
        if any(word in description + all_functions for word in 
               ["video", "image", "detection", "yolo", "opencv", "directml", "openvino"]):
            return "computer_vision"
        elif any(word in description + all_functions for word in
                ["web", "html", "url", "scrape", "fetch", "parse", "beautifulsoup"]):
            return "web_scraping"
        elif any(word in description + all_functions for word in
                ["text", "nlp", "summarize", "translate", "sentiment"]):
            return "text_processing"
        elif any(word in description + all_functions for word in
                ["data", "csv", "analytics", "database", "visualization"]):
            return "data_processing"
        else:
            return "general"
    
    def _extract_patterns_from_nodes(self, nodes: List[Dict]) -> List[str]:
        """Extract key patterns from node structure."""
        patterns = []
        
        # Common workflow patterns
        node_count = len(nodes)
        if node_count > 10:
            patterns.append("granular_atomic_approach")
        elif node_count <= 5:
            patterns.append("optimized_approach")
        else:
            patterns.append("balanced_approach")
        
        # Dependency patterns
        has_dependencies = any(node.get("dependencies") for node in nodes)
        if has_dependencies:
            patterns.append("sequential_processing")
        
        # Parallel processing patterns
        parallel_nodes = [n for n in nodes if n.get("dependencies") and len(n["dependencies"]) > 1]
        if len(parallel_nodes) > 2:
            patterns.append("parallel_processing")
        
        # Hardware detection pattern
        if any("detect" in node.get("function", "").lower() and "gpu" in node.get("function", "").lower() 
               for node in nodes):
            patterns.append("hardware_detection")
        
        # Performance monitoring pattern
        if any("performance" in node.get("function", "").lower() or "stats" in node.get("function", "").lower()
               for node in nodes):
            patterns.append("performance_monitoring")
        
        return patterns
    
    def _generate_example_usage(self, workflow_info: Dict, nodes: List[Dict]) -> str:
        """Generate example usage description."""
        name = workflow_info.get("name", "Unnamed Workflow")
        description = workflow_info.get("description", "")
        
        # Extract key functionality from nodes
        key_functions = []
        for node in nodes[:3]:  # First few nodes usually show main purpose
            func_name = node.get("function", "").split(".")[-1].replace("_node", "")
            if func_name:
                key_functions.append(func_name.replace("_", " "))
        
        usage = f"'{name}' - {description}"
        if key_functions:
            usage += f" (Key steps: {', '.join(key_functions)})"
            
        return usage
    
    def _extract_node_patterns(self):
        """Extract common node usage patterns across all templates."""
        for template in self.templates.values():
            for node in template.workflow_json.get("nodes", []):
                func = node.get("function", "")
                if func:
                    category = func.split(".")[1] if "." in func else "custom"
                    if category not in self.node_patterns:
                        self.node_patterns[category] = []
                    
                    pattern = {
                        "function": func,
                        "typical_inputs": list(node.get("inputs", {}).keys()),
                        "typical_dependencies": node.get("dependencies", [])
                    }
                    self.node_patterns[category].append(str(pattern))
    
    def get_relevant_templates(self, description: str, max_templates: int = 3) -> List[WorkflowTemplate]:
        """Get templates most relevant to a description."""
        scored_templates = []
        
        desc_lower = description.lower()
        
        for template in self.templates.values():
            score = 0
            
            # Domain matching
            if template.domain != "general":
                domain_keywords = {
                    "computer_vision": ["video", "image", "detect", "camera", "yolo", "opencv"],
                    "web_scraping": ["web", "url", "scrape", "html", "fetch", "page"],
                    "text_processing": ["text", "nlp", "summarize", "translate", "analyze"],
                    "data_processing": ["data", "csv", "analytics", "database", "chart"]
                }
                
                keywords = domain_keywords.get(template.domain, [])
                domain_matches = sum(1 for keyword in keywords if keyword in desc_lower)
                score += domain_matches * 3  # Weight domain matches highly
            
            # Pattern matching
            pattern_matches = sum(1 for pattern in template.key_patterns 
                                if any(word in desc_lower for word in pattern.split("_")))
            score += pattern_matches
            
            # Description similarity (simple keyword matching)
            template_desc = template.description.lower()
            common_words = set(desc_lower.split()) & set(template_desc.split())
            score += len(common_words)
            
            scored_templates.append((score, template))
        
        # Sort by score and return top templates
        scored_templates.sort(key=lambda x: x[0], reverse=True)
        return [template for _, template in scored_templates[:max_templates]]


class TemplateBasedWorkflowGenerator:
    """
    Generates workflows using in-context learning from templates.
    
    Much simpler than domain-specific composers - just shows the LLM
    examples of good workflows and lets it learn the patterns.
    """
    
    def __init__(self, model_name: str = "qwen2.5-coder:7b"):
        self.model_name = model_name
        self.template_library = WorkflowTemplateLibrary()
        self.template_library.discover_templates()
        self.logger = logging.getLogger('workflow.template_generator')
        
        # Create Ollama client with proper model info
        ollama_model_info = ModelInfo(
            vision=False,
            function_calling=False, 
            json_output=True,
            family="qwen",
            structured_output=False
        )
        
        self._client = OpenAIChatCompletionClient(
            model=model_name,
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            model_info=ollama_model_info
        )
    
    async def generate_workflow(self, description: str) -> Dict[str, Any]:
        """
        Generate workflow using in-context learning from templates.
        
        Args:
            description: Natural language description of desired workflow
            
        Returns:
            Generated workflow JSON
        """
        self.logger.info(f"Generating workflow from: {description}")
        
        # Get relevant templates for in-context learning
        relevant_templates = self.template_library.get_relevant_templates(description)
        
        if not relevant_templates:
            self.logger.warning("No relevant templates found, using generic approach")
            relevant_templates = list(self.template_library.templates.values())[:2]
        
        # Build prompt with templates as examples
        prompt = self._build_in_context_prompt(description, relevant_templates)
        
        # Generate workflow using LLM
        agent = AssistantAgent(
            name="workflow_generator",
            model_client=self._client,
            system_message=self._get_system_message()
        )
        
        response = await agent.run(task=prompt)
        
        # Extract workflow JSON from response
        workflow_json = self._extract_workflow_from_response(response.messages[-1].content)
        
        # Add metadata
        workflow_json["workflow"]["generated_by"] = "TemplateBasedGenerator"
        workflow_json["workflow"]["generation_method"] = "in_context_learning"
        workflow_json["workflow"]["templates_used"] = [t.name for t in relevant_templates]
        workflow_json["workflow"]["original_request"] = description
        
        self.logger.info(f"Generated workflow with {len(workflow_json.get('nodes', []))} nodes")
        return workflow_json
    
    def _get_system_message(self) -> str:
        """Get system message that teaches the LLM about workflow patterns."""
        return """You are a Workflow Engine Expert specializing in creating granular, atomic workflows.

WORKFLOW ENGINE ARCHITECTURE:
- Workflows are JSON files with "workflow" metadata and "nodes" array
- Each node has: id, function, inputs, dependencies
- Nodes use @workflow_node decorator and follow atomic design principles
- Dependencies create execution order: ["node_a", "node_b"]
- Inputs can reference other nodes: "$node_id.output_name" or "$node_id" 

NODE DESIGN PRINCIPLES:
1. ATOMIC: Each node does ONE thing well (single responsibility)
2. COMPOSABLE: Nodes can be chained and reused
3. PURE FUNCTIONS: No side effects, predictable outputs
4. GRANULAR: Break complex operations into small steps for flexibility

AVAILABLE NODE CATEGORIES:
- src.nodes.atomic.*: Basic operations (read, transform, save)
- src.nodes.custom.*: Domain-specific operations  
- src.nodes.infrastructure.*: Hardware detection, model management
- src.nodes.utils.*: Performance monitoring, statistics

WORKFLOW STRATEGIES:
- GRANULAR: Maximum atomic nodes (high flexibility, ~15-20 nodes)
- BALANCED: Mix of atomic and composite nodes (~8-12 nodes) 
- OPTIMIZED: Fewer, more capable nodes (high performance, ~5-7 nodes)

Your task is to analyze the examples provided and create similar workflows that follow these patterns."""
    
    def _build_in_context_prompt(self, description: str, templates: List[WorkflowTemplate]) -> str:
        """Build prompt with relevant templates as examples."""
        
        prompt = f"""Create a workflow for: "{description}"

Learn from these example workflows and follow the same patterns:

"""
        
        # Add template examples
        for i, template in enumerate(templates, 1):
            prompt += f"""
EXAMPLE {i}: {template.name}
Description: {template.description}
Domain: {template.domain}
Key Patterns: {', '.join(template.key_patterns)}

Workflow JSON:
```json
{json.dumps(template.workflow_json, indent=2)}
```

---
"""
        
        # Add node patterns guidance
        node_categories = list(self.template_library.node_patterns.keys())[:5]
        prompt += f"""
AVAILABLE NODE CATEGORIES: {', '.join(node_categories)}

INSTRUCTIONS:
1. Study the examples above and identify the workflow patterns
2. Create a similar workflow for the requested task: "{description}"
3. Follow the same JSON structure and node patterns
4. Use appropriate granularity (atomic nodes for flexibility)
5. Include proper dependencies between nodes
6. Reference node outputs using $node_id.output_name syntax
7. Choose appropriate function paths based on the domain

Generate ONLY the workflow JSON, no explanations:
"""
        
        return prompt
    
    def _extract_workflow_from_response(self, response: str) -> Dict[str, Any]:
        """Extract workflow JSON from LLM response."""
        import re
        
        # Try to find JSON in the response
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON without code blocks
        json_match = re.search(r'(\{[\s\S]*"workflow"[\s\S]*\})', response)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Fallback: create basic structure
        self.logger.warning("Could not extract JSON from response, creating fallback")
        return {
            "workflow": {
                "name": "Generated Workflow",
                "description": "Auto-generated workflow",
                "error": "Could not parse LLM response"
            },
            "nodes": []
        }
    
    def generate_workflow_sync(self, description: str) -> Dict[str, Any]:
        """Synchronous wrapper for generate_workflow."""
        return asyncio.run(self.generate_workflow(description))


# ============================================================================
# Simple CLI Interface
# ============================================================================

async def main():
    """Test the template-based approach."""
    print("🤖 Template-Based Workflow Generator (In-Context Learning)")
    print("=" * 70)
    
    generator = TemplateBasedWorkflowGenerator()
    
    print(f"📚 Templates discovered: {len(generator.template_library.templates)}")
    
    # List available templates
    print("\n📋 Available Templates:")
    for name, template in generator.template_library.templates.items():
        print(f"   • {name} ({template.domain}) - {template.description[:50]}...")
    
    # Test workflow generation
    test_cases = [
        "Create a granular workflow to fetch web pages and summarize their content",
        "Process video for object detection with high performance", 
        "Analyze CSV data and create visualization dashboard"
    ]
    
    print(f"\n🧪 Testing Workflow Generation:")
    
    for description in test_cases:
        print(f"\n📝 Request: {description}")
        
        try:
            workflow = await generator.generate_workflow(description)
            
            print(f"✅ Generated: {workflow['workflow']['name']}")
            print(f"   Nodes: {len(workflow.get('nodes', []))}")
            print(f"   Templates used: {workflow['workflow'].get('templates_used', [])}")
            
            # Save workflow
            filename = f"generated_{description.lower().replace(' ', '_')[:30]}.json"
            output_path = Path("workflows") / filename
            output_path.parent.mkdir(exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(workflow, f, indent=2)
            
            print(f"   Saved to: {output_path}")
            
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    asyncio.run(main())