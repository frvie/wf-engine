"""
Base Composer Architecture for Domain-Agnostic Workflow Generation

This module provides the foundation for creating specialized workflow composers
for any domain (web scraping, text processing, data analysis, etc.) while
maintaining consistency and reusability.

Key Principles:
1. Domain Discovery - Automatically detect available domains
2. Node Classification - Categorize nodes by domain and capability
3. Pattern Recognition - Learn workflow patterns from templates
4. Goal Abstraction - Universal goal representation across domains
5. Strategy Selection - Choose optimal approach per domain
"""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import inspect
import importlib
from enum import Enum


# ============================================================================
# Universal Data Models
# ============================================================================

class WorkflowDomain(Enum):
    """Supported workflow domains."""
    COMPUTER_VISION = "computer_vision"
    WEB_CONTENT = "web_content"
    TEXT_PROCESSING = "text_processing" 
    DATA_ANALYSIS = "data_analysis"
    ML_INFERENCE = "ml_inference"
    FILE_PROCESSING = "file_processing"
    API_INTEGRATION = "api_integration"
    UNKNOWN = "unknown"


class WorkflowStrategy(Enum):
    """Universal workflow strategies."""
    ATOMIC = "atomic"          # Maximum flexibility, composable atomic nodes
    OPTIMIZED = "optimized"    # Performance-focused, fewer nodes
    BALANCED = "balanced"      # Trade-off between flexibility and performance
    STREAMING = "streaming"    # For real-time/streaming data
    BATCH = "batch"           # For bulk processing


@dataclass
class UniversalGoal:
    """Domain-agnostic goal representation."""
    # Core identification
    domain: WorkflowDomain
    task: str                  # Domain-specific task (e.g., "object_detection", "web_scraping")
    
    # Input/Output specification
    input_type: str           # "video.mp4", "https://url", "text", "csv", etc.
    output_type: str          # "display", "file", "api", "summary", etc.
    
    # Performance requirements
    performance_target: Optional[float] = None    # Domain-specific metric
    quality_over_speed: bool = False
    
    # Resource constraints
    hardware_preference: Optional[str] = "auto"  # "gpu", "cpu", "cloud", etc.
    memory_limit: Optional[str] = None           # "1GB", "unlimited"
    timeout: Optional[int] = None                # seconds
    
    # Strategy preferences
    strategy: Optional[WorkflowStrategy] = None
    parallel_processing: bool = True
    
    # Additional constraints
    requirements: Dict[str, Any] = None          # Domain-specific requirements


@dataclass
class NodeCapability:
    """Describes what a node can do."""
    domain: WorkflowDomain
    function_type: str        # "fetch", "parse", "transform", "analyze", etc.
    input_types: List[str]    # What types of data it accepts
    output_types: List[str]   # What types of data it produces
    dependencies: List[str]   # Required packages/services
    hardware_requirements: List[str]  # "gpu", "internet", etc.
    performance_profile: Dict[str, float]  # Expected latency, throughput, etc.


# ============================================================================
# Base Composer Framework
# ============================================================================

class BaseComposer(ABC):
    """
    Abstract base class for domain-specific workflow composers.
    
    Provides common infrastructure while allowing domain specialization.
    """
    
    def __init__(self, nodes_dir: Path = None):
        self.logger = logging.getLogger(f'composer.{self.__class__.__name__.lower()}')
        self.nodes_dir = nodes_dir or Path(__file__).parent.parent / "nodes"
        self.domain_nodes: Dict[str, NodeCapability] = {}
        self._nodes_discovered = False
    
    @property
    @abstractmethod
    def supported_domain(self) -> WorkflowDomain:
        """Return the domain this composer handles."""
        pass
    
    @property
    @abstractmethod
    def supported_tasks(self) -> List[str]:
        """Return the tasks this composer can handle."""
        pass
    
    def discover_nodes(self) -> Dict[str, NodeCapability]:
        """
        Discover and classify nodes for this domain.
        
        Returns:
            Dictionary of node_id -> NodeCapability
        """
        if self._nodes_discovered:
            return self.domain_nodes
            
        discovered = {}
        
        # Scan all Python files in nodes directory
        for py_file in self.nodes_dir.rglob("*.py"):
            if py_file.name.startswith('__'):
                continue
                
            try:
                # Convert path to module name
                rel_path = py_file.relative_to(self.nodes_dir.parent)
                module_name = str(rel_path.with_suffix('')).replace('/', '.').replace('\\', '.')
                
                # Import and inspect module
                module = importlib.import_module(module_name)
                
                for name, obj in inspect.getmembers(module, inspect.isfunction):
                    if hasattr(obj, '__wrapped__'):  # Has @workflow_node decorator
                        capability = self._analyze_node_capability(obj, module_name, name)
                        if capability and capability.domain == self.supported_domain:
                            node_id = getattr(obj, 'node_name', name.replace('_node', ''))
                            discovered[node_id] = capability
                            
            except Exception as e:
                self.logger.debug(f"Failed to analyze {py_file}: {e}")
        
        self.domain_nodes = discovered
        self._nodes_discovered = True
        self.logger.info(f"Discovered {len(discovered)} {self.supported_domain.value} nodes")
        return discovered
    
    @abstractmethod
    def _analyze_node_capability(self, func, module_name: str, func_name: str) -> Optional[NodeCapability]:
        """
        Analyze a function to determine its capability.
        
        Args:
            func: Function object with @workflow_node decorator
            module_name: Module containing the function
            func_name: Name of the function
            
        Returns:
            NodeCapability if this node belongs to our domain, None otherwise
        """
        pass
    
    @abstractmethod
    def compose_from_goal(self, goal: UniversalGoal) -> Dict[str, Any]:
        """
        Compose a workflow from a universal goal.
        
        Args:
            goal: Universal goal specification
            
        Returns:
            Complete workflow JSON
        """
        pass
    
    def _create_workflow_metadata(self, goal: UniversalGoal, nodes: List[Dict]) -> Dict[str, Any]:
        """Create standardized workflow metadata."""
        return {
            "name": f"Auto-Generated: {goal.task}",
            "description": f"Autonomous {goal.domain.value} workflow for {goal.task}",
            "version": "1.0",
            "domain": goal.domain.value,
            "strategy": goal.strategy.value if goal.strategy else "auto",
            "generated_by": self.__class__.__name__,
            "generated_at": datetime.now().isoformat(),
            "goal": asdict(goal),
            "node_count": len(nodes),
            "estimated_performance": self._estimate_performance(goal, nodes)
        }
    
    @abstractmethod
    def _estimate_performance(self, goal: UniversalGoal, nodes: List[Dict]) -> Dict[str, Any]:
        """Estimate workflow performance based on goal and nodes."""
        pass


# ============================================================================
# Universal Composer Registry
# ============================================================================

class ComposerRegistry:
    """
    Registry that automatically discovers and manages domain-specific composers.
    
    This enables the system to handle any workflow domain without
    hardcoding specific composer types.
    """
    
    def __init__(self, composers_dir: Path = None):
        self.logger = logging.getLogger('composer.registry')
        self.composers_dir = composers_dir or Path(__file__).parent / "composers"
        self.composers: Dict[WorkflowDomain, BaseComposer] = {}
        self._discover_composers()
    
    def _discover_composers(self):
        """Automatically discover all available composers."""
        # Import known composers
        from .computer_vision_composer import ComputerVisionComposer
        
        # Register discovered composers
        cv_composer = ComputerVisionComposer()
        self.composers[cv_composer.supported_domain] = cv_composer
        
        self.logger.info(f"Registered {len(self.composers)} composers")
    
    def get_composer(self, domain: WorkflowDomain) -> Optional[BaseComposer]:
        """Get composer for a specific domain."""
        return self.composers.get(domain)
    
    def compose_workflow(self, goal: UniversalGoal) -> Dict[str, Any]:
        """
        Compose workflow using appropriate domain composer.
        
        Args:
            goal: Universal goal specification
            
        Returns:
            Complete workflow JSON
        """
        composer = self.get_composer(goal.domain)
        if not composer:
            raise ValueError(f"No composer available for domain: {goal.domain}")
        
        return composer.compose_from_goal(goal)
    
    def list_capabilities(self) -> Dict[WorkflowDomain, Dict[str, Any]]:
        """List all available capabilities across domains."""
        capabilities = {}
        
        for domain, composer in self.composers.items():
            capabilities[domain] = {
                "tasks": composer.supported_tasks,
                "node_count": len(composer.discover_nodes()),
                "composer": composer.__class__.__name__
            }
        
        return capabilities


# ============================================================================
# Domain Detection & Goal Parsing
# ============================================================================

class DomainDetector:
    """
    Automatically detects workflow domain from natural language descriptions.
    
    Uses keyword matching and pattern recognition to classify requests
    into appropriate domains.
    """
    
    def __init__(self):
        self.domain_keywords = {
            WorkflowDomain.COMPUTER_VISION: [
                "detect", "detection", "video", "image", "camera", "webcam", 
                "object", "face", "recognition", "tracking", "yolo", "opencv"
            ],
            WorkflowDomain.WEB_CONTENT: [
                "web", "website", "url", "scrape", "scraping", "html", "http",
                "crawl", "browser", "page", "content", "extract", "download"
            ],
            WorkflowDomain.TEXT_PROCESSING: [
                "text", "nlp", "summarize", "translate", "sentiment", "language",
                "parse", "analyze", "classify", "generate", "llm", "ai"
            ],
            WorkflowDomain.DATA_ANALYSIS: [
                "data", "csv", "excel", "database", "analytics", "statistics",
                "visualization", "chart", "graph", "metrics", "insights"
            ],
            WorkflowDomain.ML_INFERENCE: [
                "model", "inference", "prediction", "classification", "regression",
                "neural", "tensorflow", "pytorch", "onnx", "ml", "ai"
            ],
            WorkflowDomain.FILE_PROCESSING: [
                "file", "document", "pdf", "word", "convert", "transform",
                "batch", "folder", "directory", "format", "compress"
            ]
        }
    
    def detect_domain(self, description: str) -> Tuple[WorkflowDomain, float]:
        """
        Detect domain from natural language description.
        
        Args:
            description: Natural language description
            
        Returns:
            (detected_domain, confidence_score)
        """
        desc_lower = description.lower()
        scores = {}
        
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in desc_lower)
            if score > 0:
                scores[domain] = score / len(keywords)  # Normalize by keyword count
        
        if not scores:
            return WorkflowDomain.UNKNOWN, 0.0
        
        best_domain = max(scores.items(), key=lambda x: x[1])
        return best_domain[0], best_domain[1]
    
    def parse_natural_language_to_goal(self, description: str) -> UniversalGoal:
        """
        Parse natural language into a UniversalGoal.
        
        Args:
            description: Natural language description
            
        Returns:
            Parsed UniversalGoal
        """
        # Detect domain first
        domain, confidence = self.detect_domain(description)
        
        if confidence < 0.1:
            raise ValueError(f"Could not determine domain from: {description}")
        
        desc_lower = description.lower()
        
        # Extract task (domain-specific)
        task = self._extract_task(desc_lower, domain)
        
        # Extract input/output types
        input_type = self._extract_input_type(desc_lower, domain)
        output_type = self._extract_output_type(desc_lower, domain)
        
        # Extract performance preferences
        quality_over_speed = any(word in desc_lower for word in 
                               ["quality", "accurate", "detailed", "thorough", "atomic"])
        
        performance_target = self._extract_performance_target(desc_lower, domain)
        
        # Extract strategy preference
        strategy = None
        if "atomic" in desc_lower or "granular" in desc_lower or "flexible" in desc_lower:
            strategy = WorkflowStrategy.ATOMIC
        elif "fast" in desc_lower or "quick" in desc_lower or "optimized" in desc_lower:
            strategy = WorkflowStrategy.OPTIMIZED
        elif "streaming" in desc_lower or "real-time" in desc_lower:
            strategy = WorkflowStrategy.STREAMING
        elif "batch" in desc_lower or "bulk" in desc_lower:
            strategy = WorkflowStrategy.BATCH
        
        return UniversalGoal(
            domain=domain,
            task=task,
            input_type=input_type,
            output_type=output_type,
            performance_target=performance_target,
            quality_over_speed=quality_over_speed,
            strategy=strategy,
            requirements={"original_description": description, "confidence": confidence}
        )
    
    def _extract_task(self, description: str, domain: WorkflowDomain) -> str:
        """Extract specific task based on domain."""
        if domain == WorkflowDomain.COMPUTER_VISION:
            if "detect" in description or "detection" in description:
                return "object_detection"
            elif "track" in description:
                return "object_tracking"
            elif "recognize" in description or "recognition" in description:
                return "recognition"
        elif domain == WorkflowDomain.WEB_CONTENT:
            if "scrape" in description or "extract" in description:
                return "web_scraping"
            elif "summarize" in description or "summary" in description:
                return "content_summarization"
            elif "monitor" in description:
                return "content_monitoring"
        elif domain == WorkflowDomain.TEXT_PROCESSING:
            if "summarize" in description:
                return "text_summarization"
            elif "translate" in description:
                return "translation"
            elif "sentiment" in description:
                return "sentiment_analysis"
        
        return f"{domain.value}_processing"
    
    def _extract_input_type(self, description: str, domain: WorkflowDomain) -> str:
        """Extract input type based on description and domain."""
        if "webcam" in description or "camera" in description:
            return "webcam"
        elif "video" in description or any(ext in description for ext in [".mp4", ".avi", ".mov"]):
            return "video"
        elif "image" in description or any(ext in description for ext in [".jpg", ".png", ".jpeg"]):
            return "image"
        elif "url" in description or "http" in description or "website" in description:
            return "url"
        elif "file" in description or "document" in description:
            return "file"
        elif "text" in description:
            return "text"
        
        return "unknown"
    
    def _extract_output_type(self, description: str, domain: WorkflowDomain) -> str:
        """Extract output type based on description."""
        if "display" in description or "show" in description or "view" in description:
            return "display"
        elif "save" in description or "file" in description or "export" in description:
            return "file"
        elif "api" in description or "service" in description:
            return "api"
        elif "summary" in description or "report" in description:
            return "summary"
        
        return "display"
    
    def _extract_performance_target(self, description: str, domain: WorkflowDomain) -> Optional[float]:
        """Extract performance target based on domain."""
        import re
        
        # Look for explicit numbers with units
        if domain == WorkflowDomain.COMPUTER_VISION:
            fps_match = re.search(r'(\d+)\s*fps', description)
            if fps_match:
                return float(fps_match.group(1))
            elif "fast" in description or "real-time" in description:
                return 25.0
            elif "quality" in description:
                return 15.0
        elif domain == WorkflowDomain.WEB_CONTENT:
            if "fast" in description:
                return 5.0  # pages per second
            elif "thorough" in description:
                return 1.0
        
        return None


# ============================================================================
# Universal Workflow Generator
# ============================================================================

class UniversalWorkflowGenerator:
    """
    Main interface for generating workflows in any domain from natural language.
    
    This is the primary entry point that users interact with.
    """
    
    def __init__(self):
        self.logger = logging.getLogger('workflow.universal')
        self.registry = ComposerRegistry()
        self.detector = DomainDetector()
        self.logger.info("Universal Workflow Generator initialized")
    
    def create_workflow(self, description: str) -> Dict[str, Any]:
        """
        Create a complete workflow from natural language description.
        
        Args:
            description: Natural language description of what to do
            
        Returns:
            Complete workflow JSON with all required nodes
            
        Examples:
            "Extract and summarize content from web pages using atomic nodes"
            "Detect objects in video with high performance on GPU"
            "Process CSV data and generate analytics dashboard"
        """
        self.logger.info(f"Creating workflow from: {description}")
        
        # Parse natural language to goal
        goal = self.detector.parse_natural_language_to_goal(description)
        self.logger.info(f"Detected domain: {goal.domain.value} (task: {goal.task})")
        
        # Generate workflow using appropriate composer
        workflow = self.registry.compose_workflow(goal)
        
        self.logger.info(f"Generated {goal.domain.value} workflow with {len(workflow.get('nodes', []))} nodes")
        return workflow
    
    def list_capabilities(self) -> Dict[str, Any]:
        """List all available workflow capabilities."""
        return {
            "domains": [domain.value for domain in WorkflowDomain],
            "composers": self.registry.list_capabilities(),
            "total_nodes": sum(
                cap["node_count"] for cap in self.registry.list_capabilities().values()
            )
        }


if __name__ == "__main__":
    # Quick test
    generator = UniversalWorkflowGenerator()
    
    print("🌐 Universal Workflow Generator")
    print("=" * 70)
    
    capabilities = generator.list_capabilities()
    print(f"Available domains: {len(capabilities['domains'])}")
    print(f"Total nodes: {capabilities['total_nodes']}")
    
    # Test domain detection
    test_descriptions = [
        "Extract and summarize content from web pages using atomic nodes",
        "Detect objects in video with high performance",
        "Process CSV data and generate analytics"
    ]
    
    detector = DomainDetector()
    for desc in test_descriptions:
        domain, confidence = detector.detect_domain(desc)
        print(f"{desc[:50]}... -> {domain.value} ({confidence:.2f})")