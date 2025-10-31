"""
Web Content Composer - Specialized composer for web scraping and content processing workflows.

This composer demonstrates how to extend the base framework for a specific domain.
It can generate granular workflows for:
- Web scraping and content extraction
- HTML parsing and text processing  
- Content summarization and analysis
- Batch URL processing
- Content monitoring and change detection
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import inspect
import re

from .base_composer import (
    BaseComposer, 
    WorkflowDomain, 
    WorkflowStrategy,
    UniversalGoal, 
    NodeCapability
)


class WebContentComposer(BaseComposer):
    """
    Specialized composer for web content processing workflows.
    
    Handles tasks like web scraping, content extraction, summarization,
    and text analysis using atomic, composable nodes.
    """
    
    @property
    def supported_domain(self) -> WorkflowDomain:
        """This composer handles web content processing."""
        return WorkflowDomain.WEB_CONTENT
    
    @property  
    def supported_tasks(self) -> List[str]:
        """Tasks this composer can handle."""
        return [
            "web_scraping",
            "content_extraction", 
            "content_summarization",
            "url_processing",
            "content_monitoring",
            "html_parsing",
            "text_analysis"
        ]
    
    def _analyze_node_capability(self, func, module_name: str, func_name: str) -> Optional[NodeCapability]:
        """
        Analyze a function to determine if it's a web content node.
        
        Looks for web-related keywords in function names, docstrings, and imports.
        """
        # Check function name for web-related keywords
        web_keywords = [
            'fetch', 'scrape', 'html', 'parse', 'url', 'web', 'http', 'request',
            'beautifulsoup', 'content', 'extract', 'download', 'browser', 'crawl'
        ]
        
        func_name_lower = func_name.lower()
        if not any(keyword in func_name_lower for keyword in web_keywords):
            return None
        
        # Analyze function signature and docstring
        sig = inspect.signature(func)
        doc = func.__doc__ or ""
        
        # Determine function type based on name and documentation
        function_type = "unknown"
        if any(word in func_name_lower for word in ['fetch', 'download', 'get', 'request']):
            function_type = "fetch"
        elif any(word in func_name_lower for word in ['parse', 'extract', 'beautifulsoup']):
            function_type = "parse"
        elif any(word in func_name_lower for word in ['clean', 'normalize', 'format']):
            function_type = "transform"
        elif any(word in func_name_lower for word in ['summarize', 'analyze', 'process']):
            function_type = "analyze"
        elif any(word in func_name_lower for word in ['save', 'store', 'write']):
            function_type = "store"
        
        # Determine input/output types from signature and doc
        input_types = ["url", "html", "text"]  # Default assumptions
        output_types = ["html", "text", "data"]
        
        if "url" in func_name_lower or "fetch" in func_name_lower:
            input_types = ["url"]
            output_types = ["html", "text"]
        elif "parse" in func_name_lower:
            input_types = ["html"]
            output_types = ["text", "data"]
        elif "summarize" in func_name_lower:
            input_types = ["text"]
            output_types = ["summary", "data"]
        
        # Check for dependencies in docstring/imports
        dependencies = []
        if "requests" in doc.lower() or "requests" in func_name_lower:
            dependencies.append("requests")
        if "beautifulsoup" in doc.lower() or "bs4" in doc.lower():
            dependencies.append("beautifulsoup4")
        if "selenium" in doc.lower():
            dependencies.append("selenium")
        
        # Hardware requirements
        hardware_requirements = []
        if "selenium" in dependencies or "browser" in doc.lower():
            hardware_requirements.append("browser")
        if "http" in doc.lower() or "url" in doc.lower():
            hardware_requirements.append("internet")
        
        return NodeCapability(
            domain=self.supported_domain,
            function_type=function_type,
            input_types=input_types,
            output_types=output_types,
            dependencies=dependencies,
            hardware_requirements=hardware_requirements,
            performance_profile={"latency": 2.0, "throughput": 1.0}  # Default estimates
        )
    
    def compose_from_goal(self, goal: UniversalGoal) -> Dict[str, Any]:
        """
        Compose web content workflow from goal.
        
        Creates atomic workflows for web scraping, parsing, and analysis.
        """
        if goal.domain != self.supported_domain:
            raise ValueError(f"Goal domain {goal.domain} not supported by WebContentComposer")
        
        self.discover_nodes()  # Ensure nodes are discovered
        
        # Select workflow composition strategy
        if goal.task == "web_scraping":
            return self._compose_web_scraping_workflow(goal)
        elif goal.task == "content_summarization":
            return self._compose_content_summarization_workflow(goal)
        elif goal.task == "url_processing":
            return self._compose_batch_url_workflow(goal)
        else:
            # Generic web content processing
            return self._compose_generic_web_workflow(goal)
    
    def _compose_web_scraping_workflow(self, goal: UniversalGoal) -> Dict[str, Any]:
        """Compose workflow for web scraping."""
        nodes = []
        
        # 1. URL Validation (if processing single URL)
        if goal.input_type.startswith('http'):
            nodes.append({
                "id": "validate_url",
                "function": "src.nodes.custom.validate_url.validate_url_node",
                "dependencies": [],
                "inputs": {
                    "url": goal.input_type
                }
            })
            url_ref = "$validate_url.validated_url"
            url_deps = ["validate_url"]
        else:
            # Input is URL parameter
            url_ref = goal.input_type
            url_deps = []
        
        # 2. Fetch web content
        nodes.append({
            "id": "fetch_content",
            "function": "src.nodes.custom.fetch_web_page.fetch_web_page",
            "dependencies": url_deps,
            "inputs": {
                "input": url_ref
            }
        })
        
        # 3. Parse HTML content
        nodes.append({
            "id": "parse_html", 
            "function": "src.nodes.custom.parse_html_to_text.parse_html_to_text",
            "dependencies": ["fetch_content"],
            "inputs": {
                "input": "$fetch_content"
            }
        })
        
        # Strategy-dependent processing
        if goal.strategy == WorkflowStrategy.ATOMIC:
            # Maximum granularity - separate cleaning, validation, formatting
            nodes.extend(self._add_atomic_text_processing_nodes())
        else:
            # Combined processing for performance
            nodes.append({
                "id": "clean_text",
                "function": "src.nodes.custom.clean_and_format_text.clean_and_format_text_node", 
                "dependencies": ["parse_html"],
                "inputs": {
                    "input": "$parse_html",
                    "remove_extra_whitespace": True,
                    "normalize_encoding": True
                }
            })
        
        # Output handling based on goal
        if goal.output_type == "file":
            nodes.append({
                "id": "save_content",
                "function": "src.nodes.custom.save_text_content.save_text_content_node",
                "dependencies": ["clean_text"] if goal.strategy != WorkflowStrategy.ATOMIC else ["format_text"],
                "inputs": {
                    "content": "$clean_text" if goal.strategy != WorkflowStrategy.ATOMIC else "$format_text",
                    "output_path": "output/scraped_content.txt"
                }
            })
        elif goal.output_type == "summary":
            # Add summarization step
            nodes.append({
                "id": "create_summary",
                "function": "src.nodes.custom.create_content_summary.create_content_summary_node",
                "dependencies": ["clean_text"] if goal.strategy != WorkflowStrategy.ATOMIC else ["format_text"],
                "inputs": {
                    "content": "$clean_text" if goal.strategy != WorkflowStrategy.ATOMIC else "$format_text",
                    "max_length": 500,
                    "include_metadata": True
                }
            })
        
        # Create workflow metadata
        workflow = {
            "workflow": self._create_workflow_metadata(goal, nodes),
            "nodes": nodes
        }
        
        return workflow
    
    def _compose_content_summarization_workflow(self, goal: UniversalGoal) -> Dict[str, Any]:
        """Compose workflow specifically for content summarization."""
        nodes = []
        
        # 1. Fetch content (same as scraping)
        nodes.extend([
            {
                "id": "fetch_content",
                "function": "src.nodes.custom.fetch_web_page.fetch_web_page",
                "dependencies": [],
                "inputs": {"input": goal.input_type}
            },
            {
                "id": "parse_html",
                "function": "src.nodes.custom.parse_html_to_text.parse_html_to_text", 
                "dependencies": ["fetch_content"],
                "inputs": {"input": "$fetch_content"}
            }
        ])
        
        # 2. Advanced text processing for summarization
        if goal.strategy == WorkflowStrategy.ATOMIC:
            # Granular approach - multiple processing steps
            nodes.extend([
                {
                    "id": "clean_text",
                    "function": "src.nodes.custom.clean_text_content.clean_text_content_node",
                    "dependencies": ["parse_html"],
                    "inputs": {"input": "$parse_html"}
                },
                {
                    "id": "split_sections",
                    "function": "src.nodes.custom.split_text_sections.split_text_sections_node",
                    "dependencies": ["clean_text"], 
                    "inputs": {
                        "text": "$clean_text",
                        "section_size": 1000,
                        "overlap": 100
                    }
                },
                {
                    "id": "summarize_sections",
                    "function": "src.nodes.custom.summarize_text_sections.summarize_text_sections_node",
                    "dependencies": ["split_sections"],
                    "inputs": {
                        "sections": "$split_sections.sections",
                        "summary_length": 200
                    }
                },
                {
                    "id": "combine_summaries",
                    "function": "src.nodes.custom.combine_section_summaries.combine_section_summaries_node",
                    "dependencies": ["summarize_sections"],
                    "inputs": {
                        "section_summaries": "$summarize_sections.summaries",
                        "final_length": goal.requirements.get("summary_length", 500)
                    }
                }
            ])
        else:
            # Optimized approach - single summarization step
            nodes.append({
                "id": "summarize_content",
                "function": "src.nodes.custom.summarize_page.summarize_page",
                "dependencies": ["parse_html"],
                "inputs": {
                    "input": "$parse_html",
                    "max_length": goal.requirements.get("summary_length", 500)
                }
            })
        
        workflow = {
            "workflow": self._create_workflow_metadata(goal, nodes),
            "nodes": nodes
        }
        
        return workflow
    
    def _compose_batch_url_workflow(self, goal: UniversalGoal) -> Dict[str, Any]:
        """Compose workflow for processing multiple URLs."""
        nodes = []
        
        # 1. Load URL list
        nodes.append({
            "id": "load_urls",
            "function": "src.nodes.custom.load_url_list.load_url_list_node",
            "dependencies": [],
            "inputs": {
                "source": goal.input_type,  # Could be file path or list
                "validate": True
            }
        })
        
        # 2. Parallel processing setup (if enabled)
        if goal.parallel_processing:
            nodes.append({
                "id": "setup_parallel",
                "function": "src.nodes.custom.setup_parallel_processing.setup_parallel_processing_node",
                "dependencies": ["load_urls"],
                "inputs": {
                    "url_list": "$load_urls.urls",
                    "max_workers": 4,
                    "timeout": 30
                }
            })
            
            # 3. Parallel fetch and process
            nodes.append({
                "id": "process_urls_parallel",
                "function": "src.nodes.custom.process_urls_parallel.process_urls_parallel_node",
                "dependencies": ["setup_parallel"],
                "inputs": {
                    "worker_config": "$setup_parallel.config",
                    "processing_pipeline": ["fetch", "parse", "summarize"]
                }
            })
        else:
            # Sequential processing
            nodes.append({
                "id": "process_urls_sequential", 
                "function": "src.nodes.custom.process_urls_sequential.process_urls_sequential_node",
                "dependencies": ["load_urls"],
                "inputs": {
                    "url_list": "$load_urls.urls",
                    "processing_steps": ["fetch", "parse", "clean", "summarize"]
                }
            })
        
        # 4. Aggregate results
        process_ref = "$process_urls_parallel" if goal.parallel_processing else "$process_urls_sequential"
        nodes.append({
            "id": "aggregate_results",
            "function": "src.nodes.custom.aggregate_url_results.aggregate_url_results_node", 
            "dependencies": ["process_urls_parallel" if goal.parallel_processing else "process_urls_sequential"],
            "inputs": {
                "results": f"{process_ref}.results",
                "output_format": goal.output_type,
                "include_metadata": True
            }
        })
        
        workflow = {
            "workflow": self._create_workflow_metadata(goal, nodes),
            "nodes": nodes
        }
        
        return workflow
    
    def _compose_generic_web_workflow(self, goal: UniversalGoal) -> Dict[str, Any]:
        """Compose a generic web content processing workflow."""
        # Use web scraping as the base and add task-specific processing
        workflow = self._compose_web_scraping_workflow(goal)
        
        # Add task-specific nodes based on goal.task
        additional_nodes = []
        
        if "analysis" in goal.task:
            additional_nodes.append({
                "id": "analyze_content",
                "function": "src.nodes.custom.analyze_content.analyze_content_node",
                "dependencies": ["clean_text"],
                "inputs": {
                    "content": "$clean_text",
                    "analysis_type": "comprehensive"
                }
            })
        
        if "monitoring" in goal.task:
            additional_nodes.extend([
                {
                    "id": "store_baseline",
                    "function": "src.nodes.custom.store_content_baseline.store_content_baseline_node", 
                    "dependencies": ["clean_text"],
                    "inputs": {
                        "content": "$clean_text",
                        "url": goal.input_type,
                        "timestamp": True
                    }
                },
                {
                    "id": "detect_changes",
                    "function": "src.nodes.custom.detect_content_changes.detect_content_changes_node",
                    "dependencies": ["store_baseline"],
                    "inputs": {
                        "current_content": "$clean_text", 
                        "baseline_ref": "$store_baseline.baseline_id"
                    }
                }
            ])
        
        if additional_nodes:
            workflow["nodes"].extend(additional_nodes)
            workflow["workflow"]["node_count"] = len(workflow["nodes"])
        
        return workflow
    
    def _add_atomic_text_processing_nodes(self) -> List[Dict[str, Any]]:
        """Add granular atomic text processing nodes."""
        return [
            {
                "id": "remove_html_artifacts", 
                "function": "src.nodes.custom.remove_html_artifacts.remove_html_artifacts_node",
                "dependencies": ["parse_html"],
                "inputs": {"input": "$parse_html"}
            },
            {
                "id": "normalize_whitespace",
                "function": "src.nodes.custom.normalize_whitespace.normalize_whitespace_node", 
                "dependencies": ["remove_html_artifacts"],
                "inputs": {"input": "$remove_html_artifacts"}
            },
            {
                "id": "fix_encoding",
                "function": "src.nodes.custom.fix_text_encoding.fix_text_encoding_node",
                "dependencies": ["normalize_whitespace"], 
                "inputs": {"input": "$normalize_whitespace"}
            },
            {
                "id": "format_text",
                "function": "src.nodes.custom.format_final_text.format_final_text_node",
                "dependencies": ["fix_encoding"],
                "inputs": {
                    "input": "$fix_encoding",
                    "add_metadata": True
                }
            }
        ]
    
    def _estimate_performance(self, goal: UniversalGoal, nodes: List[Dict]) -> Dict[str, Any]:
        """Estimate performance for web content workflows."""
        # Base estimates for web operations
        base_fetch_time = 2.0  # seconds per URL
        base_parse_time = 0.1   # seconds per page
        base_summarize_time = 1.0  # seconds per page
        
        # Adjust based on strategy
        if goal.strategy == WorkflowStrategy.ATOMIC:
            processing_overhead = 1.5  # More nodes = more overhead
        else:
            processing_overhead = 1.0
        
        estimated_time_per_url = (base_fetch_time + base_parse_time + base_summarize_time) * processing_overhead
        
        # Parallel processing benefit
        if goal.parallel_processing and "batch" in goal.task:
            parallel_speedup = 3.0  # Estimate 3x speedup with parallel processing
            estimated_time_per_url /= parallel_speedup
        
        return {
            "estimated_time_per_url": estimated_time_per_url,
            "estimated_throughput": 1.0 / estimated_time_per_url,
            "memory_usage": "low" if goal.strategy != WorkflowStrategy.ATOMIC else "medium",
            "network_dependency": "high",
            "cpu_usage": "medium"
        }