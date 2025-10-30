"""
Agentic Workflow Engine

Combines autonomous capabilities:
1. Workflow Composition - Auto-generate workflows from goals
2. Self-Optimization - Tune parameters based on performance
3. Adaptive Pipeline Selection - Choose best strategy for requirements
5. Learning from Execution - Build knowledge base of what works

Architecture:
- WorkflowComposer: Generates workflows from high-level goals
- PerformanceOptimizer: Tracks metrics and tunes parameters
- PipelineSelector: Chooses optimal execution strategy
- ExecutionLearner: Builds performance profiles and suggests improvements
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict
import statistics


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class ExecutionRecord:
    """Record of a single workflow execution."""
    timestamp: str
    workflow_name: str
    workflow_type: str  # 'granular', 'monolithic', 'custom'
    nodes: List[str]
    parameters: Dict[str, Any]
    hardware: Dict[str, bool]  # Available devices
    performance: Dict[str, float]  # fps, latency, etc.
    success: bool
    error: Optional[str] = None


@dataclass
class PerformanceProfile:
    """Performance profile for a specific configuration."""
    node_sequence: str  # Hash of node sequence
    avg_fps: float
    avg_latency: float
    success_rate: float
    execution_count: int
    best_parameters: Dict[str, Any]
    hardware_used: Dict[str, bool]


@dataclass
class WorkflowGoal:
    """High-level goal for workflow composition."""
    task: str  # 'object_detection', 'video_processing', etc.
    input_type: str  # 'webcam', 'video_file', 'image'
    output_type: str  # 'display', 'file', 'api'
    performance_target: Optional[float] = None  # Target FPS
    hardware_preference: Optional[str] = None  # 'gpu', 'npu', 'cpu', 'auto'
    quality_over_speed: bool = False


# ============================================================================
# 1. Workflow Composer - Autonomous Workflow Generation
# ============================================================================

class WorkflowComposer:
    """Generates workflows from high-level goals."""
    
    def __init__(self, nodes_dir: Path = None):
        self.logger = logging.getLogger('workflow.composer')
        if nodes_dir is None:
            # Default to src/nodes relative to this file's parent directory
            nodes_dir = Path(__file__).parent.parent / "nodes"
        self.nodes_dir = nodes_dir
        self.logger.debug(f"Nodes directory: {self.nodes_dir.absolute()}")
        self.available_nodes = {}  # Lazy load - don't discover on init
        self._nodes_discovered = False
    
    def _ensure_nodes_discovered(self):
        """Discover nodes on first use (lazy loading)."""
        if not self._nodes_discovered:
            self.available_nodes = self._discover_nodes()
            self._nodes_discovered = True
    
    def _discover_nodes(self) -> Dict[str, Dict[str, Any]]:
        """Discover available nodes and their capabilities."""
        import importlib
        import inspect
        from pathlib import Path
        
        discovered = {}
        
        # Scan recursively for all Python files in workflow_nodes
        for py_file in self.nodes_dir.rglob("*.py"):
            if py_file.name == '__init__.py' or py_file.name.startswith('_'):
                continue
            
            try:
                # Convert path to module name for src.nodes.*
                relative_path = py_file.relative_to(self.nodes_dir)
                if relative_path.parts:
                    # Build module name: src.nodes.{file} or src.nodes.custom.{file}
                    module_parts = ['src', 'nodes'] + list(relative_path.parts[:-1]) + [relative_path.stem]
                    module_name = '.'.join(module_parts)
                else:
                    module_name = f'src.nodes.{py_file.stem}'
                
                self.logger.debug(f"Trying to import: {module_name}")
                module = importlib.import_module(module_name)
                
                # Look for functions with workflow_node decorator
                for name, obj in inspect.getmembers(module):
                    if inspect.isfunction(obj) and hasattr(obj, 'node_id'):
                        node_id = obj.node_id
                        
                        # Determine category and capabilities from module path
                        parts = module_name.split('.')
                        category = parts[1] if len(parts) > 1 else 'other'
                        
                        capabilities = []
                        hardware = []
                        
                        # Infer capabilities from category and node name
                        if 'infrastructure' in module_name:
                            capabilities = ['infrastructure']
                        elif 'loader' in module_name:
                            capabilities = ['model_loading']
                            if 'openvino' in module_name:
                                capabilities.append('openvino')
                                hardware = ['npu', 'cpu']
                        elif 'atomic' in module_name:
                            capabilities = ['atomic_operation']
                            if 'onnx' in node_id:
                                capabilities.append('inference')
                            if 'yolo' in node_id or 'detection' in node_id:
                                capabilities.append('object_detection')
                            if 'image' in node_id:
                                capabilities.append('image_processing')
                        elif 'video' in module_name:
                            capabilities = ['video_processing']
                        elif 'utils' in module_name:
                            capabilities = ['utility']
                        
                        discovered[node_id] = {
                            'module': module_name,
                            'function': name,
                            'capabilities': capabilities,
                            'hardware': hardware,
                            'dependencies': getattr(obj, 'dependencies', []),
                            'isolation_mode': getattr(obj, 'isolation_mode', 'none')
                        }
            except Exception as e:
                self.logger.debug(f"Failed to scan {py_file}: {e}")
        
        self.logger.info(f"Discovered {len(discovered)} workflow nodes")
        return discovered
    
    def compose_from_goal(self, goal: WorkflowGoal) -> Dict[str, Any]:
        """
        Generate a complete workflow from a high-level goal.
        
        Args:
            goal: High-level description of what to achieve
            
        Returns:
            Complete workflow JSON
        """
        self._ensure_nodes_discovered()  # Lazy load nodes
        self.logger.info(f"Composing workflow for goal: {goal.task}")
        
        # Select appropriate nodes based on goal
        if goal.task == "object_detection":
            # Determine if it's video or image based on input_type
            if goal.input_type.endswith(('.mp4', '.avi', '.mov', '.mkv')) or goal.input_type == "webcam":
                if goal.input_type == "webcam":
                    return self._compose_realtime_detection(goal)
                else:
                    return self._compose_video_detection(goal)
            else:
                return self._compose_image_detection(goal)
        
        raise ValueError(f"Unsupported task: {goal.task}")
    
    def _compose_video_detection(self, goal: WorkflowGoal) -> Dict[str, Any]:
        """Compose video object detection workflow."""
        nodes = []
        
        # Determine if we need fully atomic (granular) or wrapper approach
        # Fully atomic gives maximum flexibility for experimentation
        use_fully_atomic = goal.quality_over_speed  # Quality focus = more control
        
        # 1. Detect available hardware
        nodes.append({
            "id": "detect_hardware",
            "function": "workflow_nodes.infrastructure.detect_gpus.detect_gpus_node",
            "inputs": {}
        })
        
        # 2. Download model if needed
        nodes.append({
            "id": "download_model",
            "function": "workflow_nodes.infrastructure.download_model.download_model_node",
            "depends_on": [],
            "inputs": {
                "model_name": "yolov8s.onnx",
                "models_dir": "models"
            }
        })
        
        # 3. Open video capture
        nodes.append({
            "id": "video_capture",
            "function": "workflow_nodes.atomic.video_ops.open_video_capture_node",
            "depends_on": [],
            "inputs": {
                "source": goal.input_type
            }
        })
        
        # 4. Create ONNX session for inference
        nodes.append({
            "id": "create_session",
            "function": "workflow_nodes.atomic.onnx_ops.create_onnx_directml_session_node",
            "depends_on": ["download_model", "detect_hardware"],
            "inputs": {
                "model_path": "$download_model.model_path",
                "device_id": "$detect_hardware.directml_device_id"
            }
        })
        
        if use_fully_atomic:
            # Create fully atomic granular workflow (like examples)
            self.logger.info("Using fully atomic mode for maximum flexibility")
            return self._compose_fully_atomic_video_detection(goal, nodes)
        
        # 3. Choose detection pipeline based on hardware and performance target
        if goal.hardware_preference == "npu" or (
            goal.hardware_preference == "auto" and goal.performance_target and goal.performance_target > 20
        ):
            # Use OpenVINO for NPU
            nodes.append({
                "id": "load_model",
                "function": "workflow_nodes.model_loaders.openvino_loader.load_openvino_model_node",
                "depends_on": ["download_model"],
                "inputs": {
                    "model_path": "$download_model.model_path"
                }
            })
        
        # 5. Main processing node (wrapper approach)
        nodes.append({
            "id": "process_video",
            "function": "workflow_nodes.video.granular_video_loop.granular_video_loop_node",
            "depends_on": ["video_capture", "create_session", "download_model"],
            "inputs": {
                "conf_threshold": 0.3 if goal.quality_over_speed else 0.25,
                "iou_threshold": 0.7,
                "display_fps": True,
                "save_video": goal.output_type == "file"
            }
        })
        
        # 6. Performance stats
        if goal.output_type in ["display", "file"]:
            nodes.append({
                "id": "performance_stats",
                "function": "workflow_nodes.utils.performance_stats.performance_stats_node",
                "depends_on": ["process_video"],
                "inputs": {
                    "cpu_result": "$process_video"
                }
            })
        
        workflow = {
            "workflow": {
                "name": f"Auto-Generated: {goal.task}",
                "description": f"Autonomous workflow for {goal.task} on {goal.input_type}",
                "version": "1.0",
                "generated_by": "WorkflowComposer",
                "generated_at": datetime.now().isoformat(),
                "goal": asdict(goal)
            },
            "nodes": nodes
        }
        
        self.logger.info(f"Composed workflow with {len(nodes)} nodes")
        return workflow
    
    def _compose_realtime_detection(self, goal: WorkflowGoal) -> Dict[str, Any]:
        """Compose real-time webcam detection workflow."""
        # Similar to video but optimized for low latency
        workflow = self._compose_video_detection(goal)
        
        # Adjust for real-time constraints
        for node in workflow["nodes"]:
            if node["id"] == "process_video":
                node["inputs"]["conf_threshold"] = 0.4  # Higher threshold for speed
                node["inputs"]["display_fps"] = True
        
        workflow["workflow"]["name"] = "Auto-Generated: Real-time Detection"
        return workflow
    
    def _compose_fully_atomic_video_detection(
        self,
        goal: WorkflowGoal,
        base_nodes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compose fully atomic video detection workflow.
        
        Based on granular_parallel_inference.json pattern:
        - Each node does ONE atomic operation
        - Maximum composability and reuse
        - Minimal communication overhead
        - Complete YOLO pipeline breakdown
        
        For video: Uses granular_video_loop which internally orchestrates
        the atomic pipeline per frame. For images: Full atomic breakdown.
        """
        nodes = base_nodes.copy()
        
        # Determine if input is video or image
        is_video = goal.input_type.endswith(('.mp4', '.avi', '.mov', '.mkv'))
        
        if is_video:
            # Video workflow: Setup + granular_video_loop (which uses atomic nodes internally)
            return self._compose_atomic_video_workflow(goal, nodes)
        else:
            # Image workflow: Full atomic breakdown like granular_parallel_inference.json
            return self._compose_atomic_image_workflow(goal, nodes)
    
    def _compose_atomic_video_workflow(
        self,
        goal: WorkflowGoal,
        base_nodes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Atomic video workflow using granular_video_loop.
        
        The loop internally uses atomic nodes for each frame, providing
        composability while handling video efficiently.
        """
        nodes = base_nodes.copy()
        
        # Video capture (atomic)
        nodes.append({
            "id": "video_capture",
            "function": "workflow_nodes.atomic.video_ops.open_video_capture_node",
            "inputs": {
                "source": goal.input_type
            },
            "dependencies": []
        })
        
        # Session creation (atomic) - hardware aware
        if goal.hardware_preference in ["gpu", "auto"]:
            nodes.append({
                "id": "create_session",
                "function": "workflow_nodes.atomic.onnx_ops.create_onnx_directml_session_node",
                "inputs": {
                    "model_path": "models/yolov8s.onnx",
                    "device_id": 0
                },
                "dependencies": ["download_model", "detect_hardware"]
            })
        else:
            nodes.append({
                "id": "create_session",
                "function": "workflow_nodes.atomic.onnx_ops.create_onnx_cpu_session_node",
                "inputs": {
                    "model_path": "models/yolov8s.onnx"
                },
                "dependencies": ["download_model"]
            })
        
        # Video loop (orchestrates atomic operations per frame)
        conf_threshold = 0.3 if goal.quality_over_speed else 0.25
        nodes.append({
            "id": "video_loop",
            "function": "workflow_nodes.video.granular_video_loop.granular_video_loop_node",
            "inputs": {
                "conf_threshold": conf_threshold,
                "iou_threshold": 0.7,
                "display_fps": True,
                "save_video": goal.output_type == "file",
                "output_path": "output.mp4" if goal.output_type == "file" else None,
                "max_duration": 0
            },
            "dependencies": ["video_capture", "create_session", "download_model"]
        })
        
        # Performance stats (atomic)
        nodes.append({
            "id": "performance_stats",
            "function": "workflow_nodes.utils.performance_stats.performance_stats_node",
            "depends_on": ["video_loop"],
            "inputs": {
                "video_result": "$video_loop"
            }
        })
        
        return {
            "workflow": {
                "name": f"Atomic Video: {goal.task}",
                "description": f"Atomic video workflow using composable nodes for {goal.task}",
                "version": "1.0",
                "generated_by": "WorkflowComposer (Atomic Mode)",
                "generated_at": datetime.now().isoformat(),
                "goal": asdict(goal),
                "strategy": "fully_atomic",
                "optimized_parameters": {
                    "conf_threshold": conf_threshold,
                    "iou_threshold": 0.7
                }
            },
            "nodes": nodes
        }
    
    def _compose_atomic_image_workflow(
        self,
        goal: WorkflowGoal,
        base_nodes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Full atomic breakdown for image processing.
        
        Pattern from granular_parallel_inference.json:
        - Read image (1 node)
        - Preprocess (4 nodes): resize → normalize → transpose → add_batch
        - Session creation (1 node)
        - Inference (1 node)
        - Post-process (6 nodes): decode → filter → convert → nms → scale → format
        - Summary (1 node)
        
        Total: ~15 atomic nodes, each doing ONE thing.
        """
        nodes = base_nodes.copy()
        conf_threshold = 0.3 if goal.quality_over_speed else 0.25
        
        # ===================================================================
        # IMAGE PREPROCESSING PIPELINE (5 atomic nodes)
        # ===================================================================
        
        # 1. Read image (atomic)
        nodes.append({
            "id": "read_img",
            "function": "workflow_nodes.atomic.image_ops.read_image_node",
            "inputs": {
                "image_path": goal.input_type
            },
            "dependencies": []
        })
        
        # 2. Resize with letterbox (atomic)
        nodes.append({
            "id": "resize",
            "function": "workflow_nodes.atomic.image_ops.resize_image_letterbox_node",
            "inputs": {
                "target_width": 640,
                "target_height": 640
            },
            "dependencies": ["read_img"]
        })
        
        # 3. Normalize (atomic)
        nodes.append({
            "id": "normalize",
            "function": "workflow_nodes.atomic.image_ops.normalize_image_node",
            "inputs": {
                "scale": 255.0
            },
            "dependencies": ["resize"]
        })
        
        # 4. Transpose HWC → CHW (atomic)
        nodes.append({
            "id": "transpose",
            "function": "workflow_nodes.atomic.image_ops.hwc_to_chw_node",
            "inputs": {},
            "dependencies": ["normalize"]
        })
        
        # 5. Add batch dimension (atomic)
        nodes.append({
            "id": "add_batch",
            "function": "workflow_nodes.atomic.image_ops.add_batch_dimension_node",
            "inputs": {},
            "dependencies": ["transpose"]
        })
        
        # ===================================================================
        # SESSION CREATION (1 atomic node)
        # ===================================================================
        
        if goal.hardware_preference in ["gpu", "auto"]:
            nodes.append({
                "id": "create_session",
                "function": "workflow_nodes.atomic.onnx_ops.create_onnx_directml_session_node",
                "inputs": {
                    "model_path": "models/yolov8s.onnx",
                    "device_id": 0
                },
                "dependencies": ["download_model", "detect_hardware"]
            })
        else:
            nodes.append({
                "id": "create_session",
                "function": "workflow_nodes.atomic.onnx_ops.create_onnx_cpu_session_node",
                "inputs": {
                    "model_path": "models/yolov8s.onnx"
                },
                "dependencies": ["download_model"]
            })
        
        # ===================================================================
        # INFERENCE (1 atomic node)
        # ===================================================================
        
        nodes.append({
            "id": "inference",
            "function": "workflow_nodes.atomic.onnx_ops.run_onnx_inference_benchmark_node",
            "inputs": {
                "iterations": 1,  # Single inference for image
                "warmup_iterations": 0
            },
            "dependencies": ["create_session", "add_batch"]
        })
        
        # ===================================================================
        # YOLO POST-PROCESSING PIPELINE (6 atomic nodes)
        # ===================================================================
        
        # 1. Decode YOLO output (atomic)
        nodes.append({
            "id": "decode",
            "function": "workflow_nodes.atomic.yolo_ops.decode_yolo_v8_output_node",
            "inputs": {
                "num_classes": 80
            },
            "dependencies": ["inference"]
        })
        
        # 2. Filter by confidence (atomic)
        nodes.append({
            "id": "filter",
            "function": "workflow_nodes.atomic.yolo_ops.filter_by_confidence_node",
            "inputs": {
                "confidence_threshold": conf_threshold
            },
            "dependencies": ["decode"]
        })
        
        # 3. Convert box format (atomic)
        nodes.append({
            "id": "convert_boxes",
            "function": "workflow_nodes.atomic.yolo_ops.convert_cxcywh_to_xyxy_node",
            "inputs": {},
            "dependencies": ["filter"]
        })
        
        # 4. Apply NMS (atomic)
        nodes.append({
            "id": "nms",
            "function": "workflow_nodes.atomic.yolo_ops.apply_nms_node",
            "inputs": {
                "iou_threshold": 0.7
            },
            "dependencies": ["convert_boxes"]
        })
        
        # 5. Scale boxes to original size (atomic)
        nodes.append({
            "id": "scale_boxes",
            "function": "workflow_nodes.atomic.yolo_ops.scale_boxes_to_original_node",
            "inputs": {
                "input_width": 640,
                "input_height": 640
            },
            "dependencies": ["nms", "read_img", "resize"]
        })
        
        # 6. Format detections (atomic)
        nodes.append({
            "id": "format",
            "function": "workflow_nodes.atomic.yolo_ops.format_detections_coco_node",
            "inputs": {},
            "dependencies": ["scale_boxes"]
        })
        
        # ===================================================================
        # SUMMARY (1 atomic node)
        # ===================================================================
        
        provider = "DirectML" if goal.hardware_preference in ["gpu", "auto"] else "CPU"
        nodes.append({
            "id": "summary",
            "function": "workflow_nodes.atomic.yolo_ops.create_detection_summary_node",
            "inputs": {
                "provider": provider
            },
            "dependencies": ["format", "inference"]
        })
        
        return {
            "workflow": {
                "name": f"Atomic Image: {goal.task}",
                "description": f"Fully atomic workflow with {len(nodes)} composable nodes for {goal.task}",
                "version": "1.0",
                "generated_by": "WorkflowComposer (Atomic Mode - Full Breakdown)",
                "generated_at": datetime.now().isoformat(),
                "goal": asdict(goal),
                "strategy": "fully_atomic",
                "node_breakdown": {
                    "preprocessing": 5,
                    "session": 1,
                    "inference": 1,
                    "postprocessing": 6,
                    "summary": 1,
                    "infrastructure": 2,
                    "total": len(nodes)
                },
                "optimized_parameters": {
                    "conf_threshold": conf_threshold,
                    "iou_threshold": 0.7
                }
            },
            "nodes": nodes
        }
    
    def _compose_image_detection(self, goal: WorkflowGoal) -> Dict[str, Any]:
        """Compose single image detection workflow."""
        # Check if atomic mode requested
        if goal.quality_over_speed:
            self.logger.info("Using fully atomic mode for image detection (maximum flexibility)")
            # Create base infrastructure nodes
            base_nodes = [
                {
                    "id": "detect_hardware",
                    "function": "workflow_nodes.inference_node.detect_available_hardware",
                    "inputs": {},
                    "dependencies": []
                },
                {
                    "id": "download_model",
                    "function": "workflow_nodes.directml_model_loader_node.download_yolo_model",
                    "inputs": {"model_name": "yolov8s.onnx"},
                    "dependencies": []
                }
            ]
            return self._compose_atomic_image_workflow(goal, base_nodes)
        
        # Regular mode: simpler 4-node workflow for single images
        self.logger.info("Using regular mode for image detection")
        return self._compose_regular_image_workflow(goal)
    
    def _select_backend(self, hardware_preference: Optional[str] = None) -> str:
        """
        Select the best backend based on hardware preference.
        
        Args:
            hardware_preference: Preferred hardware ('directml', 'cuda', 'cpu', 'openvino')
            
        Returns:
            Backend name ('directml', 'cuda', 'cpu', or 'openvino')
        """
        if hardware_preference:
            return hardware_preference.lower()
        
        # Auto-select based on available nodes
        if 'create_onnx_directml_session' in self.available_nodes:
            return 'directml'
        elif 'create_onnx_cuda_session' in self.available_nodes:
            return 'cuda'
        elif 'load_openvino_model' in self.available_nodes:
            return 'openvino'
        else:
            return 'cpu'
    
    def _compose_regular_image_workflow(self, goal: WorkflowGoal) -> Dict[str, Any]:
        """Compose regular (non-atomic) image detection workflow."""
        nodes = []
        
        # Node 1: Detect hardware
        nodes.append({
            "id": "detect_hardware",
            "function": "workflow_nodes.inference_node.detect_available_hardware",
            "inputs": {},
            "outputs": ["hardware_status"]
        })
        
        # Node 2: Download model
        nodes.append({
            "id": "download_model",
            "function": "workflow_nodes.directml_model_loader_node.download_yolo_model",
            "inputs": {"model_name": "yolov8s.onnx"},
            "outputs": ["model_path"]
        })
        
        # Node 3: Process image (wrapper node)
        backend = self._select_backend(goal.hardware_preference)
        nodes.append({
            "id": "process_image",
            "function": f"workflow_nodes.image_reader_node.process_image_{backend}",
            "inputs": {
                "image_path": goal.input_type,
                "model_path": {"from": "download_model", "key": "model_path"},
                "conf_threshold": 0.25,
                "iou_threshold": 0.7
            },
            "outputs": ["detections", "inference_time"]
        })
        
        # Node 4: Performance stats
        nodes.append({
            "id": "performance_stats",
            "function": "workflow_nodes.performance_stats_node.log_performance_summary",
            "inputs": {
                "total_frames": 1,
                "inference_time": {"from": "process_image", "key": "inference_time"}
            },
            "outputs": []
        })
        
        return {
            "workflow": {
                "name": f"image_detection_{backend}",
                "description": f"Single image object detection using {backend}",
                "version": "1.0",
                "created": datetime.now().isoformat(),
                "strategy": "regular_image_pipeline",
                "backend": backend,
                "node_breakdown": {
                    "infrastructure": 2,
                    "processing": 1,
                    "summary": 1,
                    "total": 4
                }
            },
            "nodes": nodes
        }


# ============================================================================
# 2. Performance Optimizer - Self-Tuning Parameters
# ============================================================================

class PerformanceOptimizer:
    """Monitors performance and auto-tunes parameters."""
    
    def __init__(self, history_file: Path = Path("performance_history.json")):
        self.logger = logging.getLogger('workflow.optimizer')
        self.history_file = history_file
        self.execution_history: List[ExecutionRecord] = []
        self._load_history()
        
    def _load_history(self):
        """Load execution history from file."""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                    self.execution_history = [
                        ExecutionRecord(**record) for record in data
                    ]
                self.logger.info(f"Loaded {len(self.execution_history)} execution records")
            except Exception as e:
                self.logger.warning(f"Could not load history: {e}")
    
    def _save_history(self):
        """Save execution history to file."""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(
                    [asdict(record) for record in self.execution_history],
                    f,
                    indent=2
                )
        except Exception as e:
            self.logger.error(f"Could not save history: {e}")
    
    def record_execution(self, record: ExecutionRecord):
        """Record a workflow execution."""
        self.execution_history.append(record)
        self._save_history()
        self.logger.info(f"Recorded execution: {record.workflow_name} @ {record.performance.get('fps', 0):.1f} FPS")
    
    def suggest_parameters(self, workflow_type: str, target_fps: float = None) -> Dict[str, Any]:
        """
        Suggest optimal parameters based on execution history.
        
        Args:
            workflow_type: Type of workflow ('granular', 'monolithic')
            target_fps: Target FPS if specified
            
        Returns:
            Suggested parameters
        """
        # Filter relevant executions
        relevant = [
            r for r in self.execution_history
            if r.workflow_type == workflow_type and r.success
        ]
        
        if not relevant:
            # Return defaults if no history
            return {
                "conf_threshold": 0.25,
                "iou_threshold": 0.7,
                "batch_size": 1
            }
        
        # Sort by FPS
        relevant.sort(key=lambda r: r.performance.get('fps', 0), reverse=True)
        
        if target_fps:
            # Find executions close to target
            suitable = [
                r for r in relevant
                if abs(r.performance.get('fps', 0) - target_fps) < 5
            ]
            if suitable:
                best = suitable[0]
            else:
                # Use fastest
                best = relevant[0]
        else:
            # Use best performing
            best = relevant[0]
        
        suggested = best.parameters.copy()
        self.logger.info(f"Suggested parameters based on {best.workflow_name} @ {best.performance.get('fps', 0):.1f} FPS")
        
        return suggested
    
    def analyze_performance_trend(self, workflow_type: str, window: int = 10) -> Dict[str, Any]:
        """
        Analyze performance trends over recent executions.
        
        Args:
            workflow_type: Type of workflow
            window: Number of recent executions to analyze
            
        Returns:
            Trend analysis
        """
        relevant = [
            r for r in self.execution_history
            if r.workflow_type == workflow_type and r.success
        ][-window:]
        
        if not relevant:
            return {"trend": "unknown", "message": "No execution history"}
        
        fps_values = [r.performance.get('fps', 0) for r in relevant]
        
        analysis = {
            "executions": len(relevant),
            "avg_fps": statistics.mean(fps_values),
            "min_fps": min(fps_values),
            "max_fps": max(fps_values),
            "std_dev": statistics.stdev(fps_values) if len(fps_values) > 1 else 0
        }
        
        # Determine trend
        if len(relevant) >= 3:
            recent_avg = statistics.mean(fps_values[-3:])
            older_avg = statistics.mean(fps_values[:-3] if len(relevant) > 3 else fps_values)
            
            if recent_avg > older_avg * 1.1:
                analysis["trend"] = "improving"
            elif recent_avg < older_avg * 0.9:
                analysis["trend"] = "declining"
            else:
                analysis["trend"] = "stable"
        else:
            analysis["trend"] = "insufficient_data"
        
        return analysis


# ============================================================================
# 3. Pipeline Selector - Adaptive Strategy Selection
# ============================================================================

class PipelineSelector:
    """Chooses optimal pipeline strategy based on requirements."""
    
    def __init__(self):
        self.logger = logging.getLogger('workflow.selector')
        
        # Performance profiles for different strategies
        self.strategy_profiles = {
            "granular": {
                "typical_fps": 19,
                "composability": "high",
                "flexibility": "high",
                "performance": "good",
                "use_when": "Need flexibility, testing, or custom pipelines"
            },
            "monolithic": {
                "typical_fps": 25,
                "composability": "low",
                "flexibility": "low",
                "performance": "excellent",
                "use_when": "Need maximum performance, fixed pipeline"
            },
            "fast_pipeline": {
                "typical_fps": 22,
                "composability": "medium",
                "flexibility": "low",
                "performance": "very_good",
                "use_when": "Balance of performance and maintainability"
            }
        }
    
    def select_strategy(
        self,
        fps_target: Optional[float] = None,
        flexibility_needed: bool = True,
        hardware_available: Dict[str, bool] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Select optimal pipeline strategy.
        
        Args:
            fps_target: Target FPS requirement
            flexibility_needed: Whether workflow needs to be customizable
            hardware_available: Available hardware devices
            
        Returns:
            (strategy_name, configuration)
        """
        self.logger.info(f"Selecting strategy: target={fps_target} FPS, flexible={flexibility_needed}")
        
        # Decision logic
        if fps_target and fps_target > 23:
            # Need high performance
            if flexibility_needed:
                strategy = "fast_pipeline"
                self.logger.info("Selected fast_pipeline: High FPS + some flexibility")
            else:
                strategy = "monolithic"
                self.logger.info("Selected monolithic: Maximum performance")
        elif flexibility_needed:
            strategy = "granular"
            self.logger.info("Selected granular: Maximum flexibility")
        else:
            # Default to balanced approach
            strategy = "fast_pipeline"
            self.logger.info("Selected fast_pipeline: Balanced approach")
        
        config = {
            "strategy": strategy,
            "profile": self.strategy_profiles[strategy],
            "recommended_nodes": self._get_nodes_for_strategy(strategy)
        }
        
        return strategy, config
    
    def _get_nodes_for_strategy(self, strategy: str) -> List[str]:
        """Get recommended nodes for a strategy."""
        if strategy == "granular":
            return [
                "workflow_nodes.video.granular_video_loop.granular_video_loop_node"
            ]
        elif strategy == "monolithic":
            return [
                "workflow_nodes.video.monolithic_video_detection"  # If exists
            ]
        else:  # fast_pipeline
            return [
                "workflow_nodes.video.granular_video_loop.granular_video_loop_node"
            ]


# ============================================================================
# 5. Execution Learner - Knowledge Base & Suggestions
# ============================================================================

class ExecutionLearner:
    """Learns from execution history and provides insights."""
    
    def __init__(self, optimizer: PerformanceOptimizer):
        self.logger = logging.getLogger('workflow.learner')
        self.optimizer = optimizer
        self.knowledge_base: Dict[str, Any] = {}
    
    def build_knowledge_base(self):
        """Build knowledge base from execution history."""
        self.logger.info("Building knowledge base from execution history...")
        
        # Group by workflow type
        by_type = defaultdict(list)
        for record in self.optimizer.execution_history:
            if record.success:
                by_type[record.workflow_type].append(record)
        
        # Analyze each type
        for wf_type, records in by_type.items():
            fps_values = [r.performance.get('fps', 0) for r in records]
            
            self.knowledge_base[wf_type] = {
                "executions": len(records),
                "avg_fps": statistics.mean(fps_values),
                "best_fps": max(fps_values),
                "worst_fps": min(fps_values),
                "success_rate": len(records) / len([
                    r for r in self.optimizer.execution_history
                    if r.workflow_type == wf_type
                ]),
                "best_config": max(records, key=lambda r: r.performance.get('fps', 0)).parameters
            }
        
        self.logger.info(f"Knowledge base built: {len(self.knowledge_base)} workflow types")
    
    def get_insights(self) -> Dict[str, Any]:
        """Get insights and recommendations from knowledge base."""
        if not self.knowledge_base:
            self.build_knowledge_base()
        
        insights = {
            "total_executions": len(self.optimizer.execution_history),
            "workflow_types": list(self.knowledge_base.keys()),
            "recommendations": []
        }
        
        # Generate recommendations
        for wf_type, data in self.knowledge_base.items():
            if data["avg_fps"] < 15:
                insights["recommendations"].append({
                    "type": wf_type,
                    "issue": "low_fps",
                    "suggestion": "Consider using fast_pipeline or monolithic approach"
                })
            
            if data["success_rate"] < 0.8:
                insights["recommendations"].append({
                    "type": wf_type,
                    "issue": "low_success_rate",
                    "suggestion": "Check hardware compatibility and dependencies"
                })
        
        return insights
    
    def suggest_optimizations(self, current_fps: float, workflow_type: str) -> List[str]:
        """Suggest optimizations based on current performance."""
        suggestions = []
        
        if workflow_type in self.knowledge_base:
            best_fps = self.knowledge_base[workflow_type]["best_fps"]
            
            if current_fps < best_fps * 0.8:
                suggestions.append(
                    f"Your FPS ({current_fps:.1f}) is below historical best ({best_fps:.1f}). "
                    "Try the best known configuration."
                )
                suggestions.append(
                    f"Best config: {self.knowledge_base[workflow_type]['best_config']}"
                )
        
        if current_fps < 15:
            suggestions.append("Consider using SILENT log level for atomic nodes")
            suggestions.append("Ensure no debug logging is enabled")
            suggestions.append("Try reducing conf_threshold to 0.3 or higher")
        
        if current_fps < 20 and workflow_type == "granular":
            suggestions.append("Consider switching to fast_pipeline for better performance")
        
        return suggestions


# ============================================================================
# Main Agentic Workflow System
# ============================================================================

class AgenticWorkflowSystem:
    """
    Complete agentic workflow system combining all capabilities.
    
    Supports:
    - Rule-based composition (fast, no dependencies)
    - LLM-powered composition (AutoGen + Ollama for natural language)
    - Automatic fallback when LLM unavailable
    """
    
    def __init__(self, enable_llm: bool = True):
        """
        Initialize the agentic workflow system.
        
        Args:
            enable_llm: Try to enable LLM capabilities (AutoGen + Ollama).
                       Falls back to rule-based if unavailable.
        """
        self.logger = logging.getLogger('workflow.agent')
        
        # Core rule-based components (always available)
        self.composer = WorkflowComposer()
        self.optimizer = PerformanceOptimizer()
        self.selector = PipelineSelector()
        self.learner = ExecutionLearner(self.optimizer)
        
        # LLM components (optional)
        self.llm_composer = None
        self.llm_available = False
        
        if enable_llm:
            self._initialize_llm()
        
        mode = "LLM + Rule-based" if self.llm_available else "Rule-based only"
        self.logger.info(f"Agentic Workflow System initialized ({mode})")
    
    def _initialize_llm(self):
        """Initialize LLM capabilities (AutoGen + Ollama)."""
        try:
            from src.agentic.agent_llm import LLMWorkflowComposer, verify_ollama_connection
            
            # Check if Ollama is available (don't log too verbosely)
            import logging
            llm_logger = logging.getLogger('workflow.llm')
            original_level = llm_logger.level
            llm_logger.setLevel(logging.WARNING)  # Suppress info during init check
            
            ollama_available, message = verify_ollama_connection()
            
            llm_logger.setLevel(original_level)  # Restore level
            
            if ollama_available:
                self.llm_composer = LLMWorkflowComposer()
                self.llm_available = True
                self.logger.info(f"✅ LLM capabilities enabled: {message}")
            else:
                self.logger.warning(f"⚠️  LLM unavailable: {message}")
                self.logger.info("Falling back to rule-based composition")
        except ImportError as e:
            self.logger.warning(f"⚠️  LLM dependencies not installed: {e}")
            self.logger.info("Install with: pip install autogen-agentchat autogen-ext")
        except Exception as e:
            self.logger.warning(f"⚠️  Could not initialize LLM: {e}")
            self.logger.info("Falling back to rule-based composition")
    
    def create_workflow_from_goal(
        self, 
        goal: WorkflowGoal,
        use_llm: bool = False,
        natural_language: Optional[str] = None,
        use_multi_agent: bool = False
    ) -> Dict[str, Any]:
        """
        Create optimized workflow from high-level goal.
        
        Args:
            goal: WorkflowGoal specifying requirements
            use_llm: Force LLM usage (fails if unavailable)
            natural_language: Optional natural language description
                            (automatically enables LLM if available)
            use_multi_agent: Use full AutoGen multi-agent collaboration (async)
                           Requires Ollama and enables sophisticated planning
        
        This combines:
        - Workflow composition (generates nodes)
        - Parameter optimization (uses historical data)
        - Pipeline selection (chooses best strategy)
        
        Composition modes:
        1. Multi-agent LLM mode (if use_multi_agent=True)
           - Uses AutoGen RoundRobin team (Planner, Optimizer, Validator)
           - Agents collaborate asynchronously to design workflow
           - Extracts workflow from agent conversation
           - Falls back to rule-based if unavailable
        
        2. LLM-assisted mode (if natural_language provided or use_llm=True)
           - Uses intelligent natural language parsing
           - Fast workflow generation
           - Falls back to rule-based if unavailable
        
        3. Rule-based mode (default)
           - Fast, deterministic workflow generation
           - No external dependencies
           - Always available
        """
        self.logger.info(f"Creating workflow for goal: {goal.task}")
        
        # Determine composition mode
        should_use_llm = (use_llm or natural_language is not None or use_multi_agent)
        
        if should_use_llm:
            if not self.llm_available:
                if use_llm or use_multi_agent:
                    raise RuntimeError(
                        "LLM mode requested but unavailable. "
                        "Ensure Ollama is running and dependencies installed."
                    )
                else:
                    self.logger.warning("Natural language provided but LLM unavailable, using rule-based")
                    should_use_llm = False
        
        # Try LLM composition first
        if should_use_llm and self.llm_available:
            try:
                # Use natural language if provided, otherwise convert goal to NL
                nl_description = natural_language or self._goal_to_natural_language(goal)
                
                if use_multi_agent:
                    self.logger.info("🤖 Using multi-agent LLM composition (AutoGen collaboration)")
                    
                    # Use async multi-agent system
                    workflow = self.llm_composer.compose_workflow(
                        user_requirement=nl_description,
                        use_multi_agent=True
                    )
                else:
                    self.logger.info("🤖 Using LLM-powered composition (AutoGen + Ollama)")
                    
                    # LLM-assisted composition (fast)
                    workflow = self.llm_composer.compose_workflow(
                        user_requirement=nl_description,
                        use_multi_agent=False
                    )
                
                self.logger.info("✅ LLM composition successful")
                return workflow
                
            except Exception as e:
                self.logger.warning(f"⚠️  LLM composition failed: {e}")
                self.logger.info("Falling back to rule-based composition")
        
        # Rule-based composition (fallback or default)
        self.logger.info("⚙️  Using rule-based composition")
        
        # 1. Select optimal strategy
        strategy, config = self.selector.select_strategy(
            fps_target=goal.performance_target,
            flexibility_needed=not goal.quality_over_speed
        )
        
        # 2. Compose workflow
        workflow = self.composer.compose_from_goal(goal)
        
        # 3. Optimize parameters based on history
        suggested_params = self.optimizer.suggest_parameters(
            workflow_type=strategy,
            target_fps=goal.performance_target
        )
        
        # 4. Apply suggested parameters to workflow nodes
        for node in workflow["nodes"]:
            if "process_video" in node["id"]:
                node["inputs"].update(suggested_params)
        
        # 5. Add metadata
        workflow["workflow"]["strategy"] = strategy
        workflow["workflow"]["optimized_parameters"] = suggested_params
        workflow["workflow"]["composition_mode"] = "rule-based"
        
        return workflow
    
    def _goal_to_natural_language(self, goal: WorkflowGoal) -> str:
        """Convert WorkflowGoal to natural language description."""
        nl = f"Create a {goal.task} workflow "
        nl += f"for {goal.input_type} "
        nl += f"with output to {goal.output_type}. "
        
        if goal.performance_target:
            nl += f"Target {goal.performance_target} FPS. "
        
        if goal.quality_over_speed:
            nl += "Prioritize quality and flexibility over speed (use atomic nodes). "
        else:
            nl += "Prioritize speed and performance. "
        
        if goal.hardware_preference and goal.hardware_preference != "auto":
            nl += f"Use {goal.hardware_preference} hardware. "
        
        return nl
    
    def create_workflow_from_natural_language(self, description: str) -> Dict[str, Any]:
        """
        Create workflow from natural language description.
        
        Args:
            description: Natural language workflow requirements
        
        Returns:
            Generated workflow
        
        Examples:
            "Detect objects in video.mp4 using GPU at 30 FPS"
            "Process image.jpg with YOLO, use atomic nodes for flexibility"
            "Run video detection with DirectML, prioritize speed"
        """
        # Parse natural language to extract goal parameters
        goal = self._parse_natural_language_to_goal(description)
        
        # Use LLM composition with natural language
        return self.create_workflow_from_goal(
            goal=goal,
            natural_language=description
        )
    
    def _parse_natural_language_to_goal(self, description: str) -> WorkflowGoal:
        """Parse natural language into WorkflowGoal (basic rule-based parsing)."""
        desc_lower = description.lower()
        
        # Detect task
        if any(word in desc_lower for word in ["detect", "detection", "yolo", "object"]):
            task = "object_detection"
        elif any(word in desc_lower for word in ["classify", "classification"]):
            task = "classification"
        else:
            task = "object_detection"  # Default
        
        # Detect input
        import re
        video_match = re.search(r'[\w\-\.]+\.(mp4|avi|mov|mkv)', description)
        image_match = re.search(r'[\w\-\.]+\.(jpg|jpeg|png|bmp)', description)
        
        if video_match:
            input_type = video_match.group(0)
        elif image_match:
            input_type = image_match.group(0)
        else:
            input_type = "video.mp4"  # Default
        
        # Detect performance target
        fps_match = re.search(r'(\d+)\s*fps', desc_lower)
        performance_target = float(fps_match.group(1)) if fps_match else 20.0
        
        # Detect hardware preference
        if "gpu" in desc_lower or "directml" in desc_lower:
            hardware_preference = "gpu"
        elif "cpu" in desc_lower:
            hardware_preference = "cpu"
        elif "npu" in desc_lower:
            hardware_preference = "npu"
        else:
            hardware_preference = "auto"
        
        # Detect quality vs speed
        quality_keywords = ["atomic", "flexible", "quality", "granular", "breakdown"]
        speed_keywords = ["fast", "speed", "performance", "quick"]
        
        if any(word in desc_lower for word in quality_keywords):
            quality_over_speed = True
        elif any(word in desc_lower for word in speed_keywords):
            quality_over_speed = False
        else:
            quality_over_speed = False  # Default to speed
        
        return WorkflowGoal(
            task=task,
            input_type=input_type,
            output_type="display",
            performance_target=performance_target,
            hardware_preference=hardware_preference,
            quality_over_speed=quality_over_speed
        )
    
    def analyze_execution(
        self,
        workflow_name: str,
        workflow_type: str,
        performance: Dict[str, float],
        parameters: Dict[str, Any],
        hardware: Dict[str, bool]
    ):
        """
        Analyze execution and learn from it.
        
        Args:
            workflow_name: Name of executed workflow
            workflow_type: Type (granular/monolithic/etc)
            performance: Performance metrics (fps, latency, etc)
            parameters: Parameters used
            hardware: Hardware availability
        """
        # Record execution
        record = ExecutionRecord(
            timestamp=datetime.now().isoformat(),
            workflow_name=workflow_name,
            workflow_type=workflow_type,
            nodes=[],  # Can be populated if needed
            parameters=parameters,
            hardware=hardware,
            performance=performance,
            success=True
        )
        
        self.optimizer.record_execution(record)
        
        # Get insights
        insights = self.learner.get_insights()
        suggestions = self.learner.suggest_optimizations(
            performance.get('fps', 0),
            workflow_type
        )
        
        # Log recommendations
        if suggestions:
            self.logger.info("💡 Optimization suggestions:")
            for suggestion in suggestions:
                self.logger.info(f"  - {suggestion}")
        
        return {
            "recorded": True,
            "insights": insights,
            "suggestions": suggestions
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    print("🤖 Agentic Workflow System Demo")
    print("=" * 70)
    
    # Initialize system (with LLM support)
    agent = AgenticWorkflowSystem(enable_llm=True)
    
    # Example 1: Rule-based composition
    print("\n📋 Example 1: Rule-based workflow composition")
    print("-" * 70)
    
    goal = WorkflowGoal(
        task="object_detection",
        input_type="istockphoto-1585137173-640_adpp_is.mp4",
        output_type="display",
        performance_target=20.0,
        hardware_preference="auto",
        quality_over_speed=False
    )
    
    workflow = agent.create_workflow_from_goal(goal)
    
    print(f"\n✅ Generated workflow: {workflow['workflow']['name']}")
    print(f"   Mode: {workflow['workflow'].get('composition_mode', 'N/A')}")
    print(f"   Strategy: {workflow['workflow']['strategy']}")
    print(f"   Nodes: {len(workflow['nodes'])}")
    print(f"   Optimized params: {workflow['workflow']['optimized_parameters']}")
    
    # Save workflow
    output_file = Path("workflows/auto_generated_detection.json")
    with open(output_file, 'w') as f:
        json.dump(workflow, f, indent=2)
    print(f"\n💾 Saved to: {output_file}")
    
    # Example 2: Natural language composition (uses LLM if available)
    if agent.llm_available:
        print("\n🤖 Example 2: Natural language workflow composition (LLM)")
        print("-" * 70)
        
        nl_workflow = agent.create_workflow_from_natural_language(
            "Detect objects in soccer.jpg using GPU with atomic nodes for maximum flexibility"
        )
        
        print(f"\n✅ Generated from NL: {nl_workflow['workflow']['name']}")
        print(f"   Mode: {nl_workflow['workflow'].get('composition_mode', 'LLM')}")
        print(f"   Nodes: {len(nl_workflow['nodes'])}")
        
        # Save
        nl_output = Path("workflows/nl_generated_detection.json")
        with open(nl_output, 'w') as f:
            json.dump(nl_workflow, f, indent=2)
        print(f"   Saved to: {nl_output}")
    else:
        print("\n⚠️  Example 2: Skipped (LLM unavailable)")
        print("   To enable: Start Ollama and install dependencies")
    
    # Example 3: Analyze hypothetical execution
    print("\n📊 Example 3: Analyze execution and get suggestions")
    print("-" * 70)
    
    agent.analyze_execution(
        workflow_name="granular_video_detection",
        workflow_type="granular",
        performance={"fps": 18.7, "latency_ms": 53.5},
        parameters={"conf_threshold": 0.25, "iou_threshold": 0.7},
        hardware={"gpu": True, "npu": False, "cpu": True}
    )
    
    # Example 4: Get performance trends
    print("\n📈 Example 4: Performance trend analysis")
    print("-" * 70)
    
    trend = agent.optimizer.analyze_performance_trend("granular", window=5)
    print(f"Trend: {trend.get('trend', 'unknown')}")
    print(f"Avg FPS: {trend.get('avg_fps', 0):.1f}")
    print(f"Range: {trend.get('min_fps', 0):.1f} - {trend.get('max_fps', 0):.1f}")
    
    print("\n✅ Demo complete!")
    print("\n💡 System Capabilities:")
    print(f"  - Rule-based composition: ✅ Available")
    print(f"  - LLM composition (AutoGen): {'✅ Available' if agent.llm_available else '⚠️  Unavailable'}")
    print(f"  - Natural language: {'✅ Available' if agent.llm_available else '⚠️  Unavailable'}")
    print(f"  - Performance optimization: ✅ Available")
    print(f"  - Execution learning: ✅ Available")
    print("\n💡 Next steps:")
    print("  - Integrate with function_workflow_engine.py")
    print("  - Add real-time parameter tuning during execution")
    print("  - Use MCP server for external tool integration")


