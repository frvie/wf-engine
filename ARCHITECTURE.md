# Workflow Engine Architecture & Dependencies

## File System Tree

```
workflow_engine/
│
├── 🎯 CORE ENGINE (Workflow Execution)
│   ├── function_workflow_engine.py         # Main workflow executor (FunctionWorkflowEngine class)
│   ├── workflow_decorator.py                # @workflow_node decorator, _GLOBAL_CACHE
│   ├── workflow_environment_manager.py      # Environment isolation & management
│   └── workflow_data_session.py             # Data session management
│
├── 🤖 AGENTIC SYSTEM (Autonomous Capabilities)
│   ├── workflow_agent.py                    # Core agentic system (WorkflowComposer, PerformanceOptimizer, etc.)
│   ├── workflow_agent_llm.py                # LLM-powered composition (AutoGen + Ollama)
│   ├── workflow_agent_poc.py                # Proof of concept implementations
│   ├── workflow_agent_poc_v2.py             # Updated POC
│   └── agentic_integration.py               # Integration layer (AgenticWorkflowEngine)
│
├── 🔌 INTERFACES (User & AI Integration)
│   ├── wf.py                                # CLI interface (simplified commands)
│   ├── workflow_cli.py                      # Original CLI
│   ├── mcp_server.py                        # Model Context Protocol server
│   └── workflow_builder.py                  # Interactive workflow builder
│
├── 📊 DEMONSTRATIONS
│   ├── demo_agent.py                        # Agentic system demo
│   ├── demo_llm_mcp.py                      # LLM + MCP integration demo
│   └── demo_capabilities.py                 # Feature showcase
│
├── 🧩 WORKFLOW NODES (Atomic Operations)
│   ├── workflow_nodes/
│   │   ├── atomic/
│   │   │   ├── image_ops.py                 # 9 image processing nodes
│   │   │   ├── onnx_ops.py                  # 6 ONNX inference nodes
│   │   │   ├── video_ops.py                 # 2 video I/O nodes
│   │   │   └── yolo_ops.py                  # 7 YOLO post-processing nodes
│   │   │
│   │   ├── infrastructure/
│   │   │   ├── detect_gpus.py               # Hardware detection (GPU, NPU, CUDA, DirectML)
│   │   │   └── download_model.py            # Model management
│   │   │
│   │   ├── model_loaders/
│   │   │   └── openvino_loader.py           # OpenVINO NPU model loading & inference
│   │   │
│   │   ├── video/
│   │   │   ├── granular_video_loop.py       # Video processing orchestrator
│   │   │   └── fast_yolo_pipeline.py        # Optimized pipeline
│   │   │
│   │   ├── generator/                       # 🆕 NODE GENERATOR
│   │   │   ├── __init__.py
│   │   │   └── node_generator.py            # LLM-powered node code generation
│   │   │
│   │   ├── custom/                          # 🆕 GENERATED NODES
│   │   │   ├── README.md                    # Usage guide
│   │   │   └── <auto_generated>.py          # Nodes created by generator
│   │   │
│   │   └── utils/
│   │       ├── performance_stats.py         # Performance comparison & metrics
│   │       └── visualize_detections.py      # Detection visualization
│
├── 🗂️ WORKFLOWS (JSON Configurations)
│   └── workflows/
│       ├── granular_parallel_inference.json # Multi-backend benchmark (35 nodes)
│       ├── granular_video_detection.json    # Standard video detection
│       ├── test_working.json                # Auto-generated working example
│       └── [18 more workflow templates]
│
├── 🛠️ UTILITIES
│   ├── utilities/
│   │   ├── logging_config.py                # Centralized logging
│   │   └── shared_memory_utils.py           # IPC for subprocess isolation
│
└── 📚 DOCUMENTATION
    ├── CLI_GUIDE.md                         # CLI usage guide
    ├── ARCHITECTURE.md                      # This file
    ├── LLM_MCP_INTEGRATION.md              # LLM + MCP documentation
    ├── NODE_GENERATOR.md                   # 🆕 Node Generator documentation
    ├── REFACTORING_SUMMARY.md              # Refactoring history
    └── [10 more documentation files]
```

---

## Dependency Graph

### 1. Core Execution Layer

```
┌─────────────────────────────────────────────────────────────┐
│                  function_workflow_engine.py                 │
│  - FunctionWorkflowEngine (main executor)                   │
│  - Node discovery with rglob()                              │
│  - Dependency resolution & topological sort                 │
│  - Subprocess isolation with shared memory                  │
└─────────────────────────────────────────────────────────────┘
                          ↓ depends on
        ┌─────────────────┬───────────────────┬────────────────┐
        ↓                 ↓                   ↓                ↓
┌─────────────────┐ ┌──────────────┐ ┌────────────┐ ┌──────────────┐
│workflow_        │ │workflow_     │ │shared_     │ │workflow_     │
│decorator.py     │ │environment_  │ │memory_     │ │data_session  │
│                 │ │manager.py    │ │utils.py    │ │.py           │
│- @workflow_node │ │              │ │            │ │              │
│- _GLOBAL_CACHE  │ │- venv mgmt   │ │- FLAG sync │ │- sessions    │
└─────────────────┘ └──────────────┘ └────────────┘ └──────────────┘
```

**Key Responsibilities:**
- **function_workflow_engine.py**: Executes workflows, manages node lifecycle
- **workflow_decorator.py**: Provides `@workflow_node` decorator, global cache for unpicklable objects
- **workflow_environment_manager.py**: Manages isolated Python environments per node
- **shared_memory_utils.py**: IPC for subprocess communication (pickle + shared memory)

---

### 2. Agentic Intelligence Layer

```
┌─────────────────────────────────────────────────────────────┐
│                     workflow_agent.py                        │
│  - WorkflowComposer (discovers & composes workflows)        │
│  - PerformanceOptimizer (tunes parameters)                  │
│  - PipelineSelector (chooses strategies)                    │
│  - ExecutionLearner (builds knowledge base)                 │
│  - AgenticWorkflowSystem (orchestrates all)                 │
└─────────────────────────────────────────────────────────────┘
                          ↓ extends with
        ┌─────────────────┴───────────────────┐
        ↓                                     ↓
┌─────────────────────┐           ┌─────────────────────┐
│workflow_agent_llm.py│           │agentic_integration.py│
│                     │           │                      │
│- LLMWorkflowComposer│           │- AgenticWorkflowEngine│
│- AutoGen agents     │           │  (extends core engine)│
│- Ollama integration │           │- Natural language    │
│- Multi-agent system │           │  parsing             │
└─────────────────────┘           └─────────────────────┘
        ↓ uses                              ↓ uses
┌─────────────────────┐           ┌─────────────────────┐
│   Ollama Server     │           │ workflow_agent.py   │
│   (External LLM)    │           │ (rule-based)        │
└─────────────────────┘           └─────────────────────┘
```

**Dependencies:**
```python
# workflow_agent.py (Core Agentic System)
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json, logging, statistics
# NO dependency on function_workflow_engine.py

# agentic_integration.py (Integration Layer)
from workflow_agent import AgenticWorkflowSystem, WorkflowGoal, ExecutionRecord
from function_workflow_engine import FunctionWorkflowEngine  # EXTENDS core engine

# workflow_agent_llm.py (LLM Enhancement)
from workflow_agent import WorkflowComposer
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
# Requires: Ollama running locally
```

**Agentic System is LOOSELY COUPLED** - it can function independently of the core engine!

---

### 3. User Interface Layer

```
┌─────────────────────────────────────────────────────────────┐
│                           wf.py                              │
│                     (Simplified CLI)                         │
│  - cmd_run, cmd_create, cmd_optimize                        │
│  - cmd_status, cmd_devices, cmd_nodes                       │
│  - cmd_mcp, cmd_demo, cmd_templates                         │
└─────────────────────────────────────────────────────────────┘
                          ↓ uses
        ┌─────────────────┬───────────────────┬────────────────┐
        ↓                 ↓                   ↓                ↓
┌─────────────────┐ ┌──────────────┐ ┌────────────┐ ┌──────────────┐
│function_        │ │agentic_      │ │workflow_   │ │workflow_     │
│workflow_engine  │ │integration   │ │agent       │ │nodes/*       │
└─────────────────┘ └──────────────┘ └────────────┘ └──────────────┘
```

```
┌─────────────────────────────────────────────────────────────┐
│                        mcp_server.py                         │
│                  (Model Context Protocol)                    │
│  - 9 MCP tools for AI agents                                │
│  - execute_workflow, create_workflow_from_nl                │
│  - optimize_workflow, analyze_performance                   │
└─────────────────────────────────────────────────────────────┘
                          ↓ uses
        ┌─────────────────┬───────────────────┬────────────────┐
        ↓                 ↓                   ↓                ↓
┌─────────────────┐ ┌──────────────┐ ┌────────────┐ ┌──────────────┐
│function_        │ │agentic_      │ │workflow_   │ │workflow_     │
│workflow_engine  │ │integration   │ │agent_llm   │ │agent         │
└─────────────────┘ └──────────────┘ └────────────┘ └──────────────┘
        ↓
    Claude Desktop / Cline / MCP Clients
```

---

### 4. Workflow Nodes Layer

```
workflow_nodes/
├── atomic/                      # 24 atomic operations
│   ├── image_ops.py            # Independent, composable
│   ├── onnx_ops.py             # Independent, composable
│   ├── video_ops.py            # Independent, composable
│   └── yolo_ops.py             # Independent, composable
│
├── infrastructure/              # 2 infrastructure nodes
│   ├── detect_gpus.py          # NO dependencies on other nodes
│   └── download_model.py       # NO dependencies on other nodes
│
├── model_loaders/               # 2 model loader nodes
│   └── openvino_loader.py      # Uses _GLOBAL_CACHE from decorator
│
├── video/                       # 2 orchestrator nodes
│   ├── granular_video_loop.py  # USES atomic nodes (imports them)
│   └── fast_yolo_pipeline.py   # Optimized variant
│
└── utils/                       # 2 utility nodes
    ├── performance_stats.py     # Independent
    └── visualize_detections.py  # Independent
```

**Node Dependencies:**
```python
# All nodes depend ONLY on:
from workflow_decorator import workflow_node, _GLOBAL_CACHE

# Orchestrator nodes (granular_video_loop.py) import atomics:
from workflow_nodes.atomic.image_ops import resize_image_letterbox_node
from workflow_nodes.atomic.onnx_ops import run_onnx_inference_single_node
from workflow_nodes.atomic.yolo_ops import decode_yolo_v8_output_node
# etc...

# NO circular dependencies!
# Nodes are completely independent of the workflow engine
```

---

## Dependency Summary

### Core Dependencies (Required)
```
function_workflow_engine.py
    ├── workflow_decorator.py
    ├── workflow_environment_manager.py
    ├── workflow_data_session.py
    └── utilities/shared_memory_utils.py
```

### Agentic Dependencies (Optional - for autonomous features)
```
workflow_agent.py (NO deps on core engine!)
    ├── Standard library only: json, logging, pathlib, dataclasses
    └── Discovers nodes by scanning filesystem

agentic_integration.py
    ├── workflow_agent.py
    └── function_workflow_engine.py  # EXTENDS it

workflow_agent_llm.py
    ├── workflow_agent.py
    ├── autogen_agentchat
    └── Ollama (external service)
```

### CLI Dependencies
```
wf.py
    ├── function_workflow_engine.py  # For cmd_run
    ├── agentic_integration.py       # For cmd_create
    └── workflow_agent.py            # For cmd_status, cmd_optimize
```

### MCP Server Dependencies
```
mcp_server.py
    ├── mcp.server (MCP SDK)
    ├── function_workflow_engine.py
    ├── agentic_integration.py
    ├── workflow_agent.py
    └── workflow_agent_llm.py
```

### Node Dependencies (Minimal!)
```
All workflow nodes/*.py
    └── workflow_decorator.py  # ONLY dependency!

Orchestrator nodes (video/granular_video_loop.py)
    └── atomic nodes (import as functions)
```

---

## Key Architectural Principles

### 1. **Loose Coupling**
- Agentic system (`workflow_agent.py`) is **independent** of core engine
- Can compose workflows without executing them
- Core engine can run without agentic features

### 2. **Layered Architecture**
```
┌─────────────────────────────────────┐
│  User Interfaces (CLI, MCP, demos) │  ← High-level
├─────────────────────────────────────┤
│  Agentic System (optional layer)   │  ← Intelligence
├─────────────────────────────────────┤
│  Core Workflow Engine (required)   │  ← Execution
├─────────────────────────────────────┤
│  Workflow Nodes (atomic units)     │  ← Operations
└─────────────────────────────────────┘
```

### 3. **Separation of Concerns**
- **Execution**: `function_workflow_engine.py`
- **Intelligence**: `workflow_agent.py`
- **Integration**: `agentic_integration.py`
- **User Interface**: `wf.py`, `mcp_server.py`
- **Operations**: `workflow_nodes/*`

### 4. **Plugin Architecture**
- Nodes are discovered dynamically via filesystem scan
- No hardcoded node registry
- New nodes automatically detected

### 5. **Environment Isolation**
- Each node can run in separate Python environment
- Managed by `workflow_environment_manager.py`
- Subprocess communication via shared memory

---

## Data Flow

### Workflow Execution Flow
```
1. User Input (CLI/MCP)
        ↓
2. Workflow JSON (manual or auto-generated)
        ↓
3. FunctionWorkflowEngine.execute()
        ↓
4. Node Discovery (rglob scan)
        ↓
5. Dependency Resolution (topological sort)
        ↓
6. Sequential Execution
        ├── In-process (isolation_mode="none")
        └── Subprocess (isolation_mode="subprocess" + shared memory)
        ↓
7. Results Aggregation
        ↓
8. Performance Logging (if agentic enabled)
        ↓
9. Output to User
```

### Agentic Workflow Creation Flow
```
1. Natural Language Input
        ↓
2. WorkflowGoal Extraction (agentic_integration.py)
        ↓
3. Strategy Selection (PipelineSelector)
        ↓
4. Workflow Composition (WorkflowComposer)
        ├── Rule-based (workflow_agent.py)
        └── LLM-powered (workflow_agent_llm.py + Ollama)
        ↓
5. Parameter Optimization (PerformanceOptimizer)
        ↓
6. Workflow JSON Generation
        ↓
7. Optional: Execute Immediately
        ↓
8. Learning: Record execution metrics
```

---

## Component Interaction Matrix

|                          | Core Engine | Agentic System | CLI | MCP | Nodes | Node Generator |
|--------------------------|-------------|----------------|-----|-----|-------|----------------|
| **Core Engine**          | -           | No             | Yes | Yes | Yes   | No             |
| **Agentic System**       | No          | -              | Yes | Yes | No    | Yes            |
| **CLI**                  | Yes         | Yes            | -   | No  | Yes   | Yes            |
| **MCP**                  | Yes         | Yes            | No  | -   | Yes   | No             |
| **Nodes**                | No          | No             | No  | No  | -     | No             |
| **Node Generator** 🆕    | No          | No             | Yes | No  | Yes   | -              |

**Key Insights**: 
- Nodes are completely isolated! They only depend on the decorator.
- Node Generator can create new nodes that are immediately discovered.

---

## Node Generator System 🆕

### Self-Extending Capability

The **Node Generator** enables the workflow engine to create new nodes on-demand:

```
┌─────────────────────────────────────────────────────────────┐
│                      Node Generator                          │
│  - LLM-powered code generation (Ollama)                     │
│  - Automatic @workflow_node decorator application           │
│  - Syntax validation & AST parsing                          │
│  - Template-based fallback generation                       │
│  - Auto-save to workflow_nodes/custom/                      │
└─────────────────────────────────────────────────────────────┘
                          ↓ generates
        ┌─────────────────┴───────────────────┐
        ↓                                     ↓
┌─────────────────────┐           ┌─────────────────────┐
│ Custom Nodes        │           │ Atomic Nodes        │
│ (custom/)           │           │ (atomic/)           │
│                     │           │                     │
│- User-defined logic │           │- Specialized ops    │
│- Auto-discovered    │           │- Composable units   │
└─────────────────────┘           └─────────────────────┘
        ↓                                     ↓
   Immediately available in WorkflowComposer
```

### Generation Flow

```
1. Natural Language Description
        ↓
2. NodeSpec Creation (goal, inputs, outputs)
        ↓
3. LLM Prompt Generation
        ↓
4. Code Generation (Ollama qwen2.5-coder:7b)
        ↓
5. Validation (AST parsing, decorator check)
        ↓
6. Auto-save to workflow_nodes/custom/
        ↓
7. Immediate Discovery by rglob() scan
        ↓
8. Available in workflows!
```

### Usage

**CLI:**
```bash
# Generate new node
wf generate "apply median filter" -i image -o filtered

# Specify category
wf generate "detect faces" -i image -o faces -c atomic

# Show generated code
wf generate "enhance contrast" --show-code
```

**Python API:**
```python
from workflow_nodes.generator.node_generator import NodeGenerator, NodeSpec

spec = NodeSpec(
    goal="Apply Gaussian blur",
    inputs=["image", "kernel_size"],
    outputs=["blurred"],
    category="custom"
)

result = await generator.generate_node(spec)
```

---

## File Size & Complexity

| Component | Lines of Code | Complexity | Purpose |
|-----------|--------------|------------|---------|
| `function_workflow_engine.py` | ~657 | High | Core execution engine |
| `workflow_agent.py` | ~1,455 | Very High | Agentic system (4 subsystems) |
| `workflow_agent_llm.py` | ~600 | High | LLM integration |
| `agentic_integration.py` | ~256 | Medium | Integration layer |
| `wf.py` | ~500 | Medium | CLI interface (with generate cmd) |
| `mcp_server.py` | ~590 | Medium | MCP server |
| `workflow_decorator.py` | ~150 | Low | Decorator definition |
| `node_generator.py` 🆕 | ~450 | Medium | LLM-powered node generation |
| **Total Atomic Nodes** | ~1,200 | Medium | 32 independent nodes |
| **Generated Custom Nodes** 🆕 | Variable | Variable | User-created via generator |

---

## Summary

**The system has a clean layered architecture:**

1. **Core Layer** (`function_workflow_engine.py`) - Can run standalone
2. **Intelligence Layer** (`workflow_agent.py`) - Optional, loosely coupled
3. **Enhancement Layer** (`workflow_agent_llm.py`) - Optional LLM features
4. **Interface Layer** (`wf.py`, `mcp_server.py`) - User-facing
5. **Operations Layer** (`workflow_nodes/*`) - Independent atomic units
6. **🆕 Generation Layer** (`node_generator.py`) - Self-extending capability

**Key Strengths:**
- ✅ Loose coupling between layers
- ✅ Core engine works without agentic features
- ✅ Agentic system works without LLM
- ✅ Nodes are completely independent
- ✅ 🆕 **Self-extending** - Can generate new nodes on-demand
- ✅ 🆕 **LLM-powered** - Automatic code generation from natural language
- ✅ Plugin architecture for extensibility
- ✅ Multiple interfaces (CLI, MCP, programmatic)

**You can use:**
- Just the core engine (basic workflow execution)
- Core + Agentic (autonomous optimization)
- Core + Agentic + LLM (natural language composition)
- Any combination based on your needs!
