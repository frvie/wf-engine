# Cleanup Recommendations for Workflow Engine

## Files to Archive (Move to archive/ or delete)

### Test Files (7 files - can be deleted or archived)
- `test_agent_simple.py` - Simple agent test
- `test_fully_atomic.py` - Atomic workflow test
- `test_integrated_llm.py` - LLM integration test
- `test_mcp.py` - MCP server test
- `test_multi_agent.py` - Multi-agent test
- `test_true_atomic.py` - True atomic test
- `test_workflow_builder.py` - Workflow builder test

**Recommendation**: Move to `archive/tests/` or delete if not needed

### POC Files (2 files - archive)
- `workflow_agent_poc.py` - Original POC
- `workflow_agent_poc_v2.py` - Updated POC

**Recommendation**: Move to `archive/poc/` (keep for reference)

### Demo Files (3 files - KEEP but organize)
- `demo_agent.py` - Useful for demonstrating agentic features ✅ KEEP
- `demo_capabilities.py` - Feature showcase
- `demo_llm_mcp.py` - LLM + MCP demo ✅ KEEP

**Recommendation**: Keep demo_agent.py and demo_llm_mcp.py, archive demo_capabilities.py

### Utility/Debug Scripts (4 files)
- `ATOMIC_SUMMARY.py` - Summary script (one-time use)
- `LLM_INTEGRATION_STATUS.py` - Status check script (one-time use)
- `profile_granular.py` - Profiling script (debugging)
- `show_performance.py` - Performance display

**Recommendation**: Move to `utilities/scripts/`

### Old/Redundant Files (3 files)
- `run_video_detection.py` - Superseded by wf.py CLI
- `verify_core_features.py` - One-time verification
- `workflow_cli.py` - Superseded by wf.py

**Recommendation**: Archive or delete

### Legacy Workflow Files
Check `workflows/` for old/test workflows:
- `cli_demo.json`, `cli_demo_fixed.json`, `test_final.json` - temporary test files
- `auto_generated_detection.json`, `generated_workflow.json` - old auto-generated files

**Recommendation**: Delete temporary test workflows

---

## Recommended Directory Structure After Cleanup

```
workflow_engine/
│
├── 🎯 CORE (Keep - 5 files)
│   ├── function_workflow_engine.py
│   ├── workflow_decorator.py
│   ├── workflow_environment_manager.py
│   ├── workflow_data_session.py
│   └── inference_engine.py
│
├── 🤖 AGENTIC (Keep - 3 files)
│   ├── workflow_agent.py
│   ├── workflow_agent_llm.py
│   └── agentic_integration.py
│
├── 🔌 INTERFACES (Keep - 3 files)
│   ├── wf.py                    # Main CLI
│   ├── mcp_server.py            # MCP server
│   └── workflow_builder.py      # Interactive builder
│
├── 📊 DEMOS (Keep - 2 files)
│   ├── demo_agent.py            # Agentic demo
│   └── demo_llm_mcp.py          # LLM + MCP demo
│
├── 🧩 workflow_nodes/ (Keep - all)
│   ├── atomic/
│   ├── infrastructure/
│   ├── model_loaders/
│   ├── utils/
│   └── video/
│
├── 🗂️ workflows/ (Clean up)
│   ├── granular_parallel_inference.json  ✅ KEEP
│   ├── granular_video_detection.json     ✅ KEEP
│   ├── granular_video_detection_mp4.json ✅ KEEP
│   └── [Keep production workflows, delete test ones]
│
├── 🛠️ utilities/ (Keep + move scripts here)
│   ├── logging_config.py
│   ├── shared_memory_utils.py
│   └── scripts/  (NEW - move utility scripts here)
│       ├── ATOMIC_SUMMARY.py
│       ├── LLM_INTEGRATION_STATUS.py
│       ├── profile_granular.py
│       └── show_performance.py
│
└── 📦 archive/ (NEW - archive old files)
    ├── tests/
    │   ├── test_agent_simple.py
    │   ├── test_fully_atomic.py
    │   ├── test_integrated_llm.py
    │   ├── test_mcp.py
    │   ├── test_multi_agent.py
    │   ├── test_true_atomic.py
    │   └── test_workflow_builder.py
    │
    ├── poc/
    │   ├── workflow_agent_poc.py
    │   └── workflow_agent_poc_v2.py
    │
    ├── old_demos/
    │   └── demo_capabilities.py
    │
    ├── old_scripts/
    │   ├── run_video_detection.py
    │   ├── verify_core_features.py
    │   └── workflow_cli.py
    │
    └── legacy_nodes/ (already exists)
```

---

## Cleanup Commands

### Option 1: Safe Cleanup (Archive everything)

```powershell
# Create archive directories
New-Item -ItemType Directory -Force -Path "archive/tests"
New-Item -ItemType Directory -Force -Path "archive/poc"
New-Item -ItemType Directory -Force -Path "archive/old_demos"
New-Item -ItemType Directory -Force -Path "archive/old_scripts"
New-Item -ItemType Directory -Force -Path "utilities/scripts"

# Move test files
Move-Item test_*.py archive/tests/

# Move POC files
Move-Item workflow_agent_poc*.py archive/poc/

# Move old demos
Move-Item demo_capabilities.py archive/old_demos/

# Move old scripts
Move-Item run_video_detection.py archive/old_scripts/
Move-Item verify_core_features.py archive/old_scripts/
Move-Item workflow_cli.py archive/old_scripts/

# Move utility scripts
Move-Item ATOMIC_SUMMARY.py utilities/scripts/
Move-Item LLM_INTEGRATION_STATUS.py utilities/scripts/
Move-Item profile_granular.py utilities/scripts/
Move-Item show_performance.py utilities/scripts/

# Clean up temporary workflow files
Remove-Item workflows/cli_demo.json -ErrorAction SilentlyContinue
Remove-Item workflows/cli_demo_fixed.json -ErrorAction SilentlyContinue
Remove-Item workflows/test_final.json -ErrorAction SilentlyContinue
Remove-Item workflows/test_working.json -ErrorAction SilentlyContinue
```

### Option 2: Aggressive Cleanup (Delete tests, archive POCs)

```powershell
# Create archive directories
New-Item -ItemType Directory -Force -Path "archive/poc"
New-Item -ItemType Directory -Force -Path "utilities/scripts"

# DELETE test files (if you don't need them)
Remove-Item test_*.py

# Archive POC files (keep for reference)
Move-Item workflow_agent_poc*.py archive/poc/

# Delete old demos and scripts
Remove-Item demo_capabilities.py
Remove-Item run_video_detection.py
Remove-Item verify_core_features.py
Remove-Item workflow_cli.py

# Move utility scripts
Move-Item ATOMIC_SUMMARY.py utilities/scripts/
Move-Item LLM_INTEGRATION_STATUS.py utilities/scripts/
Move-Item profile_granular.py utilities/scripts/
Move-Item show_performance.py utilities/scripts/

# Clean up temporary workflow files
Remove-Item workflows/*cli_demo*.json -ErrorAction SilentlyContinue
Remove-Item workflows/test_*.json -ErrorAction SilentlyContinue
Remove-Item workflows/auto_generated_detection.json -ErrorAction SilentlyContinue
Remove-Item workflows/generated_workflow.json -ErrorAction SilentlyContinue
Remove-Item workflows/nl_generated_workflow.json -ErrorAction SilentlyContinue
```

---

## Summary

**Core Files (Keep - 13 files):**
- 5 core engine files
- 3 agentic system files
- 3 interface files (wf.py, mcp_server.py, workflow_builder.py)
- 2 demo files (demo_agent.py, demo_llm_mcp.py)

**Files to Archive/Delete (16 files):**
- 7 test files
- 2 POC files
- 1 old demo
- 3 old scripts
- 4 utility scripts (move to utilities/scripts/)

**Temporary Workflows to Delete (~8 files):**
- cli_demo*.json
- test_*.json
- auto_generated_*.json
- generated_workflow.json
- nl_generated_workflow.json

**After cleanup:**
- Root directory: 13 clean, production-ready files
- Archive: Historical reference files preserved
- Utilities: Scripts organized in utilities/scripts/
- Workflows: Only production templates

---

## Execute Cleanup?

Run the script you prefer (Option 1 or Option 2) in PowerShell to clean up your workspace!
