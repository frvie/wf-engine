# Atomic Nodes - Known Issues & Solutions

## Issue 1: Auto-Injection with Numpy Arrays

### Problem
Atomic nodes fail with "missing required positional argument" errors even though they have proper dependencies.

**Root Cause:**
- Atomic nodes use typed parameters: `image: np.ndarray`
- Workflow engine auto-injection works via **kwargs unpacking
- Numpy arrays can't be serialized through the standard JSON workflow pipeline
- Engine tries to pass `**inputs` but numpy arrays aren't in the inputs dict

### Error Example
```
12:43:44 | workflow.inference.resize_image_letterbox | ERROR | 
❌ Node resize_image_letterbox failed: resize_image_letterbox_node() 
missing 1 required positional argument: 'image'
```

Even though `resize` depends on `read_img` which returns `{"image": <numpy_array>, ...}`

---

## Solutions

### Solution 1: In-Memory Context (RECOMMENDED) ✅

Keep node outputs in memory between nodes instead of serializing.

**Implementation:**
```python
# In function_workflow_engine.py
class FunctionWorkflowEngine:
    def __init__(self):
        self._node_outputs_cache = {}  # Keep outputs in memory
    
    def _prepare_inputs(self, node, results):
        inputs = node.get('inputs', {}).copy()
        
        # Inject dependency outputs from cache (not serialized)
        for dep_id in node.get('dependencies', []):
            if dep_id in self._node_outputs_cache:
                inputs.update(self._node_outputs_cache[dep_id])
        
        return inputs
    
    def _execute_function_node(self, node, inputs):
        result = func(**inputs)
        
        # Cache result for dependent nodes
        self._node_outputs_cache[node['id']] = result
        return result
```

**Benefits:**
- ✅ No serialization needed
- ✅ Zero-copy for numpy arrays
- ✅ Works with current node signatures
- ✅ Fast (no file I/O)

**Limitations:**
- ❌ Doesn't work for subprocess nodes (need shared memory)

---

### Solution 2: Shared Memory for Subprocess Nodes

Use existing shared memory system for subprocess isolation.

**Implementation:**
```python
# For subprocess nodes (e.g., DirectML)
if isolation_mode == "subprocess":
    # Serialize simple types to JSON
    simple_inputs = {k: v for k, v in inputs.items() 
                     if not isinstance(v, np.ndarray)}
    
    # Put numpy arrays in shared memory
    for k, v in inputs.items():
        if isinstance(v, np.ndarray):
            shm_name = f"{node_id}_{k}"
            create_shared_memory_with_header(shm_name, v)
            simple_inputs[f"{k}_shm"] = shm_name
    
    # Execute in subprocess with shared memory references
    result = _execute_in_environment(func_name, simple_inputs, env_info)
```

**Benefits:**
- ✅ Works across processes
- ✅ Zero-copy IPC
- ✅ Already implemented in engine

---

### Solution 3: Global Cache (Quick Fix) ⚠️

Use global cache like monolithic nodes.

**Implementation:**
```python
# workflow_nodes/atomic/image_ops.py
_ATOMIC_CACHE = {}

@workflow_node("read_image")
def read_image_node(image_path: str) -> dict:
    image = cv2.imread(image_path)
    
    # Store in global cache
    _ATOMIC_CACHE['image'] = image
    
    return {
        "image_cached": True,  # Flag for next node
        "path": image_path,
        "height": h,
        "width": w
    }

@workflow_node("resize_image_letterbox")
def resize_image_letterbox_node(
    image_cached: bool = None,  # Signal to use cache
    target_width: int = 640,
    target_height: int = 640
) -> dict:
    # Get from cache
    image = _ATOMIC_CACHE.get('image')
    
    # Resize...
    resized = ...
    
    # Update cache
    _ATOMIC_CACHE['image'] = resized
    
    return {"image_cached": True, ...}
```

**Benefits:**
- ✅ Quick to implement
- ✅ Works with current engine

**Limitations:**
- ❌ Not thread-safe
- ❌ Global state (bad practice)
- ❌ Doesn't work for subprocess nodes

---

## Recommended Approach

**Hybrid Solution:**

1. **In-process nodes** → Use Solution 1 (in-memory context)
2. **Subprocess nodes** → Use Solution 2 (shared memory)

**Implementation Priority:**
1. ✅ Update `function_workflow_engine.py` to keep outputs in memory
2. ✅ Update `_prepare_inputs()` to inject from memory cache
3. ✅ Update `_execute_in_environment()` to use shared memory for numpy arrays
4. ✅ Test granular workflows

---

## Testing Strategy

### Test 1: Simple In-Process Pipeline
```python
# Test atomic nodes without subprocess
read_image → resize → normalize → transpose → add_batch
```

**Expected:** All should pass if in-memory context works

### Test 2: DirectML Subprocess Pipeline
```python
# Test with subprocess node
[preprocessing] → create_directml_session → benchmark → [postprocessing]
```

**Expected:** Shared memory needed for DirectML subprocess

### Test 3: Multi-Backend Comparison
```python
# Test parallel execution with shared preprocessing
[preprocessing - shared]
    ├→ CPU session → inference
    ├→ DirectML session (subprocess) → inference
    └→ CUDA session → inference
```

**Expected:** Both in-memory and shared memory needed

---

## Next Steps

1. **Implement Solution 1** - In-memory context for in-process nodes
2. **Test simple pipeline** - Verify auto-injection works
3. **Extend for subprocess** - Add shared memory support
4. **Test complex workflows** - Multi-backend, parallel execution
5. **Document** - Update granular nodes guide

---

**Status:** 🔧 In Progress  
**Priority:** High (blocking atomic node usage)  
**Estimated Fix Time:** 1-2 hours
