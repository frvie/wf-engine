"""
Custom Workflow Node Template

Copy this template to create your own workflow nodes.
The engine will automatically discover and execute them.
"""

from workflow_decorator import workflow_node
from typing import Dict, Any


# ============================================================================
# SIMPLE NODE TEMPLATE (No dependencies, runs in main environment)
# ============================================================================

@workflow_node(
    "my_simple_node",           # Unique node ID
    dependencies=[],            # No external dependencies
    isolation_mode="none"       # Run in main environment
)
def my_simple_node(input_param: str, optional_param: int = 10) -> Dict[str, Any]:
    """
    Simple node that processes data without external dependencies.
    
    Args:
        input_param: Description of input parameter
        optional_param: Optional parameter with default value
        
    Returns:
        Dictionary with results that can be used by downstream nodes
    """
    # Your logic here
    result = f"Processed: {input_param} with {optional_param}"
    
    return {
        "output": result,
        "status": "success",
        "metadata": {
            "input_length": len(input_param),
            "param_used": optional_param
        }
    }


# ============================================================================
# NODE WITH DEPENDENCIES (Installs packages automatically)
# ============================================================================

@workflow_node(
    "data_analysis_node",
    dependencies=["pandas", "numpy", "scipy"],  # Auto-installed if missing
    isolation_mode="none"
)
def data_analysis_node(data: Dict, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Node that uses pandas/numpy for data analysis.
    Dependencies are automatically installed if missing.
    """
    import pandas as pd
    import numpy as np
    
    # Convert dict to DataFrame
    df = pd.DataFrame(data)
    
    # Perform analysis
    stats = {
        "mean": df.mean().to_dict(),
        "std": df.std().to_dict(),
        "count": len(df)
    }
    
    return {
        "statistics": stats,
        "filtered_count": len(df[df > threshold]),
        "status": "complete"
    }


# ============================================================================
# ISOLATED NODE (Runs in separate environment to avoid conflicts)
# ============================================================================

@workflow_node(
    "ml_inference_node",
    dependencies=["tensorflow", "torch"],  # Conflicting packages
    isolation_mode="subprocess"            # Runs in isolated environment
)
def ml_inference_node(model_path: str, input_data: list) -> Dict[str, Any]:
    """
    Node that runs ML inference in an isolated environment.
    Use this when packages might conflict with other nodes.
    
    Process isolation ensures:
    - No dependency conflicts (e.g., TensorFlow vs PyTorch)
    - Clean environment for each run
    - Automatic cleanup after execution
    """
    import tensorflow as tf
    import numpy as np
    
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Run inference
    predictions = model.predict(np.array(input_data))
    
    return {
        "predictions": predictions.tolist(),
        "model_used": model_path,
        "num_predictions": len(predictions)
    }


# ============================================================================
# NODE WITH MULTIPLE OUTPUTS (For branching workflows)
# ============================================================================

@workflow_node(
    "split_data_node",
    dependencies=["pandas"],
    isolation_mode="none"
)
def split_data_node(data: Dict, train_ratio: float = 0.8) -> Dict[str, Any]:
    """
    Node that splits data for downstream parallel processing.
    Multiple outputs can be referenced separately in workflow JSON.
    """
    import pandas as pd
    
    df = pd.DataFrame(data)
    split_idx = int(len(df) * train_ratio)
    
    train_data = df[:split_idx].to_dict()
    test_data = df[split_idx:].to_dict()
    
    return {
        "train_data": train_data,      # Can reference as $split.train_data
        "test_data": test_data,         # Can reference as $split.test_data
        "train_size": split_idx,
        "test_size": len(df) - split_idx
    }


# ============================================================================
# ERROR HANDLING NODE (Graceful failure)
# ============================================================================

@workflow_node(
    "safe_processing_node",
    dependencies=["requests"],
    isolation_mode="none"
)
def safe_processing_node(url: str, timeout: int = 10) -> Dict[str, Any]:
    """
    Node with proper error handling.
    Returns error information that downstream nodes can handle.
    """
    import requests
    
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        
        return {
            "status": "success",
            "data": response.text,
            "status_code": response.status_code
        }
    except requests.RequestException as e:
        return {
            "status": "error",
            "error_message": str(e),
            "error_type": type(e).__name__,
            "data": None
        }


# ============================================================================
# FILE I/O NODE (Reading/writing files)
# ============================================================================

@workflow_node(
    "file_processor_node",
    dependencies=["pandas"],
    isolation_mode="none"
)
def file_processor_node(
    input_file: str, 
    output_file: str = None,
    operation: str = "read"
) -> Dict[str, Any]:
    """
    Node that handles file I/O operations.
    """
    import pandas as pd
    import os
    
    if operation == "read":
        if not os.path.exists(input_file):
            return {"error": f"File not found: {input_file}"}
        
        # Detect file type and read appropriately
        if input_file.endswith('.csv'):
            df = pd.read_csv(input_file)
        elif input_file.endswith('.json'):
            df = pd.read_json(input_file)
        else:
            return {"error": "Unsupported file type"}
        
        return {
            "data": df.to_dict(),
            "rows": len(df),
            "columns": list(df.columns),
            "file_path": input_file
        }
    
    elif operation == "write":
        # Write data to file (expects data from previous node)
        if output_file:
            # Write logic here
            return {
                "status": "written",
                "output_file": output_file
            }


# ============================================================================
# USAGE IN WORKFLOW JSON
# ============================================================================

"""
Example workflow JSON that uses these nodes:

{
  "workflow": {
    "name": "Custom Data Processing Pipeline",
    "description": "Example workflow using custom nodes"
  },
  "nodes": [
    {
      "id": "load",
      "function": "workflow_nodes.custom_node_template.file_processor_node",
      "inputs": {
        "input_file": "data/input.csv",
        "operation": "read"
      }
    },
    {
      "id": "analyze",
      "function": "workflow_nodes.custom_node_template.data_analysis_node",
      "depends_on": ["load"],
      "inputs": {
        "data": "$load.data",
        "threshold": 0.7
      }
    },
    {
      "id": "split",
      "function": "workflow_nodes.custom_node_template.split_data_node",
      "depends_on": ["load"],
      "inputs": {
        "data": "$load.data",
        "train_ratio": 0.8
      }
    },
    {
      "id": "ml_inference",
      "function": "workflow_nodes.custom_node_template.ml_inference_node",
      "depends_on": ["split"],
      "inputs": {
        "model_path": "models/my_model.h5",
        "input_data": "$split.test_data"
      }
    }
  ]
}

To run:
  python function_workflow_engine.py workflows/my_custom_workflow.json
"""
