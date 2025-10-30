#!/usr/bin/env python3
"""
Workflow Data Session - Shared In-Memory Data Store
===================================================
Provides a thread-safe, namespace-organized data store for sharing data between
workflow nodes without explicit passing through inputs/outputs.

Primary Use Case: Sharing large objects (models, tensors) between nodes
- Loader node stores ONNX session → Data session (namespace: 'directml')
- Inference node retrieves session → Data session (namespace: 'directml')

Benefits:
- Avoids serialization overhead for large objects
- Simplifies workflow JSON (no need to wire every output to input)
- Namespace organization prevents key collisions
"""

from typing import Any, Dict
import threading
from src.utils.logging_config import get_node_logger


class WorkflowDataSession:
    """
    Thread-safe in-memory key-value store with namespaces.
    
    Features:
    - Thread-safe operations with locks
    - Optional namespaces for data organization
    - Simple get/set/delete/has interface
    
    Example:
        session = WorkflowDataSession()
        session.set('model', onnx_session, namespace='gpu')
        model = session.get('model', namespace='gpu')
    """
    
    def __init__(self, session_id: str = None):
        """
        Initialize a new data session.
        
        Args:
            session_id: Optional unique identifier (auto-generated if not provided)
        """
        import uuid
        self.session_id = session_id or str(uuid.uuid4())
        self.data = {}  # Storage: {full_key: value}
        self.lock = threading.Lock()  # Thread safety
        self.logger = get_node_logger("session")
        
        self.logger.info(f"📦 Created data session: {self.session_id}")
    
    def set(self, key: str, value: Any, namespace: str = None):
        """
        Store data in the session.
        
        Args:
            key: Data identifier (e.g., 'model', 'session', 'device')
            value: Any Python object (models, tensors, configs, etc.)
            namespace: Optional namespace for organization (e.g., 'gpu', 'cpu', 'npu')
        
        Note: Full key format is 'namespace.key' if namespace provided
        """
        with self.lock:
            full_key = f"{namespace}.{key}" if namespace else key
            self.data[full_key] = value
            self.logger.debug(f"📝 Session set: {full_key}")
    
    def get(self, key: str, namespace: str = None, default: Any = None):
        """
        Retrieve data from the session.
        
        Args:
            key: Data identifier
            namespace: Optional namespace
            default: Default value if key not found
            
        Returns:
            Stored value or default if not found
        """
        with self.lock:
            full_key = f"{namespace}.{key}" if namespace else key
            value = self.data.get(full_key, default)
            if value is not None:
                self.logger.debug(f"📖 Session get: {full_key}")
            return value
    
    def has(self, key: str, namespace: str = None) -> bool:
        """
        Check if a key exists in the session.
        
        Args:
            key: Data identifier
            namespace: Optional namespace
            
        Returns:
            True if key exists, False otherwise
        """
        with self.lock:
            full_key = f"{namespace}.{key}" if namespace else key
            return full_key in self.data
    
    def delete(self, key: str, namespace: str = None):
        """
        Remove data from the session.
        
        Args:
            key: Data identifier
            namespace: Optional namespace
        """
        with self.lock:
            full_key = f"{namespace}.{key}" if namespace else key
            if full_key in self.data:
                del self.data[full_key]
                self.logger.debug(f"🗑️  Session delete: {full_key}")
    
    def clear(self):
        """Clear all session data (useful for cleanup)."""
        with self.lock:
            self.data.clear()
            self.logger.info(f"🗑️  Session cleared: {self.session_id}")
    
    def get_all(self, namespace: str = None) -> Dict[str, Any]:
        """
        Get all data in a namespace or entire session.
        
        Args:
            namespace: If specified, only return keys in this namespace
            
        Returns:
            Dictionary of all matching key-value pairs
        """
        with self.lock:
            if namespace:
                prefix = f"{namespace}."
                # Return only keys in this namespace, stripped of prefix
                return {
                    k.replace(prefix, ""): v 
                    for k, v in self.data.items() 
                    if k.startswith(prefix)
                }
            return self.data.copy()
    
    def keys(self, namespace: str = None) -> list:
        """
        Get all keys in session or namespace.
        
        Args:
            namespace: If specified, only return keys in this namespace
            
        Returns:
            List of keys (namespace prefix removed if namespace specified)
        """
        with self.lock:
            if namespace:
                prefix = f"{namespace}."
                return [
                    k.replace(prefix, "") 
                    for k in self.data.keys() 
                    if k.startswith(prefix)
                ]
            return list(self.data.keys())
    
    def __repr__(self):
        """String representation for debugging."""
        return f"<WorkflowDataSession id={self.session_id} keys={len(self.data)}>"


# =============================================================================
# Global Session Management (Optional)
# =============================================================================
# Allows nodes to access sessions by ID across different parts of the application

_global_sessions = {}  # Cache: {session_id: WorkflowDataSession}
_global_lock = threading.Lock()


def get_or_create_session(session_id: str = None) -> WorkflowDataSession:
    """
    Get existing session or create a new one.
    
    Args:
        session_id: Session identifier (creates anonymous session if None)
        
    Returns:
        WorkflowDataSession instance
    """
    with _global_lock:
        if session_id and session_id in _global_sessions:
            return _global_sessions[session_id]
        
        session = WorkflowDataSession(session_id)
        if session_id:
            _global_sessions[session_id] = session
        return session


def get_session(session_id: str) -> WorkflowDataSession:
    """
    Get existing session by ID.
    
    Args:
        session_id: Session identifier
        
    Returns:
        WorkflowDataSession or None if not found
    """
    with _global_lock:
        return _global_sessions.get(session_id)


def delete_session(session_id: str):
    """
    Delete session from global cache.
    
    Args:
        session_id: Session identifier
    """
    with _global_lock:
        if session_id in _global_sessions:
            session = _global_sessions[session_id]
            session.clear()
            del _global_sessions[session_id]


def list_sessions() -> list:
    """
    List all active session IDs.
    
    Returns:
        List of session ID strings
    """
    with _global_lock:
        return list(_global_sessions.keys())


if __name__ == "__main__":
    print("🧪 WORKFLOW DATA SESSION TEST")
    print("=" * 60)
    
    # Create session
    session = WorkflowDataSession("test-session")
    
    # Store model data
    session.set("model", {"type": "yolov8s"}, namespace="cuda")
    session.set("device", "cuda:0", namespace="cuda")
    session.set("session", "onnx_session_obj", namespace="directml")
    
    # Retrieve data
    print(f"\nCUDA model: {session.get('model', namespace='cuda')}")
    print(f"CUDA device: {session.get('device', namespace='cuda')}")
    print(f"DirectML session: {session.get('session', namespace='directml')}")
    
    # Get all in namespace
    print(f"\nAll CUDA data: {session.get_all(namespace='cuda')}")
    
    # List keys
    print(f"\nAll keys: {session.keys()}")
    print(f"CUDA keys: {session.keys(namespace='cuda')}")
    
    print(f"\nSession info: {session}")
    print("\n✅ Test complete!")

