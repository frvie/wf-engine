"""
SharedMemory Utilities for Process Isolation
=============================================
Provides zero-copy inter-process communication (IPC) for isolated workflow nodes.

Key Features:
- Flag-based synchronization (no SyncManager required)
- Works across different Python executables and virtual environments
- Header-based protocol for safe data transfer
- Support for numpy arrays, dicts, and pickled objects

Architecture:
- HEADER (8 bytes): FLAG (4 bytes) + RESERVED (4 bytes)
- DATA: Variable length payload after header
- FLAGS: EMPTY(0), READY(1), PROCESSING(2), DONE(3), ERROR(99)

Usage Pattern:
1. Main process creates SharedMemory with FLAG_EMPTY
2. Main process writes data and sets FLAG_READY
3. Subprocess attaches, waits for FLAG_READY, reads data
4. Subprocess writes result and sets FLAG_DONE
5. Main process waits for FLAG_DONE, reads result
6. Both processes close SharedMemory
7. Main process unlinks (deletes) SharedMemory

Critical: Copy bytes before closing to avoid BufferError!
"""

import numpy as np
from multiprocessing.shared_memory import SharedMemory
import struct
import time
from typing import Dict, Any, Tuple, Optional
import pickle
from src.utils.logging_config import get_logger

logger = get_logger('shared_memory')


# =============================================================================
# Synchronization Flags
# =============================================================================
FLAG_EMPTY = 0       # Shared memory created but not ready
FLAG_READY = 1       # Data ready for consumer
FLAG_PROCESSING = 2  # Consumer is processing (optional)
FLAG_DONE = 3        # Consumer finished processing
FLAG_ERROR = 99      # Error occurred

HEADER_SIZE = 8  # 4 bytes for flag + 4 bytes reserved for future use

# =============================================================================
# Flag Operations
# =============================================================================

def set_flag(buf, value: int):
    """Write synchronization flag to shared memory buffer header."""
    struct.pack_into("i", buf, 0, value)


def get_flag(buf) -> int:
    """Read synchronization flag from shared memory buffer header."""
    return struct.unpack_from("i", buf, 0)[0]


def wait_for_flag(buf, expected_value: int, timeout: float = 30.0) -> bool:
    """
    Wait for flag to reach expected value (polling with 1ms sleep).
    
    Args:
        buf: Shared memory buffer
        expected_value: Expected flag value (FLAG_READY, FLAG_DONE, etc.)
        timeout: Maximum time to wait in seconds
        
    Returns:
        True if flag reached expected value, False if timeout
    """
    start = time.time()
    while get_flag(buf) != expected_value:
        time.sleep(0.001)  # 1ms sleep to avoid busy-waiting (CPU-friendly)
        if time.time() - start > timeout:
            return False
    return True


def create_shared_memory_with_header(
    name: str,
    data_size: int
) -> Tuple[SharedMemory, memoryview]:
    """
    Create shared memory block with header for synchronization.
    
    Args:
        name: Name of the shared memory block
        data_size: Size of data area in bytes (header is additional)
        
    Returns:
        Tuple of (SharedMemory instance, data buffer view)
    """
    total_size = HEADER_SIZE + data_size
    
    try:
        shm = SharedMemory(name=name, create=True, size=total_size)
        logger.debug(
            f"Created shared memory '{name}' "
            f"({total_size} bytes: {HEADER_SIZE} header + {data_size} data)"
        )
    except FileExistsError:
        # Clean up existing block and recreate
        logger.warning(f"Shared memory '{name}' exists, recreating")
        old = SharedMemory(name=name)
        old.close()
        old.unlink()
        shm = SharedMemory(name=name, create=True, size=total_size)
        logger.debug(
            f"Recreated shared memory '{name}' "
            f"({total_size} bytes: {HEADER_SIZE} header + {data_size} data)"
        )
    
    # Initialize flag to EMPTY
    set_flag(shm.buf, FLAG_EMPTY)
    
    # Return data buffer (after header)
    data_buf = shm.buf[HEADER_SIZE:]
    return shm, data_buf


def attach_shared_memory_with_header(
    name: str
) -> Tuple[SharedMemory, memoryview]:
    """
    Attach to existing shared memory block with header.
    
    Args:
        name: Name of the shared memory block
        
    Returns:
        Tuple of (SharedMemory instance, data buffer view)
    """
    shm = SharedMemory(name=name)
    logger.debug(f"Attached to shared memory '{name}'")
    
    # Return data buffer (after header)
    data_buf = shm.buf[HEADER_SIZE:]
    return shm, data_buf


def create_shared_memory(name: str, size: int) -> SharedMemory:
    """
    Create or recreate shared memory block.
    
    Args:
        name: Name of the shared memory block
        size: Size in bytes
        
    Returns:
        SharedMemory instance
    """
    try:
        shm = SharedMemory(name=name, create=True, size=size)
        logger.debug(f"Created shared memory '{name}' ({size} bytes)")
        return shm
    except FileExistsError:
        # Clean up existing block and recreate
        logger.warning(f"Shared memory '{name}' exists, recreating")
        old = SharedMemory(name=name)
        old.close()
        old.unlink()
        shm = SharedMemory(name=name, create=True, size=size)
        logger.debug(f"Recreated shared memory '{name}' ({size} bytes)")
        return shm


def attach_shared_memory(name: str) -> SharedMemory:
    """
    Attach to existing shared memory block.
    
    Args:
        name: Name of the shared memory block
        
    Returns:
        SharedMemory instance
    """
    shm = SharedMemory(name=name)
    logger.debug(f"Attached to shared memory '{name}'")
    return shm


def numpy_to_shared_memory_with_header(
    arr: np.ndarray,
    name: str
) -> Tuple[SharedMemory, Dict[str, Any]]:
    """
    Store numpy array in shared memory with synchronization header.
    
    Args:
        arr: Numpy array to share
        name: Name for the shared memory block
        
    Returns:
        Tuple of (SharedMemory instance, metadata dict)
    """
    data_size = arr.nbytes
    shm, data_buf = create_shared_memory_with_header(name, data_size)
    
    # Copy array data to shared memory (after header)
    shared_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=data_buf)
    shared_arr[:] = arr[:]
    
    # Signal ready
    set_flag(shm.buf, FLAG_READY)
    
    metadata = {
        "shm_name": name,
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "nbytes": data_size
    }
    
    logger.debug(f"Stored array {arr.shape} {arr.dtype} in shared memory")
    return shm, metadata


def numpy_from_shared_memory_with_header(
    metadata: Dict[str, Any],
    wait_for_ready: bool = True,
    timeout: float = 30.0
) -> Tuple[SharedMemory, np.ndarray]:
    """
    Load numpy array from shared memory with synchronization header.
    
    Args:
        metadata: Dictionary with shm_name, shape, dtype
        wait_for_ready: Whether to wait for FLAG_READY
        timeout: Maximum time to wait for ready signal
        
    Returns:
        Tuple of (SharedMemory instance, numpy array view)
    """
    shm, data_buf = attach_shared_memory_with_header(metadata["shm_name"])
    
    # Wait for ready signal if requested
    if wait_for_ready:
        if not wait_for_flag(shm.buf, FLAG_READY, timeout):
            raise TimeoutError(
                f"Timeout waiting for shared memory "
                f"'{metadata['shm_name']}' to be ready"
            )
    
    arr = np.ndarray(
        metadata["shape"],
        dtype=np.dtype(metadata["dtype"]),
        buffer=data_buf
    )
    logger.debug(
        f"Loaded array {metadata['shape']} "
        f"{metadata['dtype']} from shared memory"
    )
    return shm, arr


def pickle_to_shared_memory(
    obj: Any,
    name: str
) -> Tuple[SharedMemory, Dict[str, Any]]:
    """
    Pickle an object and store in shared memory.
    
    Useful for ONNX session objects or other complex data.
    
    Args:
        obj: Object to pickle and share
        name: Name for the shared memory block
        
    Returns:
        Tuple of (SharedMemory instance, metadata dict)
    """
    pickled = pickle.dumps(obj)
    nbytes = len(pickled)
    
    shm = create_shared_memory(name, nbytes)
    shm.buf[:nbytes] = pickled
    
    metadata = {
        "shm_name": name,
        "nbytes": nbytes,
        "type": "pickle"
    }
    
    logger.debug(f"Stored pickled object ({nbytes} bytes) in shared memory")
    return shm, metadata


def pickle_from_shared_memory(metadata: Dict[str, Any]) -> Any:
    """
    Load pickled object from shared memory.
    
    Args:
        metadata: Dictionary with shm_name, nbytes
        
    Returns:
        Unpickled object
    """
    shm = attach_shared_memory(metadata["shm_name"])
    pickled = bytes(shm.buf[:metadata["nbytes"]])
    obj = pickle.loads(pickled)
    logger.debug("Loaded pickled object from shared memory")
    return obj


def dict_to_shared_memory_with_header(
    data: Dict[str, Any],
    name: str
) -> Tuple[SharedMemory, Dict[str, Any]]:
    """
    Store dictionary (model config) in shared memory with synchronization.
    
    Args:
        data: Dictionary to share
        name: Name for the shared memory block
        
    Returns:
        Tuple of (SharedMemory instance, metadata dict)
    """
    # Pickle the dictionary
    data_bytes = pickle.dumps(data)
    data_size = len(data_bytes)
    
    shm, data_buf = create_shared_memory_with_header(name, data_size)
    
    # Copy data to shared memory
    data_buf[:data_size] = data_bytes
    
    # Signal ready
    set_flag(shm.buf, FLAG_READY)
    
    metadata = {
        "shm_name": name,
        "data_size": data_size,
        "type": "dict"
    }
    
    logger.debug(f"Stored dict ({data_size} bytes) in shared memory '{name}'")
    return shm, metadata


def dict_from_shared_memory_with_header(
    metadata: Dict[str, Any],
    wait_for_ready: bool = True,
    timeout: float = 30.0
) -> Tuple[SharedMemory, Dict[str, Any]]:
    """
    Load dictionary from shared memory with synchronization.
    
    Args:
        metadata: Dictionary with shm_name, data_size
        wait_for_ready: Whether to wait for FLAG_READY
        timeout: Maximum time to wait for ready signal
        
    Returns:
        Tuple of (SharedMemory instance, dictionary)
    """
    shm, data_buf = attach_shared_memory_with_header(metadata["shm_name"])
    
    # Wait for ready signal if requested
    if wait_for_ready:
        if not wait_for_flag(shm.buf, FLAG_READY, timeout):
            raise TimeoutError(
                f"Timeout waiting for shared memory "
                f"'{metadata['shm_name']}' to be ready"
            )
    
    # Read and unpickle data
    data_size = metadata.get('data_size')
    data_bytes = bytes(data_buf[:data_size])
    data = pickle.loads(data_bytes)
    
    logger.debug(f"Loaded dict from shared memory '{metadata['shm_name']}'")
    return shm, data


def cleanup_shared_memory(name: str):
    """
    Clean up shared memory block.
    
    Args:
        name: Name of the shared memory block to clean up
    """
    try:
        shm = SharedMemory(name=name)
        shm.close()
        shm.unlink()
        logger.debug(f"Cleaned up shared memory '{name}'")
    except FileNotFoundError:
        logger.debug(f"Shared memory '{name}' already cleaned up")
    except Exception as e:
        logger.warning(f"Failed to cleanup shared memory '{name}': {e}")

