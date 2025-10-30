"""
Centralized logging configuration for AI Workflow Engine

This module provides a unified logging setup with proper formatters,
handlers, and log levels for the entire application.
"""

import logging
import logging.handlers
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import threading

# Global logger registry to avoid duplicate loggers
_loggers: Dict[str, logging.Logger] = {}
_logger_lock = threading.Lock()
_initialized = False

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output for console logging"""
    
    # Color codes for different log levels
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        # Add color to levelname
        if record.levelname in self.COLORS:
            colored_level = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
            record.levelname = colored_level
        
        return super().format(record)

def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[str] = None,
    console_output: bool = True,
    file_output: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> None:
    """
    Set up centralized logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (default: logs/)
        console_output: Enable console logging
        file_output: Enable file logging
        max_file_size: Maximum size per log file in bytes
        backup_count: Number of backup log files to keep
    """
    global _initialized
    
    with _logger_lock:
        if _initialized:
            return
        
        # Set up log directory
        if log_dir is None:
            log_dir = "logs"
        
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)
        
        # Get numeric log level
        numeric_level = getattr(logging, log_level.upper(), logging.INFO)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(numeric_level)
        
        # Remove existing handlers to avoid duplicates
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            fmt='%(asctime)s | %(name)-20s | %(levelname)-8s | %(funcName)-15s:%(lineno)-3d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        console_formatter = ColoredFormatter(
            fmt='%(asctime)s | %(name)-15s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        
        simple_formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        if console_output:
            # Set UTF-8 encoding for console to handle unicode characters
            import codecs
            if hasattr(sys.stdout, 'reconfigure'):
                try:
                    sys.stdout.reconfigure(encoding='utf-8')
                except Exception:
                    pass  # Ignore if reconfigure fails
            
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(numeric_level)
            console_handler.setFormatter(console_formatter)
            # Handle unicode errors gracefully
            console_handler.setStream(
                codecs.getwriter('utf-8')(sys.stdout.buffer, errors='replace')
                if hasattr(sys.stdout, 'buffer') else sys.stdout
            )
            root_logger.addHandler(console_handler)
        
        # File handlers
        if file_output:
            # Main application log
            main_log_file = log_path / "workflow.log"
            main_handler = logging.handlers.RotatingFileHandler(
                main_log_file,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            main_handler.setLevel(numeric_level)
            main_handler.setFormatter(detailed_formatter)
            root_logger.addHandler(main_handler)
            
            # Error-only log file
            error_log_file = log_path / "errors.log"
            error_handler = logging.handlers.RotatingFileHandler(
                error_log_file,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(detailed_formatter)
            root_logger.addHandler(error_handler)
            
            # Performance log (INFO and above)
            perf_log_file = log_path / "performance.log"
            perf_handler = logging.handlers.RotatingFileHandler(
                perf_log_file,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            perf_handler.setLevel(logging.INFO)
            perf_handler.setFormatter(simple_formatter)
            # Add filter to only capture performance-related logs
            perf_handler.addFilter(lambda record: 'performance' in record.name.lower() or 
                                 'fps' in record.getMessage().lower() or 
                                 'completed in' in record.getMessage().lower() or
                                 'benchmark' in record.getMessage().lower())
            root_logger.addHandler(perf_handler)
        
        _initialized = True

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name
    
    Args:
        name: Logger name (usually __name__ from the calling module)
        
    Returns:
        Logger instance
    """
    global _loggers
    
    with _logger_lock:
        if name not in _loggers:
            # Ensure logging is initialized
            if not _initialized:
                setup_logging()
            
            logger = logging.getLogger(name)
            _loggers[name] = logger
        
        return _loggers[name]

def log_system_info(logger: logging.Logger) -> None:
    """Log basic system information"""
    import platform
    import psutil
    
    logger.info("=== SYSTEM INFORMATION ===")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python: {platform.python_version()}")
    logger.info(f"CPU: {platform.processor()}")
    logger.info(f"CPU Cores: {psutil.cpu_count()} ({psutil.cpu_count(logical=False)} physical)")
    
    memory = psutil.virtual_memory()
    logger.info(f"Memory: {memory.total / (1024**3):.1f} GB total, {memory.available / (1024**3):.1f} GB available")
    
    # GPU info if available
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU: {gpu_name}")
            logger.info(f"CUDA: {torch.version.cuda}")
        else:
            logger.info("GPU: CUDA not available")
    except ImportError:
        logger.info("GPU: PyTorch not available for GPU detection")
    
    logger.info("=== END SYSTEM INFO ===")

def log_workflow_start(logger: logging.Logger, workflow_name: str, nodes_count: int) -> None:
    """Log workflow execution start"""
    logger.info("=" * 80)
    logger.info(f"STARTING WORKFLOW: {workflow_name}")
    logger.info(f"Total nodes: {nodes_count}")
    logger.info("=" * 80)

def log_workflow_end(logger: logging.Logger, workflow_name: str, total_time: float, completed_nodes: int) -> None:
    """Log workflow execution completion"""
    logger.info("\n" + "=" * 80)
    logger.info(f"WORKFLOW COMPLETED: {workflow_name}")
    logger.info(f"Total execution time: {total_time:.2f}s")
    logger.info(f"Nodes completed: {completed_nodes}")
    logger.info("=" * 80)

def log_node_start(logger: logging.Logger, node_id: str, node_type: str, execution_mode: str = "direct") -> None:
    """Log node execution start"""
    logger.info(f"Starting {node_id} ({node_type}) - {execution_mode} mode")

def log_node_success(logger: logging.Logger, node_id: str, execution_time: float) -> None:
    """Log successful node execution"""
    logger.info(f"SUCCESS {node_id}: Completed in {execution_time:.3f}s")

def log_node_error(logger: logging.Logger, node_id: str, error_msg: str) -> None:
    """Log node execution error"""
    logger.error(f"ERROR {node_id}: {error_msg}")

def log_performance_metrics(logger: logging.Logger, component: str, **metrics: Any) -> None:
    """Log performance metrics"""
    metrics_str = ", ".join([f"{k}: {v}" for k, v in metrics.items()])
    logger.info(f"PERFORMANCE {component} - {metrics_str}")

# Specialized loggers for different components
def get_workflow_logger() -> logging.Logger:
    """Get logger for workflow engine"""
    return get_logger("workflow.engine")

def get_inference_logger(backend: str) -> logging.Logger:
    """Get logger for inference backends (cpu, gpu, npu, cuda)"""
    logger = get_logger(f"workflow.inference.{backend}")
    return logger

def get_node_logger(node_type: str) -> logging.Logger:
    """Get logger for workflow nodes - reduced verbosity"""
    logger = get_logger(f"workflow.nodes.{node_type}")
    logger.setLevel(logging.WARNING)  # Reduce verbosity for workflow nodes
    return logger

def get_isolation_logger() -> logging.Logger:
    """Get logger for process isolation"""
    return get_logger("workflow.isolation")

def get_loader_logger() -> logging.Logger:
    """Get logger for workflow loader"""
    return get_logger("workflow.loader")

# Context managers for timing operations
class LoggedOperation:
    """Context manager for logging timed operations"""
    
    def __init__(self, logger: logging.Logger, operation_name: str, log_level: int = logging.INFO):
        self.logger = logger
        self.operation_name = operation_name
        self.log_level = log_level
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.log(self.log_level, f"Starting {self.operation_name}...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            if exc_type is None:
                self.logger.log(self.log_level, f"Completed {self.operation_name} in {duration:.3f}s")
            else:
                self.logger.error(f"Failed {self.operation_name} after {duration:.3f}s: {exc_val}")

# Initialize logging when module is imported
def initialize_default_logging():
    """Initialize logging with default settings"""
    setup_logging(
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        console_output=True,
        file_output=True
    )

# Auto-initialize unless NO_AUTO_INIT is set
if not os.getenv("NO_AUTO_INIT"):
    initialize_default_logging()
