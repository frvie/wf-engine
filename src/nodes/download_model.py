"""
Model Download Node

Downloads YOLOv8s PT model and converts to ONNX if not present locally.
"""

import logging
from pathlib import Path
from typing import Dict
from src.core.decorator import workflow_node


@workflow_node("download_model",
               dependencies=["ultralytics", "onnx"],
               isolation_mode="none")
def download_model_node(model_name: str = "yolov8s.onnx",
                        models_dir: str = "models") -> Dict:
    """Download YOLOv8s and export to ONNX format if not present"""
    logger = logging.getLogger('workflow.model_download')
    
    try:
        # Create models directory if it doesn't exist
        models_path = Path(models_dir)
        models_path.mkdir(exist_ok=True)
        
        model_file = models_path / model_name
        
        # Check if model already exists
        if model_file.exists():
            file_size_mb = model_file.stat().st_size / (1024 * 1024)
            logger.info(f"Model already exists: {model_file} ({file_size_mb:.1f} MB)")
            return {
                "model_path": str(model_file),
                "status": "already_exists",
                "size_mb": file_size_mb
            }
        
        # Download and convert model using ultralytics
        logger.info(f"Downloading YOLOv8s PT model and converting to ONNX...")
        
        from ultralytics import YOLO
        
        # Download the PT model (this will auto-download from ultralytics)
        logger.info("Downloading yolov8s.pt...")
        model = YOLO("yolov8s.pt")
        
        # Export to ONNX
        logger.info("Exporting to ONNX format...")
        export_path = model.export(format="onnx", imgsz=640, simplify=True, opset=12)
        
        # Move to models directory if needed
        export_file = Path(export_path)
        if export_file != model_file:
            import shutil
            shutil.move(str(export_file), str(model_file))
            logger.info(f"Moved {export_file} to {model_file}")
        
        file_size_mb = model_file.stat().st_size / (1024 * 1024)
        logger.info(f"Model ready: {model_file} ({file_size_mb:.1f} MB)")
        
        return {
            "model_path": str(model_file),
            "status": "downloaded",
            "size_mb": file_size_mb
        }
        
    except Exception as e:
        logger.error(f"Model download/conversion failed: {e}")
        return {
            "error": str(e),
            "status": "failed"
        }


