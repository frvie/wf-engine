"""
Performance Stats Node

Aggregates and compares performance across multiple inference backends.
No dependency conflicts - runs in-process.
"""

import logging
from typing import Dict, Optional
from workflow_decorator import workflow_node


@workflow_node("performance_stats", isolation_mode="auto")
def performance_stats_node(directml_result: Optional[Dict] = None, 
                          gpu_result: Optional[Dict] = None,
                          npu_result: Optional[Dict] = None, 
                          cpu_result: Optional[Dict] = None):
    """Aggregate and compare performance across all backends"""
    try:
        results = []
        
        # Process each backend result
        for backend_name, result in [
            ("DirectML", directml_result),
            ("CUDA_GPU", gpu_result), 
            ("OpenVINO_NPU", npu_result),
            ("CPU", cpu_result)
        ]:
            if result and "inference_time_ms" in result:
                detections = result.get("detections", [])
                results.append({
                    "backend": backend_name,
                    "avg_inference_time_ms": result["inference_time_ms"],
                    "total_time_ms": result["total_time_ms"],
                    "iterations": result.get("iterations", 0),
                    "detections_count": len(detections),
                    "detections": detections,
                    "provider": result.get("provider", backend_name)
                })
        
        # Sort by performance (fastest first)
        results.sort(key=lambda x: x["avg_inference_time_ms"])
        
        # Calculate performance ratios and log results
        if results:
            # Find CPU baseline for speedup calculations
            cpu_result = next((r for r in results if "CPU" in r["backend"]), None)
            baseline_time = cpu_result["avg_inference_time_ms"] if cpu_result else results[0]["avg_inference_time_ms"]
            
            logger = logging.getLogger('workflow.performance')
            logger.info(f"📊 Performance Comparison ({len(results)} backends tested, baseline: CPU):")
            
            for i, result in enumerate(results):
                result["performance_ratio"] = (result["avg_inference_time_ms"] / 
                                             baseline_time)
                result["speedup"] = (f"{baseline_time / result['avg_inference_time_ms']:.2f}x")
                fps = (1000.0 / result["avg_inference_time_ms"] 
                      if result["avg_inference_time_ms"] > 0 else 0)
                result["fps"] = fps  # Store FPS in result dict
                
                rank_map = {0: "1st", 1: "2nd", 2: "3rd"}
                rank = rank_map.get(i, f"{i+1}th")
                logger.info(
                    f"  {rank} {result['backend']}: "
                    f"{result['avg_inference_time_ms']:.1f}ms "
                    f"({fps:.1f} FPS) - {result['speedup']}"
                )
            
            # Display top detections from each backend
            logger.info("\nDetection Results:")
            for result in results:
                detections = result.get("detections", [])
                if detections:
                    # Sort by confidence and get top 5
                    top_dets = sorted(
                        detections,
                        key=lambda x: x['confidence'],
                        reverse=True
                    )[:5]
                    
                    logger.info(
                        f"\n  {result['backend']} - "
                        f"Top {len(top_dets)} of "
                        f"{result['detections_count']} detections:"
                    )
                    for idx, det in enumerate(top_dets, 1):
                        bbox = det['bbox']
                        logger.info(
                            f"    {idx}. {det['class']} "
                            f"(conf: {det['confidence']:.3f}) "
                            f"bbox: [{bbox[0]:.0f}, {bbox[1]:.0f}, "
                            f"{bbox[2]:.0f}, {bbox[3]:.0f}]"
                        )
                else:
                    logger.info(
                        f"\n  {result['backend']} - No detections"
                    )
        
        return {
            "backend_results": results,
            "fastest_backend": results[0]["backend"] if results else None,
            "comparison_summary": {
                "total_backends_tested": len(results),
                "best_performance_ms": (
                    results[0]["avg_inference_time_ms"]
                    if results else None
                ),
                "worst_performance_ms": (
                    results[-1]["avg_inference_time_ms"]
                    if results else None
                )
            }
        }
    except Exception as e:
        return {"error": f"Performance stats failed: {str(e)}"}
