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
            if result:
                # Handle nested performance dict (from create_detection_summary_node)
                if "performance" in result and isinstance(result["performance"], dict):
                    perf = result["performance"]
                    inference_time = perf.get("inference_time_ms", perf.get("avg_time_ms", 0))
                    fps_from_result = perf.get("fps", 0)
                    iterations = perf.get("iterations", 0)
                    total_time = perf.get("total_time_ms", inference_time)
                    provider = perf.get("provider", backend_name)
                    num_detections = result.get("num_detections", 0)
                # Handle flat dict (legacy format)
                elif "inference_time_ms" in result:
                    inference_time = result["inference_time_ms"]
                    fps_from_result = result.get("fps", 0)
                    iterations = result.get("iterations", 0)
                    total_time = result.get("total_time_ms", inference_time)
                    provider = result.get("provider", backend_name)
                    # Handle both detection count (int) and detection list
                    detections = result.get("detections", [])
                    if isinstance(detections, int):
                        num_detections = detections
                    elif isinstance(detections, list):
                        num_detections = len(detections)
                    else:
                        num_detections = 0
                else:
                    continue
                    
                results.append({
                    "backend": backend_name,
                    "avg_inference_time_ms": inference_time,
                    "fps": fps_from_result,
                    "total_time_ms": total_time,
                    "iterations": iterations,
                    "detections_count": num_detections,
                    "provider": provider
                })
        
        # Sort by performance (fastest first)
        results.sort(key=lambda x: x["avg_inference_time_ms"])
        
        # Calculate performance ratios and log results
        if results:
            # Find CPU baseline for speedup calculations
            cpu_result = next((r for r in results if "CPU" in r["backend"]), None)
            baseline_time = cpu_result["avg_inference_time_ms"] if cpu_result else results[0]["avg_inference_time_ms"]
            
            logger = logging.getLogger('workflow.inference.performance_stats')
            logger.info("=" * 80)
            logger.info(f"📊 PERFORMANCE COMPARISON ({len(results)} backends tested)")
            logger.info("=" * 80)
            
            for i, result in enumerate(results):
                result["performance_ratio"] = (result["avg_inference_time_ms"] / baseline_time)
                speedup = baseline_time / result['avg_inference_time_ms']
                result["speedup"] = f"{speedup:.2f}x"
                
                # Use FPS from result if available, otherwise calculate
                fps = result.get("fps", 0)
                if fps == 0 and result["avg_inference_time_ms"] > 0:
                    fps = 1000.0 / result["avg_inference_time_ms"]
                result["fps"] = fps
                
                rank_map = {0: "🥇 1st", 1: "🥈 2nd", 2: "🥉 3rd"}
                rank = rank_map.get(i, f"  {i+1}th")
                logger.info(
                    f"{rank} {result['backend']:15s}: "
                    f"{result['avg_inference_time_ms']:6.1f}ms | "
                    f"{fps:6.1f} FPS | "
                    f"Speedup: {result['speedup']:>6s} | "
                    f"{result['detections_count']} detections"
                )
            
            logger.info("=" * 80)
        
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
