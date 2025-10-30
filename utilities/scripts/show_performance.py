"""
Show Performance Metrics from Workflow Execution
"""
import json
import sys
from pathlib import Path

# Execute workflow and capture results
from function_workflow_engine import run_function_workflow

workflow_path = sys.argv[1] if len(sys.argv) > 1 else "workflows/granular_parallel_inference.json"

print("\n" + "="*70)
print("🚀 WORKFLOW PERFORMANCE ANALYSIS")
print("="*70)

# Run workflow
result = run_function_workflow(workflow_path)

# Extract performance data
if isinstance(result, dict):
    # Look for performance comparison node output
    for key, value in result.items():
        if isinstance(value, dict):
            # Check for backend results
            if 'backend_results' in value:
                print("\n📊 BACKEND PERFORMANCE COMPARISON\n")
                print(f"{'Rank':<6} {'Backend':<15} {'Avg Time':<12} {'FPS':<10} {'Speedup':<10}")
                print("-" * 70)
                
                for i, backend in enumerate(value['backend_results'], 1):
                    rank_map = {1: "🥇", 2: "🥈", 3: "🥉"}
                    rank = rank_map.get(i, f"{i}.")
                    
                    backend_name = backend['backend']
                    avg_time = backend['avg_inference_time_ms']
                    fps = backend.get('fps', 0)
                    speedup = backend.get('speedup', 'N/A')
                    
                    print(f"{rank:<6} {backend_name:<15} {avg_time:>8.2f} ms   {fps:>6.1f}     {speedup:<10}")
                
                # Summary
                summary = value.get('comparison_summary', {})
                print("\n" + "-" * 70)
                print(f"Backends Tested: {summary.get('total_backends_tested', 0)}")
                print(f"Best: {summary.get('best_performance_ms', 0):.2f} ms")
                print(f"Worst: {summary.get('worst_performance_ms', 0):.2f} ms")
                
                # Fastest backend
                fastest = value.get('fastest_backend', 'Unknown')
                print(f"\n🏆 Winner: {fastest}")
                
            # Check for individual backend summaries
            elif 'provider' in value and 'avg_inference_ms' in value:
                provider = value['provider']
                avg_time = value['avg_inference_ms']
                fps = 1000.0 / avg_time if avg_time > 0 else 0
                detections = len(value.get('detections', []))
                
                print(f"\n{provider} Backend:")
                print(f"  Avg Inference: {avg_time:.2f} ms")
                print(f"  FPS: {fps:.1f}")
                print(f"  Detections: {detections}")

print("\n" + "="*70)
