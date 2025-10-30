"""
Quick Launcher for Video Object Detection Workflow

Easily configure and run video object detection with DirectML GPU acceleration.
"""

import subprocess
import json
import sys
from pathlib import Path


def main():
    print("=" * 70)
    print("VIDEO OBJECT DETECTION with DirectML GPU Acceleration")
    print("=" * 70)
    print()
    
    # Configuration options
    print("Configuration:")
    print("1. Webcam (default)")
    print("2. Video file")
    print()
    
    choice = input("Select video source [1]: ").strip() or "1"
    
    if choice == "2":
        video_path = input("Enter video file path: ").strip()
        if not Path(video_path).exists():
            print(f"ERROR: File not found: {video_path}")
            return
        source = video_path
    else:
        source = "0"
    
    # Ask for max frames
    max_frames_input = input("Maximum frames to process [300]: ").strip() or "300"
    try:
        max_frames = int(max_frames_input)
    except ValueError:
        print("WARNING: Invalid number, using default 300")
        max_frames = 300
    
    # Ask for confidence threshold
    conf_input = input("Confidence threshold (0.0-1.0) [0.25]: ").strip() or "0.25"
    try:
        conf_threshold = float(conf_input)
        if not 0 <= conf_threshold <= 1:
            raise ValueError
    except ValueError:
        print("WARNING: Invalid threshold, using default 0.25")
        conf_threshold = 0.25
    
    # Ask for display option
    display_input = input("Display frames on screen? (y/n) [y]: ").strip().lower() or "y"
    display = display_input == "y"
    
    # Ask for save option
    save_input = input("Save output video? (y/n) [y]: ").strip().lower() or "y"
    save_video = save_input == "y"
    
    if save_video:
        output_path = input("Output video path [output_detections.mp4]: ").strip() or "output_detections.mp4"
    else:
        output_path = "output_detections.mp4"
    
    print()
    print("=" * 70)
    print("Configuration Summary:")
    print(f"  Video Source: {source}")
    print(f"  Max Frames: {max_frames}")
    print(f"  Confidence Threshold: {conf_threshold}")
    print(f"  Display: {display}")
    print(f"  Save Video: {save_video}")
    if save_video:
        print(f"  Output Path: {output_path}")
    print("=" * 70)
    print()
    
    # Load workflow template
    workflow_path = Path("workflows/video_detection_directml.json")
    
    if not workflow_path.exists():
        print(f"ERROR: Workflow file not found: {workflow_path}")
        return
    
    with open(workflow_path, 'r') as f:
        workflow = json.load(f)
    
    # Update workflow with user settings
    for node in workflow["nodes"]:
        if node["id"] == "video_stream":
            node["inputs"]["source"] = source
            node["inputs"]["max_frames"] = max_frames
        elif node["id"] == "batch_inference":
            node["inputs"]["conf_threshold"] = conf_threshold
        elif node["id"] == "visualize":
            node["inputs"]["display"] = display
            node["inputs"]["save_video"] = save_video
            node["inputs"]["output_path"] = output_path
    
    # Save temporary workflow
    temp_workflow_path = Path("workflows/video_detection_temp.json")
    with open(temp_workflow_path, 'w') as f:
        json.dump(workflow, f, indent=2)
    
    print(">> Starting workflow engine...")
    print()
    
    # Run workflow
    try:
        result = subprocess.run(
            [sys.executable, "function_workflow_engine.py", str(temp_workflow_path)],
            check=True
        )
        
        print()
        print("=" * 70)
        print("SUCCESS: Workflow completed successfully!")
        
        if save_video:
            print(f"OUTPUT: Video saved to: {output_path}")
        
        print("=" * 70)
        
    except subprocess.CalledProcessError as e:
        print()
        print("=" * 70)
        print(f"ERROR: Workflow failed with exit code: {e.returncode}")
        print("=" * 70)
    except KeyboardInterrupt:
        print()
        print("=" * 70)
        print("WARNING: Workflow interrupted by user")
        print("=" * 70)
    finally:
        # Clean up temp file
        if temp_workflow_path.exists():
            temp_workflow_path.unlink()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
