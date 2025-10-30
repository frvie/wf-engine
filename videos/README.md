# Videos Directory

This directory contains test video files for the MP4 video detection workflow.

## Usage

Place your test video files (`.mp4`, `.avi`, `.mov`, etc.) in this directory and reference them in your workflow JSON:

```json
{
  "inputs": {
    "source": "videos/your_video.mp4"
  }
}
```

## Example Workflows

- **Webcam detection**: `workflows/video_detection.json` (uses webcam)
- **Video file detection**: `workflows/video_detection_mp4.json` (uses video files from this folder)

## Supported Formats

The workflow supports any video format that OpenCV can read:
- MP4 (H.264, H.265)
- AVI
- MOV
- MKV
- WebM

## Notes

- Video files are gitignored to keep the repository size small
- You can download sample videos or use your own test footage
- The workflow will process the video and display detections in real-time
