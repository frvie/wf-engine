"""
Profile granular pipeline performance
"""
import time
import cv2
import numpy as np
from workflow_nodes.atomic.image_ops import (
    resize_image_letterbox_node,
    normalize_image_node,
    hwc_to_chw_node,
    add_batch_dimension_node
)
from workflow_nodes.atomic.onnx_ops import (
    create_onnx_directml_session_node,
    run_onnx_inference_single_node
)
from workflow_nodes.atomic.yolo_ops import (
    decode_yolo_v8_output_node,
    filter_by_confidence_node,
    convert_cxcywh_to_xyxy_node,
    apply_nms_node,
    scale_boxes_to_original_node,
    format_detections_coco_node
)

# Load test image
frame = cv2.imread("input/soccer.jpg")
if frame is None:
    print("Error: Could not load test image")
    exit(1)

# Create session
session_result = create_onnx_directml_session_node(
    model_path="models/yolov8s.onnx",
    device_id=1
)
session = session_result['session']
input_name = session_result['input_name']

print("Warming up...")
for _ in range(10):
    resize_result = resize_image_letterbox_node(image=frame, target_width=640, target_height=640)
    norm_result = normalize_image_node(image=resize_result['image'], scale=255.0)
    transpose_result = hwc_to_chw_node(image=norm_result['image'])
    batch_result = add_batch_dimension_node(image=transpose_result['image'])
    inference_result = run_onnx_inference_single_node(session=session, input_name=input_name, image=batch_result['image'])
    decode_result = decode_yolo_v8_output_node(outputs=inference_result['outputs'], num_classes=80)
    filter_result = filter_by_confidence_node(boxes=decode_result['boxes'], scores=decode_result['scores'], class_ids=decode_result['class_ids'], conf_threshold=0.25)
    convert_result = convert_cxcywh_to_xyxy_node(boxes=filter_result['boxes'], scores=filter_result.get('scores'), class_ids=filter_result.get('class_ids'))
    nms_result = apply_nms_node(boxes=convert_result['boxes'], scores=convert_result['scores'], class_ids=convert_result['class_ids'], iou_threshold=0.45)
    scale_result = scale_boxes_to_original_node(boxes=nms_result['boxes'], scores=nms_result.get('scores'), class_ids=nms_result.get('class_ids'), input_width=640, input_height=640, original_width=frame.shape[1], original_height=frame.shape[0], pad_x=resize_result.get('pad_x', 0), pad_y=resize_result.get('pad_y', 0), scale=resize_result.get('scale', 1.0))
    format_result = format_detections_coco_node(boxes=scale_result['boxes'], scores=scale_result['scores'], class_ids=scale_result['class_ids'])

print("\nProfiling individual operations (100 iterations)...")
iterations = 100

# Time each operation
timings = {}

start = time.perf_counter()
for _ in range(iterations):
    resize_result = resize_image_letterbox_node(image=frame, target_width=640, target_height=640)
timings['resize'] = (time.perf_counter() - start) / iterations * 1000

start = time.perf_counter()
for _ in range(iterations):
    norm_result = normalize_image_node(image=resize_result['image'], scale=255.0)
timings['normalize'] = (time.perf_counter() - start) / iterations * 1000

start = time.perf_counter()
for _ in range(iterations):
    transpose_result = hwc_to_chw_node(image=norm_result['image'])
timings['transpose'] = (time.perf_counter() - start) / iterations * 1000

start = time.perf_counter()
for _ in range(iterations):
    batch_result = add_batch_dimension_node(image=transpose_result['image'])
timings['add_batch'] = (time.perf_counter() - start) / iterations * 1000

start = time.perf_counter()
for _ in range(iterations):
    inference_result = run_onnx_inference_single_node(session=session, input_name=input_name, image=batch_result['image'])
timings['inference'] = (time.perf_counter() - start) / iterations * 1000

start = time.perf_counter()
for _ in range(iterations):
    decode_result = decode_yolo_v8_output_node(outputs=inference_result['outputs'], num_classes=80)
timings['decode'] = (time.perf_counter() - start) / iterations * 1000

start = time.perf_counter()
for _ in range(iterations):
    filter_result = filter_by_confidence_node(boxes=decode_result['boxes'], scores=decode_result['scores'], class_ids=decode_result['class_ids'], conf_threshold=0.25)
timings['filter'] = (time.perf_counter() - start) / iterations * 1000

start = time.perf_counter()
for _ in range(iterations):
    convert_result = convert_cxcywh_to_xyxy_node(boxes=filter_result['boxes'], scores=filter_result.get('scores'), class_ids=filter_result.get('class_ids'))
timings['convert'] = (time.perf_counter() - start) / iterations * 1000

start = time.perf_counter()
for _ in range(iterations):
    nms_result = apply_nms_node(boxes=convert_result['boxes'], scores=convert_result['scores'], class_ids=convert_result['class_ids'], iou_threshold=0.45)
timings['nms'] = (time.perf_counter() - start) / iterations * 1000

start = time.perf_counter()
for _ in range(iterations):
    scale_result = scale_boxes_to_original_node(boxes=nms_result['boxes'], scores=nms_result.get('scores'), class_ids=nms_result.get('class_ids'), input_width=640, input_height=640, original_width=frame.shape[1], original_height=frame.shape[0], pad_x=resize_result.get('pad_x', 0), pad_y=resize_result.get('pad_y', 0), scale=resize_result.get('scale', 1.0))
timings['scale'] = (time.perf_counter() - start) / iterations * 1000

start = time.perf_counter()
for _ in range(iterations):
    format_result = format_detections_coco_node(boxes=scale_result['boxes'], scores=scale_result['scores'], class_ids=scale_result['class_ids'])
timings['format'] = (time.perf_counter() - start) / iterations * 1000

print("\n=== TIMING BREAKDOWN (per frame) ===")
total = 0
for op, ms in sorted(timings.items(), key=lambda x: x[1], reverse=True):
    print(f"{op:15s}: {ms:6.3f} ms")
    total += ms

print(f"{'TOTAL':15s}: {total:6.3f} ms")
print(f"\nEstimated FPS: {1000/total:.1f}")
print(f"Overhead per operation: {(total - timings['inference']) / 10:.3f} ms")
