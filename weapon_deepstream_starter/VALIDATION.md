# Target GPU validation checklist

## Current status
- `best_nms_b2.onnx` was exported with input `[2,3,640,640]` and output `[2,300,6]`.
- ONNX Runtime reported CUDAExecutionProvider available on a Colab Tesla T4.
- DeepStream is not installed in the current Colab runtime.
- No DeepStream/TensorRT run has validated this configuration or parser.

## On a compatible NVIDIA system
1. Record OS, NVIDIA driver, DeepStream, CUDA and TensorRT versions.
2. Run `nvidia-smi` and `deepstream-app --version-all`.
3. Copy `best_nms_b2.onnx` and this starter directory to the target.
4. Check TensorRT can parse/build an engine from the ONNX model. An ONNX Runtime success does not guarantee TensorRT import success.
5. Build the custom parser using the sample Makefile/toolchain for that exact DeepStream SDK version.
6. Update every absolute path and both video URIs.
7. Test a single local video first with batch-size 1 if necessary; compare the same frame's detections with ONNX Runtime.
8. Test two local video files with streammux/inference batch-size 2.
9. Only then configure actual RTSP camera URIs; use live-source settings appropriate for live streams.
10. Measure GPU memory, throughput, latency, missed detections and false alarms.
11. Replace the example analytics ROI with a polygon measured for each camera. Tracker IDs are not guaranteed to persist across separate cameras.

## If it fails
- If TensorRT rejects the NMS ONNX graph, inspect the exact unsupported-op error. Consider the raw `best.onnx` plus a compatible YOLO parser, or re-export a TensorRT-compatible model.
- If parser dimensions differ, update the parser only after inspecting the actual output binding shape and data type.
- If boxes are shifted/scaled, confirm RGB/BGR ordering, normalization, letterbox/padding behavior and coordinate conventions.

This starter is for development/testing, not a certified security system. Validate on representative camera footage and keep human review for security alerts.
