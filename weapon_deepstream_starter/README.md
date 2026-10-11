# Weapon Detection — DeepStream Starter Pack

**Status: configuration starter only; not yet validated in DeepStream/TensorRT.**

Candidate model in Google Drive: `WeaponDetection/best_nms_b2.onnx`
- Input shape: `[2, 3, 640, 640]`
- Output shape: `[2, 300, 6]`
- Expected rows: `[x1, y1, x2, y2, confidence, class_id]`
- Class map: `0=knife`, `1=person`, `2=weapon`

## Contents
- `config/deepstream_app_config.txt`: two-file-source reference-app template
- `config/config_infer_primary.txt`: inference configuration template
- `config/config_nvdsanalytics.txt`: example ROI template (coordinates are placeholders)
- `config/labels.txt`: labels in model class-ID order
- `src/nvdsinfer_custom_yolo_nms.cpp`: candidate custom parser for NMS-enabled output
- `VALIDATION.md`: steps and caveats for target-machine testing

## Important limitations
This is not a ready-to-run or validated deployment. DeepStream configuration property names and paths can vary by SDK release. The custom parser must be compiled against the exact installed SDK and its output dimensions verified. The ONNX model working with ONNX Runtime CUDA does not prove TensorRT can build an engine from it.

The current parser assumes output rows are float32 in network-input pixel coordinates and that the first output layer is the `[N,6]` detections tensor. Confirm this against TensorRT/DeepStream output before relying on it. If parsing or TensorRT import fails, use the raw-output `best.onnx` with a matching parser or re-export a compatible model.

Replace all `/ABSOLUTE/PATH/...` values and the sample video paths before running. Start with one local video, then two local videos, then real RTSP feeds. The analytics ROI is only an example; measure a polygon for each actual camera view.
