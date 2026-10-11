# Real-Time Weapon Detection and Video Analytics using YOLO and NVIDIA DeepStream

An AI-based computer vision proof of concept for detecting weapons and people in surveillance footage, tracking detections across video frames, and preparing a multi-camera video analytics pipeline for factory security monitoring.

## Overview

The objective of this project is to develop a weapon-detection system for a factory entrance monitored by two CCTV cameras. The proposed system detects potentially dangerous objects, tracks detections across consecutive frames, and records information that can support security-event monitoring.

The project focuses on two main areas:

1. **Model development:** Dataset preparation, YOLO11n training, model evaluation, video inference, and object tracking.
2. **Deployment preparation:** ONNX export, GPU inference experiments, and preparation for NVIDIA DeepStream integration.

Since actual factory CCTV footage and RTSP streams were not available during development, the initial implementation uses a public weapon-detection dataset and recorded-video experiments. The model is a prototype and requires further validation before production deployment.

## Problem Statement

Factory security systems may need to identify weapons at entrances while monitoring multiple camera feeds. A practical solution must balance detection quality, inference latency, tracking reliability, and false-alarm reduction.

The proposed system aims to:

- Detect weapons and people in video frames.
- Generate bounding boxes, class labels, and confidence scores.
- Track detections across successive frames.
- Record detection timestamps and object information.
- Support future region-based analytics and security alerts.
- Prepare for multi-camera inference using NVIDIA DeepStream.

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Object detection | Ultralytics YOLO11n |
| Video processing | OpenCV |
| Object tracking | ByteTrack |
| Model export | ONNX |
| ONNX inference | ONNX Runtime |
| GPU experimentation | Google Colab, NVIDIA Tesla T4 |
| Deployment target | NVIDIA DeepStream and TensorRT |
| Event logging | CSV |

## System Architecture

### Current Prototype

```text
Public Weapon Dataset
        |
        v
Dataset Preparation and Annotation Checks
        |
        v
YOLO11n Training
        |
        v
Trained Model (best.pt)
        |
        +----------------------+
        |                      |
        v                      v
  Video Detection        ONNX Export
        |                      |
        v                      v
 ByteTrack Tracking      ONNX Runtime Tests
        |                      |
        v                      v
 Annotated Video        Sample Predictions
        |
        v
 CSV Detection Logs
```

### Proposed Production Architecture

```text
        CCTV Camera 1          CCTV Camera 2
              |                      |
             RTSP                   RTSP
              |                      |
              +----------+-----------+
                         |
                         v
                 NVIDIA DeepStream
                         |
                     nvstreammux
                         |
                      nvinfer
                         |
                     nvtracker
                         |
                   nvdsanalytics
                         |
                  Custom Alert Logic
                         |
             +-----------+-----------+
             |                       |
             v                       v
       Event Logging           Operator Display
```

The production architecture is a planned integration design, not a claim that the complete DeepStream pipeline has been executed.

## Dataset Preparation

The project started with a public weapon-detection dataset converted into YOLO-compatible annotation format.

Each object annotation follows this structure:

```text
class_id x_center y_center width height
```

The coordinates are normalized relative to the image dimensions.

The original training dataset used these class IDs:

| Class ID | Class name |
|---|---|
| 0 | GUN |
| 1 | KNIFE |
| 2 | PERSON |

**Dataset limitations discovered during development:**

- The training data had a substantial class imbalance.
- Only seven knife instances were counted in the original training split.
- Some bounding-box annotations appeared questionable and required further inspection.
- The validation and test splits had inconsistent class distributions.
- The recorded test split contained gun instances only, so it could not establish performance across all target classes.

These findings highlighted the importance of annotation quality, representative data, and class-wise evaluation before deployment.

> Note: The class names and class IDs of the final checkpoint must be verified from the model itself. They may differ from the original dataset mapping after conversion or retraining.

## Model Training

A lightweight YOLO11n model was used as the initial detector because real-time surveillance requires bounding-box predictions with practical inference latency.

The initial baseline was trained locally for 10 epochs. Training took approximately 5.35 hours on a CPU-only laptop.

Because the laptop did not provide an available NVIDIA CUDA GPU, Google Colab with an NVIDIA Tesla T4 GPU was used for subsequent GPU-related experiments.

The resulting trained checkpoint was saved as `best.pt`.

The initial training run is a baseline rather than evidence of production-level accuracy. Further training and evaluation should follow dataset validation and correction.

## Video Detection

OpenCV was used to read recorded video frames, and the trained YOLO model was used to perform frame-by-frame inference.

The detection workflow records:

- Frame number
- Timestamp in seconds
- Predicted class
- Confidence score
- Bounding-box coordinates

Detection records are exported to CSV for inspection and further analysis.

Example CSV columns:

```text
timestamp_seconds,frame_number,class_name,confidence,x1,y1,x2,y2
```

The recorded video can be used to examine detection behavior over time before live CCTV streams become available.

## Object Tracking with ByteTrack

Ultralytics tracking was used to associate detections across consecutive video frames using ByteTrack.

The tracking workflow generates an annotated video and a CSV containing available tracking IDs.

Example CSV columns:

```text
frame,timestamp_seconds,track_id,class_name,confidence,x1,y1,x2,y2
```

Tracking IDs help associate detections across frames when tracking succeeds. They do not guarantee permanent object identity across long gaps, occlusion, or tracking failures.

Repeated rows for the same tracked object are detection records, not necessarily separate security incidents.

## ONNX Export and Inference

The trained PyTorch checkpoint was exported to ONNX to test model inference independently of the original training workflow and prepare for deployment integration.

The ONNX model was tested using ONNX Runtime. A sample image produced a weapon-class detection with approximately 86.2% confidence.

This value represents the confidence score for one prediction, not the overall accuracy of the model.

### NMS-Enabled Export

Non-Maximum Suppression (NMS) removes redundant overlapping bounding boxes associated with the same predicted object.

An additional ONNX model was exported with NMS enabled. The observed output changed from 8,400 candidate predictions to a maximum of 300 detection rows per image.

The batch-size-two export was verified with the following shapes:

| Tensor | Shape |
|---|---|
| Input | `[2, 3, 640, 640]` |
| Output | `[2, 300, 6]` |

The six output values were observed in the format:

```text
x1, y1, x2, y2, confidence, class_id
```

The batch-size-two export was explored for the intended two-camera use case. However, a batch dimension of two does not by itself implement or validate a two-camera video pipeline.

The inference configuration and output parser must match the exported model's actual output format.

## NVIDIA DeepStream Integration Plan

NVIDIA DeepStream is the intended deployment framework for real-time video analytics using NVIDIA hardware.

The planned pipeline includes:

| Component | Purpose |
|---|---|
| `nvstreammux` | Batches frames from multiple video sources |
| `nvinfer` | Runs inference using a compatible configured model and inference engine |
| `nvtracker` | Associates detections across frames |
| `nvdsanalytics` | Supports configured region, direction, and line-crossing analytics |
| `nvosd` | Draws bounding boxes and labels on video output |
| Custom application logic | Applies security-event rules and records alerts |

The ONNX model is intended to be integrated through a compatible inference configuration. TensorRT engine generation, model output decoding, NMS handling, and DeepStream compatibility must be verified on a supported NVIDIA system.

### Current Integration Status

The ONNX model export and inference experiments have been completed, and DeepStream configuration and candidate parser templates have been prepared.

The complete DeepStream pipeline has not yet been executed end to end. Actual RTSP integration, TensorRT compatibility, multi-camera operation, analytics rules, and the final alert workflow remain to be validated.

## Security Event Design

The intended system will combine detections, tracking, spatial association, and temporal confirmation before generating a security event.

A future event record could contain:

```text
Camera ID
Person Track ID
Weapon Class
Detection Confidence
First Detection Timestamp
Confirmation Timestamp
Snapshot or Clip Reference
```

For example, a potential event could be raised when a weapon-class detection persists above a validated confidence threshold within a monitored region.

This is a proposed event format. A reliable person-to-weapon association system and production alert rules still need implementation and validation.

A single-frame detection should not automatically be treated as a confirmed security incident.

## Challenges and Lessons Learned

- **CPU training time:** Training locally was slow, so GPU-based experiments were moved to Google Colab.
- **Class imbalance:** The small number of knife examples highlighted the need for better class coverage.
- **Annotation quality:** Bounding-box inspection showed why dataset validation matters before further training.
- **Model output handling:** The ONNX export required inspection to understand raw predictions and NMS-enabled outputs.
- **Deployment compatibility:** Exporting a model does not automatically make it compatible with TensorRT or DeepStream.
- **Real-world domain gap:** Public datasets cannot fully represent the camera angles, lighting, occlusion, and motion blur of a specific factory.

## Current Status

### Implemented and tested

- YOLO11n baseline training.
- Initial dataset and annotation inspection.
- Recorded-video detection experiments.
- ByteTrack-based tracking experiments.
- CSV logging of detection and tracking records.
- ONNX export and sample ONNX Runtime inference.
- NMS-enabled ONNX export.
- Batch-size-two ONNX input/output shape verification.

### Prepared or investigated

- GPU inference experiments on Google Colab.
- DeepStream multi-camera architecture.
- DeepStream inference and analytics configuration templates.
- Candidate custom parser for the exported model's output.

### Pending validation

- Dataset correction and representative class-wise evaluation.
- Reliable gun, knife, and person detection on representative footage.
- TensorRT engine compatibility and performance.
- Custom parser compilation and output verification.
- Full DeepStream pipeline execution.
- Two-camera RTSP integration.
- Person-to-weapon association and validated temporal alert rules.
- End-to-end latency, throughput, false-positive, and false-negative measurements.

## Future Improvements

1. Improve dataset quality and class balance.
2. Evaluate precision, recall, mAP, and per-class performance on representative test data.
3. Collect and annotate footage from the intended factory cameras.
4. Fine-tune the detector using factory-specific examples.
5. Validate ONNX outputs against the TensorRT implementation.
6. Complete DeepStream integration and test two video sources.
7. Implement reliable tracking, spatial association, and temporal confirmation.
8. Measure throughput, end-to-end latency, GPU utilization, false alarms, and missed detections.
9. Add camera-disconnection handling, event persistence, and operator review.
10. Deploy only after the system has been validated under realistic operating conditions.

## Conclusion

This project explores the development of a weapon-detection and video-analytics proof of concept using YOLO11n, OpenCV, ByteTrack, ONNX, and NVIDIA deployment technologies.

The work progressed from dataset preparation and baseline training to recorded-video experiments, tracking, ONNX export, and preparation for DeepStream integration.

The main technical focus going forward is improving data quality, establishing reliable detection performance, and validating the complete multi-camera inference and alert pipeline on supported NVIDIA hardware.

**Project status:** Prototype and deployment preparation; production readiness has not yet been established.
