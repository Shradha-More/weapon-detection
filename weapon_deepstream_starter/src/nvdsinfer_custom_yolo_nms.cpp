/*
 * Candidate custom parser for an NMS-enabled ONNX output shaped [N, 6].
 * Each row is assumed to be float32: x1, y1, x2, y2, score, class_id.
 * Compile against the exact target DeepStream SDK headers/libraries.
 * Verify output dimensions and coordinate scaling before deployment.
 */
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
#include "nvdsinfer_custom_impl.h"

extern "C" bool NvDsInferParseCustomYoloNms(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferObjectDetectionInfo> &objectList)
{
    if (outputLayersInfo.empty() || outputLayersInfo[0].buffer == nullptr) {
        std::cerr << "YoloNms parser: missing output layer/buffer" << std::endl;
        return false;
    }
    const NvDsInferLayerInfo &layer = outputLayersInfo[0];
    if (layer.inferDims.numDims < 2 || layer.inferDims.d[1] != 6) {
        std::cerr << "YoloNms parser: expected [N,6] output, verify TensorRT output dimensions" << std::endl;
        return false;
    }
    const unsigned int rows = layer.inferDims.d[0];
    const float *data = static_cast<const float *>(layer.buffer);
    const float threshold = 0.25f;

    for (unsigned int i = 0; i < rows; ++i) {
        const float x1 = data[i * 6 + 0];
        const float y1 = data[i * 6 + 1];
        const float x2 = data[i * 6 + 2];
        const float y2 = data[i * 6 + 3];
        const float score = data[i * 6 + 4];
        const float classFloat = data[i * 6 + 5];
        if (!std::isfinite(score) || score < threshold || !std::isfinite(classFloat)) continue;
        const int classId = static_cast<int>(std::round(classFloat));
        if (classId < 0 || classId >= static_cast<int>(detectionParams.numClassesConfigured)) continue;

        const float left = std::clamp(x1, 0.0f, static_cast<float>(networkInfo.width));
        const float top = std::clamp(y1, 0.0f, static_cast<float>(networkInfo.height));
        const float right = std::clamp(x2, 0.0f, static_cast<float>(networkInfo.width));
        const float bottom = std::clamp(y2, 0.0f, static_cast<float>(networkInfo.height));
        if (right <= left || bottom <= top) continue;

        NvDsInferObjectDetectionInfo obj{};
        obj.classId = classId;
        obj.detectionConfidence = score;
        obj.left = left;
        obj.top = top;
        obj.width = right - left;
        obj.height = bottom - top;
        objectList.push_back(obj);
    }
    return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloNms);
