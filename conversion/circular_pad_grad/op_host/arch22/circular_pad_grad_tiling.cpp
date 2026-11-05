/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file circular_pad_grad_tiling.cpp
 * \brief
 */

#include <string>
#include <vector>

#include "circular_pad_grad_tiling.h"
#include "log/log.h"
#include "register/op_def_registry.h"
#include "op_host/math_tiling_templates_registry.h"
namespace optiling {
static void ReportTilingFailure(const char* opName, const char* reason)
{
    const std::string opNameStr(opName);
    const std::string reasonStr(reason);
    const std::string msg = "Failed to execute operator " + opNameStr + ". Reason: " + reasonStr + ".";
    const std::vector<const char*> msgKey = {"opName", "reason"};
    const std::vector<const char*> msgValue = {opNameStr.c_str(), reasonStr.c_str()};
    OP_LOGE_WITHOUT_REPORT(opNameStr.c_str(), "%s", msg.c_str());
    REPORT_PREDEFINED_ERR_MSG("EZ1008", msgKey, msgValue);
}
constexpr int32_t ALIGN = 32;
constexpr int32_t WTYPE_SIZE = 4;
constexpr int64_t BUFFER_NUM = 2;
constexpr int64_t DIM_1 = 1;
constexpr int64_t DIM_2 = 2;
constexpr int64_t DIM_3 = 3;
constexpr int64_t TYPE_MODE1 = 1;
constexpr int64_t TYPE_MODE2 = 2;
constexpr int64_t TYPE_MODE3 = 3;

// CircularPadGradTiling
ge::graphStatus CircularPadGradTiling::CheckLeftAndRight()
{
    std::string shapeMsg = "left: " + std::to_string(left) + ", right: " + std::to_string(right) +
                           ", x's W: " + std::to_string(inputW);
    if (left > 0 && right > 0 && (left + right > inputW * DIM_2 / DIM_3)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context_->GetNodeName(), "left, right and x's W", shapeMsg.c_str(),
            "left + right should not be greater than inputW * 2 / 3 when left and right are greater than 0.");
        return ge::GRAPH_FAILED;
    } else if (left > 0 && right <= 0 && (left > inputW / DIM_2)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context_->GetNodeName(), "left, right and x's W", shapeMsg.c_str(),
            "left should not be greater than inputW / 2 when left is greater than 0 and right is not greater than 0.");
        return ge::GRAPH_FAILED;
    } else if (left <= 0 && right > 0 && (right > inputW / DIM_2)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context_->GetNodeName(), "left, right and x's W", shapeMsg.c_str(),
            "right should not be greater than inputW / 2 when right is greater than 0 and left is not greater than 0.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CircularPadGradTiling::CheckTopAndBottom()
{
    std::string shapeMsg = "top: " + std::to_string(top) + ", bottom: " + std::to_string(bottom) +
                           ", x's H: " + std::to_string(inputH);
    if (top > 0 && bottom > 0 && (top + bottom > inputH * DIM_2 / DIM_3)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context_->GetNodeName(), "top, bottom and x's H", shapeMsg.c_str(),
            "top + bottom should not be greater than inputH * 2 / 3 when top and bottom are greater than 0.");
        return ge::GRAPH_FAILED;
    } else if (top > 0 && bottom <= 0 && (top > inputH / DIM_2)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context_->GetNodeName(), "top, bottom and x's H", shapeMsg.c_str(),
            "top should not be greater than inputH / 2 when top is greater than 0 and bottom is not greater than 0.");
        return ge::GRAPH_FAILED;
    } else if (top <= 0 && bottom > 0 && (bottom > inputH / DIM_2)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "top, bottom and x's H", shapeMsg.c_str(),
                                               "bottom should not be greater than inputH / 2 when bottom is greater "
                                               "than 0 and top is not greater than 0.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CircularPadGradTiling::CheckFrontAndBack()
{
    std::string shapeMsg = "front: " + std::to_string(front) + ", back: " + std::to_string(back) +
                           ", x's D: " + std::to_string(inputL);
    if (front > 0 && back > 0 && (front + back > inputL * DIM_2 / DIM_3)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context_->GetNodeName(), "front, back and x's D", shapeMsg.c_str(),
            "front + back should not be greater than inputL * 2 / 3 when front and back are greater than 0.");
        return ge::GRAPH_FAILED;
    } else if (front > 0 && back <= 0 && (front > inputL / DIM_2)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context_->GetNodeName(), "front, back and x's D", shapeMsg.c_str(),
            "front should not be greater than inputL / 2 when front is greater than 0 and back is not greater than 0.");
        return ge::GRAPH_FAILED;
    } else if (front <= 0 && back > 0 && (back > inputL / DIM_2)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context_->GetNodeName(), "front, back and x's D", shapeMsg.c_str(),
            "back should not be greater than inputL / 2 when back is greater than 0 and front is not greater than 0.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CircularPadGradTiling::CheckInput()
{
    if (inputW >= usedUBSize / tSize) {
        std::string valueMsg = "inputW=" + std::to_string(inputW) + ", usable=" + std::to_string(usedUBSize / tSize);
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "x", valueMsg.c_str(),
                                              "the last dim of x exceeds the usable UB capacity.");
        return ge::GRAPH_FAILED;
    }
    if ((outputH != inputH - top - bottom || outputW != inputW - left - right || outputL != inputL - front - back)) {
        std::string shapeMsg = "y: [" + std::to_string(outputH) + ", " + std::to_string(outputW) + ", " +
                               std::to_string(outputL) + "], expected: [" + std::to_string(inputH - top - bottom) +
                               ", " + std::to_string(inputW - left - right) + ", " +
                               std::to_string(inputL - front - back) + "]";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "y, x and paddings", shapeMsg.c_str(),
                                               "the shape of y must equal the shape of x minus the paddings.");
        return ge::GRAPH_FAILED;
    }
    auto ret = CheckLeftAndRight();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = CheckTopAndBottom();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = CheckFrontAndBack();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CircularPadGradTiling::CheckDtype()
{
    // 读取dtype，确定数据类型
    auto xDataType = context_->GetInputDesc(0)->GetDataType();
    if (xDataType == ge::DataType::DT_FLOAT) {
        dataType = TYPE_MODE1;
        tSize = sizeof(float);
    } else if (xDataType == ge::DataType::DT_FLOAT16) {
        dataType = TYPE_MODE2;
        tSize = sizeof(uint16_t);
    } else if (xDataType == ge::DataType::DT_BF16) {
        dataType = TYPE_MODE3;
        tSize = sizeof(uint16_t);
    } else {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "x", Ops::Base::ToString(xDataType).c_str(),
                                              "the dtype of x must be float32, float16 or bfloat16.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CircularPadGradTiling::DoOpTiling()
{
    auto ret = CheckDtype();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    CalculateParams();
    inOutputH = inputH - pTop - pBottom;
    inOutputW = inputW - pLeft - pRight;
    inOutputL = inputL - pFront - pBack;
    if (dataType == TYPE_MODE1) {
        usedUBSize = ubSize / BUFFER_NUM;
    } else {
        usedUBSize = ubSize / BUFFER_NUM / DIM_3;
    }
    ret = CheckInput();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    if (inputH * inputWAlign < usedUBSize / tSize) {
        shapeType = DIM_1;
        workspaceLen = inputH * (ALIGN / WTYPE_SIZE + inputWAlign + ALIGN / WTYPE_SIZE);
    } else {
        shapeType = DIM_2;
        workspaceLen = inOutputH * (ALIGN / WTYPE_SIZE + inputWAlign + ALIGN / WTYPE_SIZE);
    }

    DivCore();
    SetTilingKey();
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4CircularPadGrad(gert::TilingContext* context)
{
    CircularPadGradTiling circularPadGradTiling(context);
    auto ret = circularPadGradTiling.GetShapeAttrsInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = circularPadGradTiling.GetPlatformInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = circularPadGradTiling.DoOpTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = circularPadGradTiling.GetWorkspaceSize();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = circularPadGradTiling.PostTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    context->SetTilingKey(circularPadGradTiling.GetTilingKey());
    circularPadGradTiling.DumpTilingInfo();
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4CircularPadGrad(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepare4CircularPadGrad enter.");
    auto compileInfo = context->GetCompiledInfo<Tiling4CircularPadCommonCompileInfo>();
    OP_CHECK_IF(compileInfo == nullptr,
                ReportTilingFailure(context->GetNodeName(),
                                    "TilingPrepare4CircularPadGrad failed because compile info is null"),
                return ge::GRAPH_FAILED);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr,
                ReportTilingFailure(context->GetNodeName(),
                                    "TilingPrepare4CircularPadGrad failed because platform info is null"),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    if (compileInfo->coreNum <= 0) {
        std::string reason = "TilingPrepare4CircularPadGrad failed to get AIV core number, actual=" +
                             std::to_string(compileInfo->coreNum);
        ReportTilingFailure(context->GetNodeName(), reason.c_str());
        return ge::GRAPH_FAILED;
    }

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    if (compileInfo->ubSize <= 0) {
        std::string reason = "TilingPrepare4CircularPadGrad failed to get UB size, actual=" +
                             std::to_string(compileInfo->ubSize);
        ReportTilingFailure(context->GetNodeName(), reason.c_str());
        return ge::GRAPH_FAILED;
    }

    compileInfo->sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    OP_LOGD(context->GetNodeName(), "TilingPrepare4CircularPadGrad exit.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(CircularPadGrad)
    .Tiling(Tiling4CircularPadGrad)
    .TilingParse<Tiling4CircularPadCommonCompileInfo>(TilingPrepare4CircularPadGrad)
    .TilingInputsDataDependency({1});
} // namespace optiling
