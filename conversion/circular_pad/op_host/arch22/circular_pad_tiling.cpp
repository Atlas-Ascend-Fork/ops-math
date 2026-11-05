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
 * \file circular_pad_tiling_tiling.cpp
 * \brief
 */

#include <string>
#include <vector>

#include "circular_pad_tiling.h"
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
constexpr int64_t BUFFER_NUM = 2;
constexpr int64_t DIM_1 = 1;
constexpr int64_t DIM_2 = 2;
constexpr int64_t TYPE_MODE1 = 1;
constexpr int64_t TYPE_MODE2 = 2;
constexpr int64_t TYPE_MODE3 = 3;
constexpr int64_t TYPE_MODE4 = 4;
constexpr int64_t TYPE_MODE5 = 5;
// CircularPadTiling
ge::graphStatus CircularPadTiling::CheckLeftAndRight()
{
    std::string shapeMsg = "left: " + std::to_string(left) + ", right: " + std::to_string(right) +
                           ", x's W: " + std::to_string(inputW);
    if ((left > 0 && left > inputW) || (right > 0 && right > inputW)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context_->GetNodeName(), "left, right and x's W", shapeMsg.c_str(),
            "left/right should not be greater than inputW when left/right is greater than 0.");
        return ge::GRAPH_FAILED;
    } else if (left > 0 && right < 0 && (left > inputW + right || -right >= inputW)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context_->GetNodeName(), "left, right and x's W", shapeMsg.c_str(),
            "left should not be greater than inputW + right and abs(right) should not be greater than inputW when left "
            "is greater than 0 and right is not greater than 0.");
        return ge::GRAPH_FAILED;
    } else if (left < 0 && right > 0 && (-left >= inputW || right > inputW + left)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context_->GetNodeName(), "left, right and x's W", shapeMsg.c_str(),
            "right should not be greater than inputW + left and abs(left) should not be greater than inputW when right "
            "is greater than 0 and left is not greater than 0.");
        return ge::GRAPH_FAILED;
    } else if (left < 0 && right < 0 && (-left - right >= inputW)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "left, right and x's W", shapeMsg.c_str(),
                                               "abs(left + right) must be less than inputW.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CircularPadTiling::CheckTopAndBottom()
{
    std::string shapeMsg = "top: " + std::to_string(top) + ", bottom: " + std::to_string(bottom) +
                           ", x's H: " + std::to_string(inputH);
    if ((top > 0 && top > inputH) || (bottom > 0 && bottom > inputH)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context_->GetNodeName(), "top, bottom and x's H", shapeMsg.c_str(),
            "top/bottom should not be greater than inputH when top/bottom is greater than 0.");
        return ge::GRAPH_FAILED;
    } else if (top > 0 && bottom < 0 && (top > inputH + bottom || -bottom >= inputH)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context_->GetNodeName(), "top, bottom and x's H", shapeMsg.c_str(),
            "top should not be greater than inputH + bottom and abs(bottom) should not be greater than inputH when top "
            "is greater than 0 and bottom is not greater than 0.");
        return ge::GRAPH_FAILED;
    } else if (top < 0 && bottom > 0 && (-top >= inputH || bottom > inputH + top)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context_->GetNodeName(), "top, bottom and x's H", shapeMsg.c_str(),
            "bottom should not be greater than inputH + top and abs(top) should not be greater than inputH when bottom "
            "is greater than 0 and top is not greater than 0.");
        return ge::GRAPH_FAILED;
    } else if (top < 0 && bottom < 0 && (-top - bottom >= inputH)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "top, bottom and x's H", shapeMsg.c_str(),
                                               "abs(top + bottom) must be less than inputH.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CircularPadTiling::CheckFrontAndBack()
{
    std::string shapeMsg = "front: " + std::to_string(front) + ", back: " + std::to_string(back) +
                           ", x's D: " + std::to_string(inputL);
    if ((front > 0 && front > inputL) || (back > 0 && back > inputL)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context_->GetNodeName(), "front, back and x's D", shapeMsg.c_str(),
            "front/back should not be greater than inputL when front/back is greater than 0.");
        return ge::GRAPH_FAILED;
    } else if (front > 0 && back < 0 && (front > inputL + back || -back >= inputL)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context_->GetNodeName(), "front, back and x's D", shapeMsg.c_str(),
            "front should not be greater than inputL + back and abs(back) should not be greater than inputL when front "
            "is greater than 0 and back is not greater than 0.");
        return ge::GRAPH_FAILED;
    } else if (front < 0 && back > 0 && (-front >= inputL || back > inputL + front)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context_->GetNodeName(), "front, back and x's D", shapeMsg.c_str(),
            "back should not be greater than inputL + front and abs(front) should not be greater than inputL when back "
            "is greater than 0 and front is not greater than 0.");
        return ge::GRAPH_FAILED;
    } else if (front < 0 && back < 0 && (-front - back >= inputL)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "front, back and x's D", shapeMsg.c_str(),
                                               "abs(front + back) must be less than inputL.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CircularPadTiling::CheckInput()
{
    if ((inputW >= (int64_t)ubSize / BUFFER_NUM / tSize)) {
        std::string valueMsg = "inputW=" + std::to_string(inputW) +
                               ", usable=" + std::to_string(ubSize / BUFFER_NUM / tSize);
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "x", valueMsg.c_str(),
                                              "the last dim of x exceeds the usable UB capacity.");
        return ge::GRAPH_FAILED;
    }
    if ((outputH != inputH + top + bottom || outputW != inputW + left + right || outputL != inputL + front + back)) {
        std::string shapeMsg = "y: [" + std::to_string(outputH) + ", " + std::to_string(outputW) + ", " +
                               std::to_string(outputL) + "], expected: [" + std::to_string(inputH + top + bottom) +
                               ", " + std::to_string(inputW + left + right) + ", " +
                               std::to_string(inputL + front + back) + "]";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "y, x and paddings", shapeMsg.c_str(),
                                               "the shape of y must equal the shape of x plus the paddings.");
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

ge::graphStatus CircularPadTiling::CheckDtype()
{
    // 读取dtype，确定数据类型
    auto xDataType = context_->GetInputDesc(0)->GetDataType();
    if (xDataType == ge::DataType::DT_INT8) {
        dataType = TYPE_MODE1;
        tSize = sizeof(int8_t);
    } else if (xDataType == ge::DataType::DT_FLOAT16) {
        dataType = TYPE_MODE2;
        tSize = sizeof(uint16_t);
    } else if (xDataType == ge::DataType::DT_BF16) {
        dataType = TYPE_MODE3;
        tSize = sizeof(uint16_t);
    } else if (xDataType == ge::DataType::DT_FLOAT) {
        dataType = TYPE_MODE4;
        tSize = sizeof(float);
    } else if (xDataType == ge::DataType::DT_INT32) {
        dataType = TYPE_MODE5;
        tSize = sizeof(int32_t);
    } else {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "x", Ops::Base::ToString(xDataType).c_str(),
                                              "the dtype of x must be int8, float16, bfloat16, float or int32.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CircularPadTiling::DoOpTiling()
{
    auto ret = CheckDtype();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    CalculateParams();
    inOutputH = inputH + nTop + nBottom;
    inOutputW = inputW + nLeft + nRight;
    inOutputL = inputL + nFront + nBack;
    outputWAlign = (outputW * tSize + ALIGN - DIM_1) / ALIGN * ALIGN / tSize;
    inOutputWAlign = (inOutputW * tSize + ALIGN - DIM_1) / ALIGN * ALIGN / tSize;
    ret = CheckInput();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    if (inOutputH * outputWAlign < (int64_t)ubSize / BUFFER_NUM / tSize) {
        shapeType = DIM_1;
        workspaceLen = inOutputH * (leftAlign + inOutputWAlign + rightAlign);
    } else {
        shapeType = DIM_2;
        leftAlign = left > 0 ? leftAlign : ALIGN / tSize;
        rightAlign = right > 0 ? rightAlign : ALIGN / tSize;
        leftAlign = leftAlign > inputW ? left : leftAlign;
        rightAlign = rightAlign > inputW ? right : rightAlign;
        workspaceLen = inOutputH * (leftAlign + rightAlign);
    }
    DivCore();
    SetTilingKey();
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4CircularPad(gert::TilingContext* context)
{
    CircularPadTiling circularPadTiling(context);
    auto ret = circularPadTiling.GetShapeAttrsInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = circularPadTiling.GetPlatformInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = circularPadTiling.DoOpTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = circularPadTiling.GetWorkspaceSize();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = circularPadTiling.PostTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    context->SetTilingKey(circularPadTiling.GetTilingKey());
    circularPadTiling.DumpTilingInfo();
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4CircularPad(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepare4CircularPad enter.");
    auto compileInfo = context->GetCompiledInfo<Tiling4CircularPadCommonCompileInfo>();
    OP_CHECK_IF(
        compileInfo == nullptr,
        ReportTilingFailure(context->GetNodeName(), "TilingPrepare4CircularPad failed because compile info is null"),
        return ge::GRAPH_FAILED);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(
        platformInfo == nullptr,
        ReportTilingFailure(context->GetNodeName(), "TilingPrepare4CircularPad failed because platform info is null"),
        return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    if (compileInfo->coreNum <= 0) {
        std::string reason = "TilingPrepare4CircularPad failed to get AIV core number, actual=" +
                             std::to_string(compileInfo->coreNum);
        ReportTilingFailure(context->GetNodeName(), reason.c_str());
        return ge::GRAPH_FAILED;
    }

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    if (compileInfo->ubSize <= 0) {
        std::string reason = "TilingPrepare4CircularPad failed to get UB size, actual=" +
                             std::to_string(compileInfo->ubSize);
        ReportTilingFailure(context->GetNodeName(), reason.c_str());
        return ge::GRAPH_FAILED;
    }

    compileInfo->sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    OP_LOGD(context->GetNodeName(), "TilingPrepare4CircularPad exit.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(CircularPad)
    .Tiling(Tiling4CircularPad)
    .TilingParse<Tiling4CircularPadCommonCompileInfo>(TilingPrepare4CircularPad)
    .TilingInputsDataDependency({1});
} // namespace optiling
