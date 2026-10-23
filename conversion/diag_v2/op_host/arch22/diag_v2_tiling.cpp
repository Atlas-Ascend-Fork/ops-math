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
 * \file diag_v2_tiling.cpp
 * \brief
 */
#include "diag_v2_tiling.h"
#include "log/log.h"
#include "platform/platform_ascendc.h"
#include "util/math_util.h"

#include <string>
#include <vector>

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

static const int32_t SIZE_1 = 1;
static const int32_t SIZE_2 = 2;
static const int32_t SIZE_4 = 4;
static const int32_t SIZE_8 = 8;
static const int32_t SIZE_16 = 16;
static const int32_t LENGTH_1024 = 1024;
static const int32_t LENGTH_128 = 128;
static const int32_t LENGTH_64 = 64;
static const int32_t LENGTH_32 = 32;

inline static ge::graphStatus DiagV2SetTilingData(gert::TilingContext* context, DiagV2TilingData& tilingData)
{
    auto rawTilingData = context->GetRawTilingData();
    tilingData.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

static inline ge::graphStatus CalcAuxMatrixTiling(const int32_t typeSize, DiagV2TilingData& tilingData)
{
    switch (typeSize) {
        case SIZE_1:
            tilingData.set_tilingKey(static_cast<int64_t>(DiagV2TilingKey::ASSIST_SIZE_1));
            tilingData.set_matrixRowLength(LENGTH_128);
            break;
        case SIZE_2:
            tilingData.set_tilingKey(static_cast<int64_t>(DiagV2TilingKey::ASSIST_SIZE_2));
            tilingData.set_matrixRowLength(LENGTH_128);
            break;
        case SIZE_4:
            tilingData.set_tilingKey(static_cast<int64_t>(DiagV2TilingKey::ASSIST_SIZE_4));
            tilingData.set_matrixRowLength(LENGTH_64);
            break;
        case SIZE_8:
            tilingData.set_tilingKey(static_cast<int64_t>(DiagV2TilingKey::ASSIST_SIZE_8));
            tilingData.set_matrixRowLength(LENGTH_64);
            break;
        case SIZE_16:
            tilingData.set_tilingKey(static_cast<int64_t>(DiagV2TilingKey::ASSIST_SIZE_16));
            tilingData.set_matrixRowLength(LENGTH_32);
            break;
        default:
            break;
    }

    return ge::GRAPH_SUCCESS;
}

static inline ge::graphStatus processDiagFlat(gert::TilingContext* context)
{
    // call DiagFlat tiling for DiagV2
    return TilingDiagFlat(context);
}

static inline int64_t CalcLeastNumPerCore(const int32_t typeSize, ge::DataType dataType,
                                          const gert::TilingContext* context)
{
    OP_CHECK_IF(
        (typeSize <= 0),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "x", Ops::Base::ToString(dataType).c_str(),
                                              "dtype size must be positive; supported sizes are 1, 2, 4, 8 and 16"),
        return -1);

    int64_t leastNumPerCore = BLOCK_SIZE / typeSize;
    return (leastNumPerCore > 0) ? leastNumPerCore : 1;
}

static ge::graphStatus Tiling4DiagV2(gert::TilingContext* context)
{
    OP_LOGD(context, "Tiling4DiagV2 running begin");
    auto inputShapePtr = context->GetInputShape(0);
    OP_CHECK_IF(inputShapePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "x"),
                return ge::GRAPH_FAILED);
    auto inputShape = inputShapePtr->GetStorageShape();
    OP_CHECK_IF((inputShape.GetDimNum() != SIZE_2 && inputShape.GetDimNum() != 1),
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    context->GetNodeName(), "x", std::to_string(inputShape.GetDimNum()).c_str(), "rank must be 1 or 2"),
                return ge::GRAPH_FAILED);
    if (inputShape.GetDimNum() == 1) {
        return processDiagFlat(context);
    }

    DiagV2TilingData tilingData;
    tilingData.set_xHeight(inputShape.GetDim(0));
    tilingData.set_xWidth(inputShape.GetDim(1));
    auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "diagonal", "null",
                                                      "required attribute is missing"),
                return ge::GRAPH_FAILED);
    const int64_t* diagonalPtr = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_IF(diagonalPtr == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "diagonal", "null",
                                                      "required attribute is missing"),
                return ge::GRAPH_FAILED);
    const int64_t diagonal = *diagonalPtr;
    OP_CHECK_IF((diagonal >= 0 && diagonal > tilingData.get_xWidth()) ||
                    (diagonal < 0 && std::abs(diagonal) > tilingData.get_xHeight()),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context->GetNodeName(), "diagonal", std::to_string(diagonal).c_str(),
                    ("diagonal must be within [-x_height, x_width], x_width=" +
                     std::to_string(tilingData.get_xWidth()) + ", x_height=" + std::to_string(tilingData.get_xHeight()))
                        .c_str()),
                return ge::GRAPH_FAILED);
    tilingData.set_gmOffset(diagonal > 0 ? diagonal : std::abs(diagonal) * tilingData.get_xWidth());
    tilingData.set_numOut((diagonal >= 0) ? std::min((tilingData.get_xWidth() - diagonal), tilingData.get_xHeight()) :
                                            std::min((tilingData.get_xHeight() + diagonal), tilingData.get_xWidth()));
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_IF(inputDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "x"), return ge::GRAPH_FAILED);
    auto dataType = inputDesc->GetDataType();
    const int32_t typeSize = ge::GetSizeByDataType(dataType);
    int64_t leastNumPerCore = CalcLeastNumPerCore(typeSize, dataType, context);
    OP_CHECK_IF(
        (leastNumPerCore <= 0),
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "Tiling4DiagV2 leastNumPerCore is invalid after dtype check."),
        return ge::GRAPH_FAILED);

    auto compileInfo = reinterpret_cast<const DiagV2CompileInfo*>(context->GetCompileInfo());
    OP_CHECK_IF(compileInfo == nullptr,
                ReportTilingFailure(context->GetNodeName(), "Tiling4DiagV2 failed because compile info is null"),
                return ge::GRAPH_FAILED);
    uint32_t coreNum = compileInfo->totalCoreNum;
    int64_t tmpRealCoreNum = tilingData.get_numOut() < coreNum ? tilingData.get_numOut() : coreNum;
    int64_t tmpCorePerNum = Ops::Base::CeilDiv(tilingData.get_numOut(), tmpRealCoreNum);
    tilingData.set_numPerCore(Ops::Base::CeilDiv(tmpCorePerNum, leastNumPerCore) * leastNumPerCore);
    tilingData.set_realCoreNum(Ops::Base::CeilDiv(tilingData.get_numOut(), tilingData.get_numPerCore()));
    tilingData.set_tailNum(tilingData.get_numOut() - (tilingData.get_realCoreNum() - 1) * tilingData.get_numPerCore());
    OP_CHECK_IF(CalcAuxMatrixTiling(typeSize, tilingData) != ge::GRAPH_SUCCESS,
                ReportTilingFailure(context->GetNodeName(), "CalcAuxMatrixTiling failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(DiagV2SetTilingData(context, tilingData) != ge::GRAPH_SUCCESS,
                ReportTilingFailure(context->GetNodeName(), "DiagV2SetTilingData failed to serialize tiling data"),
                return ge::GRAPH_FAILED);
    context->SetBlockDim(tilingData.get_realCoreNum());
    context->SetTilingKey(tilingData.get_tilingKey());

    size_t* workspaces = context->GetWorkspaceSizes(1);
    workspaces[0] = SIZE_16 * LENGTH_1024 * LENGTH_1024;

    OP_LOGD(
        context,
        "tilingData is xWidth:%ld, xHeight:%ld, gmOffset:%ld, numOut:%ld, realCoreNum:%ld, numPerCore:%ld, tailNum:%ld,\
                              tilingKey:%ld, matrixRowLength:%ld",
        tilingData.get_xWidth(), tilingData.get_xHeight(), tilingData.get_gmOffset(), tilingData.get_numOut(),
        tilingData.get_realCoreNum(), tilingData.get_numPerCore(), tilingData.get_tailNum(), tilingData.get_tilingKey(),
        tilingData.get_matrixRowLength());

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4DiagV2(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<DiagV2CompileInfo>();
    OP_CHECK_IF(compileInfo == nullptr,
                ReportTilingFailure(context->GetNodeName(), "TilingPrepare4DiagV2 failed because compile info is null"),
                return ge::GRAPH_FAILED);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(
        platformInfo == nullptr,
        ReportTilingFailure(context->GetNodeName(), "TilingPrepare4DiagV2 failed because platform info is null"),
        return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    if (compileInfo->totalCoreNum <= 0) {
        std::string reason = "TilingPrepare4DiagV2 failed to get AIV core number, actual=" +
                             std::to_string(compileInfo->totalCoreNum);
        ReportTilingFailure(context->GetNodeName(), reason.c_str());
        return ge::GRAPH_FAILED;
    }

    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = static_cast<int64_t>(ubSizePlatForm);
    if (compileInfo->ubSizePlatForm <= 0) {
        std::string reason = "TilingPrepare4DiagV2 failed to get UB size, actual=" +
                             std::to_string(compileInfo->ubSizePlatForm);
        ReportTilingFailure(context->GetNodeName(), reason.c_str());
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DiagV2).Tiling(Tiling4DiagV2).TilingParse<DiagV2CompileInfo>(TilingPrepare4DiagV2);
} // namespace optiling
