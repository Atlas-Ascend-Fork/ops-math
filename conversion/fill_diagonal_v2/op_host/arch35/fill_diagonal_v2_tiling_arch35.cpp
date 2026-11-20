/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fill_diagonal_v2_tiling_arch35.cpp
 * \brief FillDiagonalV2 arch35 (Ascend950/350) tiling
 */
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "log/log.h"
#include "platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "../fill_diagonal_v2_tiling.h"

constexpr size_t MAX_DIM_NUM = 8;
constexpr size_t MIN_DIM_NUM = 2;
constexpr uint64_t CACHE_LINE_BYTES = 128;
constexpr uint64_t DENSE_END_THRESHOLD = 1000000;
constexpr uint64_t DENSE_STEP_THRESHOLD_1B = 1000;
constexpr uint64_t DENSE_STEP_THRESHOLD_2B = 300;
constexpr uint64_t DENSE_STEP_THRESHOLD_4B = 200;
constexpr uint64_t DENSE_STEP_THRESHOLD_8B = 80;

namespace optiling {
struct FillDiagonalV2TilingParams {
    const FillDiagonalV2CompileInfo* compileInfo = nullptr;
    uint64_t totalLength = 0;
    uint64_t step = 0;
    uint64_t end = 0;
    ge::DataType dtype = ge::DT_UNDEFINED;
    bool wrap = false;
};

inline ge::graphStatus SetTilingKey4FillDiagonalV2(gert::TilingContext* context, uint64_t step, uint64_t end, bool wrap)
{
    uint64_t stepThreshold = 0;
    auto dtype = context->GetInputDesc(0)->GetDataType();
    int64_t tilingKey = 0;
    if (dtype == ge::DT_INT8 || dtype == ge::DT_UINT8 || dtype == ge::DT_BOOL) {
        stepThreshold = DENSE_STEP_THRESHOLD_1B;
    } else if (dtype == ge::DT_INT16 || dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) {
        stepThreshold = DENSE_STEP_THRESHOLD_2B;
    } else if (dtype == ge::DT_INT32 || dtype == ge::DT_FLOAT) {
        stepThreshold = DENSE_STEP_THRESHOLD_4B;
    } else if (dtype == ge::DT_INT64 || dtype == ge::DT_DOUBLE) {
        stepThreshold = DENSE_STEP_THRESHOLD_8B;
    } else {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            context->GetNodeName(), "x", Ops::Base::ToString(dtype).c_str(),
            "supported dtypes are DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_UINT8, DT_BOOL, DT_INT8, "
            "DT_INT16, DT_INT32, DT_BFLOAT16, DT_INT64");
        return ge::GRAPH_FAILED;
    }
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "platformInfo"),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    bool isDav3510Arch = ascendcPlatform.GetCurNpuArch() == NpuArch::DAV_3510;
    if (!isDav3510Arch && wrap && (end > DENSE_END_THRESHOLD) &&
        (step < stepThreshold)) { // dense: A2A3 only, A5 uses sparse due to L2 cache coherence issue
        tilingKey = 1;
    } else {
        tilingKey = 0;
    }
    context->SetTilingKey(tilingKey);
    OP_LOGD(context, "Tiling key is %ld.", tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ParseFillDiagonalV2Tiling(gert::TilingContext* context, FillDiagonalV2TilingParams& params)
{
    OP_LOGD(context, "Begin the tiling process for FillDiagonalV2 arch35.");
    params.compileInfo = reinterpret_cast<const FillDiagonalV2CompileInfo*>(context->GetCompileInfo());
    if (params.compileInfo == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "compileInfo");
        return ge::GRAPH_FAILED;
    }
    auto attrPtr = context->GetAttrs();
    if (attrPtr == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "attrs");
        return ge::GRAPH_FAILED;
    }
    auto shape = context->GetInputShape(0);
    if (shape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "x");
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(shape->GetStorageShape().GetDimNum() > MAX_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "x",
                                                         std::to_string(shape->GetStorageShape().GetDimNum()).c_str(),
                                                         "The dim num of x must be less than or equal to 8"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(shape->GetStorageShape().GetDimNum() < MIN_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "x",
                                                         std::to_string(shape->GetStorageShape().GetDimNum()).c_str(),
                                                         "The dim num of x must be greater than or equal to 2"),
                return ge::GRAPH_FAILED);

    if (shape->GetStorageShape().GetDimNum() > MIN_DIM_NUM) {
        const int64_t dimLength = shape->GetStorageShape().GetDim(0);
        for (size_t i = 1; i < shape->GetStorageShape().GetDimNum(); ++i) {
            OP_CHECK_IF(shape->GetStorageShape().GetDim(i) != dimLength,
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                            context->GetNodeName(), "x",
                            ("dim " + std::to_string(i) + " is " + std::to_string(shape->GetStorageShape().GetDim(i)) +
                             ", dim 0 is " + std::to_string(dimLength))
                                .c_str(),
                            "When the dim num of x is greater than 2, all dimensions must be of equal length"),
                        return ge::GRAPH_FAILED);
        }
    }

    auto inputDesc = context->GetInputDesc(0);
    if (inputDesc == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "x");
        return ge::GRAPH_FAILED;
    }
    params.dtype = inputDesc->GetDataType();
    const std::set<ge::DataType> supportedDtypes = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_DOUBLE, ge::DT_UINT8,
                                                    ge::DT_BOOL,    ge::DT_INT8,  ge::DT_INT16,  ge::DT_INT32,
                                                    ge::DT_BF16,    ge::DT_INT64};
    OP_CHECK_IF(supportedDtypes.count(params.dtype) == 0,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    context->GetNodeName(), "x", Ops::Base::ToString(params.dtype).c_str(),
                    "supported dtypes are DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_UINT8, DT_BOOL, DT_INT8, "
                    "DT_INT16, DT_INT32, DT_BFLOAT16, DT_INT64"),
                return ge::GRAPH_FAILED);

    auto fillValueDesc = context->GetInputDesc(1);
    if (fillValueDesc == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "fill_value");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(
        fillValueDesc->GetDataType() != params.dtype,
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            context->GetNodeName(), "x, fill_value",
            (Ops::Base::ToString(params.dtype) + ", " + Ops::Base::ToString(fillValueDesc->GetDataType())).c_str(),
            "fill_value dtype must be the same as x dtype"),
        return ge::GRAPH_FAILED);

    uint64_t totalLength = 1;
    for (size_t i = 0; i < shape->GetStorageShape().GetDimNum(); ++i) {
        totalLength *= shape->GetStorageShape().GetDim(i);
    }
    params.totalLength = totalLength;
    uint64_t step = 0;
    constexpr uint64_t matrixDim = 2;
    if (shape->GetStorageShape().GetDimNum() == matrixDim) {
        step = shape->GetStorageShape().GetDim(1) + 1;
    } else {
        uint64_t diagStride = 1;
        for (size_t i = 0; i < shape->GetStorageShape().GetDimNum(); ++i) {
            step += diagStride;
            diagStride *= shape->GetStorageShape().GetDim(i);
        }
    }
    params.step = step;
    auto wrapPtr = attrPtr->GetAttrPointer<bool>(0);
    params.wrap = wrapPtr != nullptr ? *wrapPtr : false;
    uint64_t end = totalLength;
    if (shape->GetStorageShape().GetDimNum() == matrixDim &&
        shape->GetStorageShape().GetDim(0) > shape->GetStorageShape().GetDim(1) && !params.wrap) {
        end = shape->GetStorageShape().GetDim(1) * shape->GetStorageShape().GetDim(1);
    }
    params.end = end;
    return ge::GRAPH_SUCCESS;
}

static uint64_t ComputeUsedCoreNumForRegbase(const FillDiagonalV2TilingParams& params)
{
    const uint64_t tilingCoreNum = params.compileInfo->totalCoreNum;
    const int64_t elemSize = ge::GetSizeByDataType(params.dtype);
    if (elemSize <= 0) {
        return tilingCoreNum;
    }
    const uint64_t minElemsPerCore = CACHE_LINE_BYTES / static_cast<uint64_t>(elemSize);
    if (params.totalLength / tilingCoreNum >= minElemsPerCore) {
        return tilingCoreNum;
    }
    const uint64_t usedCoreNum = params.totalLength / minElemsPerCore;
    return usedCoreNum < 1 ? 1 : usedCoreNum;
}

static ge::graphStatus SaveFillDiagonalV2Tiling(gert::TilingContext* context, const FillDiagonalV2TilingParams& params,
                                                uint64_t usedCoreNum)
{
    FillDiagonalV2TilingData tiling;
    const uint64_t blockLength = params.totalLength / usedCoreNum;
    const uint64_t lastBlockLength = params.totalLength - params.totalLength / usedCoreNum * (usedCoreNum - 1);
    tiling.set_totalLength(params.totalLength);
    tiling.set_step(params.step);
    tiling.set_end(params.end);
    tiling.set_ubSize(params.compileInfo->ubSizePlatForm);
    tiling.set_blockLength(blockLength);
    tiling.set_lastBlockLength(lastBlockLength);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    if (SetTilingKey4FillDiagonalV2(context, params.step, params.end, params.wrap) == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    context->SetBlockDim(usedCoreNum);
    size_t* currentWorkspaceSize = context->GetWorkspaceSizes(1);
    currentWorkspaceSize[0] = params.compileInfo->sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4FillDiagonalV2ForRegbase(gert::TilingContext* context)
{
    FillDiagonalV2TilingParams params;
    if (ParseFillDiagonalV2Tiling(context, params) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return SaveFillDiagonalV2Tiling(context, params, ComputeUsedCoreNumForRegbase(params));
}

static ge::graphStatus TilingPrepare4FillDiagonalV2(gert::TilingParseContext* context)
{
    OP_LOGD(context, "Tiling Prepare For FillDiagonalV2 start.");
    auto compileInfo = context->GetCompiledInfo<FillDiagonalV2CompileInfo>();
    if (compileInfo == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "compileInfo");
        return ge::GRAPH_FAILED;
    }
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "platformInfo");
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(compileInfo->totalCoreNum < 1,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "core num",
                                                      std::to_string(compileInfo->totalCoreNum).c_str(),
                                                      "must be greater than or equal to 1"),
                return ge::GRAPH_FAILED);
    uint64_t ubSizePlatForm = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    OP_CHECK_IF(ubSizePlatForm == 0, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "ubSize"),
                return ge::GRAPH_FAILED);
    compileInfo->ubSizePlatForm = static_cast<int64_t>(ubSizePlatForm);
    OP_LOGD(context, "ub_size_platform is %llu.", static_cast<unsigned long long>(compileInfo->ubSizePlatForm));
    compileInfo->sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    uint64_t totalUbSize = 0;
    platformInfo->GetLocalMemSize(fe::LocalMemType::UB, totalUbSize);
    OP_LOGD(context, "total_ub_size is %llu.", static_cast<unsigned long long>(totalUbSize));
    OP_LOGD(context, "Tiling Prepare For FillDiagonalV2 end.");
    return ge::GRAPH_SUCCESS;
}

// register tiling interface of the FillDiagonalV2 op.
IMPL_OP_OPTILING(FillDiagonalV2)
    .Tiling(Tiling4FillDiagonalV2ForRegbase)
    .TilingParse<FillDiagonalV2CompileInfo>(TilingPrepare4FillDiagonalV2);
} // namespace optiling
