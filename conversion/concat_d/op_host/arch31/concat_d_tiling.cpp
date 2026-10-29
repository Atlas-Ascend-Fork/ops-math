/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "concat_d_tiling.h"

#include "concat_d_plan.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr uint32_t BLOCK_DIM = 1U;
constexpr uint32_t WORKSPACE_BYTES = 32U;
constexpr size_t MAX_RANK = 8U;

uint32_t ElementBytes(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_BOOL:
        case ge::DT_INT8:
        case ge::DT_UINT8:
            return 1U;
        case ge::DT_BF16:
        case ge::DT_FLOAT16:
        case ge::DT_INT16:
        case ge::DT_UINT16:
            return 2U;
        case ge::DT_FLOAT:
        case ge::DT_INT32:
        case ge::DT_UINT32:
            return 4U;
        case ge::DT_DOUBLE:
        case ge::DT_COMPLEX64:
        case ge::DT_INT64:
        case ge::DT_UINT64:
            return 8U;
        default:
            return 0U;
    }
}

bool CheckOutputShape(gert::TilingContext* context, const int64_t* dims, size_t tensorCount, size_t rank, int64_t axis)
{
    const auto* outputShape = context->GetOutputShape(0);
    if (outputShape == nullptr || outputShape->GetStorageShape().GetDimNum() != rank) {
        return false;
    }
    const size_t normalizedAxis = static_cast<size_t>(axis < 0 ? axis + static_cast<int64_t>(rank) : axis);
    uint64_t axisTotal = 0U;
    for (size_t tensor = 0U; tensor < tensorCount; ++tensor) {
        const uint64_t axisDim = static_cast<uint64_t>(dims[tensor * rank + normalizedAxis]);
        if (axisTotal > static_cast<uint64_t>(INT64_MAX) - axisDim) {
            return false;
        }
        axisTotal += axisDim;
    }
    for (size_t dim = 0U; dim < rank; ++dim) {
        const int64_t expected = dim == normalizedAxis ? static_cast<int64_t>(axisTotal) : dims[dim];
        if (outputShape->GetStorageShape().GetDim(dim) != expected) {
            return false;
        }
    }
    return true;
}

bool SaveTilingData(gert::TilingContext* context, const concat_d_arch31::ConcatDTilingDataPod& plan)
{
    ConcatDArch31TilingData tilingData;
    uint64_t inputRowBytes[concat_d_arch31::MAX_INPUTS] = {};
    for (size_t index = 0U; index < concat_d_arch31::MAX_INPUTS; ++index) {
        inputRowBytes[index] = plan.inputRowBytes[index];
    }
    tilingData.set_version(plan.version);
    tilingData.set_flags(plan.flags);
    tilingData.set_tensorNum(plan.tensorNum);
    tilingData.set_tileBytes(plan.tileBytes);
    tilingData.set_outerCount(plan.outerCount);
    tilingData.set_outputRowBytes(plan.outputRowBytes);
    tilingData.set_inputRowBytes(inputRowBytes);

    auto* raw = context->GetRawTilingData();
    if (raw == nullptr || raw->GetData() == nullptr || tilingData.GetDataSize() != sizeof(plan) ||
        tilingData.GetDataSize() > raw->GetCapacity()) {
        return false;
    }
    tilingData.SaveToBuffer(raw->GetData(), raw->GetCapacity());
    raw->SetDataSize(tilingData.GetDataSize());
    return true;
}
} // namespace

static ge::graphStatus TilingConcatDArch31(gert::TilingContext* context)
{
    if (context == nullptr || context->GetAttrs() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t* axis = context->GetAttrs()->GetAttrPointer<int64_t>(0);
    const int64_t* count = context->GetAttrs()->GetAttrPointer<int64_t>(1);
    const uint64_t actualInputCount = context->GetComputeNodeInputNum();
    if (axis == nullptr || count == nullptr || *count <= 0 ||
        *count > static_cast<int64_t>(concat_d_arch31::MAX_INPUTS) ||
        actualInputCount != static_cast<uint64_t>(*count)) {
        return ge::GRAPH_FAILED;
    }

    const auto* firstShape = context->GetDynamicInputShape(0, 0);
    const auto* firstDesc = context->GetDynamicInputDesc(0, 0);
    const auto* outputDesc = context->GetOutputDesc(0);
    if (firstShape == nullptr || firstDesc == nullptr || outputDesc == nullptr ||
        outputDesc->GetDataType() != firstDesc->GetDataType()) {
        return ge::GRAPH_FAILED;
    }
    const size_t rank = firstShape->GetStorageShape().GetDimNum();
    if (rank == 0U || rank > MAX_RANK) {
        return ge::GRAPH_FAILED;
    }

    int64_t dims[concat_d_arch31::MAX_INPUTS * MAX_RANK] = {};
    for (int64_t tensor = 0; tensor < *count; ++tensor) {
        const auto* shape = context->GetDynamicInputShape(0, static_cast<size_t>(tensor));
        const auto* desc = context->GetDynamicInputDesc(0, static_cast<size_t>(tensor));
        if (shape == nullptr || desc == nullptr || desc->GetDataType() != firstDesc->GetDataType() ||
            shape->GetStorageShape().GetDimNum() != rank) {
            return ge::GRAPH_FAILED;
        }
        for (size_t dim = 0U; dim < rank; ++dim) {
            dims[static_cast<size_t>(tensor) * rank + dim] = shape->GetStorageShape().GetDim(dim);
        }
    }

    auto* platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    platform_ascendc::PlatformAscendC platform(platformInfo);
    uint64_t ubBytes = 0U;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);

    concat_d_arch31::ConcatDTilingDataPod plan{};
    if (!BuildConcatDArch31Plan(
            {dims, static_cast<size_t>(*count), rank, *axis, ElementBytes(firstDesc->GetDataType()), ubBytes}, plan) ||
        !CheckOutputShape(context, dims, static_cast<size_t>(*count), rank, *axis) || !SaveTilingData(context, plan)) {
        return ge::GRAPH_FAILED;
    }

    size_t* workspace = context->GetWorkspaceSizes(1);
    if (workspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspace[0] = WORKSPACE_BYTES;
    context->SetBlockDim(BLOCK_DIM);
    context->SetTilingKey(0U);
    return ge::GRAPH_SUCCESS;
}

// Arch31 derives its plan from runtime descriptors and platform information,
// so it needs no DSL compile metadata, but must own the parse registration.
struct ConcatDArch31CompileInfo {};

static ge::graphStatus TilingPrepare4ConcatDArch31(gert::TilingParseContext* context)
{
    return context != nullptr && context->GetCompiledInfo<ConcatDArch31CompileInfo>() != nullptr ? ge::GRAPH_SUCCESS :
                                                                                                   ge::GRAPH_FAILED;
}

IMPL_OP_OPTILING(ConcatD)
    .Tiling(TilingConcatDArch31)
    .TilingParse<ConcatDArch31CompileInfo>(TilingPrepare4ConcatDArch31);
} // namespace optiling
