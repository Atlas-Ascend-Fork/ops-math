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
 * \file reduce_std_with_mean_v2_partial_tiling.cpp
 * \brief Host-side tiling for ReduceStdWithMeanV2Partial (two-stage stage 1)
 *
 * Stage 1 splits the reduce dim R across K cores (K = partialK attr, passed
 * from the L2 layer which computes it as min(coreNum, ceil(R/2048))). Core k
 * reduces every row's segment [k*partialSeg, +segLen) and writes (n, mean,
 * m2) to the partial output tensor [M, K, 3]. blockDim = K.
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/reduce_std_with_mean_v2_tiling_data.h"
#include "../op_kernel/reduce_std_with_mean_v2_partial_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;
using Ops::Base::FloorDiv;
using Ops::Base::GetUbBlockSize;

static constexpr int64_t UB_RESERVED = 8 * 1024;
static constexpr int64_t BUFFER_NUM = 2;
static constexpr int64_t R_SEG_MIN = 2048;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape)
{
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return in_shape;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum==0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize==0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, int64_t& totalNum, int64_t& reduceLen,
                                         ge::DataType& dataType, int64_t& correction, float& eps, bool& invert,
                                         int64_t& partialK)
{
    auto inputSelf = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputSelf);
    auto selfShape = EnsureNotScalar(inputSelf->GetStorageShape());
    totalNum = selfShape.GetShapeSize();
    int64_t rank = selfShape.GetDimNum();

    const gert::RuntimeAttrs* dimAttrs = context->GetAttrs();
    reduceLen = 1;
    if (dimAttrs) {
        auto dimList = dimAttrs->GetListInt(0);
        if (dimList != nullptr) {
            const int64_t* dimData = dimList->GetData();
            int64_t dimSize = dimList->GetSize();
            for (int64_t d = 0; d < dimSize; d++) {
                int64_t axis = dimData[d];
                if (axis < 0)
                    axis += rank;
                if (axis >= 0 && axis < rank) {
                    reduceLen *= selfShape.GetDim(axis);
                }
            }
            if (dimSize == 0) {
                // Empty dim means reduce all dimensions, matching InferShape.
                reduceLen = totalNum;
            }
        } else {
            reduceLen = selfShape.GetDim(rank - 1);
        }
    } else {
        reduceLen = selfShape.GetDim(rank - 1);
    }
    OP_CHECK_IF(reduceLen <= 0, OP_LOGE(context, "reduceLen <= 0"), return ge::GRAPH_FAILED);

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();

    correction = 0;
    eps = 0.0f;
    invert = false;
    partialK = 0;
    if (dimAttrs) {
        if (dimAttrs->GetInt(5)) {
            correction = *(dimAttrs->GetInt(5));
        }
        if (dimAttrs->GetBool(3)) {
            invert = *(dimAttrs->GetBool(3));
        }
        if (dimAttrs->GetFloat(4)) {
            eps = *(dimAttrs->GetFloat(4));
        }
        if (dimAttrs->GetInt(1)) {
            partialK = *(dimAttrs->GetInt(1));
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetWorkspace(gert::TilingContext* context, size_t size)
{
    constexpr size_t WORKSPACE_NUM = 1;
    size_t* ws = context->GetWorkspaceSizes(WORKSPACE_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, ws);
    ws[0] = size;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ReduceStdWithMeanV2PartialTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo"),
                return ge::GRAPH_FAILED);

    int64_t totalNum = 0, reduceLen = 0;
    ge::DataType dataType = ge::DT_FLOAT16;
    int64_t correction = 0;
    float eps = 0.0f;
    bool invert = false;
    int64_t partialK = 0;
    OP_CHECK_IF(GetShapeAttrsInfo(context, totalNum, reduceLen, dataType, correction, eps, invert, partialK) !=
                    ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetShapeAttrsInfo"), return ge::GRAPH_FAILED);

    int64_t M = (reduceLen > 0) ? (totalNum / reduceLen) : 0;
    if (M == 0 || reduceLen == 0) {
        context->SetBlockDim(1);
        uint64_t tk = GET_TPL_TILING_KEY(REDUCE_STD_WITH_MEAN_V2_PARTIAL_FP16);
        if (dataType == ge::DT_FLOAT)
            tk = GET_TPL_TILING_KEY(REDUCE_STD_WITH_MEAN_V2_PARTIAL_FP32);
        else if (dataType == ge::DT_BF16)
            tk = GET_TPL_TILING_KEY(REDUCE_STD_WITH_MEAN_V2_PARTIAL_BF16);
        context->SetTilingKey(tk);
        return ge::GRAPH_SUCCESS;
    }

    // K from the L2 attr; defensive recompute when absent
    int64_t K = partialK;
    int64_t maxK = CeilDiv(reduceLen, R_SEG_MIN);
    if (K <= 0 || K > maxK) {
        K = maxK;
    }
    if (K < 1) {
        K = 1;
    }
    if (K > coreNum) {
        K = coreNum;
    }
    int64_t R_seg = CeilDiv(reduceLen, K);

    ReduceStdWithMeanV2TilingData* tiling = context->GetTilingData<ReduceStdWithMeanV2TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(ReduceStdWithMeanV2TilingData), 0, sizeof(ReduceStdWithMeanV2TilingData)) != EOK,
        OP_LOGE(context, "memset tiling"), return ge::GRAPH_FAILED);

    int64_t ubBlockSize = GetUbBlockSize(context);
    // per-tile UB budget for the in-segment Welford loop (serial-path formula)
    int64_t typeSize = (dataType == ge::DT_FLOAT) ? 4 : 2;
    int64_t perElemUB = BUFFER_NUM * typeSize;
    perElemUB += 2 * static_cast<int64_t>(sizeof(float));
    if (dataType != ge::DT_FLOAT) {
        perElemUB += static_cast<int64_t>(sizeof(float));
    }
    int64_t ubAvail = static_cast<int64_t>(ubSize) - UB_RESERVED;
    int64_t ubForData = ubAvail - static_cast<int64_t>(sizeof(float));
    if (ubForData <= 0) {
        ubForData = ubAvail;
    }
    int64_t ubLength = FloorAlign(FloorDiv(ubForData, perElemUB), ubBlockSize);
    if (ubLength < 1) {
        ubLength = 1;
    }
    if (ubLength > reduceLen) {
        ubLength = reduceLen;
    }

    tiling->totalNonReduce = M;
    tiling->reduceLength = reduceLen;
    tiling->blockFactor = M;
    tiling->ubLength = ubLength;
    tiling->correction = correction;
    tiling->eps = eps;
    tiling->invert = invert;
    tiling->vectorize = false;
    tiling->ubM = 0;
    tiling->stage = 1;
    tiling->partialK = K;
    tiling->partialSeg = R_seg;

    context->SetBlockDim(K);
    // workspace for the partial [M, K, 3] fp32 results (P3 plan)
    OP_CHECK_IF(SetWorkspace(context, static_cast<size_t>(M * K * 3 * sizeof(float))) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "SetWorkspace"), return ge::GRAPH_FAILED);

    uint64_t tk = GET_TPL_TILING_KEY(REDUCE_STD_WITH_MEAN_V2_PARTIAL_FP16);
    if (dataType == ge::DT_FLOAT)
        tk = GET_TPL_TILING_KEY(REDUCE_STD_WITH_MEAN_V2_PARTIAL_FP32);
    else if (dataType == ge::DT_BF16)
        tk = GET_TPL_TILING_KEY(REDUCE_STD_WITH_MEAN_V2_PARTIAL_BF16);
    context->SetTilingKey(tk);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForReduceStdWithMeanV2Partial([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReduceStdWithMeanV2PartialCompileInfo {};
IMPL_OP_OPTILING(ReduceStdWithMeanV2Partial)
    .Tiling(ReduceStdWithMeanV2PartialTilingFunc)
    .TilingParse<ReduceStdWithMeanV2PartialCompileInfo>(TilingParseForReduceStdWithMeanV2Partial);

} // namespace optiling
