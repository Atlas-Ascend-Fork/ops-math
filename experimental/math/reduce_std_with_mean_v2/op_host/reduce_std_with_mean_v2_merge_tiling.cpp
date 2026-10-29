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
 * \file reduce_std_with_mean_v2_merge_tiling.cpp
 * \brief Host-side tiling for ReduceStdWithMeanV2Merge (two-stage stage 2)
 *
 * Stage 2 reads the partial tensor [M, K, 3] (float32), Chan-merges the K
 * triples of each row and writes std/mean. blockDim = min(M, coreNum); each
 * core handles a contiguous row range.
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/reduce_std_with_mean_v2_tiling_data.h"
#include "../op_kernel/reduce_std_with_mean_v2_merge_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;

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

static ge::graphStatus ReduceStdWithMeanV2MergeTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo"),
                return ge::GRAPH_FAILED);

    // partial tensor [M, K, 3]
    auto partialInShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, partialInShape);
    auto partialShape = EnsureNotScalar(partialInShape->GetStorageShape());
    int64_t rank = partialShape.GetDimNum();
    int64_t M = (rank > 0) ? partialShape.GetDim(0) : 0;
    int64_t K = (rank > 1) ? partialShape.GetDim(1) : 1;
    if (M < 0) {
        M = 0;
    }
    if (K < 1) {
        K = 1;
    }

    const gert::RuntimeAttrs* dimAttrs = context->GetAttrs();
    int64_t correction = 0;
    float eps = 0.0f;
    bool invert = false;
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
    }

    ReduceStdWithMeanV2TilingData* tiling = context->GetTilingData<ReduceStdWithMeanV2TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(ReduceStdWithMeanV2TilingData), 0, sizeof(ReduceStdWithMeanV2TilingData)) != EOK,
        OP_LOGE(context, "memset tiling"), return ge::GRAPH_FAILED);

    int64_t blockDim = (M < coreNum) ? M : coreNum;
    if (blockDim < 1) {
        blockDim = 1;
    }

    tiling->totalNonReduce = M;
    tiling->reduceLength = 0;
    tiling->blockFactor = CeilDiv(M, blockDim);
    tiling->ubLength = 0;
    tiling->correction = correction;
    tiling->eps = eps;
    tiling->invert = invert;
    tiling->vectorize = false;
    tiling->ubM = 0;
    tiling->stage = 2;
    tiling->partialK = K;
    tiling->partialSeg = 0;

    context->SetBlockDim(blockDim);

    // The launcher requires a workspace size to be set even when the kernel
    // does not use one ("Enable kernel dfx, but dont have workspace").
    constexpr size_t WORKSPACE_NUM = 1;
    size_t* ws = context->GetWorkspaceSizes(WORKSPACE_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, ws);
    ws[0] = 0;

    // The partial input and the Merge outputs are all float32, so the
    // tiling key is always the FP32 one.
    uint64_t tk = GET_TPL_TILING_KEY(REDUCE_STD_WITH_MEAN_V2_MERGE_FP32);
    context->SetTilingKey(tk);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForReduceStdWithMeanV2Merge([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReduceStdWithMeanV2MergeCompileInfo {};
IMPL_OP_OPTILING(ReduceStdWithMeanV2Merge)
    .Tiling(ReduceStdWithMeanV2MergeTilingFunc)
    .TilingParse<ReduceStdWithMeanV2MergeCompileInfo>(TilingParseForReduceStdWithMeanV2Merge);

} // namespace optiling
