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
 * \file reduce_std_with_mean_v2_tiling.cpp
 * \brief Host-side tiling for ReduceStdWithMeanV2
 *
 * Tiling strategy:
 *   - Multi-core split along non-reduce dimension
 *   - UB tile along reduce dimension (single-dimension tiling)
 *   - TilingKey: schMode-based (REDUCE_STD_WITH_MEAN_V2_FP16=0, FP32=1, BF16=2)
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/reduce_std_with_mean_v2_tiling_data.h"
#include "../op_kernel/reduce_std_with_mean_v2_tiling_key.h"

namespace optiling {

using Ops::Base::CeilAlign;
using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;
using Ops::Base::FloorDiv;
using Ops::Base::GetUbBlockSize;

static constexpr int64_t UB_RESERVED = 8 * 1024;
static constexpr int64_t BUFFER_NUM = 2;
static constexpr int64_t BLOCK_REDUCE_FP32_REPEAT_ELEMENTS = 64;
// Up to 30 segments per row keeps at least 8 rows within the 255-repeat limit.
static constexpr int64_t MAX_SEGMENTED_REDUCE_LENGTH = 1920;

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
                                         ge::DataType& dataType, int64_t& correction, float& eps, bool& invert)
{
    // Welford kernel reads self only (single input, no mean input)
    auto inputSelf = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputSelf);
    auto selfShape = EnsureNotScalar(inputSelf->GetStorageShape());
    totalNum = selfShape.GetShapeSize();
    int64_t rank = selfShape.GetDimNum();

    // Read dim attr to compute the correct reduce length
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

    const std::set<ge::DataType> supported = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();
    if (supported.count(dataType) == 0) {
        OP_LOGE(context, "unsupported dtype %d", static_cast<int>(dataType));
        return ge::GRAPH_FAILED;
    }

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    correction = 0;
    eps = 0.0f;
    invert = false;
    if (attrs) {
        if (attrs->GetInt(5)) {
            correction = *(attrs->GetInt(5));
        }
        if (attrs->GetBool(3)) {
            invert = *(attrs->GetBool(3));
        }
        if (attrs->GetFloat(4)) {
            eps = *(attrs->GetFloat(4));
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetWorkspace(gert::TilingContext* context)
{
    constexpr size_t WORKSPACE_NUM = 1;
    size_t* ws = context->GetWorkspaceSizes(WORKSPACE_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, ws);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ReduceStdWithMeanV2TilingFunc(gert::TilingContext* context)
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
    OP_CHECK_IF(GetShapeAttrsInfo(context, totalNum, reduceLen, dataType, correction, eps, invert) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetShapeAttrsInfo"), return ge::GRAPH_FAILED);

    if (totalNum == 0 || reduceLen == 0) {
        context->SetBlockDim(1);
        uint64_t tk = GET_TPL_TILING_KEY(REDUCE_STD_WITH_MEAN_V2_FP16);
        if (dataType == ge::DT_FLOAT)
            tk = GET_TPL_TILING_KEY(REDUCE_STD_WITH_MEAN_V2_FP32);
        else if (dataType == ge::DT_BF16)
            tk = GET_TPL_TILING_KEY(REDUCE_STD_WITH_MEAN_V2_BF16);
        context->SetTilingKey(tk);
        return ge::GRAPH_SUCCESS;
    }

    int64_t nonReduceNum = totalNum / reduceLen;
    OP_CHECK_IF(SetWorkspace(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "SetWorkspace"), return ge::GRAPH_FAILED);

    ReduceStdWithMeanV2TilingData* tiling = context->GetTilingData<ReduceStdWithMeanV2TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(ReduceStdWithMeanV2TilingData), 0, sizeof(ReduceStdWithMeanV2TilingData)) != EOK,
        OP_LOGE(context, "memset tiling"), return ge::GRAPH_FAILED);

    int64_t ubBlockSize = GetUbBlockSize(context);

    // Multi-core split along non-reduce dimension
    int64_t blockFactor = CeilAlign(CeilDiv(nonReduceNum, coreNum), ubBlockSize);
    if (blockFactor < ubBlockSize) {
        blockFactor = ubBlockSize;
    }
    int64_t usedCoreNum = CeilDiv(nonReduceNum, blockFactor);

    // UB tile length:
    //   self(buffered) + scratch(diff) + work(diff^2) + opt castBuf
    // The mean VECIN queue is gone (Welford computes mean inside kernel).
    int64_t typeSize = (dataType == ge::DT_FLOAT) ? 4 : 2;
    int64_t perElemUB = BUFFER_NUM * typeSize;            // self VECIN queue
    perElemUB += 2 * static_cast<int64_t>(sizeof(float)); // scratch + work (both fp32)
    if (dataType != ge::DT_FLOAT) {
        perElemUB += static_cast<int64_t>(sizeof(float)); // fp16/bf16 cast buffer
    }
    int64_t ubAvail = static_cast<int64_t>(ubSize) - UB_RESERVED;
    int64_t ubForData = ubAvail - static_cast<int64_t>(sizeof(float)); // output buffer
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

    // M-vectorized path enablement (docs/DESIGN_MVECTOR.md §2 + PARALLEL_PLAN.md
    // P1), with the corrections required by the actual BlockReduceSum (vcgadd)
    // semantics:
    //   - the fp32 compute domain limits the per-repeat mask (element count)
    //     to 64, i.e. one vcgadd repeat covers 64 fp32 elements (the design's
    //     "reduceLen*typeSize/32 <= 255" assumed mask in 32B-block units,
    //     which the API does not);
    //   - R <= 64 (v1): one repeat per row (S = 1, mask = R); rows must be
    //     32B aligned (R*typeSize % 32 == 0).
    //   - 64 < R <= 1920 with R % 64 == 0: P1 segmented pass1, S = R/64
    //     segments of 64 elements per row (mask = 64), per-row ReduceSum
    //     pass2. R = 1920 cap: S <= 30 keeps pass1 repeat = curM*S <= 255
    //     with ubM >= 8 (ubM <= 255/S below).
    // Single tile only (R <= ubLength): rows longer than ubLength keep the
    // serial Process() path (R > 1920, e.g. [64,65536], goes there too).
    bool vectorize = false;
    int64_t ubM = 0;
    if (reduceLen <= ubLength && nonReduceNum >= 8) {
        bool v1Range = (reduceLen <= BLOCK_REDUCE_FP32_REPEAT_ELEMENTS) && ((reduceLen * typeSize) % 32 == 0);
        bool p1Range = (reduceLen > BLOCK_REDUCE_FP32_REPEAT_ELEMENTS) && (reduceLen <= MAX_SEGMENTED_REDUCE_LENGTH) &&
                       (reduceLen % BLOCK_REDUCE_FP32_REPEAT_ELEMENTS == 0);
        if (v1Range || p1Range) {
            int64_t S = (reduceLen > BLOCK_REDUCE_FP32_REPEAT_ELEMENTS) ?
                            (reduceLen / BLOCK_REDUCE_FP32_REPEAT_ELEMENTS) :
                            1;
            // UB budget per row (fp32 compute domain):
            //   vecDataBuf R*4 + vecMeanBuf R*4 + tmpBuf 8*S*4
            //   (tmp is one 32B slot per segment: 8*S = max(R/8, 8) floats)
            //   + vecSumBuf/vecM2Buf/vecStateBuf 3 * 8 * 4
            //   + vecRawBuf R*2 (non-fp32)
            int64_t perMElem = 8 * reduceLen +
                               ((reduceLen > BLOCK_REDUCE_FP32_REPEAT_ELEMENTS) ? (reduceLen / 2) : 32) + 96;
            if (dataType != ge::DT_FLOAT) {
                perMElem += 2 * reduceLen;
            }
            int64_t mTile = FloorDiv(ubAvail - 2 * 1024, perMElem);
            // pass1 repeat = curM * S <= 255 (BlockReduceSum repeatTime cap),
            // so ubM is capped at 255/S.
            int64_t maxM = 255 / S;
            if (mTile > maxM) {
                mTile = maxM;
            }
            // ProcessVectorized keeps the per-batch state split as
            // meanVec [0, ubM) + m2Vec [ubM, 2*ubM) inside one fp32 buffer;
            // the merge step runs vector ops on state[ubM], whose byte
            // offset (ubM*4) must be 32B-aligned on real hardware
            // (tikicpulib does not check this). Force ubM to a multiple of 8
            // floats, e.g. 255 -> 248, otherwise the vadd on state[ubM]
            // traps with "UB address not aligned" (ATK 910B: all vectorize
            // cases crash at the same pc).
            mTile = FloorAlign(mTile, static_cast<int64_t>(8));
            if (mTile >= 8) {
                vectorize = true;
                ubM = mTile;
            }
        }
    }

    tiling->totalNonReduce = nonReduceNum;
    tiling->reduceLength = reduceLen;
    tiling->blockFactor = blockFactor;
    tiling->ubLength = ubLength;
    tiling->correction = correction;
    tiling->eps = eps;
    tiling->invert = invert;
    tiling->vectorize = vectorize;
    tiling->ubM = ubM;

    context->SetBlockDim(usedCoreNum);

    uint64_t tk = GET_TPL_TILING_KEY(REDUCE_STD_WITH_MEAN_V2_FP16);
    if (dataType == ge::DT_FLOAT)
        tk = GET_TPL_TILING_KEY(REDUCE_STD_WITH_MEAN_V2_FP32);
    else if (dataType == ge::DT_BF16)
        tk = GET_TPL_TILING_KEY(REDUCE_STD_WITH_MEAN_V2_BF16);
    context->SetTilingKey(tk);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForReduceStdWithMeanV2([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReduceStdWithMeanV2CompileInfo {};
IMPL_OP_OPTILING(ReduceStdWithMeanV2)
    .Tiling(ReduceStdWithMeanV2TilingFunc)
    .TilingParse<ReduceStdWithMeanV2CompileInfo>(TilingParseForReduceStdWithMeanV2);

} // namespace optiling
