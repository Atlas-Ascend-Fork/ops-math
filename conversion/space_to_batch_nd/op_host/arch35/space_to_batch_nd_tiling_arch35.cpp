/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <limits>
#include "register/op_impl_registry.h"
#include "conversion/space_to_batch_nd/op_kernel/arch35/space_to_batch_nd_tiling_data.h"
#include "conversion/space_to_batch_nd/op_kernel/arch35/space_to_batch_nd_tiling_key.h"
#include "platform/platform_ascendc.h"
#include "op_host/util/platform_util.h"
#include "op_host/util/math_util.h"
#include "op_host/util/const_util.h"

namespace optiling {

namespace {
// Checked arithmetic helpers: return false on overflow.
inline bool SafeAddI64(int64_t a, int64_t b, int64_t& r) { return !__builtin_add_overflow(a, b, &r); }
inline bool SafeMulI64(int64_t a, int64_t b, int64_t& r) { return !__builtin_mul_overflow(a, b, &r); }
inline bool SafeMulU64(uint64_t a, uint64_t b, uint64_t& r) { return !__builtin_mul_overflow(a, b, &r); }

inline bool IsSupportedXDataType(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_INT8:
        case ge::DT_UINT8:
        case ge::DT_INT16:
        case ge::DT_UINT16:
        case ge::DT_INT32:
        case ge::DT_UINT32:
        case ge::DT_INT64:
        case ge::DT_UINT64:
        case ge::DT_BF16:
        case ge::DT_FLOAT16:
        case ge::DT_FLOAT:
        case ge::DT_DOUBLE:
        case ge::DT_BOOL:
        case ge::DT_COMPLEX32:
        case ge::DT_COMPLEX64:
            return true;
        default:
            return false;
    }
}
} // namespace

static constexpr size_t INPUT_IDX_X = 0;
static constexpr size_t INPUT_IDX_BS = 1;
static constexpr size_t INPUT_IDX_PADS = 2;
static constexpr size_t OUTPUT_IDX_Y = 0;
static constexpr uint32_t MAX_BUFFER_SIZE = 64 * 1024;
static constexpr uint32_t BUFFER_NUM = 2;
static constexpr double MIN_CORE_UTIL_RATIO = 0.8;

class SpaceToBatchNDTiling {
public:
    explicit SpaceToBatchNDTiling(gert::TilingContext* context) : context_(context) {}
    ge::graphStatus DoTiling();

private:
    ge::graphStatus GetSocInfo();
    ge::graphStatus GetParams();
    ge::graphStatus ComputeShapes();
    ge::graphStatus ComputeAndSetTiling();
    bool IsEmptyOutput() const;

    void InitElementSizes();
    ge::graphStatus DoCacheLineTiling(int64_t alignedInner, int32_t& startAxis, int64_t& clInnerProduct,
                                      int64_t& minUbFactor, int64_t& outerProduct);
    ge::graphStatus DoTiling(int64_t alignedInner, int32_t startAxis, int64_t minUbFactor, int64_t clInnerProduct,
                             int64_t outerProduct);
    ge::graphStatus SetTilingData();

    gert::TilingContext* context_;
    uint32_t ubSize_{0};
    uint32_t coreNum_{0};
    uint32_t ubBlockSize_{0};
    uint32_t cacheLineSize_{0};

    int64_t inShape_[MAX_RANK]{0};
    int64_t outShape_[MAX_RANK]{0};
    uint8_t rank_{0};
    uint8_t numSpatialDims_{0};
    int64_t blockShape_[MAX_SPATIAL]{0};
    int64_t padTop_[MAX_SPATIAL]{0};
    int64_t padBottom_[MAX_SPATIAL]{0};
    int64_t paddedShape_[MAX_SPATIAL]{0};
    int64_t batchProd_{0};
    int64_t innerSize_{0};
    int32_t dSize_{0};

    uint32_t bufferSize_{0};
    uint32_t bufferSizeElements_{0};
    uint32_t ubBlockElements_{0};
    uint32_t cacheLineElements_{0};

    uint8_t ubAxis_{0};
    uint32_t ubFactor_{0};
    uint64_t totalCount_{0};
    uint64_t perCoreCount_{0};
};

ge::graphStatus SpaceToBatchNDTiling::GetSocInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum_ = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((coreNum_ == 0U), OP_LOGE(context_, "coreNum is 0"), return ge::GRAPH_FAILED);
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF((ubSize == 0U), OP_LOGE(context_, "ubSize is 0"), return ge::GRAPH_FAILED);
    ubSize_ = static_cast<uint32_t>(ubSize);
    cacheLineSize_ = Ops::Base::GetCacheLineSize(context_);
    OP_CHECK_IF((cacheLineSize_ == 0U), OP_LOGE(context_, "Failed to get cache line size."), return ge::GRAPH_FAILED);
    ubBlockSize_ = Ops::Base::GetUbBlockSize(context_);
    OP_CHECK_IF((ubBlockSize_ == 0U), OP_LOGE(context_, "Failed to get ub block size."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SpaceToBatchNDTiling::GetParams()
{
    auto inputValueDesc = context_->GetInputDesc(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputValueDesc);
    auto inputDataType = inputValueDesc->GetDataType();
    OP_CHECK_IF(!IsSupportedXDataType(inputDataType),
                OP_LOGE(context_, "input x dtype %d is not supported", static_cast<int>(inputDataType)),
                return ge::GRAPH_FAILED);
    dSize_ = ge::GetSizeByDataType(inputDataType);
    OP_CHECK_IF(dSize_ <= 0, OP_LOGE(context_, "data size should be positive"), return ge::GRAPH_FAILED);

    // The kernel assumes contiguous ND storage. Reject any other storage format
    // instead of letting an illegal format silently flow into the ND data path.
    // The kernel consumes ND **storage**. Origin format may legitimately differ after
    // framework format conversion, so only the storage format is constrained here.
    for (size_t idx = INPUT_IDX_X; idx <= INPUT_IDX_PADS; ++idx) {
        auto inputDesc = context_->GetInputDesc(idx);
        OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
        OP_CHECK_IF(inputDesc->GetStorageFormat() != ge::FORMAT_ND,
                    OP_LOGE(context_, "input[%zu] format %d is not supported, only FORMAT_ND is allowed", idx,
                            static_cast<int>(inputDesc->GetStorageFormat())),
                    return ge::GRAPH_FAILED);
    }

    // The kernel writes elements with the input dtype. If y declares another
    // dtype, the output buffer is sized with a different element width and the
    // kernel overruns it, so y dtype must match x dtype (DT_UNDEFINED is left to
    // the graph-side InferDataType callback).
    auto outputDesc = context_->GetOutputDesc(OUTPUT_IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    // The kernel consumes ND **storage**. Origin format may legitimately differ after
    // framework format conversion, so only the storage format is constrained here.
    OP_CHECK_IF(outputDesc->GetStorageFormat() != ge::FORMAT_ND,
                OP_LOGE(context_, "output y format %d is not supported, only FORMAT_ND is allowed",
                        static_cast<int>(outputDesc->GetStorageFormat())),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(outputDesc->GetDataType() != inputDataType && outputDesc->GetDataType() != ge::DT_UNDEFINED,
                OP_LOGE(context_, "output y dtype %d must match input x dtype %d",
                        static_cast<int>(outputDesc->GetDataType()), static_cast<int>(inputDataType)),
                return ge::GRAPH_FAILED);

    auto xInputShape = context_->GetInputShape(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xInputShape);
    auto xShape = xInputShape->GetStorageShape();
    const size_t inputRank = xShape.GetDimNum();
    OP_CHECK_IF(inputRank < 2 || inputRank > static_cast<size_t>(MAX_INPUT_RANK),
                OP_LOGE(context_, "invalid rank %zu, must be in [2, %d]", inputRank, MAX_INPUT_RANK),
                return ge::GRAPH_FAILED);

    auto bsTensor = context_->GetInputTensor(INPUT_IDX_BS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, bsTensor);
    const int64_t blockShapeLen = bsTensor->GetShapeSize();
    OP_CHECK_IF(blockShapeLen < 1 || blockShapeLen > static_cast<int64_t>(MAX_SPATIAL),
                OP_LOGE(context_, "invalid numSpatialDims %ld, must be in [1, %d]", blockShapeLen, MAX_SPATIAL),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(blockShapeLen >= static_cast<int64_t>(inputRank),
                OP_LOGE(context_, "block_shape length M=%ld must be less than x rank N=%zu", blockShapeLen, inputRank),
                return ge::GRAPH_FAILED);

    rank_ = static_cast<uint8_t>(inputRank);
    numSpatialDims_ = static_cast<uint8_t>(blockShapeLen);

    gert::Shape bsVec;
    OP_CHECK_IF(!Ops::Base::GetConstIntToShape(context_, INPUT_IDX_BS, bsVec),
                OP_LOGE(context_, "get block_shape const data failed"), return ge::GRAPH_FAILED);

    auto padsShape = context_->GetInputShape(INPUT_IDX_PADS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, padsShape);
    auto padsStorageShape = padsShape->GetStorageShape();
    OP_CHECK_IF(padsStorageShape.GetDimNum() != 2,
                OP_LOGE(context_, "paddings shape dim num=%zu must be 2, paddings shape must be [%u,2]",
                        padsStorageShape.GetDimNum(), static_cast<uint32_t>(numSpatialDims_)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(padsStorageShape.GetDim(0) != static_cast<int64_t>(numSpatialDims_),
                OP_LOGE(context_,
                        "paddings shape[0]=%ld must be equal to numSpatialDims=%u, "
                        "paddings shape must be [%u,2]",
                        padsStorageShape.GetDim(0), static_cast<uint32_t>(numSpatialDims_),
                        static_cast<uint32_t>(numSpatialDims_)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(padsStorageShape.GetDim(1) != 2,
                OP_LOGE(context_, "paddings shape[1]=%ld must be 2, paddings shape must be [%u,2]",
                        padsStorageShape.GetDim(1), static_cast<uint32_t>(numSpatialDims_)),
                return ge::GRAPH_FAILED);

    gert::Shape padsVec;
    OP_CHECK_IF(!Ops::Base::GetConstIntToShape(context_, INPUT_IDX_PADS, padsVec),
                OP_LOGE(context_, "get paddings const data failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        padsVec.GetDimNum() < static_cast<size_t>(numSpatialDims_) * 2,
        OP_LOGE(context_,
                "paddings element count=%zu is less than 2*numSpatialDims=%zu, "
                "paddings shape must be [%u,2]",
                padsVec.GetDimNum(), static_cast<size_t>(numSpatialDims_) * 2, static_cast<uint32_t>(numSpatialDims_)),
        return ge::GRAPH_FAILED);

    for (size_t i = 0; i < numSpatialDims_; i++) {
        blockShape_[i] = bsVec.GetDim(i);
        OP_CHECK_IF(blockShape_[i] <= 0, OP_LOGE(context_, "block_shape[%zu]=%ld must be > 0", i, blockShape_[i]),
                    return ge::GRAPH_FAILED);
        padTop_[i] = padsVec.GetDim(i * 2);
        padBottom_[i] = padsVec.GetDim(i * 2 + 1);
        OP_CHECK_IF(padTop_[i] < 0 || padBottom_[i] < 0, OP_LOGE(context_, "paddings must be non-negative"),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SpaceToBatchNDTiling::ComputeShapes()
{
    auto xInputShape = context_->GetInputShape(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xInputShape);
    auto xShape = xInputShape->GetStorageShape();

    int64_t batch = xShape.GetDim(0);
    batchProd_ = batch;
    for (size_t i = 0; i < numSpatialDims_; i++) {
        int64_t spatial = xShape.GetDim(i + 1);
        inShape_[i + 1] = spatial;
        int64_t paddedDim = 0;
        if (!SafeAddI64(spatial, padTop_[i], paddedDim) || !SafeAddI64(paddedDim, padBottom_[i], paddedDim)) {
            OP_LOGE(context_, "padded spatial overflow at axis %zu: spatial=%ld padTop=%ld padBottom=%ld", i, spatial,
                    padTop_[i], padBottom_[i]);
            return ge::GRAPH_FAILED;
        }
        paddedShape_[i] = paddedDim;
        OP_CHECK_IF(paddedShape_[i] % blockShape_[i] != 0,
                    OP_LOGE(context_, "padded spatial[%zu]=%ld not divisible by block_shape[%zu]=%ld", i,
                            paddedShape_[i], i, blockShape_[i]),
                    return ge::GRAPH_FAILED);
        if (!SafeMulI64(batchProd_, blockShape_[i], batchProd_)) {
            OP_LOGE(context_, "batch product overflow at axis %zu: batchProd=%ld blockShape=%ld", i, batchProd_,
                    blockShape_[i]);
            return ge::GRAPH_FAILED;
        }
    }

    innerSize_ = 1;
    for (size_t i = numSpatialDims_ + 1; i < rank_; i++) {
        int64_t dim = xShape.GetDim(i);
        if (!SafeMulI64(innerSize_, dim, innerSize_)) {
            OP_LOGE(context_, "inner size overflow at axis %zu: innerSize=%ld dim=%ld", i, innerSize_, dim);
            return ge::GRAPH_FAILED;
        }
    }

    inShape_[0] = batch;
    inShape_[numSpatialDims_ + 1] = innerSize_;

    outShape_[0] = batchProd_;
    for (size_t i = 0; i < numSpatialDims_; i++) {
        outShape_[i + 1] = paddedShape_[i] / blockShape_[i];
    }
    outShape_[numSpatialDims_ + 1] = innerSize_;

    // rank_ here is the internal flattened rank (batch + M spatial dims + merged inner dim),
    // not the rank of the input/output tensors.
    rank_ = numSpatialDims_ + 2;

    return ge::GRAPH_SUCCESS;
}

bool SpaceToBatchNDTiling::IsEmptyOutput() const
{
    // Judge by outShape_ rather than inShape_: when a spatial dim of the input is 0 but padding
    // makes the output non-empty, the zero-padded region still has to be written out.
    for (uint8_t i = 0; i < rank_; i++) {
        if (outShape_[i] == 0) {
            return true;
        }
    }
    return false;
}

void SpaceToBatchNDTiling::InitElementSizes()
{
    bufferSize_ = std::min(ubSize_ / BUFFER_NUM, MAX_BUFFER_SIZE);
    bufferSizeElements_ = bufferSize_ / static_cast<uint32_t>(dSize_);
    ubBlockElements_ = ubBlockSize_ / static_cast<uint32_t>(dSize_);
    cacheLineElements_ = cacheLineSize_ / static_cast<uint32_t>(dSize_);
}

ge::graphStatus SpaceToBatchNDTiling::DoCacheLineTiling(int64_t alignedInner, int32_t& startAxis,
                                                        int64_t& clInnerProduct, int64_t& minUbFactor,
                                                        int64_t& outerProduct)
{
    int32_t lastAxis = static_cast<int32_t>(rank_) - 1;
    startAxis = lastAxis;
    clInnerProduct = 1;
    bool stopped = false;
    for (int32_t ax = lastAxis; ax >= 0; --ax) {
        int64_t dimSize = (ax == lastAxis) ? alignedInner : outShape_[ax];
        int64_t clProduct = 0;
        if (!SafeMulI64(dimSize, clInnerProduct, clProduct)) {
            // 溢出代表乘积已远超 cache line，直接以当前轴作为 startAxis
            startAxis = ax;
            stopped = true;
            break;
        }
        if (clProduct >= static_cast<int64_t>(cacheLineElements_)) {
            startAxis = ax;
            stopped = true;
            break;
        }
        if (!SafeMulI64(clInnerProduct, dimSize, clInnerProduct)) {
            OP_LOGE(context_, "clInnerProduct overflow at axis %d: clInnerProduct=%ld dimSize=%ld", ax, clInnerProduct,
                    dimSize);
            return ge::GRAPH_FAILED;
        }
    }

    if (!stopped) {
        // 全轴乘积仍小于 cacheline：startAxis 保持 lastAxis，其后方没有轴，inner 乘积应为 1
        startAxis = lastAxis;
        clInnerProduct = 1;
    }

    int64_t startAxisStep = (startAxis == lastAxis) ? static_cast<int64_t>(ubBlockElements_) : 1;
    int64_t clMinFactor = Ops::Base::CeilDiv(static_cast<int64_t>(cacheLineElements_), clInnerProduct);
    minUbFactor = Ops::Base::CeilAlign(std::max(startAxisStep, clMinFactor), startAxisStep);

    outerProduct = 1;
    for (int32_t ax = 0; ax < startAxis; ++ax) {
        if (!SafeMulI64(outerProduct, outShape_[ax], outerProduct)) {
            OP_LOGE(context_, "outerProduct overflow at axis %d: outerProduct=%ld dim=%ld", ax, outerProduct,
                    outShape_[ax]);
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SpaceToBatchNDTiling::DoTiling(int64_t alignedInner, int32_t startAxis, int64_t minUbFactor,
                                               int64_t clInnerProduct, int64_t outerProduct)
{
    int32_t lastAxis = static_cast<int32_t>(rank_) - 1;
    uint8_t bestAxis = 0;
    uint32_t bestFactor = 1;
    uint64_t bestTotalCount = 1;
    uint64_t bestRealCore = 1;

    int64_t cumInnerProduct = clInnerProduct;
    for (int32_t ax = startAxis; ax >= 1; --ax) { // Only tile on spatial dimensions (ax >= 1)
        int64_t dimSize = (ax == lastAxis) ? alignedInner : outShape_[ax];
        int64_t step = (ax == lastAxis) ? static_cast<int64_t>(ubBlockElements_) : 1;
        int64_t factorStart = (ax == startAxis) ? minUbFactor : step;

        for (int64_t factor = factorStart; factor <= dimSize;) {
            // 统一公式：cumInnerProduct 含 tail axis align, × dSize 即得每 factor 占用的字节数
            int64_t bufferBytesUsed = 0;
            if (!SafeMulI64(factor, cumInnerProduct, bufferBytesUsed) ||
                !SafeMulI64(bufferBytesUsed, static_cast<int64_t>(dSize_), bufferBytesUsed)) {
                // factor 越大占用只会更多，溢出即视为该 factor 超 UB，后续候选无需再试
                break;
            }
            if (bufferBytesUsed > static_cast<int64_t>(bufferSize_)) {
                break;
            }
            uint64_t totalCount = 0;
            if (SafeMulU64(static_cast<uint64_t>(outerProduct),
                           static_cast<uint64_t>(Ops::Base::CeilDiv(dimSize, factor)), totalCount)) {
                uint64_t perCoreCount = Ops::Base::CeilDiv(totalCount, static_cast<uint64_t>(coreNum_));
                uint64_t realCoreNum = Ops::Base::CeilDiv(totalCount, perCoreCount);
                if (static_cast<double>(realCoreNum) >= static_cast<double>(coreNum_) * MIN_CORE_UTIL_RATIO ||
                    realCoreNum > bestRealCore) {
                    bestAxis = static_cast<uint8_t>(ax);
                    bestFactor = static_cast<uint32_t>(factor);
                    bestTotalCount = totalCount;
                    if (realCoreNum > bestRealCore) {
                        bestRealCore = realCoreNum;
                    }
                }
            }
            // totalCount 溢出时跳过该候选，继续尝试更大的 factor
            if (step > dimSize - factor) {
                break;
            }
            factor += step;
        }

        if (ax > 1) {
            outerProduct /= outShape_[ax - 1];
            if (!SafeMulI64(cumInnerProduct, dimSize, cumInnerProduct)) {
                OP_LOGE(context_, "cumInnerProduct overflow at axis %d: cumInnerProduct=%ld dimSize=%ld", ax,
                        cumInnerProduct, dimSize);
                return ge::GRAPH_FAILED;
            }
        }
    }

    if (bestAxis == 0) {
        uint64_t fallbackTotalCount = 1;
        for (int32_t ax = 0; ax < lastAxis; ++ax) {
            if (!SafeMulU64(fallbackTotalCount, static_cast<uint64_t>(outShape_[ax]), fallbackTotalCount)) {
                OP_LOGE(context_, "fallback total count overflow at axis %d: totalCount=%lu dim=%ld", ax,
                        fallbackTotalCount, outShape_[ax]);
                return ge::GRAPH_FAILED;
            }
        }
        ubAxis_ = static_cast<uint8_t>(lastAxis);
        ubFactor_ = static_cast<uint32_t>(alignedInner);
        totalCount_ = fallbackTotalCount;
        return ge::GRAPH_SUCCESS;
    }

    ubAxis_ = bestAxis;
    ubFactor_ = bestFactor;
    totalCount_ = bestTotalCount;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SpaceToBatchNDTiling::SetTilingData()
{
    if (totalCount_ == 0) {
        perCoreCount_ = 1;
    } else {
        perCoreCount_ = Ops::Base::CeilDiv(totalCount_, static_cast<uint64_t>(coreNum_));
    }
    uint64_t realCoreNum = (totalCount_ == 0) ? 1 : Ops::Base::CeilDiv(totalCount_, perCoreCount_);

    const uint64_t kInt64Max = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
    OP_CHECK_IF(
        totalCount_ > kInt64Max || perCoreCount_ > kInt64Max,
        OP_LOGE(context_, "tiling count exceeds int64: totalCount=%lu perCoreCount=%lu", totalCount_, perCoreCount_),
        return ge::GRAPH_FAILED);

    OP_LOGI(context_, "SpaceToBatchND tiling: ubAxis=%u ubFactor=%u totalCount=%lu perCoreCount=%lu realCoreNum=%lu",
            ubAxis_, ubFactor_, totalCount_, perCoreCount_, realCoreNum);

    auto tilingData = context_->GetTilingData<SpaceToBatchNDTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, tilingData);

    tilingData->rank = rank_;
    for (size_t i = 0; i < rank_; i++) {
        tilingData->inShape[i] = inShape_[i];
        tilingData->outShape[i] = outShape_[i];
    }
    tilingData->totalCount = static_cast<int64_t>(totalCount_);
    tilingData->perCoreCount = static_cast<int64_t>(perCoreCount_);
    tilingData->ubAxis = ubAxis_;
    tilingData->ubFactor = ubFactor_;
    tilingData->bufferSize = bufferSize_;
    tilingData->numSpatialDims = numSpatialDims_;
    for (size_t i = 0; i < numSpatialDims_; i++) {
        tilingData->blockShape[i] = blockShape_[i];
        tilingData->padTop[i] = padTop_[i];
        tilingData->padBottom[i] = padBottom_[i];
    }

    OP_LOGI(context_,
            "SpaceToBatchND TilingData: rank=%ld inShape=[%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld] "
            "outShape=[%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld] totalCount=%ld perCoreCount=%ld ubAxis=%ld "
            "ubFactor=%ld bufferSize=%ld numSpatialDims=%ld blockShape=[%ld,%ld,%ld,%ld,%ld,%ld,%ld] "
            "padTop=[%ld,%ld,%ld,%ld,%ld,%ld,%ld] padBottom=[%ld,%ld,%ld,%ld,%ld,%ld,%ld]",
            tilingData->rank, tilingData->inShape[0], tilingData->inShape[1], tilingData->inShape[2],
            tilingData->inShape[3], tilingData->inShape[4], tilingData->inShape[5], tilingData->inShape[6],
            tilingData->inShape[7], tilingData->inShape[8], tilingData->outShape[0], tilingData->outShape[1],
            tilingData->outShape[2], tilingData->outShape[3], tilingData->outShape[4], tilingData->outShape[5],
            tilingData->outShape[6], tilingData->outShape[7], tilingData->outShape[8], tilingData->totalCount,
            tilingData->perCoreCount, tilingData->ubAxis, tilingData->ubFactor, tilingData->bufferSize,
            tilingData->numSpatialDims, tilingData->blockShape[0], tilingData->blockShape[1], tilingData->blockShape[2],
            tilingData->blockShape[3], tilingData->blockShape[4], tilingData->blockShape[5], tilingData->blockShape[6],
            tilingData->padTop[0], tilingData->padTop[1], tilingData->padTop[2], tilingData->padTop[3],
            tilingData->padTop[4], tilingData->padTop[5], tilingData->padTop[6], tilingData->padBottom[0],
            tilingData->padBottom[1], tilingData->padBottom[2], tilingData->padBottom[3], tilingData->padBottom[4],
            tilingData->padBottom[5], tilingData->padBottom[6]);

    OP_CHECK_IF(ubAxis_ >= rank_,
                OP_LOGE(context_, "invalid ubAxis %u for rank %u", static_cast<uint32_t>(ubAxis_),
                        static_cast<uint32_t>(rank_)),
                return ge::GRAPH_FAILED);
    // key = rank - ubAxis: KeyN means CopyInKeyN handles the N consecutive axes from ubAxis to inner.
    // key=1 corresponds to ubAxis=rank-1 (the inner axis), which is a reachable fallback path.
    uint64_t tilingKey = GET_TPL_TILING_KEY(static_cast<uint64_t>(rank_) - static_cast<uint64_t>(ubAxis_));

    OP_LOGI(context_, "tilingKey is %lu, ubAxis %u", tilingKey, ubAxis_);
    context_->SetTilingKey(tilingKey);
    context_->SetBlockDim(realCoreNum);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SpaceToBatchNDTiling::ComputeAndSetTiling()
{
    InitElementSizes();
    OP_CHECK_IF(bufferSizeElements_ == 0U || ubBlockElements_ == 0U || cacheLineElements_ == 0U,
                OP_LOGE(context_, "dSize %d is too large for UB buffer/block/cacheline", dSize_),
                return ge::GRAPH_FAILED);

    int64_t alignedInner = Ops::Base::CeilAlign(outShape_[rank_ - 1], static_cast<int64_t>(ubBlockElements_));
    int32_t startAxis;
    int64_t clInnerProduct, minUbFactor, outerProduct;
    auto ret = DoCacheLineTiling(alignedInner, startAxis, clInnerProduct, minUbFactor, outerProduct);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context_, "DoCacheLineTiling failed"), return ret);
    ret = DoTiling(alignedInner, startAxis, minUbFactor, clInnerProduct, outerProduct);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context_, "DoTiling failed"), return ret);
    return SetTilingData();
}

ge::graphStatus SpaceToBatchNDTiling::DoTiling()
{
    auto ret = GetSocInfo();
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context_, "GetSocInfo failed"), return ret);
    ret = GetParams();
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context_, "GetParams failed"), return ret);
    ret = ComputeShapes();
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context_, "ComputeShapes failed"), return ret);
    if (IsEmptyOutput()) {
        ubAxis_ = static_cast<uint8_t>(rank_ - 1);
        ubFactor_ = 1;
        totalCount_ = 0;
        InitElementSizes();
        ret = SetTilingData();
    } else {
        ret = ComputeAndSetTiling();
    }
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context_, "ComputeAndSetTiling failed"), return ret);

    size_t* workSpaceSize = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workSpaceSize);
    workSpaceSize[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SpaceToBatchNDTilingFunc(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin the tiling process for Arch35 architecture");
    SpaceToBatchNDTiling tiling(context);
    return tiling.DoTiling();
}

static ge::graphStatus TilingParseForSpaceToBatchND([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SpaceToBatchND)
    .Tiling(SpaceToBatchNDTilingFunc)
    .TilingInputsDataDependency({INPUT_IDX_BS, INPUT_IDX_PADS})
    .TilingParse<SpaceToBatchNDCompileInfo>(TilingParseForSpaceToBatchND);
} // namespace optiling
