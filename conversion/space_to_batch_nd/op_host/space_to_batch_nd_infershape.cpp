/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "op_host/util/const_util.h"
#include "op_host/util/shape_util.h"

using namespace ge;
namespace ops {
static constexpr size_t INPUT_IDX_X = 0;
static constexpr size_t INPUT_IDX_BS = 1;
static constexpr size_t INPUT_IDX_PADS = 2;
static constexpr size_t OUTPUT_IDX_Y = 0;
static constexpr size_t PADS_DIM2 = 2;
static constexpr int64_t MAX_RANK = 8;

namespace {
// Checked arithmetic helpers: return false on overflow.
inline bool SafeAddI64(int64_t a, int64_t b, int64_t& r) { return !__builtin_add_overflow(a, b, &r); }
inline bool SafeMulI64(int64_t a, int64_t b, int64_t& r) { return !__builtin_mul_overflow(a, b, &r); }
} // namespace

class SpaceToBatchNDInferShapeHelper {
public:
    explicit SpaceToBatchNDInferShapeHelper(gert::InferShapeContext* context) : context_(context) {}

    ge::graphStatus Inference();

private:
    ge::graphStatus Init();
    ge::graphStatus CheckAndInfer();

    gert::InferShapeContext* context_;
    const gert::Shape* xShape_{nullptr};
    gert::Shape* yShape_{nullptr};
    size_t N_{0};
    gert::Shape blockVec_;
    gert::Shape padsVec_;
    // 值依赖输入在编译期可能不是常量：读不到数据时降级为 UNKNOWN_DIM(-1)，不中断 shape 推导。
    bool blockShapeIsConst_{false};
    bool paddingsIsConst_{false};
};

ge::graphStatus SpaceToBatchNDInferShapeHelper::Init()
{
    const gert::Tensor* bsTensor = context_->GetInputTensor(INPUT_IDX_BS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, bsTensor);
    N_ = bsTensor->GetShapeSize();
    OP_CHECK_IF(
        static_cast<int64_t>(N_) < 0,
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context_->GetNodeName(), "block_shape", std::to_string(N_).c_str(),
                                                  "The element count of block_shape must be non-negative"),
        return ge::GRAPH_FAILED);

    // block_shape 允许在编译期不是常量：读不到数据时后续把相关输出维降级为 -1，不再直接失败。
    blockShapeIsConst_ = Ops::Base::GetConstIntToShape<gert::InferShapeContext>(context_, INPUT_IDX_BS, blockVec_);
    if (!blockShapeIsConst_) {
        OP_LOGI(context_->GetNodeName(),
                "block_shape is not a compile-time constant, related output dims will be inferred as unknown(-1).");
    }

    const gert::Tensor* padsTensor = context_->GetInputTensor(INPUT_IDX_PADS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, padsTensor);
    auto padsShape = context_->GetInputShape(INPUT_IDX_PADS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, padsShape);
    OP_CHECK_IF(padsShape->GetDimNum() != 2 || padsShape->GetDim(0) != static_cast<int64_t>(N_) ||
                    padsShape->GetDim(1) != static_cast<int64_t>(PADS_DIM2),
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    context_->GetNodeName(), "paddings", Ops::Base::ToString(*padsShape).c_str(),
                    "The shape of paddings must be [" + std::to_string(N_) + ", 2]"),
                return ge::GRAPH_FAILED);

    // paddings 同样允许在编译期不是常量：读不到数据时相关空间维降级为 -1。
    paddingsIsConst_ = Ops::Base::GetConstIntToShape<gert::InferShapeContext>(context_, INPUT_IDX_PADS, padsVec_);
    if (paddingsIsConst_) {
        OP_CHECK_IF(
            padsVec_.GetDimNum() < N_ * PADS_DIM2,
            OP_LOGE(context_, "paddings element count=%zu is less than %zu", padsVec_.GetDimNum(), N_ * PADS_DIM2),
            return ge::GRAPH_FAILED);
        for (size_t i = 0; i < N_ * PADS_DIM2; ++i) {
            OP_CHECK_IF(padsVec_.GetDim(i) < 0,
                        OP_LOGE(context_, "paddings[%zu]=%ld must be non-negative", i, padsVec_.GetDim(i)),
                        return ge::GRAPH_FAILED);
        }
    } else {
        OP_LOGI(context_->GetNodeName(),
                "paddings is not a compile-time constant, related output dims will be inferred as unknown(-1).");
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SpaceToBatchNDInferShapeHelper::CheckAndInfer()
{
    auto rank = xShape_->GetDimNum();
    OP_CHECK_IF(static_cast<size_t>(rank) < N_ + 1 || rank > MAX_RANK,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    context_->GetNodeName(), "x", std::to_string(rank).c_str(),
                    "The rank of x must be in [" + std::to_string(N_ + 1) + ", " + std::to_string(MAX_RANK) + "]"),
                return ge::GRAPH_FAILED);

    // block_shape 非常量时 batch 倍率不可知，输出 batch 维退化为 -1。
    int64_t batchMul = 1;
    if (blockShapeIsConst_) {
        for (size_t i = 0; i < N_; i++) {
            int64_t bs = blockVec_.GetDim(i);
            OP_CHECK_IF(bs <= 0,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "block_shape",
                                                              std::to_string(bs).c_str(),
                                                              "Each value of block_shape must be greater than 0"),
                        return ge::GRAPH_FAILED);
            int64_t nextBatchMul = 0;
            OP_CHECK_IF(
                !SafeMulI64(batchMul, bs, nextBatchMul),
                OP_LOGE(context_, "batch multiplier overflow: batchMul=%ld * block_shape[%zu]=%ld", batchMul, i, bs),
                return ge::GRAPH_FAILED);
            batchMul = nextBatchMul;
        }
    }

    yShape_->SetDimNum(0);

    int64_t firstDim = xShape_->GetDim(0);
    if (firstDim == -1 || !blockShapeIsConst_) {
        yShape_->AppendDim(-1);
    } else {
        int64_t outFirstDim = 0;
        OP_CHECK_IF(!SafeMulI64(firstDim, batchMul, outFirstDim),
                    OP_LOGE(context_, "first dim overflow: x dim[0]=%ld * batchMul=%ld", firstDim, batchMul),
                    return ge::GRAPH_FAILED);
        yShape_->AppendDim(outFirstDim);
    }

    for (size_t i = 0; i < N_; i++) {
        int64_t xDim = xShape_->GetDim(i + 1);
        // x 该维未知，或 block_shape/paddings 非常量导致整除关系不可知时，该空间维降级为 -1。
        if (xDim == -1 || !blockShapeIsConst_ || !paddingsIsConst_) {
            yShape_->AppendDim(-1);
            continue;
        }
        int64_t bs = blockVec_.GetDim(i);
        int64_t dimWithTop = 0;
        int64_t paddedDim = 0;
        OP_CHECK_IF(!SafeAddI64(xDim, padsVec_.GetDim(i * 2), dimWithTop),
                    OP_LOGE(context_, "spatial dim[%zu]=%ld + paddings[%zu]=%ld overflow", i, xDim, i * 2,
                            padsVec_.GetDim(i * 2)),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(!SafeAddI64(dimWithTop, padsVec_.GetDim(i * 2 + 1), paddedDim),
                    OP_LOGE(context_, "spatial dim[%zu] + paddings[%zu]=%ld overflow", i, i * 2 + 1,
                            padsVec_.GetDim(i * 2 + 1)),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(paddedDim % bs != 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "paddings",
                                                          std::to_string(paddedDim).c_str(),
                                                          "The padded spatial dim must be divisible by block_shape"),
                    return ge::GRAPH_FAILED);
        yShape_->AppendDim(paddedDim / bs);
    }
    for (size_t i = N_ + 1; i < static_cast<size_t>(rank); i++) {
        yShape_->AppendDim(xShape_->GetDim(i));
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SpaceToBatchNDInferShapeHelper::Inference()
{
    xShape_ = context_->GetInputShape(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShape_);
    yShape_ = context_->GetOutputShape(OUTPUT_IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShape_);
    if (Ops::Base::IsUnknownRank(*xShape_)) {
        Ops::Base::SetUnknownRank(*yShape_);
        OP_LOGI(context_->GetNodeName(), "SpaceToBatchND output shape: %s.", Ops::Base::ToString(*yShape_).c_str());
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(Init() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "init failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckAndInfer() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "infer failed"), return ge::GRAPH_FAILED);
    OP_LOGI(context_->GetNodeName(), "SpaceToBatchND output shape: %s.", Ops::Base::ToString(*yShape_).c_str());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Infershape4SpaceToBatchND(gert::InferShapeContext* context)
{
    SpaceToBatchNDInferShapeHelper helper(context);
    return helper.Inference();
}

IMPL_OP_INFERSHAPE(SpaceToBatchND)
    .InferShape(Infershape4SpaceToBatchND)
    .InputsDataDependency({INPUT_IDX_BS, INPUT_IDX_PADS});
} // namespace ops
