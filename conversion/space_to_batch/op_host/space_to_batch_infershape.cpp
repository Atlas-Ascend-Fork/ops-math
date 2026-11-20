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
static constexpr size_t INPUT_IDX_PADDINGS = 1;
static constexpr size_t OUTPUT_IDX_Y = 0;
static constexpr int64_t UNKNOWN_DIM = -1;
static constexpr size_t PADDINGS_ROWS = 2;
static constexpr size_t PADDINGS_COLS = 2;

namespace {
// Checked arithmetic helpers: return false on overflow.
inline bool SafeAddI64(int64_t a, int64_t b, int64_t& r) { return !__builtin_add_overflow(a, b, &r); }
inline bool SafeMulI64(int64_t a, int64_t b, int64_t& r) { return !__builtin_mul_overflow(a, b, &r); }
} // namespace

class SpaceToBatchInferShapeHelper {
public:
    explicit SpaceToBatchInferShapeHelper(gert::InferShapeContext* context) : context_(context) {}

    ge::graphStatus Inference();

private:
    ge::graphStatus Init();
    int64_t GetBlockSize();

private:
    gert::InferShapeContext* context_;
    const gert::Shape* xShape_{nullptr};
    gert::Shape* yShape_{nullptr};
    int64_t blockSize_{0};
    bool isConstPaddings_{false};
    gert::Shape paddingsVec_;
};

ge::graphStatus SpaceToBatchInferShapeHelper::Init()
{
    xShape_ = context_->GetInputShape(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShape_);

    yShape_ = context_->GetOutputShape(OUTPUT_IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShape_);

    const gert::CompileTimeTensorDesc* xDesc = context_->GetInputDesc(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);

    const gert::CompileTimeTensorDesc* paddingsDesc = context_->GetInputDesc(INPUT_IDX_PADDINGS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, paddingsDesc);
    OP_CHECK_IF(paddingsDesc->GetOriginFormat() != ge::FORMAT_ND,
                OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(context_->GetNodeName(), "paddings",
                                                       Ops::Base::ToString(paddingsDesc->GetOriginFormat()).c_str(),
                                                       "The format of paddings must be ND"),
                return ge::GRAPH_FAILED);

    blockSize_ = GetBlockSize();
    OP_CHECK_IF(
        blockSize_ <= 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "block_size", std::to_string(blockSize_).c_str(),
                                              "The value of block_size must be positive"),
        return ge::GRAPH_FAILED);

    const gert::Tensor* paddingsTensor = context_->GetInputTensor(INPUT_IDX_PADDINGS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, paddingsTensor);

    isConstPaddings_ = Ops::Base::GetConstIntToShape<gert::InferShapeContext>(context_, INPUT_IDX_PADDINGS,
                                                                              paddingsVec_);
    if (isConstPaddings_) {
        OP_CHECK_IF(paddingsVec_.GetDimNum() != PADDINGS_ROWS * PADDINGS_COLS,
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context_->GetNodeName(), "paddings",
                                                              std::to_string(paddingsVec_.GetDimNum()).c_str(),
                                                              "The element count of paddings must be 4"),
                    return ge::GRAPH_FAILED);
        for (size_t i = 0; i < PADDINGS_ROWS * PADDINGS_COLS; ++i) {
            OP_CHECK_IF(paddingsVec_.GetDim(i) < 0,
                        OP_LOGE(context_, "paddings[%zu] must be non-negative, got %ld", i, paddingsVec_.GetDim(i)),
                        return ge::GRAPH_FAILED);
        }
    }

    return ge::GRAPH_SUCCESS;
}

int64_t SpaceToBatchInferShapeHelper::GetBlockSize()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const int64_t* blockSizePtr = attrs->GetInt(0);
    OP_CHECK_IF(blockSizePtr == nullptr, OP_LOGE(context_, "get block_size attr failed"), return -1);
    return *blockSizePtr;
}

ge::graphStatus SpaceToBatchInferShapeHelper::Inference()
{
    auto ret = Init();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    if (Ops::Base::IsUnknownRank(*xShape_)) {
        Ops::Base::SetUnknownRank(*yShape_);
        OP_LOGI(context_->GetNodeName(), "SpaceToBatch output shape: %s.", Ops::Base::ToString(*yShape_).c_str());
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(
        xShape_->GetDimNum() != 4,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            context_->GetNodeName(), "x", std::to_string(xShape_->GetDimNum()).c_str(), "The rank of x must be 4"),
        return ge::GRAPH_FAILED);

    const gert::CompileTimeTensorDesc* xDesc = context_->GetInputDesc(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    const bool isNchw = ge::GetPrimaryFormat(xDesc->GetOriginFormat()) == ge::FORMAT_NCHW;
    const size_t hDim = isNchw ? 2 : 1;
    const size_t wDim = isNchw ? 3 : 2;
    const size_t cDim = isNchw ? 1 : 3;

    int64_t batch = xShape_->GetDim(0);
    if (batch != UNKNOWN_DIM) {
        int64_t batchMul = 0;
        OP_CHECK_IF(!SafeMulI64(batch, blockSize_, batchMul) || !SafeMulI64(batchMul, blockSize_, batchMul),
                    OP_LOGE(context_, "output N overflow: x dim[0]=%ld * block_size=%ld * block_size=%ld", batch,
                            blockSize_, blockSize_),
                    return ge::GRAPH_FAILED);
        batch = batchMul;
    }

    int64_t hOut = UNKNOWN_DIM;
    if (isConstPaddings_) {
        int64_t padTop = paddingsVec_.GetDim(0);
        int64_t padBottom = paddingsVec_.GetDim(1);
        int64_t hIn = xShape_->GetDim(hDim);
        if (hIn != UNKNOWN_DIM) {
            int64_t hPadded = 0;
            OP_CHECK_IF(
                !SafeAddI64(hIn, padTop, hPadded) || !SafeAddI64(hPadded, padBottom, hPadded),
                OP_LOGE(context_, "H_padded overflow: H_in=%ld pad_top=%ld pad_bottom=%ld", hIn, padTop, padBottom),
                return ge::GRAPH_FAILED);
            OP_CHECK_IF(hPadded % blockSize_ != 0,
                        OP_LOGE(context_, "H_padded (%ld) not divisible by block_size (%ld)", hPadded, blockSize_),
                        return ge::GRAPH_FAILED);
            hOut = hPadded / blockSize_;
        }
    }

    int64_t wOut = UNKNOWN_DIM;
    if (isConstPaddings_) {
        int64_t padLeft = paddingsVec_.GetDim(2);
        int64_t padRight = paddingsVec_.GetDim(3);
        int64_t wIn = xShape_->GetDim(wDim);
        if (wIn != UNKNOWN_DIM) {
            int64_t wPadded = 0;
            OP_CHECK_IF(
                !SafeAddI64(wIn, padLeft, wPadded) || !SafeAddI64(wPadded, padRight, wPadded),
                OP_LOGE(context_, "W_padded overflow: W_in=%ld pad_left=%ld pad_right=%ld", wIn, padLeft, padRight),
                return ge::GRAPH_FAILED);
            OP_CHECK_IF(wPadded % blockSize_ != 0,
                        OP_LOGE(context_, "W_padded (%ld) not divisible by block_size (%ld)", wPadded, blockSize_),
                        return ge::GRAPH_FAILED);
            wOut = wPadded / blockSize_;
        }
    }

    int64_t cOut = xShape_->GetDim(cDim);

    yShape_->SetDimNum(0);
    if (isNchw) {
        yShape_->AppendDim(batch);
        yShape_->AppendDim(cOut);
        yShape_->AppendDim(hOut);
        yShape_->AppendDim(wOut);
    } else {
        yShape_->AppendDim(batch);
        yShape_->AppendDim(hOut);
        yShape_->AppendDim(wOut);
        yShape_->AppendDim(cOut);
    }

    OP_LOGI(context_->GetNodeName(), "SpaceToBatch output shape: %s.", Ops::Base::ToString(*yShape_).c_str());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Infershape4SpaceToBatch(gert::InferShapeContext* context)
{
    SpaceToBatchInferShapeHelper helper(context);
    return helper.Inference();
}

static ge::graphStatus InferDataType4SpaceToBatch(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(OUTPUT_IDX_Y, context->GetInputDataType(INPUT_IDX_X));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SpaceToBatch)
    .InferShape(Infershape4SpaceToBatch)
    .InferDataType(InferDataType4SpaceToBatch)
    .InputsDataDependency({INPUT_IDX_PADDINGS});
} // namespace ops
