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
#include "op_api/op_util.h"
#include "op_host/util/const_util.h"
#include "op_host/util/shape_util.h"

using namespace ge;
namespace ops {
static constexpr size_t INPUT_IDX_X = 0;
static constexpr size_t INPUT_IDX_CROPS = 1;
static constexpr size_t OUTPUT_IDX_Y = 0;
static constexpr size_t DIM_N = 0;
static constexpr size_t DIM_H = 1;
static constexpr size_t DIM_W = 2;
static constexpr size_t DIM_C = 3;
static constexpr int64_t UNKNOWN_DIM = -1;
static constexpr size_t CROPS_SIZE = 4;
static constexpr size_t X_RANK = 4;

class BatchToSpaceInferShapeHelper {
public:
    explicit BatchToSpaceInferShapeHelper(gert::InferShapeContext* context) : context_(context) {}

    ge::graphStatus Inference();

private:
    ge::graphStatus Init();

    gert::InferShapeContext* context_;
    const gert::Shape* xShape_{nullptr};
    gert::Shape* yShape_{nullptr};
    int64_t blockSize_{0};
    int64_t crops_[CROPS_SIZE]{0};
    bool hasCropsConst_{false};
};

ge::graphStatus BatchToSpaceInferShapeHelper::Init()
{
    xShape_ = context_->GetInputShape(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShape_);

    yShape_ = context_->GetOutputShape(OUTPUT_IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShape_);

    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    auto blockSizePtr = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, blockSizePtr);
    blockSize_ = *blockSizePtr;
    OP_CHECK_IF(blockSize_ <= 0, OP_LOGE(context_, "block_size must be positive"), return ge::GRAPH_FAILED);

    const gert::Tensor* cropsTensor = context_->GetInputTensor(INPUT_IDX_CROPS);
    gert::Shape cropsShape;
    // 对齐 batch_to_space_nd：仅在 crops 为 const（数据可用）时消费其形状并读取取值。
    // 动态图下非 const crops 的形状是未知描述符（-1/-2），GetShapeSize() 会得到无意义值
    // （如 (-2,)→-2、(-1,-1)→1），据此拒绝会阻断 DynamicShapeSupportFlag 声明的动态图执行；
    // 非 const crops 的形状与取值校验由运行时 tiling 以真实形状执行（tiling 侧已有同款检查）。
    const bool isConstCrops = IsConstTensor(cropsTensor);
    if (isConstCrops) {
        OP_CHECK_IF(static_cast<size_t>(cropsTensor->GetShapeSize()) != CROPS_SIZE,
                    OP_LOGE(context_, "crops must have %zu elements, got %ld", CROPS_SIZE, cropsTensor->GetShapeSize()),
                    return ge::GRAPH_FAILED);
    }
    if (isConstCrops && Ops::Base::GetConstIntToShape<gert::InferShapeContext>(context_, INPUT_IDX_CROPS, cropsShape)) {
        for (size_t i = 0; i < CROPS_SIZE; ++i) {
            crops_[i] = cropsShape[i];
            OP_CHECK_IF(crops_[i] < 0, OP_LOGE(context_, "crops[%zu] must be non-negative, got %ld", i, crops_[i]),
                        return ge::GRAPH_FAILED);
        }
        hasCropsConst_ = true;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchToSpaceInferShapeHelper::Inference()
{
    auto ret = Init();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    if (Ops::Base::IsUnknownRank(*xShape_)) {
        Ops::Base::SetUnknownRank(*yShape_);
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(xShape_->GetDimNum() != X_RANK,
                OP_LOGE(context_, "x must be %zuD, got %zu", X_RANK, xShape_->GetDimNum()), return ge::GRAPH_FAILED);

    yShape_->SetDimNum(0);

    // N = N_in / (block_size * block_size)
    int64_t bs2 = blockSize_ * blockSize_;
    if (xShape_->GetDim(DIM_N) != UNKNOWN_DIM) {
        OP_CHECK_IF(xShape_->GetDim(DIM_N) % bs2 != 0,
                    OP_LOGE(context_, "N_in must be divisible by block_size^2, got N_in=%ld, block_size=%ld",
                            xShape_->GetDim(DIM_N), blockSize_),
                    return ge::GRAPH_FAILED);
        yShape_->AppendDim(xShape_->GetDim(DIM_N) / bs2);
    } else {
        yShape_->AppendDim(UNKNOWN_DIM);
    }

    // H_out = H_in * block_size - crop_top - crop_bottom
    if (xShape_->GetDim(DIM_H) != UNKNOWN_DIM && hasCropsConst_) {
        int64_t hOut = xShape_->GetDim(DIM_H) * blockSize_ - crops_[0] - crops_[1];
        OP_CHECK_IF(hOut <= 0, OP_LOGE(context_, "H_out must be positive, got %ld", hOut), return ge::GRAPH_FAILED);
        yShape_->AppendDim(hOut);
    } else {
        yShape_->AppendDim(UNKNOWN_DIM);
    }

    // W_out = W_in * block_size - crop_left - crop_right
    if (xShape_->GetDim(DIM_W) != UNKNOWN_DIM && hasCropsConst_) {
        int64_t wOut = xShape_->GetDim(DIM_W) * blockSize_ - crops_[2] - crops_[3];
        OP_CHECK_IF(wOut <= 0, OP_LOGE(context_, "W_out must be positive, got %ld", wOut), return ge::GRAPH_FAILED);
        yShape_->AppendDim(wOut);
    } else {
        yShape_->AppendDim(UNKNOWN_DIM);
    }

    // C: unchanged
    if (xShape_->GetDim(DIM_C) != UNKNOWN_DIM) {
        yShape_->AppendDim(xShape_->GetDim(DIM_C));
    } else {
        yShape_->AppendDim(UNKNOWN_DIM);
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Infershape4BatchToSpace(gert::InferShapeContext* context)
{
    BatchToSpaceInferShapeHelper helper(context);
    const ge::graphStatus ret = helper.Inference();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    const gert::Shape* yShape = context->GetOutputShape(OUTPUT_IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    OP_LOGI(context->GetNodeName(), "BatchToSpace output shape: %s.", Ops::Base::ToString(*yShape).c_str());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4BatchToSpace(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(OUTPUT_IDX_Y, context->GetInputDataType(INPUT_IDX_X));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(BatchToSpace)
    .InferShape(Infershape4BatchToSpace)
    .InferDataType(InferDataType4BatchToSpace)
    .InputsDataDependency({INPUT_IDX_CROPS});
} // namespace ops
