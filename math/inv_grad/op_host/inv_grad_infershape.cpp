/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "util/shape_util.h"
#include "register/op_impl_registry.h"
#include "log/log.h"

#include <string>

using namespace ge;
namespace ops {
namespace {
// ND 格式合法 rank 范围为 0-8，rank>8 由 InferShape 独立拒绝（与 tiling 闸门对称，不依赖下游拦截）
constexpr size_t INV_GRAD_MAX_RANK = 8;
// 未知 rank 标记（-1 = 该维运行时确定，任意负维统一按未知放行，无需单列常量）
constexpr int64_t INV_GRAD_UNKNOWN_RANK = -2;

// 同形契约的动态 shape 兼容判断（编译期语义）：
// - rank 相同：逐维比较，仅当某维双方均已知（>=0）且不等时才拒绝；任一方未知（<0，-1/-2）放行；
// - rank 不同：仅当任一侧含 -2（未知 rank）时放行，否则拒绝。
// 放行的未知维/rank 透传到输出，由运行时 tiling 侧 GetShapeInfo 在具体 shape 上兜底校验。
bool ShapeDynCompatible(const gert::Shape& lhs, const gert::Shape& rhs)
{
    if (lhs.GetDimNum() == rhs.GetDimNum()) {
        for (size_t i = 0; i < lhs.GetDimNum(); ++i) {
            const int64_t lhsDim = lhs.GetDim(i);
            const int64_t rhsDim = rhs.GetDim(i);
            if (lhsDim >= 0 && rhsDim >= 0 && lhsDim != rhsDim) {
                return false;
            }
        }
        return true;
    }
    for (size_t i = 0; i < lhs.GetDimNum(); ++i) {
        if (lhs.GetDim(i) == INV_GRAD_UNKNOWN_RANK) {
            return true;
        }
    }
    for (size_t i = 0; i < rhs.GetDimNum(); ++i) {
        if (rhs.GetDim(i) == INV_GRAD_UNKNOWN_RANK) {
            return true;
        }
    }
    return false;
}
} // namespace

static ge::graphStatus InferShape4InvGrad(gert::InferShapeContext* context)
{
    OP_LOGI("Begin InferShape4InvGrad");
    const gert::Shape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape* gradShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradShape);
    gert::Shape* yShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    const std::string rankReason = "The rank must not exceed " + std::to_string(INV_GRAD_MAX_RANK) + ".";
    OP_CHECK_IF(xShape->GetDimNum() > INV_GRAD_MAX_RANK,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    context->GetNodeName(), "x", std::to_string(xShape->GetDimNum()).c_str(), rankReason.c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(gradShape->GetDimNum() > INV_GRAD_MAX_RANK,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    context->GetNodeName(), "grad", std::to_string(gradShape->GetDimNum()).c_str(), rankReason.c_str()),
                return ge::GRAPH_FAILED);

    // 同形契约：不支持广播，x 与 grad 必须完全同形；需要广播时由调用方外部预广播。
    // 动态 shape（-1 未知维 / -2 未知 rank）编译期放行并透传，运行时由 tiling 在具体 shape 上兜底校验
    if (!ShapeDynCompatible(*xShape, *gradShape)) {
        const std::string shapes = Ops::Base::ToString(*xShape) + " and " + Ops::Base::ToString(*gradShape);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "x and grad", shapes.c_str(),
                                               "InvGrad does not support broadcast, x and grad must have the same "
                                               "shape, pre-broadcast externally if needed");
        return ge::GRAPH_FAILED;
    }

    *yShape = *xShape;
    OP_LOGI(context->GetNodeName(), "[InferShape] output shape=%s", Ops::Base::ToString(*yShape).c_str());
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(InvGrad).InferShape(InferShape4InvGrad);
} // namespace ops
