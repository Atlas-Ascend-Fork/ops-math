/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
 * LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file minimum_grad_infershape.cpp
 * \brief MinimumGrad 算子 Shape 推导实现
 *
 * Shape 推导规则：
 *   broadcastShape = broadcast(grads, x1, x2) 三者广播后的最大 shape
 *   y1 shape = y2 shape = broadcastShape
 *   根据 grad_x/grad_y 属性动态确定输出数量
 */

#include <string>
#include <vector>
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/runtime_attrs.h"
#include "op_common/log/log.h"
#include "op_common/op_host/infershape_broadcast_util.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeForMinimumGrad(gert::InferShapeContext* context)
{
    // 获取属性
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    // grad_x 和 grad_y 是第 0 和第 1 个属性
    const bool* gradXPtr = attrs->GetBool(0); // grad_x
    const bool* gradYPtr = attrs->GetBool(1); // grad_y
    OP_CHECK_NULL_WITH_CONTEXT(context, gradXPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradYPtr);

    bool gradXAttr = *gradXPtr;
    bool gradYAttr = *gradYPtr;

    // 校验：grad_x 和 grad_y 不能同时为 False
    if (!gradXAttr && !gradYAttr) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context->GetNodeName(), "grad_x, grad_y", "false, false",
                                               "grad_x and grad_y cannot both be false");
        return ge::GRAPH_FAILED;
    }

    const gert::Shape* gradsShape = context->GetInputShape(0);
    const gert::Shape* x1Shape = context->GetInputShape(1);
    const gert::Shape* x2Shape = context->GetInputShape(2);

    OP_CHECK_NULL_WITH_CONTEXT(context, gradsShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2Shape);

    // 计算 broadcast shape
    std::vector<const gert::Shape*> inShapes = {gradsShape, x1Shape, x2Shape};
    gert::Shape broadcastShape;
    if (!Ops::Base::BroadcastShape(inShapes, &broadcastShape)) {
        const std::string shapeMsg = "grads: " + Ops::Base::ToString(*gradsShape) +
                                     ", x1: " + Ops::Base::ToString(*x1Shape) +
                                     ", x2: " + Ops::Base::ToString(*x2Shape);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "grads, x1, x2", shapeMsg.c_str(),
                                               "the shapes cannot be broadcast together");
        return ge::GRAPH_FAILED;
    }

    // 两个输出 shape 均为 broadcast shape。不能按 grad_x/grad_y 跳过未启用输出：
    gert::Shape* y1Shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, y1Shape);
    *y1Shape = broadcastShape;

    gert::Shape* y2Shape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, y2Shape);
    *y2Shape = broadcastShape;

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MinimumGrad).InferShape(InferShapeForMinimumGrad);

} // namespace ops
