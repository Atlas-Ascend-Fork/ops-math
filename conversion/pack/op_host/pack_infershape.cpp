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
 * \file pack_infershape.cpp
 * \brief
 */

#include <cinttypes>
#include "op_api/op_util.h"
#include "log/log.h"
#include "op_common/op_host/util/shape_util.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {
constexpr size_t INDEX_AXIS = 0;
constexpr size_t INDEX_N = 1;

static ge::graphStatus PackInferShapeCommon(gert::InferShapeContext* context, const int64_t dynamic_input_idx,
                                            int64_t pack_num, int64_t axis)
{
    auto out_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);

    // 未知秩(-2)兼容: 与 rt1.0 动态路径 JoinShapes 对齐, 未知秩输入视为通配,
    // 取首个已知秩输入作为输出基准; 若全部输入均为未知秩, 输出保持未知秩。
    const gert::Shape* base_shape = nullptr;
    int64_t base_index = 0;
    for (int64_t input_idx = 0; input_idx < pack_num; input_idx++) {
        const gert::Shape* input_shape = context->GetDynamicInputShape(dynamic_input_idx, input_idx);
        OP_CHECK_NULL_WITH_CONTEXT(context, input_shape);
        if (!Ops::Base::IsUnknownRank(*input_shape)) {
            base_shape = input_shape;
            base_index = input_idx;
            break;
        }
    }
    if (base_shape == nullptr) {
        Ops::Base::SetUnknownRank(*out_shape);
        return ge::GRAPH_SUCCESS;
    }
    *out_shape = *base_shape;

    for (int64_t relative_index = 0; relative_index < pack_num; relative_index++) {
        const gert::Shape* input_i_shape = context->GetDynamicInputShape(dynamic_input_idx, relative_index);
        OP_CHECK_NULL_WITH_CONTEXT(context, input_i_shape);
        if (Ops::Base::IsUnknownRank(*input_i_shape)) {
            // 未知秩输入通配, 跳过一致性校验
            continue;
        }
        if (*input_i_shape != *base_shape) {
            // input shape i is not equal base shape
            OP_LOGE(context->GetNodeName(), "input shape[%" PRId64 "] %s, must be equal to shape[%" PRId64 "] %s!",
                    relative_index, Ops::Base::ToString(*input_i_shape).c_str(), base_index,
                    Ops::Base::ToString(*base_shape).c_str());
            return ge::GRAPH_FAILED;
        }
    }

    if (out_shape->IsScalar()) {
        // scalar to shape [1]
        out_shape->AppendDim(pack_num);
        return ge::GRAPH_SUCCESS;
    }

    size_t output_dim = out_shape->GetDimNum() + 1;
    out_shape->SetDimNum(output_dim);
    OP_CHECK_IF(!IsDimValid(output_dim, axis),
                OP_LOGE(context->GetNodeName(), "%s", GenInvalidDimMsg("axis", output_dim, axis).c_str()),
                return ge::GRAPH_FAILED);
    if (axis < 0) {
        axis += output_dim;
    }
    int64_t i = output_dim;
    while (i > axis) {
        out_shape->SetDim(static_cast<uint32_t>(i), out_shape->GetDim(static_cast<uint32_t>(i - 1)));
        i--;
    }
    out_shape->SetDim(axis, pack_num);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4Pack(gert::InferShapeContext* context)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* axis = attrs->GetAttrPointer<int64_t>(INDEX_AXIS);
    OP_CHECK_NULL_WITH_CONTEXT(context, axis);
    const int64_t* N = attrs->GetAttrPointer<int64_t>(INDEX_N);
    OP_CHECK_NULL_WITH_CONTEXT(context, N);
    const int64_t num = context->GetComputeNodeInputNum();
    OP_CHECK_IF((*N != num && *N != 1),
                OP_LOGE(context->GetNodeName(), "The N should be equal to input size or default value 1."),
                return GRAPH_FAILED);
    return PackInferShapeCommon(context, 0, num, *axis);
}

IMPL_OP_INFERSHAPE(Pack).InferShape(InferShape4Pack);
} // namespace ops
