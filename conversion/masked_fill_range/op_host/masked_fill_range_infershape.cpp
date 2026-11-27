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
 * \file masked_fill_range_infershape.cpp
 * \brief MaskedFillRange 算子的 shape 推导
 *
 * y 与 x 同形同型：算子按 start/end 区间原地覆写 x 的 [start, end) 段，不改
 * 变布局，故 InferShape 为 same-as-input 推导；unknown rank（rank 未知，-2
 * 场景）显式传播，常规场景整体赋值。InferDataType（y=x 同型）按 conversion
 * 仓 op_host/op_graph 分工（matrix_diag/masked_fill 等）放在
 * op_graph/masked_fill_range_graph_infer.cpp 的 IMPL_OP 注册，本文件仅注册
 * InferShape，避免双注册。y 仅依赖 x 的 shape，无需读取任何输入张量的值，
 * 故不声明 InputsDataDependency。
 *
 * 输入校验对齐 canndev Verifier（IMPLEMT_VERIFIER(MaskedFillRange,
 * MaskedFillRangeVerify)）的三条：x/value dtype 相等、start/end dtype 相等、
 * axis 不越界；另加 start/end 均为 DT_INT32 前置校验（canndev Verifier 无此
 * 条，属 tiling 校验的图构建期前置化，语义不变仅报错时机提前）。
 * gert 新 API 无独立 Verifier 注册点，校验随 InferShape 执行。
 * axis 判定与 tiling 一致采用负轴归一化（axis < 0 时 axis += dim），而非
 * canndev 的 std::abs(axis)——后者会误杀 axis == -dim 的合法负轴。
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"
#include "graph/utils/type_utils.h"

namespace ops {
namespace {
constexpr size_t IDX_X = 0;
constexpr size_t IDX_START = 1;
constexpr size_t IDX_END = 2;
constexpr size_t IDX_VALUE = 3;
constexpr size_t IDX_Y = 0;
constexpr size_t IDX_ATTR_AXIS = 0;
} // namespace

/*!
 * \brief 推理 MaskedFillRange 的输出 shape，输出 shape 与输入 x 相同
 */
static ge::graphStatus InferShapeMaskedFillRange(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeMaskedFillRange");

    // x 与 value dtype 相等（canndev Verifier 校验 1）
    const auto* xDesc = context->GetInputDesc(IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    const auto* valueDesc = context->GetInputDesc(IDX_VALUE);
    OP_CHECK_NULL_WITH_CONTEXT(context, valueDesc);
    OP_CHECK_IF(
        xDesc->GetDataType() != valueDesc->GetDataType(),
        OP_LOGE_FOR_INVALID_DTYPE(
            context->GetNodeName(), "x and value",
            (Ops::Base::ToString(xDesc->GetDataType()) + ", " + Ops::Base::ToString(valueDesc->GetDataType())).c_str(),
            "equal dtypes"),
        return ge::GRAPH_FAILED);

    // start 与 end dtype 相等（canndev Verifier 校验 2）
    const auto* startDesc = context->GetInputDesc(IDX_START);
    OP_CHECK_NULL_WITH_CONTEXT(context, startDesc);
    const auto* endDesc = context->GetInputDesc(IDX_END);
    OP_CHECK_NULL_WITH_CONTEXT(context, endDesc);
    OP_CHECK_IF(startDesc->GetDataType() != endDesc->GetDataType(),
                OP_LOGE_FOR_INVALID_DTYPE(
                    context->GetNodeName(), "start and end",
                    (Ops::Base::ToString(startDesc->GetDataType()) + ", " + Ops::Base::ToString(endDesc->GetDataType()))
                        .c_str(),
                    "equal dtypes"),
                return ge::GRAPH_FAILED);

    // start 与 end 均为 INT32（tiling 侧 INT32 校验的图构建期前置化：canndev
    // Verifier 无此条，语义不变仅报错时机提前；tiling.cpp 侧校验保留作双保险，
    // 错误消息与其保持一致便于跨阶段日志检索）
    OP_CHECK_IF(startDesc->GetDataType() != ge::DT_INT32 || endDesc->GetDataType() != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPE(
                    context->GetNodeName(), "start/end",
                    (Ops::Base::ToString(startDesc->GetDataType()) + ", " + Ops::Base::ToString(endDesc->GetDataType()))
                        .c_str(),
                    "both INT32"),
                return ge::GRAPH_FAILED);

    // axis 属性（canndev Verifier 校验 3，判定随 unknown rank 延后）
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* axisPtr = attrs->GetAttrPointer<int64_t>(IDX_ATTR_AXIS);
    OP_CHECK_NULL_WITH_CONTEXT(context, axisPtr);

    const gert::Shape* xShape = context->GetInputShape(IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    gert::Shape* yShape = context->GetOutputShape(IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    if (Ops::Base::IsUnknownRank(*xShape)) {
        Ops::Base::SetUnknownRank(*yShape);
    } else {
        const int64_t dimNum = static_cast<int64_t>(xShape->GetDimNum());
        int64_t axis = *axisPtr;
        // 下界校验先于归一化：axis 为外部 attr 可传任意 int64（含 INT64_MIN 级
        // 极端负值），先拦截 axis < -dimNum 再归一化，保证 axis += dimNum 不发生
        // 有符号溢出 UB（与 tiling 侧 masked_fill_range_tiling.cpp 同步防御）
        OP_CHECK_IF(axis < -dimNum, OP_LOGE(context, "axis %ld is out of range for input dim %ld", *axisPtr, dimNum),
                    return ge::GRAPH_FAILED);
        if (axis < 0) {
            axis += dimNum;
        }
        OP_CHECK_IF(axis < 0 || axis >= dimNum,
                    OP_LOGE(context, "axis %ld is out of range for input dim %ld", *axisPtr, dimNum),
                    return ge::GRAPH_FAILED);
        *yShape = *xShape;
    }

    OP_LOGD(context->GetNodeName(), "End to do InferShapeMaskedFillRange");
    return ge::GRAPH_SUCCESS;
}

// infershape 注册入口：InferDataType 按 conversion 仓分工放在
// op_graph/masked_fill_range_graph_infer.cpp 的 IMPL_OP 注册
IMPL_OP_INFERSHAPE(MaskedFillRange).InferShape(InferShapeMaskedFillRange);
} // namespace ops
