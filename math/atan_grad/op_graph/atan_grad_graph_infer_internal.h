/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_MATH_ATAN_GRAD_GRAPH_INFER_INTERNAL_H
#define OPS_MATH_ATAN_GRAD_GRAPH_INFER_INTERNAL_H

#include <string>

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_datatype_context.h"
#include "log/log.h"

namespace ops {
namespace atan_grad_graph_infer_internal {

inline ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeForAtanGrad");
    const ge::DataType yDtype = context->GetInputDataType(0);
    const ge::DataType dyDtype = context->GetInputDataType(1);
    // 必选输入非空校验：输入缺失时 GetInputDataType 返回 DT_UNDEFINED，
    // 必须显式拦截并给出明确语义，避免误报为 dtype 不支持
    OP_CHECK_IF(
        yDtype == ge::DT_UNDEFINED || dyDtype == ge::DT_UNDEFINED || yDtype == ge::DT_MAX || dyDtype == ge::DT_MAX,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context->GetNodeName(), "y and dy", "DT_UNDEFINED",
            "required input y/dy is missing or has undefined dtype, both y and dy are REQUIRED inputs"),
        return ge::GRAPH_FAILED);
    // 接口层拦截：dtype 白名单（README 约束仅支持 FLOAT16/FLOAT/BF16，
    // y=bool、y=complex128 等在此直接拦截，不下沉到 tiling/framework 层）
    OP_CHECK_IF(yDtype != ge::DT_FLOAT16 && yDtype != ge::DT_FLOAT && yDtype != ge::DT_BF16,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "y", Ops::Base::ToString(yDtype),
                                                      "unsupported dtype, only FLOAT16/FLOAT/BF16 are supported"),
                return ge::GRAPH_FAILED);
    // README 约束：y、dy、z 数据类型必须完全一致，不支持隐式类型转换
    OP_CHECK_IF(yDtype != dyDtype,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    context->GetNodeName(), "y and dy",
                    (Ops::Base::ToString(yDtype) + " and " + Ops::Base::ToString(dyDtype)).c_str(),
                    "The dtypes of input y and dy must be identical"),
                return ge::GRAPH_FAILED);
    context->SetOutputDataType(0, yDtype);
    return ge::GRAPH_SUCCESS;
}

} // namespace atan_grad_graph_infer_internal
} // namespace ops

#endif // OPS_MATH_ATAN_GRAD_GRAPH_INFER_INTERNAL_H
