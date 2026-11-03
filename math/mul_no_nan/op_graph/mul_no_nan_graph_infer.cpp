/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log/log.h"
#include "register/op_impl_registry.h"

namespace ops {
ge::graphStatus InferDataType4MulNoNan(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto dtype = context->GetInputDataType(0);
    OP_CHECK_IF(dtype != ge::DT_FLOAT16 && dtype != ge::DT_FLOAT && dtype != ge::DT_INT32 && dtype != ge::DT_BF16,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "x1", Ops::Base::ToString(dtype),
                                                      "only float16, float32, int32 and bfloat16 are supported"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(dtype != context->GetInputDataType(1),
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    context->GetNodeName(), "x1, x2",
                    Ops::Base::ToString(dtype) + ", " + Ops::Base::ToString(context->GetInputDataType(1)),
                    "inputs must have the same dtype"),
                return ge::GRAPH_FAILED);
    return context->SetOutputDataType(0, dtype);
}

IMPL_OP(MulNoNan).InferDataType(InferDataType4MulNoNan);
} // namespace ops
