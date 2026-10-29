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
namespace ops {
static ge::graphStatus InferDataTypeDenseBincount(gert::InferDataTypeContext* context)
{
    const ge::DataType inputDtype = context->GetInputDataType(0);
    const ge::DataType sizeDtype = context->GetInputDataType(1);
    const ge::DataType weightsDtype = context->GetInputDataType(2);
    OP_CHECK_IF(
        inputDtype != ge::DT_INT32 && inputDtype != ge::DT_INT64,
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "input", Ops::Base::ToString(inputDtype), "int32 or int64"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        sizeDtype != inputDtype,
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context->GetNodeName(), "input, size",
                                               Ops::Base::ToString(inputDtype) + ", " + Ops::Base::ToString(sizeDtype),
                                               "input and size must have the same dtype"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        weightsDtype != ge::DT_FLOAT,
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "weights", Ops::Base::ToString(weightsDtype), "float32"),
        return ge::GRAPH_FAILED);
    return context->SetOutputDataType(0, ge::DT_FLOAT);
}
IMPL_OP(DenseBincount).InferDataType(InferDataTypeDenseBincount);
} // namespace ops
