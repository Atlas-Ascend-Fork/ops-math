/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>

#include "exe_graph/runtime/infer_datatype_context.h"
#include "register/op_impl_registry.h"
#include "op_common/log/log.h"
#include "accumulate_nv2_graph_infer.h"

namespace ops {

static ge::graphStatus InferDataType4AccumulateNV2(gert::InferDataTypeContext* context)
{
    const size_t inputNum = context->GetComputeNodeInputNum();
    if (inputNum == 0U) {
        OP_LOGE("AccumulateNV2", "input count must not be zero");
        return ge::GRAPH_FAILED;
    }
    const ge::DataType firstDtype = context->GetInputDataType(0);
    for (size_t i = 1U; i < inputNum; ++i) {
        const ge::DataType dtype = context->GetInputDataType(i);
        if (dtype != firstDtype) {
            OP_LOGE("AccumulateNV2",
                    "all input dtypes must be the same as the first input, but input[%zu] dtype %d differs from "
                    "input[0] dtype %d",
                    i, static_cast<int32_t>(dtype), static_cast<int32_t>(firstDtype));
            return ge::GRAPH_FAILED;
        }
    }
    context->SetOutputDataType(0, firstDtype);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(AccumulateNV2).InferDataType(InferDataType4AccumulateNV2);

} // namespace ops
