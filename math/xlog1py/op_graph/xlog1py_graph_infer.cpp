/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/utils/type_utils.h"
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops {
static bool IsSupportedDtype(const ge::DataType dtype)
{
    return dtype == ge::DT_FLOAT || dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16;
}

static ge::graphStatus InferDataTypeForXlog1py(gert::InferDataTypeContext* context)
{
    OP_LOGI("Begin InferDataTypeForXlog1py");
    const size_t inputNum = context->GetComputeNodeInputNum();
    if (inputNum == 0U) {
        OP_LOGE_FOR_INVALID_TENSORNUM("Xlog1py", "input", static_cast<int64_t>(inputNum), "at least 1");
        return ge::GRAPH_FAILED;
    }
    const ge::DataType xDataType = context->GetInputDataType(0);
    if (!IsSupportedDtype(xDataType)) {
        OP_LOGE_FOR_INVALID_DTYPE("Xlog1py", "input[0]", ge::TypeUtils::DataTypeToSerialString(xDataType).c_str(),
                                  "fp16, bf16, fp32");
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 1U; i < inputNum; ++i) {
        const ge::DataType dtype = context->GetInputDataType(i);
        if (!IsSupportedDtype(dtype)) {
            std::string paramName = "input[" + std::to_string(i) + "]";
            OP_LOGE_FOR_INVALID_DTYPE("Xlog1py", paramName.c_str(),
                                      ge::TypeUtils::DataTypeToSerialString(dtype).c_str(), "fp16, bf16, fp32");
            return ge::GRAPH_FAILED;
        }
        if (dtype != xDataType) {
            std::string paramNames = "input[0] and input[" + std::to_string(i) + "]";
            std::string dtypes = ge::TypeUtils::DataTypeToSerialString(xDataType) + " and " +
                                 ge::TypeUtils::DataTypeToSerialString(dtype);
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON("Xlog1py", paramNames.c_str(), dtypes.c_str(),
                                                   "The dtypes of all inputs must be the same");
            return ge::GRAPH_FAILED;
        }
    }
    context->SetOutputDataType(0, xDataType);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(Xlog1py).InferDataType(InferDataTypeForXlog1py);
} // namespace ops
