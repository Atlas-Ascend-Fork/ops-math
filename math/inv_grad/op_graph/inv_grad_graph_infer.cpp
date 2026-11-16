/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

#include <string>

using namespace ge;
namespace ops {
namespace {
const char* INV_GRAD_SUPPORTED_DTYPES = "fp16, bf16, fp32, int32, int8";
} // namespace

static bool IsSupportedDtype4InvGrad(const ge::DataType dtype)
{
    return dtype == ge::DT_FLOAT || dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16 || dtype == ge::DT_INT32 ||
           dtype == ge::DT_INT8;
}

static ge::graphStatus InferDataType4InvGrad(gert::InferDataTypeContext* context)
{
    OP_LOGI("Begin InferDataType4InvGrad");
    const ge::DataType xDataType = context->GetInputDataType(0);
    const ge::DataType gradDataType = context->GetInputDataType(1);
    // dtype 白名单独立拒绝（与 proto/OpDef 声明集合一致，不依赖 tiling 边界拦截）
    OP_CHECK_IF(!IsSupportedDtype4InvGrad(xDataType),
                OP_LOGE_FOR_INVALID_DTYPE("InvGrad", "x", ge::TypeUtils::DataTypeToSerialString(xDataType).c_str(),
                                          INV_GRAD_SUPPORTED_DTYPES),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        !IsSupportedDtype4InvGrad(gradDataType),
        OP_LOGE_FOR_INVALID_DTYPE("InvGrad", "grad", ge::TypeUtils::DataTypeToSerialString(gradDataType).c_str(),
                                  INV_GRAD_SUPPORTED_DTYPES),
        return ge::GRAPH_FAILED);
    // x 与 grad 必须同 dtype
    OP_CHECK_IF(gradDataType != xDataType,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON("InvGrad", "x and grad",
                                                       (ge::TypeUtils::DataTypeToSerialString(xDataType) + " and " +
                                                        ge::TypeUtils::DataTypeToSerialString(gradDataType))
                                                           .c_str(),
                                                       "The dtypes of x and grad must be the same"),
                return ge::GRAPH_FAILED);
    context->SetOutputDataType(0, xDataType);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(InvGrad).InferDataType(InferDataType4InvGrad);
} // namespace ops
