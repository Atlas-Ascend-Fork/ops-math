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
 * \file complex_infershape.cpp
 * \brief Complex InferShape and InferDataType.
 *        Output shape is the broadcast result of real and imag,
 *        output dtype is the corresponding complex type of the input dtype.
 */
#include "register/op_impl_registry.h"
#include "infershape_broadcast_util.h"
#include "graph/utils/type_utils.h"
#include "log/log.h"

using namespace ge;

namespace ops {
constexpr size_t INPUT_COUNT = 2;
constexpr size_t INPUT_REAL_INDEX = 0;
constexpr size_t INPUT_IMAG_INDEX = 1;
constexpr size_t OUTPUT_OUT_INDEX = 0;

static ge::graphStatus InferShapeForComplex(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeForComplex");
    // kernel 不支持混合 dtype，real 和 imag 必须为相同 dtype
    const auto* realDesc = context->GetInputDesc(INPUT_REAL_INDEX);
    if (realDesc == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "real");
        return ge::GRAPH_FAILED;
    }
    const auto* imagDesc = context->GetInputDesc(INPUT_IMAG_INDEX);
    if (imagDesc == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "imag");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(realDesc->GetDataType() != imagDesc->GetDataType(),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "imag",
                                                      std::to_string(static_cast<int>(imagDesc->GetDataType())).c_str(),
                                                      "real and imag dtype must be the same"),
                return GRAPH_FAILED);
    // 对 real、imag 做广播得到输出 out 的 shape
    return Ops::Base::InferShape4Broadcast(context, INPUT_COUNT);
}

static ge::graphStatus InferDataTypeForComplex(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeForComplex");
    const auto realDtype = context->GetInputDataType(INPUT_REAL_INDEX);
    DataType outDtype = realDtype;
    switch (realDtype) {
        case DT_FLOAT:
            outDtype = DT_COMPLEX64;
            break;
        case DT_FLOAT16:
            outDtype = DT_COMPLEX32;
            break;
        case DT_DOUBLE:
            outDtype = DT_COMPLEX128;
            break;
        default: {
            const std::string realDtypeStr = ge::TypeUtils::DataTypeToSerialString(realDtype);
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "real", realDtypeStr.c_str(),
                                                  "only FLOAT, FLOAT16 and DOUBLE are supported");
            return GRAPH_FAILED;
        }
    }
    // out.dtype -> real.dtype 对应的复数类型
    context->SetOutputDataType(OUTPUT_OUT_INDEX, outDtype);
    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeForComplex");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Complex).InferShape(InferShapeForComplex).InferDataType(InferDataTypeForComplex);
} // namespace ops
