/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file slice_last_dim_graph_infer.cpp
 * \brief Data type inference for SliceLastDim operator (dtype whitelist aligned with the OpDef baseDataType).
 */
#include "register/op_impl_registry.h"
#include "graph/utils/type_utils.h"
#include "log/log.h"

using namespace ge;
namespace ops {
static constexpr size_t INPUT_IDX_X = 0;
static constexpr size_t OUTPUT_IDX_Y = 0;

static ge::graphStatus InferDataType4SliceLastDim(gert::InferDataTypeContext* context)
{
    const ge::DataType xDataType = context->GetInputDataType(INPUT_IDX_X);
    // The dtype whitelist is aligned with the OpDef baseDataType (see slice_last_dim_def.cpp),
    // the operator info dtype list and README: int8/int16/int32/int64/float16/float32/bfloat16.
    OP_CHECK_IF(xDataType != ge::DT_INT8 && xDataType != ge::DT_INT16 && xDataType != ge::DT_INT32 &&
                    xDataType != ge::DT_INT64 && xDataType != ge::DT_FLOAT16 && xDataType != ge::DT_FLOAT &&
                    xDataType != ge::DT_BF16,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "x",
                                                      ge::TypeUtils::DataTypeToSerialString(xDataType),
                                                      "The dtype of x must be one of [int8, int16, int32, int64, "
                                                      "float16, float32, bfloat16]."),
                return ge::GRAPH_FAILED);
    context->SetOutputDataType(OUTPUT_IDX_Y, xDataType);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(SliceLastDim).InferDataType(InferDataType4SliceLastDim);
} // namespace ops
