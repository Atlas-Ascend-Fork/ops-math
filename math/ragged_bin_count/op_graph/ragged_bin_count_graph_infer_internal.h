/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_MATH_RAGGED_BIN_COUNT_GRAPH_INFER_INTERNAL_H
#define OPS_MATH_RAGGED_BIN_COUNT_GRAPH_INFER_INTERNAL_H

#include <string>

#include "log/log.h"
#include "register/op_impl_registry.h"

namespace ops {
namespace ragged_bin_count_graph_infer_internal {
namespace {
constexpr size_t INPUT_SPLITS = 0U;
constexpr size_t INPUT_VALUES = 1U;
constexpr size_t INPUT_SIZE = 2U;
constexpr size_t INPUT_WEIGHTS = 3U;
constexpr size_t OUTPUT_RESULT = 0U;
} // namespace

inline std::string FormatDtype(ge::DataType dtype)
{
    auto name = Ops::Base::ToString(dtype);
    if (name.empty() || name == "UNDEFINED") {
        name = "UNKNOWN";
    }
    return name + "(" + std::to_string(static_cast<int>(dtype)) + ")";
}

inline ge::graphStatus ValidateNativeDataTypes(const ge::char_t* nodeName, ge::DataType splitsDtype,
                                               ge::DataType valuesDtype, ge::DataType sizeDtype,
                                               ge::DataType weightsDtype)
{
    OP_CHECK_IF(
        splitsDtype != ge::DT_INT64,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(Ops::Base::GetOpInfo(nodeName), "splits", FormatDtype(splitsDtype),
                                              "The dtype of splits must be " + FormatDtype(ge::DT_INT64)),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        valuesDtype != ge::DT_INT32 && valuesDtype != ge::DT_INT64,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(Ops::Base::GetOpInfo(nodeName), "values", FormatDtype(valuesDtype),
                                              "The dtype of values must be within the range {" +
                                                  FormatDtype(ge::DT_INT32) + ", " + FormatDtype(ge::DT_INT64) + "}"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(sizeDtype != valuesDtype,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    Ops::Base::GetOpInfo(nodeName), "size", FormatDtype(sizeDtype),
                    "The dtype of size must match the dtype of values, which is " + FormatDtype(valuesDtype)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        weightsDtype != ge::DT_FLOAT,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(Ops::Base::GetOpInfo(nodeName), "weights", FormatDtype(weightsDtype),
                                              "The dtype of weights must be " + FormatDtype(ge::DT_FLOAT)),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckInputDataTypes(gert::InferDataTypeContext* context)
{
    return ValidateNativeDataTypes(context->GetNodeName(), context->GetInputDataType(INPUT_SPLITS),
                                   context->GetInputDataType(INPUT_VALUES), context->GetInputDataType(INPUT_SIZE),
                                   context->GetInputDataType(INPUT_WEIGHTS));
}

inline ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to infer the data type for RaggedBinCount.");
    OP_CHECK_IF(CheckInputDataTypes(context) != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "The input dtype combination is not supported."),
                return ge::GRAPH_FAILED);

    return context->SetOutputDataType(OUTPUT_RESULT, ge::DT_FLOAT);
}
} // namespace ragged_bin_count_graph_infer_internal
} // namespace ops

#endif // OPS_MATH_RAGGED_BIN_COUNT_GRAPH_INFER_INTERNAL_H
