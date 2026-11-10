/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fill.cc
 * \brief
 */
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_range_context.h"
#include "log/log.h"
#include "op_host/util/const_util.h"
#include "op_host/util/shape_util.h"

using namespace ge;
namespace ops {
constexpr size_t INPUT_INDEX_DIMS = 0;
constexpr size_t OUTPUT_INDEX_Y = 0;

static ge::graphStatus InferShape4Fill(gert::InferShapeContext* context)
{
    gert::Shape* out_shape = context->GetOutputShape(OUTPUT_INDEX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);

    if (!Ops::Base::GetConstIntToShape<gert::InferShapeContext>(context, INPUT_INDEX_DIMS, *out_shape)) {
        // dims不是常量，无法获取具体值
        const gert::Shape* dims_shape = context->GetInputShape(INPUT_INDEX_DIMS);
        if (dims_shape == nullptr) {
            Ops::Base::SetUnknownRank(*out_shape);
            OP_LOGW(context->GetNodeName(), "dims shape is nullptr, set output to UNKNOWN_RANK");
            return GRAPH_SUCCESS;
        }
        const int64_t dim_value = dims_shape->GetDimNum() == 0 ? 0 : dims_shape->GetDim(0);
        if (dim_value <= 0) {
            Ops::Base::SetUnknownRank(*out_shape);
        } else {
            Ops::Base::SetUnknownShape(dim_value, *out_shape);
        }
        OP_LOGI(context->GetNodeName(), "dims is not const, dim value is %ld", dim_value);
        return GRAPH_SUCCESS;
    }

    OP_LOGI(context->GetNodeName(), "InferShape4Fill: output shape is %s", Ops::Base::ToString(*out_shape).c_str());
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Fill).InferShape(InferShape4Fill).InputsDataDependency({INPUT_INDEX_DIMS});
} // namespace ops
