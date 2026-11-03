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
 * \file as_strided_infershape.cpp
 * \brief
 */
#include "util/const_util.h"
#include "log/log.h"
#include "op_common/op_host/util/shape_util.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {
static constexpr size_t IN_SIZE = 1;
static constexpr size_t IN_STRIDE = 2;
static constexpr size_t IN_OFFSET = 3;
static constexpr size_t OUT_Y = 0;

static ge::graphStatus InferShapeForAsStrided(gert::InferShapeContext* context)
{
    gert::Shape* yShape = context->GetOutputShape(OUT_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    const gert::Tensor* size_tensor = context->GetInputTensor(IN_SIZE);
    // 对齐 rt1.0 SetOutputForRank：size 缺失 / shape 含未知维或未知秩 → 输出未知秩并成功
    if (size_tensor == nullptr || Ops::Base::IsUnknownShape(size_tensor->GetOriginShape()) ||
        Ops::Base::IsUnknownRank(size_tensor->GetOriginShape())) {
        Ops::Base::SetUnknownRank(*yShape);
        return ge::GRAPH_SUCCESS;
    }

    ge::DataType size_dtype = size_tensor->GetDataType();

    bool get_value_ok = false;
    switch (size_dtype) {
        case ge::DT_INT32: {
            get_value_ok = Ops::Base::GetValueToShape<int32_t>(size_tensor, *yShape);
            break;
        }
        case ge::DT_INT64: {
            get_value_ok = Ops::Base::GetValueToShape<int64_t>(size_tensor, *yShape);
            break;
        }
        default:
            OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "size", Ops::Base::ToString(size_dtype).c_str(),
                                      "int32 or int64");
            return ge::GRAPH_FAILED;
    }

    // size 非常量（取 const 值失败）→ 未知秩，而非静默保留空 shape
    if (!get_value_ok) {
        Ops::Base::SetUnknownRank(*yShape);
    }

    OP_LOGD(context->GetNodeName(), "output y = %s", Ops::Base::ToString(*yShape).c_str());
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AsStrided).InferShape(InferShapeForAsStrided).InputsDataDependency({IN_SIZE, IN_STRIDE, IN_OFFSET});
} // namespace ops
