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
 * \file permute_infershape.cpp
 * \brief Permute shape and dtype inference.
 */

#include <vector>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/shape_util.h"
#include "conversion/transpose/op_graph/permute_order_util.h"

using namespace ge;
namespace ops {
namespace {
constexpr size_t IDX_X = 0;
constexpr size_t IDX_Y = 0;
constexpr size_t ATTR_ORDER = 0;
} // namespace

static ge::graphStatus InferShape4Permute(gert::InferShapeContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    const gert::Shape* xShape = context->GetInputShape(IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    gert::Shape* yShape = context->GetOutputShape(IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const gert::TypedContinuousVector<int64_t>* order = attrs->GetListInt(ATTR_ORDER);
    if (order == nullptr) {
        OP_LOGE(context->GetNodeName(), "Permute attr order is missing");
        return ge::GRAPH_FAILED;
    }
    std::vector<int64_t> orderValue;
    if (order->GetSize() > 0) {
        orderValue.assign(order->GetData(), order->GetData() + order->GetSize());
    }

    if (Ops::Base::IsUnknownRank(*xShape)) {
        if (orderValue.empty()) {
            Ops::Base::SetUnknownRank(*yShape);
            return ge::GRAPH_SUCCESS;
        }
        std::vector<int64_t> ignoredPerm;
        if (!NormalizePermuteOrder(orderValue, orderValue.size(), ignoredPerm)) {
            OP_LOGE(context->GetNodeName(), "Permute attr order is not a valid permutation of size %zu",
                    orderValue.size());
            return ge::GRAPH_FAILED;
        }
        Ops::Base::SetUnknownShape(static_cast<int64_t>(orderValue.size()), *yShape);
        return ge::GRAPH_SUCCESS;
    }

    const size_t rank = xShape->GetDimNum();
    std::vector<int64_t> perm;
    if (!NormalizePermuteOrder(orderValue, rank, perm)) {
        OP_LOGE(context->GetNodeName(), "Permute attr order is not a valid permutation of input rank %zu", rank);
        return ge::GRAPH_FAILED;
    }

    yShape->SetDimNum(rank);
    for (size_t i = 0; i < rank; ++i) {
        yShape->SetDim(i, xShape->GetDim(perm[i]));
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Permute).InferShape(InferShape4Permute);
} // namespace ops
