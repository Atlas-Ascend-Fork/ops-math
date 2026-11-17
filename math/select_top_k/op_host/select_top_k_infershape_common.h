#ifndef SELECT_TOP_K_INFERSHAPE_COMMON_H_
#define SELECT_TOP_K_INFERSHAPE_COMMON_H_
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
#include "../op_kernel/select_top_k_config.h"

namespace ops {
// 将 [outer, axis, inner] 的中间维替换为 k，推导 values、indices 的输出形状。
inline ge::graphStatus InferSelectTopKShape(gert::InferShapeContext* context)
{
    auto input = context->GetInputShape(0);
    auto values = context->GetOutputShape(0);
    auto indices = context->GetOutputShape(1);
    auto attrs = context->GetAttrs();
    auto k = attrs == nullptr ? nullptr : attrs->GetInt(0);
    if (input == nullptr || values == nullptr || indices == nullptr || k == nullptr ||
        input->GetDimNum() != ::SelectTopK::INPUT_RANK || *k < 0) {
        return ge::GRAPH_FAILED;
    }
    const int64_t inner = input->GetDim(::SelectTopK::INNER_LAYOUT_DIM);
    // 动态维 -1 留给运行时 tiling 校验；已知 inner 必须落在布局支持范围内。
    if (inner != -1 && (inner < 1 || inner > ::SelectTopK::MAX_LAYOUT_INNER)) {
        return ge::GRAPH_FAILED;
    }
    *values = *input;
    values->SetDim(1, *k);
    *indices = *values;
    return ge::GRAPH_SUCCESS;
}
} // namespace ops
#endif
