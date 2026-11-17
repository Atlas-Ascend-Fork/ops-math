/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "select_top_k.h"
#include "../op_kernel/select_top_k_config.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"

namespace l0op {
OP_TYPE_REGISTER(SelectTopK);

// 分配输出张量并注册统一的 A2 计算任务，具体算法由 Host 的 tiling key 区分。
std::tuple<aclTensor*, aclTensor*> SelectTopK(const aclTensor* self, const aclTensor* begin, const aclTensor* end,
                                              int64_t k, bool largest, int64_t order, op::DataType indicesType,
                                              int64_t indicesOffset, int64_t outputMode, float valueFill,
                                              int64_t indexFill, aclOpExecutor* executor)
{
    L0_DFX(SelectTopK, self, begin, end, k, largest, order, indicesType, indicesOffset, outputMode, valueFill,
           indexFill);
    auto shape = self->GetViewShape();
    if (shape.GetDimNum() != ::SelectTopK::INPUT_RANK) {
        return {nullptr, nullptr};
    }
    // 排序轴固定在中间维，输入、输出的 outer 和 inner 保持一致。
    shape.SetDim(1, k);
    auto values = executor->AllocTensor(shape, self->GetDataType(), op::Format::FORMAT_ND);
    auto indices = executor->AllocTensor(shape, indicesType, op::Format::FORMAT_ND);
    if (values == nullptr || indices == nullptr) {
        return {nullptr, nullptr};
    }
    ADD_TO_LAUNCHER_LIST_AICORE(
        SelectTopK, OP_INPUT(self, begin, end), OP_OUTPUT(values, indices),
        OP_ATTR(k, largest, order, indicesType, indicesOffset, outputMode, valueFill, indexFill));
    return {values, indices};
}
} // namespace l0op
