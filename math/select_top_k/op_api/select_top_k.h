/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MATH_SELECT_TOP_K_H_
#define MATH_SELECT_TOP_K_H_
#include "opdev/op_executor.h"
#include <tuple>
namespace l0op {
// 构建 SelectTopK 计算任务；输入为 [outer, axis, inner]，inner 为 1 表示排序轴连续。
std::tuple<aclTensor*, aclTensor*> SelectTopK(const aclTensor* self, const aclTensor* begin, const aclTensor* end,
                                              int64_t k, bool largest, int64_t order, op::DataType indicesType,
                                              int64_t indicesOffset, int64_t outputMode, float valueFill,
                                              int64_t indexFill, aclOpExecutor* executor);
} // namespace l0op
#endif
