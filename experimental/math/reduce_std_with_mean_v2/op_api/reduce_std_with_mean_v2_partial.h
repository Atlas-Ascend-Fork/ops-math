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
 * \file reduce_std_with_mean_v2_partial.h
 * \brief L0 API declaration for ReduceStdWithMeanV2Partial (two-stage stage 1)
 */

#ifndef OP_API_INC_LEVEL0_REDUCE_STD_WITH_MEAN_WELFORD_PARTIAL_H
#define OP_API_INC_LEVEL0_REDUCE_STD_WITH_MEAN_WELFORD_PARTIAL_H
#include "opdev/op_executor.h"

namespace l0op {
// Stage 1 of the two-stage cross-core reduction: splits R into partialK segments,
// one core per segment, and returns the float32 partial workspace tensor of
// shape [M, partialK, 3] holding per-(row, segment) (n, mean, m2) triples.
const aclTensor* ReduceStdWithMeanV2Partial(const aclTensor* self, const aclIntArray* dim, int64_t partialK,
                                            int64_t correction, bool keepdim, bool invert, float eps,
                                            aclOpExecutor* executor);
} // namespace l0op

#endif
