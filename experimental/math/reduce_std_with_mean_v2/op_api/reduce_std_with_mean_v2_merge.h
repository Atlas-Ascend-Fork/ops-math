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
 * \file reduce_std_with_mean_v2_merge.h
 * \brief L0 API declaration for ReduceStdWithMeanV2Merge (two-stage stage 2)
 */

#ifndef OP_API_INC_LEVEL0_REDUCE_STD_WITH_MEAN_WELFORD_MERGE_H
#define OP_API_INC_LEVEL0_REDUCE_STD_WITH_MEAN_WELFORD_MERGE_H
#include "opdev/op_executor.h"
#include <tuple>

namespace l0op {
// Stage 2 of the two-stage cross-core reduction: Chan-merges the partial tensor's
// [M, partialK, 3] (n, mean, m2) triples per row and returns (std, mean) as
// float32 (the partial tensor itself is fp32); the caller Casts to the
// original input dtype when needed.
const std::tuple<const aclTensor*, const aclTensor*> ReduceStdWithMeanV2Merge(const aclTensor* partial,
                                                                              const aclIntArray* dim, int64_t partialK,
                                                                              int64_t correction, bool keepdim,
                                                                              bool invert, float eps,
                                                                              aclOpExecutor* executor);
} // namespace l0op

#endif
