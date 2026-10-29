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
 * \file reduce_std_with_mean_v2_partial.cpp
 * \brief L0 implementation for ReduceStdWithMeanV2Partial (two-stage stage 1)
 */

#include "reduce_std_with_mean_v2_partial.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ReduceStdWithMeanV2Partial);

const aclTensor* ReduceStdWithMeanV2Partial(const aclTensor* self, const aclIntArray* dim, int64_t partialK,
                                            int64_t correction, bool keepdim, bool invert, float eps,
                                            aclOpExecutor* executor)
{
    L0_DFX(ReduceStdWithMeanV2Partial, self, dim, partialK, correction, keepdim, invert, eps);
    auto partial = executor->AllocTensor(op::DataType::DT_FLOAT, self->GetStorageFormat(), self->GetOriginalFormat());
    CHECK_RET(partial != nullptr, nullptr);
    INFER_SHAPE(ReduceStdWithMeanV2Partial, OP_INPUT(self), OP_OUTPUT(partial),
                OP_ATTR(dim, partialK, keepdim, invert, eps, correction));
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(ReduceStdWithMeanV2Partial, OP_INPUT(self), OP_OUTPUT(partial),
                                           OP_ATTR(dim, partialK, keepdim, invert, eps, correction));
    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ReduceStdWithMeanV2Partial ADD_TO_LAUNCHER_LIST_AICORE failed."),
             return nullptr);
    return partial;
}
} // namespace l0op
