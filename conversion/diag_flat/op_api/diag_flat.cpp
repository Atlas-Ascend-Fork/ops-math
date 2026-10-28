/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <limits>
#include "diag_flat.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

// *******************************************************************************************************
// ******************************** [AI CORE] DiagFlat 算子入口 ***************************************
// *******************************************************************************************************

namespace l0op {

OP_TYPE_REGISTER(DiagFlat);

// AICORE算子kernel
static const aclTensor* DiagFlatAiCore(const aclTensor* self, int64_t diagonal, aclTensor* diagFlatOut,
                                       aclOpExecutor* executor)
{
    L0_DFX(DiagFlatAiCore, self, diagonal, diagFlatOut);
    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将AiCore DiagFlat算子加入任务队列
    // self是算子的输入，diagonal是算子的属性，out是算子的输出
    auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(DiagFlat, OP_INPUT(self), OP_OUTPUT(diagFlatOut), OP_ATTR(diagonal));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(retAicore != ACLNN_SUCCESS, return nullptr,
                                         "DiagFlat ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return diagFlatOut;
}

const aclTensor* DiagFlat(const aclTensor* self, int64_t diagonal, aclOpExecutor* executor)
{
    // |INT64_MIN| 无法用 int64 正值表示，std::abs 结果未定义（aclnn 入口已拒绝该取值，此处防御）
    if (diagonal == std::numeric_limits<int64_t>::min()) {
        OP_LOGW("diagonal is INT64_MIN, its absolute value overflows int64; keep nullptr output.");
        return nullptr;
    }
    int64_t absDiagonal = diagonal >= 0 ? diagonal : -diagonal;
    auto outputLength = self->Numel() + absDiagonal;
    op::Shape diagflatOutShape = self->GetViewShape();

    diagflatOutShape.SetDimNum(2);
    diagflatOutShape.SetDim(0, outputLength);
    diagflatOutShape.SetDim(1, outputLength);

    auto diagFlatOut = executor->AllocTensor(diagflatOutShape, self->GetDataType(), op::Format::FORMAT_ND);
    return DiagFlatAiCore(self, diagonal, diagFlatOut, executor);
}

} // namespace l0op
