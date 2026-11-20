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
 * \file is_inf.cpp
 * \brief
 */

#include "is_inf_l0.h"
#include "opdev/op_dfx.h"
#include "opdev/make_op_executor.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(IsInf);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static bool IsAiCoreSupport(const aclTensor* self) { return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST); }

// AiCore逻辑
static const aclTensor* IsInfAiCore(const aclTensor* self, aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(IsInfAiCore, self, out);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(IsInf, OP_INPUT(self), OP_OUTPUT(out));
    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_INNER_NULLPTR,
                                    "IsInfAiCore ADD_TO_LAUNCHER_LIST_AICORE failed; propagate the executor launch "
                                    "failure."),
             return nullptr);
    return out;
}

const aclTensor* IsInf(const aclTensor* self, aclOpExecutor* executor)
{
    L0_DFX(IsInf, self);
    if (!IsAiCoreSupport(self)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "IsInf does not support input dtype %s.",
                ToString(self->GetDataType()).GetString());
        return nullptr;
    }
    // 根据输入shape申请输出tensor，输出dtype固定为BOOL
    auto out = executor->AllocTensor(self->GetViewShape(), op::DataType::DT_BOOL);
    OP_CHECK(out != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "[is_inf] alloc out tensor failed."), return nullptr);
    return IsInfAiCore(self, out, executor);
}
} // namespace l0op
