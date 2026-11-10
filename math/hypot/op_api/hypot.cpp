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
 * \file hypot.cpp
 * \brief hypot L0 implementation: broadcast infershape + AiCore launch
 */

#include "hypot.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/data_type_utils.h"
#include "op_api/aclnn_check.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Hypot);

// AiCore 支持 FLOAT / FLOAT16 / BFLOAT16
static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

// 根据dtype判断算子是否支持走AiCore
static bool IsAiCoreSupport(const aclTensor* self, const aclTensor* other)
{
    return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST) &&
           CheckType(other->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

// AICORE算子kernel
static const aclTensor* HypotAiCore(const aclTensor* self, const aclTensor* other, const aclTensor* hypotOut,
                                    aclOpExecutor* executor)
{
    L0_DFX(HypotAiCore, self, other, hypotOut)
    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将Hypot算子加入任务队列
    ADD_TO_LAUNCHER_LIST_AICORE(Hypot, OP_INPUT(self, other), OP_OUTPUT(hypotOut));
    return hypotOut;
}

const aclTensor* Hypot(const aclTensor* self, const aclTensor* other, aclOpExecutor* executor)
{
    op::Shape broadcastShape;
    if (!BroadcastInferShape(self->GetViewShape(), other->GetViewShape(), broadcastShape)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Broadcast %s and %s failed.", op::ToString(self->GetViewShape()).GetString(),
                op::ToString(other->GetViewShape()).GetString());
        return nullptr;
    }

    DataType outType = PromoteType(self->GetDataType(), other->GetDataType());
    auto hypotOut = executor->AllocTensor(broadcastShape, outType);
    if (IsAiCoreSupport(self, other)) {
        return HypotAiCore(self, other, hypotOut, executor);
    }
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Hypot dtype %s is not supported on AiCore.",
            op::ToString(self->GetDataType()).GetString());
    return nullptr;
}
} // namespace l0op
