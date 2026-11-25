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
 * \file rrelu_with_noise.cpp
 * \brief
 */

#include "rrelu_with_noise_train.h"
#include "op_api/aclnn_check.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;
namespace l0op {

OP_TYPE_REGISTER(RreluWithNoiseTrain);
// AICore算子kernel
static const aclTensor* RreluWithNoiseTrainAiCore(const aclTensor* x, const aclTensor* seed, const aclTensor* offset,
                                                  aclTensor* y, aclTensor* noise, const float lower, const float upper,
                                                  aclOpExecutor* executor)
{
    L0_DFX(RreluWithNoiseTrainAiCore, x, seed, offset, lower, upper);
    ADD_TO_LAUNCHER_LIST_AICORE(RreluWithNoiseTrain, OP_ATTR_NAMES({"lower", "upper"}), OP_INPUT(x, seed, offset),
                                OP_OUTPUT(y, noise), OP_ATTR(lower, upper));
    return y;
}

static aclTensor* ScalarToSeedOffsetTensor(aclOpExecutor* executor, const int64_t value)
{
    return executor->AllocHostTensor(&value, 1, op::DataType::DT_INT64);
}

const aclTensor* RreluWithNoiseTrain(const aclTensor* x, const int64_t seed, const int64_t offset, const float lower,
                                     const float upper, const aclTensor** noise, aclOpExecutor* executor)
{
    if (x == nullptr || noise == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "input of RreluWithNoiseTrain must not be nullptr.");
        return nullptr;
    }
    constexpr int64_t offsetMultiple = 4;
    if (offset % offsetMultiple != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "offset of RreluWithNoiseTrain must be a multiple of 4.");
        return nullptr;
    }

    auto seedTensor = ScalarToSeedOffsetTensor(executor, seed);
    if (seedTensor == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "create seed tensor failed.");
        return nullptr;
    }
    auto offsetTensor = ScalarToSeedOffsetTensor(executor, offset);
    if (offsetTensor == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "create offset tensor failed.");
        return nullptr;
    }

    auto y = executor->AllocTensor(x->GetViewShape(), x->GetDataType(), x->GetViewFormat());
    auto noiseTensor = executor->AllocTensor(x->GetViewShape(), x->GetDataType(), x->GetViewFormat());
    if (y == nullptr || noiseTensor == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AllocTensor failed for y or noise in RreluWithNoiseTrain.");
        return nullptr;
    }
    *noise = noiseTensor;

    return RreluWithNoiseTrainAiCore(x, seedTensor, offsetTensor, y, noiseTensor, lower, upper, executor);
}
} // namespace l0op
