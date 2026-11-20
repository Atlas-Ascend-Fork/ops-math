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
 * \file aclnn_is_inf.cpp
 * \brief
 */

#include "aclnn_is_inf.h"
#include "is_inf_l0.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/platform.h"
#include "op_api/aclnn_check.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
// Ascend910B/Ascend910_93/Ascend950 支持的输入数据类型
static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_910B = {DataType::DT_FLOAT, DataType::DT_FLOAT16,
                                                                        DataType::DT_BF16};

// Ascend310P 及 kirin 系列支持的输入数据类型
static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_310P = {DataType::DT_FLOAT, DataType::DT_FLOAT16};

static inline bool CheckNotNull(const aclTensor* x, const aclTensor* out)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* x, const aclTensor* out)
{
    auto socVersion = op::GetCurrentPlatformInfo().GetSocVersion();
    bool is910BSocVersion = (socVersion == SocVersion::ASCEND910B || socVersion == SocVersion::ASCEND910_93 ||
                             IsRegBase());
    const std::initializer_list<DataType>& supportList = is910BSocVersion ? DTYPE_SUPPORT_LIST_910B :
                                                                            DTYPE_SUPPORT_LIST_310P;

    // 检查x的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(x, supportList, return false);

    // 检查out的数据类型是否为BOOL
    OP_CHECK_DTYPE_NOT_MATCH(out, DataType::DT_BOOL, return false);

    return true;
}

static bool CheckShape(const aclTensor* x, const aclTensor* out)
{
    // x与out的shape必须一致
    OP_CHECK_SHAPE_NOT_EQUAL(x, out, return false);

    return true;
}

static aclnnStatus CheckParams(const aclTensor* x, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(x, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查参数的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid(x, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查x和out的shape是否一致
    CHECK_RET(CheckShape(x, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}
} // namespace

aclnnStatus aclnnIsInfGetWorkspaceSize(const aclTensor* x, const aclTensor* out, uint64_t* workspaceSize,
                                       aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnIsInf, DFX_IN(x), DFX_OUT(out));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(x, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空Tensor处理
    if (x->IsEmpty() || out->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // x如果非连续，需要转换
    auto xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用l0算子IsInf进行计算
    auto isInfResult = l0op::IsInf(xContiguous, uniqueExecutor.get());
    CHECK_RET(isInfResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto viewCopyResult = l0op::ViewCopy(isInfResult, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnIsInf(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnIsInf);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
