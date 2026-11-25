/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include "aclnn_minimum.h"
#include "minimum.h"
#include "math/logical_and/op_api/logical_and.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/aclnn_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "op_api/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

constexpr size_t MAX_DIM_LEN = 8;
// 特定数据类型下，广播轴与连续非广播轴合并后允许的最大段数
constexpr int64_t MAX_BROADCAST_SEGMENT_NUM = 5;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT,   op::DataType::DT_INT32,  op::DataType::DT_INT64,
    op::DataType::DT_FLOAT16, op::DataType::DT_INT16,  op::DataType::DT_INT8,
    op::DataType::DT_UINT8,   op::DataType::DT_DOUBLE, op::DataType::DT_BOOL};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_FLOAT16,
    op::DataType::DT_INT16, op::DataType::DT_INT8,  op::DataType::DT_UINT8, op::DataType::DT_DOUBLE,
    op::DataType::DT_BOOL,  op::DataType::DT_BF16};

static bool CheckNotNull(const aclTensor* self, const aclTensor* other, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(other, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static inline const std::initializer_list<op::DataType>& GetDtypeSupportList()
{
    if ((GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
         GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E) ||
        IsRegBase()) {
        return ASCEND910B_DTYPE_SUPPORT_LIST;
    } else {
        return ASCEND910_DTYPE_SUPPORT_LIST;
    }
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* other, const aclTensor* out)
{
    // 检查self的数据类型是否在Minimum算子的支持列表内
    auto supportList = GetDtypeSupportList();
    OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);

    // 检查other的数据类型是否在Minimum算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(other, supportList, return false);

    // 检查out的数据类型是否在Minimum算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(out, supportList, return false);
    return true;
}

static bool CheckComputeDtype(const aclTensor* self, const aclTensor* other, const aclTensor* out)
{
    // 根据平台选择数据类型推导方式
    op::DataType promoteType = IsRegBase() ? op::BinaryOpTypePromote(self, other) :
                                             op::PromoteType(self->GetDataType(), other->GetDataType());
    if (promoteType == DataType::DT_UNDEFINED) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Self dtype %s and other dtype %s can not promote dtype.",
                op::ToString(self->GetDataType()).GetString(), op::ToString(other->GetDataType()).GetString());
        return false;
    }

    // 检查推导后的数据类型能否转换为输出的数据类型
    OP_CHECK_RESULT_DTYPE_CAST_FAILED(promoteType, out->GetDataType(), return false);
    return true;
}

static bool CheckShape(const aclTensor* self, const aclTensor* other, const aclTensor* out)
{
    OP_CHECK_MAX_DIM(self, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(other, MAX_DIM_LEN, return false);

    op::Shape broadcastShape;
    OP_CHECK_BROADCAST_AND_INFER_SHAPE(self, other, broadcastShape, return false);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(out, broadcastShape, return false);
    return true;
}

// 参考 docs/zh/context/broadcast_relationship.md 的限制：当参与广播的数据类型（或推导后的数据类型）为
// COMPLEX64、COMPLEX128、DOUBLE、INT16、UINT16、UINT64 时，连续广播轴与连续非广播轴合并后的段数需不超过
// MAX_BROADCAST_SEGMENT_NUM
static const std::initializer_list<op::DataType> BROADCAST_SEGMENT_LIMIT_DTYPE_LIST = {
    op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128, op::DataType::DT_DOUBLE,
    op::DataType::DT_INT16,     op::DataType::DT_UINT16,     op::DataType::DT_UINT64};

static bool IsBroadcastSegmentLimitDtype(op::DataType dtype)
{
    for (auto supportDtype : BROADCAST_SEGMENT_LIMIT_DTYPE_LIST) {
        if (supportDtype == dtype) {
            return true;
        }
    }
    return false;
}

// 将两个输入按广播规则逐轴标记为“需要广播”/“不需要广播”，相邻同标记轴合并后统计段数
static int64_t CountBroadcastSegments(const aclTensor* self, const aclTensor* other)
{
    auto selfShape = self->GetViewShape();
    auto otherShape = other->GetViewShape();
    size_t selfRank = selfShape.GetDimNum();
    size_t otherRank = otherShape.GetDimNum();
    size_t rank = std::max(selfRank, otherRank);
    int64_t segments = 0;
    bool lastBroadcast = false;
    for (size_t i = 0; i < rank; i++) {
        int64_t selfDim = i + selfRank >= rank ? selfShape.GetDim(i + selfRank - rank) : 1;
        int64_t otherDim = i + otherRank >= rank ? otherShape.GetDim(i + otherRank - rank) : 1;
        bool broadcast = (selfDim != otherDim);
        if (i == 0 || broadcast != lastBroadcast) {
            segments++;
            lastBroadcast = broadcast;
        }
    }
    return segments;
}

static bool CheckBroadcastSegmentLimit(const aclTensor* self, const aclTensor* other, op::DataType computeType)
{
    if (!IsBroadcastSegmentLimitDtype(computeType)) {
        return true;
    }
    int64_t segments = CountBroadcastSegments(self, other);
    if (segments > MAX_BROADCAST_SEGMENT_NUM) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Broadcast segments of self and other is %ld after merging adjacent axes, exceeds the limit %ld "
                "for dtype %s.",
                segments, MAX_BROADCAST_SEGMENT_NUM, op::ToString(computeType).GetString());
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclTensor* other, const aclTensor* out)
{
    CHECK_RET(CheckNotNull(self, other, out), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(self, other, out), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckComputeDtype(self, other, out), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(self, other, out), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

// RegBase平台计算类型优化，输入类型安全时直接计算，否则推导后统一转换
static void GetComputeType(const aclTensor* self, const aclTensor* other, const aclTensor* out, DataType& computeType,
                           bool& needOutputCast)
{
    needOutputCast = true;
    if (!IsRegBase()) {
        computeType = op::PromoteType(self->GetDataType(), other->GetDataType());
        return;
    }
    auto inDtype = self->GetDataType();
    auto outDtype = out->GetDataType();
    if (inDtype == other->GetDataType() && op::IsMaxMinSafeInputDtype(inDtype, outDtype)) {
        computeType = inDtype;
        needOutputCast = (computeType != outDtype);
    } else if (inDtype == other->GetDataType()) {
        computeType = outDtype;
        needOutputCast = false;
    } else {
        computeType = op::BinaryOpTypePromote(self, other);
        needOutputCast = (computeType != outDtype);
    }
}

aclnnStatus aclnnMinimumGetWorkspaceSize(const aclTensor* self, const aclTensor* other, aclTensor* out,
                                         uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnMinimum, DFX_IN(self, other), DFX_OUT(out));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto ret = CheckParams(self, other, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    if (self->IsEmpty() || other->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    if (self->GetStorageFormat() != Format::FORMAT_ND) {
        OP_LOGW("Format only supports ND");
    }
    DataType computeType = DataType::DT_UNDEFINED;
    bool needOutputCast = true;
    GetComputeType(self, other, out, computeType, needOutputCast);
    CHECK_RET(computeType != DataType::DT_UNDEFINED, ACLNN_ERR_INNER);
    CHECK_RET(CheckBroadcastSegmentLimit(self, other, computeType), ACLNN_ERR_PARAM_INVALID);
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto selfCasted = l0op::Cast(selfContiguous, computeType, uniqueExecutor.get());
    CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto otherContiguous = l0op::Contiguous(other, uniqueExecutor.get());
    CHECK_RET(otherContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto otherCasted = l0op::Cast(otherContiguous, computeType, uniqueExecutor.get());
    CHECK_RET(otherCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor* minimumOpOut = nullptr;
    if (computeType == op::DataType::DT_BOOL) {
        minimumOpOut = l0op::LogicalAnd(selfCasted, otherCasted, uniqueExecutor.get());
    } else {
        minimumOpOut = l0op::Minimum(selfCasted, otherCasted, uniqueExecutor.get());
    }
    CHECK_RET(minimumOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor* finalOut = minimumOpOut;
    if (needOutputCast) {
        finalOut = l0op::Cast(minimumOpOut, out->GetDataType(), uniqueExecutor.get());
        CHECK_RET(finalOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    auto viewCopyResult = l0op::ViewCopy(finalOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMinimum(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMinimum);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
