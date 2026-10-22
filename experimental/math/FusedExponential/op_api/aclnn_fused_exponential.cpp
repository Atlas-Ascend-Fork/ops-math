/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <cstdint>
#include <array>
#include <limits>
#include "aclnn_fused_exponential.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/tensor_view_utils.h"
#include "../op_host/fused_exponential_utils.h"

namespace l0op {
OP_TYPE_REGISTER(FusedExponential);
static aclnnStatus Launch(const aclTensor* u, aclTensor* out, float scale, int64_t exponent, bool zero,
                          aclOpExecutor* executor)
{
    return ADD_TO_LAUNCHER_LIST_AICORE(FusedExponential, OP_INPUT(u), OP_OUTPUT(out), OP_ATTR(scale, exponent, zero));
}
} // namespace l0op

namespace {
bool HasUnsafeOutputLayout(const aclTensor* out)
{
    const auto& shape = out->GetViewShape();
    const auto& strides = out->GetViewStrides();
    std::array<size_t, fused_exponential::MAX_RANK> order{};
    if (shape.GetDimNum() > order.size()) {
        return true;
    }
    size_t activeDims = 0;
    for (size_t dim = 0; dim < shape.GetDimNum(); ++dim) {
        if (shape.GetDim(dim) < 0 || strides[dim] < 0 || (shape.GetDim(dim) > 1 && strides[dim] == 0)) {
            return true;
        }
        if (shape.GetDim(dim) > 1) {
            order[activeDims++] = dim;
        }
    }
    for (size_t index = 1; index < activeDims; ++index) {
        const size_t dim = order[index];
        size_t position = index;
        while (position > 0 && strides[order[position - 1]] > strides[dim]) {
            order[position] = order[position - 1];
            --position;
        }
        order[position] = dim;
    }
    // Sufficient non-overlap proof for dense, transposed and strided slices.
    // Reject layouts for which disjoint element storage cannot be established.
    uint64_t span = 1;
    const uint64_t limit = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
    for (size_t index = 0; index < activeDims; ++index) {
        const auto dim = order[index];
        const auto stride = static_cast<uint64_t>(strides[dim]);
        const auto repeats = static_cast<uint64_t>(shape.GetDim(dim) - 1);
        if (stride < span || repeats > (limit - span) / stride) {
            return true;
        }
        span += repeats * stride;
    }
    return false;
}

bool HasPartialContiguousOverlap(const aclTensor* u, const aclTensor* out, uint64_t bytes)
{
    if (!op::IsContiguous(u) || !op::IsContiguous(out)) {
        // Contiguous/AllocTensor + ViewCopy snapshots these paths before writing out.
        return false;
    }
    const auto inputAddress = reinterpret_cast<uintptr_t>(u->GetData());
    const auto outputAddress = reinterpret_cast<uintptr_t>(out->GetData());
    if (inputAddress == outputAddress) {
        return false;
    }
    const uintptr_t distance = inputAddress < outputAddress ? outputAddress - inputAddress :
                                                              inputAddress - outputAddress;
    return distance < bytes;
}
} // namespace

extern "C" aclnnStatus aclnnFusedExponentialGetWorkspaceSize(const aclTensor* u, const aclScalar* lambd, aclTensor* out,
                                                             uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnFusedExponential, DFX_IN(u, lambd), DFX_OUT(out));
    if (!u || !lambd || !out || !workspaceSize || !executor) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "u, lambd, out, workspaceSize and executor must be non-null.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_2201) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "FusedExponential supports Atlas A2/A3 (DAV_2201) only.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const auto dtype = u->GetDataType();
    if ((dtype != op::DataType::DT_FLOAT && dtype != op::DataType::DT_FLOAT16 && dtype != op::DataType::DT_BF16) ||
        out->GetDataType() != dtype || lambd->GetDataType() != op::DataType::DT_DOUBLE ||
        u->GetViewShape() != out->GetViewShape() || u->GetViewShape().GetDimNum() > fused_exponential::MAX_RANK ||
        u->GetStorageFormat() != op::Format::FORMAT_ND || out->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Expected matching ND FLOAT16/BFLOAT16/FLOAT32 tensors of rank 0-8 and a DOUBLE lambd scalar.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    uint64_t count = 0;
    const uint64_t elementBytes = dtype == op::DataType::DT_FLOAT ? sizeof(float) : sizeof(uint16_t);
    if (!fused_exponential::GetElementCount(u->GetViewShape(), elementBytes, count)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Negative dimensions or tensor byte size overflow.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    auto owner = CREATE_EXECUTOR();
    if (!owner.get()) {
        OP_LOGE(ACLNN_ERR_INNER_CREATE_EXECUTOR, "Failed to create executor.");
        return ACLNN_ERR_INNER_CREATE_EXECUTOR;
    }
    if (count == 0) {
        *workspaceSize = 0;
        owner.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    if (HasUnsafeOutputLayout(out) || HasPartialContiguousOverlap(u, out, count * elementBytes)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Output storage must be non-overlapping; partial contiguous alias is unsupported.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const double lambda = lambd->ToDouble();
    const bool zero = std::isinf(lambda) && lambda > 0;
    float scale = 0;
    int64_t exponent = 0;
    if (!zero) {
        int power = 0;
        const double mantissa = std::frexp(lambda, &power);
        scale = static_cast<float>(-1.0 / mantissa);
        // lambda = mantissa * 2^power, so -1/lambda = scale * 2^(-power).
        exponent = -static_cast<int64_t>(power);
    }
    auto input = l0op::Contiguous(u, owner.get());
    if (!input) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Failed to make input contiguous.");
        return ACLNN_ERR_INNER_NULLPTR;
    }
    auto result = out;
    if (!op::IsContiguous(out)) {
        result = owner->AllocTensor(out->GetViewShape(), dtype);
    }
    if (!result) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Failed to allocate contiguous output.");
        return ACLNN_ERR_INNER_NULLPTR;
    }
    auto status = l0op::Launch(input, result, scale, exponent, zero, owner.get());
    if (status != ACLNN_SUCCESS) {
        OP_LOGE(status, "Failed to launch FusedExponential.");
        return status;
    }
    if (result != out && !l0op::ViewCopy(result, out, owner.get())) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Failed to copy the result into the output view.");
        return ACLNN_ERR_INNER_NULLPTR;
    }
    *workspaceSize = owner->GetWorkspaceSize();
    owner.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnFusedExponential(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                             aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFusedExponential);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
