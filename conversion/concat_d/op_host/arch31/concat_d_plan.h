/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CONCAT_D_PLAN_ARCH31_H_
#define CONCAT_D_PLAN_ARCH31_H_

#include <cstddef>
#include <cstdint>

#include "../../op_kernel/arch31/concat_d_tiling_data.h"

namespace optiling {
constexpr uint64_t CONCAT_D_ARCH31_VEC_LOCAL_BYTES = 112U * 1024U;
constexpr uint64_t CONCAT_D_ARCH31_UB_RESERVE_BYTES = 16U * 1024U;
constexpr uint64_t CONCAT_D_ARCH31_QUEUE_BUFFER_COUNT = 2U;
constexpr uint64_t CONCAT_D_ARCH31_MAX_TILE_BYTES = 48U * 1024U;

struct ConcatDArch31PlanInput {
    const int64_t* dims;
    size_t tensorCount;
    size_t rank;
    int64_t axis;
    uint32_t elementBytes;
    uint64_t ubBytes;
};

inline uint64_t ConcatDArch31CompactScatterBufferBytes(size_t tensorCount)
{
    if (tensorCount < 2U || tensorCount > concat_d_arch31::MAX_INPUTS) {
        return 0U;
    }
    const uint64_t tableBytes = tensorCount * concat_d_arch31::COMPACT_SCATTER_TABLE_BYTES;
    if (tableBytes >= concat_d_arch31::COMPACT_SCATTER_UB_BYTES) {
        return 0U;
    }
    return ((concat_d_arch31::COMPACT_SCATTER_UB_BYTES - tableBytes) / 4U / 32U) * 32U;
}

inline bool ConcatDArch31SafeMultiply(uint64_t lhs, uint64_t rhs, uint64_t& value)
{
    if (rhs != 0U && lhs > UINT64_MAX / rhs) {
        return false;
    }
    value = lhs * rhs;
    return true;
}

inline bool ConcatDArch31SafeAdd(uint64_t lhs, uint64_t rhs, uint64_t& value)
{
    if (lhs > UINT64_MAX - rhs) {
        return false;
    }
    value = lhs + rhs;
    return true;
}

inline bool ConcatDArch31CanUseCompactScatter(const ConcatDArch31PlanInput& input,
                                              const concat_d_arch31::ConcatDTilingDataPod& plan)
{
    const uint64_t scatterBufferBytes = ConcatDArch31CompactScatterBufferBytes(input.tensorCount);
    if (input.tensorCount < 2U || input.tensorCount > concat_d_arch31::MAX_INPUTS ||
        input.elementBytes != sizeof(uint16_t) || input.ubBytes < concat_d_arch31::COMPACT_SCATTER_UB_BYTES ||
        scatterBufferBytes == 0U || plan.outputRowBytes > scatterBufferBytes) {
        return false;
    }
    for (size_t tensor = 0U; tensor < input.tensorCount; ++tensor) {
        if (plan.inputRowBytes[tensor] == 0U || plan.inputRowBytes[tensor] % sizeof(uint16_t) != 0U) {
            return false;
        }
    }
    return true;
}

inline bool BuildConcatDArch31Plan(const ConcatDArch31PlanInput& input, concat_d_arch31::ConcatDTilingDataPod& plan)
{
    if (input.dims == nullptr || input.tensorCount == 0U || input.tensorCount > concat_d_arch31::MAX_INPUTS ||
        input.rank == 0U || input.rank > 8U || input.elementBytes == 0U ||
        input.axis < -static_cast<int64_t>(input.rank) || input.axis >= static_cast<int64_t>(input.rank) ||
        input.ubBytes < 32U) {
        return false;
    }
    const size_t axis = static_cast<size_t>(input.axis < 0 ? input.axis + static_cast<int64_t>(input.rank) :
                                                             input.axis);
    uint64_t outer = 1U;
    uint64_t inner = 1U;
    for (size_t dim = 0U; dim < input.rank; ++dim) {
        const int64_t reference = input.dims[dim];
        if (reference <= 0) {
            return false;
        }
        for (size_t tensor = 1U; tensor < input.tensorCount; ++tensor) {
            const int64_t current = input.dims[tensor * input.rank + dim];
            if (current <= 0 || (dim != axis && current != reference)) {
                return false;
            }
        }
        if (dim < axis && !ConcatDArch31SafeMultiply(outer, static_cast<uint64_t>(reference), outer)) {
            return false;
        }
        if (dim > axis && !ConcatDArch31SafeMultiply(inner, static_cast<uint64_t>(reference), inner)) {
            return false;
        }
    }

    uint64_t innerBytes = 0U;
    if (!ConcatDArch31SafeMultiply(inner, input.elementBytes, innerBytes)) {
        return false;
    }
    plan = {};
    plan.version = concat_d_arch31::TILING_VERSION;
    plan.tensorNum = static_cast<uint32_t>(input.tensorCount);
    plan.outerCount = outer;
    for (size_t tensor = 0U; tensor < input.tensorCount; ++tensor) {
        const uint64_t axisElements = static_cast<uint64_t>(input.dims[tensor * input.rank + axis]);
        if (!ConcatDArch31SafeMultiply(axisElements, innerBytes, plan.inputRowBytes[tensor]) ||
            plan.inputRowBytes[tensor] == 0U ||
            !ConcatDArch31SafeAdd(plan.outputRowBytes, plan.inputRowBytes[tensor], plan.outputRowBytes)) {
            return false;
        }
    }

    const uint64_t usableUb = input.ubBytes < CONCAT_D_ARCH31_VEC_LOCAL_BYTES ? input.ubBytes :
                                                                                CONCAT_D_ARCH31_VEC_LOCAL_BYTES;
    const uint64_t queueUb = usableUb > CONCAT_D_ARCH31_UB_RESERVE_BYTES ? usableUb - CONCAT_D_ARCH31_UB_RESERVE_BYTES :
                                                                           usableUb;
    uint64_t tileBytes = queueUb / CONCAT_D_ARCH31_QUEUE_BUFFER_COUNT;
    if (tileBytes > CONCAT_D_ARCH31_MAX_TILE_BYTES) {
        tileBytes = CONCAT_D_ARCH31_MAX_TILE_BYTES;
    }
    tileBytes = (tileBytes / 32U) * 32U;
    if (tileBytes == 0U) {
        return false;
    }
    plan.tileBytes = static_cast<uint32_t>(tileBytes);

    if (ConcatDArch31CanUseCompactScatter(input, plan)) {
        plan.flags |= concat_d_arch31::FLAG_COMPACT_SCATTER;
        plan.tileBytes = static_cast<uint32_t>(ConcatDArch31CompactScatterBufferBytes(input.tensorCount));
    }
    return true;
}
} // namespace optiling

#endif // CONCAT_D_PLAN_ARCH31_H_
