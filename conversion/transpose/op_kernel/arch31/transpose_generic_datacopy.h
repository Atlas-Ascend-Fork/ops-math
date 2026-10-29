/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TRANSPOSE_GENERIC_DATACOPY_ARCH31_H
#define TRANSPOSE_GENERIC_DATACOPY_ARCH31_H

#include <cstdint>

#include "transpose_base_arch31.h"

namespace Transpose {
template <typename T>
class TransposeGenericDataCopy {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const KirinTransposeOpTilingData* tiling, TPipe* pipe)
    {
        tiling_ = tiling;
        input_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
        (void)pipe;
    }

    __aicore__ inline void Process()
    {
        uint64_t start = 0;
        uint64_t count = 0;
        if (!GetCoreRange(tiling_, GetBlockIdx(), start, count)) {
            return;
        }

        for (uint64_t offset = 0; offset < count; ++offset) {
            const uint64_t outputIndex = start + offset;
            const uint64_t inputIndex = GetInputOffset(outputIndex);
            output_.SetValue(outputIndex, input_.GetValue(inputIndex));
        }
    }

private:
    __aicore__ inline uint64_t GetInputOffset(uint64_t outputOffset) const
    {
        uint64_t inputOffset = 0;
        for (uint32_t axis = 0; axis < tiling_->reducedRank; ++axis) {
            const uint64_t coordinate = outputOffset / tiling_->outStride[axis] % tiling_->outShape[axis];
            inputOffset += coordinate * tiling_->srcStride[axis];
        }
        return inputOffset;
    }

    const KirinTransposeOpTilingData* tiling_ = nullptr;
    GlobalTensor<T> input_;
    GlobalTensor<T> output_;
};
} // namespace Transpose

#endif // TRANSPOSE_GENERIC_DATACOPY_ARCH31_H
