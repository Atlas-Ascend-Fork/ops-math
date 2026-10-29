/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TRANSPOSE_TENSOR_MOVE_ARCH31_H
#define TRANSPOSE_TENSOR_MOVE_ARCH31_H

#include "transpose_base_arch31.h"

namespace Transpose {
template <typename T>
class TransposeTensorMove {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const KirinTransposeOpTilingData* tiling, TPipe* pipe)
    {
        tiling_ = tiling;
        input_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
        pipe->InitBuffer(queue_, 1, tiling_->ubTileElements * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        uint64_t start = 0;
        uint64_t count = 0;
        if (!GetCoreRange(tiling_, GetBlockIdx(), start, count)) {
            return;
        }

        for (uint64_t offset = 0; offset < count;) {
            const uint32_t tileElements = static_cast<uint32_t>(
                count - offset < tiling_->ubTileElements ? count - offset : tiling_->ubTileElements);
            DataCopyExtParams params{1, static_cast<uint32_t>(tileElements * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
            LocalTensor<T> local = queue_.AllocTensor<T>();
            DataCopyPad(local, input_[start + offset], params, padParams);
            queue_.EnQue(local);
            local = queue_.DeQue<T>();
            DataCopyPad(output_[start + offset], local, params);
            queue_.FreeTensor(local);
            offset += tileElements;
        }
    }

private:
    const KirinTransposeOpTilingData* tiling_ = nullptr;
    GlobalTensor<T> input_;
    GlobalTensor<T> output_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> queue_;
};
} // namespace Transpose

#endif // TRANSPOSE_TENSOR_MOVE_ARCH31_H
