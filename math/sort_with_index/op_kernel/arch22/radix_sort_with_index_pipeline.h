/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SORT_WITH_INDEX_RADIX_PIPELINE_H
#define SORT_WITH_INDEX_RADIX_PIPELINE_H

#include "kernel_operator.h"
#include "sort_with_index_constants.h"

namespace SortWithIndexA2 {
using namespace AscendC;

// 每个队列有两个独立 UB 槽；EnQue/DeQue 同步生产和消费，FreeTensor/AllocTensor 确保消费完成后才复用槽位。
class RadixSortWithIndexPipeline {
public:
    __aicore__ inline void Init(TPipe& pipe, uint32_t length)
    {
        inputLength_ = length;
        outputLength_ = length + WORK_GUARD_ELEMENTS;
        pipe.InitBuffer(inputQueue_, RADIX_PIPELINE_BUFFERS, length * (VALUE_BYTES + INDEX_BYTES));
        pipe.InitBuffer(outputQueue_, RADIX_PIPELINE_BUFFERS, outputLength_ * (VALUE_BYTES + INDEX_BYTES));
    }

    __aicore__ inline void Prefetch(const GlobalTensor<uint16_t>& source, const GlobalTensor<int32_t>& sourceIndex,
                                    uint32_t offset, uint32_t count, bool needIndex)
    {
        auto input = inputQueue_.AllocTensor<uint16_t>();
        DataCopyPad(input, source[offset], {1, count * VALUE_BYTES, 0, 0, 0}, {false, 0, 0, 0});
        if (needIndex) {
            DataCopyPad(InputIndices(input), sourceIndex[offset], {1, count * INDEX_BYTES, 0, 0, 0}, {false, 0, 0, 0});
        }
        inputQueue_.EnQue(input);
    }

    __aicore__ inline LocalTensor<uint16_t> AcquireInput() { return inputQueue_.DeQue<uint16_t>(); }

    __aicore__ inline void ReleaseInput(LocalTensor<uint16_t> input) { inputQueue_.FreeTensor(input); }

    __aicore__ inline LocalTensor<int32_t> InputIndices(LocalTensor<uint16_t> input)
    {
        return input.ReinterpretCast<int32_t>()[inputLength_ / HALFWORDS_PER_INDEX];
    }

    __aicore__ inline LocalTensor<uint16_t> AllocateOutput() { return outputQueue_.AllocTensor<uint16_t>(); }

    __aicore__ inline LocalTensor<int32_t> OutputIndices(LocalTensor<uint16_t> output)
    {
        return output.ReinterpretCast<int32_t>()[outputLength_ / HALFWORDS_PER_INDEX];
    }

    __aicore__ inline void WriteOutput(LocalTensor<uint16_t> output, const GlobalTensor<uint16_t>& target,
                                       const GlobalTensor<int32_t>& targetIndex, uint32_t offset, uint32_t count)
    {
        outputQueue_.EnQue(output);
        auto ready = outputQueue_.DeQue<uint16_t>();
        DataCopyPad(target[offset], ready, {1, count * VALUE_BYTES, 0, 0, 0});
        DataCopyPad(targetIndex[offset], OutputIndices(ready), {1, count * INDEX_BYTES, 0, 0, 0});
        outputQueue_.FreeTensor(ready);
    }

private:
    TQue<TPosition::VECIN, RADIX_PIPELINE_BUFFERS> inputQueue_;
    TQue<TPosition::VECOUT, RADIX_PIPELINE_BUFFERS> outputQueue_;
    uint32_t inputLength_, outputLength_;
};
} // namespace SortWithIndexA2

#endif // SORT_WITH_INDEX_RADIX_PIPELINE_H
