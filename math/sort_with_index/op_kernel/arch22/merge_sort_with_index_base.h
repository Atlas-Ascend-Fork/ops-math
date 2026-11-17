/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SORT_WITH_INDEX_MERGE_BASE_ARCH22_H
#define SORT_WITH_INDEX_MERGE_BASE_ARCH22_H
#include "merge_sort_with_index_common.h"

namespace SortWithIndexA2 {
// 原生归并共用的张量、UB 访问和排序键构造；具体排序及输出流程由派生类实现。
template <typename T>
class MergeSortWithIndexBase {
public:
    // 此初始化用于 StreamNative 的单核整行处理；Native GM 和 SingleRun 分别实现自己的缓冲规划。
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR index, GM_ADDR y, GM_ADDR outputIndex, GM_ADDR workspace,
                                const SortWithIndexTilingDataA2& data)
    {
        length_ = data.length;
        rows_ = data.rows;
        chunk_ = data.alignedLength;
        total_ = chunk_ * data.tileCount;
        descending_ = data.descending;
        input_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(x));
        inputIndex_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(index));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(y));
        outputIndex_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(outputIndex));
        // 按 Host 约定的每核跨度划分 workspace；传入指针已越过系统预留区。
        const uint64_t rowBytes = uint64_t(total_) * (MERGE_PROPOSAL_BYTES + INDEX_BYTES);
        workspace += uint64_t(GetBlockIdx()) * rowBytes;
        pingGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
        pipe_.InitBuffer(arena_, total_ * MERGE_PROPOSAL_BYTES);
        pipe_.InitBuffer(small_, STREAM_MERGE_PAYLOAD_ELEMENTS * INDEX_BYTES);
    }

protected:
    static constexpr uint32_t KEY_OFFSET = VALUE_BYTES;
    static constexpr uint32_t SCORE_OFFSET = VALUE_KEY_BYTES;
    static constexpr uint32_t TEMP_OFFSET = SCORE_OFFSET + sizeof(float);

    template <typename U>
    __aicore__ inline LocalTensor<U> At(uint32_t byteOffset = 0)
    {
        return arena_.template Get<U>()[byteOffset / sizeof(U)];
    }

    __aicore__ inline void MakeKeys(uint32_t count)
    {
        MakeSortKeys<T>(At<int16_t>(), At<int16_t>(KEY_OFFSET * chunk_), At<int16_t>(SCORE_OFFSET * chunk_),
                        At<int16_t>(SCORE_OFFSET * chunk_)[chunk_], At<float>(TEMP_OFFSET * chunk_),
                        small_.template Get<uint8_t>(), descending_, count);
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> arena_, small_;
    GlobalTensor<uint16_t> input_, output_;
    GlobalTensor<int32_t> inputIndex_, outputIndex_;
    GlobalTensor<float> pingGm_;
    uint32_t length_, rows_, chunk_, fanout_, total_;
    bool descending_;
};
} // namespace SortWithIndexA2
#endif // SORT_WITH_INDEX_MERGE_BASE_ARCH22_H
