/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SELECT_TOP_K_BASE_INDEX_H
#define SELECT_TOP_K_BASE_INDEX_H

#include "select_top_k_base_common.h"

namespace SelectTopKBase {
// count 是 32 的倍数。每次复制当前已有序列并加偏移，向量操作次数随 count 对数增长。
__aicore__ inline void CreateLinearIndex(const LocalTensor<int32_t>& destination, uint32_t first, uint32_t count)
{
    Fence<HardEvent::V_S>();
    for (uint32_t i = 0; i < INDEX_SEQUENCE_SEED; ++i) {
        destination.SetValue(i, static_cast<int32_t>(first + i));
    }
    Fence<HardEvent::S_V>();
    SetMaskNorm();
    ResetMask();
    for (uint32_t generated = INDEX_SEQUENCE_SEED; generated < count; generated *= INDEX_SEQUENCE_GROWTH) {
        const uint32_t copied = Min<uint32_t>(generated, count - generated);
        Adds(destination[generated], destination, static_cast<int32_t>(generated), copied);
        PipeBarrier<PIPE_V>();
    }
}
} // namespace SelectTopKBase
#endif
