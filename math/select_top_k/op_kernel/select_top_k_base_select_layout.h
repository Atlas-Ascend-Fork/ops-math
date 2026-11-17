/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SELECT_TOP_K_BASE_SELECT_LAYOUT_H
#define SELECT_TOP_K_BASE_SELECT_LAYOUT_H

#include <cstdint>
#include "select_top_k_base_config.h"

#if defined(__CCE_AICORE__)
#define TOPKV2_SELECT_LAYOUT_FUNCTION __aicore__
#else
#define TOPKV2_SELECT_LAYOUT_FUNCTION
#endif

namespace SelectTopKBase {
// 通用排序/归并的同生命周期布局。Host 容量核验和 Device 取址共用这份账本。
struct GeneralMergeLayout {
    uint64_t raw = 0;
    uint64_t index = 0;
    uint64_t key = 0;
    uint64_t ping = 0;
    uint64_t pong = 0;
    uint64_t windows = 0;
    uint64_t merged = 0;
    uint64_t appendScratch = 0;
    uint64_t offsets = 0;
    uint64_t accumulator = 0;
    uint64_t nextAccumulator = 0;
    uint64_t bytes = 0;

    // 按计算路径划分各阶段存活缓冲，计算可复用偏移和 UB 总量。
    TOPKV2_SELECT_LAYOUT_FUNCTION constexpr GeneralMergeLayout(uint32_t tile, uint32_t window, uint64_t k,
                                                               uint32_t dtypeBytes, uint32_t route)
    {
        if (route == ROUTE_GM_MERGE || route == ROUTE_COMPACT_ROW_SORT) {
            // 叶子排序、流式归并、输出转换三个阶段分时复用同一 UB 区域。
            // Sort32 后原值、键、索引不再使用，可复用为第二份 pair 缓冲。
            key = static_cast<uint64_t>(tile) * VALUE32_BYTES;
            index = static_cast<uint64_t>(tile) * KEY_INDEX_BYTES;
            ping = static_cast<uint64_t>(tile) * RAW_KEY_INDEX_BYTES;
            pong = 0;
            windows = 0;
            merged = static_cast<uint64_t>(window) * MERGE_INPUT_BYTES;
            // GM 归并从推进后的地址重载窗口，不再需要压缩缓冲。
            appendScratch = offsets = 0;
            bytes = tile * COMPACT_SORT_BYTES_64 > window * MERGE_WORK_BYTES_64 ? tile * COMPACT_SORT_BYTES_64 :
                                                                                  window * MERGE_WORK_BYTES_64;
            return;
        }
        // tile 为 32 的倍数，以下数组均天然满足 32B 对齐。
        index = static_cast<uint64_t>(tile) * dtypeBytes;
        bytes = index + static_cast<uint64_t>(tile) * sizeof(uint32_t);
        if (route == ROUTE_COPY) {
            return;
        }
        key = bytes;
        ping = key + static_cast<uint64_t>(tile) * sizeof(float);
        pong = ping + static_cast<uint64_t>(tile) * PAIR_BYTES;
        windows = pong + static_cast<uint64_t>(tile) * PAIR_BYTES;
        merged = windows + static_cast<uint64_t>(MERGE_WAYS) * window * PAIR_BYTES;
        appendScratch = merged + static_cast<uint64_t>(MERGE_WAYS) * window * PAIR_BYTES;
        offsets = appendScratch + static_cast<uint64_t>(window) * PAIR_BYTES;
        bytes = offsets + static_cast<uint64_t>(MERGE_ALL_SOURCE_COMPONENTS) * window * sizeof(int32_t);
        if (route == ROUTE_UB_ACCUMULATE) {
            const uint64_t accumulatorBytes = (k * PAIR_BYTES + UB_BLOCK_BYTES - 1) / UB_BLOCK_BYTES * UB_BLOCK_BYTES;
            accumulator = bytes;
            nextAccumulator = accumulator + accumulatorBytes;
            bytes = nextAccumulator + accumulatorBytes;
        }
    }
};

// Host 和 Device 使用同一份偏移账本。调用前保证 2<=k<=4096，block 为 64 的倍数且不超过 2048。
struct SelectLayout {
    uint32_t raw;
    uint32_t key;
    uint32_t index;
    uint32_t mask;
    uint32_t pendingKey;
    uint32_t pendingIndex;
    uint32_t ping;
    uint32_t pong;
    uint32_t accumulator;
    uint32_t mergeAfterPing;
    uint32_t mergeAfterPong;
    uint32_t capacity;
    uint32_t alignedK;
    uint32_t bytes = 0;

    // 在当前 UB 布局末尾分配块对齐区域并返回起始偏移。
    TOPKV2_SELECT_LAYOUT_FUNCTION constexpr uint32_t Allocate(uint32_t size)
    {
        const uint32_t offset = bytes;
        bytes += (size + (UB_BLOCK_BYTES - 1)) / UB_BLOCK_BYTES * UB_BLOCK_BYTES;
        return offset;
    }

    // 计算逐页过滤的原值、掩码、候选和累积段的 UB 布局。
    TOPKV2_SELECT_LAYOUT_FUNCTION constexpr SelectLayout(uint32_t block, uint32_t k, uint32_t dtypeBytes)
        : raw(0),
          key(0),
          index(0),
          mask(0),
          pendingKey(0),
          pendingIndex(0),
          ping(0),
          pong(0),
          accumulator(0),
          mergeAfterPing(0),
          mergeAfterPong(0),
          capacity(DOUBLE_BUFFER_COUNT * block),
          alignedK((k + (SORT32_ELEMENTS - 1)) / SORT32_ELEMENTS * SORT32_ELEMENTS)
    {
        raw = Allocate(block * dtypeBytes);
        key = Allocate(block * VALUE32_BYTES);
        index = Allocate(block * VALUE32_BYTES);
        mask = Allocate(block / BITS_PER_BYTE);
        // 新候选触发量为 block；一次过滤最多再加入 block 个槽位，因此容量为 2*block。
        // 排列为 ping[D]、middle[max(D,Ka)]、pong[D]，单位均为8B pair。
        // middle 的前8D字节存放候选key/index；两侧排序缓冲都不覆盖这段输入。
        ping = Allocate(capacity * PAIR_BYTES);
        const uint32_t middleSlots = capacity > alignedK ? capacity : alignedK;
        const uint32_t middle = Allocate(middleSlots * PAIR_BYTES);
        pendingKey = middle;
        pendingIndex = middle + capacity * INDEX32_BYTES;
        pong = Allocate(capacity * PAIR_BYTES);
        accumulator = Allocate(alignedK * PAIR_BYTES);

        // 排序落在ping时，完整归并写middle+pong；落在pong时，写ping+middle。
        // 两个输出范围均可容纳Ka+D个pair，且不覆盖仍存活的排序结果及Acc。
        // middle按max(D,Ka)扩展，因此小UB导致D<Ka时，两种落点仍然安全。
        mergeAfterPing = middle;
        mergeAfterPong = ping;
    }
};

// 单块分片排序：无需过滤缓冲；pong 额外容纳两份 K 的完整归并结果。
struct LeafSelectLayout {
    uint32_t raw;
    uint32_t key;
    uint32_t index;
    uint32_t ping;
    uint32_t pong;
    uint32_t accumulator;
    uint32_t bytes;

    // 计算叶子排序和最终候选归并所需的共享 UB 布局。
    TOPKV2_SELECT_LAYOUT_FUNCTION constexpr LeafSelectLayout(uint32_t block, uint32_t k, uint32_t dtypeBytes,
                                                             uint32_t finalCapacity = 0)
        : raw(0),
          key(block * dtypeBytes),
          index(key + block * VALUE32_BYTES),
          ping(index + block * VALUE32_BYTES),
          pong(ping + block * KEY_INDEX_BYTES),
          accumulator(0),
          bytes(0)
    {
        const uint32_t alignedK = (k + (SORT32_ELEMENTS - 1)) / SORT32_ELEMENTS * SORT32_ELEMENTS;
        const uint32_t mergeSlots = block > DOUBLE_BUFFER_COUNT * alignedK ? block : DOUBLE_BUFFER_COUNT * alignedK;
        if (block > Tuning::MAX_LEAF_BLOCK) {
            raw = 0;
            key = block * VALUE32_BYTES;
            index = block * KEY_INDEX_BYTES;
            ping = block * RAW_KEY_INDEX_BYTES;
            pong = 0;
            accumulator = ping + mergeSlots * PAIR_BYTES;
            if (accumulator < finalCapacity * DOUBLE_PAIR_BYTES)
                accumulator = finalCapacity * DOUBLE_PAIR_BYTES;
            bytes = accumulator + alignedK * PAIR_BYTES;
            if (k >= Tuning::PREFIX_MERGE_MIN_K) {
                // 叶子回写后才启用累积段；最终 ping/pong 复用缓冲前部，累积段位于其后。
                const uint32_t finalSlots = mergeSlots > finalCapacity ? mergeSlots : finalCapacity;
                accumulator = finalSlots * DOUBLE_PAIR_BYTES;
                bytes = accumulator + alignedK * PAIR_BYTES;
                if (bytes < block * COMPACT_SORT_BYTES)
                    bytes = block * COMPACT_SORT_BYTES;
            }
            return;
        }
        accumulator = pong + mergeSlots * PAIR_BYTES;
        // 写回分片后扫描区全部失效。最终两份缓冲从 arena 起点复用，Acc 位于二者之后。
        if (accumulator < finalCapacity * DOUBLE_PAIR_BYTES) {
            accumulator = finalCapacity * DOUBLE_PAIR_BYTES;
        }
        bytes = accumulator + alignedK * PAIR_BYTES;
    }
};

// 宽块过滤：三个 B 项数组在扫描阶段分别保存 key、index、原值。
// 筛选结束后 index 区失效，复用为候选排序的双缓冲；最终归并再复用整个扫描区。
struct WideSelectLayout {
    uint32_t key;
    uint32_t index;
    uint32_t value;
    uint32_t mask;
    uint32_t accumulator;
    uint32_t bytes;

    // 计算宽块扫描、掩码与累积段的 UB 布局。
    TOPKV2_SELECT_LAYOUT_FUNCTION constexpr WideSelectLayout(uint32_t block, uint32_t k)
        : key(0),
          index(block * VALUE32_BYTES),
          value(block * KEY_INDEX_BYTES),
          mask(block * RAW_KEY_INDEX_BYTES),
          accumulator(mask + (block / BITS_PER_BYTE + (UB_BLOCK_BYTES - 1)) / UB_BLOCK_BYTES * UB_BLOCK_BYTES),
          bytes(accumulator + (k + (SORT32_ELEMENTS - 1)) / SORT32_ELEMENTS * SORT32_ELEMENTS * PAIR_BYTES)
    {}
};
} // namespace SelectTopKBase

#undef TOPKV2_SELECT_LAYOUT_FUNCTION
#endif
