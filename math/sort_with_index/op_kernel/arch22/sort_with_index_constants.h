/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SORT_WITH_INDEX_ARCH22_CONSTANTS_H
#define SORT_WITH_INDEX_ARCH22_CONSTANTS_H

#include <cstdint>

// Host 与 Device 共用的常量；调整容量上限或核数策略时需同步核对 UB 用量和性能。
namespace SortWithIndexA2 {
constexpr uint32_t SIMD_ALIGN_ELEMENTS = 128;
constexpr uint32_t MULTI_BLOCK_MODE = 2;
constexpr uint32_t MERGE_LOCAL_MODE = 4;
constexpr uint32_t MERGE_STREAM_MODE = 6;
constexpr uint32_t MERGE_GM_MODE = 8;
constexpr uint32_t MERGE_STREAM_NATIVE_MODE = 9;
// 原生归并的分段容量受 UB 限制，算法切换阈值由性能决定；行与核的划分由 Host 选择。
constexpr uint32_t NATIVE_MERGE_AXIS_GRAIN = 2048;
constexpr uint32_t NATIVE_MERGE_RUN_ELEMENTS = 4096;
constexpr uint32_t NATIVE_MERGE_COOPERATIVE_RUN_LIMIT = 2 * NATIVE_MERGE_RUN_ELEMENTS;
constexpr uint32_t NATIVE_MERGE_LOCAL_BYTES_PER_ELEMENT = 16;
constexpr uint32_t NATIVE_MERGE_ARENA_BYTES = 186368;
// 整行 NaN 检测对每个补齐元素使用一位掩码，扩大单段长度时须同步预留掩码空间。
constexpr uint32_t NATIVE_MERGE_SMALL_BYTES = 5 * 256;
constexpr uint32_t NATIVE_MERGE_POSITION_SEED_BYTES = 256;
constexpr uint32_t NATIVE_MERGE_ROW_BYTES_PER_ELEMENT = 16;
// 单行包含源索引、字节偏移、原生排序记录及取回的输出。
constexpr uint32_t STREAM_SINGLE_RUN_BYTES_PER_ELEMENT = 5 * sizeof(uint32_t);
// 按对齐粒度向下取 UB 可容纳的整行长度，A2 为 9216 项。
constexpr uint32_t STREAM_SINGLE_RUN_AXIS_LIMIT = NATIVE_MERGE_ARENA_BYTES / STREAM_SINGLE_RUN_BYTES_PER_ELEMENT /
                                                  SIMD_ALIGN_ELEMENTS * SIMD_ALIGN_ELEMENTS;
constexpr uint32_t STREAM_NATIVE_AXIS_LIMIT = 4 * NATIVE_MERGE_RUN_ELEMENTS;
static_assert(STREAM_SINGLE_RUN_AXIS_LIMIT * STREAM_SINGLE_RUN_BYTES_PER_ELEMENT <= NATIVE_MERGE_ARENA_BYTES,
              "single-run stream layout must fit the native arena");
static_assert(STREAM_SINGLE_RUN_AXIS_LIMIT <= NATIVE_MERGE_SMALL_BYTES * 8,
              "single-run NaN fallback mask must fit the small buffer");
// FP32 在该范围内可精确表示整数位置，超过上限改用基数排序。
constexpr uint32_t NATIVE_MERGE_POSITION_LIMIT = 1U << 24;
// 基数排序起始长度按 40 个 AIV 核标定，属于性能阈值，不是 UB 容量限制；输出均携带 INT32 索引。
constexpr uint32_t RADIX_SINGLE_ROW_FIRST_LENGTH = 32769;
constexpr uint32_t RADIX_FEW_ROWS_FIRST_LENGTH = 262144;
constexpr uint32_t RADIX_MANY_ROWS_FIRST_LENGTH = 524288;
constexpr uint32_t RADIX_FEW_ROWS_LIMIT = 8;
// 短行按组合编码归并；较长的 UB 内排序直接携带 INT32 索引，有限值路径省去编码构造和 Gather。
constexpr uint32_t MERGE_AXIS_LIMIT = 2048;
constexpr uint32_t MERGE_SORT_RUN_ELEMENTS = 32;
// 短行批处理不跨行归并，批大小同时受位置编码和实际 UB 容量约束。
constexpr uint32_t MERGE_BATCH_POWER_OF_TWO_LIMIT = 128;
// Sort32 最多重复 255 次，13 位位置编码可覆盖该范围；实际批大小还受 Host 的 UB 预算约束。
constexpr uint32_t MERGE_BATCH_ELEMENTS = UINT8_MAX * MERGE_SORT_RUN_ELEMENTS;
constexpr uint32_t MERGE_LIST_COUNT = 4;
constexpr uint32_t MERGE_PROPOSAL_BYTES = 2 * sizeof(uint32_t);
constexpr uint32_t MERGE_POSITION_BITS = 13;
constexpr uint32_t MERGE_KEY_MASK = 0xffff;
// 将组合整数按位解释为正的正规 FP32，保持键和位置的顺序，避免数值转换丢失低位。
constexpr int32_t MERGE_SCORE_BASE_BITS = 0x30000000;
constexpr float MERGE_PADDING_SCORE = -3.402823466e38F;
static_assert(MERGE_AXIS_LIMIT <= (1U << MERGE_POSITION_BITS), "merge positions must fit the score");
static_assert(MERGE_BATCH_ELEMENTS <= (1U << MERGE_POSITION_BITS), "batch positions must fit the score");
static_assert(MERGE_SCORE_BASE_BITS + (MERGE_KEY_MASK << MERGE_POSITION_BITS) + MERGE_BATCH_ELEMENTS - 1 < 0x7f800000,
              "merge scores must stay finite");

constexpr uint32_t NATIVE_MERGE_MIN_COOPERATIVE_CORES = 4;
constexpr uint32_t STREAM_MERGE_AXIS_LIMIT = 1U << 15;
constexpr uint32_t MERGE_PROPOSAL_ELEMENTS = MERGE_PROPOSAL_BYTES / sizeof(float);
constexpr uint32_t MERGE_PAIR_LIST_COUNT = 2;
constexpr uint32_t MERGE_WORKSPACE_BUFFERS = 2;
// 独立行先排序四段，再使用硬件四路归并流式输出。
constexpr uint32_t STREAM_MERGE_RUNS = 4;
constexpr uint32_t STREAM_MERGE_WINDOW = 2048;
constexpr uint32_t STREAM_MERGE_PAYLOAD_ELEMENTS = 4096;
constexpr uint32_t RADIX_MIN_TILE_ELEMENTS = 1024;
constexpr uint32_t UB_BLOCK_BYTES = 32;
constexpr uint32_t BITS_PER_BYTE = 8;
constexpr uint32_t SDK_RESERVED_UB_BYTES = 8192;
constexpr uint32_t VALUE_BYTES = sizeof(uint16_t);
constexpr uint32_t INDEX_BYTES = sizeof(int32_t);
constexpr uint32_t VALUE_KEY_BYTES = VALUE_BYTES * 2;
constexpr uint32_t VALUE_BITS = VALUE_BYTES * BITS_PER_BYTE;
constexpr uint32_t SIGN_SHIFT = VALUE_BITS - 1;
constexpr uint16_t SIGN_MASK = 0x8000;
constexpr int16_t SIGN_MASK_SIGNED = -32768;
constexpr int16_t ABS_MASK = 0x7fff;
constexpr int32_t FP16_INFINITY_BITS = 0x7c00;
constexpr int32_t BF16_INFINITY_BITS = 0x7f80;

constexpr uint32_t RADIX_BITS = 2;
constexpr uint32_t RADIX_BINS = 1U << RADIX_BITS;
constexpr uint32_t RADIX_MASK = RADIX_BINS - 1;
constexpr uint32_t MAX_RADIX_PASSES = VALUE_BITS / RADIX_BITS;
constexpr uint32_t PREFIX_BATCH_TILES = 256;
constexpr uint32_t RADIX_PIPELINE_BUFFERS = 2;
// tile 记录和核摘要各占一个 DMA 块，均含四个桶计数；核摘要另存最小、最大键，剩余两项保留。
// 偏移以 int32 元素为单位。
constexpr uint32_t HISTOGRAM_RECORD_BYTES = UB_BLOCK_BYTES;
constexpr uint32_t HISTOGRAM_RECORD_ELEMENTS = HISTOGRAM_RECORD_BYTES / INDEX_BYTES;
constexpr uint32_t HISTOGRAM_MIN_OFFSET = RADIX_BINS;
constexpr uint32_t HISTOGRAM_MAX_OFFSET = HISTOGRAM_MIN_OFFSET + 1;

// 保护区容纳向量归约及逐桶 GatherMask 的对齐补齐。
constexpr uint32_t WORK_GUARD_ELEMENTS = 64;
constexpr uint32_t FP32_REPEAT_ELEMENTS = 64;
constexpr uint32_t DMA_VALUE_ELEMENTS = UB_BLOCK_BYTES / VALUE_BYTES;
constexpr uint32_t DMA_INDEX_ELEMENTS = UB_BLOCK_BYTES / INDEX_BYTES;
constexpr uint32_t FP32_POSITION_SEED_ELEMENTS = DMA_INDEX_ELEMENTS;
constexpr uint32_t REDUCE_MAX_OFFSET = DMA_INDEX_ELEMENTS;
constexpr uint32_t HALFWORDS_PER_INDEX = INDEX_BYTES / VALUE_BYTES;
constexpr uint32_t VALUE_BYTE_SHIFT = 1;
constexpr uint32_t BATCH_SCHEDULE_MODE = 1;
} // namespace SortWithIndexA2

#endif // SORT_WITH_INDEX_ARCH22_CONSTANTS_H
