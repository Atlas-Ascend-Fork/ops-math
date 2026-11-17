/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SELECT_TOP_K_BASE_CONFIG_H
#define SELECT_TOP_K_BASE_CONFIG_H
#include <cstdint>

namespace SelectTopKBase {
// 通用计算路径编号，同时用于 Host 规划、内核分发和日志诊断。
enum Route : uint32_t {
    ROUTE_COPY = 0,
    ROUTE_ROW_SORT = 1,
    ROUTE_UB_ACCUMULATE = 2,
    ROUTE_GM_MERGE = 3,
    ROUTE_EMPTY = 4,
    ROUTE_FILTER_MERGE = 5,
    ROUTE_SHARDED_SELECT = 6,
    ROUTE_WIDE_FILTER = 7,
    ROUTE_COMPACT_ROW_SORT = 8,
};

// 2201 硬件/API 约束；当前构建未开启 --cce-disable-asc-reserved-ubuf。
// 架构依据：AscendC 开发指南 / 硬件实现 / NPU Arch 2201。
constexpr uint64_t UB_API_RESERVE = 8 * 1024;
constexpr uint64_t UB_FRAMEWORK_RESERVE = 256;
constexpr uint64_t UB_RESERVE = UB_API_RESERVE + UB_FRAMEWORK_RESERVE;
constexpr uint32_t UB_BLOCK_BYTES = 32;
constexpr uint32_t PAIR_BYTES = 8; // FP32 key + UINT32 原始轴内索引。
constexpr uint32_t SORT32_ELEMENTS = 32;
constexpr uint32_t VECTOR_MAX_REPEATS = 255;
constexpr uint32_t SORT32_MAX_ELEMENTS = SORT32_ELEMENTS * VECTOR_MAX_REPEATS;
constexpr uint32_t FLOAT_VECTOR_ELEMENTS = 64;
constexpr uint32_t MERGE_MAX_ALIGNED_ITEMS = 4092; // 单路最多 4095 项，DMA 按 4 个 pair 对齐。

// 数据表示和搬运单位；同值常量按用途区分，避免混淆字节数、位数和元素数。
constexpr auto VALUE16_BYTES = 2;               // FP16/BF16 原值的字节数。
constexpr auto VALUE32_BYTES = 4;               // FP32 排序键的字节数。
constexpr auto INDEX32_BYTES = 4;               // 核内原轴索引的字节数。
constexpr auto INDEX64_BYTES = 8;               // 对外 INT64 索引的字节数。
constexpr auto BITS_PER_BYTE = 8;               // 比较掩码的字节与元素位数换算。
constexpr auto MASK_WORD16_BITS = 16;           // 按 UINT16 执行掩码逻辑运算时的位数。
constexpr auto MASK_WORD32_BITS = 32;           // 按 UINT32 修剪尾掩码时的位数。
constexpr auto FLOAT_BLOCK_ELEMENTS = 8;        // 一个 32B DMA 块容纳的 FP32/INT32 元素数。
constexpr auto VECTOR_REPEAT_BLOCKS = 8;        // 一个向量 repeat 跨越的 32B 块数。
constexpr auto PAIR_COMPONENTS = 2;             // 一个排序 pair 包含 key 和 index 两个 32 位分量。
constexpr auto PAIR_ALIGNMENT = 4;              // 一个 32B 块容纳的排序 pair 数。
constexpr auto DOUBLE_BUFFER_COUNT = 2;         // ping/pong 缓冲数量。
constexpr auto TWO_MERGE_RUNS = 2;              // 双路累积或最终归并的输入路数。
constexpr auto MERGE_WAYS = 4;                  // MrgSort 每次最多合并的有序路数。
constexpr auto MERGE_THIRD_SOURCE = 2;          // 第三路在定长源数组中的位置。
constexpr auto MERGE_FOURTH_SOURCE = 3;         // 第四路在定长源数组中的位置。
constexpr auto MERGE_ALL_SOURCES_MASK = 15;     // 四路均有效时的 MrgSort 位掩码。
constexpr auto MERGE_TWO_SOURCES_MASK = 3;      // 前两路有效时的 MrgSort 位掩码。
constexpr auto MERGE_THREE_SOURCES_MASK = 7;    // 前三路有效时的 MrgSort 位掩码。
constexpr auto MERGE_THIRD_SOURCE_MASK = 4;     // 单独标记第三路有效。
constexpr auto MERGE_FOURTH_SOURCE_MASK = 8;    // 单独标记第四路有效。
constexpr auto VECTOR_MASK_WORDS = 2;           // AscendC 掩码参数由两个 64 位字组成。
constexpr auto HALF_PREFIX_ALIGNMENT = 8;       // 半 K 前缀先按八项向上取整，再映射成四个 pair。
constexpr auto LOCAL_PREFIX_MAX_LISTS = 8;      // 最终前缀归并由单核接管的最大列表数。
constexpr auto SHORT_ROW_REPEAT_BLOCKS = 4;     // 一行 32 个 FP32 项对应四个 32B 块。
constexpr auto GATHER_INDEX_PATTERN = 2;        // GatherMask 从交错 pair 中提取奇数位索引。
constexpr auto MERGE_ALL_SOURCE_COMPONENTS = 8; // 四路 pair 对应的 32 位分量数。
constexpr auto MERGE_FOURTH_SOURCE_COMPONENT_OFFSET = 6; // 第四路 pair 的分量偏移。
constexpr auto HALF_DIVISOR = 2;                         // 二分查找或容量减半时使用的除数。
constexpr auto INDEX_SEQUENCE_SEED = 8;                  // 线性索引先以标量写满一个 32B 块。
constexpr auto INDEX_SEQUENCE_GROWTH = 2;                // 复制已有序列时每轮扩展一倍。
constexpr auto KEY_INDEX_BYTES = 8;                      // 分离的 key、index 数组每项总字节数。
constexpr auto INDEX_AND_OFFSET_BYTES = 8;               // INT32 索引和 Gather 字节偏移的总字节数。
constexpr auto KEY_INDEX_BYTES_64 = 8ULL;                // 大地址空间内的 key/index 字节偏移。
constexpr auto RAW_KEY_INDEX_BYTES = 12;                 // 紧凑排序中原值、键和索引三份 32 位存储。
constexpr auto DOUBLE_PAIR_BYTES = 16;                   // 两份 pair 缓冲每项总字节数。
constexpr auto COMPACT_SORT_BYTES = 20;                  // 紧凑排序的三份 32 位数组和一份 pair。
constexpr auto COMPACT_SORT_BYTES_64 = 20ULL;            // Host 计算大块容量时保留 64 位乘法。
constexpr auto MERGE_INPUT_BYTES = 32;                   // 每路各一个 pair 的四路输入总字节数。
constexpr auto MERGE_INPUT_BYTES_64 = 32ULL;             // Host 检查四路输入窗口时使用的 64 位字节数。
constexpr auto MERGE_WORK_BYTES = 64;                    // 四路输入加四路输出的总字节数。
constexpr auto MERGE_WORK_BYTES_64 = 64ULL;              // Host 容量计算使用的 64 位版本。
constexpr auto INDEX32_BYTES_64 = 4ULL;                  // 工作区偏移乘法显式保持 64 位。
constexpr auto SORT32_ELEMENTS_64 = 32ULL;               // 批量短行容量计算使用的 64 位项数。
constexpr auto SORT32_PAIR_COMPONENTS = 64;              // 一组 Sort32 输出包含的 key/index 分量数。
constexpr auto WIDE_SCAN_BITS_PER_ITEM = 97;             // 三份 32 位数组加一位筛选掩码。
constexpr auto SORT_STAGE2_ITEMS = 128;                  // Sort32 四路归并一次后的有序段长度。
constexpr auto SORT_STAGE3_ITEMS = 512;                  // 再次四路归并后的有序段长度。
constexpr auto SORT_STAGE4_ITEMS = 2048;                 // 第三次四路归并后的有序段长度。
constexpr auto SORT_LEAF_SMALL_ITEMS = 1024;             // 小叶子固定排序模板的项数。
constexpr auto TILING_DATA_BYTES = 128;                  // Host/Device 共享 tiling 结构的固定字节数。
constexpr auto FP32_PADDING_BITS = 0xffffffffU;          // 比有效 NaN 更小的无效尾部编码。
constexpr auto FP32_QUIET_NAN_BITS = 0x7fc00000U;        // 有效 NaN 使用的统一排序键。
constexpr auto FP32_ABS_BITS = 0x7fffffffU;              // 去除 FP32 符号位的掩码。
constexpr auto FP32_POS_INF_BITS = 0x7f800000U;          // FP32 正无穷编码。
constexpr auto FP32_NEG_INF_BITS = 0xff800000U;          // FP32 负无穷编码。
constexpr auto FP32_SIGN_BIT = 0x80000000U;              // FP32 符号位，用于恢复排序方向。
constexpr auto FP16_POS_INF_BITS = 0x7c00U;              // FP16 正无穷编码。
constexpr auto FP16_NEG_INF_BITS = 0xfc00U;              // FP16 负无穷编码。
constexpr auto BF16_POS_INF_BITS = 0x7f80U;              // BF16 正无穷编码。
constexpr auto BF16_NEG_INF_BITS = 0xff80U;              // BF16 负无穷编码。
constexpr auto MASK_BYTE_ALL_BITS = 0xffULL;             // 取出比较掩码中的一个完整字节。
constexpr auto PACKED_INDEX_LOW_BITS = 0xffffffffULL;    // 打包索引中低 32 位的有效位。
constexpr auto BOUND_HAS_BEGIN = 1U;                     // 可选区间起点存在。
constexpr auto BOUND_HAS_END = 2U;                       // 可选区间终点存在。
constexpr auto BOUND_BROADCAST_BEGIN = 4U;               // 起点按单元素广播。
constexpr auto BOUND_BROADCAST_END = 8U;                 // 终点按单元素广播。

namespace Tuning {
// 下列数值是算法选择阈值；硬件容量仍由 Host 查询并验证。
constexpr uint32_t SHORT_ROWS_MIN_BATCH = 16;
// 用户特化范围；K=1 保留已验证的通用实现，过滤模式从 K=2 开始。
constexpr uint64_t FILTER_MIN_AXIS = 16384;
constexpr uint64_t FILTER_MIN_K = 2;
constexpr uint64_t FILTER_MAX_K = 4096;
// 大 K 优先使用较大过滤页；较小 UB 下逐级减半并重新检查布局容量。
constexpr uint32_t FILTER_PREFERRED_BLOCK = 2048;
constexpr uint32_t FILTER_MIN_BLOCK = 64;
// 少行路径按 K 的大小选择宽块过滤、叶子排序和前缀归并。
constexpr uint32_t TINY_K = 32;
constexpr uint32_t SMALL_K = 256;
constexpr uint32_t PREFIX_MERGE_MIN_K = 1024;
constexpr uint32_t TINY_LEAF_BLOCK = 1024;
constexpr uint32_t SMALL_LEAF_BLOCK = 2048;
constexpr uint32_t MAX_LEAF_BLOCK = 4096;
// 扩展叶子将 Sort32 拆为最多 255 次 repeat 和一个尾部 repeat。
constexpr uint32_t MAX_EXTENDED_LEAF_BLOCK = 8192;
constexpr uint32_t FINAL_MERGE_CAPACITY = 4096; // 候选收尾双缓冲的经验容量上限，独立于输入块长。
constexpr uint32_t WIDE_MIN_BLOCK = 4096;
constexpr uint32_t WIDE_TINY_SEED = 1024;
constexpr uint32_t WIDE_SMALL_SEED = 2048;
constexpr uint32_t WIDE_SORT_CHUNK = 2048;
constexpr uint32_t SMALL_K_MIN_SHARD_ITEMS = 2048;
constexpr uint32_t LARGE_K_MIN_SHARD_ITEMS = 4096;
constexpr uint32_t GENERAL_WINDOW_DIVISOR = 8;
constexpr uint32_t PAGE_STEP_MULTIPLIER = 2654435761U;
constexpr uint32_t ROW_PAGE_PHASE = 17;
// 前缀归并至少四路；5~7片单核收尾避免额外同步，更多片使用已验证的稀疏树。
constexpr uint32_t PREFIX_MIN_SHARDS = 5;
constexpr uint32_t LOCAL_FINAL_MAX_SHARDS = 7;
constexpr auto LARGE_LEAF_MIN_K = 2048;         // 达到此 K 后优先减少叶子候选段数量。
constexpr auto LEAF_CANDIDATE_COUNT = 5;        // 三档常规叶子及前后两个扩展叶子候选。
constexpr auto LAST_REGULAR_LEAF_CANDIDATE = 4; // 此位置仅用于补充扩展叶子候选。
constexpr auto PARALLEL_FINAL_MIN_K = 8192;     // 双路最终归并启用多核切分的最小 K。
} // namespace Tuning
} // namespace SelectTopKBase
#endif
