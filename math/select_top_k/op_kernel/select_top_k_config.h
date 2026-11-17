/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SELECT_TOP_K_CONFIG_H_
#define SELECT_TOP_K_CONFIG_H_
#include <cstdint>

// 编译器从预处理文本登记入口；每个完整模板仅有一个 key。
#define SELECT_TOP_K_TILING_ROW_SORT 0
#define SELECT_TOP_K_TILING_MERGE_SORT 1
#define SELECT_TOP_K_TILING_RANGE_MERGE 2
#define SELECT_TOP_K_TILING_TOP1 3
#define SELECT_TOP_K_TILING_SHORT_ROWS 4
#define SELECT_TOP_K_TILING_SELECT 5
#define SELECT_TOP_K_TILING_WIDE_SELECT 6
#define SELECT_TOP_K_TILING_LEAF_SELECT 7
#define SELECT_TOP_K_TILING_COPY_ALL 8
#define SELECT_TOP_K_TILING_TINY_ROW 9
#define SELECT_TOP_K_TILING_PREFIX_ROWS 10
#include "select_top_k_strategy.h"

namespace SelectTopK {
// L0 输入和融合布局约定；属性位置必须与算子定义的注册顺序一致。
constexpr auto INPUT_RANK = 3;       // L0 统一接收 [outer, axis, inner]，末轴路径使用 [rows, axis, 1]。
constexpr auto INNER_LAYOUT_DIM = 2; // 输入、输出布局中 inner 所在的维度。
constexpr auto MAX_LAYOUT_INNER = 256; // 融合布局支持的轴后元素数上限，块大小另按UB与Gather容量裁剪。
constexpr auto INPUT_BEGIN = 1; // 可选区间起点输入的位置。
constexpr auto INPUT_END = 2;   // 可选区间终点输入的位置。
constexpr auto ATTR_ORDER = 2;
constexpr auto ATTR_INDICES_TYPE = 3;
constexpr auto ATTR_INDICES_OFFSET = 4;
constexpr auto ATTR_OUTPUT_MODE = 5;
constexpr auto ATTR_VALUE_FILL = 6;
constexpr auto ATTR_INDEX_FILL = 7;
constexpr auto OUTPUT_BOTH = 2;         // 同时输出值和索引。
constexpr auto ORDER_NONE = 0;          // 输出不要求排序。
constexpr auto ORDER_VALUE = 1;         // 有效结果按值排序。
constexpr auto ORDER_INDEX = 2;         // 索引排序预留值，当前不支持。
constexpr auto TILING_DATA_BYTES = 256; // 含 基础调度；只改变候选包内部 Host/Device ABI。

// 只在最终生成执行信息时编码，规划及资源计算不比较或改写 key。
constexpr uint32_t TilingKey(KernelTemplate kernel)
{
    switch (kernel) {
        case KernelTemplate::ROW_SORT:
            return SELECT_TOP_K_TILING_ROW_SORT;
        case KernelTemplate::MERGE_SORT:
            return SELECT_TOP_K_TILING_MERGE_SORT;
        case KernelTemplate::RANGE_MERGE:
            return SELECT_TOP_K_TILING_RANGE_MERGE;
        case KernelTemplate::TOP1:
            return SELECT_TOP_K_TILING_TOP1;
        case KernelTemplate::SHORT_ROWS:
            return SELECT_TOP_K_TILING_SHORT_ROWS;
        case KernelTemplate::SELECT:
            return SELECT_TOP_K_TILING_SELECT;
        case KernelTemplate::WIDE_SELECT:
            return SELECT_TOP_K_TILING_WIDE_SELECT;
        case KernelTemplate::LEAF_SELECT:
            return SELECT_TOP_K_TILING_LEAF_SELECT;
        case KernelTemplate::COPY_ALL:
            return SELECT_TOP_K_TILING_COPY_ALL;
        case KernelTemplate::TINY_ROW:
            return SELECT_TOP_K_TILING_TINY_ROW;
        case KernelTemplate::PREFIX_ROWS:
            return SELECT_TOP_K_TILING_PREFIX_ROWS;
        default:
            return UINT32_MAX;
    }
}
constexpr uint32_t DENSE_PREFIX_DIVISOR = 3;     // 四路试探每路约K/3，产出不足时完整回退。
constexpr uint32_t DENSE_PREFIX_MAX_LEAVES = 16; // 单核依次归并至多四个四路组。
constexpr uint32_t TINY_ROW_ITEMS = 128;         // 单行短轴最多四个Sort32有序段。
constexpr uint32_t SCALAR_ROW_ITEMS = 16; // 极短单行用标量插入排序，避免向量初始化和掩码切换。
constexpr uint32_t SCALAR_SORT_MAX_ITEMS = 8;    // 插入排序只用于更短的行；较长行复用单次 Sort32。
constexpr uint32_t COPY_ALL_TILE = 8192;         // 一次连续搬运及索引生成的上限。
constexpr uint32_t SINGLE_ROW_MAX_AXIS = 2048;   // 单行Gather原值路径最多四层Sort32/四路归并。
constexpr uint32_t COMPACT_ROW_MAX_AXIS = 2048;  // 少行紧凑排序只覆盖无需跨核的短轴。
constexpr uint32_t PREFIX_ROW_WIDTH = 2048;      // 批量小 K 路径使用完整四路归并层级。
constexpr uint32_t PREFIX_ROW_GROUP_ITEMS = 512; // 两级四路归并覆盖的输入数；按整组裁剪无效尾部。
constexpr uint32_t PREFIX_ROW_MIN_AXIS = 1024;   // 轴长超过此值时裁剪中间候选能减少归并工作。
constexpr uint32_t PREFIX_ROW_ITEMS = 64;        // 每个有序分段保留的候选数，覆盖本路径的最大 K。
constexpr uint32_t PREFIX_ROW_MIN_BATCHES = 2;   // 每核至少两行，摊薄固定索引初始化成本。
constexpr uint32_t ROUTE_TOP1_REDUCE = 9;        // 内部单值归约路线，与对外的 tiling key 独立。
constexpr uint32_t TOP1_TILE = 8192;             // 单次归约扫描的最大项数。
constexpr uint32_t TOP1_FIXED_UB = 4608;         // 归约中间结果与临时区的固定容量。
constexpr auto TOP1_REDUCE_TEMP_ITEMS = 1024;    // 归约结果之前预留的 FP32 临时项数。
constexpr auto FINALIZE_MAX_BLOCK = 8192;        // 索引收尾阶段一次转换的最大项数。

// 输入和输出重排的资源阈值，均需额外通过实际 UB 容量检查。
constexpr auto LARGE_K_THRESHOLD = 4096;
constexpr auto GM_MIN_INNER = 32;             // 较宽的非末轴输出才启用 GM 归并的复用重排路径。
constexpr auto PAD_MIN_INNER = 3;             // 超过此行宽后给大 K 源行加间隔以减少 UB 冲突。
constexpr auto OUTPUT_ROW_PADDING = 16;       // 常规输出布局中每行追加的元素数。
constexpr auto REUSED_FP32_PADDING = 8;       // 复用布局下 FP32 行的追加元素数。
constexpr auto MIN_LAYOUT_BLOCK = 16;         // 输入/输出重排的最小列块。
constexpr auto MAX_INPUT_BLOCK = 512;         // 融合输入重排的最大列块。
constexpr auto MAX_OUTPUT_BLOCK = 1024;       // 窄 inner 输出重排的最大列块。
constexpr auto WIDE_OUTPUT_BLOCK = 64;        // 大 K、宽 inner 的输出列块上限。
constexpr auto REUSED_MID_BLOCK = 96;         // 复用布局中 64 和 128 之间的资源档位。
constexpr auto MAX_REUSED_OUTPUT_BLOCK = 128; // 复用布局允许的最大列块。
constexpr auto MAX_OUTPUT_CAPACITY = 6144;    // 常规重排一次 Gather 的容量上限。
constexpr auto MAX_REUSED_CAPACITY = 12288;   // 复用布局和输入重排的容量上限。

namespace Tuning {
constexpr auto PREFIX_SMALL_K = 512;    // 少量叶子时启用前缀归并的 K 上限。
constexpr auto BALANCED_LEAVES = 8;     // 少行场景用于平衡叶子工作的目标分片数。
constexpr auto MAX_FIXED_LEAVES = 20;   // 用固定叶子替代扩展叶子时的分片数上限。
constexpr auto SMALL_LEAF_MAX_K = 64;   // 小叶子替代宽块扫描的 K 上限。
constexpr auto SHORT_AXIS_MAX_ROWS = 8; // 短目标轴保留固定叶子排序的行数上限。
} // namespace Tuning
} // namespace SelectTopK
#endif
