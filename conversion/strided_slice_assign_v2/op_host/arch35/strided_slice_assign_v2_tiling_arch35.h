/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file strided_slice_assign_v2_tiling_arch35.h
 * \brief strided_slice_assign_v2_tiling_arch35.h
 */

#ifndef STRIDED_SLICE_ASSIGN_V2_TILING_ARCH35_H
#define STRIDED_SLICE_ASSIGN_V2_TILING_ARCH35_H

#include <cstdint>

#include "platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"

#include "../../op_kernel/arch35/strided_slice_assign_v2_tiling_data.h"
#include "../../op_kernel/arch35/strided_slice_assign_v2_tilingkey.h"

namespace optiling {

// ---- 输入 / 输出索引 ----
constexpr size_t SSA_INDEX_VAR = 0;
constexpr size_t SSA_INDEX_INPUT_VALUE = 1;
constexpr size_t SSA_INDEX_BEGIN = 2;
constexpr size_t SSA_INDEX_END = 3;
constexpr size_t SSA_INDEX_STRIDES = 4;
constexpr size_t SSA_INDEX_AXES = 5;
constexpr size_t SSA_INDEX_VAR_OUT = 0;

// ---- 方向 ID ----
constexpr int32_t SSA_DIR_ALL_POSITIVE = 0;
constexpr int32_t SSA_DIR_ALL_NEGATIVE = 1;
constexpr int32_t SSA_DIR_MIXED = 2;

// ---- 判据 / 切分常量 ----
constexpr int64_t SSA_SMALL_SHAPE_THRESHOLD = 1024; // 小 shape 阈值（元素）
constexpr int64_t SSA_SPARSITY_THRESHOLD = 1;       // 极稀疏阈值（每窗口有效元素数）
constexpr int64_t SSA_SPARSE_WINDOW_BYTES = 192;    // 稀疏判定窗口（字节）
constexpr int64_t SSA_MIN_CONTIG_RUN_BYTES = 32;    // T2 最小连续段（字节，另要求段首 32B 对齐）
constexpr int64_t SSA_T2_MIN_CHUNK_BYTES = 256;     // T2 chunksPerSeg 下限粒度（字节）：小 slice 收敛铺核
constexpr int64_t SSA_MIN_ELEM_PER_CORE = 1024;     // 核间切分每核最小元素数
constexpr int64_t SSA_ALIGN_32_BYTES = 32;          // 32B 对齐字节数（元素粒度 = 32/elemSize）
constexpr int64_t SSA_ALIGN_UNIT_ELEMS_STEP = 16; // T1 ubFactor 对齐粒度：16 元素整数倍（跨 dtype 保守上界）
constexpr int64_t SSA_PARAM_BUF_BYTES = 512;       // 常驻参数区字节数
constexpr int64_t SSA_BUFFER_NUM = 2;              // inputQueue 双缓冲槽位数（T1/T2）
constexpr int64_t SSA_NEG_SLOT_NUM = 4;            // 全负方向 UB 槽位总数（input 2 + reverse 2，T2 Key 21）
constexpr int64_t SSA_BUFFER_NUM_VAR = 4;          // varQue 槽位数（T3 Scatter）
constexpr int64_t SSA_SCATTER_BUFFER_NUM = 4;      // Scatter inputQueue 槽位数（T3）
constexpr int64_t SSA_MIN_PER_TASK_BYTES = 128;    // T3 均衡：单任务最小字节数（1 cacheline）
constexpr float SSA_MIN_USED_CORES_RATIO = 0.75F;  // T3 均衡：核数利用率下限
constexpr int64_t SSA_MAX_DMA_BLOCK_COUNT = 4095;  // k>=2 接口 blockCount 上限
constexpr int64_t SSA_MAX_DMA_LOOP_SIZE = 1 << 21; // Loop 模式单级循环次数上限（size < 2^21）
constexpr int32_t SSA_SIMT_THREAD_NUM = 1024;      // T1 SIMT 线程数
constexpr int64_t SSA_SIMT_DCACHE_BYTES = 32 * 1024;    // T1 SIMT DCache 预留（UB 尾部，须 ≥32KB）
constexpr int64_t SSA_VF_OVERREAD_BYTES = 256;          // Reg 寄存器整宽：UB 缓冲尾部越读预留
constexpr int64_t SSA_INT32_MAX = 0x7FFFFFFF;           // int32 上界
constexpr int64_t SSA_INT64_MAX = 0x7FFFFFFFFFFFFFFFLL; // int64 上界

// TilingParse 填充，全模板共用
struct StridedSliceAssignV2CompileInfo {
    int32_t totalCoreNum = 0;
    int64_t ubSize = 0;
    int64_t ubUsable = 0;
};

ge::graphStatus Tiling4StridedSliceAssignV2Arch35(gert::TilingContext* context);

ge::graphStatus TilingPrepare4StridedSliceAssignV2Arch35(gert::TilingParseContext* context);

} // namespace optiling

#endif // STRIDED_SLICE_ASSIGN_V2_TILING_ARCH35_H
