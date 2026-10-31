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
 * \file strided_slice_assign_v2_tiling_data.h
 * \brief strided_slice_assign_v2_tiling_data.h
 */

#ifndef STRIDED_SLICE_ASSIGN_V2_TILING_DATA_H_
#define STRIDED_SLICE_ASSIGN_V2_TILING_DATA_H_

#include <cstdint>

constexpr int32_t SSA_MAX_DIM_NUM = 8;

// ---- TilingKey 10 / 11 / 12：T1 StridedSliceAssignV2Hybrid ----
struct StridedSliceAssignV2HybridTilingData {
    int64_t dimNum = 0;                       // 约简维数 n'（1-8）
    int64_t sliceElemNum = 0;                 // = Π S'[dimIdx]
    int64_t blockDim = 0;                     // = usedCoreNum
    int64_t elemPerCore = 0;                  // = CeilDiv(sliceElemNum, blockDim)
    int64_t lastCoreElemNum = 0;              // 尾核元素数
    int64_t varBase = 0;                      // 切片首元素 var 线性偏移（标量）
    int64_t sliceShape[SSA_MAX_DIM_NUM] = {}; // S'[]，供 flat 反解
    int64_t axisStep[SSA_MAX_DIM_NUM] = {};   // 约简元素步长 T'[]（含符号）
    int32_t ubFactor = 0;                     // 核内 tile 元素数
};

// ---- TilingKey 20 / 21：T2 StridedSliceAssignV2Contig ----
struct StridedSliceAssignV2ContigTilingData {
    int32_t templateId = 0;                   // = 20
    int32_t directionId = 0;                  // 0=全正 1=全负
    int64_t dimNum = 0;                       // 约简维数 n'
    int64_t sliceElemNum = 0;                 // = Π S'[j]
    int64_t contigRun = 0;                    // 最内连续段长度 = S'[n'-1]（元素）
    int64_t segNum = 0;                       // = sliceElemNum / contigRun
    int64_t blockDim = 0;                     // 实际铺核数
    int64_t chunksPerSeg = 0;                 // 每段 chunk 数（≥1）
    int64_t chunkLen = 0;                     // 每 chunk 基准长度（元素）
    int64_t chunkRem = 0;                     // 前 chunkRem 个 chunk 长 chunkLen+1
    int64_t totalChunks = 0;                  // = segNum × chunksPerSeg
    int64_t chunksPerCore = 0;                // 每核 chunk 数
    int64_t lastCoreChunks = 0;               // 尾核 chunk 数
    int64_t varBase = 0;                      // 切片首元素 var 线性偏移（标量）
    int64_t sliceShape[SSA_MAX_DIM_NUM] = {}; // 约简形状 S'[]
    int64_t axisStep[SSA_MAX_DIM_NUM] = {};   // 约简元素步长 T'[]（含符号）
    int64_t segCum[SSA_MAX_DIM_NUM] = {};     // 段形状后缀积（基于 S'，最内维置 1）
    int32_t ubFactor = 0;                     // UB 缓冲元素容量（= chunk 长度上界，元素）
};

// ---- TilingKey 30 / 31：T3 StridedSliceAssignV2Scatter ----
struct StridedSliceAssignV2ScatterTilingData {
    uint8_t ubAxis = 0;                        // 切分轴 a ∈ [max(0,n-4), n-1]
    uint32_t ubFactor = 0;                     // 切分轴单任务最大元素数 f
    uint32_t ubPerCount = 0;                   // 每核任务单元数
    uint32_t ubTotalCount = 0;                 // 任务单元总数
    uint32_t tileElemNum = 0;                  // = f·innerElems(a)（单任务有效元素上限）
    int32_t idxDtypeSize = 0;                  // Scatter 索引字节数（2/4，按 dtype 配对）
    int64_t dimNum = 0;                        // n'
    int64_t innerElems = 0;                    // = Π_{d>a}S[d]
    int64_t rows = 0;                          // var 帧行数上限 = DMA blockCount
    int64_t cols = 0;                          // var 帧列数上限
    int64_t ubSpan = 0;                        // = (cols-1)·|axisStep[n-1]|+1
    int64_t pitchUb = 0;                       // UB 内行距（元素，32B 对齐）
    int64_t varBase = 0;                       // 切片首元素 var 线性偏移（标量）
    int64_t sliceShape[SSA_MAX_DIM_NUM] = {};  // 约简形状 S'[]
    int64_t axisStep[SSA_MAX_DIM_NUM] = {};    // 约简元素步长 T'[]（含符号）
    int64_t sliceCum[SSA_MAX_DIM_NUM] = {};    // 切片后缀积（仅填 d ∈ [0, n')，下标 n' 不下发）
    int64_t rowStepVar[SSA_MAX_DIM_NUM] = {};  // = |axisStep[d]|，d ∈ [a, n-2]
    int64_t blkStartVar[SSA_MAX_DIM_NUM] = {}; // 各维升地址序起点，d ∈ [a, n-1]
};

// ---- TilingKey 40：T4 StridedSliceAssignV2Empty ----
struct StridedSliceAssignV2EmptyTilingData {
    uint32_t blockNum = 0; // = 实际铺核数（恒为 1）
};

#endif // STRIDED_SLICE_ASSIGN_V2_TILING_DATA_H_
