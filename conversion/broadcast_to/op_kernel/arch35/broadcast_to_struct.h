/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BROADCAST_TO_STRUCT_H_
#define BROADCAST_TO_STRUCT_H_

#include <cstddef>
#include <cstdint>

constexpr size_t BRC_TO_MAX_DMA_DIM_NUM = static_cast<size_t>(0x5);
constexpr size_t BRC_TO_MAX_A_DIM_NUM = static_cast<size_t>(0x8) * 3;
constexpr size_t BRC_TO_MAX_B_DIM_NUM = static_cast<size_t>(0x8) * 2;

// 布局约定: 8B字段区 -> 4B字段区 -> 2B/1B字段区, 按对齐分组排列消除padding
// 收窄依据(零风险): 仅收窄受硬件硬约束的字段, 其余保持int64
//   dFactor/usedCoreCnt: 受核数约束(<=AIV核数)
//   aLpUnit/uLpUnit: 受tensorSize_约束(<=ub/dtype)
//   isUNotB/isLastDimB/aAxesNum/bAxesNum: 布尔或小范围有符号计数
// ntcULen/tcULen保持int64: 非U轴主切分时直接取uAxisLen_(MergeAxis合并长度, 与tensorSize无关), 不可收窄
// 其余int64字段(ntcALen/tcALen/ntcBLen/tcBLen等)为后续收窄候选, 需host侧补numel<2^31校验
struct BroadcastToTilingData {
    // 8B区
    int64_t ntcALen;
    int64_t tcALen;
    int64_t ntcBLen;
    int64_t tcBLen;
    int64_t uInOffset;
    int64_t uOutOffset;
    int64_t ntcULen;
    int64_t tcULen;
    uint64_t xSrcStride[BRC_TO_MAX_DMA_DIM_NUM];
    int64_t aAxesParams[BRC_TO_MAX_A_DIM_NUM];
    int64_t bAxesParams[BRC_TO_MAX_B_DIM_NUM];
    // 4B区
    uint32_t tensorSize;
    uint32_t dFactor;
    uint32_t usedCoreCnt;
    uint32_t aLpUnit;
    uint32_t uLpUnit;
    uint32_t xDstStride[BRC_TO_MAX_DMA_DIM_NUM];
    uint32_t xSize[BRC_TO_MAX_DMA_DIM_NUM];
    // 1B区
    uint8_t doubleMode;
    uint8_t uAxisCnt;
    uint8_t bufferCnt;
    uint8_t blockAxis;
    uint8_t isUNotB;
    uint8_t isLastDimB;
    int8_t aAxesNum;
    int8_t bAxesNum;
};

struct TailAxisTilingData {
    // 8B区
    int64_t ntcALen;
    int64_t tcALen;
    int64_t ntcBLen;
    int64_t tcBLen;
    int64_t ntcULen;
    int64_t tcULen;
    int64_t aAxesParams[BRC_TO_MAX_A_DIM_NUM];
    int64_t bAxesParams[BRC_TO_MAX_B_DIM_NUM];
    // 4B区
    uint32_t tensorSize;
    uint32_t dFactor;
    uint32_t usedCoreCnt;
    uint32_t aLpUnit;
    uint32_t uLpUnit; // 供复用base CalcAxesLoopInfo: LAST_DIM_LARGE_B下=tensorSize
    // 1B区
    uint8_t doubleMode;
    uint8_t bufferCnt;
    uint8_t blockAxis;
    int8_t aAxesNum;
    int8_t bAxesNum;
    uint8_t pad0[3];
};

// UB广播kernel(TILING_MODE_UB_BRC/LAST_DIM_SMALL_A)专属tiling
// 裁剪kernel未使用字段, usedCoreCnt/dFactor收窄为uint16(<=AIV核数, 值无损), 其余长度保持int64
// 488B -> 424B
struct UbBrcTilingData {
    // 8B区
    int64_t ntcALen;
    int64_t tcALen;
    int64_t ntcBLen;
    int64_t tcBLen;
    int64_t uInOffset;
    int64_t uOutOffset;
    int64_t ntcULen;
    int64_t tcULen;
    int64_t aAxesParams[BRC_TO_MAX_A_DIM_NUM];
    int64_t bAxesParams[BRC_TO_MAX_B_DIM_NUM];
    // 4B区
    uint32_t tensorSize;
    uint32_t uLpUnit;
    uint32_t xSize[BRC_TO_MAX_DMA_DIM_NUM];
    // 2B区
    uint16_t usedCoreCnt;
    uint16_t dFactor;
    // 1B区
    uint8_t doubleMode;
    uint8_t uAxisCnt;
    uint8_t bufferCnt;
    uint8_t blockAxis;
    uint8_t isUNotB;
    uint8_t isLastDimB;
    int8_t aAxesNum;
    int8_t bAxesNum;
};

// DataCopyPad直通kernel(TILING_MODE_LAST_DIM_LARGE_A)专属tiling
// 裁剪kernel未使用字段; LARGE_A下uAxis恒为尾轴, uInOffset/uOutOffset/isUNotB恒为1已内化;
// usedCoreCnt/dFactor收窄为uint16(<=AIV核数, 值无损), 其余长度保持int64
// 488B -> 384B
struct CopyPadTilingData {
    // 8B区
    int64_t ntcALen;
    int64_t tcALen;
    int64_t ntcBLen;
    int64_t tcBLen;
    int64_t ntcULen;
    int64_t tcULen;
    int64_t aAxesParams[BRC_TO_MAX_A_DIM_NUM];
    int64_t bAxesParams[BRC_TO_MAX_B_DIM_NUM];
    // 4B区
    uint32_t tensorSize;
    uint32_t uLpUnit;
    // 2B区
    uint16_t usedCoreCnt;
    uint16_t dFactor;
    // 1B区
    uint8_t doubleMode;
    uint8_t blockAxis;
    int8_t aAxesNum;
    int8_t bAxesNum;
};

#endif // BROADCAST_TO_STRUCT_H_
