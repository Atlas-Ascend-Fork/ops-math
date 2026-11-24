/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file slice_struct.h
 * \brief slice struct
 */
#ifndef SLICE_TILINGDATA_H_
#define SLICE_TILINGDATA_H_

#include <cstdint>

constexpr int64_t SLICE_MAX_AXIS_NUM = 8;
constexpr int64_t SLICE_MAX_NDDMA_UB_SPLIT_AXIS_NUM = 5;
constexpr int64_t SLICE_MAX_SIMT_UB_SPLIT_AXIS_NUM = 8;
constexpr int64_t SLICE_NUMBER_TWO = 2;

struct SliceBaseTilingData {
    int8_t isBeginConst;
    int64_t blkFactor;
    int64_t blkTailFactor;
    int64_t ubInLoopSteps;
    int32_t ubSize;
    int32_t ubFactor;
    int32_t ubTailFactor;
    int32_t ubTailTailFactor;
    int16_t realCoreNum;
    int16_t inputDims;
    int16_t blkIndex;
    int16_t ubIndex;
    int64_t outputShape[SLICE_MAX_AXIS_NUM];
    int64_t begin[SLICE_MAX_AXIS_NUM];
    int64_t inputSteps[SLICE_MAX_AXIS_NUM];
    int64_t rowsOffsetSteps[SLICE_MAX_AXIS_NUM];
};

struct StridedSliceMoveAlignParams2 {
    uint16_t blockCount;
    uint32_t blockLen;
    uint32_t srcStride;
    uint32_t dstStride;
    uint16_t loop1Size;
    uint16_t loop2Size;
    uint32_t loop1SrcStride;
    uint16_t loop1DstStride;
    uint32_t loop2SrcStride;
    uint16_t loop2DstStride;
};

struct SliceMoveAlignLastDimTilingData {
    SliceBaseTilingData sliceBaseTilingData;
};

struct SliceMoveAlignParams {
    uint16_t blockCount;
    uint32_t blockLen;
    uint32_t srcStride;
    uint32_t dstStride;
};

struct SliceMoveAlignLast2DimTilingData {
    int64_t blkFactor;
    int64_t blkTailFactor;
    int64_t ubInLoopSteps;
    int64_t ubOutLoopSteps;
    int32_t ubSize;
    int32_t ubFactor;
    int32_t ubTailFactor;
    int32_t ubTailTailFactor;
    int16_t realCoreNum;
    int8_t isBeginConst;
    SliceMoveAlignParams moveAlignParams;
    int64_t outputShape[SLICE_NUMBER_TWO];
    int64_t begin[SLICE_NUMBER_TWO];
    int64_t inputSteps[SLICE_NUMBER_TWO];
};

struct SliceMoveAlignGatherTilingData {
    SliceBaseTilingData sliceBaseTilingData;

    int64_t ubOutLoopSteps;
    int32_t ubSizeInput;
    uint32_t lastOneInputDim;
    uint32_t outBlockLen;
    StridedSliceMoveAlignParams2 moveAlignParams;
};

struct StridedSliceTilingData2 {
    int64_t ubSize;
    int64_t ubSizeInput; // stride为负时使用
    int64_t coreNum;
    int64_t ubIndex;
    int64_t ubFactor;
    int64_t ubTailFactor;
    int64_t ubTailTailFactor;
    int64_t realCoreNum;
    int64_t inputDims;
    int64_t blkIndex;
    int64_t blkFactor;
    int64_t blkTailFactor;
    int64_t xDtypeSize;
    int64_t tilingKey;
    int64_t nddmaTotalNum;
    int64_t ubInLoopSteps;
    int64_t ubOutLoopSteps;
    uint32_t isShapeExceedUint32;
    uint32_t isEmptyTensor;
    int8_t isBeginConst;
    StridedSliceMoveAlignParams2 moveAlignParams;
    int64_t outputShape[SLICE_MAX_AXIS_NUM];
    int64_t begin[SLICE_MAX_AXIS_NUM];
    int64_t strides[SLICE_MAX_AXIS_NUM];
    int64_t rowsOffsetSteps[SLICE_MAX_AXIS_NUM];
    int64_t inputSteps[SLICE_MAX_AXIS_NUM];
    int64_t nddmaLoopSize[SLICE_MAX_NDDMA_UB_SPLIT_AXIS_NUM];
    int64_t nddmaLoopSrcStride[SLICE_MAX_NDDMA_UB_SPLIT_AXIS_NUM];
    int64_t nddmaLoopDstStride[SLICE_MAX_NDDMA_UB_SPLIT_AXIS_NUM];
    int64_t outputShapeProd[SLICE_MAX_SIMT_UB_SPLIT_AXIS_NUM];
    int64_t inputShapeProd[SLICE_MAX_SIMT_UB_SPLIT_AXIS_NUM];
};

struct SliceTilingData {
    StridedSliceTilingData2 stridedSliceTilingData;
};

struct SliceMoveAlignTilingData {
    SliceBaseTilingData sliceBaseTilingData;
    StridedSliceMoveAlignParams2 moveAlignParams;
    int64_t ubOutLoopSteps;
};

struct SliceNDDMATilingData {
    SliceBaseTilingData sliceBaseTilingData;

    int64_t ubOutLoopSteps;
    int64_t nddmaTotalNum;
    int64_t nddmaLoopSize[SLICE_MAX_NDDMA_UB_SPLIT_AXIS_NUM];
    int64_t nddmaLoopSrcStride[SLICE_MAX_NDDMA_UB_SPLIT_AXIS_NUM];
    int64_t nddmaLoopDstStride[SLICE_MAX_NDDMA_UB_SPLIT_AXIS_NUM];
};

struct SliceNDDMALastDimTilingData {
    SliceBaseTilingData sliceBaseTilingData;

    int64_t nddmaLoopSrcStride[SLICE_MAX_NDDMA_UB_SPLIT_AXIS_NUM];
    int64_t nddmaLoopDstStride[SLICE_MAX_NDDMA_UB_SPLIT_AXIS_NUM];
};

struct SliceTwoDimSmallSapeTilingData {
    int8_t isBeginConst;
    int16_t realCoreNum;
    int16_t mainCoreNum;
    uint32_t ubSize;
    uint32_t blockLen;
    uint32_t blkFactor;
    uint64_t lastOneInputDim;
    uint64_t lastOneOutputDim;
    uint64_t lastOneDimOffset;
};

// host侧按realCoreNum份写入tiling buffer，kernel侧每个核按核偏移dcpreload并读取自己核对应的副本。
// stride不再统一做64B对齐/guard，仅key102保留64B对齐；总dataSize需低于GE的32KB上限
constexpr uint32_t SLICE_TWO_DIM_SMALL_TILING_ALIGN = 64;
// 注册tiling buffer容量的编译期上界；运行期限核以平台实际AIV核数（compileInfo->block_dim）为准
constexpr uint32_t SLICE_TWO_DIM_SMALL_MAX_CORE_NUM = 72;

// 以下为按核多份布局常量：默认stride=sizeof，不做对齐/guard；key102单独按64B对齐
constexpr uint32_t SLICE_MOVE_ALIGN_TILING_STRIDE = sizeof(SliceMoveAlignTilingData);
constexpr uint32_t SLICE_MOVE_ALIGN_LAST_DIM_TILING_STRIDE = sizeof(SliceMoveAlignLastDimTilingData);
constexpr uint32_t SLICE_NDDMA_TILING_STRIDE = (sizeof(SliceNDDMATilingData) + SLICE_TWO_DIM_SMALL_TILING_ALIGN - 1) /
                                               SLICE_TWO_DIM_SMALL_TILING_ALIGN * SLICE_TWO_DIM_SMALL_TILING_ALIGN;
constexpr uint32_t SLICE_NDDMA_LAST_DIM_TILING_STRIDE = sizeof(SliceNDDMALastDimTilingData);
constexpr uint32_t SLICE_MOVE_ALIGN_TWO_DIM_TILING_STRIDE = sizeof(SliceMoveAlignLast2DimTilingData);
constexpr uint32_t SLICE_MOVE_ALIGN_GATHER_TILING_STRIDE = sizeof(SliceMoveAlignGatherTilingData);
constexpr uint32_t SLICE_TWO_DIM_SMALL_TILING_STRIDE = sizeof(SliceTwoDimSmallSapeTilingData);

// 注册的tiling buffer容量：最大核数 * 所有key中最大stride
constexpr uint32_t SliceMaxTilingStride()
{
    uint32_t maxStride = SLICE_MOVE_ALIGN_TILING_STRIDE;
    if (maxStride < SLICE_MOVE_ALIGN_LAST_DIM_TILING_STRIDE) {
        maxStride = SLICE_MOVE_ALIGN_LAST_DIM_TILING_STRIDE;
    }
    if (maxStride < SLICE_NDDMA_TILING_STRIDE) {
        maxStride = SLICE_NDDMA_TILING_STRIDE;
    }
    if (maxStride < SLICE_NDDMA_LAST_DIM_TILING_STRIDE) {
        maxStride = SLICE_NDDMA_LAST_DIM_TILING_STRIDE;
    }
    if (maxStride < SLICE_MOVE_ALIGN_TWO_DIM_TILING_STRIDE) {
        maxStride = SLICE_MOVE_ALIGN_TWO_DIM_TILING_STRIDE;
    }
    if (maxStride < SLICE_MOVE_ALIGN_GATHER_TILING_STRIDE) {
        maxStride = SLICE_MOVE_ALIGN_GATHER_TILING_STRIDE;
    }
    if (maxStride < SLICE_TWO_DIM_SMALL_TILING_STRIDE) {
        maxStride = SLICE_TWO_DIM_SMALL_TILING_STRIDE;
    }
    return maxStride;
}

constexpr uint32_t SLICE_TILING_MAX_STRIDE = SliceMaxTilingStride();
constexpr uint32_t SLICE_TILING_DATA_SIZE = SLICE_TWO_DIM_SMALL_MAX_CORE_NUM * SLICE_TILING_MAX_STRIDE;

struct SliceMoveAlignTilingDataBuffer {
    uint8_t tilingData[SLICE_TWO_DIM_SMALL_MAX_CORE_NUM * SLICE_MOVE_ALIGN_TILING_STRIDE];
};
struct SliceMoveAlignLastDimTilingDataBuffer {
    uint8_t tilingData[SLICE_TWO_DIM_SMALL_MAX_CORE_NUM * SLICE_MOVE_ALIGN_LAST_DIM_TILING_STRIDE];
};
struct SliceNDDMATilingDataBuffer {
    uint8_t tilingData[SLICE_TWO_DIM_SMALL_MAX_CORE_NUM * SLICE_NDDMA_TILING_STRIDE];
};
struct SliceNDDMALastDimTilingDataBuffer {
    uint8_t tilingData[SLICE_TWO_DIM_SMALL_MAX_CORE_NUM * SLICE_NDDMA_LAST_DIM_TILING_STRIDE];
};
struct SliceMoveAlignLast2DimTilingDataBuffer {
    uint8_t tilingData[SLICE_TWO_DIM_SMALL_MAX_CORE_NUM * SLICE_MOVE_ALIGN_TWO_DIM_TILING_STRIDE];
};
struct SliceMoveAlignGatherTilingDataBuffer {
    uint8_t tilingData[SLICE_TWO_DIM_SMALL_MAX_CORE_NUM * SLICE_MOVE_ALIGN_GATHER_TILING_STRIDE];
};
struct SliceTwoDimSmallSapeTilingDataBuffer {
    uint8_t tilingData[SLICE_TWO_DIM_SMALL_MAX_CORE_NUM * SLICE_TWO_DIM_SMALL_TILING_STRIDE];
};

#endif // SLICE_TILINGDATA__H_
