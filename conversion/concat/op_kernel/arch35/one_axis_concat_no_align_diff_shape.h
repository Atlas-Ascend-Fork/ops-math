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
 * \file one_axis_concat_no_align_diff_shape.h
 * \brief one axis concat no align diff shape
 */

#ifndef ONE_AXIS_CONCAT_NO_ALIGN_DIFF_SHAPE
#define ONE_AXIS_CONCAT_NO_ALIGN_DIFF_SHAPE

#include "concat_base.h"
#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "op_kernel/math_util.h"

namespace Concat {
using namespace AscendC;
using namespace Ops::Base;
template <typename T, typename U, typename TILINGDATA = ConcatTilingData>
class OneAxisConcatNoAlignDiffShape {
public:
    __aicore__ inline OneAxisConcatNoAlignDiffShape(__tiling_data_ptr__ TILINGDATA* tilingData)
        : tilingData_(tilingData){};
    static __aicore__ inline void PreloadTiling(__tiling_data_ptr__ TILINGDATA* tilingData)
    {
        PreloadMergedInfo(tilingData);
    }
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR dst);
    __aicore__ inline void Process();

private:
    __aicore__ inline __gm__ T* GetTensorAddr(uint32_t index, int64_t offset);
    __aicore__ inline void CopyInNoSplitDim1(int64_t srcRowsOffset, int64_t rows, LocalTensor<T>& srcLocal,
                                             LocalTensor<T>& dstLocal);
    __aicore__ inline void CopyOut(LocalTensor<T> dstLocal, int64_t dstOffset, uint16_t rows, int64_t cols);
    __aicore__ inline void CopyOut(int64_t dstOffset, int64_t dataLen, LocalTensor<T>& dstLocal);
    __aicore__ inline void ProcessBlockSplitDim0NoSplitDim1();
    __aicore__ inline void ProcessBlockSplitDim0SplitDim1();
    __aicore__ inline void ProcessBlockSplitDim1();
    __aicore__ inline void ComputeSplitDim1(LocalTensor<T> dstLocal, LocalTensor<T> srcLocal, uint32_t rows,
                                            uint32_t cols, uint32_t dstOffset, uint32_t curLoopHandleCols);
    __aicore__ inline void GenScatterIndex(U curLoopHandleCols, U curTensorStartCols, U curLoopHandleCurTensorCols,
                                           LocalTensor<U>& indexLocal);
    __aicore__ inline void ScatterSplitDim1(LocalTensor<T> dstLocal, LocalTensor<T> srcLocal, LocalTensor<U> indexLocal,
                                            uint32_t curLoopHandleRows, uint32_t curLoopHandleCols,
                                            uint32_t curLoopHandleCurTensorCols);
    __aicore__ inline bool GatherSubBlockSeg(int64_t tensorIdx, int64_t dim0stride, int64_t seg, int64_t numSeg,
                                             int64_t startSeg, int64_t rowSrcStride, int64_t copyCols, int64_t rows,
                                             int64_t srcRowBase, int64_t dstColBase);
    __aicore__ inline void GenGatherSubBlockIndex(int64_t seg, int64_t dim0stride, LocalTensor<U>& indexLocal);
    // 计算当前 tensor 切片参数(dim1Size/dim0stride/copyCols/extraCols/isSplit)
    __aicore__ inline void CalcSliceParams(int64_t tensorIdx, int64_t colsOffset, int64_t totalCopyCols,
                                           int64_t endTensorIdx, int64_t endTensorOffset, int64_t& dim1Size,
                                           int64_t& dim0stride, int64_t& copyCols, int64_t& extraCols, bool& isSplit);
    // gather 逐段 2D DMA 兜底(块级, srcRowBase 统一行偏移)
    __aicore__ inline void GatherSegDma(int64_t tensorIdx, int64_t dim0stride, int64_t seg, int64_t numSeg,
                                        int64_t startSeg, int64_t rowSrcStride, int64_t copyCols, uint16_t rows,
                                        int64_t srcRowBase, int64_t outTensorColsOffset,
                                        DataCopyPadExtParams<T>& padParams);
    // 老路径攒批写入(srcLocal 偏移累攒 + ComputeSplitDim1 重排, dstOffset=totalCopyCols)
    __aicore__ inline void LegacyBatchWrite(int64_t tensorIdx, int64_t dim0stride, int64_t copyCols, uint16_t rows,
                                            int64_t colsOffset, int64_t srcRowBase, int64_t totalCopyCols,
                                            uint32_t curLoopHandleCols, LocalTensor<T>& srcLocal,
                                            LocalTensor<T>& dstLocal, int64_t tensorStride,
                                            DataCopyPadExtParams<T>& padParams);
    // gather 域整循环: 逐 tensor 切片即写(PBS1 块形)
    __aicore__ inline void ProcessGatherSlicesDim1(uint16_t rows);
    // 老路径整循环: 攒批 CopyOut(PBS1 块形)
    __aicore__ inline void ProcessLegacyBatchDim1(uint16_t rows, uint32_t curLoopHandleCols);
    // gather 域整循环: 逐列块整段搬运(PBS01 块形)
    __aicore__ inline void ProcessGatherSlicesDim0SplitDim1(int64_t loopSizeDim0);
    // 老路径整循环: 逐列块攒批 CopyOut(PBS01 块形)
    __aicore__ inline void ProcessLegacyBatchDim0SplitDim1(int64_t loopSizeDim0, uint32_t curLoopHandleCols);

private:
    constexpr static event_t EVT_ID0 = static_cast<event_t>(0);
    constexpr static event_t EVT_ID1 = static_cast<event_t>(1);

    GlobalTensor<T> dstGlobal_;
    __tiling_data_ptr__ TILINGDATA* tilingData_;
    int64_t blockOffset_ = 0;
    int64_t numPerBlock_ = BYTES_PER_BLOCK / sizeof(T);
    int64_t rowsUsedCoreNum_ = 1;
    int64_t colsUsedCoreNum_ = 1;
    int64_t startTensorIdx_ = 0;
    int64_t startTensorOffset_ = 0;
    int64_t endTensorIdx_ = 0;
    int64_t endTensorOffset_ = 0;
    TensorDesc<T> desc_;
    ListTensorDesc inputList_;
    uint64_t buf_[ARRAY_SIZE];
    static constexpr uint32_t SCATTER_MAX_LEN = GetVRegSize() * DIGIT_FOUR / sizeof(U);
    static constexpr uint32_t COLS_SIZE_UNROLL4 = GetVRegSize() * DIGIT_THREE / sizeof(U);
    static constexpr uint32_t COLS_SIZE_UNROLL3 = GetVRegSize() * DIGIT_TWO / sizeof(U);
    static constexpr uint32_t COLS_SIZE_UNROLL2 = GetVRegSize() / sizeof(U);
    static constexpr uint32_t COLS_SIZE_UNROLL1 = GetVRegSize() / DIGIT_TWO / sizeof(U);

    uint32_t bufSize_{0};
    uint32_t bufElems_{0};
    uint32_t inOffset0_{0};
    uint32_t inOffset1_{0};
    uint32_t outOffset0_{0};
    uint32_t outOffset1_{0};
    uint32_t idxOffset_{0};
    uint32_t idxElems_{0};
    event_t curEvtX_{static_cast<event_t>(0)};
    event_t curEvtY_{static_cast<event_t>(0)};
    // gather 逐段 2D DMA 兜底路径的双 buffer 轮转槽位
    uint16_t dmaSlot_{0};

    int16_t ubSplitDim1_{0};
    int16_t dim_{0};
    int16_t tensorNum_{0};
    int16_t isNonContiguous_{0};
    int16_t isFP4Type_{0};
    int16_t isGather_{0};
    int16_t isRowConcat_{0};
    int64_t gatherSeg_{0};
    int64_t rowConcatSegNum_{0};
    int32_t ubFactorDim0_{0};
    int32_t ubFactorDim1_{0};
    int32_t tailUbFactorDim0_{0};
    int32_t tailUbFactorDim1_{0};
    int32_t bufferSize_{0};
    int64_t blockFactor_{0};
    int64_t tailBlockFactor_{0};
    int64_t uoDim0_{0};
    int64_t catDim1_{0};
    int64_t sameShapeTensorDim1_{0};
};

template <typename T, typename U, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignDiffShape<T, U, TILINGDATA>::Init(GM_ADDR x, GM_ADDR dst)
{
    int64_t blockIdx = GetBlockIdx();
    if constexpr (IsSame<TILINGDATA, ConcatTilingDataNoArray>::value ||
                  IsSame<TILINGDATA, ConcatTilingDataNoArrayCompact>::value) {
        blockOffset_ = blockIdx * tilingData_->blockFactor * tilingData_->ubFactorDim0;
        dstGlobal_.SetGlobalBuffer((__gm__ T*)dst + blockOffset_ * tilingData_->catDim1);
    } else {
        rowsUsedCoreNum_ = tilingData_->uoDim0;
        colsUsedCoreNum_ = GetBlockNum() / rowsUsedCoreNum_;
        int64_t blockIdxInCols = blockIdx / colsUsedCoreNum_;
        int64_t blockIdxInRow = blockIdx - blockIdxInCols * colsUsedCoreNum_;
        if (blockIdxInRow != 0) {
            // 数组按核行重复展开，blockIdx-1 即同列上一核定界，与本核定界对相邻
            startTensorOffset_ = tilingData_->arrays.mergedInfo[CORE_INFO_BASE + 2 * (blockIdx - 1) + 1];
            startTensorIdx_ = tilingData_->arrays.mergedInfo[CORE_INFO_BASE + 2 * (blockIdx - 1)];
        }
        endTensorIdx_ = tilingData_->arrays.mergedInfo[CORE_INFO_BASE + 2 * blockIdx];
        endTensorOffset_ = tilingData_->arrays.mergedInfo[CORE_INFO_BASE + 2 * blockIdx + 1];
        blockOffset_ = blockIdxInCols * tilingData_->ubFactorDim0;
        int64_t colOffset = blockIdxInRow * tilingData_->blockFactor * tilingData_->ubFactorDim1;
        dstGlobal_.SetGlobalBuffer((__gm__ T*)dst + blockOffset_ * tilingData_->catDim1 + colOffset);
    }

    bufSize_ = static_cast<uint32_t>(tilingData_->bufferSize * sizeof(T));
    bufElems_ = static_cast<uint32_t>(tilingData_->bufferSize);
    idxOffset_ = 0;
    idxElems_ = INDEX_SIZE / sizeof(U);
    inOffset0_ = static_cast<uint32_t>((INDEX_SIZE + 31) / 32 * 32);
    inOffset1_ = inOffset0_ + bufSize_;
    outOffset0_ = inOffset1_ + bufSize_;
    outOffset1_ = outOffset0_ + bufSize_;

    ubSplitDim1_ = tilingData_->ubSplitDim1;
    dim_ = tilingData_->dim;
    tensorNum_ = tilingData_->tensorNum;
    isNonContiguous_ = tilingData_->isNonContiguous;
    isFP4Type_ = tilingData_->isFP4Type;
    isGather_ = tilingData_->isGather;
    isRowConcat_ = tilingData_->isRowConcat;
    gatherSeg_ = tilingData_->gatherSeg;
    rowConcatSegNum_ = tilingData_->rowConcatSegNum;
    ubFactorDim0_ = tilingData_->ubFactorDim0;
    ubFactorDim1_ = tilingData_->ubFactorDim1;
    tailUbFactorDim0_ = tilingData_->tailUbFactorDim0;
    tailUbFactorDim1_ = tilingData_->tailUbFactorDim1;
    bufferSize_ = tilingData_->bufferSize;
    blockFactor_ = tilingData_->blockFactor;
    tailBlockFactor_ = tilingData_->tailBlockFactor;
    uoDim0_ = tilingData_->uoDim0;
    catDim1_ = tilingData_->catDim1;
    sameShapeTensorDim1_ = tilingData_->sameShapeTensorDim1;

    inputList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(x));
    desc_.SetShapeAddr(&buf_[0]);
    int64_t startTensorDim1_ = GetNonConDimSize<TILINGDATA, T>(tilingData_, startTensorIdx_, inputList_, desc_) *
                               tilingData_->sameShapeTensorDim1;
    if (tilingData_->isFP4Type) {
        startTensorDim1_ /= 2;
    }
    if (startTensorOffset_ == startTensorDim1_) {
        startTensorOffset_ = 0;
        startTensorIdx_ += 1;
    }
}

template <typename T, typename U, typename TILINGDATA>
__aicore__ inline __gm__ T* OneAxisConcatNoAlignDiffShape<T, U, TILINGDATA>::GetTensorAddr(uint32_t index,
                                                                                           int64_t offset)
{
    return inputList_.GetDataPtr<T>(index) + offset;
}

template <typename T, typename U, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignDiffShape<T, U, TILINGDATA>::Process()
{
    if (GetBlockIdx() >= GetBlockNum()) {
        return;
    }
    // gather/rowConcat 的中转 work 槽位（复用 in 双 buffer）依赖 MTE3_MTE2 事件保护复用：
    // 预挂两槽标志使首次 MTE2 写入前不阻塞。普通/Scatter 路径不消费该事件, 须按路径
    // 条件预挂, 否则形成未配对 SetFlag(计数残留会在 super-kernel 拼接场景被后续算子
    // 的 WaitFlag 误消费, 造成假放行)
    if (isGather_ || isRowConcat_) {
        SetFlag<HardEvent::MTE3_MTE2>(EVT_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(EVT_ID1);
    }
    if constexpr (IsSame<TILINGDATA, ConcatTilingData>::value || IsSame<TILINGDATA, ConcatTilingDataCompact>::value) {
        ProcessBlockSplitDim1();
    } else {
        if (ubSplitDim1_ == 1) {
            ProcessBlockSplitDim0SplitDim1();
        } else {
            ProcessBlockSplitDim0NoSplitDim1();
        }
    }
    // 收尾成对清零: 逐片路径每 chunk Wait/Set 净零, 末态每槽各残留 1 次归还的许可
    if (isGather_ || isRowConcat_) {
        WaitFlag<HardEvent::MTE3_MTE2>(EVT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVT_ID1);
    }
}

template <typename T, typename U, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignDiffShape<T, U, TILINGDATA>::ProcessBlockSplitDim1()
{
    uint16_t rows = static_cast<uint16_t>(ubFactorDim0_);
    if (GetBlockIdx() / colsUsedCoreNum_ == rowsUsedCoreNum_ - 1) {
        rows = static_cast<uint16_t>(tailUbFactorDim0_);
    }
    uint32_t curLoopHandleCols = static_cast<uint32_t>(ubFactorDim1_);
    // 域一刀切: gather 走逐片即写, 老路径走基线攒批(攒满一批 CopyOut 一次)
    if (isGather_) {
        ProcessGatherSlicesDim1(rows);
    } else {
        ProcessLegacyBatchDim1(rows, curLoopHandleCols);
    }
}

template <typename T, typename U, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignDiffShape<T, U, TILINGDATA>::ProcessBlockSplitDim0SplitDim1()
{
    int64_t loopSizeDim0 = blockFactor_;
    if (GetBlockIdx() == GetBlockNum() - 1) {
        loopSizeDim0 = tailBlockFactor_;
    }
    uint32_t curLoopHandleCols = static_cast<uint32_t>(ubFactorDim1_);
    // 域一刀切(与 ProcessBlockSplitDim1 同): gather 逐片, 老路径攒批
    if (isGather_) {
        ProcessGatherSlicesDim0SplitDim1(loopSizeDim0);
    } else {
        ProcessLegacyBatchDim0SplitDim1(loopSizeDim0, curLoopHandleCols);
    }
}

// gather 域逐片搬运: PBS1 块形(按行分块列直切), 整段快路径优先、段级 2D DMA 兜底、逐片即写
template <typename T, typename U, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignDiffShape<T, U, TILINGDATA>::ProcessGatherSlicesDim1(uint16_t rows)
{
    DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
    int64_t totalCopyCols = 0;
    int64_t colsOffset = startTensorOffset_;
    int64_t tensorIdx = startTensorIdx_;
    int64_t outTensorColsOffset = 0;
    while (tensorIdx <= endTensorIdx_) {
        int64_t dim1Size;
        int64_t dim0stride;
        int64_t copyCols;
        int64_t extraCols;
        bool isSplit;
        CalcSliceParams(tensorIdx, colsOffset, totalCopyCols, endTensorIdx_, endTensorOffset_, dim1Size, dim0stride,
                        copyCols, extraCols, isSplit);
        if (extraCols <= 0 || isSplit) {
            if (isSplit) {
                copyCols = ubFactorDim1_ - totalCopyCols;
            }
            int64_t seg = gatherSeg_;
            int64_t startSeg = colsOffset / seg;
            int64_t rowSrcStride = dim0stride * (dim1Size / seg);
            // 整段优先：一次搬完该 tensor 在本 block 的全部剩余列，内部按跨度分 chunk 并
            // 预取流水（MTE2 背靠背），消除按 ubFactorDim1 切片带来的逐 call 串行等待；
            // 不满足快路径条件时回退既有逐片逻辑，行为不变
            int64_t fullPortion = (tensorIdx == endTensorIdx_) ? (endTensorOffset_ - colsOffset) :
                                                                 (dim1Size - colsOffset);
            if (GatherSubBlockSeg(tensorIdx, dim0stride, seg, fullPortion / seg, startSeg, rowSrcStride, fullPortion,
                                  rows, blockOffset_, outTensorColsOffset)) {
                outTensorColsOffset += fullPortion;
                totalCopyCols = 0;
                colsOffset = 0;
                tensorIdx++;
                continue;
            }
            int64_t numSeg = copyCols / seg;
            // sub-32B 段优先走整跨度连读+向量紧缩快路径（含按列 chunk），大段回退逐段 2D 拷贝
            if (!GatherSubBlockSeg(tensorIdx, dim0stride, seg, numSeg, startSeg, rowSrcStride, copyCols, rows,
                                   blockOffset_, outTensorColsOffset)) {
                GatherSegDma(tensorIdx, dim0stride, seg, numSeg, startSeg, rowSrcStride, copyCols, rows, blockOffset_,
                             outTensorColsOffset, padParams);
            }
            if (isSplit) {
                colsOffset += copyCols;
            } else {
                colsOffset = 0;
                tensorIdx++;
            }
            outTensorColsOffset += copyCols;
            totalCopyCols += copyCols;
            if (totalCopyCols >= ubFactorDim1_) {
                totalCopyCols = 0;
            }
        } else {
            // 小溢出兜底(还原基线 flush 第三条款): 本 tensor 装入会溢出不足一个 32B 块,
            // 不值得切片; 已即写的数据无需回写, 仅重置预算计数让本 tensor 按新预算重试。
            // 重试后要么装得下(extraCols<=0), 要么触发 isSplit 的 totalCopyCols==0 条款,
            // 游标必然前进——缺失此分支会在该窗口下 while 死循环(AIC 100% 挂死)
            totalCopyCols = 0;
        }
    }
}

// 老路径攒批: PBS1 块形, 多 tensor 依 tensorStride 攒入 UB, 攒满一批 CopyOut 一次(与基线结构一致)
template <typename T, typename U, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignDiffShape<T, U, TILINGDATA>::ProcessLegacyBatchDim1(
    uint16_t rows, uint32_t curLoopHandleCols)
{
    DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
    int64_t totalCopyCols = 0;
    int64_t colsOffset = startTensorOffset_;
    int64_t tensorIdx = startTensorIdx_;
    int64_t outTensorColsOffset = 0;
    int64_t tensorStride = 0;

    SetFlag<HardEvent::V_MTE2>(EVT_ID0);
    SetFlag<HardEvent::V_MTE2>(EVT_ID1);
    SetFlag<HardEvent::MTE3_V>(EVT_ID0);
    SetFlag<HardEvent::MTE3_V>(EVT_ID1);
    uint16_t bufIdX = 0;
    uint16_t bufIdY = 0;
    curEvtX_ = EVT_ID0;
    curEvtY_ = EVT_ID0;
    WaitFlag<HardEvent::V_MTE2>(curEvtX_);
    WaitFlag<HardEvent::MTE3_V>(curEvtY_);
    LocalTensor<T> srcLocal(TPosition::VECIN, inOffset0_, bufElems_);
    LocalTensor<T> dstLocal(TPosition::VECOUT, outOffset0_, bufElems_);

    while (tensorIdx <= endTensorIdx_) {
        int64_t dim1Size;
        int64_t dim0stride;
        int64_t copyCols;
        int64_t extraCols;
        bool isSplit;
        CalcSliceParams(tensorIdx, colsOffset, totalCopyCols, endTensorIdx_, endTensorOffset_, dim1Size, dim0stride,
                        copyCols, extraCols, isSplit);
        if (extraCols <= 0 || isSplit) {
            if (isSplit) {
                copyCols = ubFactorDim1_ - totalCopyCols;
            }
            LegacyBatchWrite(tensorIdx, dim0stride, copyCols, rows, colsOffset, blockOffset_, totalCopyCols,
                             curLoopHandleCols, srcLocal, dstLocal, tensorStride, padParams);
            if (isSplit) {
                colsOffset += copyCols;
            } else {
                colsOffset = 0;
                tensorIdx++;
            }
            totalCopyCols += copyCols;
            tensorStride += CeilAlign(rows * copyCols, numPerBlock_);
        }
        if (tensorIdx > endTensorIdx_ || totalCopyCols == ubFactorDim1_ ||
            (extraCols > 0 && extraCols < numPerBlock_)) {
            SetFlag<HardEvent::V_MTE2>(curEvtX_);
            bufIdX ^= 1;
            curEvtX_ = (bufIdX == 0) ? EVT_ID0 : EVT_ID1;
            WaitFlag<HardEvent::V_MTE2>(curEvtX_);
            srcLocal = LocalTensor<T>(TPosition::VECIN, (bufIdX == 0) ? inOffset0_ : inOffset1_, bufElems_);

            CopyOut(dstLocal, outTensorColsOffset, rows, totalCopyCols);

            SetFlag<HardEvent::MTE3_V>(curEvtY_);
            bufIdY ^= 1;
            curEvtY_ = (bufIdY == 0) ? EVT_ID0 : EVT_ID1;
            WaitFlag<HardEvent::MTE3_V>(curEvtY_);
            dstLocal = LocalTensor<T>(TPosition::VECOUT, (bufIdY == 0) ? outOffset0_ : outOffset1_, bufElems_);

            outTensorColsOffset += totalCopyCols;
            totalCopyCols = 0;
            tensorStride = 0;
        }
    }
    SetFlag<HardEvent::V_MTE2>(curEvtX_);
    SetFlag<HardEvent::MTE3_V>(curEvtY_);
    WaitFlag<HardEvent::V_MTE2>(EVT_ID0);
    WaitFlag<HardEvent::V_MTE2>(EVT_ID1);
    WaitFlag<HardEvent::MTE3_V>(EVT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVT_ID1);
}

// gather 域逐片搬运: PBS01 块形(行列双切), 逐列块(i)内整段搬运, 处理完整 tensor 区间无截断
template <typename T, typename U, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignDiffShape<T, U, TILINGDATA>::ProcessGatherSlicesDim0SplitDim1(
    int64_t loopSizeDim0)
{
    DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
    for (int64_t i = 0; i < loopSizeDim0; i++) {
        int64_t totalCopyCols = 0;
        int64_t colsOffset = 0;
        int64_t tensorIdx = 0;
        int64_t loopOffsetInCols = i * ubFactorDim0_;
        int64_t outTensorColsOffset = loopOffsetInCols * catDim1_;
        uint16_t rows = static_cast<uint16_t>(ubFactorDim0_);
        if (GetBlockIdx() == GetBlockNum() - 1 && i == loopSizeDim0 - 1) {
            rows = static_cast<uint16_t>(tailUbFactorDim0_);
        }
        while (tensorIdx < tensorNum_) {
            int64_t dim1Size;
            int64_t dim0stride;
            int64_t copyCols;
            int64_t extraCols;
            bool isSplit;
            // PBS01 处理完整 tensor 区间(0..tensorNum), 无 endTensorIdx 截断, 传 -1 使截断条款恒不触发;
            // 误传 endTensorIdx_(NoArray/Compact 下恒 0) 会使首 tensor copyCols 被截为 0, 输出缺数据
            CalcSliceParams(tensorIdx, colsOffset, totalCopyCols, -1, 0, dim1Size, dim0stride, copyCols, extraCols,
                            isSplit);
            if (extraCols <= 0 || isSplit) {
                if (isSplit) {
                    copyCols = ubFactorDim1_ - totalCopyCols;
                }
                int64_t seg = gatherSeg_;
                int64_t startSeg = colsOffset / seg;
                int64_t numSeg = copyCols / seg;
                int64_t rowSrcStride = dim0stride * (dim1Size / seg);
                if (!GatherSubBlockSeg(tensorIdx, dim0stride, seg, numSeg, startSeg, rowSrcStride, copyCols, rows,
                                       blockOffset_ + loopOffsetInCols, outTensorColsOffset)) {
                    GatherSegDma(tensorIdx, dim0stride, seg, numSeg, startSeg, rowSrcStride, copyCols, rows,
                                 blockOffset_ + loopOffsetInCols, outTensorColsOffset, padParams);
                }
                if (isSplit) {
                    colsOffset += copyCols;
                } else {
                    colsOffset = 0;
                    tensorIdx++;
                }
                outTensorColsOffset += copyCols;
                totalCopyCols += copyCols;
                if (totalCopyCols >= ubFactorDim1_) {
                    totalCopyCols = 0;
                }
            } else {
                // 小溢出兜底(还原基线 flush 第三条款, 与 ProcessBlockSplitDim1 同):
                // 溢出不足一个 32B 块时仅重置预算计数重试, 防止 while 死循环
                totalCopyCols = 0;
            }
        }
    }
}

// 老路径攒批: PBS01 块形(行列双切), 逐列块(i)内攒批 CopyOut(与基线结构一致)
template <typename T, typename U, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignDiffShape<T, U, TILINGDATA>::ProcessLegacyBatchDim0SplitDim1(
    int64_t loopSizeDim0, uint32_t curLoopHandleCols)
{
    DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};

    SetFlag<HardEvent::V_MTE2>(EVT_ID0);
    SetFlag<HardEvent::V_MTE2>(EVT_ID1);
    SetFlag<HardEvent::MTE3_V>(EVT_ID0);
    SetFlag<HardEvent::MTE3_V>(EVT_ID1);
    uint16_t bufIdX = 0;
    uint16_t bufIdY = 0;
    for (int64_t i = 0; i < loopSizeDim0; i++) {
        int64_t totalCopyCols = 0;
        int64_t colsOffset = 0;
        int64_t tensorIdx = 0;
        int64_t tensorStride = 0;
        int64_t loopOffsetInCols = i * ubFactorDim0_;
        int64_t outTensorColsOffset = loopOffsetInCols * catDim1_;
        uint16_t rows = static_cast<uint16_t>(ubFactorDim0_);
        if (GetBlockIdx() == GetBlockNum() - 1 && i == loopSizeDim0 - 1) {
            rows = static_cast<uint16_t>(tailUbFactorDim0_);
        }
        curEvtX_ = (bufIdX == 0) ? EVT_ID0 : EVT_ID1;
        curEvtY_ = (bufIdY == 0) ? EVT_ID0 : EVT_ID1;
        WaitFlag<HardEvent::V_MTE2>(curEvtX_);
        WaitFlag<HardEvent::MTE3_V>(curEvtY_);
        LocalTensor<T> srcLocal(TPosition::VECIN, (bufIdX == 0) ? inOffset0_ : inOffset1_, bufElems_);
        LocalTensor<T> dstLocal(TPosition::VECOUT, (bufIdY == 0) ? outOffset0_ : outOffset1_, bufElems_);
        while (tensorIdx < tensorNum_) {
            int64_t dim1Size;
            int64_t dim0stride;
            int64_t copyCols;
            int64_t extraCols;
            bool isSplit;
            // PBS01 处理完整 tensor 区间(0..tensorNum), 无 endTensorIdx 截断, 传 -1 使截断条款恒不触发;
            // 误传 endTensorIdx_(NoArray/Compact 下恒 0) 会使首 tensor copyCols 被截为 0, 输出缺数据
            CalcSliceParams(tensorIdx, colsOffset, totalCopyCols, -1, 0, dim1Size, dim0stride, copyCols, extraCols,
                            isSplit);
            if (extraCols <= 0 || isSplit) {
                if (isSplit) {
                    copyCols = ubFactorDim1_ - totalCopyCols;
                }
                LegacyBatchWrite(tensorIdx, dim0stride, copyCols, rows, colsOffset, blockOffset_ + loopOffsetInCols,
                                 totalCopyCols, curLoopHandleCols, srcLocal, dstLocal, tensorStride, padParams);
                if (isSplit) {
                    colsOffset += copyCols;
                } else {
                    colsOffset = 0;
                    tensorIdx++;
                }
                totalCopyCols += copyCols;
                tensorStride += CeilAlign(rows * copyCols, numPerBlock_);
            }
            if (tensorIdx >= tensorNum_ || totalCopyCols == ubFactorDim1_ ||
                (extraCols > 0 && extraCols < numPerBlock_)) {
                SetFlag<HardEvent::V_MTE2>(curEvtX_);
                bufIdX ^= 1;
                curEvtX_ = (bufIdX == 0) ? EVT_ID0 : EVT_ID1;
                WaitFlag<HardEvent::V_MTE2>(curEvtX_);
                srcLocal = LocalTensor<T>(TPosition::VECIN, (bufIdX == 0) ? inOffset0_ : inOffset1_, bufElems_);

                CopyOut(dstLocal, outTensorColsOffset, rows, totalCopyCols);

                SetFlag<HardEvent::MTE3_V>(curEvtY_);
                bufIdY ^= 1;
                curEvtY_ = (bufIdY == 0) ? EVT_ID0 : EVT_ID1;
                WaitFlag<HardEvent::MTE3_V>(curEvtY_);
                dstLocal = LocalTensor<T>(TPosition::VECOUT, (bufIdY == 0) ? outOffset0_ : outOffset1_, bufElems_);

                outTensorColsOffset += totalCopyCols;
                totalCopyCols = 0;
                tensorStride = 0;
            }
        }
        SetFlag<HardEvent::V_MTE2>(curEvtX_);
        SetFlag<HardEvent::MTE3_V>(curEvtY_);
        bufIdX ^= 1;
        bufIdY ^= 1;
    }
    WaitFlag<HardEvent::V_MTE2>(EVT_ID0);
    WaitFlag<HardEvent::V_MTE2>(EVT_ID1);
    WaitFlag<HardEvent::MTE3_V>(EVT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVT_ID1);
}

template <typename T, typename U, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignDiffShape<T, U, TILINGDATA>::ProcessBlockSplitDim0NoSplitDim1()
{
    int64_t loopSize = blockFactor_;
    if (GetBlockIdx() == GetBlockNum() - 1) {
        loopSize = tailBlockFactor_ - 1;
    }

    SetFlag<HardEvent::V_MTE2>(EVT_ID0);
    SetFlag<HardEvent::V_MTE2>(EVT_ID1);
    SetFlag<HardEvent::MTE3_V>(EVT_ID0);
    SetFlag<HardEvent::MTE3_V>(EVT_ID1);
    uint16_t bufIdX = 0;
    uint16_t bufIdY = 0;
    for (int64_t i = 0; i < loopSize; i++) {
        curEvtX_ = (bufIdX == 0) ? EVT_ID0 : EVT_ID1;
        curEvtY_ = (bufIdY == 0) ? EVT_ID0 : EVT_ID1;
        WaitFlag<HardEvent::V_MTE2>(curEvtX_);
        WaitFlag<HardEvent::MTE3_V>(curEvtY_);
        LocalTensor<T> srcLocal(TPosition::VECIN, (bufIdX == 0) ? inOffset0_ : inOffset1_, bufElems_);
        LocalTensor<T> dstLocal(TPosition::VECOUT, (bufIdY == 0) ? outOffset0_ : outOffset1_, bufElems_);
        CopyInNoSplitDim1(i * ubFactorDim0_, ubFactorDim0_, srcLocal, dstLocal);
        SetFlag<HardEvent::V_MTE2>(curEvtX_);
        // gather/rowConcat 域在 CopyInNoSplitDim1 内部逐段即写完成输出, 无需整块 CopyOut
        if (!isRowConcat_ && !isGather_) {
            CopyOut(i * ubFactorDim0_ * catDim1_, ubFactorDim0_ * catDim1_, dstLocal);
        }
        SetFlag<HardEvent::MTE3_V>(curEvtY_);
        bufIdX ^= 1;
        bufIdY ^= 1;
    }
    if (GetBlockIdx() == GetBlockNum() - 1) {
        curEvtX_ = (bufIdX == 0) ? EVT_ID0 : EVT_ID1;
        curEvtY_ = (bufIdY == 0) ? EVT_ID0 : EVT_ID1;
        WaitFlag<HardEvent::V_MTE2>(curEvtX_);
        WaitFlag<HardEvent::MTE3_V>(curEvtY_);
        LocalTensor<T> srcLocal(TPosition::VECIN, (bufIdX == 0) ? inOffset0_ : inOffset1_, bufElems_);
        LocalTensor<T> dstLocal(TPosition::VECOUT, (bufIdY == 0) ? outOffset0_ : outOffset1_, bufElems_);
        CopyInNoSplitDim1(loopSize * ubFactorDim0_, tailUbFactorDim0_, srcLocal, dstLocal);
        SetFlag<HardEvent::V_MTE2>(curEvtX_);
        if (!isRowConcat_ && !isGather_) {
            CopyOut(loopSize * ubFactorDim0_ * catDim1_, tailUbFactorDim0_ * catDim1_, dstLocal);
        }
        SetFlag<HardEvent::MTE3_V>(curEvtY_);
        bufIdX ^= 1;
        bufIdY ^= 1;
    }
    WaitFlag<HardEvent::V_MTE2>(EVT_ID0);
    WaitFlag<HardEvent::V_MTE2>(EVT_ID1);
    WaitFlag<HardEvent::MTE3_V>(EVT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVT_ID1);
}

template <typename T, typename U, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignDiffShape<T, U, TILINGDATA>::ScatterSplitDim1(
    LocalTensor<T> dstLocal, LocalTensor<T> srcLocal, LocalTensor<U> indexLocal, uint32_t curLoopHandleRows,
    uint32_t curLoopHandleCols, uint32_t curLoopHandleCurTensorCols)
{
    constexpr uint32_t vfLen = GetVRegSize() / sizeof(U);
    uint16_t regFactorDim0 = vfLen / curLoopHandleCurTensorCols;
    uint16_t size0 = curLoopHandleRows / regFactorDim0;
    uint16_t tailRegFactorDim0 = curLoopHandleRows - size0 * regFactorDim0;
    auto indexAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    auto dstAddr = (__ubuf__ T*)dstLocal.GetPhyAddr();
    auto srcAddr = (__ubuf__ T*)srcLocal.GetPhyAddr();

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<T> vd2;
        AscendC::Reg::RegTensor<T> src;
        AscendC::Reg::RegTensor<T> tmp;
        AscendC::Reg::RegTensor<T> dst0;
        AscendC::Reg::RegTensor<T> dst1;
        AscendC::Reg::RegTensor<U> vd0;
        AscendC::Reg::RegTensor<U> vd1;
        AscendC::Reg::UnalignRegForLoad u0;

        uint32_t num = (uint32_t)(regFactorDim0 * curLoopHandleCurTensorCols);
        uint32_t tailNum = (uint32_t)(tailRegFactorDim0 * curLoopHandleCurTensorCols);
        uint32_t pnum = num;
        uint32_t tailPnum = tailNum;
        AscendC::Reg::MaskReg p0 = AscendC::Reg::UpdateMask<U>(num);
        AscendC::Reg::MaskReg p1 = AscendC::Reg::UpdateMask<U>(tailNum);

        AscendC::Reg::LoadAlign(vd0, indexAddr);
        AscendC::Reg::LoadUnAlignPre(u0, srcAddr);
        for (uint16_t i = 0; i < size0; i++) {
            AscendC::Reg::LoadUnAlign(vd2, u0, srcAddr, pnum);
            AscendC::Reg::Adds(vd1, vd0, (U)(i * curLoopHandleCols * regFactorDim0), p0);
            if constexpr (sizeof(T) == 1) {
                AscendC::Reg::Interleave(dst0, dst1, vd2, tmp);
                AscendC::Reg::Scatter(dstAddr, dst0, vd1, p0);
            } else {
                AscendC::Reg::Scatter(dstAddr, vd2, vd1, p0);
            }
        }
        AscendC::Reg::LoadUnAlign(vd2, u0, srcAddr, tailPnum);
        AscendC::Reg::Adds(vd1, vd0, (U)(size0 * curLoopHandleCols * regFactorDim0), p1);
        if constexpr (sizeof(T) == 1) {
            AscendC::Reg::Interleave(dst0, dst1, vd2, tmp);
            AscendC::Reg::Scatter(dstAddr, dst0, vd1, p1);
        } else {
            AscendC::Reg::Scatter(dstAddr, vd2, vd1, p1);
        }
    }
}

template <typename T, typename U, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignDiffShape<T, U, TILINGDATA>::ComputeSplitDim1(LocalTensor<T> dstLocal,
                                                                                         LocalTensor<T> srcLocal,
                                                                                         uint32_t rows, uint32_t cols,
                                                                                         uint32_t dstOffset,
                                                                                         uint32_t curLoopHandleCols)
{
    LocalTensor<U> indexLocal(TPosition::VECCALC, idxOffset_, idxElems_);
    SetFlag<HardEvent::MTE2_V>(curEvtX_);
    WaitFlag<HardEvent::MTE2_V>(curEvtX_);
    if (cols > SCATTER_MAX_LEN) {
        Copy(dstLocal[dstOffset], srcLocal, rows, cols, curLoopHandleCols);
    } else if (cols > COLS_SIZE_UNROLL4) {
        ScatterConcat<T, U, DIGIT_FOUR>(dstLocal[dstOffset], srcLocal, rows, cols, curLoopHandleCols);
    } else if (cols > COLS_SIZE_UNROLL3) {
        ScatterConcat<T, U, DIGIT_THREE>(dstLocal[dstOffset], srcLocal, rows, cols, curLoopHandleCols);
    } else if (cols > COLS_SIZE_UNROLL2) {
        ScatterConcat<T, U, DIGIT_TWO>(dstLocal[dstOffset], srcLocal, rows, cols, curLoopHandleCols);
    } else if (cols >= COLS_SIZE_UNROLL1 || rows == 1) {
        ScatterConcat<T, U, 1>(dstLocal[dstOffset], srcLocal, rows, cols, curLoopHandleCols);
    } else {
        GenScatterIndex(curLoopHandleCols, dstOffset, cols, indexLocal);
        ScatterSplitDim1(dstLocal, srcLocal, indexLocal, rows, curLoopHandleCols, cols);
    }
}

template <typename T, typename U, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignDiffShape<T, U, TILINGDATA>::CopyInNoSplitDim1(int64_t srcRowsOffset,
                                                                                          int64_t rows,
                                                                                          LocalTensor<T>& srcLocal,
                                                                                          LocalTensor<T>& dstLocal)
{
    GlobalTensor<T> srcGlobal;
    DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
    if (isGather_) {
        uint32_t curDim1Offset = 0;
        uint16_t wSlot = 0;
        for (int64_t i = 0; i < tensorNum_; i++) {
            int64_t dim1 = GetNonConDimSize<TILINGDATA, T>(tilingData_, i, inputList_, desc_) * sameShapeTensorDim1_;
            if (isFP4Type_) {
                dim1 /= 2;
            }
            int64_t seg = gatherSeg_;
            int64_t segNum = dim1 / seg;
            int64_t dim0stride = GetTensorDim0Stride<TILINGDATA>(tilingData_, i, dim1);
            int64_t rowSrcStride = dim0stride * segNum;
            if (!GatherSubBlockSeg(i, dim0stride, seg, segNum, 0, rowSrcStride, dim1, rows,
                                   blockOffset_ + srcRowsOffset, srcRowsOffset * catDim1_ + curDim1Offset)) {
                uint32_t rowStride = static_cast<uint32_t>((dim1 + numPerBlock_ - 1) / numPerBlock_ * numPerBlock_);
                event_t evt = wSlot == 0 ? EVT_ID0 : EVT_ID1;
                WaitFlag<HardEvent::MTE3_MTE2>(evt);
                LocalTensor<T> workLocal(TPosition::VECIN, wSlot == 0 ? inOffset0_ : inOffset1_, bufElems_);
                DataCopyExtParams copyInParam = {static_cast<uint16_t>(segNum), static_cast<uint32_t>(seg * sizeof(T)),
                                                 static_cast<int64_t>((dim0stride - seg) * sizeof(T)), 0, 0};
                srcGlobal.SetGlobalBuffer(GetTensorAddr(i, blockOffset_ * rowSrcStride));
                for (int64_t r = 0; r < rows; r++) {
                    DataCopyPad<T, PaddingMode::Compact>(workLocal[r * rowStride],
                                                         srcGlobal[(srcRowsOffset + r) * rowSrcStride], copyInParam,
                                                         padParams);
                }
                SetFlag<HardEvent::MTE2_MTE3>(evt);
                WaitFlag<HardEvent::MTE2_MTE3>(evt);
                DataCopyExtParams copyOutParam = {static_cast<uint16_t>(rows), static_cast<uint32_t>(dim1 * sizeof(T)),
                                                  0, static_cast<int64_t>((catDim1_ - dim1) * sizeof(T)), 0};
                DataCopyPad(dstGlobal_[srcRowsOffset * catDim1_ + curDim1Offset], workLocal, copyOutParam);
                SetFlag<HardEvent::MTE3_MTE2>(evt);
                wSlot ^= 1;
            }
            curDim1Offset += static_cast<uint32_t>(dim1);
        }
        return;
    }
    if (isRowConcat_) {
        int64_t segNum = rowConcatSegNum_;
        int64_t segOut = catDim1_ / segNum;
        uint16_t wSlot = 0;
        for (int64_t i = 0; i < tensorNum_; i++) {
            int64_t dim1 = GetNonConDimSize<TILINGDATA, T>(tilingData_, i, inputList_, desc_) * sameShapeTensorDim1_;
            if (isFP4Type_) {
                dim1 /= 2;
            }
            int64_t seg = dim1 / segNum;
            int64_t dim0stride = GetTensorDim0Stride<TILINGDATA>(tilingData_, i, dim1);
            // 断点轴以下整行完全连续（tiling FindBreakAxis 保证唯一断点），行内各段均匀分布，
            // 段间距恒为 seg = dim1/segNum；不能用断点轴下一轴的物理 stride
            // （rowConcatSegStrideList = stride(断点轴+1) = seg * ∏dims[断点轴+2..dim)，
            // 仅当断点轴与 concat 轴之间只有 1 个中间轴时才与 seg 相等，多中间轴场景会读错源地址）
            int64_t tensorOffsetInSeg = 0;
            for (int64_t j = 0; j < i; j++) {
                int64_t prevDim1 = GetNonConDimSize<TILINGDATA, T>(tilingData_, j, inputList_, desc_) *
                                   sameShapeTensorDim1_;
                if (isFP4Type_) {
                    prevDim1 /= 2;
                }
                tensorOffsetInSeg += prevDim1 / segNum;
            }
            // 统一 2D 搬运（对齐/非对齐段通用）：
            // 源侧：断点轴以下整行连续，段以 seg 为间距紧密排列，单条 2D Normal 读即可覆盖
            //       （dstStride=0 时 Normal 模式 UB 块间距自动 = AlignUp(seg*sizeof(T),32) = alignedSeg，
            //       非对齐段落入 32B padded 槽位；MTE3 源侧 Normal 模式同样按 32B 对齐推进，恰好读回 padded 槽位）
            // 目的侧：段以 segOut 间距交错落位，单条 2D 写（dstStride=(segOut-seg)*sizeof(T)）。
            // 每段一个队列周期的老 fallback 仅对 seg 超 UB 容量的巨段保留。
            int64_t alignedSeg = CeilAlign(seg, numPerBlock_);
            if (alignedSeg <= bufferSize_) {
                int64_t segsPerGroup = bufferSize_ / alignedSeg;
                if (segsPerGroup > UINT16_MAX) {
                    segsPerGroup = UINT16_MAX; // blockCount 为 uint16
                }
                for (int64_t r = 0; r < rows; r++) {
                    for (int64_t k0 = 0; k0 < segNum; k0 += segsPerGroup) {
                        int64_t nSeg = segNum - k0 < segsPerGroup ? segNum - k0 : segsPerGroup;
                        event_t evt = wSlot == 0 ? EVT_ID0 : EVT_ID1;
                        WaitFlag<HardEvent::MTE3_MTE2>(evt);
                        LocalTensor<T> workLocal(TPosition::VECIN, wSlot == 0 ? inOffset0_ : inOffset1_, bufElems_);
                        srcGlobal.SetGlobalBuffer(
                            GetTensorAddr(i, blockOffset_ * dim0stride + (srcRowsOffset + r) * dim0stride + k0 * seg));
                        DataCopyExtParams copyInParam = {static_cast<uint16_t>(nSeg),
                                                         static_cast<uint32_t>(seg * sizeof(T)), 0, 0, 0};
                        DataCopyPad<T>(workLocal, srcGlobal, copyInParam, padParams);
                        SetFlag<HardEvent::MTE2_MTE3>(evt);
                        WaitFlag<HardEvent::MTE2_MTE3>(evt);
                        DataCopyExtParams copyOutParam = {static_cast<uint16_t>(nSeg),
                                                          static_cast<uint32_t>(seg * sizeof(T)), 0,
                                                          static_cast<int64_t>((segOut - seg) * sizeof(T)), 0};
                        DataCopyPad(dstGlobal_[(srcRowsOffset + r) * catDim1_ + k0 * segOut + tensorOffsetInSeg],
                                    workLocal, copyOutParam);
                        SetFlag<HardEvent::MTE3_MTE2>(evt);
                        wSlot ^= 1;
                    }
                }
                continue;
            }
            for (int64_t k = 0; k < segNum; k++) {
                int64_t startCol = 0;
                while (startCol < seg) {
                    int64_t copyCols = (seg - startCol) < bufferSize_ ? (seg - startCol) : bufferSize_;
                    for (int64_t r = 0; r < rows; r++) {
                        event_t evt = wSlot == 0 ? EVT_ID0 : EVT_ID1;
                        WaitFlag<HardEvent::MTE3_MTE2>(evt);
                        LocalTensor<T> segLocal(TPosition::VECIN, wSlot == 0 ? inOffset0_ : inOffset1_, bufElems_);
                        srcGlobal.SetGlobalBuffer(GetTensorAddr(i, blockOffset_ * dim0stride + k * seg + startCol));
                        DataCopyExtParams copyInParam = {1, static_cast<uint32_t>(copyCols * sizeof(T)), 0, 0, 0};
                        DataCopyPad<T, PaddingMode::Compact>(segLocal, srcGlobal[(srcRowsOffset + r) * dim0stride],
                                                             copyInParam, padParams);
                        SetFlag<HardEvent::MTE2_MTE3>(evt);
                        WaitFlag<HardEvent::MTE2_MTE3>(evt);
                        DataCopyExtParams copyOutParam = {1, static_cast<uint32_t>(copyCols * sizeof(T)), 0, 0, 0};
                        DataCopyPad(
                            dstGlobal_[(srcRowsOffset + r) * catDim1_ + k * segOut + tensorOffsetInSeg + startCol],
                            segLocal, copyOutParam);
                        SetFlag<HardEvent::MTE3_MTE2>(evt);
                        wSlot ^= 1;
                    }
                    startCol += copyCols;
                }
            }
        }
        return;
    }
    uint32_t curDim1Offset = 0;
    int64_t tensorStride = 0;
    uint32_t curLoopHandleCols = static_cast<uint32_t>(catDim1_);
    for (int64_t i = 0; i < tensorNum_; i++) {
        int64_t dim1 = GetNonConDimSize<TILINGDATA, T>(tilingData_, i, inputList_, desc_) * sameShapeTensorDim1_;
        if (isFP4Type_) {
            dim1 /= 2;
        }
        int64_t dim0stride = GetTensorDim0Stride<TILINGDATA>(tilingData_, i, dim1);
        srcGlobal.SetGlobalBuffer(GetTensorAddr(i, blockOffset_ * dim0stride));
        DataCopyExtParams copyInParam = {static_cast<uint16_t>(rows), static_cast<uint32_t>(dim1 * sizeof(T)),
                                         static_cast<int64_t>((dim0stride - dim1) * sizeof(T)), 0, 0};
#if defined(__DAV_920R1__) || defined(__DAV_920R2__)

        DataCopyPad<T, PaddingMode::Compact>(srcLocal[tensorStride], srcGlobal[srcRowsOffset * dim0stride], copyInParam,
                                             padParams, true);
#else
        DataCopyPad<T, PaddingMode::Compact>(srcLocal[tensorStride], srcGlobal[srcRowsOffset * dim0stride], copyInParam,
                                             padParams);
#endif
        ComputeSplitDim1(dstLocal, srcLocal[tensorStride], rows, dim1, curDim1Offset, curLoopHandleCols);
        curDim1Offset += dim1;
        tensorStride += (rows * dim1 + numPerBlock_ - 1) / numPerBlock_ * numPerBlock_;
    }
}

template <typename T, typename U, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignDiffShape<T, U, TILINGDATA>::GenScatterIndex(U curLoopHandleCols,
                                                                                        U curTensorStartCols,
                                                                                        U curLoopHandleCurTensorCols,
                                                                                        LocalTensor<U>& indexLocal)
{
    constexpr uint32_t vfLen = GetVRegSize() / sizeof(U);
    auto dstAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<U> v1;
        AscendC::Reg::RegTensor<U> vd0;
        AscendC::Reg::RegTensor<U> vd2;
        AscendC::Reg::RegTensor<U> vd3;
        AscendC::Reg::RegTensor<U> vd6;
        AscendC::Reg::RegTensor<U> vd7;
        AscendC::Reg::RegTensor<U> vd10;
        uint32_t num = vfLen;
        AscendC::Reg::MaskReg p0 = AscendC::Reg::UpdateMask<U>(num);
        AscendC::Reg::RegTensor<U> v0;
        using regType = typename VciTypeGet<U>::T;
        AscendC::Reg::RegTensor<regType> tmp;
        AscendC::Reg::Arange(tmp, 0);
        v0 = (AscendC::Reg::RegTensor<U>&)tmp;
        AscendC::Reg::Duplicate(v1, curLoopHandleCurTensorCols, p0);
        AscendC::Reg::Div(vd2, v0, v1, p0);
        AscendC::Reg::Muls(vd6, vd2, curLoopHandleCurTensorCols, p0);
        AscendC::Reg::Sub(vd7, v0, vd6, p0);
        AscendC::Reg::Muls(vd0, vd2, curLoopHandleCols, p0);
        AscendC::Reg::Add(vd3, vd0, vd7, p0);
        AscendC::Reg::Adds(vd10, vd3, curTensorStartCols, p0);
        AscendC::Reg::StoreAlign(dstAddr, vd10, p0);
    }
}

template <typename T, typename U, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignDiffShape<T, U, TILINGDATA>::CopyOut(LocalTensor<T> dstLocal,
                                                                                int64_t dstOffset, uint16_t rows,
                                                                                int64_t cols)
{
    // VEC(ComputeSplitDim1 写 dstLocal) -> MTE3(DataCopyPad 读)同步: 基线原有此屏障,
    // PR 改动中被误删导致 19 例精度失败(B11/B17/B18), 现按基线原文还原
    SetFlag<HardEvent::V_MTE3>(curEvtY_);
    WaitFlag<HardEvent::V_MTE3>(curEvtY_);
    DataCopyExtParams copyOutParam = {rows, static_cast<uint32_t>(cols * sizeof(T)),
                                      (ubFactorDim1_ - cols) / numPerBlock_,
                                      static_cast<int64_t>((catDim1_ - cols) * sizeof(T)), 0};
    DataCopyPad(dstGlobal_[dstOffset], dstLocal, copyOutParam);
}

template <typename T, typename U, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignDiffShape<T, U, TILINGDATA>::CopyOut(int64_t dstOffset, int64_t dataLen,
                                                                                LocalTensor<T>& dstLocal)
{
    SetFlag<HardEvent::V_MTE3>(curEvtY_);
    WaitFlag<HardEvent::V_MTE3>(curEvtY_);
    DataCopyExtParams copyOutParam = {0, 0, 0, 0, 0};
    copyOutParam.blockCount = 1;
    copyOutParam.blockLen = dataLen * sizeof(T);
    copyOutParam.dstStride = 0;
    copyOutParam.srcStride = 0;
    DataCopyPad(dstGlobal_[dstOffset], dstLocal, copyOutParam);
}

template <typename T, typename U, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignDiffShape<T, U, TILINGDATA>::GenGatherSubBlockIndex(
    int64_t seg, int64_t dim0stride, LocalTensor<U>& indexLocal)
{
    constexpr uint32_t vfLen = GetVRegSize() / sizeof(U);
    auto indexAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    __VEC_SCOPE__
    {
        using regType = typename VciTypeGet<U>::T;
        AscendC::Reg::RegTensor<regType> tmp;
        AscendC::Reg::RegTensor<U> v0;
        AscendC::Reg::RegTensor<U> v1;
        AscendC::Reg::RegTensor<U> v2;
        AscendC::Reg::RegTensor<U> vd0;
        AscendC::Reg::RegTensor<U> vd1;
        AscendC::Reg::RegTensor<U> vd2;
        AscendC::Reg::RegTensor<U> vd3;
        uint32_t num = vfLen;
        AscendC::Reg::MaskReg p0 = AscendC::Reg::UpdateMask<U>(num);
        AscendC::Reg::Arange(tmp, 0);
        v0 = (AscendC::Reg::RegTensor<U>&)tmp;
        AscendC::Reg::Duplicate(v1, (U)seg, p0);
        AscendC::Reg::Duplicate(v2, (U)dim0stride, p0);
        AscendC::Reg::Div(vd0, v0, v1, p0);
        AscendC::Reg::Mul(vd1, vd0, v1, p0);
        AscendC::Reg::Sub(vd2, v0, vd1, p0);
        AscendC::Reg::Mul(vd1, vd0, v2, p0);
        AscendC::Reg::Add(vd3, vd1, vd2, p0);
        AscendC::Reg::StoreAlign(indexAddr, vd3, p0);
    }
}

template <typename T, typename U, typename TILINGDATA>
__aicore__ inline bool OneAxisConcatNoAlignDiffShape<T, U, TILINGDATA>::GatherSubBlockSeg(
    int64_t tensorIdx, int64_t dim0stride, int64_t seg, int64_t numSeg, int64_t startSeg, int64_t rowSrcStride,
    int64_t copyCols, int64_t rows, int64_t srcRowBase, int64_t dstColBase)
{
    // sub-32B 段（seg*sizeof(T) < 32B）场景：逐段 2D MTE1 会退化成海量 16B 小 burst，
    // 搬运效率极低。改为整行 padded 连续读（Normal 模式，UB 内按 32B 对齐分行）+
    // 向量 DataCopyGather 原地紧缩（idx(p) = (p/seg)*dim0stride + p%seg >= p，前向原位覆写安全）+
    // 2D 连续写出。布局与搬运模式参考 one_axis_concat_no_align_same_shape_gather.h。
    // B8 gather 产物为 b16 需 Pack、B64 不被 gather 支持（编译期即不成立），回退老路径。
    // 整段跨度超出 bufferSize 时（dim=0 且断点轴>dim 的 catDim0=1 大列宽场景，逐段小 burst
    // 同样低效），按列切 chunk：每 chunk 跨度可入 UB，chunk 内仍是整跨度连读+向量紧缩，
    // 索引模式跨 chunk 不变（仅源基址平移）；大段（seg>=32B）逐段搬运本身高效，保持回退不放大读。
    if constexpr (sizeof(T) != 2 && sizeof(T) != 4) {
        return false;
    } else {
        if (numSeg < 2 || copyCols != numSeg * seg || rows < 1) {
            return false;
        }
        // 新路径固定开销（索引生成+MTE2_V/V_MTE3 两次事件同步+队列往返）约 0.5us，
        // 小数据量时得不偿失，回退逐段老路径。
        // sub-32B 段的实际 MTE2 搬运量是跨度（放大 dim0stride/seg 倍），按跨度判定，
        // 避免大跨度小列宽场景（dim=0 且断点轴>dim 的 [X,b,1]）被误杀回退逐段 4B 小 burst
        if (seg < numPerBlock_) {
            if (rows * ((numSeg - 1) * dim0stride + seg) < 4096) {
                return false;
            }
        } else if (rows * copyCols < 4096) {
            return false;
        }
        constexpr uint32_t vfLen = GetVRegSize() / sizeof(U);
        if (seg > vfLen) {
            return false;
        }
        // 列分块：整段跨度装不下 UB 时，按 bufferSize 反解每 chunk 段数
        // （预留 32B 对齐余量，保证 CeilAlign 后跨度仍 <= bufferSize）
        int64_t segsPerChunk = numSeg;
        int64_t fullSpanElems = (numSeg - 1) * dim0stride + seg;
        bool spanOverflow = CeilAlign(fullSpanElems, numPerBlock_) > bufferSize_;
        if (spanOverflow) {
            if (seg >= numPerBlock_) {
                return false;
            }
            int64_t usableBuf = bufferSize_ / numPerBlock_ * numPerBlock_;
            segsPerChunk = (usableBuf - seg) / dim0stride + 1;
            if (segsPerChunk < 2) {
                return false;
            }
        }
        // 索引推进步长需同时为 seg 的倍数(保证 Adds 基址向量可整体平移)与 32B 对齐(保证 StoreAlign 落址对齐)
        int64_t gcdVal = seg;
        int64_t modVal = numPerBlock_;
        while (modVal != 0) {
            int64_t rem = gcdVal % modVal;
            gcdVal = modVal;
            modVal = rem;
        }
        int64_t lcmVal = seg / gcdVal * numPerBlock_;
        int64_t regChunk = (vfLen / lcmVal) * lcmVal;
        if (regChunk < seg) {
            return false;
        }
        DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
        GlobalTensor<T> srcGlobal;
        LocalTensor<U> indexLocal(TPosition::VECCALC, idxOffset_, idxElems_);
        GenGatherSubBlockIndex(seg, dim0stride, indexLocal);
        auto indexAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
        if (spanOverflow) {
            // 列分块 + 预取流水：处理 chunk k 前先下发 chunk k+1 的 MTE2 整跨度读，
            // MTE2 引擎背靠背传输，与 VEC(k)/MTE3(k) 并行，消除逐 chunk 同步造成的 MTE2 空转
            int64_t c0 = 0;
            int64_t r0 = 0;
            int64_t firstSegs = segsPerChunk < numSeg ? segsPerChunk : numSeg;
            int64_t firstSpan = (firstSegs - 1) * dim0stride + seg;
            int64_t firstRowsPerBatch = bufferSize_ / CeilAlign(firstSpan, numPerBlock_);
            int64_t firstBatchRows = firstRowsPerBatch < rows ? firstRowsPerBatch : rows;
            uint16_t curSlot = 0;
            WaitFlag<HardEvent::MTE3_MTE2>(EVT_ID0);
            LocalTensor<T> workLocal(TPosition::VECIN, inOffset0_, bufElems_);
            DataCopyExtParams firstInParam = {static_cast<uint16_t>(firstBatchRows),
                                              static_cast<uint32_t>(firstSpan * sizeof(T)),
                                              static_cast<int64_t>((rowSrcStride - firstSpan) * sizeof(T)), 0, 0};
            srcGlobal.SetGlobalBuffer(GetTensorAddr(tensorIdx, srcRowBase * rowSrcStride + startSeg * dim0stride));
            DataCopyPad<T>(workLocal, srcGlobal, firstInParam, padParams);
            while (true) {
                int64_t chunkSegs = segsPerChunk < numSeg - c0 ? segsPerChunk : numSeg - c0;
                int64_t chunkCols = chunkSegs * seg;
                int64_t spanElems = (chunkSegs - 1) * dim0stride + seg;
                int64_t spanAligned = CeilAlign(spanElems, numPerBlock_);
                int64_t rowsPerBatch = bufferSize_ / spanAligned;
                if (rowsPerBatch < 1) {
                    // 当前槽位已有本 chunk 的 MTE2 写在途且无 MTE3 读，补挂 MTE3_MTE2 标志后回退老路径
                    SetFlag<HardEvent::MTE3_MTE2>(curSlot == 0 ? EVT_ID0 : EVT_ID1);
                    return false;
                }
                int64_t batchRows = rowsPerBatch < rows - r0 ? rowsPerBatch : rows - r0;
                int64_t size1 = chunkCols / regChunk;
                int64_t tailChunk = chunkCols - size1 * regChunk;
                // 尾块归一化：整除时把最后一个满块并入尾块，使尾块必走且 mask=全量，
                // 消除 __VEC_SCOPE__ 行循环内的运行时分支（PTO TExtract 同款尾处理 idiom）
                if (tailChunk == 0 && size1 > 0) {
                    size1 -= 1;
                    tailChunk = regChunk;
                }
                // SetFlag 先行入 MTE2 流（位于 CopyPad(k) 之后），随后才下发 CopyPad(k+1)，
                // 保证 WaitFlag 只同步当前 chunk，MTE2 引擎背靠背不空转
                SetFlag<HardEvent::MTE2_V>(EVT_ID0);
                int64_t nextC0 = c0;
                int64_t nextR0 = r0 + rowsPerBatch;
                if (nextR0 >= rows) {
                    nextR0 = 0;
                    nextC0 = c0 + segsPerChunk;
                }
                uint16_t nextSlot = curSlot ^ 1;
                if (nextC0 < numSeg) {
                    int64_t nextSegs = segsPerChunk < numSeg - nextC0 ? segsPerChunk : numSeg - nextC0;
                    int64_t nextSpan = (nextSegs - 1) * dim0stride + seg;
                    int64_t nextRowsPerBatch = bufferSize_ / CeilAlign(nextSpan, numPerBlock_);
                    int64_t nextBatchRows = nextRowsPerBatch < rows - nextR0 ? nextRowsPerBatch : rows - nextR0;
                    WaitFlag<HardEvent::MTE3_MTE2>(nextSlot == 0 ? EVT_ID0 : EVT_ID1);
                    LocalTensor<T> nextLocal(TPosition::VECIN, nextSlot == 0 ? inOffset0_ : inOffset1_, bufElems_);
                    DataCopyExtParams nextInParam = {static_cast<uint16_t>(nextBatchRows),
                                                     static_cast<uint32_t>(nextSpan * sizeof(T)),
                                                     static_cast<int64_t>((rowSrcStride - nextSpan) * sizeof(T)), 0, 0};
                    srcGlobal.SetGlobalBuffer(GetTensorAddr(
                        tensorIdx, (srcRowBase + nextR0) * rowSrcStride + (startSeg + nextC0) * dim0stride));
                    DataCopyPad<T>(nextLocal, srcGlobal, nextInParam, padParams);
                }
                WaitFlag<HardEvent::MTE2_V>(EVT_ID0);
                auto srcAddr = (__ubuf__ T*)workLocal.GetPhyAddr();
                __VEC_SCOPE__
                {
                    AscendC::Reg::RegTensor<U> vd0;
                    AscendC::Reg::RegTensor<U> vd1;
                    AscendC::Reg::RegTensor<T> vd2;
                    uint32_t chunkMask = static_cast<uint32_t>(regChunk);
                    uint32_t tailMask = static_cast<uint32_t>(tailChunk);
                    AscendC::Reg::MaskReg p0 = AscendC::Reg::UpdateMask<U>(chunkMask);
                    AscendC::Reg::MaskReg p1 = AscendC::Reg::UpdateMask<U>(tailMask);
                    AscendC::Reg::LoadAlign(vd0, indexAddr);
                    uint16_t vecBatchRows = static_cast<uint16_t>(batchRows);
                    uint16_t vecSize1 = static_cast<uint16_t>(size1);
                    for (uint16_t r = 0; r < vecBatchRows; r++) {
                        auto curDstAddr = srcAddr + r * spanAligned;
                        for (uint16_t c = 0; c < vecSize1; c++) {
                            AscendC::Reg::Adds(vd1, vd0, (U)(r * spanAligned + c * (regChunk / seg) * dim0stride), p0);
                            AscendC::Reg::DataCopyGather(vd2, srcAddr, vd1, p0);
                            AscendC::Reg::StoreAlign(curDstAddr + c * regChunk, vd2, p0);
                        }
                        // 尾块已归一化（tailChunk ∈ [1, regChunk]，整除时并入最后一个满块），无条件执行
                        AscendC::Reg::Adds(vd1, vd0, (U)(r * spanAligned + size1 * (regChunk / seg) * dim0stride), p1);
                        AscendC::Reg::DataCopyGather(vd2, srcAddr, vd1, p1);
                        AscendC::Reg::StoreAlign(curDstAddr + size1 * regChunk, vd2, p1);
                    }
                }
                SetFlag<HardEvent::V_MTE3>(EVT_ID0);
                WaitFlag<HardEvent::V_MTE3>(EVT_ID0);
                DataCopyExtParams copyOutParam = {static_cast<uint16_t>(batchRows),
                                                  static_cast<uint32_t>(chunkCols * sizeof(T)),
                                                  (spanAligned - CeilAlign(chunkCols, numPerBlock_)) / numPerBlock_,
                                                  static_cast<int64_t>((catDim1_ - chunkCols) * sizeof(T)), 0};
                DataCopyPad(dstGlobal_[dstColBase + c0 * seg + r0 * catDim1_], workLocal, copyOutParam);
                SetFlag<HardEvent::MTE3_MTE2>(curSlot == 0 ? EVT_ID0 : EVT_ID1);
                if (nextC0 >= numSeg) {
                    break;
                }
                c0 = nextC0;
                r0 = nextR0;
                curSlot = nextSlot;
                workLocal = LocalTensor<T>(TPosition::VECIN, curSlot == 0 ? inOffset0_ : inOffset1_, bufElems_);
            }
            return true;
        }
        uint16_t curSlot = 0;
        for (int64_t c0 = 0; c0 < numSeg; c0 += segsPerChunk) {
            int64_t chunkSegs = segsPerChunk < numSeg - c0 ? segsPerChunk : numSeg - c0;
            int64_t chunkCols = chunkSegs * seg;
            int64_t spanElems = (chunkSegs - 1) * dim0stride + seg;
            int64_t spanAligned = CeilAlign(spanElems, numPerBlock_);
            int64_t rowsPerBatch = bufferSize_ / spanAligned;
            if (rowsPerBatch < 1) {
                return false;
            }
            int64_t chunkStartSeg = startSeg + c0;
            int64_t size1 = chunkCols / regChunk;
            int64_t tailChunk = chunkCols - size1 * regChunk;
            // 尾块归一化：整除时把最后一个满块并入尾块，使尾块必走且 mask=全量，
            // 消除 __VEC_SCOPE__ 行循环内的运行时分支（PTO TExtract 同款尾处理 idiom）
            if (tailChunk == 0 && size1 > 0) {
                size1 -= 1;
                tailChunk = regChunk;
            }
            for (int64_t r0 = 0; r0 < rows; r0 += rowsPerBatch) {
                int64_t batchRows = rowsPerBatch < rows - r0 ? rowsPerBatch : rows - r0;
                WaitFlag<HardEvent::MTE3_MTE2>(curSlot == 0 ? EVT_ID0 : EVT_ID1);
                LocalTensor<T> workLocal(TPosition::VECIN, curSlot == 0 ? inOffset0_ : inOffset1_, bufElems_);
                // Normal 模式 dstStride=0：UB 行距 = align32(spanElems*sizeof(T)) = spanAligned*sizeof(T)
                DataCopyExtParams copyInParam = {static_cast<uint16_t>(batchRows),
                                                 static_cast<uint32_t>(spanElems * sizeof(T)),
                                                 static_cast<int64_t>((rowSrcStride - spanElems) * sizeof(T)), 0, 0};
                srcGlobal.SetGlobalBuffer(
                    GetTensorAddr(tensorIdx, (srcRowBase + r0) * rowSrcStride + chunkStartSeg * dim0stride));
                DataCopyPad<T>(workLocal, srcGlobal, copyInParam, padParams);
                SetFlag<HardEvent::MTE2_V>(EVT_ID0);
                WaitFlag<HardEvent::MTE2_V>(EVT_ID0);
                auto srcAddr = (__ubuf__ T*)workLocal.GetPhyAddr();
                __VEC_SCOPE__
                {
                    AscendC::Reg::RegTensor<U> vd0;
                    AscendC::Reg::RegTensor<U> vd1;
                    AscendC::Reg::RegTensor<T> vd2;
                    uint32_t chunkMask = static_cast<uint32_t>(regChunk);
                    uint32_t tailMask = static_cast<uint32_t>(tailChunk);
                    AscendC::Reg::MaskReg p0 = AscendC::Reg::UpdateMask<U>(chunkMask);
                    AscendC::Reg::MaskReg p1 = AscendC::Reg::UpdateMask<U>(tailMask);
                    AscendC::Reg::LoadAlign(vd0, indexAddr);
                    uint16_t vecBatchRows = static_cast<uint16_t>(batchRows);
                    uint16_t vecSize1 = static_cast<uint16_t>(size1);
                    for (uint16_t r = 0; r < vecBatchRows; r++) {
                        auto curDstAddr = srcAddr + r * spanAligned;
                        for (uint16_t c = 0; c < vecSize1; c++) {
                            AscendC::Reg::Adds(vd1, vd0, (U)(r * spanAligned + c * (regChunk / seg) * dim0stride), p0);
                            AscendC::Reg::DataCopyGather(vd2, srcAddr, vd1, p0);
                            AscendC::Reg::StoreAlign(curDstAddr + c * regChunk, vd2, p0);
                        }
                        // 尾块已归一化（tailChunk ∈ [1, regChunk]，整除时并入最后一个满块），无条件执行
                        AscendC::Reg::Adds(vd1, vd0, (U)(r * spanAligned + size1 * (regChunk / seg) * dim0stride), p1);
                        AscendC::Reg::DataCopyGather(vd2, srcAddr, vd1, p1);
                        AscendC::Reg::StoreAlign(curDstAddr + size1 * regChunk, vd2, p1);
                    }
                }
                SetFlag<HardEvent::V_MTE3>(EVT_ID0);
                WaitFlag<HardEvent::V_MTE3>(EVT_ID0);
                DataCopyExtParams copyOutParam = {static_cast<uint16_t>(batchRows),
                                                  static_cast<uint32_t>(chunkCols * sizeof(T)),
                                                  (spanAligned - CeilAlign(chunkCols, numPerBlock_)) / numPerBlock_,
                                                  static_cast<int64_t>((catDim1_ - chunkCols) * sizeof(T)), 0};
                DataCopyPad(dstGlobal_[dstColBase + c0 * seg + r0 * catDim1_], workLocal, copyOutParam);
                SetFlag<HardEvent::MTE3_MTE2>(curSlot == 0 ? EVT_ID0 : EVT_ID1);
                curSlot ^= 1;
            }
        }
        return true;
    }
    return false;
}

// 计算当前 tensor 切片参数: 公共子过程, 被两个 ProcessBlock 函数复用
template <typename T, typename U, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignDiffShape<T, U, TILINGDATA>::CalcSliceParams(
    int64_t tensorIdx, int64_t colsOffset, int64_t totalCopyCols, int64_t endTensorIdx, int64_t endTensorOffset,
    int64_t& dim1Size, int64_t& dim0stride, int64_t& copyCols, int64_t& extraCols, bool& isSplit)
{
    dim1Size = GetNonConDimSize<TILINGDATA, T>(tilingData_, tensorIdx, inputList_, desc_) * sameShapeTensorDim1_;
    if (isFP4Type_) {
        dim1Size /= 2;
    }
    dim0stride = GetTensorDim0Stride<TILINGDATA>(tilingData_, tensorIdx, dim1Size);
    copyCols = dim1Size - colsOffset;
    if (tensorIdx == endTensorIdx) {
        copyCols = endTensorOffset - colsOffset;
    }
    extraCols = totalCopyCols + copyCols - ubFactorDim1_;
    isSplit = extraCols >= numPerBlock_ || (extraCols > 0 && totalCopyCols == 0);
}

// gather 逐段 2D DMA 兜底: PBS1/PBS01 共用, srcRowBase 统一行偏移
template <typename T, typename U, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignDiffShape<T, U, TILINGDATA>::GatherSegDma(
    int64_t tensorIdx, int64_t dim0stride, int64_t seg, int64_t numSeg, int64_t startSeg, int64_t rowSrcStride,
    int64_t copyCols, uint16_t rows, int64_t srcRowBase, int64_t outTensorColsOffset,
    DataCopyPadExtParams<T>& padParams)
{
    GlobalTensor<T> srcGlobal;
    DataCopyExtParams gatherInParam = {static_cast<uint16_t>(numSeg), static_cast<uint32_t>(seg * sizeof(T)),
                                       static_cast<int64_t>((dim0stride - seg) * sizeof(T)), 0, 0};
    // 双 buffer 轮转: 上一片 MTE3 读出与下一片 MTE2 读入可并行
    uint16_t slot = dmaSlot_;
    dmaSlot_ ^= 1;
    event_t evt = slot == 0 ? EVT_ID0 : EVT_ID1;
    WaitFlag<HardEvent::MTE3_MTE2>(evt);
    LocalTensor<T> workLocal(TPosition::VECIN, slot == 0 ? inOffset0_ : inOffset1_, bufElems_);
    for (int64_t r = 0; r < rows; r++) {
        srcGlobal.SetGlobalBuffer(GetTensorAddr(tensorIdx, (srcRowBase + r) * rowSrcStride + startSeg * dim0stride));
        DataCopyPad<T, PaddingMode::Compact>(workLocal[r * CeilAlign(copyCols, numPerBlock_)], srcGlobal, gatherInParam,
                                             padParams);
    }
    SetFlag<HardEvent::MTE2_MTE3>(evt);
    WaitFlag<HardEvent::MTE2_MTE3>(evt);
    for (int64_t r = 0; r < rows; r++) {
        DataCopyExtParams rowOutParam = {1, static_cast<uint32_t>(copyCols * sizeof(T)), 0, 0, 0};
        DataCopyPad(dstGlobal_[outTensorColsOffset + r * catDim1_], workLocal[r * CeilAlign(copyCols, numPerBlock_)],
                    rowOutParam);
    }
    SetFlag<HardEvent::MTE3_MTE2>(evt);
}

// 老路径攒批写入: PBS1/PBS01 共用, 读入 srcLocal[tensorStride] 偏移 + ComputeSplitDim1 重排
template <typename T, typename U, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignDiffShape<T, U, TILINGDATA>::LegacyBatchWrite(
    int64_t tensorIdx, int64_t dim0stride, int64_t copyCols, uint16_t rows, int64_t colsOffset, int64_t srcRowBase,
    int64_t totalCopyCols, uint32_t curLoopHandleCols, LocalTensor<T>& srcLocal, LocalTensor<T>& dstLocal,
    int64_t tensorStride, DataCopyPadExtParams<T>& padParams)
{
    GlobalTensor<T> srcGlobal;
    DataCopyExtParams copyInParam = {rows, static_cast<uint32_t>(copyCols * sizeof(T)),
                                     static_cast<int64_t>((dim0stride - copyCols) * sizeof(T)),
                                     static_cast<int64_t>(copyCols * sizeof(T)), 0};
    srcGlobal.SetGlobalBuffer(GetTensorAddr(tensorIdx, srcRowBase * dim0stride + colsOffset));
#if defined(__DAV_920R1__) || defined(__DAV_920R2__)
    DataCopyPad<T, PaddingMode::Compact>(srcLocal[tensorStride], srcGlobal, copyInParam, padParams, true);
#else
    DataCopyPad<T, PaddingMode::Compact>(srcLocal[tensorStride], srcGlobal, copyInParam, padParams);
#endif
    ComputeSplitDim1(dstLocal, srcLocal[tensorStride], rows, copyCols, totalCopyCols, curLoopHandleCols);
}

} // namespace Concat
#endif // ONE_AXIS_CONCAT_NO_ALIGN_DIFF_SHAPE
