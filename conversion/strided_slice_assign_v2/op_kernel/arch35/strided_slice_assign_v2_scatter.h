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
 * \file strided_slice_assign_v2_scatter.h
 * \brief strided_slice_assign_v2_scatter.h
 */

#ifndef STRIDED_SLICE_ASSIGN_V2_SCATTER_H_
#define STRIDED_SLICE_ASSIGN_V2_SCATTER_H_

#include <cstdint>
#include <type_traits>

#include "kernel_operator.h"
#include "strided_slice_assign_v2_tiling_data.h"

namespace StridedSliceAssignV2ScatterNs {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 4;      // input / var 队列乒乓槽位（4 槽）
constexpr int64_t ALIGN_32_BYTES = 32; // 32B 对齐字节数
constexpr int64_t VEC_WIDTH_BYTES = 256;

__aicore__ inline int64_t SsaCeilDiv(int64_t a, int64_t b) { return (b == 0) ? 0 : ((a + b - 1) / b); }

__aicore__ inline int64_t SsaMin(int64_t a, int64_t b) { return (a < b) ? a : b; }

__aicore__ inline int64_t SsaAbs(int64_t v) { return (v < 0) ? (-v) : v; }

__aicore__ inline int64_t SsaAlignUp(int64_t value, int64_t align)
{
    return (align <= 0) ? value : SsaCeilDiv(value, align) * align;
}

struct SsaCopyDims {
    uint16_t bcCount = 1;
    uint32_t blkByt = 0;
    int64_t bcGmGap = 0;
    uint32_t l1Count = 1;
    uint64_t l1UbStep = 0;
    uint64_t l1GmStep = 0;
    uint32_t l2Count = 1;
    uint64_t l2UbStep = 0;
    uint64_t l2GmStep = 0;
};

template <typename T>
class StridedSliceAssignV2Scatter {
    using IdxT = std::conditional_t<sizeof(T) <= sizeof(int16_t), uint16_t, uint32_t>;
    using RangeT = std::conditional_t<sizeof(IdxT) == sizeof(uint16_t), int16_t, int32_t>;

    static constexpr int64_t ELEM_SIZE = static_cast<int64_t>(sizeof(T));
    static constexpr int64_t VF_WIDTH_ELEMS = VEC_WIDTH_BYTES /
                                              ((ELEM_SIZE == 1) ? static_cast<int64_t>(sizeof(uint16_t)) : ELEM_SIZE);
    using MaskT = std::conditional_t<ELEM_SIZE == 1, uint16_t, T>;

    static constexpr int64_t IDX_WIDTH_ELEMS = VEC_WIDTH_BYTES / static_cast<int64_t>(sizeof(IdxT));

public:
    __aicore__ inline explicit StridedSliceAssignV2Scatter(TPipe* pipe) : Ppipe_(pipe) {}
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR inputValue, GM_ADDR varOut,
                                const StridedSliceAssignV2ScatterTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline int64_t ComputePositionIndices(int64_t* posIdx, int64_t task) const;
    __aicore__ inline void ProcessOneFactor(const int64_t* posIdx, int64_t curFactor);
    __aicore__ inline int64_t BlkVarOff(const int64_t* posIdx, int64_t curFactor) const;
    __aicore__ inline int64_t BlkInputOff(const int64_t* posIdx) const;
    __aicore__ inline int64_t TblOff(int64_t curFactor) const;
    __aicore__ inline void MultiDimCopyBlock(const LocalTensor<T>& dstBlk, const GlobalTensor<T>& gm, int64_t gmOff,
                                             int64_t curFactor, bool isLoad);
    __aicore__ inline void LoadVarBlock(const LocalTensor<T>& dstBlk, const GlobalTensor<T>& gm, int64_t gmOff,
                                        int64_t curFactor);
    __aicore__ inline void StoreVarBlock(const GlobalTensor<T>& gm, const LocalTensor<T>& srcBlk, int64_t gmOff,
                                         int64_t curFactor);
    __aicore__ inline SsaCopyDims CalcCopyDims(int64_t curFactor) const;
    __aicore__ inline void CopyLoopCore(const LocalTensor<T>& ubBlk, const GlobalTensor<T>& gm, int64_t gmOff,
                                        uint16_t bcCount, uint32_t blkByt, int64_t bcGmGap, uint32_t l1Count,
                                        uint64_t l1UbStep, uint64_t l1GmStep, uint32_t l2Count, uint64_t l2UbStep,
                                        uint64_t l2GmStep, bool isLoad);
    __aicore__ inline void ComputeScatter(const LocalTensor<T>& varLocal, const LocalTensor<T>& inLocal,
                                          int64_t curFactor);
    __aicore__ inline void PrepareOffsetTable();

private:
    TPipe* Ppipe_{nullptr};
    TQue<TPosition::VECIN, BUFFER_NUM> inputQueue_;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, BUFFER_NUM> varQue_;
    TBuf<TPosition::VECCALC> offsetBuf_;
    GlobalTensor<T> varGm_;
    GlobalTensor<T> inputValueGm_;
    GlobalTensor<T> varOutGm_;
    DataCopyPadExtParams<T> padParams_{false, 0, 0, 0};

    int32_t ubAxis_{0};
    int64_t ubFactor_{0};
    int64_t ubPerCount_{0};
    int64_t ubTotalCount_{0};
    int64_t tileElemNum_{0};
    int32_t idxDtypeSize_{0};
    int32_t dimNum_{0};
    int32_t k_{0};
    int64_t innerElems_{0};
    int64_t rows_{0};
    int64_t cols_{0};
    int64_t ubSpan_{0};
    int64_t pitchUb_{0};
    int64_t sLast_{0};
    int64_t varBase_{0};
    bool positive_{true};

    int64_t axisStep_[SSA_MAX_DIM_NUM] = {};
    int64_t sliceShape_[SSA_MAX_DIM_NUM] = {};
    int64_t sliceCum_[SSA_MAX_DIM_NUM] = {};
    int64_t rowStepVar_[SSA_MAX_DIM_NUM] = {};
    int64_t blkStartVar_[SSA_MAX_DIM_NUM] = {};
    int64_t startIdx_{0};
    int64_t endIdx_{0};
};

template <typename T>
__aicore__ inline void StridedSliceAssignV2Scatter<T>::Init(GM_ADDR var, GM_ADDR inputValue, GM_ADDR varOut,
                                                            const StridedSliceAssignV2ScatterTilingData* tilingData)
{
    ubAxis_ = static_cast<int32_t>(tilingData->ubAxis);
    ubFactor_ = static_cast<int64_t>(tilingData->ubFactor);
    ubPerCount_ = static_cast<int64_t>(tilingData->ubPerCount);
    ubTotalCount_ = static_cast<int64_t>(tilingData->ubTotalCount);
    tileElemNum_ = static_cast<int64_t>(tilingData->tileElemNum);
    idxDtypeSize_ = tilingData->idxDtypeSize;
    dimNum_ = static_cast<int32_t>(tilingData->dimNum);
    innerElems_ = tilingData->innerElems;
    rows_ = tilingData->rows;
    cols_ = tilingData->cols;
    ubSpan_ = tilingData->ubSpan;
    pitchUb_ = tilingData->pitchUb;
    varBase_ = tilingData->varBase;
    for (int32_t i = 0; i < SSA_MAX_DIM_NUM; ++i) {
        sliceShape_[i] = tilingData->sliceShape[i];
        axisStep_[i] = tilingData->axisStep[i];
        sliceCum_[i] = tilingData->sliceCum[i];
        rowStepVar_[i] = tilingData->rowStepVar[i];
        blkStartVar_[i] = tilingData->blkStartVar[i];
    }

    k_ = dimNum_ - ubAxis_;
    sLast_ = SsaAbs(axisStep_[dimNum_ - 1]);
    positive_ = (axisStep_[dimNum_ - 1] > 0);

    varGm_.SetGlobalBuffer((__gm__ T*)var);
    inputValueGm_.SetGlobalBuffer((__gm__ T*)inputValue);
    varOutGm_.SetGlobalBuffer((__gm__ T*)varOut);

    const uint32_t varBytes = static_cast<uint32_t>(SsaAlignUp(rows_ * pitchUb_ * ELEM_SIZE, ALIGN_32_BYTES));
    const uint32_t inBytes = static_cast<uint32_t>(
        SsaAlignUp(tileElemNum_ * ELEM_SIZE + VEC_WIDTH_BYTES, ALIGN_32_BYTES));
    const uint32_t offBytes = static_cast<uint32_t>(
        SsaAlignUp(tileElemNum_ * static_cast<int64_t>(idxDtypeSize_) + VEC_WIDTH_BYTES, ALIGN_32_BYTES));

    Ppipe_->InitBuffer(varQue_, BUFFER_NUM, varBytes);
    Ppipe_->InitBuffer(inputQueue_, BUFFER_NUM, inBytes);
    Ppipe_->InitBuffer(offsetBuf_, offBytes);

    PrepareOffsetTable();

    const int64_t blk = static_cast<int64_t>(GetBlockIdx());
    startIdx_ = blk * ubPerCount_;
    endIdx_ = SsaMin((blk + 1) * ubPerCount_, ubTotalCount_);
}

template <typename T>
__aicore__ inline void StridedSliceAssignV2Scatter<T>::Process()
{
    if (startIdx_ >= endIdx_) {
        return;
    }
    int64_t posIdx[SSA_MAX_DIM_NUM] = {0};
    for (int64_t task = startIdx_; task < endIdx_; ++task) {
        const int64_t curFactor = ComputePositionIndices(posIdx, task);
        if (curFactor <= 0) {
            continue;
        }
        ProcessOneFactor(posIdx, curFactor);
    }
}

// 以 dim 0 为最低位、起点恒为合法切片坐标，mixed-radix 反解：
//   factor(i) = (i == ubAxis) ? CeilDiv(sliceShape[ubAxis], ubFactor) : sliceShape[i]
//   digit     = cur % factor(i); cur /= factor(i)
//   posIdx[i] = (i == ubAxis) ? digit · ubFactor : digit
// 返回 curFactor = min(ubFactor, sliceShape[ubAxis] − posIdx[ubAxis])（尾块收缩）
template <typename T>
__aicore__ inline int64_t StridedSliceAssignV2Scatter<T>::ComputePositionIndices(int64_t* posIdx, int64_t task) const
{
    int64_t cur = task;
    int64_t curFactor = 1;
    for (int32_t i = ubAxis_; i >= 0; --i) {
        const int64_t factor = (i == ubAxis_) ? SsaCeilDiv(sliceShape_[i], ubFactor_) : sliceShape_[i];
        if (factor <= 0) {
            continue;
        }
        const int64_t digit = cur % factor;
        cur /= factor;
        if (i == ubAxis_) {
            posIdx[i] = digit * ubFactor_;
            curFactor = SsaMin(ubFactor_, sliceShape_[i] - posIdx[i]);
        } else {
            posIdx[i] = digit;
        }
    }
    return curFactor;
}

template <typename T>
__aicore__ inline void StridedSliceAssignV2Scatter<T>::ProcessOneFactor(const int64_t* posIdx, int64_t curFactor)
{
    const int64_t blockElems = curFactor * innerElems_;
    const int64_t blkVarOff = BlkVarOff(posIdx, curFactor);
    const int64_t blkInOff = BlkInputOff(posIdx);

    LocalTensor<T> varLocal = varQue_.AllocTensor<T>();
    MultiDimCopyBlock(varLocal, varGm_, blkVarOff, curFactor, true);

    LocalTensor<T> inLocal = inputQueue_.AllocTensor<T>();
    const DataCopyExtParams cpIn = {1, static_cast<uint32_t>(blockElems * ELEM_SIZE), 0, 0, 0};
    DataCopyPad(inLocal, inputValueGm_[blkInOff], cpIn, padParams_);

    inputQueue_.EnQue(inLocal);
    varQue_.EnQue(varLocal);

    inLocal = inputQueue_.DeQue<T>();
    varLocal = varQue_.DeQue<T>();

    const event_t evMte2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(evMte2V);
    WaitFlag<HardEvent::MTE2_V>(evMte2V);

    ComputeScatter(varLocal, inLocal, curFactor);

    const event_t evVMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(evVMte3);
    WaitFlag<HardEvent::V_MTE3>(evVMte3);

    MultiDimCopyBlock(varLocal, varOutGm_, blkVarOff, curFactor, false);

    varQue_.FreeTensor(varLocal);
    inputQueue_.FreeTensor(inLocal);
}

template <typename T>
__aicore__ inline int64_t StridedSliceAssignV2Scatter<T>::BlkVarOff(const int64_t* posIdx, int64_t curFactor) const
{
    int64_t off = varBase_;
    for (int32_t d = 0; d < ubAxis_; ++d) {
        off += posIdx[d] * axisStep_[d];
    }
    for (int32_t d = ubAxis_; d < dimNum_; ++d) {
        off += blkStartVar_[d];
    }
    if (axisStep_[ubAxis_] > 0) {
        off += posIdx[ubAxis_] * axisStep_[ubAxis_];
    } else {
        off += (sliceShape_[ubAxis_] - posIdx[ubAxis_] - curFactor) * (-axisStep_[ubAxis_]);
    }
    return off;
}

template <typename T>
__aicore__ inline int64_t StridedSliceAssignV2Scatter<T>::BlkInputOff(const int64_t* posIdx) const
{
    int64_t off = 0;
    for (int32_t d = 0; d < ubAxis_; ++d) {
        off += posIdx[d] * sliceCum_[d + 1];
    }
    off += posIdx[ubAxis_] * innerElems_;
    return off;
}

template <typename T>
__aicore__ inline int64_t StridedSliceAssignV2Scatter<T>::TblOff(int64_t curFactor) const
{
    return positive_ ? 0 : (ubFactor_ - curFactor) * innerElems_;
}

// 计算 var 帧多维 DMA 搬运的各级计数与步长（按 k_ = dimNum_ - ubAxis_ 逐维递增）：
//   k_ == 1：blockLen（最内维，切分轴在 blockLen 内收缩）
//   k_ == 2：blockLen + blockCount（切分轴置于 blockCount）
//   k_ == 3：blockLen + blockCount + loop1（切分轴置于 loop1）
//   k_ == 4：blockLen + blockCount + loop1 + loop2（切分轴置于 loop2）
template <typename T>
__aicore__ inline SsaCopyDims StridedSliceAssignV2Scatter<T>::CalcCopyDims(int64_t curFactor) const
{
    SsaCopyDims d;
    const int64_t ubRowByt = pitchUb_ * ELEM_SIZE;

    if (k_ == 1) {
        const int64_t curCols = SsaMin(cols_, curFactor);
        d.blkByt = static_cast<uint32_t>(((curCols - 1) * sLast_ + 1) * ELEM_SIZE);
        return d;
    }

    // k_ >= 2：最内维整行搬满 cols_，倒数第二维（dimNum_-2）由 blockCount 承载
    d.blkByt = static_cast<uint32_t>(ubSpan_ * ELEM_SIZE);
    const int64_t bcStep = rowStepVar_[dimNum_ - 2] * ELEM_SIZE;
    if (k_ == 2) {
        d.bcCount = static_cast<uint16_t>(curFactor);
        d.bcGmGap = bcStep - static_cast<int64_t>(d.blkByt);
        return d;
    }
    d.bcCount = static_cast<uint16_t>(sliceShape_[dimNum_ - 2]);
    d.bcGmGap = bcStep - static_cast<int64_t>(d.blkByt);

    // k_ == 3：倒数第三维（切分轴 dimNum_-3）由 loop1 承载
    if (k_ == 3) {
        d.l1Count = static_cast<uint32_t>(curFactor);
        d.l1GmStep = static_cast<uint64_t>(SsaAbs(axisStep_[ubAxis_]) * ELEM_SIZE);
        d.l1UbStep = static_cast<uint64_t>(ubRowByt * sliceShape_[dimNum_ - 2]);
        return d;
    }

    // k_ == 4：倒数第三维由 loop1、倒数第四维（切分轴 dimNum_-4）由 loop2 承载
    d.l1Count = static_cast<uint32_t>(sliceShape_[dimNum_ - 3]);
    d.l1GmStep = static_cast<uint64_t>(rowStepVar_[dimNum_ - 3] * ELEM_SIZE);
    d.l1UbStep = static_cast<uint64_t>(ubRowByt * sliceShape_[dimNum_ - 2]);
    d.l2Count = static_cast<uint32_t>(curFactor);
    d.l2GmStep = static_cast<uint64_t>(SsaAbs(axisStep_[ubAxis_]) * ELEM_SIZE);
    d.l2UbStep = static_cast<uint64_t>(ubRowByt * sliceShape_[dimNum_ - 2] * sliceShape_[dimNum_ - 3]);
    return d;
}

template <typename T>
__aicore__ inline void StridedSliceAssignV2Scatter<T>::CopyLoopCore(const LocalTensor<T>& ubBlk,
                                                                    const GlobalTensor<T>& gm, int64_t gmOff,
                                                                    uint16_t bcCount, uint32_t blkByt, int64_t bcGmGap,
                                                                    uint32_t l1Count, uint64_t l1UbStep,
                                                                    uint64_t l1GmStep, uint32_t l2Count,
                                                                    uint64_t l2UbStep, uint64_t l2GmStep, bool isLoad)
{
    const DataCopyExtParams p = {bcCount, blkByt, isLoad ? bcGmGap : static_cast<int64_t>(0),
                                 isLoad ? static_cast<int64_t>(0) : bcGmGap, 0};

    if (l1Count <= 1 && l2Count <= 1) {
        if (isLoad) {
            DataCopyPad(ubBlk, gm[gmOff], p, padParams_);
        } else {
            DataCopyPad(gm[gmOff], ubBlk, p);
        }
        return;
    }

    LoopModeParams lp;
    lp.loop1Size = l1Count;
    lp.loop2Size = l2Count;
    lp.loop1SrcStride = isLoad ? l1GmStep : l1UbStep;
    lp.loop1DstStride = isLoad ? l1UbStep : l1GmStep;
    lp.loop2SrcStride = isLoad ? l2GmStep : l2UbStep;
    lp.loop2DstStride = isLoad ? l2UbStep : l2GmStep;

    const DataCopyMVType dir = isLoad ? DataCopyMVType::OUT_TO_UB : DataCopyMVType::UB_TO_OUT;
    SetLoopModePara(lp, dir);
    if (isLoad) {
        DataCopyPad(ubBlk, gm[gmOff], p, padParams_);
    } else {
        DataCopyPad(gm[gmOff], ubBlk, p);
    }
    ResetLoopModePara(dir);
}

template <typename T>
__aicore__ inline void StridedSliceAssignV2Scatter<T>::LoadVarBlock(const LocalTensor<T>& dstBlk,
                                                                    const GlobalTensor<T>& gm, int64_t gmOff,
                                                                    int64_t curFactor)
{
    const SsaCopyDims d = CalcCopyDims(curFactor);
    CopyLoopCore(dstBlk, gm, gmOff, d.bcCount, d.blkByt, d.bcGmGap, d.l1Count, d.l1UbStep, d.l1GmStep, d.l2Count,
                 d.l2UbStep, d.l2GmStep, true);
}

template <typename T>
__aicore__ inline void StridedSliceAssignV2Scatter<T>::StoreVarBlock(const GlobalTensor<T>& gm,
                                                                     const LocalTensor<T>& srcBlk, int64_t gmOff,
                                                                     int64_t curFactor)
{
    const SsaCopyDims d = CalcCopyDims(curFactor);
    CopyLoopCore(srcBlk, gm, gmOff, d.bcCount, d.blkByt, d.bcGmGap, d.l1Count, d.l1UbStep, d.l1GmStep, d.l2Count,
                 d.l2UbStep, d.l2GmStep, false);
}

template <typename T>
__aicore__ inline void StridedSliceAssignV2Scatter<T>::MultiDimCopyBlock(const LocalTensor<T>& dstBlk,
                                                                         const GlobalTensor<T>& gm, int64_t gmOff,
                                                                         int64_t curFactor, bool isLoad)
{
    if (isLoad) {
        LoadVarBlock(dstBlk, gm, gmOff, curFactor);
    } else {
        StoreVarBlock(gm, dstBlk, gmOff, curFactor);
    }
}

template <typename T>
__aicore__ inline void StridedSliceAssignV2Scatter<T>::ComputeScatter(const LocalTensor<T>& varLocal,
                                                                      const LocalTensor<T>& inLocal, int64_t curFactor)
{
    const int64_t blockElems = curFactor * innerElems_;

    if (blockElems <= 0) {
        return;
    }

    const int64_t tblOff = TblOff(curFactor);
    LocalTensor<IdxT> offLocal = offsetBuf_.Get<IdxT>();

    __ubuf__ T* varAddr = reinterpret_cast<__ubuf__ T*>(varLocal.GetPhyAddr());
    __ubuf__ T* inAddr = reinterpret_cast<__ubuf__ T*>(inLocal.GetPhyAddr());
    __ubuf__ IdxT* tblPtr = reinterpret_cast<__ubuf__ IdxT*>(offLocal.GetPhyAddr()) + tblOff;

    const uint16_t reps = static_cast<uint16_t>(SsaCeilDiv(blockElems, VF_WIDTH_ELEMS));

    __VEC_SCOPE__
    {
        Reg::RegTensor<T> regData;
        Reg::RegTensor<IdxT> regIdx;
        Reg::UnalignRegForLoad uReg;

        Reg::MaskReg mask;

        uint32_t cnt = static_cast<uint32_t>(blockElems);

        __ubuf__ T* dataPtr = inAddr;
        __ubuf__ IdxT* idxPtr = tblPtr;

        Reg::LoadUnAlignPre(uReg, idxPtr);
        for (uint16_t rep = 0; rep < reps; ++rep) {
            mask = Reg::UpdateMask<MaskT>(cnt);
            Reg::LoadUnAlign(regIdx, uReg, idxPtr, static_cast<uint32_t>(VF_WIDTH_ELEMS));
            if constexpr (ELEM_SIZE == 1) {
                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B8>(regData, dataPtr + rep * VF_WIDTH_ELEMS);
            } else {
                Reg::LoadAlign(regData, dataPtr + rep * VF_WIDTH_ELEMS);
            }
            Reg::Scatter(varAddr, regData, regIdx, mask);
        }
    }
}

template <typename T>
__aicore__ inline void StridedSliceAssignV2Scatter<T>::PrepareOffsetTable()
{
    if (tileElemNum_ <= 0 || cols_ <= 0) {
        return;
    }

    const int64_t sign = positive_ ? 1 : -1;

    const IdxT pConst = static_cast<IdxT>(positive_ ? 0 : ((rows_ - 1) * pitchUb_ + (cols_ - 1) * sLast_));
    const IdxT dConst = static_cast<IdxT>(sign * sLast_);
    const IdxT eConst = static_cast<IdxT>(sign * (pitchUb_ - sLast_ * cols_));

    __ubuf__ IdxT* tblBase = reinterpret_cast<__ubuf__ IdxT*>(offsetBuf_.Get<IdxT>().GetPhyAddr());

    const uint16_t reps = static_cast<uint16_t>(SsaCeilDiv(tileElemNum_, IDX_WIDTH_ELEMS));

    __VEC_SCOPE__
    {
        Reg::MaskReg mask;
        Reg::RegTensor<RangeT> regRange;
        Reg::RegTensor<IdxT> regIdx;
        Reg::RegTensor<IdxT> regQuo;
        Reg::RegTensor<IdxT> regCols;
        Reg::Duplicate(regCols, static_cast<IdxT>(cols_));
        uint32_t cnt = static_cast<uint32_t>(tileElemNum_);
        IdxT firstEntry = 0;
        for (uint16_t rep = 0; rep < reps; ++rep) {
            mask = Reg::UpdateMask<IdxT>(cnt);
            Reg::Arange(regRange, static_cast<RangeT>(firstEntry));
            regIdx = (Reg::RegTensor<IdxT>&)regRange;
            firstEntry += static_cast<IdxT>(IDX_WIDTH_ELEMS);
            Reg::Div(regQuo, regIdx, regCols, mask);
            Reg::Muls(regIdx, regIdx, dConst, mask);
            Reg::Adds(regIdx, regIdx, pConst, mask);
            Reg::Muls(regQuo, regQuo, eConst, mask);
            Reg::Add(regIdx, regIdx, regQuo, mask);
            Reg::StoreAlign<IdxT>(tblBase + rep * IDX_WIDTH_ELEMS, regIdx, mask);
        }
    }
}

} // namespace StridedSliceAssignV2ScatterNs

#endif // STRIDED_SLICE_ASSIGN_V2_SCATTER_H_
