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
 * \file transpose_gather.h
 * \brief TilingKey=10006 GATHER 策略实现
 *
 * 适用场景：尾轴转置（isLastAxisTranspose==true）+ 大shape（非小shape场景）。
 *
 * 核心机制：使用 DAV_3510 的 Gather 硬件指令（RegBase 路径）
 *   Gather 根据预计算的索引数组从 UB 中按任意跨步读取数据，
 *   实现尾轴转置。索引数组在 Init 阶段预生成，最多4组覆盖4种 main/tail 组合。
 *
 * Buffer 规划（静态 Tensor，不使用 TPipe/TQue，Mutex 做流水同步）：
 *   - xInLocal_[2]：输入静态双缓冲，存放 CopyIn 的数据
 *   - xOutLocal_[2]：输出静态双缓冲，存放 Gather 后的数据
 *   - idxLocal_：Gather 索引缓冲，存放预计算的4组索引
 *
 * Gather 索引生成算法：
 *   支持1/2/3维索引（GenIndex4OneDim / GenIndex4TwoDim / GenIndex4ThreeDim）
 *   使用 Reg::Arange + Reg::Div + Reg::Mul + Reg::Muls 等 RegBase 指令计算索引
 *   核心思路：将输出线性索引分解到各维度，再按 perm 映射计算输入偏移
 *
 * 类型适配：
 *   - RangeType_：8bit→int16_t，16bit→int16_t，32/64bit→int32_t（索引计算类型）
 *   - IdxType_：8bit→uint16_t，16bit→uint16_t，32/64bit→uint32_t（Gather 索引类型）
 *   - CastType_：8bit→uint16_t/int16_t（8bit 需 Cast 到16bit 进行 DataCopyGather）
 *
 * 4种 phase 的索引偏移控制（gIndexId_）：
 *   0: out_cut=false, in_cut=false  （主块）
 *   1: out_cut=false, in_cut=true   （输入尾块）
 *   2: out_cut=true,  in_cut=false  （输出尾块）
 *   3: out_cut=true,  in_cut=true   （双尾块）
 */
#ifndef KERNEL_TRANSPOSE_WITH_GATHER_H_
#define KERNEL_TRANSPOSE_WITH_GATHER_H_

#include <type_traits>

#include "op_kernel/platform_util.h"
#include "transpose_base.h"

namespace Transpose {
using namespace AscendC;
using AscendC::Reg::CreateMask;
using AscendC::Reg::Gather;
using AscendC::Reg::MaskReg;
using AscendC::Reg::RegTensor;
using AscendC::Reg::UpdateMask;

constexpr int8_t NUM_TWO = 2;
constexpr int8_t NUM_THREE = 3;

/**
 * @brief GATHER 策略类，使用 Gather 硬件指令实现尾轴转置
 *
 * 执行流程：
 * 1. InitAxes: 初始化 UB 轴尺寸
 * 2. GenGatherIndex4AllPhase: 预生成4种 phase 的 Gather 索引
 * 3. for each block loop:
 *    a. CalcBlkAddr: 计算当前 block 的 GM 输入/输出地址
 *    b. UpdateUbAxes: 根据 block 位置更新 UB 轴尺寸（main/tail）
 *    c. CopyDataIn: DataCopyPad(GM → UB) + LoopMode
 *    d. GetOutLoopAxes: 获取输出循环轴参数
 *    e. GatherData: Gather(UB_in → UB_out) 使用 RegBase 硬件
 *    f. CopyDataOut: DataCopyPad(UB → GM) + LoopMode
 *
 * @tparam T 数据元素类型
 */
template <typename T>
class TransposeGather {
public:
    __aicore__ inline TransposeGather(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TransposeGatherTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void GetCoreLoopRange();
    __aicore__ inline void InitAxes();
    __aicore__ inline int32_t CalcUbAxesInOffset(int8_t begIdx);
    __aicore__ inline int32_t CalcUbAxesOutOffset(int8_t begIdx);
    __aicore__ inline int64_t CalcBlkAxesOffset(int8_t begIdx);
    __aicore__ inline void GenIndex4OneDim(int32_t goffset);
    __aicore__ inline void GenIndex4TwoDim(int32_t goffset);
    __aicore__ inline void GenIndex4ThreeDim(int32_t goffset);
    __aicore__ inline void GenGatherIndex(int32_t goffset);
    __aicore__ inline void GenGatherIndex4AllPhase();
    __aicore__ inline void CalcBlkAddr(int64_t idx);
    __aicore__ inline void SetCopyInParams();
    __aicore__ inline void CopyDataIn(uint32_t bufId);
    __aicore__ inline void GetOutLoopAxes();
    __aicore__ inline void SetCopyOutParams();
    __aicore__ inline void CopyDataOut(uint32_t bufId);
    __aicore__ inline void UpdateUbAxes(int64_t blkLpIdx);
    __aicore__ inline void GatherData(uint32_t bufId);

private:
    int64_t blockIdx_ = 0;
    const TransposeGatherTilingData* td_ = nullptr;
    GlobalTensor<T> xGM_;
    GlobalTensor<T> yGM_;

    int64_t blkBeg_ = 0;
    int64_t blkEnd_ = 0;
    int64_t blkInAddr_ = 0;                                            ///< 当前 block 的 GM 输入起始地址偏移
    int64_t blkOutAddr_ = 0;                                           ///< 当前 block 的 GM 输出起始地址偏移
    int64_t blkInCutROffset_ = 0;                                      ///< 输入切分轴的块级循环偏移
    int64_t blkOutCutROffset_ = 0;                                     ///< 输出切分轴的块级循环偏移
    int64_t blkInCutAxisSize_ = 0;                                     ///< 输入切分轴的块级尺寸
    int64_t blkOutCutAxisSize_ = 0;                                    ///< 输出切分轴的块级尺寸
    int32_t inUbAxes_[TP_GATHER_UB_MAX_DIM_NUM] = {1, 1, 1, 1, 1, 1};  ///< 输入 UB 各轴尺寸（动态更新）
    int32_t outUbAxes_[TP_GATHER_UB_MAX_DIM_NUM] = {1, 1, 1, 1, 1, 1}; ///< 输出 UB 各轴尺寸（动态更新）
    /* 4种 Gather 索引 phase 的控制变量
     *  0: out_cut=false, in_cut=false  （主块）
     *  1: out_cut=false, in_cut=true   （输入尾块）
     *  2: out_cut=true,  in_cut=false  （输出尾块）
     *  3: out_cut=true,  in_cut=true   （双尾块）
     */
    int32_t gIndexId_ = 0; // 控制 Gather 索引的偏移量

    /* 类型适配：根据元素字节宽度选择合适的索引计算类型和 Gather 索引类型 */
    using RangeType_ = std::conditional_t<sizeof(T) <= sizeof(int16_t), int16_t, int32_t>; ///< 索引计算类型
    using IdxType_ = std::conditional_t<sizeof(T) <= sizeof(int16_t), uint16_t, uint32_t>; ///< Gather 索引类型
    using CastType_ = std::conditional_t<sizeof(T) == 1,                                   ///< 8bit Cast 类型
                                         std::conditional_t<std::is_same_v<T, uint8_t>, uint16_t, int16_t>, T>;
    uint32_t vlSize_ = static_cast<uint32_t>(Ops::Base::GetVRegSize() / sizeof(CastType_)); ///< 向量寄存器元素数
    uint32_t idxVLSize_ = static_cast<uint32_t>(Ops::Base::GetVRegSize() /
                                                sizeof(RangeType_)); ///< 索引向量寄存器元素数
    // 静态 Tensor：输入/输出各双缓冲 + Gather 索引缓冲；不使用 TPipe/TQue。
    LocalTensor<T> xInLocal_[BUFFER_NUM];  ///< 输入静态双缓冲
    LocalTensor<T> xOutLocal_[BUFFER_NUM]; ///< 输出静态双缓冲
    LocalTensor<RangeType_> idxLocal_;     ///< Gather 索引的 Local Tensor
    int32_t gElemPerBlock_ = static_cast<int32_t>(BLOCK_SIZE_BYTE / sizeof(RangeType_)); ///< 索引每 Block 元素数
    int32_t elemPerBlock_ = static_cast<int32_t>(BLOCK_SIZE_BYTE / sizeof(T));           ///< 数据每 Block 元素数
    int32_t gIdxOffset_ = 0; ///< 每组 Gather 索引的偏移量（字节数）

    LoopModeParams lpModeInParams_ = {1, 1, 0, 0, 0, 0};
    LoopModeParams lpModeOutParams_ = {1, 1, 0, 0, 0, 0};
    DataCopyExtParams copyInParams_;
    DataCopyExtParams copyOutParams_;
    uint32_t outUbAxis0_ = 1;
    uint32_t outUbAxis1_ = 1;
    uint32_t outUbAxis2_ = 1;
    uint32_t outUbAxis0InROffset_ = 0;
    uint32_t outUbAxis1InROffset_ = 0;
    uint32_t outUbAxis2InROffset_ = 0;
    int32_t inUbTailFactor_ = 0;
    int32_t outUbTailFactor_ = 0;
};

template <typename T>
__aicore__ inline void TransposeGather<T>::Init(GM_ADDR x, GM_ADDR y, const TransposeGatherTilingData* tilingData)
{
    blockIdx_ = GetBlockIdx();
    td_ = tilingData;

    // 静态 UB 布局：in0 | in1 | out0 | out1 | index（各段按 ubBlockSize 块对齐）
    const uint32_t dataSize = td_->dataTensorSize;
    for (uint32_t i = 0; i < BUFFER_NUM; ++i) {
        xInLocal_[i] = LocalTensor<T>(TPosition::VECIN, i * dataSize, dataSize / sizeof(T));
        xOutLocal_[i] = LocalTensor<T>(TPosition::VECOUT, (BUFFER_NUM + i) * dataSize, dataSize / sizeof(T));
    }
    idxLocal_ = LocalTensor<RangeType_>(TPosition::VECCALC, BUFFER_NUM * 2 * dataSize,
                                        td_->indexTensorSize / sizeof(RangeType_));

    xGM_.SetGlobalBuffer((__gm__ T*)x);
    yGM_.SetGlobalBuffer((__gm__ T*)y);
}

template <typename T>
__aicore__ inline void TransposeGather<T>::Process()
{
    if (blockIdx_ >= static_cast<int64_t>(GetBlockNum())) {
        return;
    }
    GetCoreLoopRange();
    InitAxes();
    GenGatherIndex4AllPhase();
    for (int64_t blkLpIdx = blkBeg_; blkLpIdx < blkEnd_; ++blkLpIdx) {
        const uint32_t bufId = static_cast<uint32_t>(blkLpIdx - blkBeg_) & 1U;
        CalcBlkAddr(blkLpIdx);
        UpdateUbAxes(blkLpIdx);
        CopyDataIn(bufId);
        GetOutLoopAxes();
        GatherData(bufId);
        CopyDataOut(bufId);
    }
}

template <typename T>
__aicore__ inline void TransposeGather<T>::GetCoreLoopRange()
{
    blkBeg_ = blockIdx_ * td_->blkFactor;
    if (blockIdx_ != static_cast<int64_t>(GetBlockNum()) - 1) {
        blkEnd_ = blkBeg_ + td_->blkFactor;
    } else {
        blkEnd_ = blkBeg_ + td_->blkTailFactor;
    }
}

template <typename T>
__aicore__ inline void TransposeGather<T>::InitAxes()
{
    for (int8_t i = 0; i < td_->ubAxesCnt; ++i) {
        inUbAxes_[i] = td_->inUbAxes[i];
        outUbAxes_[i] = td_->outUbAxes[i];
    }
    if (td_->blkInUbCutPos != -1) {
        blkInCutROffset_ = CalcBlkAxesOffset(td_->blkInUbCutPos);
        blkInCutAxisSize_ = td_->blkAxes[td_->blkInUbCutPos];
    }
    if (td_->blkOutUbCutPos != -1) {
        blkOutCutROffset_ = CalcBlkAxesOffset(td_->blkOutUbCutPos);
        blkOutCutAxisSize_ = td_->blkAxes[td_->blkOutUbCutPos];
    }
}

template <typename T>
__aicore__ inline void TransposeGather<T>::UpdateUbAxes(int64_t blkLpIdx)
{
    bool isInSwitch = false;
    bool isOutSwitch = false;

    if (td_->blkInUbCutPos != -1) {
        if (blkLpIdx / blkInCutROffset_ % blkInCutAxisSize_ == blkInCutAxisSize_ - 1) {
            inUbAxes_[td_->inUbInCutPos] = inUbTailFactor_;
            outUbAxes_[td_->outUbInCutPos] = inUbTailFactor_;
            isInSwitch = true;
        } else {
            inUbAxes_[td_->inUbInCutPos] = td_->inUbCutAxisFactor;
            outUbAxes_[td_->outUbInCutPos] = td_->inUbCutAxisFactor;
            isInSwitch = false;
        }
    }
    if (td_->blkOutUbCutPos != -1) {
        if (blkLpIdx / blkOutCutROffset_ % blkOutCutAxisSize_ == blkOutCutAxisSize_ - 1) {
            inUbAxes_[td_->inUbOutCutPos] = outUbTailFactor_;
            outUbAxes_[td_->outUbOutCutPos] = outUbTailFactor_;
            isOutSwitch = true;
        } else {
            inUbAxes_[td_->inUbOutCutPos] = td_->outUbCutAxisFactor;
            outUbAxes_[td_->outUbOutCutPos] = td_->outUbCutAxisFactor;
            isOutSwitch = false;
        }
    }

    if (!isOutSwitch) {
        if (!isInSwitch) {
            gIndexId_ = 0;
        } else {
            gIndexId_ = 1;
        }
    } else {
        if (!isInSwitch) {
            gIndexId_ = NUM_TWO;
        } else {
            gIndexId_ = NUM_THREE;
        }
    }
}

template <typename T>
__aicore__ inline int32_t TransposeGather<T>::CalcUbAxesInOffset(int8_t begIdx)
{
    int32_t res = 1;

    // DataCopyPad blockCount and blockLen are consecutive
    if (begIdx + 1 >= td_->inUbInCutPos) {
        for (int8_t i = begIdx + 1; i < td_->ubAxesCnt; ++i) {
            res *= inUbAxes_[i];
        }
    } else {
        for (int8_t i = td_->inUbInCutPos - 1; i < td_->ubAxesCnt; ++i) {
            res *= inUbAxes_[i];
        }
        // DataCopyPad loop2, loop1 are block align
        res = Ops::Base::CeilAlign(res, elemPerBlock_);
        for (int8_t j = begIdx + 1; j < td_->inUbInCutPos - 1; ++j) {
            res *= inUbAxes_[j];
        }
    }
    return res;
}

template <typename T>
__aicore__ inline int32_t TransposeGather<T>::CalcUbAxesOutOffset(int8_t begIdx)
{
    int32_t res = 1;
    for (int8_t i = begIdx + 1; i < td_->ubAxesCnt; ++i) {
        res *= outUbAxes_[i];
    }
    return res;
}

template <typename T>
__aicore__ inline int64_t TransposeGather<T>::CalcBlkAxesOffset(int8_t begIdx)
{
    int32_t res = 1;
    for (int8_t i = begIdx + 1; i < td_->blkAxesCnt; ++i) {
        res *= td_->blkAxes[i];
    }
    return res;
}

/**
 * @brief 执行 Gather 硬件转置操作
 *
 * 使用 RegBase 路径的 Gather 指令，根据预计算的索引从 UB_in 中
 * 按任意跨步读取数据写入 UB_out，实现尾轴转置。
 *
 * 算法步骤：
 * 1. 获取输入/输出/索引的 UB 物理地址
 * 2. 按 gIndexId_ 选择对应的 Gather 索引偏移
 * 3. 按 burstLpCnt 分批处理（每批 vlSize_ 个元素）
 * 4. 对每批数据，按 outUbAxis2/1/0 三层循环展开：
 *    a. 计算当前轴的偏移更新值 idxUpdate
 *    b. Reg::Adds(idxReg, idxOriReg, idxUpdate) 更新索引
 *    c. Gather(xReg, xInAddr, idxReg, mask) 按索引读取
 *    d. Reg::DataCopy 写入输出 UB
 * 5. 8bit 类型特殊处理：使用 DIST_PACK_B16 存储模式
 */
template <typename T>
__aicore__ inline void TransposeGather<T>::GatherData(uint32_t bufId)
{
    uint32_t outLen = static_cast<uint32_t>(CalcUbAxesOutOffset(td_->outUbOutCutPos - 1));
    uint32_t outLenAlign = Ops::Base::CeilAlign(outLen, static_cast<uint32_t>(elemPerBlock_));
    uint16_t burstLpCnt = Ops::Base::CeilDiv(outLen, vlSize_);
    uint32_t maskValue = outLen;
    if constexpr (sizeof(T) == sizeof(int8_t)) {
        maskValue *= sizeof(int16_t);
    }

    LocalTensor<T> xInLocal = xInLocal_[bufId];
    LocalTensor<T> xOutLocal = xOutLocal_[bufId];
    __ubuf__ T* xInAddr = (__ubuf__ T*)xInLocal.GetPhyAddr();
    __ubuf__ T* xOutAddr = (__ubuf__ T*)xOutLocal.GetPhyAddr();
    __ubuf__ RangeType_* idxAddr = (__ubuf__ RangeType_*)idxLocal_.GetPhyAddr();
    idxAddr += static_cast<uint32_t>(gIndexId_ * gIdxOffset_);

    // V 转置：mIn 门控本轮搬入完成（buf 复用由 MTE2/V 同一 mutex 保证），
    // mOut 门控外输出缓冲空闲（上一轮 MTE3 已搬出归还）。
    const MutexID mIn = static_cast<MutexID>(bufId);
    const MutexID mOut = static_cast<MutexID>(BUFFER_NUM + bufId);
    Mutex::Lock<PIPE_V>(mIn);
    Mutex::Lock<PIPE_V>(mOut);
    __VEC_SCOPE__
    {
        RegTensor<T> xReg;
        RegTensor<RangeType_> idxReg;
        RegTensor<RangeType_> idxOriReg;
        MaskReg maskIdx = CreateMask<RangeType_, Reg::MaskPattern::ALL>();
        MaskReg mask;

        for (uint16_t lpIdx = 0; lpIdx < burstLpCnt; ++lpIdx) {
            mask = UpdateMask<T>(maskValue);
            Reg::LoadAlign(idxOriReg, idxAddr + lpIdx * vlSize_);
            uint32_t outIdx = 0;
            for (uint16_t axis2Idx = 0; axis2Idx < static_cast<uint16_t>(outUbAxis2_); ++axis2Idx) {
                RangeType_ axis2Update = static_cast<RangeType_>(axis2Idx * outUbAxis2InROffset_);
                for (uint16_t axis1Idx = 0; axis1Idx < static_cast<uint16_t>(outUbAxis1_); ++axis1Idx) {
                    RangeType_ axis1Update = static_cast<RangeType_>(axis1Idx * outUbAxis1InROffset_);
                    for (uint16_t axis0Idx = 0; axis0Idx < static_cast<uint16_t>(outUbAxis0_); ++axis0Idx) {
                        RangeType_ axis0Update = static_cast<RangeType_>(axis0Idx * outUbAxis0InROffset_);
                        RangeType_ idxUpdate = axis2Update + axis1Update + axis0Update;
                        Reg::Adds(idxReg, idxOriReg, idxUpdate, maskIdx);
                        Reg::Gather((Reg::RegTensor<CastType_>&)xReg, xInAddr, (Reg::RegTensor<IdxType_>&)idxReg, mask);
                        if constexpr (sizeof(T) != sizeof(int8_t)) {
                            Reg::StoreAlign(xOutAddr + outIdx * outLenAlign + lpIdx * vlSize_, xReg, mask);
                        } else {
                            __ubuf__ CastType_* xOutAddrB16 = reinterpret_cast<__ubuf__ CastType_*>(
                                xOutAddr + outIdx * outLenAlign + lpIdx * vlSize_);
                            Reg::StoreAlign<CastType_, Reg::StoreDist::DIST_PACK_B16>(
                                xOutAddrB16, (Reg::RegTensor<CastType_>&)xReg, mask);
                        }
                        ++outIdx;
                    }
                }
            }
        }
    }
    Mutex::Unlock<PIPE_V>(mIn);  // 输入缓冲读完即归还，MTE2(i+2) 可继续搬入
    Mutex::Unlock<PIPE_V>(mOut); // 输出缓冲写完，MTE3(i) 可搬出
}

template <typename T>
__aicore__ inline void TransposeGather<T>::GenIndex4OneDim(int32_t goffset)
{
    int8_t lastPerm = td_->ubPerm[td_->ubAxesCnt - 1];
    uint32_t lastDimSize = static_cast<uint32_t>(outUbAxes_[td_->ubAxesCnt - 1]);
    RangeType_ lastDimInOffset = static_cast<RangeType_>(CalcUbAxesInOffset(lastPerm));

    __ubuf__ RangeType_* idxAddr = (__ubuf__ RangeType_*)idxLocal_.GetPhyAddr();
    idxAddr += goffset;
    uint16_t loopCnt = static_cast<uint16_t>(Ops::Base::CeilDiv(lastDimSize, idxVLSize_));

    __VEC_SCOPE__
    {
        RegTensor<RangeType_> srcReg;
        RegTensor<RangeType_> dstReg;
        MaskReg mask;
        Reg::Arange(srcReg, 0);
        for (uint16_t lpIdx = 0; lpIdx < loopCnt; ++lpIdx) {
            mask = UpdateMask<RangeType_>(lastDimSize);
            Reg::Muls(dstReg, srcReg, lastDimInOffset, mask);
            Reg::StoreAlign(idxAddr + lpIdx * idxVLSize_, dstReg, mask);
            Reg::Adds(srcReg, srcReg, idxVLSize_, mask);
        }
    }
}

template <typename T>
__aicore__ inline void TransposeGather<T>::GenIndex4TwoDim(int32_t goffset)
{
    int8_t lastPerm = td_->ubPerm[td_->ubAxesCnt - 1];
    uint32_t lastDimSize = static_cast<uint32_t>(outUbAxes_[td_->ubAxesCnt - 1]);
    RangeType_ lastDimInOffset = static_cast<RangeType_>(CalcUbAxesInOffset(lastPerm));
    int8_t last2ndPerm = td_->ubPerm[td_->ubAxesCnt - NUM_TWO];
    uint32_t last2ndDimSize = static_cast<uint32_t>(outUbAxes_[td_->ubAxesCnt - NUM_TWO]);
    RangeType_ last2ndDimInOffset = static_cast<RangeType_>(CalcUbAxesInOffset(last2ndPerm));
    auto totalDimSize = lastDimSize * last2ndDimSize;
    RangeType_ lastDimInc = static_cast<RangeType_>(idxVLSize_ % lastDimSize);
    RangeType_ last2ndDimInc = static_cast<RangeType_>(idxVLSize_ / lastDimSize);

    __ubuf__ RangeType_* idxAddr = (__ubuf__ RangeType_*)idxLocal_.GetPhyAddr();
    idxAddr += goffset;
    uint16_t loopCnt = 0;
    uint32_t leftDimSize = 0;
    if (totalDimSize > idxVLSize_) {
        leftDimSize = totalDimSize - idxVLSize_;
        loopCnt = static_cast<uint16_t>(Ops::Base::CeilDiv(leftDimSize, idxVLSize_));
    }

    /*   input: (a, b, c, d)
     *  output: (d, c, b, a)
     *  generate index for a, b
     */
    __VEC_SCOPE__
    {
        RegTensor<RangeType_> idxReg;
        RegTensor<RangeType_> dim0Reg;
        RegTensor<RangeType_> tmpReg;
        RegTensor<RangeType_> dim1Reg;
        RegTensor<RangeType_> dstReg;
        MaskReg mask = CreateMask<RangeType_, Reg::MaskPattern::ALL>();
        // vec_a: VL % a
        Reg::Arange(idxReg, 0);
        Reg::Duplicate(dim0Reg, lastDimSize);
        Reg::Div(tmpReg, idxReg, dim0Reg, mask);
        Reg::Move(dim1Reg, tmpReg); // vec_b: VL / a
        Reg::Mul(tmpReg, tmpReg, dim0Reg, mask);
        Reg::Sub(dim0Reg, idxReg, tmpReg, mask);
        // index: vec_a * a_in_offset + vec_b * b_in_offset
        Reg::Muls(tmpReg, dim0Reg, lastDimInOffset, mask);
        Reg::Muls(dstReg, dim1Reg, last2ndDimInOffset, mask);
        Reg::Add(dstReg, dstReg, tmpReg, mask);
        Reg::StoreAlign(idxAddr, dstReg, mask);

        MaskReg lpMask;
        MaskReg selMask;
        RegTensor<RangeType_> zeroReg;
        RegTensor<RangeType_> oneReg;
        RegTensor<RangeType_> cmpReg;
        Reg::Duplicate(zeroReg, 0);
        Reg::Duplicate(oneReg, 1);
        for (uint16_t lpIdx = 0; lpIdx < loopCnt; ++lpIdx) {
            lpMask = UpdateMask<RangeType_>(leftDimSize);
            /*   vec_a += a_inc
             *   cmp_a = vec_a >= a
             *   vec_a = vec_a - cmp_a * a
             */
            Reg::Adds(dim0Reg, dim0Reg, lastDimInc, lpMask);
            Reg::Compares<RangeType_, CMPMODE::GE>(selMask, dim0Reg, lastDimSize, lpMask);
            Reg::Select(cmpReg, oneReg, zeroReg, selMask);
            Reg::Muls(tmpReg, cmpReg, lastDimSize, lpMask);
            Reg::Sub(dim0Reg, dim0Reg, tmpReg, lpMask);
            // vec_b += (b_inc + cmp_a)
            Reg::Adds(dim1Reg, dim1Reg, last2ndDimInc, lpMask);
            Reg::Add(dim1Reg, dim1Reg, cmpReg, lpMask);
            // index: vec_a * a_in_offset + vec_b * b_in_offset
            Reg::Muls(tmpReg, dim0Reg, lastDimInOffset, lpMask);
            Reg::Muls(dstReg, dim1Reg, last2ndDimInOffset, lpMask);
            Reg::Add(dstReg, dstReg, tmpReg, lpMask);
            Reg::StoreAlign(idxAddr + (lpIdx + 1) * idxVLSize_, dstReg, lpMask);
        }
    }
}

template <typename T>
__aicore__ inline void TransposeGather<T>::GenIndex4ThreeDim(int32_t goffset)
{
    int8_t lastPerm = td_->ubPerm[td_->ubAxesCnt - 1];
    uint32_t lastDimSize = static_cast<uint32_t>(outUbAxes_[td_->ubAxesCnt - 1]);
    RangeType_ lastDimInOffset = static_cast<RangeType_>(CalcUbAxesInOffset(lastPerm));
    int8_t last2ndPerm = td_->ubPerm[td_->ubAxesCnt - NUM_TWO];
    uint32_t last2ndDimSize = static_cast<uint32_t>(outUbAxes_[td_->ubAxesCnt - NUM_TWO]);
    RangeType_ last2ndDimInOffset = static_cast<RangeType_>(CalcUbAxesInOffset(last2ndPerm));
    int8_t last3rdPerm = td_->ubPerm[td_->ubAxesCnt - NUM_THREE];
    uint32_t last3rdDimSize = static_cast<uint32_t>(outUbAxes_[td_->ubAxesCnt - NUM_THREE]);
    RangeType_ last3rdDimInOffset = static_cast<RangeType_>(CalcUbAxesInOffset(last3rdPerm));
    auto totalDimSize = lastDimSize * last2ndDimSize * last3rdDimSize;
    RangeType_ lastDimInc = static_cast<RangeType_>(idxVLSize_ % lastDimSize);
    RangeType_ last2ndDimInc = static_cast<RangeType_>(idxVLSize_ / lastDimSize % last2ndDimSize);
    RangeType_ last3rdDimInc = static_cast<RangeType_>(idxVLSize_ / (lastDimSize * last2ndDimSize));

    __ubuf__ RangeType_* idxAddr = (__ubuf__ RangeType_*)idxLocal_.GetPhyAddr();
    idxAddr += goffset;
    uint16_t loopCnt = 0;
    uint32_t leftDimSize = 0;
    if (totalDimSize > idxVLSize_) {
        leftDimSize = totalDimSize - idxVLSize_;
        loopCnt = static_cast<uint16_t>(Ops::Base::CeilDiv(leftDimSize, idxVLSize_));
    }

    /*   input: (a, b, c, d)
     *  output: (d, c, b, a)
     *  generate index for a, b, c
     */
    __VEC_SCOPE__
    {
        RegTensor<RangeType_> idxReg;
        RegTensor<RangeType_> dim0Reg;
        RegTensor<RangeType_> tmpReg;
        RegTensor<RangeType_> dim1Reg;
        RegTensor<RangeType_> tmp1Reg;
        RegTensor<RangeType_> dim2Reg;
        RegTensor<RangeType_> dstReg;
        MaskReg mask = CreateMask<RangeType_, Reg::MaskPattern::ALL>();
        // vec_a: VL % a
        Reg::Arange(idxReg, 0);
        Reg::Duplicate(dim0Reg, lastDimSize);
        Reg::Move(dim2Reg, dim0Reg); // backup a
        Reg::Div(tmpReg, idxReg, dim0Reg, mask);
        Reg::Move(dim1Reg, tmpReg); // backup VL / a
        Reg::Mul(tmpReg, tmpReg, dim0Reg, mask);
        Reg::Sub(dim0Reg, idxReg, tmpReg, mask);
        // vec_b: VL / a % b
        Reg::Duplicate(tmp1Reg, last2ndDimSize);
        Reg::Mul(dim2Reg, dim2Reg, tmp1Reg, mask); // backup b
        Reg::Div(tmpReg, dim1Reg, tmp1Reg, mask);
        Reg::Mul(tmpReg, tmpReg, tmp1Reg, mask);
        Reg::Sub(dim1Reg, dim1Reg, tmpReg, mask);
        // vec_c: VL / (a * b)
        Reg::Div(dim2Reg, idxReg, dim2Reg, mask);
        // index: vec_a * a_in_offset + vec_b * b_in_offset + vec_c * c_in_offset
        Reg::Muls(tmpReg, dim0Reg, lastDimInOffset, mask);
        Reg::Muls(tmp1Reg, dim1Reg, last2ndDimInOffset, mask);
        Reg::Muls(dstReg, dim2Reg, last3rdDimInOffset, mask);
        Reg::Add(dstReg, dstReg, tmpReg, mask);
        Reg::Add(dstReg, dstReg, tmp1Reg, mask);
        Reg::StoreAlign(idxAddr, dstReg, mask);

        MaskReg lpMask;
        MaskReg selMask;
        RegTensor<RangeType_> zeroReg;
        RegTensor<RangeType_> oneReg;
        RegTensor<RangeType_> cmpReg;
        Reg::Duplicate(zeroReg, 0);
        Reg::Duplicate(oneReg, 1);
        for (uint16_t lpIdx = 0; lpIdx < loopCnt; ++lpIdx) {
            lpMask = UpdateMask<RangeType_>(leftDimSize);
            /*   vec_a += a_inc
             *   cmp_a = vec_a >= a
             *   vec_a = vec_a - cmp_a * a
             */
            Reg::Adds(dim0Reg, dim0Reg, lastDimInc, lpMask);
            Reg::Compares<RangeType_, CMPMODE::GE>(selMask, dim0Reg, lastDimSize, lpMask);
            Reg::Select(cmpReg, oneReg, zeroReg, selMask);
            Reg::Muls(tmpReg, cmpReg, lastDimSize, lpMask);
            Reg::Sub(dim0Reg, dim0Reg, tmpReg, lpMask);
            /*   vec_b += (b_inc + cmp_a)
             *   cmp_b = vec_b >= b
             *   vec_b = vec_b - cmp_b * b
             */
            Reg::Adds(cmpReg, cmpReg, last2ndDimInc, lpMask);
            Reg::Add(dim1Reg, dim1Reg, cmpReg, lpMask);
            Reg::Compares<RangeType_, CMPMODE::GE>(selMask, dim1Reg, last2ndDimSize, lpMask);
            Reg::Select(cmpReg, oneReg, zeroReg, selMask);
            Reg::Muls(tmpReg, cmpReg, last2ndDimSize, lpMask);
            Reg::Sub(dim1Reg, dim1Reg, tmpReg, lpMask);
            // vec_c += (c_inc + cmp_b)
            Reg::Adds(dim2Reg, dim2Reg, last3rdDimInc, lpMask);
            Reg::Add(dim2Reg, dim2Reg, cmpReg, lpMask);
            // index: vec_a * a_in_offset + vec_b * b_in_offset + vec_c * c_in_offset
            Reg::Muls(tmpReg, dim0Reg, lastDimInOffset, lpMask);
            Reg::Muls(tmp1Reg, dim1Reg, last2ndDimInOffset, lpMask);
            Reg::Muls(dstReg, dim2Reg, last3rdDimInOffset, lpMask);
            Reg::Add(dstReg, dstReg, tmpReg, lpMask);
            Reg::Add(dstReg, dstReg, tmp1Reg, lpMask);
            Reg::StoreAlign(idxAddr + (lpIdx + 1) * idxVLSize_, dstReg, lpMask);
        }
    }
}

template <typename T>
__aicore__ inline void TransposeGather<T>::GenGatherIndex(int32_t goffset)
{
    int8_t dimNum = td_->ubAxesCnt - td_->outUbOutCutPos;
    // max borrowed axes count is 3
    if (dimNum == 1) {
        GenIndex4OneDim(goffset);
    } else if (dimNum == NUM_TWO) {
        GenIndex4TwoDim(goffset);
    } else {
        GenIndex4ThreeDim(goffset);
    }
}

template <typename T>
__aicore__ inline void TransposeGather<T>::GenGatherIndex4AllPhase()
{
    GenGatherIndex(0);
    gIdxOffset_ = Ops::Base::CeilAlign(CalcUbAxesOutOffset(td_->outUbOutCutPos - 1), gElemPerBlock_);
    inUbTailFactor_ = td_->inUbCutAxisSize % td_->inUbCutAxisFactor;
    outUbTailFactor_ = td_->outUbCutAxisSize % td_->outUbCutAxisFactor;
    if (td_->blkOutUbCutPos == -1 && td_->blkInUbCutPos != -1) {
        inUbAxes_[td_->inUbInCutPos] = inUbTailFactor_;
        outUbAxes_[td_->outUbInCutPos] = inUbTailFactor_;
        GenGatherIndex(gIdxOffset_);
    } else if (td_->blkOutUbCutPos != -1 && td_->blkInUbCutPos == -1) {
        inUbAxes_[td_->inUbOutCutPos] = outUbTailFactor_;
        outUbAxes_[td_->outUbOutCutPos] = outUbTailFactor_;
        GenGatherIndex(gIdxOffset_ * NUM_TWO);
    } else if (td_->blkOutUbCutPos != -1 && td_->blkInUbCutPos != -1) {
        inUbAxes_[td_->inUbInCutPos] = inUbTailFactor_;
        outUbAxes_[td_->outUbInCutPos] = inUbTailFactor_;
        GenGatherIndex(gIdxOffset_);
        inUbAxes_[td_->inUbOutCutPos] = outUbTailFactor_;
        outUbAxes_[td_->outUbOutCutPos] = outUbTailFactor_;
        inUbAxes_[td_->inUbInCutPos] = td_->inUbCutAxisFactor;
        outUbAxes_[td_->outUbInCutPos] = td_->inUbCutAxisFactor;
        GenGatherIndex(gIdxOffset_ * NUM_TWO);
        inUbAxes_[td_->inUbInCutPos] = inUbTailFactor_;
        outUbAxes_[td_->outUbInCutPos] = inUbTailFactor_;
        GenGatherIndex(gIdxOffset_ * NUM_THREE);
    }
    // recover ub axes for next process
    inUbAxes_[td_->inUbOutCutPos] = td_->outUbCutAxisFactor;
    inUbAxes_[td_->inUbInCutPos] = td_->inUbCutAxisFactor;
    outUbAxes_[td_->outUbInCutPos] = td_->inUbCutAxisFactor;
    outUbAxes_[td_->outUbOutCutPos] = td_->outUbCutAxisFactor;
}

template <typename T>
__aicore__ inline void TransposeGather<T>::CalcBlkAddr(int64_t idx)
{
    int64_t tmpIdx = 0;
    blkInAddr_ = 0;
    blkOutAddr_ = 0;
    for (int8_t i = 0; i < td_->blkAxesCnt; ++i) {
        tmpIdx = idx / CalcBlkAxesOffset(i) % td_->blkAxes[i];
        blkInAddr_ += tmpIdx * td_->blkAxesInAOffset[i];
        blkOutAddr_ += tmpIdx * td_->blkAxesOutAOffset[i];
    }
}

template <typename T>
__aicore__ inline void TransposeGather<T>::SetCopyInParams()
{
    int32_t inUbAxis0 = 1;
    int32_t inUbAxis1 = 1;
    int32_t inUbAxis2 = 1;
    if (td_->inUbInCutPos == 1) {
        inUbAxis0 = inUbAxes_[0];
    } else if (td_->inUbInCutPos == NUM_TWO) {
        inUbAxis0 = inUbAxes_[1];
        inUbAxis1 = inUbAxes_[0];
    } else if (td_->inUbInCutPos == NUM_THREE) {
        inUbAxis0 = inUbAxes_[NUM_TWO];
        inUbAxis1 = inUbAxes_[1];
        inUbAxis2 = inUbAxes_[0];
    }

    uint32_t inCube = static_cast<size_t>(CalcUbAxesInOffset(td_->inUbInCutPos - 1)) * sizeof(T);
    copyInParams_.blockCount = inUbAxis0;
    copyInParams_.blockLen = inCube;
    copyInParams_.srcStride = td_->axis0InSrcStride * sizeof(T) - inCube;
    copyInParams_.dstStride = 0;
    lpModeInParams_.loop1Size = inUbAxis1;
    lpModeInParams_.loop2Size = inUbAxis2;
    lpModeInParams_.loop1SrcStride = td_->axis1InSrcStride * sizeof(T);
    lpModeInParams_.loop2SrcStride = td_->axis2InSrcStride * sizeof(T);
    lpModeInParams_.loop1DstStride = Ops::Base::CeilAlign(
        static_cast<int64_t>(copyInParams_.blockCount * copyInParams_.blockLen), BLOCK_SIZE_BYTE);
    lpModeInParams_.loop2DstStride = lpModeInParams_.loop1Size * lpModeInParams_.loop1DstStride;
}

template <typename T>
__aicore__ inline void TransposeGather<T>::CopyDataIn(uint32_t bufId)
{
    SetCopyInParams();
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    LocalTensor<T> xInLocal = xInLocal_[bufId];
    const MutexID mIn = static_cast<MutexID>(bufId);
    // mIn 门控 inbuf 复用：上一轮读者 V(i-2) 已归还才可重新搬入
    Mutex::Lock<PIPE_MTE2>(mIn);
    SetLoopModePara(lpModeInParams_, DataCopyMVType::OUT_TO_UB);
#if defined(__DAV_920R1__) || defined(__DAV_920R2__)
    DataCopyPad<T, PaddingMode::Compact>(xInLocal, xGM_[blkInAddr_], copyInParams_, padParams, true);
#else
    DataCopyPad<T, PaddingMode::Compact>(xInLocal, xGM_[blkInAddr_], copyInParams_, padParams);
#endif
    ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    Mutex::Unlock<PIPE_MTE2>(mIn);
}

template <typename T>
__aicore__ inline void TransposeGather<T>::GetOutLoopAxes()
{
    if (td_->outUbOutCutPos == 1) {
        outUbAxis0_ = outUbAxes_[0];
        outUbAxis0InROffset_ = CalcUbAxesInOffset(td_->ubPerm[0]);
    } else if (td_->outUbOutCutPos == NUM_TWO) {
        outUbAxis0_ = outUbAxes_[1];
        outUbAxis1_ = outUbAxes_[0];
        outUbAxis0InROffset_ = CalcUbAxesInOffset(td_->ubPerm[1]);
        outUbAxis1InROffset_ = CalcUbAxesInOffset(td_->ubPerm[0]);
    } else if (td_->outUbOutCutPos == NUM_THREE) {
        outUbAxis0_ = outUbAxes_[NUM_TWO];
        outUbAxis1_ = outUbAxes_[1];
        outUbAxis2_ = outUbAxes_[0];
        outUbAxis0InROffset_ = CalcUbAxesInOffset(td_->ubPerm[NUM_TWO]);
        outUbAxis1InROffset_ = CalcUbAxesInOffset(td_->ubPerm[1]);
        outUbAxis2InROffset_ = CalcUbAxesInOffset(td_->ubPerm[0]);
    }
}

template <typename T>
__aicore__ inline void TransposeGather<T>::SetCopyOutParams()
{
    uint32_t outCube = static_cast<size_t>(CalcUbAxesOutOffset(td_->outUbOutCutPos - 1)) * sizeof(T);
    uint32_t outCubeAlign = Ops::Base::CeilAlign(outCube, static_cast<uint32_t>(BLOCK_SIZE_BYTE));
    copyOutParams_.blockCount = outUbAxis0_;
    copyOutParams_.blockLen = outCube;
    copyOutParams_.srcStride = 0;
    copyOutParams_.dstStride = td_->axis0OutDstStride * sizeof(T) - outCube;
    lpModeOutParams_.loop1Size = outUbAxis1_;
    lpModeOutParams_.loop2Size = outUbAxis2_;
    lpModeOutParams_.loop1SrcStride = outUbAxis0_ * outCubeAlign;
    lpModeOutParams_.loop2SrcStride = outUbAxis1_ * lpModeOutParams_.loop1SrcStride;
    lpModeOutParams_.loop1DstStride = td_->axis1OutDstStride * sizeof(T);
    lpModeOutParams_.loop2DstStride = td_->axis2OutDstStride * sizeof(T);
}

template <typename T>
__aicore__ inline void TransposeGather<T>::CopyDataOut(uint32_t bufId)
{
    SetCopyOutParams();
    LocalTensor<T> xOutLocal = xOutLocal_[bufId];
    const MutexID mOut = static_cast<MutexID>(BUFFER_NUM + bufId);
    // mOut 门控数据就绪与 outbuf 复用：V(i) 写完才可搬出，搬出完 V(i+2) 才可再写
    Mutex::Lock<PIPE_MTE3>(mOut);
    SetLoopModePara(lpModeOutParams_, DataCopyMVType::UB_TO_OUT);
    DataCopyPad(yGM_[blkOutAddr_], xOutLocal, copyOutParams_);
    ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
    Mutex::Unlock<PIPE_MTE3>(mOut);
}

} // namespace Transpose

#endif // TRANSPOSE_CUT_TWO_AXIS
