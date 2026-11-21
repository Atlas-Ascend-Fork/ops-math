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
 * \file concat_base.h
 * \brief concat base
 */
#ifndef CONCAT_BASE_H
#define CONCAT_BASE_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "op_kernel/platform_util.h"

namespace Concat {
using namespace AscendC;
using namespace Ops::Base;
constexpr int64_t BUFFER_NUM_ZERO_AXIS = 2;
constexpr int64_t MAX_DIM_SIZE = 8;
constexpr int64_t BUFFER_NUM = 2;
constexpr int64_t INDEX_SIZE = 512;
constexpr int64_t BYTES_PER_BLOCK = GetUbBlockSize();
constexpr int64_t ARRAY_SIZE = 10;
constexpr int64_t PRELOAD_DIM1_SIZE = 2;
constexpr int32_t DIGIT_TWO = 2;
constexpr int32_t DIGIT_THREE = 3;
constexpr int32_t DIGIT_FOUR = 4;
// ===== mergedInfo 交错布局（与 op_host/arch35/concat_tiling_arch35.h 保持同步）=====
// [2t]/[2t+1] = tensor t 的 stride/concatDim（所有变体共用，基址 0）；
// 数组变体在 CORE_INFO_BASE 之后按核交错存放 {endTensorIdx, endTensorOffset}，
// 核 i 与核 i-1 的定界对相邻（连续 32B），一次窗口读取即可拿到 start/end 定界
constexpr int64_t TENSOR_PAIR_WORDS = 128; // NON_CON_TENSOR_SIZE(64) * 2
constexpr int64_t CORE_INFO_BASE = TENSOR_PAIR_WORDS;

// CONST_TILING（静态 shape 编译）时 tiling 为编译期常量（无 GM 地址），预取无意义，
// 守卫放在 DcPreloadRange 源头即可覆盖入口调用与全部 PreloadTiling 调用点
__aicore__ inline void DcPreloadRange(uint64_t tilingAddr, uint64_t startByte, uint64_t endByte)
{
#ifndef CONST_TILING
    __gm__ uint64_t* ptr = (__gm__ uint64_t*)tilingAddr;
    for (uint64_t off = startByte; off < endByte; off += 64) {
        dc_preload(ptr, static_cast<int64_t>(off));
    }
#endif
}

using SplitInfo = struct {
    int64_t startIdx;
    int64_t startOffset;
    int64_t endIdx;
    int64_t endOffset;
};

using SplitLoopParam = struct {
    int64_t startTensorIdx;
    int64_t handleTensorNum;
    int64_t perLoopDealTensorNum;
    int64_t tailLoopDealTensorNum;
    int64_t loopSize;
    int64_t tensorStride;
};

template <typename T>
struct VciTypeGet;
template <>
struct VciTypeGet<uint32_t> {
    using T = int32_t;
};
template <>
struct VciTypeGet<uint16_t> {
    using T = int16_t;
};

template <typename Tp, Tp v>
struct IntegralConstant {
    static constexpr Tp value = v;
};
using trueType = IntegralConstant<bool, true>;
using falseType = IntegralConstant<bool, false>;
template <typename, typename>
struct IsSame : public falseType {};
template <typename Tp>
struct IsSame<Tp, Tp> : public trueType {};

// 预取本核与上一核在 mergedInfo 中的定界对（一次 32B 窗口覆盖 start/end 定界）。
// CONST_TILING（静态 shape）时 tiling 编译期常量折叠，预取无意义；且 offsetof 对
// cst 生成器的 constexpr 成员形态非法，故函数体整体不参与编译。
// 各 one_axis_* 模板的 PreloadTiling 统一转发到这里，避免逐类重复实现。
template <typename TILINGDATA>
__aicore__ inline void PreloadMergedInfo(__tiling_data_ptr__ TILINGDATA* tilingData)
{
#ifndef CONST_TILING
    uint32_t blockIdx = GetBlockIdx();
    if constexpr (IsSame<TILINGDATA, ConcatTilingData>::value) {
        constexpr uint64_t coreInfoOff = offsetof(ConcatTilingData, arrays.mergedInfo) +
                                         CORE_INFO_BASE * sizeof(int64_t);
        // 本核与上一核定界对在 mergedInfo 中相邻，预取 32B 窗口（首核仅本核定界对）
        uint64_t start = blockIdx > 0 ? coreInfoOff + (blockIdx * 2 - 2) * sizeof(int64_t) : coreInfoOff;
        uint64_t len = blockIdx > 0 ? sizeof(int64_t) * 4 : sizeof(int64_t) * 2;
        DcPreloadRange((uint64_t)tilingData, start, start + len);
    } else if constexpr (IsSame<TILINGDATA, ConcatTilingDataCompact>::value) {
        constexpr uint64_t coreInfoOff = offsetof(ConcatTilingDataCompact, arrays.mergedInfo) +
                                         CORE_INFO_BASE * sizeof(uint32_t);
        uint64_t start = blockIdx > 0 ? coreInfoOff + (blockIdx * 2 - 2) * sizeof(uint32_t) : coreInfoOff;
        uint64_t len = blockIdx > 0 ? sizeof(uint32_t) * 4 : sizeof(uint32_t) * 2;
        DcPreloadRange((uint64_t)tilingData, start, start + len);
    }
#endif
}

template <typename T>
__aicore__ inline void Copy(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, uint32_t rows,
                            uint32_t cols, uint32_t rowStride)
{
    constexpr uint32_t vfLen = GetVRegSize() / sizeof(T);
    uint16_t size0 = rows;
    uint16_t size1 = cols / vfLen;
    uint32_t tail = cols - size1 * vfLen;
    uint32_t main = vfLen;

    auto dstAddr = (__ubuf__ T*)dstLocal.GetPhyAddr();
    auto srcAddr = (__ubuf__ T*)srcLocal.GetPhyAddr();

    __VEC_SCOPE__
    {
        AscendC::Reg::UnalignRegForLoad u0;
        AscendC::Reg::UnalignRegForStore uReg;
        AscendC::Reg::RegTensor<T> vd0;
        AscendC::Reg::LoadUnAlignPre(u0, srcAddr);
        for (uint16_t i = 0; i < size0; i++) {
            auto curDstAddr = dstAddr + i * rowStride;
            for (uint16_t j = 0; j < size1; j++) {
                AscendC::Reg::LoadUnAlign(vd0, u0, srcAddr, main);
                AscendC::Reg::StoreUnAlign(curDstAddr, vd0, uReg, main);
            }
            AscendC::Reg::LoadUnAlign(vd0, u0, srcAddr, tail);
            AscendC::Reg::StoreUnAlign(curDstAddr, vd0, uReg, tail);
            AscendC::Reg::StoreUnAlignPost(curDstAddr, uReg, 0);
        }
    }
}

template <typename T, typename U, int32_t ScatterNum>
__aicore__ inline void ScatterConcat(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, uint32_t rows,
                                     uint32_t cols, uint32_t rowStride)
{
    constexpr uint32_t vfLen = GetVRegSize() / sizeof(U);
    uint16_t size0 = rows;
    uint32_t tail = cols - cols / vfLen * vfLen;
    if (tail == 0) {
        tail = vfLen;
    }
    uint32_t main = vfLen;

    auto dstAddr = (__ubuf__ T*)dstLocal.GetPhyAddr();
    auto srcAddr = (__ubuf__ T*)srcLocal.GetPhyAddr();

    __VEC_SCOPE__
    {
        AscendC::Reg::UnalignRegForLoad u0;
        AscendC::Reg::RegTensor<T> gather0;
        AscendC::Reg::RegTensor<T> gather1;
        AscendC::Reg::RegTensor<T> gather2;
        AscendC::Reg::RegTensor<T> gather3;
        AscendC::Reg::RegTensor<T> dst;
        AscendC::Reg::RegTensor<U> index;
        using regType = typename VciTypeGet<U>::T;
        AscendC::Reg::RegTensor<regType> tmp;
        AscendC::Reg::Arange(tmp, 0);
        index = (AscendC::Reg::RegTensor<U>&)tmp;
        auto allVf = main;
        auto tailVf = tail;
        AscendC::Reg::MaskReg p0 = AscendC::Reg::UpdateMask<U>(allVf);
        AscendC::Reg::MaskReg p1 = AscendC::Reg::UpdateMask<U>(tailVf);
        AscendC::Reg::LoadUnAlignPre(u0, srcAddr);
        for (uint16_t i = 0; i < size0; i++) {
            auto curDstAddr = dstAddr + i * rowStride;
            if constexpr (ScatterNum > DIGIT_THREE) {
                AscendC::Reg::LoadUnAlign(gather3, u0, srcAddr, main);
                if constexpr (sizeof(T) == 1) {
                    AscendC::Reg::UnPack((AscendC::Reg::RegTensor<uint16_t>&)dst, gather3);
                    AscendC::Reg::Scatter(curDstAddr, dst, index, p0);
                } else {
                    AscendC::Reg::Scatter(curDstAddr, gather3, index, p0);
                }
                curDstAddr = curDstAddr + main;
            }
            if constexpr (ScatterNum > DIGIT_TWO) {
                AscendC::Reg::LoadUnAlign(gather2, u0, srcAddr, main);
                if constexpr (sizeof(T) == 1) {
                    AscendC::Reg::UnPack((AscendC::Reg::RegTensor<uint16_t>&)dst, gather2);
                    AscendC::Reg::Scatter(curDstAddr, dst, index, p0);
                } else {
                    AscendC::Reg::Scatter(curDstAddr, gather2, index, p0);
                }
                curDstAddr = curDstAddr + main;
            }
            if constexpr (ScatterNum > 1) {
                AscendC::Reg::LoadUnAlign(gather1, u0, srcAddr, main);
                if constexpr (sizeof(T) == 1) {
                    AscendC::Reg::UnPack((AscendC::Reg::RegTensor<uint16_t>&)dst, gather1);
                    AscendC::Reg::Scatter(curDstAddr, dst, index, p0);
                } else {
                    AscendC::Reg::Scatter(curDstAddr, gather1, index, p0);
                }
                curDstAddr = curDstAddr + main;
            }
            AscendC::Reg::LoadUnAlign(gather0, u0, srcAddr, tail);
            if constexpr (sizeof(T) == 1) {
                AscendC::Reg::UnPack((AscendC::Reg::RegTensor<uint16_t>&)dst, gather0);
                AscendC::Reg::Scatter(curDstAddr, dst, index, p1);
            } else {
                AscendC::Reg::Scatter(curDstAddr, gather0, index, p1);
            }
        }
    }
}

template <typename TILINGDATA>
__aicore__ inline int64_t GetTensorDim0Stride(__tiling_data_ptr__ TILINGDATA* tilingData_, int64_t idx, int64_t dim1)
{
    int64_t dim0stride = dim1;
    if (tilingData_->isNonContiguous) {
        dim0stride = static_cast<int64_t>(tilingData_->arrays.mergedInfo[2 * idx]);
    }
    return dim0stride;
}

template <typename TILINGDATA, typename T>
__aicore__ inline int64_t GetNonConDimSize(__tiling_data_ptr__ TILINGDATA* tilingData_, int64_t idx,
                                           ListTensorDesc inputList_, TensorDesc<T> desc_)
{
    int64_t concatDimSize_ = 0;
    if (tilingData_->isNonContiguous) {
        concatDimSize_ = static_cast<int64_t>(tilingData_->arrays.mergedInfo[2 * idx + 1]);
    } else {
        inputList_.GetDesc(desc_, idx);
        concatDimSize_ = desc_.GetShape(tilingData_->dim);
    }
    return concatDimSize_;
}
} // namespace Concat
#endif
