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
 * \file strided_slice_assign_v2_hybrid.h
 * \brief strided_slice_assign_v2_hybrid.h
 */

#ifndef STRIDED_SLICE_ASSIGN_V2_HYBRID_H_
#define STRIDED_SLICE_ASSIGN_V2_HYBRID_H_

#include <cstdint>

#include "kernel_operator.h"
#include "simt_api/common_functions.h"
#include "simt_api/asc_simt.h"
#include "strided_slice_assign_v2_tiling_data.h"

namespace StridedSliceAssignV2HybridNs {

using namespace AscendC;

constexpr int32_t SSA_SIMT_THREAD_NUM = 1024; // VF 线程数（__launch_bounds__ 寄存器预算）
constexpr int32_t SSA_BUFFER_NUM = 2;         // inputQueue 双缓冲槽位数
constexpr int32_t SSA_PARAM_BUF_BYTES = 512;  // paramBuf 常驻参数区字节预算（与 Host 侧一致）

// UB 参数区布局（int64 槽）：varBase | sliceShape | axisStep | magic | shift
constexpr uint32_t SSA_UB_VAR_BASE_OFF = 0;
constexpr uint32_t SSA_UB_SLICE_SHAPE_OFF = 1;
constexpr uint32_t SSA_UB_AXIS_STEP_OFF = 1 + SSA_MAX_DIM_NUM;
constexpr uint32_t SSA_UB_MAGIC_OFF = 1 + 2 * SSA_MAX_DIM_NUM;
constexpr uint32_t SSA_UB_SHIFT_OFF = 1 + 3 * SSA_MAX_DIM_NUM;
constexpr uint32_t SSA_UB_TOTAL_ELEMS = 1 + 4 * SSA_MAX_DIM_NUM;
static_assert(SSA_UB_TOTAL_ELEMS * static_cast<int64_t>(sizeof(int64_t)) <= SSA_PARAM_BUF_BYTES,
              "param area must fit in the 512B budget reserved by host tiling");

// 绝对 flat 混基反解（radix = sliceShape[j]，行主序自最内维剥起）：
//   q = tmp / len 经 magic/shift 快除（Simt::UintDiv 无符号域，操作数恒非负），模由乘减 coord = tmp − q·len
//   varAddr += coord·axisStep[j]，带符号步长直接累加，全正 / 全负 / 混合三方向天然支持
template <typename T>
__simt_vf__ __aicore__ __launch_bounds__(SSA_SIMT_THREAD_NUM) inline void HybridSimtBody(
    int64_t dimNum, int64_t flatBase, int64_t curElems, __ubuf__ const int64_t* paramUb, __ubuf__ const T* inUb,
    __gm__ T* varOut)
{
    // 单 block 发射：VF 内 blockIdx.x 继承 kernel 级上下文而非 0，起点/步进只可用 threadIdx.x
    for (int64_t localFlat = threadIdx.x; localFlat < curElems; localFlat += SSA_SIMT_THREAD_NUM) {
        int64_t tmp = flatBase + localFlat;
        int64_t varAddr = paramUb[SSA_UB_VAR_BASE_OFF];
        for (int64_t j = dimNum - 1; j >= 0; --j) {
            const int64_t len = paramUb[SSA_UB_SLICE_SHAPE_OFF + j];
            const int64_t magic = paramUb[SSA_UB_MAGIC_OFF + j];
            const int64_t shift = paramUb[SSA_UB_SHIFT_OFF + j];
            const int64_t q = static_cast<int64_t>(Simt::UintDiv<uint64_t>(
                static_cast<uint64_t>(tmp), static_cast<uint64_t>(magic), static_cast<uint64_t>(shift)));
            const int64_t coord = tmp - q * len;
            varAddr += coord * paramUb[SSA_UB_AXIS_STEP_OFF + j];
            tmp = q;
        }
        varOut[varAddr] = inUb[localFlat];
    }
}

// magic/shift 按 uint64 位宽预计算，与线程侧 UintDiv<uint64_t> 实例一致（位宽不配对会算错）
__aicore__ inline void PrepareParamBuf(__ubuf__ int64_t* paramUb,
                                       const StridedSliceAssignV2HybridTilingData* tilingData)
{
    paramUb[SSA_UB_VAR_BASE_OFF] = tilingData->varBase;
    for (int64_t d = 0; d < SSA_MAX_DIM_NUM; d++) {
        paramUb[SSA_UB_SLICE_SHAPE_OFF + d] = tilingData->sliceShape[d];
        paramUb[SSA_UB_AXIS_STEP_OFF + d] = tilingData->axisStep[d];
        paramUb[SSA_UB_MAGIC_OFF + d] = 0;
        paramUb[SSA_UB_SHIFT_OFF + d] = 0;
    }
    const int64_t dimNum = tilingData->dimNum;
    for (int64_t d = 0; d < dimNum; d++) {
        const int64_t len = tilingData->sliceShape[d];
        if (len <= 0) {
            continue;
        }
        uint64_t magic = 0;
        uint64_t shift = 0;
        GetUintDivMagicAndShift<uint64_t>(magic, shift, static_cast<uint64_t>(len));
        paramUb[SSA_UB_MAGIC_OFF + d] = static_cast<int64_t>(magic);
        paramUb[SSA_UB_SHIFT_OFF + d] = static_cast<int64_t>(shift);
    }
}

template <typename T>
class StridedSliceAssignV2Hybrid {
public:
    __aicore__ inline explicit StridedSliceAssignV2Hybrid(TPipe* pipe) : Ppipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR var, GM_ADDR inputValue, GM_ADDR varOut,
                                const StridedSliceAssignV2HybridTilingData* tilingData)
    {
        dimNum_ = tilingData->dimNum;
        sliceElemNum_ = tilingData->sliceElemNum;
        blockDim_ = tilingData->blockDim;
        elemPerCore_ = tilingData->elemPerCore;
        lastCoreElemNum_ = tilingData->lastCoreElemNum;
        ubFactor_ = tilingData->ubFactor;

        (void)var; // in-place：varOut ≡ var，var 不读
        inputValueGm_.SetGlobalBuffer((__gm__ T*)inputValue);
        varOutGm_.SetGlobalBuffer((__gm__ T*)varOut);

        Ppipe_->InitBuffer(inputQueue_, SSA_BUFFER_NUM, static_cast<uint32_t>(ubFactor_ * sizeof(T)));
        Ppipe_->InitBuffer(paramBuf_, static_cast<uint32_t>(SSA_PARAM_BUF_BYTES));

        __ubuf__ int64_t* paramUb = (__ubuf__ int64_t*)paramBuf_.Get<int64_t>().GetPhyAddr();
        PrepareParamBuf(paramUb, tilingData);
        DataSyncBarrier<MemDsbT::UB>(); // scalar 写 UB → VF 读可见性

        const int64_t blk = static_cast<int64_t>(GetBlockIdx());
        coreStart_ = blk * elemPerCore_;
        coreElemNum_ = (blk == blockDim_ - 1) ? lastCoreElemNum_ : elemPerCore_;
        loopPerCore_ = (coreElemNum_ + ubFactor_ - 1) / ubFactor_;
    }

    __aicore__ inline void Process()
    {
        for (int64_t loopIdx = 0; loopIdx < loopPerCore_; ++loopIdx) {
            const int64_t curElems = (loopIdx == loopPerCore_ - 1) ? (coreElemNum_ - (loopPerCore_ - 1) * ubFactor_) :
                                                                     ubFactor_;
            CopyIn(loopIdx, curElems);
            Compute(loopIdx, curElems);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t loopIdx, int64_t curElems)
    {
        LocalTensor<T> inLocal = inputQueue_.AllocTensor<T>();
        const DataCopyExtParams copyParams = {1u, static_cast<uint32_t>(curElems * static_cast<int64_t>(sizeof(T))), 0u,
                                              0u, 0u};
        DataCopyPad(inLocal, inputValueGm_[coreStart_ + loopIdx * ubFactor_], copyParams, padParams_);
        inputQueue_.EnQue(inLocal);
    }

    __aicore__ inline void Compute(int64_t loopIdx, int64_t curElems)
    {
        LocalTensor<T> inLocal = inputQueue_.DeQue<T>();
        __ubuf__ const int64_t* paramUb = (__ubuf__ const int64_t*)paramBuf_.Get<int64_t>().GetPhyAddr();
        __ubuf__ const T* inUb = (__ubuf__ const T*)inLocal.GetPhyAddr();
        asc_vf_call<HybridSimtBody<T>>(dim3(SSA_SIMT_THREAD_NUM), dimNum_, coreStart_ + loopIdx * ubFactor_, curElems,
                                       paramUb, inUb, (__gm__ T*)varOutGm_.GetPhyAddr());
        inputQueue_.FreeTensor(inLocal);
    }

    TPipe* Ppipe_ = nullptr;
    TQue<TPosition::VECIN, SSA_BUFFER_NUM> inputQueue_;
    TBuf<TPosition::VECCALC> paramBuf_;
    GlobalTensor<T> inputValueGm_;
    GlobalTensor<T> varOutGm_;
    DataCopyPadExtParams<T> padParams_{false, 0, 0, 0};
    int64_t dimNum_ = 0;
    int64_t sliceElemNum_ = 0;
    int64_t blockDim_ = 0;
    int64_t elemPerCore_ = 0;
    int64_t lastCoreElemNum_ = 0;
    int64_t ubFactor_ = 0;
    int64_t coreStart_ = 0;
    int64_t coreElemNum_ = 0;
    int64_t loopPerCore_ = 0;
};

} // namespace StridedSliceAssignV2HybridNs
#endif // STRIDED_SLICE_ASSIGN_V2_HYBRID_H_
