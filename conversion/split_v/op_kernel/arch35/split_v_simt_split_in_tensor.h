/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SPLIT_V_SIMT_SPLIT_IN_TENSOR_H
#define SPLIT_V_SIMT_SPLIT_IN_TENSOR_H
#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"
#include "simt_api/asc_simt.h"

#ifdef __DAV_FPGA__
constexpr int32_t THREAD_DIM_SIT = 512;
#else
constexpr int32_t THREAD_DIM_SIT = 2048;
#endif

namespace SplitV {
using namespace AscendC;

template <typename T>      // T原始数据类型  U做sizeSplits的数据类型
class SplitVSIMTInTensor { // 每个核处理的数据小于128K，才会有这个模版，所以数据类型都可以降级
public:
    __aicore__ inline SplitVSIMTInTensor(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SplitVSIMTTilingData* tilingData);
    __aicore__ inline void Process();

private:
    // __aicore__ inline void CopyInSplitSize();
    __aicore__ inline __gm__ T* GetTensorAddr(uint32_t index);

private:
    const SplitVSIMTTilingData* tilingData_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    ListTensorDesc outputList_;
    int32_t blockIdx_ = 0;
    int32_t blockNums_ = 0;

    int32_t mLen_ = 0;
    int32_t nLen_ = 0;
    int32_t splitNum_ = 0;
};

template <typename T>
__simt_vf__ LAUNCH_BOUND(THREAD_DIM_SIT) __aicore__
    void ProcessSplitTensor(__gm__ T* inputGM, __gm__ volatile T* outputGM, int32_t outputSize, int32_t nCurLen,
                            int32_t nLen, int32_t preSumN, int32_t partLen, int32_t partOffset, uint32_t shift,
                            uint32_t m)
{
    for (uint32_t idx = threadIdx.x; idx < outputSize; idx += blockDim.x) {
        int32_t row = Simt::UintDiv(idx, m, shift);
        int32_t col = idx - row * nCurLen;
        int32_t inputIndex = row * nLen + col + preSumN;
        int32_t outputIndex = row * partLen + col + partOffset;
        if constexpr (sizeof(T) == 16) {
#if defined(__DAV_920R1__) || defined(__DAV_920R2__)
            // B128(数据类型提升至 16B, ascend960 增强): 显式 128 位访存内建, 单次访存粒度最大化
            asc_stcg(const_cast<__gm__ T*>(&outputGM[outputIndex]), asc_ldcg((__gm__ T*)&inputGM[inputIndex]));
#else
            // 非增强架构无 B128 内建: 不应到达(tiling 侧 IsSimtDtypePromoEnabled 门控保证), 防御性拆为窄访存
            outputGM[outputIndex] = inputGM[inputIndex];
#endif
        } else {
            outputGM[outputIndex] = inputGM[inputIndex];
        }
    }
}

// 按提升后访存宽度 W 发起 VF 拷贝(tiling 已把全部计数缩放进 W 宽度的索引空间)
template <typename W, typename PTR>
__aicore__ inline void LaunchProcessSplitTensor(PTR xPhy, PTR yPhy, uint32_t outputSize, int32_t nCurLen, int32_t nLen,
                                                int32_t preSumN, int32_t partLen, int32_t partOffset, uint32_t shift,
                                                uint32_t m)
{
    asc_vf_call<ProcessSplitTensor<W>>(dim3(THREAD_DIM_SIT), (__gm__ W*)xPhy, (__gm__ volatile W*)yPhy, outputSize,
                                       nCurLen, nLen, preSumN, partLen, partOffset, shift, m);
}

template <typename T>
__aicore__ inline void SplitVSIMTInTensor<T>::Init(GM_ADDR x, GM_ADDR y, const SplitVSIMTTilingData* tilingData)
{
    blockIdx_ = GetBlockIdx();
    blockNums_ = GetBlockNum();
    tilingData_ = tilingData;
    xGm_.SetGlobalBuffer((__gm__ T*)x);
    outputList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(y));
    mLen_ = tilingData_->mSize;
    nLen_ = tilingData_->nSize;
    splitNum_ = tilingData_->splitNum;
}

template <typename T>
__aicore__ inline void SplitVSIMTInTensor<T>::Process()
{
    if (blockIdx_ >= tilingData_->realCoreNum) {
        return;
    }

    int32_t coreNumPerTensor = blockNums_ / splitNum_;
    uint32_t tensorIdx = blockIdx_ / coreNumPerTensor;
    int32_t totalCoreNum = coreNumPerTensor * splitNum_;

    int32_t colOffset = blockIdx_ == 0 ? 0 : tilingData_->colOffset[blockIdx_ - 1];
    uint32_t nCurLen = tilingData_->colOffset[blockIdx_] - colOffset;

    int32_t partLen = tensorIdx == 0 ? tilingData_->colOffset[coreNumPerTensor - 1] :
                                       tilingData_->colOffset[(tensorIdx + 1) * coreNumPerTensor - 1] -
                                           tilingData_->colOffset[tensorIdx * coreNumPerTensor - 1];
    int32_t prePartOffest = tensorIdx == 0 ? 0 : tilingData_->colOffset[tensorIdx * coreNumPerTensor - 1];
    int32_t partOffset = tilingData_->colOffset[blockIdx_] - nCurLen - prePartOffest;
    if (nCurLen == 0) {
        return;
    }
    uint32_t shift = 0;
    uint32_t m = 0;
    GetUintDivMagicAndShift(m, shift, nCurLen);
    uint32_t outputSize = mLen_ * nCurLen;

    yGm_.SetGlobalBuffer(GetTensorAddr(blockIdx_ / coreNumPerTensor));

    // 访存宽度取数据类型提升后的值(tiling 已缩放全部计数); 0 兼容未提升场景
    uint32_t accSize = tilingData_->dtypeSize == 0 ? sizeof(T) : static_cast<uint32_t>(tilingData_->dtypeSize);
    auto xPhy = xGm_.GetPhyAddr();
    auto yPhy = yGm_.GetPhyAddr();
    switch (accSize) {
        case 2U:
            LaunchProcessSplitTensor<uint16_t>(xPhy, yPhy, outputSize, nCurLen, nLen_, colOffset, partLen, partOffset,
                                               shift, m);
            break;
        case 4U:
            LaunchProcessSplitTensor<uint32_t>(xPhy, yPhy, outputSize, nCurLen, nLen_, colOffset, partLen, partOffset,
                                               shift, m);
            break;
        case 8U:
            LaunchProcessSplitTensor<uint64_t>(xPhy, yPhy, outputSize, nCurLen, nLen_, colOffset, partLen, partOffset,
                                               shift, m);
            break;
#if defined(__DAV_920R1__) || defined(__DAV_920R2__)
        case 16U:
            LaunchProcessSplitTensor<uint4>(xPhy, yPhy, outputSize, nCurLen, nLen_, colOffset, partLen, partOffset,
                                            shift, m);
            break;
#endif
        default:
            LaunchProcessSplitTensor<T>(xPhy, yPhy, outputSize, nCurLen, nLen_, colOffset, partLen, partOffset, shift,
                                        m);
            break;
    }
}

template <typename T>
__aicore__ inline __gm__ T* SplitVSIMTInTensor<T>::GetTensorAddr(uint32_t index)
{
    return outputList_.GetDataPtr<T>(index);
}

} // namespace SplitV
#endif // namespace SplitV
