/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file rrelu_with_noise_train_impl.h
 * \brief RreluWithNoiseTrain SIMT kernel implementation
 */
#ifndef RRELU_WITH_NOISE_TRAIN_H
#define RRELU_WITH_NOISE_TRAIN_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../../random_common/arch35/random_unified_tiling_data_arch35.h"
#include "../../random_common/arch35/random_kernel_base.h"

namespace RreluWithNoise {
using namespace AscendC;
using namespace RandomKernelBase;

template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(DEFAULT_SIMT_THREAD_NUM) inline void RreluWithNoiseFusedKernel(
    __gm__ T* xGm, __gm__ T* yGm, __gm__ T* noiseGm, int64_t kernelOffset, int64_t seed, uint64_t outputSize,
    uint64_t totalThreads, float lower, float range)
{
    uint32_t key[ALG_KEY_SIZE] = {0, 0};
    uint32_t counter[ALG_COUNTER_SIZE] = {0, 0, 0, 0};
    PhiloxAlgParsInit(key, counter, seed, kernelOffset);

    constexpr uint32_t unroll = 4;
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t repeatTime = (outputSize + totalThreads * unroll - 1) / (totalThreads * unroll);
    const uint64_t elemStride = totalThreads * unroll;

    for (uint64_t linearIndex = idx; linearIndex < totalThreads; linearIndex += blockDim.x * gridDim.x) {
        uint32_t counterTmp[ALG_COUNTER_SIZE] = {0, 0, 0, 0};
        CopyArray<ALG_COUNTER_SIZE>(counterTmp, counter);
        FlashCounter(linearIndex, 0, counterTmp);

        for (uint64_t loopIdx = 0; loopIdx < repeatTime; loopIdx++) {
            const uint64_t elemBase = linearIndex + loopIdx * elemStride;
            const uint64_t elemIdx[unroll] = {elemBase, elemBase + totalThreads, elemBase + 2 * totalThreads,
                                              elemBase + 3 * totalThreads};
            T xValue[unroll] = {};
            for (uint32_t iStep = 0; iStep < unroll; iStep++) {
                xValue[iStep] = (elemIdx[iStep] < outputSize) ? xGm[elemIdx[iStep]] : xValue[iStep];
            }

            uint32_t results[SIMT_STEP];
            PhiloxRandomSimt(key, counterTmp, results);
            SkipOne(counterTmp);

            for (uint32_t iStep = 0; iStep < unroll; iStep++) {
                if (elemIdx[iStep] < outputSize) {
                    float randUniform = results[iStep] * RAND_2POW32_INV + RAND_2POW32_INV_HALF;
                    T r = static_cast<T>(randUniform);
                    float ratioFloat = static_cast<float>(r);
                    T noiseVal = static_cast<T>(ratioFloat * range + lower);
                    float xFloat = static_cast<float>(xValue[iStep]);
                    if (xFloat <= 0.0f) {
                        float factor = static_cast<float>(noiseVal);
                        yGm[elemIdx[iStep]] = static_cast<T>(xFloat * factor);
                        noiseGm[elemIdx[iStep]] = noiseVal;
                    } else {
                        yGm[elemIdx[iStep]] = xValue[iStep];
                        noiseGm[elemIdx[iStep]] = static_cast<T>(1.0f);
                    }
                }
            }
        }
    }
}

template <typename T>
struct RreluWithNoiseFusedLauncher {
    int64_t seed_;
    int64_t realOffset_;
    float lower_;
    float upper_;
    GM_ADDR xAddr_;
    GM_ADDR yAddr_;
    GM_ADDR noiseAddr_;

    __aicore__ RreluWithNoiseFusedLauncher(int64_t seed, int64_t realOffset, float lower, float upper, GM_ADDR x,
                                           GM_ADDR y, GM_ADDR noise)
        : seed_(seed), realOffset_(realOffset), lower_(lower), upper_(upper), xAddr_(x), yAddr_(y), noiseAddr_(noise)
    {}

    __aicore__ inline void operator()(const ExecutionPolicyKernel& policy, int64_t gmOffset, int64_t kernelOffset,
                                      int64_t numel, int64_t grid, int64_t totalThreads)
    {
        (void)policy;
        (void)grid;
        __gm__ T* xGm = (__gm__ T*)xAddr_ + gmOffset;
        __gm__ T* yGm = (__gm__ T*)yAddr_ + gmOffset;
        __gm__ T* noiseGm = (__gm__ T*)noiseAddr_ + gmOffset;
        AscendC::Simt::VF_CALL<RreluWithNoiseFusedKernel<T>>(
            AscendC::Simt::Dim3(DEFAULT_SIMT_THREAD_NUM), xGm, yGm, noiseGm, kernelOffset + realOffset_, seed_,
            static_cast<uint64_t>(numel), static_cast<uint64_t>(totalThreads), lower_, upper_ - lower_);
    }
};

template <typename T>
__aicore__ inline void Process(GM_ADDR x, GM_ADDR seed, GM_ADDR offset, GM_ADDR y, GM_ADDR noise,
                               const RandomUnifiedSimtTilingDataStruct* __restrict tilingData)
{
    if (AscendC::GetBlockIdx() >= static_cast<uint32_t>(tilingData->usedCoreNum)) {
        return;
    }

    int64_t realSeed = *(reinterpret_cast<__gm__ int64_t*>(seed));
    int64_t realOffset = *(reinterpret_cast<__gm__ int64_t*>(offset));

    const float lower = tilingData->fromFp32;
    const float upper = tilingData->toFp32;

    RreluWithNoiseFusedLauncher<T> launcher(realSeed, realOffset, lower, upper, x, y, noise);
    ProcessWithSplitBlocks(tilingData, launcher);
}

} // namespace RreluWithNoise
#endif // RRELU_WITH_NOISE_TRAIN_H
