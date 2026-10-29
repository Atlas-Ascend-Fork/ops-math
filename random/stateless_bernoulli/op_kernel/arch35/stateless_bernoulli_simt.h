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
 * \file stateless_bernoulli_simt.h
 * \brief
 */

#ifndef STATELESS_BERNOULLI_H
#define STATELESS_BERNOULLI_H

#include "kernel_operator.h"
#include "../../random_common/arch35/random_kernel_base.h"
#include "simt_api/asc_simt.h"

namespace StatelessBernoulli {

// 场景化 tilingKey: 100=标量(交织映射+严格小于), 101=张量非广播(直接索引),
// 102=张量广播(stride 寻址)——语义区分收敛至 entry 按 tilingKey 分发
constexpr uint32_t PHILOX_THREAD_LAUNCH = 512;
constexpr uint32_t MAX_THREADS_PER_PROCESSOR = 2048;
constexpr uint32_t PHILOX_BLOCK_THREAD = 512;
constexpr uint32_t GPU_GRID_SIZE = 2147483647;
constexpr int32_t STEP = 4;
constexpr uint16_t ALG_KEY_SIZE = 2;
constexpr uint16_t ALG_COUNTER_SIZE = 4;
constexpr int32_t CONTINUOUS_USE = 0;
// UB 传参区布局 (int64 元素下标, 各组连续 8 个元素, 基址逐级推导):
//   [UB_IDX_DIMS]=probDims, [UB_IDX_SIZES..+8)=probSizes(输出shape),
//   [UB_IDX_STRIDES..+8)=probStrides(广播步长), [UB_IDX_MAGIC..+8)/[UB_IDX_SHIFT..+8)=各维除数快除参数
constexpr int32_t PROB_MAX_DIMS = 8;
constexpr int32_t UB_IDX_DIMS = 0;
constexpr int32_t UB_IDX_SIZES = 1;
constexpr int32_t UB_IDX_STRIDES = UB_IDX_SIZES + PROB_MAX_DIMS;
constexpr int32_t UB_IDX_MAGIC = UB_IDX_STRIDES + PROB_MAX_DIMS;
constexpr int32_t UB_IDX_SHIFT = UB_IDX_MAGIC + PROB_MAX_DIMS;
// 传参区共 (UB_IDX_SHIFT + PROB_MAX_DIMS) = 33 个元素 264 字节, 长度向上对齐到 512 字节
constexpr int32_t STRIDE_UB_LEN = 512;

// 标量语义变换函数: u = x*2^-32 + 2^-33 (值域 (0,1]), 输出 = (u < p) ? 1 : 0
// prob 由 launcher 一次性读取并按值传递, 避免热循环内重复读 GM
template <typename To>
struct BernoulliScalarTransform {
    float probFp32_;

    __aicore__ BernoulliScalarTransform(float probFp32) : probFp32_(probFp32) {}

    __simt_callee__ __aicore__ inline void operator()(__gm__ volatile To* outputGm, uint64_t li,
                                                      const uint32_t* results, uint32_t iStep,
                                                      [[maybe_unused]] uint32_t unroll = 1)
    {
        float u = results[iStep] * RandomKernelBase::RAND_2POW32_INV + RandomKernelBase::RAND_2POW32_INV_HALF;
        *const_cast<__gm__ To*>(&outputGm[li]) = (u < probFp32_) ? 1 : 0;
    }
};

// 多维索引到 prob 偏移的转换: 第二层, 传入线性索引, 从 UB 自行读取参数计算
// 整除用快除 (乘高+移位) 实现, 余数 = li - q*sizes[i] (除数本身仍需保留)
__simt_callee__ __aicore__ inline int64_t ProbIndexToOffset(uint64_t li, __ubuf__ const int64_t* par)
{
    int32_t dims = static_cast<int32_t>(par[UB_IDX_DIMS]);
    int64_t offset = 0;
    for (int32_t i = dims - 1; i > 0; --i) {
        uint64_t q = AscendC::Simt::UintDiv(li, static_cast<uint64_t>(par[UB_IDX_MAGIC + i]),
                                            static_cast<uint64_t>(par[UB_IDX_SHIFT + i]));
        offset += static_cast<int64_t>(li - q * static_cast<uint64_t>(par[UB_IDX_SIZES + i])) * par[UB_IDX_STRIDES + i];
        li = q;
    }
    return offset + static_cast<int64_t>(li) * par[UB_IDX_STRIDES];
}

// 张量语义 dense kernel: prob 与输出同 shape, offset(li) ≡ li, 直接索引零寻址开销
// 独立模板实例（无 UB 参数依赖）, 编译期与广播路径分离
template <typename Tp, typename To, typename TIdx>
__simt_vf__ __aicore__ LAUNCH_BOUND(PHILOX_THREAD_LAUNCH) inline void PhiloxBernoulliTensorDense(
    __gm__ To* outGm, __gm__ Tp* probGm, int64_t offset, int64_t seed, TIdx outputSize, uint64_t magic, uint64_t shift,
    uint64_t totalThreads)
{
    uint32_t counterTmp[ALG_COUNTER_SIZE] = {0, 0, 0, 0};
    uint32_t key[ALG_KEY_SIZE] = {0, 0};
    RandomKernelBase::PhiloxAlgParsInit(key, counterTmp, seed, offset);
    for (TIdx i = (blockIdx.x * blockDim.x + threadIdx.x) * STEP; i < outputSize; i += gridDim.x * blockDim.x * STEP) {
        uint32_t counter[ALG_COUNTER_SIZE] = {0, 0, 0, 0};
        RandomKernelBase::CopyArray<ALG_COUNTER_SIZE>(counter, counterTmp);
        float results[ALG_COUNTER_SIZE] = {0, 0, 0, 0};
        // counter 的 call 分量恒 0: 对齐竞品 functor 每组 curand_init 的复位语义
        uint64_t groupIdx = static_cast<uint64_t>(i) / STEP;
        uint64_t vthread = groupIdx - AscendC::Simt::UintDiv(groupIdx, magic, shift) * totalThreads;
        RandomKernelBase::FlashCounter(vthread, 0, counter);
        RandomKernelBase::PhiloxRandomSimt(key, counter, results);
        for (uint8_t j = 0; j < 4; ++j) {
            if (i + j >= outputSize) {
                break;
            }
            float probFp32 = AscendC::IsSameType<Tp, float>::value ? probGm[i + j] : static_cast<float>(probGm[i + j]);
            outGm[i + j] = results[j] <= probFp32 ? 1 : 0;
        }
    }
}

// 张量语义广播 kernel: 连续映射, prob 按广播 stride 寻址（广播维 stride=0 反复读同一地址）
// 偏移计算下沉到 ProbIndexToOffset, stride 与快除参数经 UB 传递（布局见常量区, DSB 后下发 VF）
template <typename Tp, typename To, typename TIdx>
__simt_vf__ __aicore__ LAUNCH_BOUND(PHILOX_THREAD_LAUNCH) inline void PhiloxBernoulliTensorBcast(
    __gm__ To* outGm, __gm__ Tp* probGm, int64_t offset, int64_t seed, TIdx outputSize, uint64_t magic, uint64_t shift,
    uint64_t totalThreads, __ubuf__ const int64_t* strideBuf)
{
    uint32_t counterTmp[ALG_COUNTER_SIZE] = {0, 0, 0, 0};
    uint32_t key[ALG_KEY_SIZE] = {0, 0};
    RandomKernelBase::PhiloxAlgParsInit(key, counterTmp, seed, offset);
    for (TIdx i = (blockIdx.x * blockDim.x + threadIdx.x) * STEP; i < outputSize; i += gridDim.x * blockDim.x * STEP) {
        uint32_t counter[ALG_COUNTER_SIZE] = {0, 0, 0, 0};
        RandomKernelBase::CopyArray<ALG_COUNTER_SIZE>(counter, counterTmp);
        float results[ALG_COUNTER_SIZE] = {0, 0, 0, 0};
        // counter 的 call 分量恒 0(同 Dense 模板)
        uint64_t groupIdx = static_cast<uint64_t>(i) / STEP;
        uint64_t vthread = groupIdx - AscendC::Simt::UintDiv(groupIdx, magic, shift) * totalThreads;
        RandomKernelBase::FlashCounter(vthread, 0, counter);
        RandomKernelBase::PhiloxRandomSimt(key, counter, results);
        // 组首完整分解一次, 组内按末维 stride 递推: 连续 li 仅末维索引 +1
        // (广播维 stride=0, +0 反复读同地址); 跨末维边界时回退全量重算并归零,
        // 末维 <=4 的极端 shape 逐元素回退, 与原逐元素寻址完全等价。
        // dims=1 时无快除参数(magic 自 1 维起填), 末维即全长, 递推不会触发跨界
        const int32_t bcastDims = static_cast<int32_t>(strideBuf[UB_IDX_DIMS]);
        const int64_t lastDim = strideBuf[UB_IDX_SIZES + bcastDims - 1];
        const int64_t lastStride = strideBuf[UB_IDX_STRIDES + bcastDims - 1];
        int64_t lastIdx = 0;
        if (bcastDims > 1) {
            uint64_t qLast = AscendC::Simt::UintDiv(static_cast<uint64_t>(i),
                                                    static_cast<uint64_t>(strideBuf[UB_IDX_MAGIC + bcastDims - 1]),
                                                    static_cast<uint64_t>(strideBuf[UB_IDX_SHIFT + bcastDims - 1]));
            lastIdx = static_cast<int64_t>(static_cast<uint64_t>(i) - qLast * static_cast<uint64_t>(lastDim));
        }
        int64_t probOffset = ProbIndexToOffset(static_cast<uint64_t>(i), strideBuf);
        for (uint8_t j = 0; j < 4; ++j) {
            if (i + j >= outputSize) {
                break;
            }
            if (j > 0) {
                if (++lastIdx == lastDim) {
                    probOffset = ProbIndexToOffset(static_cast<uint64_t>(i + j), strideBuf);
                    lastIdx = 0;
                } else {
                    probOffset += lastStride;
                }
            }
            float probFp32 = AscendC::IsSameType<Tp, float>::value ? probGm[probOffset] :
                                                                     static_cast<float>(probGm[probOffset]);
            outGm[i + j] = results[j] <= probFp32 ? 1 : 0;
        }
    }
}

// 标量语义 launcher: 交织映射, 经 ProcessWithSplitBlocks 消费 tiling 下发的 splitBlocks 执行参数
// prob 在此一次性读取并转 float, 按值传递给 transform
template <typename Tp, typename To>
struct BernoulliScalarLauncher {
    int64_t seed_;
    GM_ADDR prob_;
    GM_ADDR baseAddr_;

    __aicore__ BernoulliScalarLauncher(int64_t seed, GM_ADDR prob, GM_ADDR baseAddr)
        : seed_(seed), prob_(prob), baseAddr_(baseAddr)
    {}

    __aicore__ inline float ReadProb()
    {
        AscendC::GlobalTensor<Tp> probTensor;
        probTensor.SetGlobalBuffer((__gm__ Tp*)prob_, 1);
        Tp currProb = probTensor.GetValue(0);
        if constexpr (AscendC::IsSameType<Tp, float>::value) {
            return currProb;
        } else if constexpr (AscendC::IsSameType<Tp, half>::value) {
            return static_cast<float>(currProb);
        } else if constexpr (AscendC::IsSameType<Tp, bfloat16_t>::value) {
            return AscendC::ToFloat(currProb);
        } else {
            return static_cast<float>(currProb);
        }
    }

    __aicore__ inline void operator()(const RandomKernelBase::ExecutionPolicyKernel& policy, int64_t gmOffset,
                                      int64_t kernelOffset, int64_t numel, int64_t grid, int64_t totalThreads)
    {
        // 映射反转注释见 PhiloxSimtKernelDiscontinuous 模板; transform 构造与 VF 下发分离到 Launch
        BernoulliScalarTransform<To> transform(ReadProb());
        Launch(policy, gmOffset, kernelOffset, numel, totalThreads, transform);
    }

private:
    __aicore__ inline void Launch(const RandomKernelBase::ExecutionPolicyKernel& policy, int64_t gmOffset,
                                  int64_t kernelOffset, int64_t numel, int64_t totalThreads,
                                  BernoulliScalarTransform<To>& transform)
    {
        __gm__ volatile To* gmPtr = (__gm__ volatile To*)baseAddr_ + gmOffset;
        asc_vf_call<RandomKernelBase::PhiloxSimtKernelDiscontinuous<To, BernoulliScalarTransform<To>>>(
            dim3(PHILOX_THREAD_LAUNCH), gmPtr, kernelOffset, seed_, numel, policy.magic, policy.shift, totalThreads,
            transform);
    }
};

template <typename Tp, typename To>
class StatelessBernoulliKernel {
public:
    __aicore__ inline StatelessBernoulliKernel(){};
    // 场景化入口, 与 tilingKey 一一对应(100/101/102), kernel 内不再做语义分支判断
    __aicore__ inline void ProcessScalar(GM_ADDR prob, GM_ADDR out,
                                         const RandomUnifiedSimtTilingDataStruct* __restrict tilingData);
    __aicore__ inline void ProcessTensorDense(GM_ADDR prob, GM_ADDR out,
                                              const RandomUnifiedSimtTilingDataStruct* __restrict tilingData);
    __aicore__ inline void ProcessTensorBcast(GM_ADDR prob, GM_ADDR out,
                                              const RandomUnifiedSimtTilingDataStruct* __restrict tilingData);

private:
    // 张量语义公共执行参数: T = 512 * min(ceil(n/2048), 2^31-1) 与 totalThreads 快除参数
    __aicore__ inline void CalcTensorPolicy(const RandomUnifiedSimtTilingDataStruct* __restrict tilingData,
                                            uint64_t& totalThreads, uint64_t& magic, uint64_t& shift);
};

// tilingKey=100 标量语义: 交织映射, 经 splitBlocks 取执行参数
template <typename Tp, typename To>
__aicore__ inline void StatelessBernoulliKernel<Tp, To>::ProcessScalar(
    GM_ADDR prob, GM_ADDR out, const RandomUnifiedSimtTilingDataStruct* __restrict tilingData)
{
    if (AscendC::GetBlockIdx() >= static_cast<uint32_t>(tilingData->usedCoreNum)) {
        return;
    }
    BernoulliScalarLauncher<Tp, To> launcher(tilingData->seed, prob, out);
    RandomKernelBase::ProcessWithSplitBlocks(tilingData, launcher);
}

template <typename Tp, typename To>
__aicore__ inline void StatelessBernoulliKernel<Tp, To>::CalcTensorPolicy(
    const RandomUnifiedSimtTilingDataStruct* __restrict tilingData, uint64_t& totalThreads, uint64_t& magic,
    uint64_t& shift)
{
    uint64_t minBlockNums = (tilingData->outputSize + MAX_THREADS_PER_PROCESSOR - 1) / MAX_THREADS_PER_PROCESSOR;
    minBlockNums = minBlockNums < GPU_GRID_SIZE ? minBlockNums : GPU_GRID_SIZE;
    totalThreads = minBlockNums * PHILOX_BLOCK_THREAD;
    RandomKernelBase::GetUintDivMagicAndShift(magic, shift, totalThreads);
}

// tilingKey=101 张量语义非广播: prob 与输出同 shape, 直接索引零寻址开销
template <typename Tp, typename To>
__aicore__ inline void StatelessBernoulliKernel<Tp, To>::ProcessTensorDense(
    GM_ADDR prob, GM_ADDR out, const RandomUnifiedSimtTilingDataStruct* __restrict tilingData)
{
    if (AscendC::GetBlockIdx() >= static_cast<uint32_t>(tilingData->usedCoreNum)) {
        return;
    }
    uint64_t totalThreads = 0;
    uint64_t magic = 0;
    uint64_t shift = 0;
    CalcTensorPolicy(tilingData, totalThreads, magic, shift);

    if (tilingData->outputSize <= GPU_GRID_SIZE) {
        asc_vf_call<PhiloxBernoulliTensorDense<Tp, To, uint32_t>>(
            dim3(PHILOX_THREAD_LAUNCH), (__gm__ To*)out, (__gm__ Tp*)prob, tilingData->offset, tilingData->seed,
            static_cast<uint32_t>(tilingData->outputSize), magic, shift, totalThreads);
    } else {
        asc_vf_call<PhiloxBernoulliTensorDense<Tp, To, uint64_t>>(
            dim3(PHILOX_THREAD_LAUNCH), (__gm__ To*)out, (__gm__ Tp*)prob, tilingData->offset, tilingData->seed,
            tilingData->outputSize, magic, shift, totalThreads);
    }
}

// tilingKey=102 张量语义广播: stride 与快除参数经 UB 传递给 VF（各核写各自 UB, 无跨核可见性问题）
template <typename Tp, typename To>
__aicore__ inline void StatelessBernoulliKernel<Tp, To>::ProcessTensorBcast(
    GM_ADDR prob, GM_ADDR out, const RandomUnifiedSimtTilingDataStruct* __restrict tilingData)
{
    if (AscendC::GetBlockIdx() >= static_cast<uint32_t>(tilingData->usedCoreNum)) {
        return;
    }
    uint64_t totalThreads = 0;
    uint64_t magic = 0;
    uint64_t shift = 0;
    CalcTensorPolicy(tilingData, totalThreads, magic, shift);

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> strideBuf;
    pipe.InitBuffer(strideBuf, STRIDE_UB_LEN);
    AscendC::LocalTensor<int64_t> strideLocal = strideBuf.Get<int64_t>();
    strideLocal.SetValue(UB_IDX_DIMS, tilingData->probDims);
    for (int32_t i = 0; i < PROB_MAX_DIMS; ++i) {
        strideLocal.SetValue(UB_IDX_SIZES + i, tilingData->probSizes[i]);
        strideLocal.SetValue(UB_IDX_STRIDES + i, tilingData->probStrides[i]);
    }
    int32_t probDims = static_cast<int32_t>(tilingData->probDims);
    probDims = probDims < PROB_MAX_DIMS ? probDims : PROB_MAX_DIMS;
    for (int32_t i = 1; i < probDims; ++i) {
        uint64_t divMagic = 0;
        uint64_t divShift = 0;
        AscendC::GetUintDivMagicAndShift(divMagic, divShift, static_cast<uint64_t>(tilingData->probSizes[i]));
        strideLocal.SetValue(UB_IDX_MAGIC + i, static_cast<int64_t>(divMagic));
        strideLocal.SetValue(UB_IDX_SHIFT + i, static_cast<int64_t>(divShift));
    }
    AscendC::DataSyncBarrier<AscendC::MemDsbT::UB>(); // 确保 UB 写入对 VF 可见后再下发任务
    if (tilingData->outputSize <= GPU_GRID_SIZE) {
        asc_vf_call<PhiloxBernoulliTensorBcast<Tp, To, uint32_t>>(
            dim3(PHILOX_THREAD_LAUNCH), (__gm__ To*)out, (__gm__ Tp*)prob, tilingData->offset, tilingData->seed,
            static_cast<uint32_t>(tilingData->outputSize), magic, shift, totalThreads,
            (__ubuf__ const int64_t*)(strideLocal.GetPhyAddr()));
    } else {
        asc_vf_call<PhiloxBernoulliTensorBcast<Tp, To, uint64_t>>(
            dim3(PHILOX_THREAD_LAUNCH), (__gm__ To*)out, (__gm__ Tp*)prob, tilingData->offset, tilingData->seed,
            tilingData->outputSize, magic, shift, totalThreads, (__ubuf__ const int64_t*)(strideLocal.GetPhyAddr()));
    }
}
} // namespace StatelessBernoulli
#endif // STATELESS_BERNOULLI_H
