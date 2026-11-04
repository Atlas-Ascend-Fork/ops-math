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
 * \file axpy_v2_dag.h
 * \brief axpy_v2 dag
 */

#ifndef AXPY_V2_DAG_H
#define AXPY_V2_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

namespace AxpyV2Op {
using namespace Ops::Base;

#ifdef __CCE_AICORE__
constexpr AscendC::Reg::CastTrait castTraitUp = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT,
                                                 AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_NONE};
constexpr AscendC::Reg::CastTrait castTraitRint = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT,
                                                   AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
#endif

template <typename T>
struct AxpyV2Fp16Bf16Fused : public Vec::ElemwiseTernaryOP<T, T, T, T> {
    __aicore__ inline AxpyV2Fp16Bf16Fused(LocalTensor<T>& dst, LocalTensor<T>& x1, LocalTensor<T>& x2,
                                          LocalTensor<T>& alpha, const uint32_t& count)
    {
#ifdef __CCE_AICORE__
        constexpr uint32_t vlSize = AscendC::VECTOR_REG_WIDTH / sizeof(float);
        __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();
        __ubuf__ T* x1Addr = (__ubuf__ T*)x1.GetPhyAddr();
        __ubuf__ T* x2Addr = (__ubuf__ T*)x2.GetPhyAddr();
        __ubuf__ T* alphaAddr = (__ubuf__ T*)alpha.GetPhyAddr();
        const uint16_t loopNum = (count + vlSize - 1) / vlSize;
        uint32_t remain = count;
        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<T> x1Reg;
            AscendC::Reg::RegTensor<T> x2Reg;
            AscendC::Reg::RegTensor<T> alphaReg;
            AscendC::Reg::RegTensor<T> resReg;
            AscendC::Reg::RegTensor<float> x1F;
            AscendC::Reg::RegTensor<float> x2F;
            AscendC::Reg::RegTensor<float> alphaF;
            AscendC::Reg::MaskReg mask;
            for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                const uint32_t offset = loopIdx * vlSize;
                mask = AscendC::Reg::UpdateMask<float>(remain);
                AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(x1Reg, x1Addr + offset);
                AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(x2Reg, x2Addr + offset);
                AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(alphaReg, alphaAddr + offset);
                AscendC::Reg::Cast<float, T, castTraitUp>(x1F, x1Reg, mask);
                AscendC::Reg::Cast<float, T, castTraitUp>(x2F, x2Reg, mask);
                AscendC::Reg::Cast<float, T, castTraitUp>(alphaF, alphaReg, mask);
                AscendC::Reg::MulDstAdd(x2F, alphaF, x1F, mask);
                AscendC::Reg::Cast<T, float, castTraitRint>(resReg, x2F, mask);
                AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(dstAddr + offset, resReg, mask);
            }
        }
#endif
    }
};

template <typename T>
struct AxpyV2ComputeFp16Bf16 {
    using OpInputX1 = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using OpInputX2 = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;
    using OpInputAlpha = Bind<Vec::CopyInBrc<T>, Placeholder::In2<T>>;
    using FusedRes = Bind<AxpyV2Fp16Bf16Fused<T>, OpInputX1, OpInputX2, OpInputAlpha>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, FusedRes>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

struct AxpyV2ComputeFloat {
    using OpInputX1 = Bind<Vec::CopyInBrc<float>, Placeholder::In0<float>>;
    using OpInputX2 = Bind<Vec::CopyInBrc<float>, Placeholder::In1<float>>;
    using OpInputAlpha = Bind<Vec::CopyInBrc<float>, Placeholder::In2<float>>;
    // Brc模板存在输入缓存场景，手动Copy规避FusedMulAdd计算结果覆盖OpInputX2
    using OpInputX2Copy = Bind<Vec::Copy<float>, OpInputX2>;
    using FusedMulAddRes = Bind<Vec::FusedMulAdd<float>, OpInputX1, OpInputX2Copy, OpInputAlpha>;
    using OpCopyOut = Bind<Vec::CopyOut<float>, Placeholder::Out0<float>, FusedMulAddRes>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T>
struct AxpyV2ComputeInt32Int64 {
    using OpInputX1 = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using OpInputX2 = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;
    using OpInputAlpha = Bind<Vec::CopyInBrc<T>, Placeholder::In2<T>>;
    using OpMulRes = Bind<Vec::Mul<T>, OpInputX2, OpInputAlpha>;
    using OpAddRes = Bind<Vec::Add<T>, OpInputX1, OpMulRes>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpAddRes>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T1, typename T2>
struct AxpyV2Uint8Int8Fused : public Vec::ElemwiseTernaryOP<T1, T1, T1, T1> {
    __aicore__ inline AxpyV2Uint8Int8Fused(LocalTensor<T1>& dst, LocalTensor<T1>& x1, LocalTensor<T1>& x2,
                                           LocalTensor<T1>& alpha, const uint32_t& count)
    {
#ifdef __CCE_AICORE__
        constexpr uint32_t vlSize = AscendC::VECTOR_REG_WIDTH / sizeof(T2);
        __ubuf__ T1* dstAddr = (__ubuf__ T1*)dst.GetPhyAddr();
        __ubuf__ T1* x1Addr = (__ubuf__ T1*)x1.GetPhyAddr();
        __ubuf__ T1* x2Addr = (__ubuf__ T1*)x2.GetPhyAddr();
        __ubuf__ T1* alphaAddr = (__ubuf__ T1*)alpha.GetPhyAddr();
        const uint16_t loopNum = (count + vlSize - 1) / vlSize;
        uint32_t remain = count;
        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<T1> x1B8A, x1B8B;
            AscendC::Reg::RegTensor<T1> x2B8A, x2B8B;
            AscendC::Reg::RegTensor<T1> alphaB8A, alphaB8B;
            AscendC::Reg::RegTensor<T2> mulResA, mulResB;
            AscendC::Reg::MaskReg maskA, maskB;
            uint16_t loopIdx = 0;
            // 双组寄存器交替：b 组 Load 与 a 组 Compute 重叠调度
            const uint16_t pairLoopNum = loopNum / 2 * 2;
            for (loopIdx = 0; loopIdx < pairLoopNum; loopIdx += 2) {
                const uint32_t offsetA = loopIdx * vlSize;
                maskA = AscendC::Reg::UpdateMask<T2>(remain);
                AscendC::Reg::LoadAlign<T1, AscendC::Reg::LoadDist::DIST_UNPACK_B8>(x1B8A, x1Addr + offsetA);
                AscendC::Reg::LoadAlign<T1, AscendC::Reg::LoadDist::DIST_UNPACK_B8>(x2B8A, x2Addr + offsetA);
                AscendC::Reg::LoadAlign<T1, AscendC::Reg::LoadDist::DIST_UNPACK_B8>(alphaB8A, alphaAddr + offsetA);
                const uint32_t offsetB = (loopIdx + 1) * vlSize;
                maskB = AscendC::Reg::UpdateMask<T2>(remain);
                AscendC::Reg::LoadAlign<T1, AscendC::Reg::LoadDist::DIST_UNPACK_B8>(x1B8B, x1Addr + offsetB);
                AscendC::Reg::LoadAlign<T1, AscendC::Reg::LoadDist::DIST_UNPACK_B8>(x2B8B, x2Addr + offsetB);
                AscendC::Reg::LoadAlign<T1, AscendC::Reg::LoadDist::DIST_UNPACK_B8>(alphaB8B, alphaAddr + offsetB);
                AscendC::Reg::Mul<T2>(mulResA, (AscendC::Reg::RegTensor<T2>&)x2B8A,
                                      (AscendC::Reg::RegTensor<T2>&)alphaB8A, maskA);
                AscendC::Reg::Add<T2>(mulResA, mulResA, (AscendC::Reg::RegTensor<T2>&)x1B8A, maskA);
                AscendC::Reg::StoreAlign<T1, AscendC::Reg::StoreDist::DIST_PACK_B16>(
                    dstAddr + offsetA, (AscendC::Reg::RegTensor<T1>&)mulResA, maskA);
                AscendC::Reg::Mul<T2>(mulResB, (AscendC::Reg::RegTensor<T2>&)x2B8B,
                                      (AscendC::Reg::RegTensor<T2>&)alphaB8B, maskB);
                AscendC::Reg::Add<T2>(mulResB, mulResB, (AscendC::Reg::RegTensor<T2>&)x1B8B, maskB);
                AscendC::Reg::StoreAlign<T1, AscendC::Reg::StoreDist::DIST_PACK_B16>(
                    dstAddr + offsetB, (AscendC::Reg::RegTensor<T1>&)mulResB, maskB);
            }
            if (loopIdx < loopNum) {
                const uint32_t offset = loopIdx * vlSize;
                maskA = AscendC::Reg::UpdateMask<T2>(remain);
                AscendC::Reg::LoadAlign<T1, AscendC::Reg::LoadDist::DIST_UNPACK_B8>(x1B8A, x1Addr + offset);
                AscendC::Reg::LoadAlign<T1, AscendC::Reg::LoadDist::DIST_UNPACK_B8>(x2B8A, x2Addr + offset);
                AscendC::Reg::LoadAlign<T1, AscendC::Reg::LoadDist::DIST_UNPACK_B8>(alphaB8A, alphaAddr + offset);
                AscendC::Reg::Mul<T2>(mulResA, (AscendC::Reg::RegTensor<T2>&)x2B8A,
                                      (AscendC::Reg::RegTensor<T2>&)alphaB8A, maskA);
                AscendC::Reg::Add<T2>(mulResA, mulResA, (AscendC::Reg::RegTensor<T2>&)x1B8A, maskA);
                AscendC::Reg::StoreAlign<T1, AscendC::Reg::StoreDist::DIST_PACK_B16>(
                    dstAddr + offset, (AscendC::Reg::RegTensor<T1>&)mulResA, maskA);
            }
        }
#endif
    }
};

template <typename T1, typename T2>
struct AxpyV2ComputeUint8Int8 {
    using OpInputX1 = Bind<Vec::CopyInBrc<T1>, Placeholder::In0<T1>>;
    using OpInputX2 = Bind<Vec::CopyInBrc<T1>, Placeholder::In1<T1>>;
    using OpInputAlpha = Bind<Vec::CopyInBrc<T1>, Placeholder::In2<T1>>;
    using FusedRes = Bind<AxpyV2Uint8Int8Fused<T1, T2>, OpInputX1, OpInputX2, OpInputAlpha>;
    using OpCopyOut = Bind<Vec::CopyOut<T1>, Placeholder::Out0<T1>, FusedRes>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T>
struct AxpyV2BoolFused : public Vec::ElemwiseTernaryOP<T, T, T, T> {
    __aicore__ inline AxpyV2BoolFused(LocalTensor<T>& dst, LocalTensor<T>& x1, LocalTensor<T>& x2,
                                      LocalTensor<T>& alpha, const uint32_t& count)
    {
#ifdef __CCE_AICORE__
        constexpr uint32_t vlSize = AscendC::VECTOR_REG_WIDTH / sizeof(T);
        __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();
        __ubuf__ T* x1Addr = (__ubuf__ T*)x1.GetPhyAddr();
        __ubuf__ T* x2Addr = (__ubuf__ T*)x2.GetPhyAddr();
        __ubuf__ T* alphaAddr = (__ubuf__ T*)alpha.GetPhyAddr();
        const uint16_t loopNum = (count + vlSize - 1) / vlSize;
        uint32_t remain = count;
        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<T> x1B8A, x1B8B;
            AscendC::Reg::RegTensor<T> x2B8A, x2B8B;
            AscendC::Reg::RegTensor<T> alphaB8A, alphaB8B;
            AscendC::Reg::RegTensor<T> resA, resB;
            AscendC::Reg::MaskReg maskA, maskB;
            uint16_t loopIdx = 0;
            // 双组寄存器交替：b 组 Load 与 a 组 Compute 重叠调度
            const uint16_t pairLoopNum = loopNum / 2 * 2;
            for (loopIdx = 0; loopIdx < pairLoopNum; loopIdx += 2) {
                const uint32_t offsetA = loopIdx * vlSize;
                maskA = AscendC::Reg::UpdateMask<T>(remain);
                AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_NORM>(x1B8A, x1Addr + offsetA);
                AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_NORM>(x2B8A, x2Addr + offsetA);
                AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_NORM>(alphaB8A, alphaAddr + offsetA);
                const uint32_t offsetB = (loopIdx + 1) * vlSize;
                maskB = AscendC::Reg::UpdateMask<T>(remain);
                AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_NORM>(x1B8B, x1Addr + offsetB);
                AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_NORM>(x2B8B, x2Addr + offsetB);
                AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_NORM>(alphaB8B, alphaAddr + offsetB);
                AscendC::Reg::And<T>(resA, x2B8A, alphaB8A, maskA);
                AscendC::Reg::Or<T>(resA, resA, x1B8A, maskA);
                AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_NORM_B8>(dstAddr + offsetA, resA, maskA);
                AscendC::Reg::And<T>(resB, x2B8B, alphaB8B, maskB);
                AscendC::Reg::Or<T>(resB, resB, x1B8B, maskB);
                AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_NORM_B8>(dstAddr + offsetB, resB, maskB);
            }
            if (loopIdx < loopNum) {
                const uint32_t offset = loopIdx * vlSize;
                maskA = AscendC::Reg::UpdateMask<T>(remain);
                AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_NORM>(x1B8A, x1Addr + offset);
                AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_NORM>(x2B8A, x2Addr + offset);
                AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_NORM>(alphaB8A, alphaAddr + offset);
                AscendC::Reg::And<T>(resA, x2B8A, alphaB8A, maskA);
                AscendC::Reg::Or<T>(resA, resA, x1B8A, maskA);
                AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_NORM_B8>(dstAddr + offset, resA, maskA);
            }
        }
#endif
    }
};

struct AxpyV2ComputeBool {
    using OpInputX1 = Bind<Vec::CopyInBrc<int8_t>, Placeholder::In0<int8_t>>;
    using OpInputX2 = Bind<Vec::CopyInBrc<int8_t>, Placeholder::In1<int8_t>>;
    using OpInputAlpha = Bind<Vec::CopyInBrc<int8_t>, Placeholder::In2<int8_t>>;
    using FusedRes = Bind<AxpyV2BoolFused<int8_t>, OpInputX1, OpInputX2, OpInputAlpha>;
    using OpCopyOut = Bind<Vec::CopyOut<int8_t>, Placeholder::Out0<int8_t>, FusedRes>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

} // namespace AxpyV2Op
#endif // AXPY_V2_DAG_H
