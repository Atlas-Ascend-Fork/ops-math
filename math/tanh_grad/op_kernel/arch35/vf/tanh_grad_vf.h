/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file tanh_grad_vf.h
 * \brief SIMD VF entry functions for TanhGrad on Ascend 950 (arch35).
 *
 * One VF function per dtype combination of (y, dy, z); the with_loops and
 * without_loops kernel classes of one dtype combination share the same VF
 * function. UB addresses and the element count are passed in as plain
 * arguments; GM transfers, broadcast and dispatch stay in the host classes.
 *
 * Addressing convention: AddrReg stride unit is the element count of the
 * CreateAddrReg template type (b16 unpack lanes count as float elements,
 * i.e. VECTOR_REG_WIDTH / sizeof(float) per iteration), matching the
 * original bufferIn0Addr + i * (VECTOR_REG_WIDTH / sizeof(float)) offsets.
 */

#ifndef ASCENDC_TANH_GRAD_VF_H_

#define ASCENDC_TANH_GRAD_VF_H_

#include "kernel_operator.h"

#ifdef __CCE_AICORE__

namespace TanhGrad {

using AscendC::Reg::MaskReg;

using AscendC::Reg::RegTensor;

constexpr static AscendC::Reg::CastTrait castTraitTg0 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,

                                                         AscendC::Reg::MaskMergeMode::ZEROING,

                                                         AscendC::RoundMode::UNKNOWN};

constexpr static AscendC::Reg::CastTrait castTraitTg1 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT,

                                                         AscendC::Reg::MaskMergeMode::ZEROING,

                                                         AscendC::RoundMode::CAST_RINT};

__simd_vf__ inline void TanhGradF16VF(__ubuf__ half* bufferIn0Addr, __ubuf__ half* bufferIn1Addr,

                                      __ubuf__ half* bufferOut0Addr, int64_t dataLength)

{
    RegTensor<half> vreg0;

    RegTensor<float> vreg1;

    RegTensor<float> vreg2;

    RegTensor<float> vreg3;

    RegTensor<float> vreg4;

    RegTensor<half> vreg5;

    RegTensor<float> vreg6;

    RegTensor<float> vreg7;

    RegTensor<half> vreg8;

    MaskReg preg0;

    uint32_t size = dataLength;

    uint16_t vfLoopNum = (dataLength + (AscendC::VECTOR_REG_WIDTH / 4) - 1) / (AscendC::VECTOR_REG_WIDTH / 4);

    for (uint16_t i = 0; i < vfLoopNum; i++) {
        preg0 = AscendC::Reg::UpdateMask<float>(size);

        AscendC::Reg::AddrReg addrReg = AscendC::Reg::CreateAddrReg<float>(i, (AscendC::VECTOR_REG_WIDTH / 4));

        AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(

            vreg0, bufferIn0Addr, addrReg);

        AscendC::Reg::Cast<float, half, castTraitTg0>(vreg1, vreg0, preg0);

        AscendC::Reg::Duplicate<float, float>(vreg4, 1.0);

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202)

        AscendC::Reg::FusedMulSubDst<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg4, vreg1, vreg1, preg0);

#else

        AscendC::Reg::Muls<float, float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg2, vreg1, static_cast<float>(-1),

                                                                               preg0);

        AscendC::Reg::MulAddDst<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg4, vreg2, vreg1, preg0);

#endif

        AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(

            vreg5, bufferIn1Addr, addrReg);

        AscendC::Reg::Cast<float, half, castTraitTg0>(vreg6, vreg5, preg0);

        AscendC::Reg::Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg7, vreg4, vreg6, preg0);

        AscendC::Reg::Cast<half, float, castTraitTg1>(vreg8, vreg7, preg0);

        AscendC::Reg::StoreAlign<half, AscendC::Reg::StoreDist::DIST_PACK_B32>(

            bufferOut0Addr, vreg8, addrReg, preg0);
    }
}

__simd_vf__ inline void TanhGradF32VF(__ubuf__ float* bufferIn0Addr, __ubuf__ float* bufferIn1Addr,

                                      __ubuf__ float* bufferOut0Addr, int64_t dataLength)

{
    RegTensor<float> vreg0;

    RegTensor<float> vreg1;

    RegTensor<float> vreg2;

    RegTensor<float> vreg3;

    RegTensor<float> vreg4;

    RegTensor<float> vreg5;

    MaskReg preg0;

    uint32_t size = dataLength;

    uint16_t vfLoopNum = (dataLength + (AscendC::VECTOR_REG_WIDTH / 4) - 1) / (AscendC::VECTOR_REG_WIDTH / 4);

    for (uint16_t i = 0; i < vfLoopNum; i++) {
        preg0 = AscendC::Reg::UpdateMask<float>(size);

        AscendC::Reg::AddrReg addrReg = AscendC::Reg::CreateAddrReg<float>(i, (AscendC::VECTOR_REG_WIDTH / 4));

        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(

            vreg0, bufferIn0Addr, addrReg);

        AscendC::Reg::Duplicate<float, float>(vreg3, 1.0);

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202)

        AscendC::Reg::FusedMulSubDst<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg3, vreg0, vreg0, preg0);

#else

        AscendC::Reg::Muls<float, float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg2, vreg0, static_cast<float>(-1),

                                                                               preg0);

        AscendC::Reg::MulAddDst<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg3, vreg2, vreg0, preg0);

#endif

        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(

            vreg4, bufferIn1Addr, addrReg);

        AscendC::Reg::Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg5, vreg3, vreg4, preg0);

        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(

            bufferOut0Addr, vreg5, addrReg, preg0);
    }
}

__simd_vf__ inline void TanhGradBf16VF(__ubuf__ bfloat16_t* bufferIn0Addr, __ubuf__ bfloat16_t* bufferIn1Addr,

                                       __ubuf__ bfloat16_t* bufferOut0Addr, int64_t dataLength)

{
    RegTensor<bfloat16_t> vreg0;

    RegTensor<float> vreg1;

    RegTensor<float> vreg2;

    RegTensor<float> vreg3;

    RegTensor<float> vreg4;

    RegTensor<bfloat16_t> vreg5;

    RegTensor<float> vreg6;

    RegTensor<float> vreg7;

    RegTensor<bfloat16_t> vreg8;

    MaskReg preg0;

    uint32_t size = dataLength;

    uint16_t vfLoopNum = (dataLength + (AscendC::VECTOR_REG_WIDTH / 4) - 1) / (AscendC::VECTOR_REG_WIDTH / 4);

    for (uint16_t i = 0; i < vfLoopNum; i++) {
        preg0 = AscendC::Reg::UpdateMask<float>(size);

        AscendC::Reg::AddrReg addrReg = AscendC::Reg::CreateAddrReg<float>(i, (AscendC::VECTOR_REG_WIDTH / 4));

        AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(

            vreg0, bufferIn0Addr, addrReg);

        AscendC::Reg::Cast<float, bfloat16_t, castTraitTg0>(vreg1, vreg0, preg0);

        AscendC::Reg::Duplicate<float, float>(vreg4, 1.0);

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202)

        AscendC::Reg::FusedMulSubDst<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg4, vreg1, vreg1, preg0);

#else

        AscendC::Reg::Muls<float, float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg2, vreg1, static_cast<float>(-1),

                                                                               preg0);

        AscendC::Reg::MulAddDst<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg4, vreg2, vreg1, preg0);

#endif

        AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(

            vreg5, bufferIn1Addr, addrReg);

        AscendC::Reg::Cast<float, bfloat16_t, castTraitTg0>(vreg6, vreg5, preg0);

        AscendC::Reg::Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg7, vreg4, vreg6, preg0);

        AscendC::Reg::Cast<bfloat16_t, float, castTraitTg1>(vreg8, vreg7, preg0);

        AscendC::Reg::StoreAlign<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(

            bufferOut0Addr, vreg8, addrReg, preg0);
    }
}

__simd_vf__ inline void TanhGradComb0VF(__ubuf__ half* bufferIn0Addr, __ubuf__ float* bufferIn1Addr,

                                        __ubuf__ float* bufferOut0Addr, int64_t dataLength)

{
    RegTensor<half> vreg0;

    RegTensor<float> vreg1;

    RegTensor<float> vreg2;

    RegTensor<float> vreg3;

    RegTensor<float> vreg4;

    RegTensor<float> vreg5;

    RegTensor<float> vreg6;

    MaskReg preg0;

    uint32_t size = dataLength;

    uint16_t vfLoopNum = (dataLength + (AscendC::VECTOR_REG_WIDTH / 4) - 1) / (AscendC::VECTOR_REG_WIDTH / 4);

    for (uint16_t i = 0; i < vfLoopNum; i++) {
        preg0 = AscendC::Reg::UpdateMask<float>(size);

        AscendC::Reg::AddrReg addrReg = AscendC::Reg::CreateAddrReg<float>(i, (AscendC::VECTOR_REG_WIDTH / 4));

        AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(

            vreg0, bufferIn0Addr, addrReg);

        AscendC::Reg::Cast<float, half, castTraitTg0>(vreg1, vreg0, preg0);

        AscendC::Reg::Duplicate<float, float>(vreg4, 1.0);

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202)

        AscendC::Reg::FusedMulSubDst<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg4, vreg1, vreg1, preg0);

#else

        AscendC::Reg::Muls<float, float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg2, vreg1, static_cast<float>(-1),

                                                                               preg0);

        AscendC::Reg::MulAddDst<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg4, vreg2, vreg1, preg0);

#endif

        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(

            vreg5, bufferIn1Addr, addrReg);

        AscendC::Reg::Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg6, vreg4, vreg5, preg0);

        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(

            bufferOut0Addr, vreg6, addrReg, preg0);
    }
}

__simd_vf__ inline void TanhGradComb1VF(__ubuf__ half* bufferIn0Addr, __ubuf__ bfloat16_t* bufferIn1Addr,

                                        __ubuf__ float* bufferOut0Addr, int64_t dataLength)

{
    RegTensor<half> vreg0;

    RegTensor<float> vreg1;

    RegTensor<float> vreg2;

    RegTensor<float> vreg3;

    RegTensor<float> vreg4;

    RegTensor<bfloat16_t> vreg5;

    RegTensor<float> vreg6;

    RegTensor<float> vreg7;

    MaskReg preg0;

    uint32_t size = dataLength;

    uint16_t vfLoopNum = (dataLength + (AscendC::VECTOR_REG_WIDTH / 4) - 1) / (AscendC::VECTOR_REG_WIDTH / 4);

    for (uint16_t i = 0; i < vfLoopNum; i++) {
        preg0 = AscendC::Reg::UpdateMask<float>(size);

        AscendC::Reg::AddrReg addrReg = AscendC::Reg::CreateAddrReg<float>(i, (AscendC::VECTOR_REG_WIDTH / 4));

        AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(

            vreg0, bufferIn0Addr, addrReg);

        AscendC::Reg::Cast<float, half, castTraitTg0>(vreg1, vreg0, preg0);

        AscendC::Reg::Duplicate<float, float>(vreg4, 1.0);

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202)

        AscendC::Reg::FusedMulSubDst<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg4, vreg1, vreg1, preg0);

#else

        AscendC::Reg::Muls<float, float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg2, vreg1, static_cast<float>(-1),

                                                                               preg0);

        AscendC::Reg::MulAddDst<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg4, vreg2, vreg1, preg0);

#endif

        AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(

            vreg5, bufferIn1Addr, addrReg);

        AscendC::Reg::Cast<float, bfloat16_t, castTraitTg0>(vreg6, vreg5, preg0);

        AscendC::Reg::Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg7, vreg4, vreg6, preg0);

        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(

            bufferOut0Addr, vreg7, addrReg, preg0);
    }
}

__simd_vf__ inline void TanhGradComb2VF(__ubuf__ float* bufferIn0Addr, __ubuf__ half* bufferIn1Addr,

                                        __ubuf__ float* bufferOut0Addr, int64_t dataLength)

{
    RegTensor<float> vreg0;

    RegTensor<float> vreg1;

    RegTensor<float> vreg2;

    RegTensor<float> vreg3;

    RegTensor<half> vreg4;

    RegTensor<float> vreg5;

    RegTensor<float> vreg6;

    MaskReg preg0;

    uint32_t size = dataLength;

    uint16_t vfLoopNum = (dataLength + (AscendC::VECTOR_REG_WIDTH / 4) - 1) / (AscendC::VECTOR_REG_WIDTH / 4);

    for (uint16_t i = 0; i < vfLoopNum; i++) {
        preg0 = AscendC::Reg::UpdateMask<float>(size);

        AscendC::Reg::AddrReg addrReg = AscendC::Reg::CreateAddrReg<float>(i, (AscendC::VECTOR_REG_WIDTH / 4));

        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(

            vreg0, bufferIn0Addr, addrReg);

        AscendC::Reg::Duplicate<float, float>(vreg3, 1.0);

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202)

        AscendC::Reg::FusedMulSubDst<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg3, vreg0, vreg0, preg0);

#else

        AscendC::Reg::Muls<float, float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg2, vreg0, static_cast<float>(-1),

                                                                               preg0);

        AscendC::Reg::MulAddDst<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg3, vreg2, vreg0, preg0);

#endif

        AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(

            vreg4, bufferIn1Addr, addrReg);

        AscendC::Reg::Cast<float, half, castTraitTg0>(vreg5, vreg4, preg0);

        AscendC::Reg::Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg6, vreg3, vreg5, preg0);

        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(

            bufferOut0Addr, vreg6, addrReg, preg0);
    }
}

__simd_vf__ inline void TanhGradComb3VF(__ubuf__ float* bufferIn0Addr, __ubuf__ bfloat16_t* bufferIn1Addr,

                                        __ubuf__ float* bufferOut0Addr, int64_t dataLength)

{
    RegTensor<float> vreg0;

    RegTensor<float> vreg1;

    RegTensor<float> vreg2;

    RegTensor<float> vreg3;

    RegTensor<bfloat16_t> vreg4;

    RegTensor<float> vreg5;

    RegTensor<float> vreg6;

    MaskReg preg0;

    uint32_t size = dataLength;

    uint16_t vfLoopNum = (dataLength + (AscendC::VECTOR_REG_WIDTH / 4) - 1) / (AscendC::VECTOR_REG_WIDTH / 4);

    for (uint16_t i = 0; i < vfLoopNum; i++) {
        preg0 = AscendC::Reg::UpdateMask<float>(size);

        AscendC::Reg::AddrReg addrReg = AscendC::Reg::CreateAddrReg<float>(i, (AscendC::VECTOR_REG_WIDTH / 4));

        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(

            vreg0, bufferIn0Addr, addrReg);

        AscendC::Reg::Duplicate<float, float>(vreg3, 1.0);

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202)

        AscendC::Reg::FusedMulSubDst<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg3, vreg0, vreg0, preg0);

#else

        AscendC::Reg::Muls<float, float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg2, vreg0, static_cast<float>(-1),

                                                                               preg0);

        AscendC::Reg::MulAddDst<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg3, vreg2, vreg0, preg0);

#endif

        AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(

            vreg4, bufferIn1Addr, addrReg);

        AscendC::Reg::Cast<float, bfloat16_t, castTraitTg0>(vreg5, vreg4, preg0);

        AscendC::Reg::Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg6, vreg3, vreg5, preg0);

        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(

            bufferOut0Addr, vreg6, addrReg, preg0);
    }
}

__simd_vf__ inline void TanhGradComb4VF(__ubuf__ bfloat16_t* bufferIn0Addr, __ubuf__ half* bufferIn1Addr,

                                        __ubuf__ float* bufferOut0Addr, int64_t dataLength)

{
    RegTensor<bfloat16_t> vreg0;

    RegTensor<float> vreg1;

    RegTensor<float> vreg2;

    RegTensor<float> vreg3;

    RegTensor<float> vreg4;

    RegTensor<half> vreg5;

    RegTensor<float> vreg6;

    RegTensor<float> vreg7;

    MaskReg preg0;

    uint32_t size = dataLength;

    uint16_t vfLoopNum = (dataLength + (AscendC::VECTOR_REG_WIDTH / 4) - 1) / (AscendC::VECTOR_REG_WIDTH / 4);

    for (uint16_t i = 0; i < vfLoopNum; i++) {
        preg0 = AscendC::Reg::UpdateMask<float>(size);

        AscendC::Reg::AddrReg addrReg = AscendC::Reg::CreateAddrReg<float>(i, (AscendC::VECTOR_REG_WIDTH / 4));

        AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(

            vreg0, bufferIn0Addr, addrReg);

        AscendC::Reg::Cast<float, bfloat16_t, castTraitTg0>(vreg1, vreg0, preg0);

        AscendC::Reg::Duplicate<float, float>(vreg4, 1.0);

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202)

        AscendC::Reg::FusedMulSubDst<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg4, vreg1, vreg1, preg0);

#else

        AscendC::Reg::Muls<float, float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg2, vreg1, static_cast<float>(-1),

                                                                               preg0);

        AscendC::Reg::MulAddDst<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg4, vreg2, vreg1, preg0);

#endif

        AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(

            vreg5, bufferIn1Addr, addrReg);

        AscendC::Reg::Cast<float, half, castTraitTg0>(vreg6, vreg5, preg0);

        AscendC::Reg::Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg7, vreg4, vreg6, preg0);

        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(

            bufferOut0Addr, vreg7, addrReg, preg0);
    }
}

__simd_vf__ inline void TanhGradComb5VF(__ubuf__ bfloat16_t* bufferIn0Addr, __ubuf__ float* bufferIn1Addr,

                                        __ubuf__ float* bufferOut0Addr, int64_t dataLength)

{
    RegTensor<bfloat16_t> vreg0;

    RegTensor<float> vreg1;

    RegTensor<float> vreg2;

    RegTensor<float> vreg3;

    RegTensor<float> vreg4;

    RegTensor<float> vreg5;

    RegTensor<float> vreg6;

    MaskReg preg0;

    uint32_t size = dataLength;

    uint16_t vfLoopNum = (dataLength + (AscendC::VECTOR_REG_WIDTH / 4) - 1) / (AscendC::VECTOR_REG_WIDTH / 4);

    for (uint16_t i = 0; i < vfLoopNum; i++) {
        preg0 = AscendC::Reg::UpdateMask<float>(size);

        AscendC::Reg::AddrReg addrReg = AscendC::Reg::CreateAddrReg<float>(i, (AscendC::VECTOR_REG_WIDTH / 4));

        AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(

            vreg0, bufferIn0Addr, addrReg);

        AscendC::Reg::Cast<float, bfloat16_t, castTraitTg0>(vreg1, vreg0, preg0);

        AscendC::Reg::Duplicate<float, float>(vreg4, 1.0);

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202)

        AscendC::Reg::FusedMulSubDst<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg4, vreg1, vreg1, preg0);

#else

        AscendC::Reg::Muls<float, float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg2, vreg1, static_cast<float>(-1),

                                                                               preg0);

        AscendC::Reg::MulAddDst<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg4, vreg2, vreg1, preg0);

#endif

        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(

            vreg5, bufferIn1Addr, addrReg);

        AscendC::Reg::Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg6, vreg4, vreg5, preg0);

        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(

            bufferOut0Addr, vreg6, addrReg, preg0);
    }
}

} // namespace TanhGrad

#endif

#endif // ASCENDC_TANH_GRAD_VF_H_
