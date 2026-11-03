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
 * \file lerp_vf.h
 * \brief SIMD VF entry functions for Lerp on Ascend 950 (arch35).
 *
 * One VF function per dtype combination of (start, end, weight, y); the
 * with_loops and without_loops kernel classes of one dtype combination share
 * the same VF function. UB addresses and the element count are passed in as
 * plain arguments; GM transfers, broadcast and dispatch stay in the host
 * classes.
 *
 * Addressing convention: AddrReg stride unit is the element count of the
 * CreateAddrReg template type (b16 unpack lanes count as float elements,
 * i.e. VECTOR_REG_WIDTH / sizeof(float) per iteration), matching the
 * original xAddr + i * (VECTOR_REG_WIDTH / sizeof(float)) offsets.
 */

#ifndef ASCENDC_LERP_VF_H_

#define ASCENDC_LERP_VF_H_

#include "kernel_operator.h"

#ifdef __CCE_AICORE__

namespace Lerp {

using AscendC::Reg::MaskReg;

using AscendC::Reg::RegTensor;

constexpr static AscendC::Reg::CastTrait castTraitLp0 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,

                                                         AscendC::Reg::MaskMergeMode::ZEROING,

                                                         AscendC::RoundMode::UNKNOWN};

constexpr static AscendC::Reg::CastTrait castTraitLp1 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT,

                                                         AscendC::Reg::MaskMergeMode::ZEROING,

                                                         AscendC::RoundMode::CAST_RINT};

__simd_vf__ inline void LerpF16VF(__ubuf__ half* bufferIn0Addr, __ubuf__ half* bufferIn1Addr,

                                  __ubuf__ half* bufferIn2Addr, __ubuf__ half* bufferOut0Addr, int64_t dataLength)

{
    RegTensor<half> vreg0;

    RegTensor<float> vreg1;

    RegTensor<half> vreg2;

    RegTensor<float> vreg3;

    RegTensor<half> vreg4;

    RegTensor<float> vreg5;

    RegTensor<float> vreg6;

    RegTensor<float> vreg7;

    RegTensor<float> vreg8;

    RegTensor<float> vreg9;

    RegTensor<half> vreg10;

    MaskReg preg0;

    uint32_t size = dataLength;

    MaskReg preg2;

    preg0 = AscendC::Reg::CreateMask<float>();

    AscendC::Reg::Duplicate<float, AscendC::Reg::MaskMergeMode::ZEROING, float>(vreg7, static_cast<float>(1.0), preg0);

    uint16_t vfLoopNum = (dataLength + (AscendC::VECTOR_REG_WIDTH / 4) - 1) / (AscendC::VECTOR_REG_WIDTH / 4);

    for (uint16_t i = 0; i < vfLoopNum; i++) {
        preg0 = AscendC::Reg::UpdateMask<float>(size);

        AscendC::Reg::AddrReg addrReg = AscendC::Reg::CreateAddrReg<float>(i, (AscendC::VECTOR_REG_WIDTH / 4));

        AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(

            vreg0, bufferIn0Addr, addrReg);

        AscendC::Reg::Cast<float, half, castTraitLp0>(vreg1, vreg0, preg0);

        AscendC::Reg::Compares<float, AscendC::CMPMODE::LT>(preg2, vreg1, static_cast<float>(0.5), preg0);

        AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(

            vreg2, bufferIn1Addr, addrReg);

        AscendC::Reg::Cast<float, half, castTraitLp0>(vreg3, vreg2, preg0);

        AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(

            vreg4, bufferIn2Addr, addrReg);

        AscendC::Reg::Cast<float, half, castTraitLp0>(vreg5, vreg4, preg0);

        AscendC::Reg::Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg6, vreg3, vreg5, preg0);

        AscendC::Reg::MulAddDst<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg5, vreg1, vreg6, preg0);

        AscendC::Reg::Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg8, vreg1, vreg7, preg0);

        AscendC::Reg::MulAddDst<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg3, vreg8, vreg6, preg0);

        AscendC::Reg::Select<float>(vreg9, vreg5, vreg3, preg2);

        AscendC::Reg::Cast<half, float, castTraitLp1>(vreg10, vreg9, preg0);

        AscendC::Reg::StoreAlign<half, AscendC::Reg::StoreDist::DIST_PACK_B32>(

            bufferOut0Addr, vreg10, addrReg, preg0);
    }
}

__simd_vf__ inline void LerpF32VF(__ubuf__ float* bufferIn0Addr, __ubuf__ float* bufferIn1Addr,

                                  __ubuf__ float* bufferIn2Addr, __ubuf__ float* bufferOut0Addr, int64_t dataLength)

{
    RegTensor<float> vreg0;

    RegTensor<float> vreg1;

    RegTensor<float> vreg2;

    RegTensor<float> vreg3;

    RegTensor<float> vreg4;

    RegTensor<float> vreg5;

    RegTensor<float> vreg6;

    MaskReg preg0;

    uint32_t size = dataLength;

    MaskReg preg26;

    preg0 = AscendC::Reg::CreateMask<float>();

    AscendC::Reg::Duplicate<float, AscendC::Reg::MaskMergeMode::ZEROING, float>(vreg4, static_cast<float>(1.0), preg0);

    uint16_t vfLoopNum = (dataLength + (AscendC::VECTOR_REG_WIDTH / 4) - 1) / (AscendC::VECTOR_REG_WIDTH / 4);

    for (uint16_t i = 0; i < vfLoopNum; i++) {
        preg0 = AscendC::Reg::UpdateMask<float>(size);

        AscendC::Reg::AddrReg addrReg = AscendC::Reg::CreateAddrReg<float>(i, (AscendC::VECTOR_REG_WIDTH / 4));

        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(

            vreg0, bufferIn0Addr, addrReg);

        AscendC::Reg::Compares<float, AscendC::CMPMODE::LT>(preg26, vreg0, static_cast<float>(0.5), preg0);

        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(

            vreg1, bufferIn1Addr, addrReg);

        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(

            vreg2, bufferIn2Addr, addrReg);

        AscendC::Reg::Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg3, vreg1, vreg2, preg0);

        AscendC::Reg::MulAddDst<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg2, vreg0, vreg3, preg0);

        AscendC::Reg::Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg5, vreg0, vreg4, preg0);

        AscendC::Reg::MulAddDst<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg1, vreg5, vreg3, preg0);

        AscendC::Reg::Select<float>(vreg6, vreg2, vreg1, preg26);

        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(

            bufferOut0Addr, vreg6, addrReg, preg0);
    }
}

__simd_vf__ inline void LerpBf16VF(__ubuf__ bfloat16_t* bufferIn0Addr, __ubuf__ bfloat16_t* bufferIn1Addr,

                                   __ubuf__ bfloat16_t* bufferIn2Addr, __ubuf__ bfloat16_t* bufferOut0Addr,

                                   int64_t dataLength)

{
    RegTensor<bfloat16_t> vreg0;

    RegTensor<float> vreg1;

    RegTensor<bfloat16_t> vreg2;

    RegTensor<float> vreg3;

    RegTensor<bfloat16_t> vreg4;

    RegTensor<float> vreg5;

    RegTensor<float> vreg6;

    RegTensor<float> vreg7;

    RegTensor<float> vreg8;

    RegTensor<float> vreg9;

    RegTensor<bfloat16_t> vreg10;

    MaskReg preg0;

    uint32_t size = dataLength;

    MaskReg preg48;

    preg0 = AscendC::Reg::CreateMask<float>();

    AscendC::Reg::Duplicate<float, AscendC::Reg::MaskMergeMode::ZEROING, float>(vreg7, static_cast<float>(1.0), preg0);

    uint16_t vfLoopNum = (dataLength + (AscendC::VECTOR_REG_WIDTH / 4) - 1) / (AscendC::VECTOR_REG_WIDTH / 4);

    for (uint16_t i = 0; i < vfLoopNum; i++) {
        preg0 = AscendC::Reg::UpdateMask<float>(size);

        AscendC::Reg::AddrReg addrReg = AscendC::Reg::CreateAddrReg<float>(i, (AscendC::VECTOR_REG_WIDTH / 4));

        AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(

            vreg0, bufferIn0Addr, addrReg);

        AscendC::Reg::Cast<float, bfloat16_t, castTraitLp0>(vreg1, vreg0, preg0);

        AscendC::Reg::Compares<float, AscendC::CMPMODE::LT>(preg48, vreg1, static_cast<float>(0.5), preg0);

        AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(

            vreg2, bufferIn1Addr, addrReg);

        AscendC::Reg::Cast<float, bfloat16_t, castTraitLp0>(vreg3, vreg2, preg0);

        AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(

            vreg4, bufferIn2Addr, addrReg);

        AscendC::Reg::Cast<float, bfloat16_t, castTraitLp0>(vreg5, vreg4, preg0);

        AscendC::Reg::Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg6, vreg3, vreg5, preg0);

        AscendC::Reg::MulAddDst<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg5, vreg1, vreg6, preg0);

        AscendC::Reg::Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg8, vreg1, vreg7, preg0);

        AscendC::Reg::MulAddDst<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg3, vreg8, vreg6, preg0);

        AscendC::Reg::Select<float>(vreg9, vreg5, vreg3, preg48);

        AscendC::Reg::Cast<bfloat16_t, float, castTraitLp1>(vreg10, vreg9, preg0);

        AscendC::Reg::StoreAlign<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(

            bufferOut0Addr, vreg10, addrReg, preg0);
    }
}

__simd_vf__ inline void LerpComb0VF(__ubuf__ float* bufferIn0Addr, __ubuf__ bfloat16_t* bufferIn1Addr,

                                    __ubuf__ bfloat16_t* bufferIn2Addr, __ubuf__ bfloat16_t* bufferOut0Addr,

                                    int64_t dataLength)

{
    RegTensor<float> vreg0;

    RegTensor<bfloat16_t> vreg1;

    RegTensor<float> vreg2;

    RegTensor<bfloat16_t> vreg3;

    RegTensor<float> vreg4;

    RegTensor<float> vreg5;

    RegTensor<float> vreg6;

    RegTensor<float> vreg7;

    RegTensor<float> vreg8;

    RegTensor<bfloat16_t> vreg9;

    MaskReg preg0;

    uint32_t size = dataLength;

    MaskReg preg73;

    preg0 = AscendC::Reg::CreateMask<float>();

    AscendC::Reg::Duplicate<float, AscendC::Reg::MaskMergeMode::ZEROING, float>(vreg6, static_cast<float>(1.0), preg0);

    uint16_t vfLoopNum = (dataLength + (AscendC::VECTOR_REG_WIDTH / 4) - 1) / (AscendC::VECTOR_REG_WIDTH / 4);

    for (uint16_t i = 0; i < vfLoopNum; i++) {
        preg0 = AscendC::Reg::UpdateMask<float>(size);

        AscendC::Reg::AddrReg addrReg = AscendC::Reg::CreateAddrReg<float>(i, (AscendC::VECTOR_REG_WIDTH / 4));

        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(

            vreg0, bufferIn0Addr, addrReg);

        AscendC::Reg::Compares<float, AscendC::CMPMODE::LT>(preg73, vreg0, static_cast<float>(0.5), preg0);

        AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(

            vreg1, bufferIn1Addr, addrReg);

        AscendC::Reg::Cast<float, bfloat16_t, castTraitLp0>(vreg2, vreg1, preg0);

        AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(

            vreg3, bufferIn2Addr, addrReg);

        AscendC::Reg::Cast<float, bfloat16_t, castTraitLp0>(vreg4, vreg3, preg0);

        AscendC::Reg::Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg5, vreg2, vreg4, preg0);

        AscendC::Reg::MulAddDst<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg4, vreg0, vreg5, preg0);

        AscendC::Reg::Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg7, vreg0, vreg6, preg0);

        AscendC::Reg::MulAddDst<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg2, vreg7, vreg5, preg0);

        AscendC::Reg::Select<float>(vreg8, vreg4, vreg2, preg73);

        AscendC::Reg::Cast<bfloat16_t, float, castTraitLp1>(vreg9, vreg8, preg0);

        AscendC::Reg::StoreAlign<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(

            bufferOut0Addr, vreg9, addrReg, preg0);
    }
}

__simd_vf__ inline void LerpComb1VF(__ubuf__ float* bufferIn0Addr, __ubuf__ half* bufferIn1Addr,

                                    __ubuf__ half* bufferIn2Addr, __ubuf__ half* bufferOut0Addr, int64_t dataLength)

{
    RegTensor<float> vreg0;

    RegTensor<half> vreg1;

    RegTensor<float> vreg2;

    RegTensor<half> vreg3;

    RegTensor<float> vreg4;

    RegTensor<float> vreg5;

    RegTensor<float> vreg6;

    RegTensor<float> vreg7;

    RegTensor<float> vreg8;

    RegTensor<half> vreg9;

    MaskReg preg0;

    uint32_t size = dataLength;

    MaskReg preg97;

    preg0 = AscendC::Reg::CreateMask<float>();

    AscendC::Reg::Duplicate<float, AscendC::Reg::MaskMergeMode::ZEROING, float>(vreg6, static_cast<float>(1.0), preg0);

    uint16_t vfLoopNum = (dataLength + (AscendC::VECTOR_REG_WIDTH / 4) - 1) / (AscendC::VECTOR_REG_WIDTH / 4);

    for (uint16_t i = 0; i < vfLoopNum; i++) {
        preg0 = AscendC::Reg::UpdateMask<float>(size);

        AscendC::Reg::AddrReg addrReg = AscendC::Reg::CreateAddrReg<float>(i, (AscendC::VECTOR_REG_WIDTH / 4));

        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(

            vreg0, bufferIn0Addr, addrReg);

        AscendC::Reg::Compares<float, AscendC::CMPMODE::LT>(preg97, vreg0, static_cast<float>(0.5), preg0);

        AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(

            vreg1, bufferIn1Addr, addrReg);

        AscendC::Reg::Cast<float, half, castTraitLp0>(vreg2, vreg1, preg0);

        AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(

            vreg3, bufferIn2Addr, addrReg);

        AscendC::Reg::Cast<float, half, castTraitLp0>(vreg4, vreg3, preg0);

        AscendC::Reg::Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg5, vreg2, vreg4, preg0);

        AscendC::Reg::MulAddDst<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg4, vreg0, vreg5, preg0);

        AscendC::Reg::Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg7, vreg0, vreg6, preg0);

        AscendC::Reg::MulAddDst<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg2, vreg7, vreg5, preg0);

        AscendC::Reg::Select<float>(vreg8, vreg4, vreg2, preg97);

        AscendC::Reg::Cast<half, float, castTraitLp1>(vreg9, vreg8, preg0);

        AscendC::Reg::StoreAlign<half, AscendC::Reg::StoreDist::DIST_PACK_B32>(

            bufferOut0Addr, vreg9, addrReg, preg0);
    }
}

} // namespace Lerp

#endif

#endif // ASCENDC_LERP_VF_H_
