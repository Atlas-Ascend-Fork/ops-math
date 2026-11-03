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
 * \file pow_tensor_scalar_vf.h
 * \brief Register-level vector functions for Pow (tensor * scalar) on Ascend 950.
 *
 * These __simd_vf__ functions operate on UB addresses and scalar arguments.
 * GM transfers, synchronization, and dispatch are handled by the caller.
 *
 * Addressing convention: AddrReg stride unit is the element count of the
 * CreateAddrReg template type (b16 unpack lanes count as float elements;
 * b8 unpack lanes count as uint16_t/int16_t elements), matching the original
 * srcAddr + i * (VECTOR_REG_WIDTH / sizeof(lane)) offsets.
 */

#ifndef ASCENDC_POW_TENSOR_SCALAR_VF_H_

#define ASCENDC_POW_TENSOR_SCALAR_VF_H_

#include "kernel_operator.h"

#ifdef __CCE_AICORE__

namespace Pow {

using namespace AscendC;

constexpr static Reg::CastTrait castTraitFlt0 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,

                                                 Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

constexpr static Reg::CastTrait castTraitFlt1 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,

                                                 Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

constexpr static Reg::CastTrait castTraitInt0 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,

                                                 Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

constexpr static Reg::CastTrait castTraitInt1 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,

                                                 Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

constexpr static Reg::CastTrait castTraitInt2 = {Reg::RegLayout::UNKNOWN, Reg::SatMode::UNKNOWN,

                                                 Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

constexpr static Reg::CastTrait castTraitInt3 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,

                                                 Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

template <typename T1>

__simd_vf__ inline void PowScalarSqrtVF(__ubuf__ T1* dstAddr, __ubuf__ T1* srcAddr, int64_t dataLength)

{
    if constexpr (sizeof(T1) == sizeof(float)) {
        Reg::RegTensor<float> vreg0;

        Reg::RegTensor<float> vreg1;

        Reg::MaskReg preg0;

        uint32_t size = dataLength;

        uint16_t vfLoopNum = (dataLength + (VECTOR_REG_WIDTH / sizeof(float)) - 1) / (VECTOR_REG_WIDTH / sizeof(float));

        for (uint16_t i = 0; i < vfLoopNum; i++) {
            preg0 = Reg::UpdateMask<float>(size);

            Reg::AddrReg addrReg = Reg::CreateAddrReg<float>(i, (VECTOR_REG_WIDTH / sizeof(float)));

            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(vreg0, srcAddr, addrReg);

            Reg::Sqrt<float, Reg::MaskMergeMode::ZEROING>(vreg1, vreg0, preg0);

            Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(

                dstAddr, vreg1, addrReg, preg0);
        }

    } else {
        Reg::RegTensor<T1> vreg0;

        Reg::RegTensor<float> vreg1;

        Reg::RegTensor<float> vreg2;

        Reg::RegTensor<T1> vreg3;

        Reg::MaskReg preg0;

        uint32_t size = dataLength;

        uint16_t vfLoopNum = (dataLength + (VECTOR_REG_WIDTH / sizeof(float)) - 1) / (VECTOR_REG_WIDTH / sizeof(float));

        for (uint16_t i = 0; i < vfLoopNum; i++) {
            preg0 = Reg::UpdateMask<float>(size);

            Reg::AddrReg addrReg = Reg::CreateAddrReg<float>(i, (VECTOR_REG_WIDTH / sizeof(float)));

            Reg::LoadAlign<T1, Reg::LoadDist::DIST_UNPACK_B16>(

                vreg0, srcAddr, addrReg);

            Reg::Cast<float, T1, castTraitFlt0>(vreg1, vreg0, preg0);

            Reg::Sqrt<float, Reg::MaskMergeMode::ZEROING>(vreg2, vreg1, preg0);

            Reg::Cast<T1, float, castTraitFlt1>(vreg3, vreg2, preg0);

            Reg::StoreAlign<T1, Reg::StoreDist::DIST_PACK_B32>(

                dstAddr, vreg3, addrReg, preg0);
        }
    }
}

template <typename T1>

__simd_vf__ inline void PowScalarRsqrtVF(__ubuf__ T1* dstAddr, __ubuf__ T1* srcAddr, int64_t dataLength)

{
    if constexpr (sizeof(T1) == sizeof(float)) {
        Reg::RegTensor<float> vreg0;

        Reg::RegTensor<float> vreg1;

        Reg::RegTensor<float> vreg2;

        Reg::RegTensor<float> vreg3;

        Reg::MaskReg preg0;

        uint32_t size = dataLength;

        uint16_t vfLoopNum = (dataLength + (VECTOR_REG_WIDTH / sizeof(float)) - 1) / (VECTOR_REG_WIDTH / sizeof(float));

        for (uint16_t i = 0; i < vfLoopNum; i++) {
            preg0 = Reg::UpdateMask<float>(size);

            Reg::AddrReg addrReg = Reg::CreateAddrReg<float>(i, (VECTOR_REG_WIDTH / sizeof(float)));

            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(vreg0, srcAddr, addrReg);

            Reg::Duplicate<float, Reg::MaskMergeMode::ZEROING>(vreg1, static_cast<float>(1.0f), preg0);

            Reg::Sqrt<float, Reg::MaskMergeMode::ZEROING>(vreg2, vreg0, preg0);

            Reg::Div<float, Reg::MaskMergeMode::ZEROING>(vreg3, vreg1, vreg2, preg0);

            Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(

                dstAddr, vreg3, addrReg, preg0);
        }

    } else {
        Reg::RegTensor<T1> vreg0;

        Reg::RegTensor<float> vreg1;

        Reg::RegTensor<float> vreg2;

        Reg::RegTensor<float> vreg3;

        Reg::RegTensor<float> vreg4;

        Reg::RegTensor<T1> vreg5;

        Reg::MaskReg preg0;

        uint32_t size = dataLength;

        uint16_t vfLoopNum = (dataLength + (VECTOR_REG_WIDTH / sizeof(float)) - 1) / (VECTOR_REG_WIDTH / sizeof(float));

        for (uint16_t i = 0; i < vfLoopNum; i++) {
            preg0 = Reg::UpdateMask<float>(size);

            Reg::AddrReg addrReg = Reg::CreateAddrReg<float>(i, (VECTOR_REG_WIDTH / sizeof(float)));

            Reg::LoadAlign<T1, Reg::LoadDist::DIST_UNPACK_B16>(

                vreg0, srcAddr, addrReg);

            Reg::Cast<float, T1, castTraitFlt0>(vreg1, vreg0, preg0);

            Reg::Duplicate<float, Reg::MaskMergeMode::ZEROING>(vreg2, static_cast<float>(1.0f), preg0);

            Reg::Sqrt<float, Reg::MaskMergeMode::ZEROING>(vreg3, vreg1, preg0);

            Reg::Div<float, Reg::MaskMergeMode::ZEROING>(vreg4, vreg2, vreg3, preg0);

            Reg::Cast<T1, float, castTraitFlt1>(vreg5, vreg4, preg0);

            Reg::StoreAlign<T1, Reg::StoreDist::DIST_PACK_B32>(

                dstAddr, vreg5, addrReg, preg0);
        }
    }
}

template <typename T1>

__simd_vf__ inline void PowScalarSquareVF(__ubuf__ T1* dstAddr, __ubuf__ T1* srcAddr, int64_t dataLength)

{
    if constexpr (sizeof(T1) == sizeof(float)) {
        Reg::RegTensor<float> vreg0;

        Reg::RegTensor<float> vreg1;

        Reg::MaskReg preg0;

        uint32_t size = dataLength;

        uint16_t vfLoopNum = (dataLength + (VECTOR_REG_WIDTH / sizeof(float)) - 1) / (VECTOR_REG_WIDTH / sizeof(float));

        for (uint16_t i = 0; i < vfLoopNum; i++) {
            preg0 = Reg::UpdateMask<float>(size);

            Reg::AddrReg addrReg = Reg::CreateAddrReg<float>(i, (VECTOR_REG_WIDTH / sizeof(float)));

            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(vreg0, srcAddr, addrReg);

            Reg::Mul<float, Reg::MaskMergeMode::ZEROING>(vreg1, vreg0, vreg0, preg0);

            Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(

                dstAddr, vreg1, addrReg, preg0);
        }

    } else {
        Reg::RegTensor<T1> vreg0;

        Reg::RegTensor<float> vreg1;

        Reg::RegTensor<float> vreg2;

        Reg::RegTensor<T1> vreg3;

        Reg::MaskReg preg0;

        uint32_t size = dataLength;

        uint16_t vfLoopNum = (dataLength + (VECTOR_REG_WIDTH / sizeof(float)) - 1) / (VECTOR_REG_WIDTH / sizeof(float));

        for (uint16_t i = 0; i < vfLoopNum; i++) {
            preg0 = Reg::UpdateMask<float>(size);

            Reg::AddrReg addrReg = Reg::CreateAddrReg<float>(i, (VECTOR_REG_WIDTH / sizeof(float)));

            Reg::LoadAlign<T1, Reg::LoadDist::DIST_UNPACK_B16>(

                vreg0, srcAddr, addrReg);

            Reg::Cast<float, T1, castTraitFlt0>(vreg1, vreg0, preg0);

            Reg::Mul<float, Reg::MaskMergeMode::ZEROING>(vreg2, vreg1, vreg1, preg0);

            Reg::Cast<T1, float, castTraitFlt1>(vreg3, vreg2, preg0);

            Reg::StoreAlign<T1, Reg::StoreDist::DIST_PACK_B32>(

                dstAddr, vreg3, addrReg, preg0);
        }
    }
}

template <typename T1>

__simd_vf__ inline void PowScalarCubeVF(__ubuf__ T1* dstAddr, __ubuf__ T1* srcAddr, int64_t dataLength)

{
    if constexpr (sizeof(T1) == sizeof(float)) {
        Reg::RegTensor<float> vreg0;

        Reg::RegTensor<float> vreg1;

        Reg::RegTensor<float> vreg2;

        Reg::MaskReg preg0;

        uint32_t size = dataLength;

        uint16_t vfLoopNum = (dataLength + (VECTOR_REG_WIDTH / sizeof(float)) - 1) / (VECTOR_REG_WIDTH / sizeof(float));

        for (uint16_t i = 0; i < vfLoopNum; i++) {
            preg0 = Reg::UpdateMask<float>(size);

            Reg::AddrReg addrReg = Reg::CreateAddrReg<float>(i, (VECTOR_REG_WIDTH / sizeof(float)));

            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(vreg0, srcAddr, addrReg);

            Reg::Mul<float, Reg::MaskMergeMode::ZEROING>(vreg1, vreg0, vreg0, preg0);

            Reg::Mul<float, Reg::MaskMergeMode::ZEROING>(vreg2, vreg1, vreg0, preg0);

            Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(

                dstAddr, vreg2, addrReg, preg0);
        }

    } else {
        Reg::RegTensor<T1> vreg0;

        Reg::RegTensor<float> vreg1;

        Reg::RegTensor<float> vreg2;

        Reg::RegTensor<float> vreg3;

        Reg::RegTensor<T1> vreg4;

        Reg::MaskReg preg0;

        uint32_t size = dataLength;

        uint16_t vfLoopNum = (dataLength + (VECTOR_REG_WIDTH / sizeof(float)) - 1) / (VECTOR_REG_WIDTH / sizeof(float));

        for (uint16_t i = 0; i < vfLoopNum; i++) {
            preg0 = Reg::UpdateMask<float>(size);

            Reg::AddrReg addrReg = Reg::CreateAddrReg<float>(i, (VECTOR_REG_WIDTH / sizeof(float)));

            Reg::LoadAlign<T1, Reg::LoadDist::DIST_UNPACK_B16>(

                vreg0, srcAddr, addrReg);

            Reg::Cast<float, T1, castTraitFlt0>(vreg1, vreg0, preg0);

            Reg::Mul<float, Reg::MaskMergeMode::ZEROING>(vreg2, vreg1, vreg1, preg0);

            Reg::Mul<float, Reg::MaskMergeMode::ZEROING>(vreg3, vreg2, vreg1, preg0);

            Reg::Cast<T1, float, castTraitFlt1>(vreg4, vreg3, preg0);

            Reg::StoreAlign<T1, Reg::StoreDist::DIST_PACK_B32>(

                dstAddr, vreg4, addrReg, preg0);
        }
    }
}

template <typename T1>

__simd_vf__ inline void PowScalarRecVF(__ubuf__ T1* dstAddr, __ubuf__ T1* srcAddr, int64_t dataLength)

{
    static constexpr Reg::DivSpecificMode divMode = {Reg::MaskMergeMode::ZEROING, true};

    if constexpr (sizeof(T1) == sizeof(float)) {
        Reg::RegTensor<float> vreg0;

        Reg::RegTensor<float> vreg1;

        Reg::RegTensor<float> vreg2;

        Reg::MaskReg preg0;

        uint32_t size = dataLength;

        uint16_t vfLoopNum = (dataLength + (VECTOR_REG_WIDTH / sizeof(float)) - 1) / (VECTOR_REG_WIDTH / sizeof(float));

        Reg::Duplicate(vreg0, 1.0f);

        for (uint16_t i = 0; i < vfLoopNum; i++) {
            preg0 = Reg::UpdateMask<float>(size);

            Reg::AddrReg addrReg = Reg::CreateAddrReg<float>(i, (VECTOR_REG_WIDTH / sizeof(float)));

            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(vreg1, srcAddr, addrReg);

            Reg::Div<float, &divMode>(vreg2, vreg0, vreg1, preg0);

            Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(

                dstAddr, vreg2, addrReg, preg0);
        }

    } else {
        Reg::RegTensor<float> vreg0;

        Reg::RegTensor<T1> vreg1;

        Reg::RegTensor<float> vreg2;

        Reg::RegTensor<float> vreg3;

        Reg::RegTensor<T1> vreg4;

        Reg::MaskReg preg0;

        uint32_t size = dataLength;

        uint16_t vfLoopNum = (dataLength + (VECTOR_REG_WIDTH / sizeof(float)) - 1) / (VECTOR_REG_WIDTH / sizeof(float));

        Reg::Duplicate(vreg0, 1.0f);

        for (uint16_t i = 0; i < vfLoopNum; i++) {
            preg0 = Reg::UpdateMask<float>(size);

            Reg::AddrReg addrReg = Reg::CreateAddrReg<float>(i, (VECTOR_REG_WIDTH / sizeof(float)));

            Reg::LoadAlign<T1, Reg::LoadDist::DIST_UNPACK_B16>(

                vreg1, srcAddr, addrReg);

            Reg::Cast<float, T1, castTraitFlt0>(vreg2, vreg1, preg0);

            Reg::Div<float, &divMode>(vreg3, vreg0, vreg2, preg0);

            Reg::Cast<T1, float, castTraitFlt1>(vreg4, vreg3, preg0);

            Reg::StoreAlign<T1, Reg::StoreDist::DIST_PACK_B32>(

                dstAddr, vreg4, addrReg, preg0);
        }
    }
}

template <typename T1>

__simd_vf__ inline void PowScalarSquareRecVF(__ubuf__ T1* dstAddr, __ubuf__ T1* srcAddr, int64_t dataLength)

{
    static constexpr Reg::DivSpecificMode divMode = {Reg::MaskMergeMode::ZEROING, true};

    if constexpr (sizeof(T1) == sizeof(float)) {
        Reg::RegTensor<float> vreg0;

        Reg::RegTensor<float> vreg1;

        Reg::RegTensor<float> vreg2;

        Reg::RegTensor<float> vreg3;

        Reg::MaskReg preg0;

        uint32_t size = dataLength;

        uint16_t vfLoopNum = (dataLength + (VECTOR_REG_WIDTH / sizeof(float)) - 1) / (VECTOR_REG_WIDTH / sizeof(float));

        Reg::Duplicate(vreg0, 1.0f);

        for (uint16_t i = 0; i < vfLoopNum; i++) {
            preg0 = Reg::UpdateMask<float>(size);

            Reg::AddrReg addrReg = Reg::CreateAddrReg<float>(i, (VECTOR_REG_WIDTH / sizeof(float)));

            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(vreg1, srcAddr, addrReg);

            Reg::Mul<float, Reg::MaskMergeMode::ZEROING>(vreg2, vreg1, vreg1, preg0);

            Reg::Div<float, &divMode>(vreg3, vreg0, vreg2, preg0);

            Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(

                dstAddr, vreg3, addrReg, preg0);
        }

    } else {
        Reg::RegTensor<float> vreg0;

        Reg::RegTensor<T1> vreg1;

        Reg::RegTensor<float> vreg2;

        Reg::RegTensor<float> vreg3;

        Reg::RegTensor<float> vreg4;

        Reg::RegTensor<T1> vreg5;

        Reg::MaskReg preg0;

        uint32_t size = dataLength;

        uint16_t vfLoopNum = (dataLength + (VECTOR_REG_WIDTH / sizeof(float)) - 1) / (VECTOR_REG_WIDTH / sizeof(float));

        Reg::Duplicate(vreg0, 1.0f);

        for (uint16_t i = 0; i < vfLoopNum; i++) {
            preg0 = Reg::UpdateMask<float>(size);

            Reg::AddrReg addrReg = Reg::CreateAddrReg<float>(i, (VECTOR_REG_WIDTH / sizeof(float)));

            Reg::LoadAlign<T1, Reg::LoadDist::DIST_UNPACK_B16>(

                vreg1, srcAddr, addrReg);

            Reg::Cast<float, T1, castTraitFlt0>(vreg2, vreg1, preg0);

            Reg::Mul<float, Reg::MaskMergeMode::ZEROING>(vreg3, vreg2, vreg2, preg0);

            Reg::Div<float, &divMode>(vreg4, vreg0, vreg3, preg0);

            Reg::Cast<T1, float, castTraitFlt1>(vreg5, vreg4, preg0);

            Reg::StoreAlign<T1, Reg::StoreDist::DIST_PACK_B32>(

                dstAddr, vreg5, addrReg, preg0);
        }
    }
}

template <typename T1>

__simd_vf__ inline void PowScalarIntSquareVF(__ubuf__ T1* dstAddr, __ubuf__ T1* srcAddr, int64_t dataLength)

{
    if constexpr (IsSameType<T1, int32_t>::value || IsSameType<T1, int16_t>::value) {
        Reg::RegTensor<T1> vreg0;

        Reg::RegTensor<T1> vreg1;

        Reg::MaskReg preg0;

        uint32_t size = dataLength;

        uint16_t vfLoopNum = (dataLength + (VECTOR_REG_WIDTH / sizeof(T1)) - 1) / (VECTOR_REG_WIDTH / sizeof(T1));

        for (uint16_t i = 0; i < vfLoopNum; i++) {
            preg0 = Reg::UpdateMask<T1>(size);

            Reg::AddrReg addrReg = Reg::CreateAddrReg<T1>(i, (VECTOR_REG_WIDTH / sizeof(T1)));

            Reg::LoadAlign<T1, Reg::LoadDist::DIST_NORM>(vreg0, srcAddr, addrReg);

            Reg::Mul<T1, Reg::MaskMergeMode::ZEROING>(vreg1, vreg0, vreg0, preg0);

            Reg::StoreAlign<T1, Reg::StoreDist::DIST_NORM_B32>(

                dstAddr, vreg1, addrReg, preg0);
        }

    } else if constexpr (IsSameType<T1, uint8_t>::value) {
        Reg::RegTensor<T1> vreg0;

        Reg::RegTensor<uint16_t> scalar0;

        Reg::RegTensor<uint16_t> vreg1;

        Reg::RegTensor<uint16_t> vreg2;

        Reg::RegTensor<uint16_t> vreg3;

        Reg::RegTensor<T1> vreg4;

        Reg::MaskReg preg0;

        uint32_t size = dataLength;

        uint16_t vfLoopNum = (dataLength + (VECTOR_REG_WIDTH / sizeof(uint16_t)) - 1) /

                             (VECTOR_REG_WIDTH / sizeof(uint16_t));

        Reg::Duplicate(scalar0, uint16_t(255));

        for (uint16_t i = 0; i < vfLoopNum; i++) {
            preg0 = Reg::UpdateMask<uint16_t>(size);

            Reg::AddrReg addrReg = Reg::CreateAddrReg<uint16_t>(i, (VECTOR_REG_WIDTH / sizeof(uint16_t)));

            Reg::LoadAlign<T1, Reg::LoadDist::DIST_UNPACK_B8>(

                vreg0, srcAddr, addrReg);

            Reg::Cast<uint16_t, T1, castTraitInt0>(vreg1, vreg0, preg0);

            Reg::Mul<uint16_t, Reg::MaskMergeMode::ZEROING>(vreg2, vreg1, vreg1, preg0);

            Reg::And<uint16_t, Reg::MaskMergeMode::ZEROING>(vreg3, vreg2, scalar0, preg0);

            Reg::Cast<T1, uint16_t, castTraitInt1>(vreg4, vreg3, preg0);

            Reg::StoreAlign<T1, Reg::StoreDist::DIST_PACK_B16>(

                dstAddr, vreg4, addrReg, preg0);
        }

    } else {
        Reg::RegTensor<T1> vreg0;

        Reg::RegTensor<int16_t> scalar0;

        Reg::RegTensor<int16_t> vreg1;

        Reg::RegTensor<int16_t> vreg2;

        Reg::RegTensor<int16_t> vreg3;

        Reg::RegTensor<half> vreg4;

        Reg::RegTensor<T1> vreg5;

        Reg::MaskReg preg0;

        uint32_t size = dataLength;

        uint16_t vfLoopNum = (dataLength + (VECTOR_REG_WIDTH / sizeof(int16_t)) - 1) /

                             (VECTOR_REG_WIDTH / sizeof(int16_t));

        Reg::Duplicate(scalar0, int16_t(255));

        for (uint16_t i = 0; i < vfLoopNum; i++) {
            preg0 = Reg::UpdateMask<int16_t>(size);

            Reg::AddrReg addrReg = Reg::CreateAddrReg<int16_t>(i, (VECTOR_REG_WIDTH / sizeof(int16_t)));

            Reg::LoadAlign<T1, Reg::LoadDist::DIST_UNPACK_B8>(

                vreg0, srcAddr, addrReg);

            Reg::Cast<int16_t, T1, castTraitInt0>(vreg1, vreg0, preg0);

            Reg::Mul<int16_t, Reg::MaskMergeMode::ZEROING>(vreg2, vreg1, vreg1, preg0);

            Reg::And<int16_t, Reg::MaskMergeMode::ZEROING>(vreg3, vreg2, scalar0, preg0);

            Reg::Cast<half, int16_t, castTraitInt2>(vreg4, vreg3, preg0);

            Reg::Cast<T1, half, castTraitInt3>(vreg5, vreg4, preg0);

            Reg::StoreAlign<T1, Reg::StoreDist::DIST_PACK_B16>(

                dstAddr, vreg5, addrReg, preg0);
        }
    }
}

template <typename T1>

__simd_vf__ inline void PowScalarIntCubeVF(__ubuf__ T1* dstAddr, __ubuf__ T1* srcAddr, int64_t dataLength)

{
    if constexpr (IsSameType<T1, int32_t>::value || IsSameType<T1, int16_t>::value) {
        Reg::RegTensor<T1> vreg0;

        Reg::RegTensor<T1> vreg1;

        Reg::RegTensor<T1> vreg2;

        Reg::MaskReg preg0;

        uint32_t size = dataLength;

        uint16_t vfLoopNum = (dataLength + (VECTOR_REG_WIDTH / sizeof(T1)) - 1) / (VECTOR_REG_WIDTH / sizeof(T1));

        for (uint16_t i = 0; i < vfLoopNum; i++) {
            preg0 = Reg::UpdateMask<T1>(size);

            Reg::AddrReg addrReg = Reg::CreateAddrReg<T1>(i, (VECTOR_REG_WIDTH / sizeof(T1)));

            Reg::LoadAlign<T1, Reg::LoadDist::DIST_NORM>(vreg0, srcAddr, addrReg);

            Reg::Mul<T1, Reg::MaskMergeMode::ZEROING>(vreg1, vreg0, vreg0, preg0);

            Reg::Mul<T1, Reg::MaskMergeMode::ZEROING>(vreg2, vreg1, vreg0, preg0);

            Reg::StoreAlign<T1, Reg::StoreDist::DIST_NORM_B32>(

                dstAddr, vreg2, addrReg, preg0);
        }

    } else if constexpr (IsSameType<T1, uint8_t>::value) {
        Reg::RegTensor<T1> vreg0;

        Reg::RegTensor<uint16_t> scalar0;

        Reg::RegTensor<uint16_t> vreg1;

        Reg::RegTensor<uint16_t> vreg2;

        Reg::RegTensor<uint16_t> vreg3;

        Reg::RegTensor<uint16_t> vreg4;

        Reg::RegTensor<T1> vreg5;

        Reg::MaskReg preg0;

        uint32_t size = dataLength;

        uint16_t vfLoopNum = (dataLength + (VECTOR_REG_WIDTH / sizeof(uint16_t)) - 1) /

                             (VECTOR_REG_WIDTH / sizeof(uint16_t));

        Reg::Duplicate(scalar0, uint16_t(255));

        for (uint16_t i = 0; i < vfLoopNum; i++) {
            preg0 = Reg::UpdateMask<uint16_t>(size);

            Reg::AddrReg addrReg = Reg::CreateAddrReg<uint16_t>(i, (VECTOR_REG_WIDTH / sizeof(uint16_t)));

            Reg::LoadAlign<T1, Reg::LoadDist::DIST_UNPACK_B8>(

                vreg0, srcAddr, addrReg);

            Reg::Cast<uint16_t, T1, castTraitInt0>(vreg1, vreg0, preg0);

            Reg::Mul<uint16_t, Reg::MaskMergeMode::ZEROING>(vreg2, vreg1, vreg1, preg0);

            Reg::Mul<uint16_t, Reg::MaskMergeMode::ZEROING>(vreg3, vreg2, vreg1, preg0);

            Reg::And<uint16_t, Reg::MaskMergeMode::ZEROING>(vreg4, vreg3, scalar0, preg0);

            Reg::Cast<T1, uint16_t, castTraitInt1>(vreg5, vreg4, preg0);

            Reg::StoreAlign<T1, Reg::StoreDist::DIST_PACK_B16>(

                dstAddr, vreg5, addrReg, preg0);
        }

    } else {
        Reg::RegTensor<T1> vreg0;

        Reg::RegTensor<int16_t> scalar0;

        Reg::RegTensor<int16_t> vreg1;

        Reg::RegTensor<int16_t> vreg2;

        Reg::RegTensor<int16_t> vreg3;

        Reg::RegTensor<int16_t> vreg4;

        Reg::RegTensor<half> vreg5;

        Reg::RegTensor<T1> vreg6;

        Reg::MaskReg preg0;

        uint32_t size = dataLength;

        uint16_t vfLoopNum = (dataLength + (VECTOR_REG_WIDTH / sizeof(int16_t)) - 1) /

                             (VECTOR_REG_WIDTH / sizeof(int16_t));

        Reg::Duplicate(scalar0, int16_t(255));

        for (uint16_t i = 0; i < vfLoopNum; i++) {
            preg0 = Reg::UpdateMask<int16_t>(size);

            Reg::AddrReg addrReg = Reg::CreateAddrReg<int16_t>(i, (VECTOR_REG_WIDTH / sizeof(int16_t)));

            Reg::LoadAlign<T1, Reg::LoadDist::DIST_UNPACK_B8>(

                vreg0, srcAddr, addrReg);

            Reg::Cast<int16_t, T1, castTraitInt0>(vreg1, vreg0, preg0);

            Reg::Mul<int16_t, Reg::MaskMergeMode::ZEROING>(vreg2, vreg1, vreg1, preg0);

            Reg::Mul<int16_t, Reg::MaskMergeMode::ZEROING>(vreg3, vreg2, vreg1, preg0);

            Reg::And<int16_t, Reg::MaskMergeMode::ZEROING>(vreg4, vreg3, scalar0, preg0);

            Reg::Cast<half, int16_t, castTraitInt2>(vreg5, vreg4, preg0);

            Reg::Cast<T1, half, castTraitInt3>(vreg6, vreg5, preg0);

            Reg::StoreAlign<T1, Reg::StoreDist::DIST_PACK_B16>(

                dstAddr, vreg6, addrReg, preg0);
        }
    }
}

} // namespace Pow

#endif

#endif // ASCENDC_POW_TENSOR_SCALAR_VF_H_
