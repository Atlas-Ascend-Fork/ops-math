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
 * \file left_shift_dag.h
 * \brief left_shift dag
 */

#ifndef OPS_MATH_LEFT_SHIFT_DAG_H
#define OPS_MATH_LEFT_SHIFT_DAG_H

#include <type_traits>
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

namespace LeftShiftOp {
using namespace Ops::Base;

template <typename T>
struct LeftShiftCustom8 : public Vec::ElemwiseBinaryOP<T, T, T> {
    __aicore__ inline LeftShiftCustom8(LocalTensor<T>& dst, LocalTensor<T>& src1, LocalTensor<T>& src2, uint32_t count)
    {
#ifdef __CCE_AICORE__
        uint32_t dtypeSize = sizeof(T);
        int8_t leftShifts = 7;
        constexpr uint64_t VECTOR_REG_WIDTH = 256UL;
        uint32_t vl = VECTOR_REG_WIDTH / dtypeSize;
        uint16_t loopNum = (count + vl - 1) / vl;
        uint32_t vlSize = vl;

        __ubuf__ T* src1Addr = (__ubuf__ T*)src1.GetPhyAddr();
        __ubuf__ T* src2Addr = (__ubuf__ T*)src2.GetPhyAddr();
        __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();

        Reg::RegTensor<T> xReg;
        Reg::RegTensor<T> yReg;
        Reg::RegTensor<T> zReg;
        Reg::RegTensor<T> sizeReg;

        Reg::MaskReg mask;
        Reg::MaskReg validMask;

        __VEC_SCOPE__
        {
            Reg::MaskReg scalarMaskReg = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
            Reg::Duplicate(sizeReg, leftShifts, scalarMaskReg);

            for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                mask = Reg::UpdateMask<T, Reg::RegTraitNumOne>(count);
                Reg::DataCopy(xReg, (__ubuf__ T*)(src1Addr + loopIdx * vlSize));
                Reg::DataCopy(yReg, (__ubuf__ T*)(src2Addr + loopIdx * vlSize));

                // 位移量按无符号位模式比较: [0, bits-1] 有效, 负数/越界均为无效 -> 0
                if constexpr (std::is_same_v<T, int8_t>) {
                    Reg::Compare<uint8_t, CMPMODE::LE>(validMask, (Reg::RegTensor<uint8_t>&)yReg,
                                                       (Reg::RegTensor<uint8_t>&)sizeReg, mask);
                    Reg::ShiftLeft(zReg, xReg, (Reg::RegTensor<T>&)yReg, validMask);
                } else if constexpr (std::is_same_v<T, uint8_t>) {
                    Reg::Compare<T, CMPMODE::LE>(validMask, yReg, sizeReg, mask);
                    Reg::ShiftLeft(zReg, xReg, (Reg::RegTensor<int8_t>&)yReg, validMask);
                }
                Reg::DataCopy((__ubuf__ T*)(dstAddr + loopIdx * vlSize), zReg, mask);
            }
        }
#endif
    }
};

template <typename T>
struct LeftShiftCustom16 : public Vec::ElemwiseBinaryOP<T, T, T> {
    __aicore__ inline LeftShiftCustom16(LocalTensor<T>& dst, LocalTensor<T>& src1, LocalTensor<T>& src2, uint32_t count)
    {
#ifdef __CCE_AICORE__
        uint32_t dtypeSize = sizeof(T);
        int8_t leftShifts = 15;
        constexpr uint64_t VECTOR_REG_WIDTH = 256UL;
        uint32_t vl = VECTOR_REG_WIDTH / dtypeSize;
        uint16_t loopNum = (count + vl - 1) / vl;
        uint32_t vlSize = vl;

        __ubuf__ T* src1Addr = (__ubuf__ T*)src1.GetPhyAddr();
        __ubuf__ T* src2Addr = (__ubuf__ T*)src2.GetPhyAddr();
        __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();

        Reg::RegTensor<T> xReg;
        Reg::RegTensor<T> yReg;
        Reg::RegTensor<T> zReg;
        Reg::RegTensor<T> sizeReg;

        Reg::MaskReg mask;
        Reg::MaskReg validMask;

        __VEC_SCOPE__
        {
            Reg::MaskReg scalarMaskReg = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
            Reg::Duplicate(sizeReg, leftShifts, scalarMaskReg);

            for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                mask = Reg::UpdateMask<T, Reg::RegTraitNumOne>(count);
                Reg::DataCopy(xReg, (__ubuf__ T*)(src1Addr + loopIdx * vlSize));
                Reg::DataCopy(yReg, (__ubuf__ T*)(src2Addr + loopIdx * vlSize));

                // 位移量按无符号位模式比较: [0, bits-1] 有效, 负数/越界均为无效 -> 0
                if constexpr (std::is_same_v<T, int16_t>) {
                    Reg::Compare<uint16_t, CMPMODE::LE>(validMask, (Reg::RegTensor<uint16_t>&)yReg,
                                                        (Reg::RegTensor<uint16_t>&)sizeReg, mask);
                    Reg::ShiftLeft(zReg, xReg, (Reg::RegTensor<T>&)yReg, validMask);
                } else if constexpr (std::is_same_v<T, uint16_t>) {
                    Reg::Compare<T, CMPMODE::LE>(validMask, yReg, sizeReg, mask);
                    Reg::ShiftLeft(zReg, xReg, (Reg::RegTensor<int16_t>&)yReg, validMask);
                }
                Reg::DataCopy((__ubuf__ T*)(dstAddr + loopIdx * vlSize), zReg, mask);
            }
        }
#endif
    }
};

template <typename T>
struct LeftShiftCustom32 : public Vec::ElemwiseBinaryOP<T, T, T> {
    __aicore__ inline LeftShiftCustom32(LocalTensor<T>& dst, LocalTensor<T>& src1, LocalTensor<T>& src2, uint32_t count)
    {
#ifdef __CCE_AICORE__
        uint32_t dtypeSize = sizeof(T);
        int8_t leftShifts = 31;
        constexpr uint64_t VECTOR_REG_WIDTH = 256UL;
        uint32_t vl = VECTOR_REG_WIDTH / dtypeSize;
        uint16_t loopNum = (count + vl - 1) / vl;
        uint32_t vlSize = vl;

        __ubuf__ T* src1Addr = (__ubuf__ T*)src1.GetPhyAddr();
        __ubuf__ T* src2Addr = (__ubuf__ T*)src2.GetPhyAddr();
        __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();

        Reg::RegTensor<T> xReg;
        Reg::RegTensor<T> yReg;
        Reg::RegTensor<T> zReg;
        Reg::RegTensor<T> sizeReg;

        Reg::MaskReg mask;
        Reg::MaskReg validMask;

        __VEC_SCOPE__
        {
            Reg::MaskReg scalarMaskReg = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
            Reg::Duplicate(sizeReg, leftShifts, scalarMaskReg);

            for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                mask = Reg::UpdateMask<T, Reg::RegTraitNumOne>(count);
                Reg::DataCopy(xReg, (__ubuf__ T*)(src1Addr + loopIdx * vlSize));
                Reg::DataCopy(yReg, (__ubuf__ T*)(src2Addr + loopIdx * vlSize));

                // 位移量按无符号位模式比较: [0, bits-1] 有效, 负数/越界均为无效 -> 0
                if constexpr (std::is_same_v<T, int32_t>) {
                    Reg::Compare<uint32_t, CMPMODE::LE>(validMask, (Reg::RegTensor<uint32_t>&)yReg,
                                                        (Reg::RegTensor<uint32_t>&)sizeReg, mask);
                    Reg::ShiftLeft(zReg, xReg, (Reg::RegTensor<T>&)yReg, validMask);
                } else if constexpr (std::is_same_v<T, uint32_t>) {
                    Reg::Compare<T, CMPMODE::LE>(validMask, yReg, sizeReg, mask);
                    Reg::ShiftLeft(zReg, xReg, (Reg::RegTensor<int32_t>&)yReg, validMask);
                }
                Reg::DataCopy((__ubuf__ T*)(dstAddr + loopIdx * vlSize), zReg, mask);
            }
        }
#endif
    }
};

template <typename T>
struct LeftShiftCustom64 : public Vec::ElemwiseBinaryOP<T, T, T> {
    __aicore__ inline LeftShiftCustom64(LocalTensor<T>& dst, LocalTensor<T>& src1, LocalTensor<T>& src2, uint32_t count)
    {
#ifdef __CCE_AICORE__
        uint32_t dtypeSize = sizeof(T);
        int8_t leftShifts = 63;
        constexpr uint64_t VECTOR_REG_WIDTH = 256UL;
        uint32_t vl = VECTOR_REG_WIDTH / dtypeSize;
        uint16_t loopNum = (count + vl - 1) / vl;
        uint32_t vlSize = vl;

        __ubuf__ T* src1Addr = (__ubuf__ T*)src1.GetPhyAddr();
        __ubuf__ T* src2Addr = (__ubuf__ T*)src2.GetPhyAddr();
        __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();

        Reg::RegTensor<T> xReg;
        Reg::RegTensor<T> yReg;
        Reg::RegTensor<T> zReg;
        Reg::RegTensor<T> sizeReg;

        Reg::MaskReg mask;
        Reg::MaskReg validMask;

        __VEC_SCOPE__
        {
            Reg::MaskReg scalarMaskReg = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
            Reg::Duplicate(sizeReg, leftShifts, scalarMaskReg);

            for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                mask = Reg::UpdateMask<T, Reg::RegTraitNumOne>(count);
                Reg::DataCopy(xReg, (__ubuf__ T*)(src1Addr + loopIdx * vlSize));
                Reg::DataCopy(yReg, (__ubuf__ T*)(src2Addr + loopIdx * vlSize));

                // 位移量按无符号位模式比较: [0, bits-1] 有效, 负数/越界均为无效 -> 0。
                // validMask 直接作为 ShiftLeft 的谓词 (仅支持 MaskMergeMode::ZEROING),
                // 谓词外 lane 结果归零, 等价于 "Select(y,0) + ShiftLeft + Select(z,0)",
                // 省去两条展开的 64 位 Select。
                if constexpr (std::is_same_v<T, int64_t>) {
                    Reg::Compare<uint64_t, CMPMODE::LE>(validMask, (Reg::RegTensor<uint64_t>&)yReg,
                                                        (Reg::RegTensor<uint64_t>&)sizeReg, mask);
                    Reg::ShiftLeft(zReg, xReg, (Reg::RegTensor<T>&)yReg, validMask);
                } else if constexpr (std::is_same_v<T, uint64_t>) {
                    Reg::Compare<T, CMPMODE::LE>(validMask, yReg, sizeReg, mask);
                    Reg::ShiftLeft(zReg, xReg, (Reg::RegTensor<int64_t>&)yReg, validMask);
                }
                Reg::DataCopy((__ubuf__ T*)(dstAddr + loopIdx * vlSize), zReg, mask);
            }
        }
#endif
    }
};

template <typename T>
struct LeftShiftDag8 {
    using InputX = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using InputY = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;

    using OpResult = Bind<LeftShiftCustom8<T>, InputX, InputY>;

    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpResult>;

    // 指定输出节点
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T>
struct LeftShiftDag16 {
    using InputX = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using InputY = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;

    using OpResult = Bind<LeftShiftCustom16<T>, InputX, InputY>;

    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpResult>;

    // 指定输出节点
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T>
struct LeftShiftDag32 {
    using InputX = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using InputY = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;

    using OpResult = Bind<LeftShiftCustom32<T>, InputX, InputY>;

    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpResult>;

    // 指定输出节点
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T>
struct LeftShiftDag64 {
    using InputX = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using InputY = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;

    using OpResult = Bind<LeftShiftCustom64<T>, InputX, InputY>;

    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpResult>;

    // 指定输出节点
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace LeftShiftOp

#endif // OPS_MATH_LEFT_SHIFT_DAG_H
