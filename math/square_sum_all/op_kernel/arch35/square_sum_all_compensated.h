/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SQUARE_SUM_ALL_COMPENSATED_ARCH35_H_
#define SQUARE_SUM_ALL_COMPENSATED_ARCH35_H_

#include "kernel_operator.h"

namespace SquareSumAllAccuracy {
using namespace AscendC;
constexpr uint32_t REDUCTION_LANE_COUNT = 64;

// Retain the rounding residual of each addition through every reduction stage.
// Partial sums are nonnegative. Clear the residual for Inf/NaN so that Inf-Inf
// in TwoSum cannot turn a valid overflowing square sum into NaN.
__simd_callee__ inline void AddCompensated(Reg::RegTensor<float>& sum, Reg::RegTensor<float>& correction,
                                           Reg::RegTensor<float>& value, Reg::MaskReg mask)
{
    Reg::RegTensor<float> rounded;
    Reg::RegTensor<float> recovered;
    Reg::RegTensor<float> error;
    Reg::RegTensor<float> temp;
    Reg::RegTensor<float> zero;
    Reg::MaskReg finiteMask;
    Reg::Add(rounded, sum, value, mask);
    Reg::Sub(recovered, rounded, sum, mask);
    Reg::Sub(error, value, recovered, mask);
    Reg::Sub(temp, rounded, recovered, mask);
    Reg::Sub(temp, sum, temp, mask);
    Reg::Add(error, error, temp, mask);
    Reg::Add(correction, correction, error, mask);
    Reg::Duplicate(zero, 0.0f, mask);
    Reg::CompareScalar<float, CMPMODE::LE>(finiteMask, rounded, 3.4028234663852886e38f, mask);
    Reg::Select(correction, correction, zero, finiteMask);
    Reg::Copy(sum, rounded, mask);
}

__simd_callee__ inline void ReduceCompensatedPair64(Reg::RegTensor<float>& sum, Reg::RegTensor<float>& correction)
{
    Reg::RegTensor<uint32_t> baseIndex;
    Reg::RegTensor<uint32_t> gatherIndex;
    Reg::RegTensor<uint32_t> validIndexMask;
    Reg::RegTensor<float> gathered;
    Reg::RegTensor<float> gatheredCorrection;
    Reg::MaskReg indexMask = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::Arange((Reg::RegTensor<int32_t>&)baseIndex, static_cast<int32_t>(0));
    Reg::Duplicate(validIndexMask, REDUCTION_LANE_COUNT - 1, indexMask);
    for (uint16_t offset = REDUCTION_LANE_COUNT / 2; offset > 0; offset >>= 1) {
        Reg::Adds(gatherIndex, baseIndex, static_cast<uint32_t>(offset), indexMask);
        Reg::And(gatherIndex, gatherIndex, validIndexMask, indexMask);
        Reg::Gather(gathered, sum, gatherIndex);
        Reg::Gather(gatheredCorrection, correction, gatherIndex);
        uint32_t activeElements = offset;
        Reg::MaskReg reduceMask = Reg::UpdateMask<float>(activeElements);
        Reg::Add(correction, correction, gatheredCorrection, reduceMask);
        AddCompensated(sum, correction, gathered, reduceMask);
    }
}

// Keep both the rounded square and its FMA residual until the final output.
__simd_callee__ inline void AccumulateSquare(Reg::RegTensor<float>& sum, Reg::RegTensor<float>& correction,
                                             Reg::RegTensor<float>& input, Reg::MaskReg mask)
{
    Reg::RegTensor<float> square;
    Reg::RegTensor<float> productError;
    Reg::Mul(square, input, input, mask);
    Reg::Neg(productError, square, mask);
    Reg::MulAddDst(productError, input, input, mask);
    Reg::Add(correction, correction, productError, mask);
    // AddCompensated also clears Inf-Inf residuals for overflowing squares.
    AddCompensated(sum, correction, square, mask);
}

__simd_callee__ inline void ReduceCompensated64(Reg::RegTensor<float>& dst, Reg::RegTensor<float>& sum,
                                                Reg::RegTensor<float>& correction)
{
    ReduceCompensatedPair64(sum, correction);
    Reg::MaskReg oneMask = Reg::CreateMask<float, Reg::MaskPattern::VL1>();
    Reg::Add(dst, sum, correction, oneMask);
}

} // namespace SquareSumAllAccuracy

#endif // SQUARE_SUM_ALL_COMPENSATED_ARCH35_H_
