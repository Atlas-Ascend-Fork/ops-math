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
 * \file histogram_v2_scalar_cast.h
 * \brief Scalar cast helpers. The bisheng backend does not lower bf16 static_cast in SIMT
 *        scalar context (fatal error: not support bf16 type cast), so bfloat16_t must go
 *        through explicit intrinsics: __bfloat162float inside __simt_vf__ thread context
 *        (simt_api/asc_bf16.h), AscendC::ToFloat in the plain __aicore__ scalar path
 *        (kernel_scalar_convert.h). All other types keep plain static_cast.
 */
#ifndef HISTOGRAM_V2_SCALAR_CAST_H
#define HISTOGRAM_V2_SCALAR_CAST_H

#include "simt_api/asc_bf16.h"

namespace HistogramV2Common {

// __simt_vf__ thread context: X_TYPE -> COMPUTE_TYPE
template <typename COMPUTE_TYPE, typename X_TYPE>
__simt_callee__ __aicore__ inline COMPUTE_TYPE SimtCastTo(X_TYPE v)
{
    if constexpr (AscendC::IsSameType<X_TYPE, bfloat16_t>::value) {
        return static_cast<COMPUTE_TYPE>(__bfloat162float(v));
    } else {
        return static_cast<COMPUTE_TYPE>(v);
    }
}

// plain __aicore__ scalar path (kernel main flow): X_TYPE -> COMPUTE_TYPE
template <typename COMPUTE_TYPE, typename X_TYPE>
__aicore__ inline COMPUTE_TYPE ScalarCastTo(X_TYPE v)
{
    if constexpr (AscendC::IsSameType<X_TYPE, bfloat16_t>::value) {
        return static_cast<COMPUTE_TYPE>(AscendC::ToFloat(v));
    } else {
        return static_cast<COMPUTE_TYPE>(v);
    }
}

} // namespace HistogramV2Common

#endif // HISTOGRAM_V2_SCALAR_CAST_H
