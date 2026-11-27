/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MASKED_FILL_RANGE_TILING_H_
#define MASKED_FILL_RANGE_TILING_H_

#include "exe_graph/runtime/tiling_context.h"
#include "register/tilingdata_base.h"

namespace optiling {

// Host registration mirror of the PyPTO-generated MaskedFillRange type. Keep field order
// and types in sync with op_kernel/masked_fill_range.py; the generated type is force-included
// into the tiling source by enable_pypto_kernel, as in math/pdist.
BEGIN_TILING_DATA_DEF(MaskedFillRangeTilingData)
TILING_DATA_FIELD_DEF(int64_t, rows)
TILING_DATA_FIELD_DEF(int64_t, cols)
TILING_DATA_FIELD_DEF(int64_t, rounds)
TILING_DATA_FIELD_DEF(int64_t, batches)
TILING_DATA_FIELD_DEF(int64_t, winStride)
TILING_DATA_FIELD_DEF(int64_t, repeat)
TILING_DATA_FIELD_DEF(int64_t, fillMode)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(MaskedFillRange, MaskedFillRangeTilingData)

struct MaskedFillRangeCompileInfo {};

} // namespace optiling

#endif // MASKED_FILL_RANGE_TILING_H_
