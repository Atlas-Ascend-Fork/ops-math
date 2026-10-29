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
 * \file pdist_tiling.h
 * \brief
 */
#ifndef PDIST_TILING_H_
#define PDIST_TILING_H_

#include "exe_graph/runtime/tiling_context.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

// Registered tiling data mirror of the pypto kernel dataclass PdistTiling
// (op_kernel/pdist.py, force-included codegen header PdistTiling_tiling.h).
// Field order and types must stay in sync with the dataclass:
//   N: int / M: int / B: int -> int64_t, p_val: float / inv_p: float -> float.
BEGIN_TILING_DATA_DEF(PdistTilingData)
TILING_DATA_FIELD_DEF(int64_t, N)
TILING_DATA_FIELD_DEF(int64_t, M)
TILING_DATA_FIELD_DEF(int64_t, B)
TILING_DATA_FIELD_DEF(float, p_val)
TILING_DATA_FIELD_DEF(float, inv_p)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(Pdist, PdistTilingData)

// Compile info carried across the compile/run boundary (no persistent state needed: every
// tiling input is re-derived from the runtime shapes + attr).
struct PdistCompileInfo {};

} // namespace optiling

#endif // PDIST_TILING_H_
