/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CONCAT_D_TILING_ARCH31_H_
#define CONCAT_D_TILING_ARCH31_H_

#include "register/tilingdata_base.h"
#include "../../op_kernel/arch31/concat_d_tiling_data.h"

namespace optiling {
#define CONCAT_D_ARCH31_HOST_FIELD(type, name) TILING_DATA_FIELD_DEF(type, name);
BEGIN_TILING_DATA_DEF(ConcatDArch31TilingData)
CONCAT_D_ARCH31_TILING_SCALAR_FIELDS(CONCAT_D_ARCH31_HOST_FIELD)
TILING_DATA_FIELD_DEF_ARR(uint64_t, concat_d_arch31::MAX_INPUTS, inputRowBytes);
END_TILING_DATA_DEF;
#undef CONCAT_D_ARCH31_HOST_FIELD
REGISTER_TILING_DATA_CLASS(ConcatD, ConcatDArch31TilingData)
} // namespace optiling

#endif // CONCAT_D_TILING_ARCH31_H_
