/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "tiling_case_executor.h"
#include "../../../../op_host/arch35/pow_tiling_arch35.h"

TEST(PowErrorCodeTiling, zero_ub_returns_failure)
{
    gert::StorageShape shape({2, 4}, {2, 4});
    optiling::PowCompileInfo compileInfo{64, 0, true, 256, 32};
    gert::TilingContextPara params("Pow", {{shape, ge::DT_FLOAT, ge::FORMAT_ND}, {shape, ge::DT_FLOAT, ge::FORMAT_ND}},
                                   {{shape, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo, 64, 0);
    ExecuteTestCase(params, ge::GRAPH_FAILED);
}
