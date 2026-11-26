/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../../op_kernel/diag_part.cpp"
#include "diag_part_ut_common.h"

#ifndef DTYPE_X
#define DTYPE_X half
#endif
#ifndef DTYPE_VALUE
#define DTYPE_VALUE 1
#endif
#ifndef ALIGN_SIZE
#define ALIGN_SIZE 16
#endif

class DiagPartTest : public DiagPartUtCommon::DiagPartUtFixtureBase {};

static void RunDiagPartTest(uint32_t sideLen)
{
    auto kernelFn = diag_part<ELEMENTWISE_TPL_SCH_MODE_0, DTYPE_VALUE>;
    DiagPartUtCommon::ExecuteDiagPartCase<DTYPE_X, DTYPE_VALUE, ALIGN_SIZE>(sideLen, kernelFn);
}

TEST_F(DiagPartTest, test_case_4x4) { RunDiagPartTest(4); }

TEST_F(DiagPartTest, test_case_8x8) { RunDiagPartTest(8); }
