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
#define DTYPE_X float
#endif
#ifndef DTYPE_VALUE
#define DTYPE_VALUE DIAG_PART_TPL_DTYPE_FLOAT
#endif
#ifndef ALIGN_SIZE
#define ALIGN_SIZE 8
#endif

namespace {
void RunFp32Side(uint32_t matrixSide)
{
    auto entry = diag_part<ELEMENTWISE_TPL_SCH_MODE_0, DTYPE_VALUE>;
    DiagPartUtCommon::ExecuteDiagPartCase<DTYPE_X, DTYPE_VALUE, ALIGN_SIZE>(matrixSide, entry);
}
} // namespace

class DiagPartTestFp32 : public DiagPartUtCommon::DiagPartUtFixtureBase {};

TEST_F(DiagPartTestFp32, fp32_case_4x4) { RunFp32Side(4); }

TEST_F(DiagPartTestFp32, fp32_case_8x8) { RunFp32Side(8); }
