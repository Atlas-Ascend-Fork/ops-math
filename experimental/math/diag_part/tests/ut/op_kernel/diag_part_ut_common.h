/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DIAG_PART_UT_COMMON_H_
#define DIAG_PART_UT_COMMON_H_

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

#include "data_utils.h"
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../../../op_kernel/diag_part_tiling_key.h"

namespace DiagPartUtCommon {

constexpr size_t GM_ALIGN_BYTES = 32;
constexpr size_t WORKSPACE_BYTES = 32 * 1024 * 1024;
constexpr uint32_t DEFAULT_BLOCK_DIM = 1;

inline size_t CeilAlignBytes(size_t byteCount, size_t alignBytes = GM_ALIGN_BYTES)
{
    return (byteCount + alignBytes - 1) / alignBytes * alignBytes;
}

// 根据编译宏传入的 DTYPE_VALUE 映射测试数据脚本使用的数据类型名
inline std::string ResolveDtypeName(uint32_t dtypeValue)
{
    if (dtypeValue == DIAG_PART_TPL_DTYPE_FLOAT) {
        return "float32";
    }
    if (dtypeValue == DIAG_PART_TPL_DTYPE_INT32) {
        return "int32";
    }
    return "float16";
}

class DiagPartUtFixtureBase : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        const std::string dataPath = "../../../../experimental/math/diag_part/tests/ut/op_kernel/diag_part_data";
        const std::string copyCmd = "cp -rf " + dataPath + " ./";
        ASSERT_EQ(std::system(copyCmd.c_str()), 0);
        ASSERT_EQ(std::system("chmod -R 755 ./diag_part_data/"), 0);
    }
};

inline void FillDiagPartTiling(uint32_t sideLen, uint32_t dtypeValue, uint32_t alignSize,
                               DiagPartTilingData* tilingData)
{
    uint32_t tailNum = sideLen % alignSize;
    if (tailNum == 0) {
        tailNum = alignSize;
    }
    tilingData->sideLength = sideLen;
    tilingData->dtype = dtypeValue;
    tilingData->realCoreNum = DEFAULT_BLOCK_DIM;
    tilingData->numPerCore = alignSize;
    tilingData->tailNum = tailNum;
}

template <typename ElemType, uint32_t DtypeValue, uint32_t AlignSize, typename KernelEntry>
inline void ExecuteDiagPartCase(uint32_t sideLen, KernelEntry kernelFunc)
{
    const std::string dtypeName = ResolveDtypeName(DtypeValue);
    const std::string genCmd = "cd ./diag_part_data/ && python3 gen_data.py '(" + std::to_string(sideLen) + "," +
                               std::to_string(sideLen) + ")' '" + dtypeName + "'";
    ASSERT_EQ(std::system(genCmd.c_str()), 0);

    const size_t inputBytes = static_cast<size_t>(sideLen) * sideLen * sizeof(ElemType);
    const size_t outputBytes = static_cast<size_t>(sideLen) * sizeof(ElemType);
    uint8_t* xGm = static_cast<uint8_t*>(AscendC::GmAlloc(CeilAlignBytes(inputBytes)));
    uint8_t* yGm = static_cast<uint8_t*>(AscendC::GmAlloc(CeilAlignBytes(outputBytes)));
    uint8_t* workspaceGm = static_cast<uint8_t*>(AscendC::GmAlloc(WORKSPACE_BYTES));
    uint8_t* tilingGm = static_cast<uint8_t*>(AscendC::GmAlloc(sizeof(DiagPartTilingData)));

    const std::string inputFile = "./diag_part_data/" + dtypeName + "_input_x_diag_part.bin";
    ASSERT_TRUE(ReadFile(inputFile, inputBytes, xGm, inputBytes));
    FillDiagPartTiling(sideLen, DtypeValue, AlignSize, reinterpret_cast<DiagPartTilingData*>(tilingGm));

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernelFunc, DEFAULT_BLOCK_DIM, xGm, yGm, workspaceGm, tilingGm);

    const std::string outputFile = "./diag_part_data/" + dtypeName + "_output_diag_part.bin";
    ASSERT_TRUE(WriteFile(outputFile, yGm, outputBytes));

    AscendC::GmFree(static_cast<void*>(xGm));
    AscendC::GmFree(static_cast<void*>(yGm));
    AscendC::GmFree(static_cast<void*>(workspaceGm));
    AscendC::GmFree(static_cast<void*>(tilingGm));

    const std::string compareCmd = "cd ./diag_part_data/ && python3 compare_data.py '" + dtypeName + "'";
    ASSERT_EQ(std::system(compareCmd.c_str()), 0);
}

} // namespace DiagPartUtCommon

#endif // DIAG_PART_UT_COMMON_H_
