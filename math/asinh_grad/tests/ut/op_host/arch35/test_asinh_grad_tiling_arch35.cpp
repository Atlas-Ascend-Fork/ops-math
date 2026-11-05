/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "tiling_case_executor.h"

namespace {
struct CompileInfo {};
CompileInfo compileInfo;

gert::TilingContextPara MakeContext(const gert::StorageShape& y, const gert::StorageShape& dy,
                                    const gert::StorageShape& z, ge::DataType dtype = ge::DT_FLOAT)
{
    return {"AsinhGrad",
            {{y, dtype, ge::FORMAT_ND}, {dy, dtype, ge::FORMAT_ND}},
            {{z, dtype, ge::FORMAT_ND}},
            &compileInfo};
}
} // namespace

TEST(AsinhGradTilingTest, RejectsDifferentShapesWithEqualElementCount)
{
    gert::StorageShape y = {{2, 3}, {2, 3}};
    gert::StorageShape dy = {{3, 2}, {3, 2}};
    ExecuteTestCase(MakeContext(y, dy, y), ge::GRAPH_FAILED);
    ExecuteTestCase(MakeContext(y, y, dy), ge::GRAPH_FAILED);
}

TEST(AsinhGradTilingTest, RejectsScalarVectorMismatch)
{
    gert::StorageShape scalar = {{}, {}};
    gert::StorageShape vector = {{1}, {1}};
    ExecuteTestCase(MakeContext(scalar, vector, scalar), ge::GRAPH_FAILED);
}

TEST(AsinhGradTilingTest, RejectsRankNine)
{
    gert::StorageShape shape = {{1, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1, 1, 1}};
    ExecuteTestCase(MakeContext(shape, shape, shape), ge::GRAPH_FAILED);
}

TEST(AsinhGradTilingTest, AcceptsScalarAndRankEight)
{
    for (const gert::StorageShape& shape :
         {gert::StorageShape{{}, {}}, gert::StorageShape{{1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1, 1}}}) {
        TilingInfo info;
        EXPECT_TRUE(ExecuteTiling(MakeContext(shape, shape, shape), info));
    }
}

TEST(AsinhGradTilingTest, EmptyInputSelectsSingleBuffer)
{
    gert::StorageShape empty = {{0, 3}, {0, 3}};
    gert::StorageShape nonempty = {{3}, {3}};
    for (auto dtype : {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16}) {
        TilingInfo emptyInfo;
        TilingInfo nonemptyInfo;
        ASSERT_TRUE(ExecuteTiling(MakeContext(empty, empty, empty, dtype), emptyInfo));
        ASSERT_TRUE(ExecuteTiling(MakeContext(nonempty, nonempty, nonempty, dtype), nonemptyInfo));
        EXPECT_EQ(emptyInfo.tilingKey, nonemptyInfo.tilingKey);
        EXPECT_EQ(emptyInfo.blockNum, 1);
        EXPECT_EQ(emptyInfo.workspaceSizes, std::vector<int64_t>{0});
    }
}

TEST(AsinhGradTilingTest, BufferModeDispatchIsIndependentOfDtype)
{
    for (auto dtype : {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16}) {
        for (int64_t count : {1024, 1025}) {
            gert::StorageShape shape = {{count}, {count}};
            TilingInfo info;
            ASSERT_TRUE(ExecuteTiling(MakeContext(shape, shape, shape, dtype), info));
            EXPECT_EQ(info.tilingKey, count > 1024 ? 1U : 0U);
        }
    }
}

TEST(AsinhGradTilingTest, RejectsMixedInputDtypes)
{
    gert::StorageShape shape = {{3}, {3}};
    auto context = MakeContext(shape, shape, shape);
    context.inputTensorDesc_[1].dtype_ = ge::DT_FLOAT16;
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST(AsinhGradTilingTest, RejectsUnsupportedDtype)
{
    gert::StorageShape shape = {{3}, {3}};
    ExecuteTestCase(MakeContext(shape, shape, shape, ge::DT_INT32), ge::GRAPH_FAILED);
}
