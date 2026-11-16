/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <gtest/gtest.h>
#include "math/sinh/op_api/aclnn_sinh.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

namespace {
const std::vector<std::vector<int64_t>> EMPTY_SHAPES = {{0},       {0, 2, 3}, {2, 0, 3},
                                                        {2, 3, 0}, {0, 0, 0}, {1, 1, 1, 1, 1, 1, 1, 0}};
} // namespace

TEST(SinhEmptyApiTest, OutOfPlaceRequiresNoWorkspace)
{
    for (auto dtype : {ACL_FLOAT16, ACL_BF16, ACL_FLOAT}) {
        for (const auto& shape : EMPTY_SHAPES) {
            auto tensor = TensorDesc(shape, dtype, ACL_FORMAT_ND);
            auto ut = OP_API_UT(aclnnSinh, INPUT(tensor), OUTPUT(tensor));
            uint64_t workspaceSize = 1;
            ASSERT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
            EXPECT_EQ(workspaceSize, 0U);
        }
    }
}

TEST(SinhEmptyApiTest, InplaceRequiresNoWorkspace)
{
    for (auto dtype : {ACL_FLOAT16, ACL_BF16, ACL_FLOAT}) {
        for (const auto& shape : EMPTY_SHAPES) {
            auto tensor = TensorDesc(shape, dtype, ACL_FORMAT_ND);
            auto ut = OP_API_UT(aclnnInplaceSinh, INPUT(tensor), OUTPUT());
            uint64_t workspaceSize = 1;
            ASSERT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
            EXPECT_EQ(workspaceSize, 0U);
        }
    }
}
