/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <vector>
#include "gtest/gtest.h"
#include "math/affine_grid/op_api/aclnn_affine_grid.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"

static aclnnStatus CheckAffineGridSize(const std::vector<int64_t>& size, bool alignCorners)
{
    const bool is3d = size.size() == 5;
    const int64_t n = std::max(size[0], int64_t{0});
    auto thetaDesc = TensorDesc({n, is3d ? 3 : 2, is3d ? 4 : 3}, ACL_FLOAT, ACL_FORMAT_ND);
    std::vector<int64_t> outShape{n};
    for (size_t i = 2; i < size.size(); ++i) {
        outShape.push_back(std::max(size[i], int64_t{0}));
    }
    outShape.push_back(is3d ? 3 : 2);
    auto outDesc = TensorDesc(outShape, ACL_FLOAT, ACL_FORMAT_ND);
    auto sizeDesc = IntArrayDesc(size);
    auto ut = OP_API_UT(aclnnAffineGrid, INPUT(thetaDesc, sizeDesc, alignCorners), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    return ut.TestGetWorkspaceSize(&workspaceSize);
}

TEST(l2_affine_grid_size_test, rejects_invalid_n_h_w)
{
    for (bool is3d : {false, true}) {
        for (bool alignCorners : {false, true}) {
            const std::vector<int64_t> validSize = is3d ? std::vector<int64_t>{1, 1, 1, 1, 1} :
                                                          std::vector<int64_t>{1, 1, 1, 1};
            for (size_t axis : {size_t{0}, validSize.size() - 2, validSize.size() - 1}) {
                for (int64_t value : {-1, 0, 100001}) {
                    auto size = validSize;
                    size[axis] = value;
                    EXPECT_EQ(CheckAffineGridSize(size, alignCorners), ACLNN_ERR_PARAM_INVALID);
                }
            }
        }
    }
}

TEST(l2_affine_grid_size_test, rejects_nonpositive_depth)
{
    for (int64_t depth : {-1, 0}) {
        EXPECT_EQ(CheckAffineGridSize({1, 1, depth, 1, 1}, false), ACLNN_ERR_PARAM_INVALID);
    }
}

TEST(l2_affine_grid_size_test, accepts_boundaries_and_ignores_channels)
{
    for (bool is3d : {false, true}) {
        const std::vector<int64_t> validSize = is3d ? std::vector<int64_t>{1, 1, 1, 1, 1} :
                                                      std::vector<int64_t>{1, 1, 1, 1};
        EXPECT_EQ(CheckAffineGridSize(validSize, false), ACLNN_SUCCESS);
        for (size_t axis : {size_t{0}, validSize.size() - 2, validSize.size() - 1}) {
            auto size = validSize;
            size[axis] = 100000;
            EXPECT_EQ(CheckAffineGridSize(size, true), ACLNN_SUCCESS);
        }
        for (int64_t channels : {-1, 0}) {
            auto size = validSize;
            size[1] = channels;
            EXPECT_EQ(CheckAffineGridSize(size, false), ACLNN_SUCCESS);
        }
    }
}
