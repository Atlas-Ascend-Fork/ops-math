/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <array>
#include <vector>
#include "gtest/gtest.h"

#include "../../../op_api/aclnn_arange.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_arange_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "arange_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "arange_test TearDown" << endl; }
};

TEST_F(l2_arange_test, aclnnArange_0_to_10_step_1_float)
{
    // start
    double startValue = 0.0;
    aclDataType startDtype = ACL_FLOAT;
    // end
    double endValue = 10.0;
    aclDataType endDtype = ACL_FLOAT;
    // step
    double stepValue = 1.0;
    aclDataType stepDtype = ACL_FLOAT;
    // output
    const vector<int64_t>& outShape = {10};
    aclDataType outDtype = ACL_FLOAT;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto startDesc = ScalarDesc(startValue, startDtype);
    auto endDesc = ScalarDesc(endValue, endDtype);
    auto stepDesc = ScalarDesc(stepValue, stepDtype);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnArange, INPUT(startDesc, endDesc, stepDesc), OUTPUT(outTensorDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_arange_test, aclnnArange_0_to_100_step_5_int32)
{
    // start
    int64_t startValue = 0;
    aclDataType startDtype = ACL_INT32;
    // end
    int64_t endValue = 100;
    aclDataType endDtype = ACL_INT32;
    // step
    int64_t stepValue = 5;
    aclDataType stepDtype = ACL_INT32;
    // output
    const vector<int64_t>& outShape = {20};
    aclDataType outDtype = ACL_INT32;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto startDesc = ScalarDesc(startValue);
    auto endDesc = ScalarDesc(endValue);
    auto stepDesc = ScalarDesc(stepValue);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0, 0);

    auto ut = OP_API_UT(aclnnArange, INPUT(startDesc, endDesc, stepDesc), OUTPUT(outTensorDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// out元素个数与(start,end,step)推导长度不一致时按torch语义处理（issue#3972）
TEST_F(l2_arange_test, aclnnArange_out_size_mismatch)
{
    // out=18 > 推导长度17：成功，仅前17个元素有效
    {
        auto startDesc = ScalarDesc(-0.5, ACL_FLOAT);
        auto endDesc = ScalarDesc(16.0, ACL_FLOAT);
        auto stepDesc = ScalarDesc(1.0, ACL_FLOAT);
        auto outTensorDesc = TensorDesc({18}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
        auto ut = OP_API_UT(aclnnArange, INPUT(startDesc, endDesc, stepDesc), OUTPUT(outTensorDesc));
        uint64_t workspaceSize = 0;
        EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
    }
    // out=16 < 推导长度17：维持原有截断行为
    {
        auto startDesc = ScalarDesc(-0.5, ACL_FLOAT);
        auto endDesc = ScalarDesc(16.0, ACL_FLOAT);
        auto stepDesc = ScalarDesc(1.0, ACL_FLOAT);
        auto outTensorDesc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
        auto ut = OP_API_UT(aclnnArange, INPUT(startDesc, endDesc, stepDesc), OUTPUT(outTensorDesc));
        uint64_t workspaceSize = 0;
        EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
    }
    // int64输出：start=0, end=100, step=7，推导长度15，out=14截断
    {
        auto startDesc = ScalarDesc(static_cast<int64_t>(0));
        auto endDesc = ScalarDesc(static_cast<int64_t>(100));
        auto stepDesc = ScalarDesc(static_cast<int64_t>(7));
        auto outTensorDesc = TensorDesc({14}, ACL_INT64, ACL_FORMAT_ND).Precision(0, 0);
        auto ut = OP_API_UT(aclnnArange, INPUT(startDesc, endDesc, stepDesc), OUTPUT(outTensorDesc));
        uint64_t workspaceSize = 0;
        EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
    }
}
