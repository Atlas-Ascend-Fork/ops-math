/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include "../../../op_api/aclnn_minimum.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace std;

class l2_minimum_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "minimum_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "minimum_test TearDown" << endl; }
};

// DOUBLE 输入时，连续广播轴/非广播轴合并后共 6 段，超过 broadcast_relationship.md 的限制，第一阶段应返回参数错误
TEST_F(l2_minimum_test, aclnnMinimum_027_aclnnMinimum_double_broadcast_6_segments_invalid)
{
    const vector<int64_t>& selfShape = {32, 4, 7, 4, 2, 3, 1, 7};
    aclDataType selfDtype = ACL_DOUBLE;
    aclFormat selfFormat = ACL_FORMAT_ND;
    const vector<int64_t>& otherShape = {32, 1, 7, 1, 1, 3, 1, 1};
    aclDataType otherDtype = ACL_DOUBLE;
    aclFormat otherFormat = ACL_FORMAT_ND;
    const vector<int64_t>& outShape = {32, 4, 7, 4, 2, 3, 1, 7};
    aclDataType outDtype = ACL_DOUBLE;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).ValueRange(-1, 1);
    auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat).ValueRange(-1, 1);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

    auto ut = OP_API_UT(aclnnMinimum, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// DOUBLE 输入时，广播合并后为 5 段，未超过限制，应正常放行
TEST_F(l2_minimum_test, aclnnMinimum_028_aclnnMinimum_double_broadcast_5_segments_success)
{
    const vector<int64_t>& selfShape = {32, 4, 7, 4, 2, 3, 1, 7};
    aclDataType selfDtype = ACL_DOUBLE;
    aclFormat selfFormat = ACL_FORMAT_ND;
    const vector<int64_t>& otherShape = {32, 1, 7, 1, 1, 3, 1, 7};
    aclDataType otherDtype = ACL_DOUBLE;
    aclFormat otherFormat = ACL_FORMAT_ND;
    const vector<int64_t>& outShape = {32, 4, 7, 4, 2, 3, 1, 7};
    aclDataType outDtype = ACL_DOUBLE;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).ValueRange(-1, 1);
    auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat).ValueRange(-1, 1);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

    auto ut = OP_API_UT(aclnnMinimum, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// INT16 输入同样受广播分段限制，6 段时应返回参数错误
TEST_F(l2_minimum_test, aclnnMinimum_029_aclnnMinimum_int16_broadcast_6_segments_invalid)
{
    const vector<int64_t>& selfShape = {32, 4, 7, 4, 2, 3, 1, 7};
    aclDataType selfDtype = ACL_INT16;
    aclFormat selfFormat = ACL_FORMAT_ND;
    const vector<int64_t>& otherShape = {32, 1, 7, 1, 1, 3, 1, 1};
    aclDataType otherDtype = ACL_INT16;
    aclFormat otherFormat = ACL_FORMAT_ND;
    const vector<int64_t>& outShape = {32, 4, 7, 4, 2, 3, 1, 7};
    aclDataType outDtype = ACL_INT16;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

    auto ut = OP_API_UT(aclnnMinimum, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
