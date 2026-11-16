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
#include "gtest/gtest.h"

#include "../../../op_api/aclnn_as_strided.h"

#include "op_api_ut_common/inner/types.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2AsStridedTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "l2AsStridedTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "l2AsStridedTest TearDown" << std::endl; }
};

// 正常路径：fp32，2维重叠stride
TEST_F(l2AsStridedTest, case_norm_fp32)
{
    auto x_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto y_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto size_desc = IntArrayDesc(vector<int64_t>{2, 3});
    auto stride_desc = IntArrayDesc(vector<int64_t>{4, 2});
    int64_t storageOffset = 1;
    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto ut = OP_API_UT(aclnnAsStrided, INPUT(x_desc, size_desc, stride_desc, storageOffset, y_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常路径：多dtype覆盖
TEST_F(l2AsStridedTest, case_norm_multi_dtype)
{
    vector<aclDataType> dtypeList = {ACL_FLOAT16, ACL_BF16,   ACL_INT8,  ACL_UINT8,  ACL_INT16, ACL_UINT16,
                                     ACL_INT32,   ACL_UINT32, ACL_INT64, ACL_UINT64, ACL_BOOL,  ACL_COMPLEX64};
    SetPlatformNpuArch(NpuArch::DAV_3510);
    for (auto dtype : dtypeList) {
        auto x_desc = TensorDesc({100}, dtype, ACL_FORMAT_ND);
        auto y_desc = TensorDesc({6}, dtype, ACL_FORMAT_ND);
        auto size_desc = IntArrayDesc(vector<int64_t>{6});
        auto stride_desc = IntArrayDesc(vector<int64_t>{11});
        int64_t storageOffset = 3;
        auto ut = OP_API_UT(aclnnAsStrided, INPUT(x_desc, size_desc, stride_desc, storageOffset, y_desc), OUTPUT());

        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACL_SUCCESS);
    }
}

// 正常路径：零stride广播拷贝
TEST_F(l2AsStridedTest, case_norm_zero_stride)
{
    auto x_desc = TensorDesc({10}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto y_desc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto size_desc = IntArrayDesc(vector<int64_t>{16});
    auto stride_desc = IntArrayDesc(vector<int64_t>{0});
    int64_t storageOffset = 3;
    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto ut = OP_API_UT(aclnnAsStrided, INPUT(x_desc, size_desc, stride_desc, storageOffset, y_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常路径：8维上限
TEST_F(l2AsStridedTest, case_norm_8dim)
{
    auto x_desc = TensorDesc({4096}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto y_desc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2}, ACL_INT8, ACL_FORMAT_ND);
    auto size_desc = IntArrayDesc(vector<int64_t>{2, 2, 2, 2, 2, 2, 2, 2});
    auto stride_desc = IntArrayDesc(vector<int64_t>{8, 7, 6, 5, 4, 3, 2, 1});
    int64_t storageOffset = 0;
    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto ut = OP_API_UT(aclnnAsStrided, INPUT(x_desc, size_desc, stride_desc, storageOffset, y_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常路径：空输出（size含0），快速返回
TEST_F(l2AsStridedTest, case_norm_empty_y)
{
    auto x_desc = TensorDesc({100}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto y_desc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto size_desc = IntArrayDesc(vector<int64_t>{2, 0});
    auto stride_desc = IntArrayDesc(vector<int64_t>{3, 1});
    int64_t storageOffset = 0;
    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto ut = OP_API_UT(aclnnAsStrided, INPUT(x_desc, size_desc, stride_desc, storageOffset, y_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    EXPECT_EQ(workspace_size, 0);
}

// 异常路径：非regbase平台不支持
TEST_F(l2AsStridedTest, case_platform_not_support)
{
    auto x_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto y_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto size_desc = IntArrayDesc(vector<int64_t>{4});
    auto stride_desc = IntArrayDesc(vector<int64_t>{2});
    int64_t storageOffset = 1;
    SetPlatformNpuArch(NpuArch::DAV_2201);
    auto ut = OP_API_UT(aclnnAsStrided, INPUT(x_desc, size_desc, stride_desc, storageOffset, y_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    SetPlatformNpuArch(NpuArch::DAV_3510);
}

// 异常路径：空指针
TEST_F(l2AsStridedTest, case_nullptr)
{
    auto x_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto y_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto size_desc = IntArrayDesc(vector<int64_t>{4});
    auto stride_desc = IntArrayDesc(vector<int64_t>{2});
    int64_t storageOffset = 1;
    SetPlatformNpuArch(NpuArch::DAV_3510);
    uint64_t workspace_size = 0;

    auto ut1 = OP_API_UT(aclnnAsStrided, INPUT(nullptr, size_desc, stride_desc, storageOffset, y_desc), OUTPUT());
    aclnnStatus aclRet = ut1.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut2 = OP_API_UT(aclnnAsStrided, INPUT(x_desc, size_desc, stride_desc, storageOffset, nullptr), OUTPUT());
    aclRet = ut2.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut3 = OP_API_UT(aclnnAsStrided, INPUT(x_desc, nullptr, stride_desc, storageOffset, y_desc), OUTPUT());
    aclRet = ut3.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut4 = OP_API_UT(aclnnAsStrided, INPUT(x_desc, size_desc, nullptr, storageOffset, y_desc), OUTPUT());
    aclRet = ut4.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 异常路径：dtype不支持
TEST_F(l2AsStridedTest, case_invalid_dtype)
{
    auto x_desc = TensorDesc({16}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto y_desc = TensorDesc({4}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto size_desc = IntArrayDesc(vector<int64_t>{4});
    auto stride_desc = IntArrayDesc(vector<int64_t>{2});
    int64_t storageOffset = 1;
    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto ut = OP_API_UT(aclnnAsStrided, INPUT(x_desc, size_desc, stride_desc, storageOffset, y_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常路径：y与x的dtype不一致
TEST_F(l2AsStridedTest, case_dtype_mismatch)
{
    auto x_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto y_desc = TensorDesc({4}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto size_desc = IntArrayDesc(vector<int64_t>{4});
    auto stride_desc = IntArrayDesc(vector<int64_t>{2});
    int64_t storageOffset = 1;
    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto ut = OP_API_UT(aclnnAsStrided, INPUT(x_desc, size_desc, stride_desc, storageOffset, y_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常路径：y的shape与size不一致
TEST_F(l2AsStridedTest, case_y_shape_neq_size)
{
    auto x_desc = TensorDesc({1000}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto y_desc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto size_desc = IntArrayDesc(vector<int64_t>{4});
    auto stride_desc = IntArrayDesc(vector<int64_t>{2});
    int64_t storageOffset = 1;
    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto ut = OP_API_UT(aclnnAsStrided, INPUT(x_desc, size_desc, stride_desc, storageOffset, y_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常路径：size与stride长度不一致
TEST_F(l2AsStridedTest, case_len_mismatch)
{
    auto x_desc = TensorDesc({1000}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto y_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto size_desc = IntArrayDesc(vector<int64_t>{2, 2});
    auto stride_desc = IntArrayDesc(vector<int64_t>{3, 1, 1});
    int64_t storageOffset = 0;
    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto ut = OP_API_UT(aclnnAsStrided, INPUT(x_desc, size_desc, stride_desc, storageOffset, y_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常路径：维度超过8
TEST_F(l2AsStridedTest, case_rank_over_8)
{
    auto x_desc = TensorDesc({4096}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto y_desc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto size_desc = IntArrayDesc(vector<int64_t>{2, 2, 2, 2, 2, 2, 2, 2, 2});
    auto stride_desc = IntArrayDesc(vector<int64_t>{8, 7, 6, 5, 4, 3, 2, 1, 1});
    int64_t storageOffset = 0;
    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto ut = OP_API_UT(aclnnAsStrided, INPUT(x_desc, size_desc, stride_desc, storageOffset, y_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常路径：size/stride/storageOffset负值
TEST_F(l2AsStridedTest, case_negative_value)
{
    auto x_desc = TensorDesc({100}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto y_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto stride_desc = IntArrayDesc(vector<int64_t>{2});
    SetPlatformNpuArch(NpuArch::DAV_3510);
    uint64_t workspace_size = 0;

    auto ut1 = OP_API_UT(aclnnAsStrided, INPUT(x_desc, IntArrayDesc(vector<int64_t>{-1}), stride_desc, 0, y_desc),
                         OUTPUT());
    aclnnStatus aclRet = ut1.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

    auto ut2 = OP_API_UT(aclnnAsStrided,
                         INPUT(x_desc, IntArrayDesc(vector<int64_t>{4}), IntArrayDesc(vector<int64_t>{-2}), 0, y_desc),
                         OUTPUT());
    aclRet = ut2.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

    auto ut3 = OP_API_UT(aclnnAsStrided, INPUT(x_desc, IntArrayDesc(vector<int64_t>{4}), stride_desc, -1, y_desc),
                         OUTPUT());
    aclRet = ut3.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常路径：gather范围越界（storageOffset + Σ(size-1)*stride >= numel(x)），
// L2在CheckGatherRange前置拦截（带int64溢出保护），返回参数校验错误
TEST_F(l2AsStridedTest, case_out_of_range)
{
    auto x_desc = TensorDesc({100}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto y_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto size_desc = IntArrayDesc(vector<int64_t>{4});
    auto stride_desc = IntArrayDesc(vector<int64_t>{256});
    int64_t storageOffset = 4096;
    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto ut = OP_API_UT(aclnnAsStrided, INPUT(x_desc, size_desc, stride_desc, storageOffset, y_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
