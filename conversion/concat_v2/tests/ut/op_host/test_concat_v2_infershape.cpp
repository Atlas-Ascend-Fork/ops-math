/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file test_concat_v2_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <vector>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "exe_graph/runtime/kernel_run_context.h"

using namespace ge;
class ConcatV2Test : public testing::Test {
protected:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}
};

TEST_F(ConcatV2Test, infer_axis_type_test_01)
{
    gert::InfershapeContextPara infershapeContextPara("ConcatV2",
                                                      {
                                                          {{{16, 4}, {16, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{16, 4}, {16, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(4)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(ConcatV2Test, infer_axis_type_test_02)
{
    gert::InfershapeContextPara infershapeContextPara("ConcatV2",
                                                      {
                                                          {{{16, 4, 16}, {16, 4, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{16, 4, 16}, {16, 4, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(4)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(ConcatV2Test, infer_success_int64_axis)
{
    int64_t axis_value = 2;
    gert::InfershapeContextPara infershapeContextPara("ConcatV2",
                                                      {
                                                          {{{16, 4, 8}, {16, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{16, 4, 8}, {16, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{16, 4, 8}, {16, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &axis_value},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(3)},
                                                      },
                                                      {3, 1}, {1});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {16, 4, 24},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ConcatV2Test, infer_negative_axis)
{
    int64_t axis_value = -1;
    gert::InfershapeContextPara infershapeContextPara("ConcatV2",
                                                      {
                                                          {{{16, 4, 8}, {16, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{16, 4, 8}, {16, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &axis_value},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(2)},
                                                      },
                                                      {2, 1}, {1});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {16, 4, 16},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ConcatV2Test, infer_invalid_axis_too_large)
{
    int64_t axis_value = 3;
    gert::InfershapeContextPara infershapeContextPara("ConcatV2",
                                                      {
                                                          {{{16, 4, 8}, {16, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{16, 4, 8}, {16, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &axis_value},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(2)},
                                                      },
                                                      {2, 1}, {1});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(ConcatV2Test, infer_invalid_axis_too_negative)
{
    int64_t axis_value = -4;
    gert::InfershapeContextPara infershapeContextPara("ConcatV2",
                                                      {
                                                          {{{16, 4, 8}, {16, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{16, 4, 8}, {16, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &axis_value},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(2)},
                                                      },
                                                      {2, 1}, {1});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(ConcatV2Test, infer_single_input_error)
{
    int64_t axis_value = 2;
    gert::InfershapeContextPara infershapeContextPara("ConcatV2",
                                                      {
                                                          {{{16, 4, 8}, {16, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &axis_value},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
                                                      },
                                                      {1, 1}, {1});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(ConcatV2Test, infer_axis_value_unavailable)
{
    // concat_dim 为数据依赖输入，编译期取值不可得（data-feed 场景），秩已知时输出应保秩、全维置 -1
    gert::InfershapeContextPara infershapeContextPara("ConcatV2",
                                                      {
                                                          {{{2, 100, 4}, {2, 100, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{2, 100, 4}, {2, 100, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(2)},
                                                      },
                                                      {2, 1}, {1});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, -1, -1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ConcatV2Test, infer_axis_value_unavailable_all_unknown_rank)
{
    // concat_dim 编译期不可得且所有输入均为未知秩时，输出应为未知秩
    gert::InfershapeContextPara infershapeContextPara("ConcatV2",
                                                      {
                                                          {{{-2}, {-2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{-2}, {-2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(2)},
                                                      },
                                                      {2, 1}, {1});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ConcatV2Test, infer_scalar_inputs)
{
    // 对等V1静态路径: 标量输入拼接, 输出为 [N]
    int64_t axis_value = 0;
    gert::InfershapeContextPara infershapeContextPara("ConcatV2",
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &axis_value},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(2)},
                                                      },
                                                      {2, 1}, {1});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ConcatV2Test, infer_scalar_with_rank1_input)
{
    // 对等V1静态路径: 秩1输入与标量输入沿axis 0拼接
    int64_t axis_value = 0;
    gert::InfershapeContextPara infershapeContextPara("ConcatV2",
                                                      {
                                                          {{{3}, {3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &axis_value},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(2)},
                                                      },
                                                      {2, 1}, {1});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ConcatV2Test, infer_rank_mismatch_failed)
{
    // 对等V1: 输入秩不一致应报错, 防止axis维越界读取
    int64_t axis_value = 1;
    gert::InfershapeContextPara infershapeContextPara("ConcatV2",
                                                      {
                                                          {{{16, 4, 8}, {16, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{16, 4}, {16, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &axis_value},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(2)},
                                                      },
                                                      {2, 1}, {1});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(ConcatV2Test, infer_non_concat_dim_mismatch_failed)
{
    // 对等V1: 除concat维外其余维度不一致应报错
    int64_t axis_value = 1;
    gert::InfershapeContextPara infershapeContextPara("ConcatV2",
                                                      {
                                                          {{{16, 4, 8}, {16, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{16, 4, 7}, {16, 4, 7}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &axis_value},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(2)},
                                                      },
                                                      {2, 1}, {1});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(ConcatV2Test, infer_unknown_dim_merge)
{
    // 对等V1 JoinShapes: 非concat维的-1与已知值合并取已知
    int64_t axis_value = 1;
    gert::InfershapeContextPara infershapeContextPara("ConcatV2",
                                                      {
                                                          {{{-1, 4}, {-1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{16, 4}, {16, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &axis_value},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(2)},
                                                      },
                                                      {2, 1}, {1});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {16, 8},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ConcatV2Test, infer_concat_dim_unknown)
{
    // 对等V1: 任一输入在concat维为-1时, 输出该维为-1
    int64_t axis_value = 1;
    gert::InfershapeContextPara infershapeContextPara("ConcatV2",
                                                      {
                                                          {{{16, -1, 8}, {16, -1, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{16, 4, 8}, {16, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &axis_value},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(2)},
                                                      },
                                                      {2, 1}, {1});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {16, -1, 8},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ConcatV2Test, infer_dtype_mismatch_failed)
{
    // 对等V1 ConcatV2Verify/CheckInputDtypeSame: 动态输入x的dtype必须一致
    int64_t axis_value = 1;
    gert::InfershapeContextPara infershapeContextPara("ConcatV2",
                                                      {
                                                          {{{16, 4, 8}, {16, 4, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{16, 4, 8}, {16, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &axis_value},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(2)},
                                                      },
                                                      {2, 1}, {1});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(ConcatV2Test, infer_all_unknown_rank_with_known_axis)
{
    // 对等V1 all_unknown_rank_shape: 全部输入未知秩时输出未知秩
    int64_t axis_value = 0;
    gert::InfershapeContextPara infershapeContextPara("ConcatV2",
                                                      {
                                                          {{{-2}, {-2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{-2}, {-2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &axis_value},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(2)},
                                                      },
                                                      {2, 1}, {1});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ConcatV2Test, infer_mixed_unknown_rank_with_known_axis)
{
    // 对等V1 JoinShapes: 未知秩输入与已知秩输入混合时, 以已知秩输入为基底, concat维置-1
    int64_t axis_value = 1;
    gert::InfershapeContextPara infershapeContextPara("ConcatV2",
                                                      {
                                                          {{{-2}, {-2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{16, 4, 8}, {16, 4, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &axis_value},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(2)},
                                                      },
                                                      {2, 1}, {1});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {16, -1, 8},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ConcatV2Test, infer_identical_shapes_shared_shape_pointer)
{
    // 回归(线上axis维漏加问题): 运行时可能对相同形状的多个实例返回同一个 Shape 对象(desc同源共享),
    // 基底识别必须按索引而非指针, 否则同形输入被误判为基底跳过求和, 输出axis维漏加
    int64_t axis_value = 2;
    gert::Tensor x0(gert::StorageShape({16, 4, 8}, {16, 4, 8}),
                    gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                    gert::TensorPlacement::kOnHost, ge::DT_FLOAT16, nullptr);
    gert::Tensor x1(gert::StorageShape({16, 4, 8}, {16, 4, 8}),
                    gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                    gert::TensorPlacement::kOnHost, ge::DT_FLOAT16, nullptr);
    gert::Tensor axisTensor(gert::StorageShape({1}, {1}),
                            gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                            gert::TensorPlacement::kOnHost, ge::DT_INT64, &axis_value);
    std::vector<gert::Tensor*> inputTensors = {&x0, &x1, &axisTensor};
    auto contextHolder = gert::InferShapeContextFaker()
                             .SetOpType("ConcatV2")
                             .IrInstanceNum({2, 1}, {1})
                             .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                             .InputTensors(inputTensors)
                             .Build();
    // 指针别名: 把实例1的槽位指针改为指向实例0, 模拟运行时同形实例共享 Shape 对象
    auto* kernelCtx = reinterpret_cast<KernelRunContext*>(contextHolder.GetContext());
    kernelCtx->values[1]->data.pointer = kernelCtx->values[0]->data.pointer;

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    const auto* opImpl = spaceRegistry->GetOpImpl("ConcatV2");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_shape, nullptr);
    EXPECT_EQ(opImpl->infer_shape(contextHolder.GetContext()), ge::GRAPH_SUCCESS);
    auto* outShape = contextHolder.GetContext()->GetOutputShape(0);
    ASSERT_NE(outShape, nullptr);
    const std::vector<int64_t> expectDims = {16, 4, 16};
    ASSERT_EQ(outShape->GetDimNum(), expectDims.size());
    for (size_t d = 0; d < expectDims.size(); ++d) {
        EXPECT_EQ(outShape->GetDim(d), expectDims[d]);
    }
}
