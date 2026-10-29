/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file test_reduce_std_with_mean_v2_infershape.cpp
 * \brief Infershape unit tests for ReduceStdWithMeanV2 operator
 *
 * Coverage targets:
 *   - Normal shapes: fp32/fp16, various dims, keepdim true/false
 *   - Empty dim (reduce all dimensions), with and without keepdim
 *   - Multi-dim reduce
 *   - Negative axis
 *   - Single-dim input
 *
 * Note: op has two outputs (out=std, meanOut) with identical shapes.
 */

#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include "infershape_case_executor.h"
#include "infershape_context_faker.h"

namespace {

gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape;
    shape.MutableOriginShape().SetDimNum(dims.size());
    shape.MutableStorageShape().SetDimNum(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        shape.MutableOriginShape().SetDim(i, dims[i]);
        shape.MutableStorageShape().SetDim(i, dims[i]);
    }
    return shape;
}

gert::InfershapeContextPara MakeContext(const std::vector<int64_t>& selfShape, const std::vector<int64_t>& dims,
                                        bool keepDims, ge::DataType dtype = ge::DT_FLOAT)
{
    std::vector<gert::InfershapeContextPara::TensorDescription> inputs = {
        {MakeStorageShape(selfShape), dtype, ge::FORMAT_ND},
    };
    std::vector<gert::InfershapeContextPara::TensorDescription> outputs = {
        {MakeStorageShape({}), dtype, ge::FORMAT_ND},
        {MakeStorageShape({}), dtype, ge::FORMAT_ND},
    };
    // Must pass all OpDef attrs in order (0=dim, 1=unbiased, 2=keepdim, 3=invert,
    // 4=eps, 5=correction) so that GetAttrPointer by index resolves correctly.
    std::vector<gert::InfershapeContextPara::OpAttr> attrs = {
        {"dim", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>(dims)},
        {"unbiased", Ops::Math::AnyValue::CreateFrom<bool>(true)},
        {"keepdim", Ops::Math::AnyValue::CreateFrom<bool>(keepDims)},
        {"invert", Ops::Math::AnyValue::CreateFrom<bool>(false)},
        {"eps", Ops::Math::AnyValue::CreateFrom<float>(0.0f)},
        {"correction", Ops::Math::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(0))},
    };
    return gert::InfershapeContextPara("ReduceStdWithMeanV2", inputs, outputs, attrs);
}

} // namespace

class ReduceStdWithMeanV2InferShape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ReduceStdWithMeanV2InferShape SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "ReduceStdWithMeanV2InferShape TearDown" << std::endl; }
};

// ==========================================================================
// Normal Cases — fp32
// ==========================================================================

TEST_F(ReduceStdWithMeanV2InferShape, fp32_keepdim_true_dim0)
{
    auto context = MakeContext({2, 4}, {0}, true, ge::DT_FLOAT);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{1, 4}, {1, 4}});
}

TEST_F(ReduceStdWithMeanV2InferShape, fp32_keepdim_false_last_dim)
{
    auto context = MakeContext({2, 3, 4}, {-1}, false, ge::DT_FLOAT);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{2, 3}, {2, 3}});
}

TEST_F(ReduceStdWithMeanV2InferShape, fp32_multi_dim_keepdim_false)
{
    auto context = MakeContext({2, 3, 4, 5}, {0, 2}, false, ge::DT_FLOAT);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{3, 5}, {3, 5}});
}

TEST_F(ReduceStdWithMeanV2InferShape, fp32_multi_dim_keepdim_true)
{
    auto context = MakeContext({2, 3, 4}, {0, 1}, true, ge::DT_FLOAT);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{1, 1, 4}, {1, 1, 4}});
}

// ==========================================================================
// Normal Cases — fp16
// ==========================================================================

TEST_F(ReduceStdWithMeanV2InferShape, fp16_keepdim_true_dim1)
{
    auto context = MakeContext({2, 3, 4}, {1}, true, ge::DT_FLOAT16);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{2, 1, 4}, {2, 1, 4}});
}

TEST_F(ReduceStdWithMeanV2InferShape, fp16_keepdim_false_dim0)
{
    auto context = MakeContext({2, 3, 4}, {0}, false, ge::DT_FLOAT16);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{3, 4}, {3, 4}});
}

// ==========================================================================
// Empty dim (reduce all dimensions)
// ==========================================================================

TEST_F(ReduceStdWithMeanV2InferShape, dim_empty_reduce_all_no_keepdim)
{
    auto context = MakeContext({2, 3, 4}, {}, false, ge::DT_FLOAT);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{}, {}});
}

TEST_F(ReduceStdWithMeanV2InferShape, dim_empty_reduce_all_keepdim)
{
    auto context = MakeContext({2, 3, 4}, {}, true, ge::DT_FLOAT);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{1, 1, 1}, {1, 1, 1}});
}

// ==========================================================================
// Edge cases
// ==========================================================================

TEST_F(ReduceStdWithMeanV2InferShape, reduce_all_single_dim_no_keepdim)
{
    auto context = MakeContext({128}, {}, false, ge::DT_FLOAT16);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{}, {}});
}

TEST_F(ReduceStdWithMeanV2InferShape, reduce_all_single_dim_keepdim)
{
    auto context = MakeContext({128}, {}, true, ge::DT_FLOAT16);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{1}, {1}});
}

TEST_F(ReduceStdWithMeanV2InferShape, dim_negative_axis)
{
    auto context = MakeContext({2, 3, 4, 5}, {-2}, false, ge::DT_FLOAT);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{2, 3, 5}, {2, 3, 5}});
}

TEST_F(ReduceStdWithMeanV2InferShape, fp16_single_dim_partial_reduce)
{
    auto context = MakeContext({64}, {0}, false, ge::DT_FLOAT16);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{}, {}});
}

TEST_F(ReduceStdWithMeanV2InferShape, fp16_single_dim_partial_reduce_keepdim)
{
    auto context = MakeContext({64}, {0}, true, ge::DT_FLOAT16);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{1}, {1}});
}
