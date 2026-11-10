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
 * \file test_bitcast_infershape.cpp
 * \brief ut of Bitcast infershape
 */

#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "infershape_context_faker.h"
#include "op_infer_datatype_context_builder.h"
#include "op_infer_shape_range_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace {
// Bitcast infershape reads the x dtype from GetInputDesc(0), so the input must be a real
// gert::Tensor carrying a valid desc instead of a StorageShape-to-Tensor cast.
gert::Tensor MakeXTensor(std::initializer_list<int64_t> dims, ge::DataType dtype)
{
    return gert::Tensor(gert::StorageShape(dims, dims),
                        gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                        gert::TensorPlacement::kOnHost, dtype, nullptr);
}

std::vector<int64_t> ShapeToVec(const gert::Shape& shape)
{
    std::vector<int64_t> shapeVec(shape.GetDimNum(), 0);
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        shapeVec[i] = shape.GetDim(i);
    }
    return shapeVec;
}

// Min/Max tensor of one input shape range; both tensors carry the same dtype so
// InferShapeRangeContext::GetInputDesc can read the input dtype.
struct XShapeRange {
    gert::Tensor minTensor;
    gert::Tensor maxTensor;
    gert::Range<gert::Tensor> range;
    XShapeRange(std::initializer_list<int64_t> minDims, std::initializer_list<int64_t> maxDims, ge::DataType dtype)
        : minTensor(gert::StorageShape(minDims, minDims),
                    gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                    gert::TensorPlacement::kOnHost, dtype, nullptr),
          maxTensor(gert::StorageShape(maxDims, maxDims),
                    gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                    gert::TensorPlacement::kOnHost, dtype, nullptr),
          range(&minTensor, &maxTensor)
    {}
};
} // namespace

class BitcastUT : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "BitcastUT SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "BitcastUT TearDown" << std::endl; }

    static auto GetInferShapeFunc()
    {
        auto opImpl = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("Bitcast");
        EXPECT_NE(opImpl, nullptr);
        return opImpl->infer_shape;
    }
};

// test for keep_dim
TEST_F(BitcastUT, Infershape_OK_Int32ToInt4_KeepDim)
{
    auto xTensor = MakeXTensor({1, 3, 256}, ge::DT_INT32);
    gert::StorageShape yShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Bitcast")
                      .NodeIoNum(1, 1)
                      .InputTensors({&xTensor})
                      .OutputShapes({&yShape})
                      .Attr("type", static_cast<int64_t>(ge::DT_INT4))
                      .Attr("keep_dim", true)
                      .Build();

    EXPECT_EQ(GetInferShapeFunc()(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(ShapeToVec(*holder.GetContext()->GetOutputShape(0)), (std::vector<int64_t>{1, 3, 2048}));
}

TEST_F(BitcastUT, Infershape_OK_Int4ToInt32_KeepDim)
{
    auto xTensor = MakeXTensor({1, 3, 8}, ge::DT_INT4);
    gert::StorageShape yShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Bitcast")
                      .NodeIoNum(1, 1)
                      .InputTensors({&xTensor})
                      .OutputShapes({&yShape})
                      .Attr("type", static_cast<int64_t>(ge::DT_INT32))
                      .Attr("keep_dim", true)
                      .Build();

    EXPECT_EQ(GetInferShapeFunc()(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(ShapeToVec(*holder.GetContext()->GetOutputShape(0)), (std::vector<int64_t>{1, 3, 1}));
}

TEST_F(BitcastUT, Infershape_OK_Int4ToInt32_KeepDim_2)
{
    auto xTensor = MakeXTensor({1, 3, 16}, ge::DT_INT4);
    gert::StorageShape yShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Bitcast")
                      .NodeIoNum(1, 1)
                      .InputTensors({&xTensor})
                      .OutputShapes({&yShape})
                      .Attr("type", static_cast<int64_t>(ge::DT_INT32))
                      .Attr("keep_dim", true)
                      .Build();

    EXPECT_EQ(GetInferShapeFunc()(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(ShapeToVec(*holder.GetContext()->GetOutputShape(0)), (std::vector<int64_t>{1, 3, 2}));
}

TEST_F(BitcastUT, Infershape_OK_Int4ToInt32_KeepDim_KeepDim_3)
{
    // empty tensor with keep_dim: shape keeps unchanged
    auto xTensor = MakeXTensor({1, 3, 0}, ge::DT_INT4);
    gert::StorageShape yShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Bitcast")
                      .NodeIoNum(1, 1)
                      .InputTensors({&xTensor})
                      .OutputShapes({&yShape})
                      .Attr("type", static_cast<int64_t>(ge::DT_INT32))
                      .Attr("keep_dim", true)
                      .Build();

    EXPECT_EQ(GetInferShapeFunc()(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(ShapeToVec(*holder.GetContext()->GetOutputShape(0)), (std::vector<int64_t>{1, 3, 0}));
}

TEST_F(BitcastUT, Infershape_OK_FP32ToInt4_KeepDim_TrailingAxesIs0)
{
    // empty tensor with keep_dim: shape keeps unchanged
    auto xTensor = MakeXTensor({1, 3, 0}, ge::DT_FLOAT);
    gert::StorageShape yShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Bitcast")
                      .NodeIoNum(1, 1)
                      .InputTensors({&xTensor})
                      .OutputShapes({&yShape})
                      .Attr("type", static_cast<int64_t>(ge::DT_INT4))
                      .Attr("keep_dim", true)
                      .Build();

    EXPECT_EQ(GetInferShapeFunc()(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(ShapeToVec(*holder.GetContext()->GetOutputShape(0)), (std::vector<int64_t>{1, 3, 0}));
}

TEST_F(BitcastUT, Infershape_OK_Int32ToInt4_KeepDim_LastDimUnknown)
{
    // unknown last dim keeps unknown
    auto xTensor = MakeXTensor({1, 3, -1}, ge::DT_INT32);
    gert::StorageShape yShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Bitcast")
                      .NodeIoNum(1, 1)
                      .InputTensors({&xTensor})
                      .OutputShapes({&yShape})
                      .Attr("type", static_cast<int64_t>(ge::DT_INT4))
                      .Attr("keep_dim", true)
                      .Build();

    EXPECT_EQ(GetInferShapeFunc()(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(ShapeToVec(*holder.GetContext()->GetOutputShape(0)), (std::vector<int64_t>{1, 3, -1}));
}

TEST_F(BitcastUT, Infershape_FAILED_Int4ToInt32_KeepDim_LastDimNotRight)
{
    // last dim 15 is not a multiple of (int32 bits / int4 bits = 8)
    auto xTensor = MakeXTensor({1, 3, 15}, ge::DT_INT4);
    gert::StorageShape yShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Bitcast")
                      .NodeIoNum(1, 1)
                      .InputTensors({&xTensor})
                      .OutputShapes({&yShape})
                      .Attr("type", static_cast<int64_t>(ge::DT_INT32))
                      .Attr("keep_dim", true)
                      .Build();

    EXPECT_EQ(GetInferShapeFunc()(holder.GetContext()), ge::GRAPH_FAILED);
}

TEST_F(BitcastUT, InfershapeForScalar_FAILED_Int64ToF16_KeepDim)
{
    // scalar is not allowed when keep_dim is true
    auto xTensor = MakeXTensor({}, ge::DT_INT64);
    gert::StorageShape yShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Bitcast")
                      .NodeIoNum(1, 1)
                      .InputTensors({&xTensor})
                      .OutputShapes({&yShape})
                      .Attr("type", static_cast<int64_t>(ge::DT_FLOAT16))
                      .Attr("keep_dim", true)
                      .Build();

    EXPECT_EQ(GetInferShapeFunc()(holder.GetContext()), ge::GRAPH_FAILED);
}

// no keep_dim
TEST_F(BitcastUT, Infershape_OK_Int32ToInt4)
{
    auto xTensor = MakeXTensor({1, 3, 256}, ge::DT_INT32);
    gert::StorageShape yShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Bitcast")
                      .NodeIoNum(1, 1)
                      .InputTensors({&xTensor})
                      .OutputShapes({&yShape})
                      .Attr("type", static_cast<int64_t>(ge::DT_INT4))
                      .Build();

    EXPECT_EQ(GetInferShapeFunc()(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(ShapeToVec(*holder.GetContext()->GetOutputShape(0)), (std::vector<int64_t>{1, 3, 256, 8}));
}

TEST_F(BitcastUT, Infershape_OK_Int4ToInt32)
{
    auto xTensor = MakeXTensor({1, 3, 8}, ge::DT_INT4);
    gert::StorageShape yShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Bitcast")
                      .NodeIoNum(1, 1)
                      .InputTensors({&xTensor})
                      .OutputShapes({&yShape})
                      .Attr("type", static_cast<int64_t>(ge::DT_INT32))
                      .Build();

    EXPECT_EQ(GetInferShapeFunc()(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(ShapeToVec(*holder.GetContext()->GetOutputShape(0)), (std::vector<int64_t>{1, 3}));
}

TEST_F(BitcastUT, Infershape_OK_FP32ToInt4)
{
    auto xTensor = MakeXTensor({1, 3, 8}, ge::DT_FLOAT);
    gert::StorageShape yShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Bitcast")
                      .NodeIoNum(1, 1)
                      .InputTensors({&xTensor})
                      .OutputShapes({&yShape})
                      .Attr("type", static_cast<int64_t>(ge::DT_INT4))
                      .Build();

    EXPECT_EQ(GetInferShapeFunc()(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(ShapeToVec(*holder.GetContext()->GetOutputShape(0)), (std::vector<int64_t>{1, 3, 8, 8}));
}

TEST_F(BitcastUT, Infershape_OK_Int4ToFP32)
{
    auto xTensor = MakeXTensor({1, 3, 8}, ge::DT_INT4);
    gert::StorageShape yShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Bitcast")
                      .NodeIoNum(1, 1)
                      .InputTensors({&xTensor})
                      .OutputShapes({&yShape})
                      .Attr("type", static_cast<int64_t>(ge::DT_FLOAT))
                      .Build();

    EXPECT_EQ(GetInferShapeFunc()(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(ShapeToVec(*holder.GetContext()->GetOutputShape(0)), (std::vector<int64_t>{1, 3}));
}

TEST_F(BitcastUT, Infershape_OK_Int4ToInt2)
{
    auto xTensor = MakeXTensor({1, 3, 8}, ge::DT_INT4);
    gert::StorageShape yShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Bitcast")
                      .NodeIoNum(1, 1)
                      .InputTensors({&xTensor})
                      .OutputShapes({&yShape})
                      .Attr("type", static_cast<int64_t>(ge::DT_INT2))
                      .Build();

    EXPECT_EQ(GetInferShapeFunc()(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(ShapeToVec(*holder.GetContext()->GetOutputShape(0)), (std::vector<int64_t>{1, 3, 8, 2}));
}

TEST_F(BitcastUT, Infershape_OK_Int2ToInt8)
{
    auto xTensor = MakeXTensor({1, 3, 4}, ge::DT_INT2);
    gert::StorageShape yShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Bitcast")
                      .NodeIoNum(1, 1)
                      .InputTensors({&xTensor})
                      .OutputShapes({&yShape})
                      .Attr("type", static_cast<int64_t>(ge::DT_INT8))
                      .Build();

    EXPECT_EQ(GetInferShapeFunc()(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(ShapeToVec(*holder.GetContext()->GetOutputShape(0)), (std::vector<int64_t>{1, 3}));
}

TEST_F(BitcastUT, Infershape_OK_FP32ToInt32)
{
    // same dtype size: shape keeps unchanged
    auto xTensor = MakeXTensor({1, 3, 8}, ge::DT_FLOAT);
    gert::StorageShape yShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Bitcast")
                      .NodeIoNum(1, 1)
                      .InputTensors({&xTensor})
                      .OutputShapes({&yShape})
                      .Attr("type", static_cast<int64_t>(ge::DT_INT32))
                      .Build();

    EXPECT_EQ(GetInferShapeFunc()(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(ShapeToVec(*holder.GetContext()->GetOutputShape(0)), (std::vector<int64_t>{1, 3, 8}));
}

TEST_F(BitcastUT, Infershape_OK_DoubleToUint64)
{
    // same dtype size: shape keeps unchanged
    auto xTensor = MakeXTensor({1, 3, 8}, ge::DT_DOUBLE);
    gert::StorageShape yShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Bitcast")
                      .NodeIoNum(1, 1)
                      .InputTensors({&xTensor})
                      .OutputShapes({&yShape})
                      .Attr("type", static_cast<int64_t>(ge::DT_UINT64))
                      .Build();

    EXPECT_EQ(GetInferShapeFunc()(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(ShapeToVec(*holder.GetContext()->GetOutputShape(0)), (std::vector<int64_t>{1, 3, 8}));
}

TEST_F(BitcastUT, InferDynamicshape_OK_Bf16ToUint64)
{
    auto xTensor = MakeXTensor({-1, 3, -1}, ge::DT_BF16);
    gert::StorageShape yShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Bitcast")
                      .NodeIoNum(1, 1)
                      .InputTensors({&xTensor})
                      .OutputShapes({&yShape})
                      .Attr("type", static_cast<int64_t>(ge::DT_UINT64))
                      .Build();

    EXPECT_EQ(GetInferShapeFunc()(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(ShapeToVec(*holder.GetContext()->GetOutputShape(0)), (std::vector<int64_t>{-1, 3}));
}

TEST_F(BitcastUT, InferDynamicshape_OK_Uint64ToBf16)
{
    auto xTensor = MakeXTensor({-1, 3, -1}, ge::DT_UINT64);
    gert::StorageShape yShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Bitcast")
                      .NodeIoNum(1, 1)
                      .InputTensors({&xTensor})
                      .OutputShapes({&yShape})
                      .Attr("type", static_cast<int64_t>(ge::DT_BF16))
                      .Build();

    EXPECT_EQ(GetInferShapeFunc()(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(ShapeToVec(*holder.GetContext()->GetOutputShape(0)), (std::vector<int64_t>{-1, 3, -1, 4}));
}

TEST_F(BitcastUT, Infershape_Failed_DT_STRING)
{
    auto xTensor = MakeXTensor({-1, 3, -1}, ge::DT_STRING);
    gert::StorageShape yShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Bitcast")
                      .NodeIoNum(1, 1)
                      .InputTensors({&xTensor})
                      .OutputShapes({&yShape})
                      .Attr("type", static_cast<int64_t>(ge::DT_BF16))
                      .Build();

    EXPECT_EQ(GetInferShapeFunc()(holder.GetContext()), ge::GRAPH_FAILED);
}

TEST_F(BitcastUT, Infershape_Failed_DT_STRING_REF)
{
    auto xTensor = MakeXTensor({-1, 3, -1}, ge::DT_STRING_REF);
    gert::StorageShape yShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Bitcast")
                      .NodeIoNum(1, 1)
                      .InputTensors({&xTensor})
                      .OutputShapes({&yShape})
                      .Attr("type", static_cast<int64_t>(ge::DT_BF16))
                      .Build();

    EXPECT_EQ(GetInferShapeFunc()(holder.GetContext()), ge::GRAPH_FAILED);
}

TEST_F(BitcastUT, Infershape_Failed_DT_UNDEFINED)
{
    auto xTensor = MakeXTensor({-1, 3, -1}, ge::DT_UNDEFINED);
    gert::StorageShape yShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Bitcast")
                      .NodeIoNum(1, 1)
                      .InputTensors({&xTensor})
                      .OutputShapes({&yShape})
                      .Attr("type", static_cast<int64_t>(ge::DT_BF16))
                      .Build();

    EXPECT_EQ(GetInferShapeFunc()(holder.GetContext()), ge::GRAPH_FAILED);
}

TEST_F(BitcastUT, Infershape_Failed_DT_DUAL_2_f16)
{
    // DT_DUAL has unknown dtype size (-1)
    auto xTensor = MakeXTensor({-1, 3, -1}, ge::DT_DUAL);
    gert::StorageShape yShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Bitcast")
                      .NodeIoNum(1, 1)
                      .InputTensors({&xTensor})
                      .OutputShapes({&yShape})
                      .Attr("type", static_cast<int64_t>(ge::DT_FLOAT16))
                      .Build();

    EXPECT_EQ(GetInferShapeFunc()(holder.GetContext()), ge::GRAPH_FAILED);
}

TEST_F(BitcastUT, Infershape_Failed_f16_2_DT_DUAL)
{
    // DT_DUAL has unknown dtype size (-1)
    auto xTensor = MakeXTensor({-1, 3, -1}, ge::DT_FLOAT16);
    gert::StorageShape yShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Bitcast")
                      .NodeIoNum(1, 1)
                      .InputTensors({&xTensor})
                      .OutputShapes({&yShape})
                      .Attr("type", static_cast<int64_t>(ge::DT_DUAL))
                      .Build();

    EXPECT_EQ(GetInferShapeFunc()(holder.GetContext()), ge::GRAPH_FAILED);
}

TEST_F(BitcastUT, Infershape_Failed_LastDimNotEqualToDtypeTimes)
{
    // last dim 8 is not equal to (int64 size / float16 size = 4)
    auto xTensor = MakeXTensor({-1, 3, 8}, ge::DT_FLOAT16);
    gert::StorageShape yShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Bitcast")
                      .NodeIoNum(1, 1)
                      .InputTensors({&xTensor})
                      .OutputShapes({&yShape})
                      .Attr("type", static_cast<int64_t>(ge::DT_INT64))
                      .Build();

    EXPECT_EQ(GetInferShapeFunc()(holder.GetContext()), ge::GRAPH_FAILED);
}

TEST_F(BitcastUT, InfershapeForScalar_OK_Int64ToF16)
{
    // scalar is expanded to a 1-dim vector
    auto xTensor = MakeXTensor({}, ge::DT_INT64);
    gert::StorageShape yShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Bitcast")
                      .NodeIoNum(1, 1)
                      .InputTensors({&xTensor})
                      .OutputShapes({&yShape})
                      .Attr("type", static_cast<int64_t>(ge::DT_FLOAT16))
                      .Build();

    EXPECT_EQ(GetInferShapeFunc()(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(ShapeToVec(*holder.GetContext()->GetOutputShape(0)), (std::vector<int64_t>{4}));
}

TEST_F(BitcastUT, InfershapeForScalar_FAILED_F16ToInt64)
{
    // scalar can not be squeezed any more
    auto xTensor = MakeXTensor({}, ge::DT_FLOAT16);
    gert::StorageShape yShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Bitcast")
                      .NodeIoNum(1, 1)
                      .InputTensors({&xTensor})
                      .OutputShapes({&yShape})
                      .Attr("type", static_cast<int64_t>(ge::DT_INT64))
                      .Build();

    EXPECT_EQ(GetInferShapeFunc()(holder.GetContext()), ge::GRAPH_FAILED);
}

TEST_F(BitcastUT, InfershapeForEmptyTensor_FAILED_F16ToInt64)
{
    // last dim 0 is not equal to (int64 size / float16 size = 4)
    auto xTensor = MakeXTensor({1, 3, 0}, ge::DT_FLOAT16);
    gert::StorageShape yShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Bitcast")
                      .NodeIoNum(1, 1)
                      .InputTensors({&xTensor})
                      .OutputShapes({&yShape})
                      .Attr("type", static_cast<int64_t>(ge::DT_INT64))
                      .Build();

    EXPECT_EQ(GetInferShapeFunc()(holder.GetContext()), ge::GRAPH_FAILED);
}

TEST_F(BitcastUT, InfershapeForEmptyTensor_OK_F16ToInt64)
{
    // empty tensor stays empty after squeezing the last dim
    auto xTensor = MakeXTensor({1, 3, 0, 4}, ge::DT_FLOAT16);
    gert::StorageShape yShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Bitcast")
                      .NodeIoNum(1, 1)
                      .InputTensors({&xTensor})
                      .OutputShapes({&yShape})
                      .Attr("type", static_cast<int64_t>(ge::DT_INT64))
                      .Build();

    EXPECT_EQ(GetInferShapeFunc()(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(ShapeToVec(*holder.GetContext()->GetOutputShape(0)), (std::vector<int64_t>{1, 3, 0}));
}

TEST_F(BitcastUT, InfershapeForEmptyTensor_OK_Int64ToF16)
{
    // empty tensor stays empty after appending the multiple dim
    auto xTensor = MakeXTensor({1, 3, 0}, ge::DT_INT64);
    gert::StorageShape yShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Bitcast")
                      .NodeIoNum(1, 1)
                      .InputTensors({&xTensor})
                      .OutputShapes({&yShape})
                      .Attr("type", static_cast<int64_t>(ge::DT_FLOAT16))
                      .Build();

    EXPECT_EQ(GetInferShapeFunc()(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(ShapeToVec(*holder.GetContext()->GetOutputShape(0)), (std::vector<int64_t>{1, 3, 0, 4}));
}

TEST_F(BitcastUT, InfershapeRange_OK_Int64ToF16)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry->GetOpImpl("Bitcast"), nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("Bitcast");
    ASSERT_NE(opImpl->infer_shape_range, nullptr);

    gert::OpInferShapeRangeContextBuilder builder;
    builder.OpType("Bitcast").OpName("Bitcast");
    builder.IONum(1, 1);
    builder.OutputTensorDesc(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.AppendAttr(static_cast<int64_t>(ge::DT_FLOAT16));

    XShapeRange xRange({1, 3, 0}, {1, 3, 256}, ge::DT_INT64);
    builder.InputTensorsRange({&xRange.range});
    auto contextHolder = builder.Build();
    auto* context = contextHolder.GetContext();
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(opImpl->infer_shape_range(context), ge::GRAPH_SUCCESS);
    auto* yRange = context->GetOutputShapeRange(0);
    ASSERT_NE(yRange, nullptr);
    EXPECT_EQ(ShapeToVec(*(yRange->GetMin())), std::vector<int64_t>({1, 3, 0, 4}));
    EXPECT_EQ(ShapeToVec(*(yRange->GetMax())), std::vector<int64_t>({1, 3, 256, 4}));
}

TEST_F(BitcastUT, InfershapeRange_OK_F16ToF16)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry->GetOpImpl("Bitcast"), nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("Bitcast");
    ASSERT_NE(opImpl->infer_shape_range, nullptr);

    gert::OpInferShapeRangeContextBuilder builder;
    builder.OpType("Bitcast").OpName("Bitcast");
    builder.IONum(1, 1);
    builder.OutputTensorDesc(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.AppendAttr(static_cast<int64_t>(ge::DT_FLOAT16));

    XShapeRange xRange({1, 3, 0}, {1, 3, 256}, ge::DT_FLOAT16);
    builder.InputTensorsRange({&xRange.range});
    auto contextHolder = builder.Build();
    auto* context = contextHolder.GetContext();
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(opImpl->infer_shape_range(context), ge::GRAPH_SUCCESS);
    auto* yRange = context->GetOutputShapeRange(0);
    ASSERT_NE(yRange, nullptr);
    EXPECT_EQ(ShapeToVec(*(yRange->GetMin())), std::vector<int64_t>({1, 3, 0}));
    EXPECT_EQ(ShapeToVec(*(yRange->GetMax())), std::vector<int64_t>({1, 3, 256}));
}

TEST_F(BitcastUT, InfershapeRange_OK_F16ToInt64)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry->GetOpImpl("Bitcast"), nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("Bitcast");
    ASSERT_NE(opImpl->infer_shape_range, nullptr);

    gert::OpInferShapeRangeContextBuilder builder;
    builder.OpType("Bitcast").OpName("Bitcast");
    builder.IONum(1, 1);
    builder.OutputTensorDesc(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.AppendAttr(static_cast<int64_t>(ge::DT_INT64));

    XShapeRange xRange({1, 0, 0}, {1, 3, 256}, ge::DT_FLOAT16);
    builder.InputTensorsRange({&xRange.range});
    auto contextHolder = builder.Build();
    auto* context = contextHolder.GetContext();
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(opImpl->infer_shape_range(context), ge::GRAPH_SUCCESS);
    auto* yRange = context->GetOutputShapeRange(0);
    ASSERT_NE(yRange, nullptr);
    EXPECT_EQ(ShapeToVec(*(yRange->GetMin())), std::vector<int64_t>({1, 0}));
    EXPECT_EQ(ShapeToVec(*(yRange->GetMax())), std::vector<int64_t>({1, 3}));
}

TEST_F(BitcastUT, InfershapeRange_OK_ShapeRangeIsEmpty)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry->GetOpImpl("Bitcast"), nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("Bitcast");
    ASSERT_NE(opImpl->infer_shape_range, nullptr);

    gert::OpInferShapeRangeContextBuilder builder;
    builder.OpType("Bitcast").OpName("Bitcast");
    builder.IONum(1, 1);
    builder.OutputTensorDesc(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.AppendAttr(static_cast<int64_t>(ge::DT_INT64));

    XShapeRange xRange({}, {}, ge::DT_FLOAT16);
    builder.InputTensorsRange({&xRange.range});
    auto contextHolder = builder.Build();
    auto* context = contextHolder.GetContext();
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(opImpl->infer_shape_range(context), ge::GRAPH_SUCCESS);
    auto* yRange = context->GetOutputShapeRange(0);
    ASSERT_NE(yRange, nullptr);
    EXPECT_EQ(ShapeToVec(*(yRange->GetMin())), std::vector<int64_t>({}));
    EXPECT_EQ(ShapeToVec(*(yRange->GetMax())), std::vector<int64_t>({}));
}

TEST_F(BitcastUT, InfershapeRange_Failed_MinSizeNotEqualToMaxSize)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry->GetOpImpl("Bitcast"), nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("Bitcast");
    ASSERT_NE(opImpl->infer_shape_range, nullptr);

    gert::OpInferShapeRangeContextBuilder builder;
    builder.OpType("Bitcast").OpName("Bitcast");
    builder.IONum(1, 1);
    builder.OutputTensorDesc(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.AppendAttr(static_cast<int64_t>(ge::DT_INT64));

    XShapeRange xRange({1, 3, 0, 1}, {1, 3, 256}, ge::DT_FLOAT16);
    builder.InputTensorsRange({&xRange.range});
    auto contextHolder = builder.Build();
    auto* context = contextHolder.GetContext();
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(opImpl->infer_shape_range(context), ge::GRAPH_FAILED);
}

TEST_F(BitcastUT, InferDataType_OK_TargetDtypeIsInt64)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry->GetOpImpl("Bitcast"), nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("Bitcast");
    ASSERT_NE(opImpl->infer_datatype, nullptr);

    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("Bitcast").OpName("Bitcast");
    builder.IONum(1, 1);
    builder.InputTensorDesc(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.AppendAttr(static_cast<int64_t>(ge::DT_INT64));
    auto contextHolder = builder.Build();
    auto* context = contextHolder.GetContext();
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(opImpl->infer_datatype(context), ge::GRAPH_SUCCESS);
    // output dtype is the target dtype carried by the type attr
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_INT64);
}

TEST_F(BitcastUT, Infershape_OK_Int8ToFP4e2m1_KeepDim)
{
    auto xTensor = MakeXTensor({1, 3, 256}, ge::DT_INT8);
    gert::StorageShape yShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Bitcast")
                      .NodeIoNum(1, 1)
                      .InputTensors({&xTensor})
                      .OutputShapes({&yShape})
                      .Attr("type", static_cast<int64_t>(ge::DT_FLOAT4_E2M1))
                      .Attr("keep_dim", true)
                      .Build();

    EXPECT_EQ(GetInferShapeFunc()(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(ShapeToVec(*holder.GetContext()->GetOutputShape(0)), (std::vector<int64_t>{1, 3, 512}));
}
