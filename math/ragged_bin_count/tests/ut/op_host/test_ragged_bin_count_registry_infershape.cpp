/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "base/context_builder/op_infer_shape_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "exe_graph/runtime/runtime_tensor.h"

namespace {
gert::StorageShape MakeStorageShape(const gert::Shape& shape)
{
    gert::StorageShape storage;
    storage.MutableOriginShape() = shape;
    storage.MutableStorageShape() = shape;
    return storage;
}

void CheckRegisteredInfer(const gert::Shape& valuesShape, const gert::Shape& weightsShape, const gert::Shape& expected,
                          const gert::Shape& splitsShape = {2}, const gert::Shape& sizeShape = {1},
                          ge::DataType indexType = ge::DT_INT32, void* sizeData = nullptr,
                          ge::graphStatus status = ge::GRAPH_SUCCESS)
{
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(registry, nullptr);
    const auto* impl = registry->GetOpImpl("RaggedBinCount");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->infer_shape, nullptr);
    EXPECT_TRUE(impl->IsInputDataDependency(2U));
    const gert::StorageFormat format(ge::FORMAT_ND, ge::FORMAT_ND, {});
    gert::Tensor splits(MakeStorageShape(splitsShape), format, ge::DT_INT64);
    gert::Tensor values(MakeStorageShape(valuesShape), format, indexType);
    gert::Tensor size(MakeStorageShape(sizeShape), format, gert::TensorPlacement::kOnHost, indexType, sizeData);
    gert::Tensor weights(MakeStorageShape(weightsShape), format, ge::DT_FLOAT);
    gert::OpInferShapeContextBuilder builder;
    builder.OpType("RaggedBinCount").OpName("ragged_bin_count_registered").IONum(4, 1);
    builder.InputTensors({&splits, &values, &size, &weights});
    builder.OutputTensorDesc(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    auto holder = builder.Build();
    ASSERT_NE(holder.GetContext(), nullptr);
    auto* context = holder.GetContext();
    ASSERT_NE(context->GetOutputShape(0), nullptr);
    *context->GetOutputShape(0) = gert::Shape({91, 92, 93});
    ASSERT_EQ(impl->infer_shape(context), status);
    EXPECT_EQ(*context->GetInputShape(0), splitsShape);
    EXPECT_EQ(*context->GetInputShape(1), valuesShape);
    EXPECT_EQ(*context->GetInputShape(2), sizeShape);
    EXPECT_EQ(*context->GetInputShape(3), weightsShape);
    if (status == ge::GRAPH_SUCCESS) {
        EXPECT_EQ(*context->GetOutputShape(0), expected);
        ASSERT_EQ(impl->infer_shape(context), ge::GRAPH_SUCCESS);
        EXPECT_EQ(*context->GetOutputShape(0), expected);
        EXPECT_EQ(*context->GetInputShape(1), valuesShape);
        EXPECT_EQ(*context->GetInputShape(3), weightsShape);
    }
}
} // namespace

TEST(RaggedBinCountRegisteredInfer, PreservesMatrixWeightsForVectorValues)
{
    CheckRegisteredInfer({6}, {2, 3}, {1, -1});
}

TEST(RaggedBinCountRegisteredInfer, PreservesVectorWeightsForMatrixValues)
{
    CheckRegisteredInfer({2, 3}, {6}, {1, -1});
}

TEST(RaggedBinCountRegisteredInfer, PreservesDifferentMatrixShapes) { CheckRegisteredInfer({2, 3}, {3, 2}, {1, -1}); }

TEST(RaggedBinCountRegisteredInfer, PreservesEmptyWeights)
{
    CheckRegisteredInfer({2, 3}, {0}, {1, -1});
    CheckRegisteredInfer({2, 3}, {0, 3}, {1, -1});
    CheckRegisteredInfer({2, 3}, {2, 0}, {1, -1});
}

TEST(RaggedBinCountRegisteredInfer, PreservesScalarWeights) { CheckRegisteredInfer({1}, {}, {1, -1}); }

TEST(RaggedBinCountRegisteredInfer, ReadsInt32ConstantSize)
{
    int32_t bins = 7;
    CheckRegisteredInfer({6}, {2, 3}, {1, 7}, {2}, {1}, ge::DT_INT32, &bins);
}

TEST(RaggedBinCountRegisteredInfer, ReadsInt64ConstantSizeWithoutTruncation)
{
    int64_t bins = (int64_t{1} << 32) + 7;
    CheckRegisteredInfer({6}, {2, 3}, {1, bins}, {2}, {1}, ge::DT_INT64, &bins);
}

TEST(RaggedBinCountRegisteredInfer, AcceptsZeroBins)
{
    int64_t bins = 0;
    CheckRegisteredInfer({6}, {2, 3}, {1, 0}, {2}, {1}, ge::DT_INT64, &bins);
}

TEST(RaggedBinCountRegisteredInfer, RejectsNegativeBins)
{
    int32_t bins32 = -1;
    int64_t bins64 = -1;
    CheckRegisteredInfer({6}, {2, 3}, {}, {2}, {1}, ge::DT_INT32, &bins32, ge::GRAPH_FAILED);
    CheckRegisteredInfer({6}, {2, 3}, {}, {2}, {1}, ge::DT_INT64, &bins64, ge::GRAPH_FAILED);
}

TEST(RaggedBinCountRegisteredInfer, RetainsKnownRowsWhenSizeIsUnknown)
{
    CheckRegisteredInfer({6}, {2, 3}, {3, -1}, {4});
}

TEST(RaggedBinCountRegisteredInfer, PreservesUnknownInputDimensions)
{
    CheckRegisteredInfer({-1}, {-1, 3}, {-1, -1}, {-1});
}

TEST(RaggedBinCountRegisteredInfer, PreservesUnknownInputRanks)
{
    CheckRegisteredInfer({-2}, {-2}, {-1, -1}, {-2}, {-2});
}

TEST(RaggedBinCountRegisteredInfer, RejectsInvalidSplitsAndSizeShapes)
{
    CheckRegisteredInfer({6}, {2, 3}, {}, {}, {1}, ge::DT_INT32, nullptr, ge::GRAPH_FAILED);
    CheckRegisteredInfer({6}, {2, 3}, {}, {1}, {1}, ge::DT_INT32, nullptr, ge::GRAPH_FAILED);
    CheckRegisteredInfer({6}, {2, 3}, {}, {2, 1}, {1}, ge::DT_INT32, nullptr, ge::GRAPH_FAILED);
    CheckRegisteredInfer({6}, {2, 3}, {}, {2}, {}, ge::DT_INT32, nullptr, ge::GRAPH_FAILED);
    CheckRegisteredInfer({6}, {2, 3}, {}, {2}, {int64_t{1} << 31}, ge::DT_INT32, nullptr, ge::GRAPH_FAILED);
    CheckRegisteredInfer({6}, {2, 3}, {}, {2}, {int64_t{1} << 32}, ge::DT_INT32, nullptr, ge::GRAPH_FAILED);
}

TEST(RaggedBinCountRegisteredInfer, RejectsMismatchedNonemptyElementCounts)
{
    CheckRegisteredInfer({6}, {2, 2}, {}, {2}, {1}, ge::DT_INT32, nullptr, ge::GRAPH_FAILED);
}
