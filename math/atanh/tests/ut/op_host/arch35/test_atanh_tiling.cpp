/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>
#include <limits>
#include <gtest/gtest.h>
#include "base/registry/op_impl_space_registry_v2.h"
#include "experiment/metadef/common/util/error_manager/error_manager.h"
#include "tiling_case_executor.h"
#include "math/atanh/op_kernel/arch35/atanh_tiling_data.h"

namespace {
constexpr int64_t MAX_ELEMENTS = std::numeric_limits<int64_t>::max();

gert::TilingContextPara MakeContext(const gert::Shape& inputShape, const gert::Shape& outputShape,
                                    ge::DataType inputType = ge::DT_FLOAT, ge::DataType outputType = ge::DT_FLOAT,
                                    ge::Format inputFormat = ge::FORMAT_ND, ge::Format outputFormat = ge::FORMAT_ND)
{
    gert::StorageShape inputStorageShape;
    inputStorageShape.MutableOriginShape() = inputShape;
    inputStorageShape.MutableStorageShape() = inputShape;
    gert::StorageShape outputStorageShape;
    outputStorageShape.MutableOriginShape() = outputShape;
    outputStorageShape.MutableStorageShape() = outputShape;
    // The SDK context builder requires a live compile-info object, even when the tiler does not read it.
    static uint8_t compileInfo = 0;
    return gert::TilingContextPara("Atanh", {{inputStorageShape, inputType, inputFormat}},
                                   {{outputStorageShape, outputType, outputFormat}}, {}, &compileInfo);
}

class AtanhTilingTest : public testing::Test {};

void ExpectTilingFailure(const gert::TilingContextPara& context, const char* errorCode)
{
    auto& errors = ErrorManager::GetInstance();
    ASSERT_EQ(errors.Init(), 0);
    errors.GetRawErrorMessages();
    ExecuteTestCase(context, ge::GRAPH_FAILED);
    const auto reports = errors.GetRawErrorMessages();
    ASSERT_EQ(reports.size(), 1U);
    EXPECT_EQ(reports[0].error_id, errorCode);
    EXPECT_NE(reports[0].error_message.find("Atanh"), std::string::npos);
}

TEST_F(AtanhTilingTest, SupportedTypesAndElementCounts)
{
    const std::vector<gert::Shape> shapes = {{},      {0, 3},         {1},           {1023}, {1024}, {1025},
                                             {65537}, {2147483649LL}, {MAX_ELEMENTS}};
    const std::vector<int64_t> counts = {1, 0, 1, 1023, 1024, 1025, 65537, 2147483649LL, MAX_ELEMENTS};
    const std::vector<ge::DataType> types = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
    for (size_t type = 0; type < types.size(); ++type) {
        for (size_t i = 0; i < shapes.size(); ++i) {
            SCOPED_TRACE(::testing::Message() << "dtype=" << types[type] << " case=" << i);
            TilingInfo info;
            ASSERT_TRUE(ExecuteTiling(MakeContext(shapes[i], shapes[i], types[type], types[type]), info));
            EXPECT_EQ(info.tilingKey, static_cast<int64_t>(type));
            EXPECT_GE(info.blockNum, 1U);
            EXPECT_LE(info.blockNum, 64U);
            EXPECT_EQ(info.workspaceSizes, std::vector<int64_t>({0}));
            ASSERT_EQ(info.tilingDataSize, sizeof(AtanhTilingData));
            AtanhTilingData data;
            std::memcpy(&data, info.tilingData.get(), sizeof(data));
            EXPECT_EQ(data.totalNum, counts[i]);
        }
    }
}

TEST_F(AtanhTilingTest, RejectsInvalidShapes)
{
    const std::vector<gert::Shape> invalidShapes = {
        {-1}, {-2, -2}, {0, -1}, {MAX_ELEMENTS, 2}, {4294967296LL, 4294967296LL}};
    for (size_t i = 0; i < invalidShapes.size(); ++i) {
        SCOPED_TRACE(i);
        ExpectTilingFailure(MakeContext(invalidShapes[i], invalidShapes[i]), "EZ0009");
    }
    // Equal element counts do not make different shapes compatible.
    ExpectTilingFailure(MakeContext({2, 3}, {3, 2}), "EZ0008");
    ExpectTilingFailure(MakeContext({2, 3}, {2}), "EZ0008");
    ExpectTilingFailure(MakeContext({0, 3}, {0, 4}), "EZ0008");
}

TEST_F(AtanhTilingTest, EmptyShapeDoesNotOverflow)
{
    ExecuteTestCase(MakeContext({MAX_ELEMENTS, 2, 0}, {MAX_ELEMENTS, 2, 0}), ge::GRAPH_SUCCESS, 0, "0 ", {0});
}

TEST_F(AtanhTilingTest, RejectsInvalidDtypes)
{
    ExpectTilingFailure(MakeContext({4}, {4}, ge::DT_INT32, ge::DT_INT32), "EZ0019");
    ExpectTilingFailure(MakeContext({4}, {4}, ge::DT_FLOAT, ge::DT_FLOAT16), "EZ0019");
}

TEST_F(AtanhTilingTest, SupportsPublicFormats)
{
    const std::vector<gert::Shape> shapes = {{2, 3, 4}, {2, 3, 4, 5}, {2, 3, 4, 5, 6}};
    const std::vector<ge::Format> formats = {ge::FORMAT_NCL, ge::FORMAT_NCHW, ge::FORMAT_NCDHW};
    const std::vector<int64_t> counts = {24, 120, 720};
    for (size_t i = 0; i < formats.size(); ++i) {
        for (const auto inputFormat : {ge::FORMAT_ND, formats[i]}) {
            for (const auto outputFormat : {ge::FORMAT_ND, formats[i]}) {
                SCOPED_TRACE(::testing::Message() << "input=" << inputFormat << " output=" << outputFormat);
                TilingInfo info;
                ASSERT_TRUE(ExecuteTiling(
                    MakeContext(shapes[i], shapes[i], ge::DT_FLOAT, ge::DT_FLOAT, inputFormat, outputFormat), info));
                ASSERT_EQ(info.tilingDataSize, sizeof(AtanhTilingData));
                AtanhTilingData data;
                std::memcpy(&data, info.tilingData.get(), sizeof(data));
                EXPECT_EQ(data.totalNum, counts[i]);
            }
        }
    }
}

TEST_F(AtanhTilingTest, NullContextReportsStageErrorOnce)
{
    const auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(registry, nullptr);
    const auto* implementation = registry->GetOpImpl("Atanh");
    ASSERT_NE(implementation, nullptr);
    ASSERT_NE(implementation->tiling, nullptr);
    ASSERT_NE(implementation->infer_shape, nullptr);
    auto& errors = ErrorManager::GetInstance();
    ASSERT_EQ(errors.Init(), 0);
    errors.GetRawErrorMessages();
    EXPECT_EQ(implementation->tiling(nullptr), ge::GRAPH_FAILED);
    auto reports = errors.GetRawErrorMessages();
    ASSERT_EQ(reports.size(), 1U);
    EXPECT_EQ(reports[0].error_id, "EZ1008");
    EXPECT_EQ(implementation->infer_shape(nullptr), ge::GRAPH_FAILED);
    reports = errors.GetRawErrorMessages();
    ASSERT_EQ(reports.size(), 1U);
    EXPECT_EQ(reports[0].error_id, "EZ1014");
}
} // namespace
