/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "base/registry/op_impl_space_registry_v2.h"
#include "gtest/gtest.h"
#include "platform/platform_infos_def.h"
#include "tiling_context_faker.h"
#include "../../../../op_kernel/arch35/dense_bincount_tiling_data.h"
#include "../../../../op_kernel/arch35/dense_bincount_tiling_key.h"

namespace {
constexpr uint64_t TILING_DATA_SIZE = 4096U;
constexpr uint64_t INVALID_TILING_KEY = UINT64_MAX;
const uint64_t ND_ORDERED_TILING_KEY = GET_TPL_TILING_KEY(SCH_MODE_ND_COUNT_WEIGHT_ORDERED);
const uint64_t ONE_D_ORDERED_TILING_KEY = GET_TPL_TILING_KEY(SCH_MODE_1D_COUNT_WEIGHT_ORDERED);
const uint64_t ND_DETERMINISTIC_TILING_KEY = GET_TPL_TILING_KEY(SCH_MODE_ND_COUNT_WEIGHT_DETERMINISTIC);
const uint64_t ONE_D_DETERMINISTIC_TILING_KEY = GET_TPL_TILING_KEY(SCH_MODE_1D_COUNT_WEIGHT_DETERMINISTIC);
constexpr uint64_t DETERMINISTIC_PARTIAL_BYTES = 64U;
constexpr uint64_t DETERMINISTIC_HISTOGRAM_SLOT_BYTES = 128U;
struct DenseBincountCompileInfo {};

struct DenseBincountTilingCase {
    std::vector<int64_t> inputShape{2, 4};
    std::vector<int64_t> sizeShape{1};
    std::vector<int64_t> weightsShape{2, 4};
    ge::DataType inputDtype = ge::DT_INT32;
    ge::DataType sizeDtype = ge::DT_INT32;
    ge::DataType weightsDtype = ge::DT_FLOAT;
    ge::DataType outputDtype = ge::DT_FLOAT;
    int64_t size = 4;
    bool binaryOutput = false;
    bool setBinaryOutput = true;
    int32_t deterministic = 0;
    uint32_t coreNum = 64U;
    uint64_t ubSize = 262144U;
};

struct DenseBincountTilingResult {
    ge::graphStatus status = ge::GRAPH_FAILED;
    uint64_t key = INVALID_TILING_KEY;
    uint32_t privateHistElems = 0U;
    uint32_t privateHistMode = 0U;
    uint32_t usedCoreNum = 0U;
    uint32_t deterministicMode = 0U;
    uint32_t deterministicPartitions = 0U;
    uint32_t deterministicBufferBytes = 0U;
    size_t workspaceSize = 0U;
};

gert::StorageShape MakeShape(const std::vector<int64_t>& dimensions)
{
    gert::StorageShape shape;
    for (const int64_t dimension : dimensions) {
        shape.MutableOriginShape().AppendDim(dimension);
        shape.MutableStorageShape().AppendDim(dimension);
    }
    return shape;
}

fe::PlatFormInfos MakePlatform(uint32_t coreNum, uint64_t ubSize)
{
    fe::PlatFormInfos platform;
    platform.Init();
    std::map<std::string, std::string> soc = {{"ai_core_cnt", std::to_string(coreNum)}, {"core_type_list", "AICore"}};
    std::map<std::string, std::string> spec = {{"ub_size", std::to_string(ubSize)}};
    std::map<std::string, std::string> version = {{"NpuArch", "3510"}, {"Short_SoC_version", "Ascend950"}};
    platform.SetPlatformRes("SoCInfo", soc);
    platform.SetPlatformRes("AICoreSpec", spec);
    platform.SetPlatformRes("version", version);
    platform.SetCoreNumByCoreType("AICore");
    return platform;
}

DenseBincountTilingResult ExecuteTiling(gert::ContextHolder<gert::TilingContext>& holder)
{
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    if (registry == nullptr) {
        return {};
    }
    auto functions = registry->GetOpImpl("DenseBincount");
    if (functions == nullptr || functions->tiling == nullptr) {
        return {};
    }
    DenseBincountTilingResult result;
    result.status = functions->tiling(holder.GetContext());
    if (result.status != ge::GRAPH_SUCCESS) {
        return result;
    }
    const auto* tiling = holder.GetContext()->GetTilingData<DenseBincountTilingData>();
    if (tiling == nullptr) {
        return {};
    }
    result.key = holder.GetContext()->GetTilingKey();
    result.privateHistElems = tiling->privateHistElems;
    result.privateHistMode = tiling->privateHistMode;
    result.usedCoreNum = tiling->usedCoreNum;
    result.deterministicMode = tiling->deterministicMode;
    result.deterministicPartitions = tiling->deterministicPartitions;
    result.deterministicBufferBytes = tiling->deterministicBufferBytes;
    result.workspaceSize = holder.GetContext()->GetWorkspaceSizes(1)[0];
    return result;
}

DenseBincountTilingResult RunTiling(const DenseBincountTilingCase& testCase)
{
    int64_t sizeData = testCase.size;
    const auto inputShape = MakeShape(testCase.inputShape);
    const auto weightsShape = MakeShape(testCase.weightsShape);
    const bool is1D = testCase.inputShape.size() == 1U;
    const int64_t rows = is1D ? 1 : testCase.inputShape.front();
    const auto outputShape = MakeShape(is1D ? std::vector<int64_t>{testCase.size} :
                                              std::vector<int64_t>{rows, testCase.size});
    gert::StorageFormat format(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType());
    gert::Tensor inputTensor(inputShape, format, gert::TensorPlacement::kOnHost, testCase.inputDtype, nullptr);
    gert::Tensor sizeTensor(MakeShape(testCase.sizeShape), format, gert::TensorPlacement::kOnHost, testCase.sizeDtype,
                            &sizeData);
    gert::Tensor weightsTensor(weightsShape, format, gert::TensorPlacement::kOnHost, testCase.weightsDtype, nullptr);
    gert::Tensor outputTensor(outputShape, format, gert::TensorPlacement::kOnHost, testCase.outputDtype, nullptr);
    auto platform = MakePlatform(testCase.coreNum, testCase.ubSize);
    auto data = gert::TilingData::CreateCap(TILING_DATA_SIZE);
    auto workspace = gert::ContinuousVector::Create<size_t>(1);
    DenseBincountCompileInfo compileInfo;
    int32_t deterministic = testCase.deterministic;
    gert::TilingContextFaker holderBuilder;
    holderBuilder.NodeIoNum(3, 1)
        .InputTensors({&inputTensor, &sizeTensor, &weightsTensor})
        .OutputTensors({&outputTensor})
        .DeterministicInfo(&deterministic)
        .SetOpType("DenseBincount")
        .CompileInfo(&compileInfo)
        .PlatformInfo(reinterpret_cast<char*>(&platform))
        .TilingData(data.get())
        .Workspace(reinterpret_cast<gert::ContinuousVector*>(workspace.get()));
    if (testCase.setBinaryOutput) {
        holderBuilder.Attr("binary_output", testCase.binaryOutput);
    }
    auto holder = holderBuilder.Build();
    return ExecuteTiling(holder);
}
} // namespace

TEST(DenseBincountTiling, optional_binary_output_defaults_to_false)
{
    DenseBincountTilingCase explicitFalse;
    explicitFalse.inputShape = {2, 128};
    explicitFalse.weightsShape = {2, 128};
    explicitFalse.size = 128;
    explicitFalse.binaryOutput = false;
    auto explicitResult = RunTiling(explicitFalse);
    ASSERT_EQ(explicitResult.status, ge::GRAPH_SUCCESS);

    auto omitted = explicitFalse;
    omitted.setBinaryOutput = false;
    auto omittedResult = RunTiling(omitted);
    ASSERT_EQ(omittedResult.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(omittedResult.key, explicitResult.key);
}

TEST(DenseBincountTiling, scalar_size_shape_is_supported)
{
    DenseBincountTilingCase scalarSize;
    scalarSize.sizeShape = {};
    const auto result = RunTiling(scalarSize);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.key, ND_ORDERED_TILING_KEY);
}

TEST(DenseBincountTiling, ordered_key_is_limited_to_bounded_weighted_accumulation)
{
    DenseBincountTilingCase normal;
    normal.inputShape = {2, 128};
    normal.weightsShape = {2, 128};
    normal.size = 128;
    auto result = RunTiling(normal);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.key, SCH_MODE_ND_COUNT_WEIGHT);

    auto deterministic = normal;
    deterministic.deterministic = 1;
    result = RunTiling(deterministic);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.key, ND_DETERMINISTIC_TILING_KEY);
    EXPECT_EQ(result.deterministicMode, 1U);

    deterministic.inputShape = {8};
    deterministic.weightsShape = {8};
    result = RunTiling(deterministic);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.key, ONE_D_DETERMINISTIC_TILING_KEY);

    deterministic.inputShape = {2, 4};
    deterministic.weightsShape = {0};
    result = RunTiling(deterministic);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.key, SCH_MODE_ND_COUNT_NOWEIGHT);

    deterministic.weightsShape = {2, 4};
    deterministic.binaryOutput = true;
    result = RunTiling(deterministic);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.key, SCH_MODE_ND_BINARY_WEIGHT);
}

TEST(DenseBincountTiling, deterministic_weighted_dispatch_and_workspace_boundaries)
{
    DenseBincountTilingCase partial;
    partial.inputShape = {65536};
    partial.weightsShape = {65536};
    partial.size = 256;
    partial.deterministic = 1;
    const auto partialResult = RunTiling(partial);
    ASSERT_EQ(partialResult.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(partialResult.key, ONE_D_DETERMINISTIC_TILING_KEY);
    EXPECT_EQ(partialResult.usedCoreNum, 64U);
    EXPECT_EQ(partialResult.deterministicMode, 1U);
    EXPECT_EQ(partialResult.deterministicPartitions, 4U);
    EXPECT_EQ(partialResult.privateHistElems, 0U);

    auto baseline = partial;
    baseline.deterministic = 0;
    const auto baselineResult = RunTiling(baseline);
    ASSERT_EQ(baselineResult.status, ge::GRAPH_SUCCESS);
    const size_t expectedPartialWorkspace = static_cast<size_t>(partialResult.usedCoreNum) * partial.size *
                                            partialResult.deterministicPartitions * DETERMINISTIC_PARTIAL_BYTES;
    EXPECT_EQ(partialResult.workspaceSize, baselineResult.workspaceSize + expectedPartialWorkspace);

    auto histogram = partial;
    histogram.inputShape = {65537};
    histogram.weightsShape = {65537};
    const auto histogramResult = RunTiling(histogram);
    ASSERT_EQ(histogramResult.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(histogramResult.key, ONE_D_DETERMINISTIC_TILING_KEY);
    EXPECT_EQ(histogramResult.deterministicMode, 2U);
    EXPECT_EQ(histogramResult.deterministicPartitions, 0U);
    EXPECT_EQ(histogramResult.deterministicBufferBytes, 32U * 1024U);
    EXPECT_EQ(histogramResult.workspaceSize,
              baselineResult.workspaceSize + histogram.size * DETERMINISTIC_HISTOGRAM_SLOT_BYTES);

    DenseBincountTilingCase partialBoundary;
    partialBoundary.inputShape = {64};
    partialBoundary.weightsShape = {64};
    partialBoundary.size = 262144;
    partialBoundary.coreNum = 1U;
    partialBoundary.deterministic = 1;
    const auto partialBoundaryResult = RunTiling(partialBoundary);
    ASSERT_EQ(partialBoundaryResult.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(partialBoundaryResult.deterministicMode, 1U);
    EXPECT_EQ(partialBoundaryResult.deterministicPartitions, 1U);

    partialBoundary.size = 262145;
    const auto afterPartialBoundary = RunTiling(partialBoundary);
    ASSERT_EQ(afterPartialBoundary.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(afterPartialBoundary.deterministicMode, 2U);

    DenseBincountTilingCase histogramBoundary;
    histogramBoundary.inputShape = {1};
    histogramBoundary.weightsShape = {1};
    histogramBoundary.size = 524288;
    histogramBoundary.deterministic = 1;
    const auto histogramBoundaryResult = RunTiling(histogramBoundary);
    ASSERT_EQ(histogramBoundaryResult.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(histogramBoundaryResult.deterministicMode, 2U);
    EXPECT_EQ(histogramBoundaryResult.workspaceSize,
              baselineResult.workspaceSize + histogramBoundary.size * DETERMINISTIC_HISTOGRAM_SLOT_BYTES);

    histogramBoundary.size = 524289;
    EXPECT_EQ(RunTiling(histogramBoundary).status, ge::GRAPH_FAILED);
}

TEST(DenseBincountTiling, deterministic_histogram_rejects_insufficient_ub)
{
    DenseBincountTilingCase insufficient;
    insufficient.inputShape = {4097};
    insufficient.weightsShape = {4097};
    insufficient.size = 4097;
    insufficient.deterministic = 1;
    insufficient.ubSize = 128U * 1024U + 32U * 1024U - 1U;
    EXPECT_EQ(RunTiling(insufficient).status, ge::GRAPH_FAILED);
}

TEST(DenseBincountTiling, bounded_weighted_work_uses_ordered_reduction)
{
    DenseBincountTilingCase precise;
    precise.inputShape = {512};
    precise.weightsShape = {512};
    precise.size = 4;

    auto result = RunTiling(precise);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.key, ONE_D_ORDERED_TILING_KEY);
    EXPECT_EQ(result.privateHistElems, 0U);

    precise.deterministic = 1;
    result = RunTiling(precise);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.key, ONE_D_ORDERED_TILING_KEY);
    precise.deterministic = 0;

    precise.inputShape = {1024};
    precise.weightsShape = {1024};
    result = RunTiling(precise);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.key, ONE_D_ORDERED_TILING_KEY);

    precise.inputShape = {1025};
    precise.weightsShape = {1025};
    result = RunTiling(precise);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.key, SCH_MODE_1D_COUNT_WEIGHT);

    precise.inputShape = {512};
    precise.weightsShape = {512};
    precise.size = 8;
    result = RunTiling(precise);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.key, ONE_D_ORDERED_TILING_KEY);

    precise.size = 9;
    result = RunTiling(precise);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.key, SCH_MODE_1D_COUNT_WEIGHT);

    precise.inputShape = {1048576};
    precise.weightsShape = {1048576};
    precise.size = 1;
    result = RunTiling(precise);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.key, SCH_MODE_1D_COUNT_WEIGHT);

    precise.inputShape = {2, 4};
    precise.weightsShape = {2, 4};
    precise.size = 4;
    result = RunTiling(precise);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.key, ND_ORDERED_TILING_KEY);

    precise.inputShape = {1};
    precise.weightsShape = {1};
    precise.size = 64;
    result = RunTiling(precise);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.key, ONE_D_ORDERED_TILING_KEY);

    precise.size = 65;
    result = RunTiling(precise);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.key, SCH_MODE_1D_COUNT_WEIGHT);
}

TEST(DenseBincountTiling, binary_row_private_histogram_is_selected_when_full_output_is_too_expensive)
{
    DenseBincountTilingCase rowPrivate;
    rowPrivate.inputShape = {4, 128};
    rowPrivate.weightsShape = {4, 128};
    rowPrivate.size = 1024;
    rowPrivate.binaryOutput = true;

    const auto result = RunTiling(rowPrivate);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.key, SCH_MODE_ND_BINARY_WEIGHT);
    EXPECT_EQ(result.privateHistElems, 1024U);
    EXPECT_EQ(result.privateHistMode, 2U);
}

TEST(DenseBincountTiling, large_1d_output_uses_bounded_tiled_histogram)
{
    DenseBincountTilingCase tiled;
    tiled.inputShape = {1048576};
    tiled.weightsShape = {1048576};
    tiled.inputDtype = ge::DT_INT64;
    tiled.sizeDtype = ge::DT_INT64;
    tiled.size = 65536;

    auto result = RunTiling(tiled);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.key, SCH_MODE_1D_COUNT_WEIGHT);
    EXPECT_EQ(result.privateHistMode, 3U);
    EXPECT_GT(result.privateHistElems, 0U);
    EXPECT_LT(result.privateHistElems, static_cast<uint32_t>(tiled.size));

    tiled.binaryOutput = true;
    result = RunTiling(tiled);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.key, SCH_MODE_1D_BINARY_WEIGHT);
    EXPECT_EQ(result.privateHistMode, 3U);
}

TEST(DenseBincountTiling, rejects_invalid_contracts_and_platform_limits)
{
    DenseBincountTilingCase invalidRank;
    invalidRank.inputShape = {1, 2, 4};
    invalidRank.weightsShape = {1, 2, 4};
    EXPECT_EQ(RunTiling(invalidRank).status, ge::GRAPH_FAILED);

    DenseBincountTilingCase invalidInputDtype;
    invalidInputDtype.inputDtype = ge::DT_FLOAT;
    invalidInputDtype.sizeDtype = ge::DT_FLOAT;
    EXPECT_EQ(RunTiling(invalidInputDtype).status, ge::GRAPH_FAILED);

    DenseBincountTilingCase invalidSizeShape;
    invalidSizeShape.sizeShape = {1, 1};
    EXPECT_EQ(RunTiling(invalidSizeShape).status, ge::GRAPH_FAILED);

    invalidSizeShape.sizeShape = {2};
    EXPECT_EQ(RunTiling(invalidSizeShape).status, ge::GRAPH_FAILED);

    invalidSizeShape.sizeShape = {0};
    EXPECT_EQ(RunTiling(invalidSizeShape).status, ge::GRAPH_FAILED);

    DenseBincountTilingCase invalidWeightsDtype;
    invalidWeightsDtype.weightsDtype = ge::DT_FLOAT16;
    EXPECT_EQ(RunTiling(invalidWeightsDtype).status, ge::GRAPH_FAILED);

    DenseBincountTilingCase invalidOutputDtype;
    invalidOutputDtype.outputDtype = ge::DT_FLOAT16;
    EXPECT_EQ(RunTiling(invalidOutputDtype).status, ge::GRAPH_FAILED);

    DenseBincountTilingCase invalidWeightsRank;
    invalidWeightsRank.inputShape = {8};
    invalidWeightsRank.weightsShape = {1, 1, 8};
    EXPECT_EQ(RunTiling(invalidWeightsRank).status, ge::GRAPH_FAILED);

    DenseBincountTilingCase mismatchedWeights;
    mismatchedWeights.weightsShape = {7};
    EXPECT_EQ(RunTiling(mismatchedWeights).status, ge::GRAPH_FAILED);

    DenseBincountTilingCase negativeSize;
    negativeSize.size = -1;
    EXPECT_EQ(RunTiling(negativeSize).status, ge::GRAPH_FAILED);

    DenseBincountTilingCase insufficientUb;
    insufficientUb.ubSize = 128U * 1024U;
    EXPECT_EQ(RunTiling(insufficientUb).status, ge::GRAPH_FAILED);

    DenseBincountTilingCase noCores;
    noCores.coreNum = 0U;
    EXPECT_EQ(RunTiling(noCores).status, ge::GRAPH_FAILED);

    DenseBincountTilingCase dynamicShape;
    dynamicShape.inputShape = {-1};
    dynamicShape.weightsShape = {-1};
    EXPECT_EQ(RunTiling(dynamicShape).status, ge::GRAPH_FAILED);
}
