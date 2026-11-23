/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_host/arch35/transpose_tiling_arch35.h"
// 轴变换函数在 Host 动态库中不可见，直接编译真实实现以独立验证一至八维描述。
#include "../../../../op_host/arch35/transpose_tiling_base.cpp"

namespace {

using namespace Transpose;

std::vector<int64_t> Shape2Vec(const int64_t* values, int64_t dim)
{
    return std::vector<int64_t>(values, values + dim);
}

TEST(TransposeTilingAxis, RemoveSingletonAxes)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    // 删除单例轴后 perm 为 [1, 0, 2]，没有可合并的连续轴组。
    int64_t permValues[] = {2, 1, 0, 3, 4};
    gert::TilingContextPara::TensorDescription x({{2, 1, 3, 1, 4}, {2, 1, 3, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{5}, {5}}, ge::DT_INT64, ge::FORMAT_ND, true, &permValues);
    gert::TilingContextPara::TensorDescription y({{3, 1, 2, 1, 4}, {3, 1, 2, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {y}, &compileInfo);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    // 简化后 dim=3 → 10011 二/三维 24B 精简 TilingData
    ASSERT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(10011));
    const auto* data = reinterpret_cast<const optiling::TransposeSmallShape23DTilingData*>(tilingInfo.tilingData.get());
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(optiling::TransposeSmallShape23DTilingData));
    EXPECT_EQ(data->dim, 3);
    EXPECT_EQ(data->inputShape[0], 2);
    EXPECT_EQ(data->inputShape[1], 3);
    EXPECT_EQ(data->inputShape[2], 4);
    EXPECT_EQ(data->perm[0], 1);
    EXPECT_EQ(data->perm[1], 0);
    EXPECT_EQ(data->perm[2], 2);
}

TEST(TransposeTilingAxis, MergeContinuousAxes)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    // 输入没有单例轴；分别合并输入轴 [0, 1] 和 [2, 3]。
    int64_t permValues[] = {2, 3, 0, 1, 4};
    gert::TilingContextPara::TensorDescription x({{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{5}, {5}}, ge::DT_INT64, ge::FORMAT_ND, true, &permValues);
    gert::TilingContextPara::TensorDescription y({{4, 5, 2, 3, 6}, {4, 5, 2, 3, 6}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {y}, &compileInfo);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    // 简化后 dim=3 → 10011 二/三维 24B 精简 TilingData
    ASSERT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(10011));
    const auto* data = reinterpret_cast<const optiling::TransposeSmallShape23DTilingData*>(tilingInfo.tilingData.get());
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(optiling::TransposeSmallShape23DTilingData));
    EXPECT_EQ(data->dim, 3);
    EXPECT_EQ(data->inputShape[0], 6);
    EXPECT_EQ(data->inputShape[1], 20);
    EXPECT_EQ(data->inputShape[2], 6);
    EXPECT_EQ(data->perm[0], 1);
    EXPECT_EQ(data->perm[1], 0);
    EXPECT_EQ(data->perm[2], 2);
}

using Shape = std::vector<int64_t>;
constexpr int64_t UNUSED_AXIS_VALUE = -123456789;

struct AxisCase {
    std::string name;
    Shape input;
    Shape perm;
    Shape reducedInput;
    Shape reducedPerm;
    Shape reducedOutput;
};

int64_t Volume(const Shape& input)
{
    return std::accumulate(input.begin(), input.end(), int64_t{1}, std::multiplies<int64_t>());
}

optiling::ShapeInfo MakeShapeInfo(const Shape& input, const Shape& perm)
{
    optiling::ShapeInfo info;
    // 保持构造时的八个位置；无效尾槽填哨兵，检查算法是否只使用有效前缀。
    for (auto* values :
         {&info.inShape, &info.outShape, &info.perm, &info.reducedInShape, &info.reducedOutShape, &info.reducedPerm}) {
        std::fill(values->begin(), values->end(), UNUSED_AXIS_VALUE);
    }
    info.inShapeSize = info.outShapeSize = info.permSize = info.origDim = info.dim = input.size();
    info.totalVolumeActual = Volume(input);
    info.eleLenInBytes = 4;
    std::copy(input.begin(), input.end(), info.inShape.begin());
    std::copy(perm.begin(), perm.end(), info.perm.begin());
    for (size_t i = 0; i < input.size(); ++i) {
        info.outShape[i] = input[perm[i]];
    }
    return info;
}

void ExpectOriginalUnchanged(const optiling::ShapeInfo& actual, const optiling::ShapeInfo& original)
{
    EXPECT_THAT(actual.inShape, ::testing::ElementsAreArray(original.inShape));
    EXPECT_THAT(actual.outShape, ::testing::ElementsAreArray(original.outShape));
    EXPECT_THAT(actual.perm, ::testing::ElementsAreArray(original.perm));
    EXPECT_EQ(actual.inShapeSize, original.inShapeSize);
    EXPECT_EQ(actual.outShapeSize, original.outShapeSize);
    EXPECT_EQ(actual.permSize, original.permSize);
    EXPECT_EQ(actual.origDim, original.origDim);
    EXPECT_EQ(actual.totalVolumeActual, original.totalVolumeActual);
    EXPECT_EQ(actual.eleLenInBytes, original.eleLenInBytes);
    EXPECT_EQ(actual.isLastAxisTranspose, original.isLastAxisTranspose);
    for (const auto* values : {&actual.reducedInShape, &actual.reducedOutShape, &actual.reducedPerm}) {
        ASSERT_EQ(values->size(), TP_MAX_AXIS_NUM);
        EXPECT_THAT(Shape2Vec(values->data() + original.origDim, TP_MAX_AXIS_NUM - original.origDim),
                    ::testing::Each(::testing::Eq(UNUSED_AXIS_VALUE)));
    }
}

void ExpectReduced(const optiling::ShapeInfo& actual, const AxisCase& expected, const optiling::ShapeInfo& original)
{
    ASSERT_NO_FATAL_FAILURE(ExpectOriginalUnchanged(actual, original));
    ASSERT_EQ(actual.dim, static_cast<int64_t>(expected.reducedInput.size()));
    EXPECT_THAT(Shape2Vec(actual.reducedInShape.data(), actual.dim),
                ::testing::ElementsAreArray(expected.reducedInput));
    EXPECT_THAT(Shape2Vec(actual.reducedOutShape.data(), actual.dim),
                ::testing::ElementsAreArray(expected.reducedOutput));
    EXPECT_THAT(Shape2Vec(actual.reducedPerm.data(), actual.dim), ::testing::ElementsAreArray(expected.reducedPerm));
    EXPECT_EQ(Volume(Shape2Vec(actual.reducedInShape.data(), actual.dim)), original.totalVolumeActual);
}

std::vector<AxisCase> RemoveCases()
{
    std::vector<AxisCase> cases = {
        {"OneSingletonAxis", {1}, {0}, {1}, {0}, {1}},
        {"OneNonSingletonAxis", {7}, {0}, {7}, {0}, {7}},
        {"LargeSingleAxis", {9007199254740993LL}, {0}, {9007199254740993LL}, {0}, {9007199254740993LL}},
        {"SingletonsAroundLargeAxis",
         {1, 9007199254740993LL, 1, 2},
         {3, 0, 1, 2},
         {9007199254740993LL, 2},
         {1, 0},
         {2, 9007199254740993LL}},
        {"AllSingletons", {1, 1, 1, 1}, {3, 1, 0, 2}, {1}, {0}, {1}},
        {"EightSingletons", {1, 1, 1, 1, 1, 1, 1, 1}, {6, 3, 1, 7, 0, 5, 2, 4}, {1}, {0}, {1}},
        {"NoSingletons", {2, 3, 4}, {2, 0, 1}, {2, 3, 4}, {2, 0, 1}, {4, 2, 3}},
        {"SingletonAtHead", {1, 2, 3, 4}, {3, 0, 2, 1}, {2, 3, 4}, {2, 1, 0}, {4, 3, 2}},
        {"SingletonAtTail", {2, 3, 4, 1}, {3, 1, 2, 0}, {2, 3, 4}, {1, 2, 0}, {3, 4, 2}},
        {"SingletonInMiddle", {2, 1, 3}, {2, 0, 1}, {2, 3}, {1, 0}, {3, 2}},
        {"MultipleMiddleSingletons", {2, 1, 1, 3, 4}, {2, 4, 0, 1, 3}, {2, 3, 4}, {2, 0, 1}, {4, 2, 3}},
        {"SingletonsAtBothEnds", {1, 2, 3, 1}, {3, 2, 0, 1}, {2, 3}, {1, 0}, {3, 2}},
        {"EightAxesWithSingletons",
         {1, 2, 1, 3, 1, 4, 1, 5},
         {6, 7, 4, 5, 2, 3, 0, 1},
         {2, 3, 4, 5},
         {3, 2, 1, 0},
         {5, 4, 3, 2}},
        {"EightAxesWithoutSingletons",
         {2, 3, 4, 5, 6, 7, 8, 9},
         {7, 5, 3, 1, 6, 4, 2, 0},
         {2, 3, 4, 5, 6, 7, 8, 9},
         {7, 5, 3, 1, 6, 4, 2, 0},
         {9, 7, 5, 3, 8, 6, 4, 2}}};
    // 覆盖八个位置中任意一个轴保留的情形，perm 始终是合法的完整排列。
    for (int64_t axis = 0; axis < TP_MAX_AXIS_NUM; ++axis) {
        Shape input(TP_MAX_AXIS_NUM, 1);
        input[axis] = 7;
        cases.push_back(
            {"OnlyAxis" + std::to_string(axis) + "Remains", input, {4, 0, 7, 2, 5, 1, 6, 3}, {7}, {0}, {7}});
    }
    return cases;
}

std::string CaseName(const ::testing::TestParamInfo<AxisCase>& info) { return info.param.name; }

class TransposeRemoveAxis : public ::testing::TestWithParam<AxisCase> {};

TEST_P(TransposeRemoveAxis, MatchesExpectedReducedDescription)
{
    const auto& testCase = GetParam();
    auto info = MakeShapeInfo(testCase.input, testCase.perm);
    const auto original = info;
    optiling::RemoveAxisV2(info);
    ASSERT_NO_FATAL_FAILURE(ExpectReduced(info, testCase, original));
}

INSTANTIATE_TEST_SUITE_P(AxisCases, TransposeRemoveAxis, ::testing::ValuesIn(RemoveCases()), CaseName);

class TransposeMergeAxis : public ::testing::TestWithParam<AxisCase> {};

TEST_P(TransposeMergeAxis, MatchesExpectedReducedDescription)
{
    const auto& testCase = GetParam();
    auto info = MakeShapeInfo(testCase.input, testCase.perm);
    // 合轴单测模拟已删轴状态，保留八槽存储，不依赖 RemoveAxisV2 的结果。
    info.reducedInShape = info.inShape;
    info.reducedOutShape = info.outShape;
    info.reducedPerm = info.perm;
    const auto original = info;
    optiling::MergeAxisV2(info);
    ASSERT_NO_FATAL_FAILURE(ExpectReduced(info, testCase, original));
}

INSTANTIATE_TEST_SUITE_P(
    AxisCases, TransposeMergeAxis,
    ::testing::Values(
        AxisCase{"OneSingletonAxis", {1}, {0}, {1}, {0}, {1}}, AxisCase{"OneNonSingletonAxis", {7}, {0}, {7}, {0}, {7}},
        AxisCase{"IdentityAllMerged", {2, 3, 4, 5}, {0, 1, 2, 3}, {120}, {0}, {120}},
        AxisCase{"EightAxisIdentity", {2, 3, 2, 3, 2, 3, 2, 3}, {0, 1, 2, 3, 4, 5, 6, 7}, {1296}, {0}, {1296}},
        AxisCase{"ReversedNoMerge", {2, 3, 4, 5}, {3, 2, 1, 0}, {2, 3, 4, 5}, {3, 2, 1, 0}, {5, 4, 3, 2}},
        AxisCase{"EightAxesReversedNoMerge",
                 {2, 3, 4, 5, 6, 7, 8, 9},
                 {7, 6, 5, 4, 3, 2, 1, 0},
                 {2, 3, 4, 5, 6, 7, 8, 9},
                 {7, 6, 5, 4, 3, 2, 1, 0},
                 {9, 8, 7, 6, 5, 4, 3, 2}},
        AxisCase{
            "NonMonotonicNoMerge", {2, 3, 4, 5, 6}, {2, 0, 4, 1, 3}, {2, 3, 4, 5, 6}, {2, 0, 4, 1, 3}, {4, 2, 6, 3, 5}},
        AxisCase{"GroupAtOutputStart", {2, 3, 4}, {1, 2, 0}, {2, 12}, {1, 0}, {12, 2}},
        AxisCase{"GroupAtOutputEnd", {2, 3, 4}, {2, 0, 1}, {6, 4}, {1, 0}, {4, 6}},
        AxisCase{"GroupAtOutputMiddle", {2, 3, 4, 5, 6}, {4, 1, 2, 0, 3}, {2, 12, 5, 6}, {3, 1, 0, 2}, {6, 12, 2, 5}},
        AxisCase{
            "MultipleReorderedGroups", {2, 3, 4, 5, 6, 7}, {2, 3, 0, 1, 4, 5}, {6, 20, 42}, {1, 0, 2}, {20, 6, 42}},
        AxisCase{"ReversedGroupOrder", {2, 3, 4, 5, 6, 7}, {4, 5, 2, 3, 0, 1}, {6, 20, 42}, {2, 1, 0}, {42, 20, 6}},
        AxisCase{"UnequalReorderedGroups",
                 {2, 3, 4, 5, 6, 7, 8},
                 {3, 4, 5, 0, 1, 6, 2},
                 {6, 4, 210, 8},
                 {2, 0, 3, 1},
                 {210, 6, 8, 4}},
        AxisCase{
            "EightAxes", {2, 3, 4, 5, 6, 7, 8, 9}, {6, 7, 2, 3, 4, 5, 0, 1}, {6, 840, 72}, {2, 1, 0}, {72, 840, 6}},
        AxisCase{"ProductBeyond32Bits", {2, 65537, 65537}, {1, 2, 0}, {2, 4295098369LL}, {1, 0}, {4295098369LL, 2}},
        // 大数仅计算描述；包括超过 double 精确整数范围及接近 int64 总字节上限的乘积。
        AxisCase{"ProductBeyondDoublePrecision",
                 {2, 134217729, 134217727},
                 {1, 2, 0},
                 {2, 18014398509481983LL},
                 {1, 0},
                 {18014398509481983LL, 2}},
        AxisCase{"ProductNearByteLimit",
                 {536870911, 4294967297LL},
                 {0, 1},
                 {2305843005455597567LL},
                 {0},
                 {2305843005455597567LL}}),
    CaseName);

class TransposeRemoveThenMerge : public ::testing::TestWithParam<AxisCase> {};

TEST_P(TransposeRemoveThenMerge, MatchesExpectedReducedDescription)
{
    const auto& testCase = GetParam();
    auto info = MakeShapeInfo(testCase.input, testCase.perm);
    const auto original = info;
    optiling::RemoveAxisV2(info);
    optiling::MergeAxisV2(info);
    ASSERT_NO_FATAL_FAILURE(ExpectReduced(info, testCase, original));
}

INSTANTIATE_TEST_SUITE_P(
    AxisCases, TransposeRemoveThenMerge,
    ::testing::Values(AxisCase{"DeleteWithoutMerge", {2, 1, 3, 1, 4}, {2, 1, 0, 3, 4}, {2, 3, 4}, {1, 0, 2}, {3, 2, 4}},
                      AxisCase{"DeleteThenMergeAll", {2, 1, 3, 1, 4}, {0, 1, 2, 3, 4}, {24}, {0}, {24}},
                      AxisCase{"DeleteInsideGroups", {2, 1, 3, 4, 1, 5}, {3, 4, 5, 0, 1, 2}, {6, 20}, {1, 0}, {20, 6}},
                      AxisCase{"EightAxesWithReorderedGroups",
                               {1, 2, 3, 1, 4, 5, 1, 7},
                               {7, 6, 0, 4, 5, 3, 1, 2},
                               {6, 20, 7},
                               {2, 1, 0},
                               {7, 20, 6}}),
    CaseName);

Shape TransposedInputOffsets(const Shape& input, const Shape& perm)
{
    Shape offsets(Volume(input));
    // 逐个输出坐标映射回输入坐标，再计算输入线性偏移，不复刻删轴或合轴算法。
    for (int64_t outputOffset = 0; outputOffset < static_cast<int64_t>(offsets.size()); ++outputOffset) {
        int64_t remaining = outputOffset;
        Shape inputCoords(input.size(), 0);
        for (int64_t i = static_cast<int64_t>(perm.size()) - 1; i >= 0; --i) {
            inputCoords[perm[i]] = remaining % input[perm[i]];
            remaining /= input[perm[i]];
        }
        int64_t inputOffset = 0;
        for (size_t i = 0; i < input.size(); ++i) {
            inputOffset = inputOffset * input[i] + inputCoords[i];
        }
        offsets[outputOffset] = inputOffset;
    }
    return offsets;
}

void ExpectEquivalentTranspose(const optiling::ShapeInfo& actual, const optiling::ShapeInfo& original,
                               const Shape& originalOffsets)
{
    ASSERT_NO_FATAL_FAILURE(ExpectOriginalUnchanged(actual, original));
    ASSERT_GE(actual.dim, 1);
    ASSERT_LE(actual.dim, original.origDim);
    const auto input = Shape2Vec(actual.reducedInShape.data(), actual.dim);
    const auto perm = Shape2Vec(actual.reducedPerm.data(), actual.dim);
    ASSERT_THAT(input, ::testing::Each(::testing::Gt(0)));
    Shape sortedPerm = perm;
    std::sort(sortedPerm.begin(), sortedPerm.end());
    Shape axes(actual.dim);
    std::iota(axes.begin(), axes.end(), int64_t{0});
    ASSERT_THAT(sortedPerm, ::testing::ElementsAreArray(axes));
    Shape output(actual.dim);
    for (int64_t i = 0; i < actual.dim; ++i) {
        output[i] = input[perm[i]];
    }
    EXPECT_THAT(Shape2Vec(actual.reducedOutShape.data(), actual.dim), ::testing::ElementsAreArray(output));
    ASSERT_EQ(Volume(input), original.totalVolumeActual);
    EXPECT_THAT(TransposedInputOffsets(input, perm), ::testing::ElementsAreArray(originalOffsets));
}

void VerifyBothAlgorithms(const Shape& input, const Shape& perm)
{
    SCOPED_TRACE("input=" + ::testing::PrintToString(input) + ", perm=" + ::testing::PrintToString(perm));
    auto info = MakeShapeInfo(input, perm);
    const auto original = info;
    const auto originalOffsets = TransposedInputOffsets(input, perm);
    optiling::RemoveAxisV2(info);
    const int64_t nonSingletonAxes = std::count_if(input.begin(), input.end(),
                                                   [](int64_t value) { return value != 1; });
    ASSERT_EQ(info.dim, std::max(int64_t{1}, nonSingletonAxes));
    ASSERT_NO_FATAL_FAILURE(ExpectEquivalentTranspose(info, original, originalOffsets));
    if (info.dim > 1) {
        EXPECT_THAT(Shape2Vec(info.reducedInShape.data(), info.dim), ::testing::Each(::testing::Gt(1)));
    }
    optiling::MergeAxisV2(info);
    ASSERT_NO_FATAL_FAILURE(ExpectEquivalentTranspose(info, original, originalOffsets));
    // 除了语义相同，还要求完成简化：合轴结果中不能继续存在相邻递增轴。
    for (int64_t i = 1; i < info.dim; ++i) {
        EXPECT_NE(info.reducedPerm[i], info.reducedPerm[i - 1] + 1);
    }
}

TEST(TransposeAxisSemantics, AllPermutationsAndSingletonPatternsUpToFiveAxes)
{
    int64_t caseCount = 0;
    for (int64_t rank = 1; rank <= 5; ++rank) {
        for (int64_t singletonMask = 0; singletonMask < (int64_t{1} << rank); ++singletonMask) {
            Shape input(rank);
            Shape perm(rank);
            for (int64_t i = 0; i < rank; ++i) {
                input[i] = (singletonMask & (int64_t{1} << i)) ? 1 : 2 + i % 2;
            }
            std::iota(perm.begin(), perm.end(), int64_t{0});
            do {
                ASSERT_NO_FATAL_FAILURE(VerifyBothAlgorithms(input, perm));
                ++caseCount;
            } while (std::next_permutation(perm.begin(), perm.end()));
        }
    }
    EXPECT_EQ(caseCount, 4282);
}

TEST(TransposeAxisSemantics, ReproducibleSixToEightAxisSamples)
{
    std::mt19937 random(3510);
    for (int64_t rank = 6; rank <= TP_MAX_AXIS_NUM; ++rank) {
        for (int64_t sample = 0; sample < 64; ++sample) {
            Shape input(rank);
            Shape perm(rank);
            for (auto& value : input) {
                value = 1 + random() % 3;
            }
            std::iota(perm.begin(), perm.end(), int64_t{0});
            std::shuffle(perm.begin(), perm.end(), random);
            ASSERT_NO_FATAL_FAILURE(VerifyBothAlgorithms(input, perm));
        }
    }
}

} // namespace
