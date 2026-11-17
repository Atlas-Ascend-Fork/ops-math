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

#include "tiling_case_executor.h"
#include "../../../../op_kernel/select_top_k_tiling_data.h"

namespace {
constexpr uint64_t A2_VECTOR_CORES = 40;
constexpr uint64_t A2_UB_BYTES = 196352;
constexpr int64_t OUTER_SIZE = 2;
constexpr int64_t AXIS_SIZE = 4096;
constexpr int64_t FUSED_AXIS_SIZE = 65;
constexpr int64_t FUSED_INNER_SIZE = 3;
constexpr int64_t FUSED_OUTPUT_K = 13;
constexpr int64_t RANGE_AXIS_SIZE = 16;
constexpr int64_t VALID_INDEX_OFFSET = 7;
constexpr int64_t INDEX_FILL = -1;
constexpr float VALUE_FILL = -3.0f;
constexpr int64_t RESERVED_INDEX_ORDER = 2;

// 算子没有额外编译参数，但框架构造上下文时仍要求 CompileInfo 地址非空。
struct SelectTopKCompileInfo {};
SelectTopKCompileInfo g_compileInfo;

struct ContextOptions {
    ge::DataType dtype = ge::DT_FLOAT16;
    ge::DataType indicesType = ge::DT_INT64;
    bool largest = true;
    int64_t order = 1; // 公开约定：0 不排序、1 按值排序，独立于实现常量验证编码。
    int64_t offset = 0;
};

// 按真实属性顺序构造 A2 上下文，begin/end 默认缺省；输出形状由输入中间维替换得到。
gert::TilingContextPara MakeContext(const gert::StorageShape& inputShape, int64_t k, const ContextOptions& options = {})
{
    auto outputShape = inputShape;
    outputShape.MutableOriginShape().SetDim(1, k);
    outputShape.MutableStorageShape().SetDim(1, k);
    return gert::TilingContextPara(
        "SelectTopK", {{inputShape, options.dtype, ge::FORMAT_ND}},
        {{outputShape, options.dtype, ge::FORMAT_ND}, {outputShape, options.indicesType, ge::FORMAT_ND}},
        {
            {"k", Ops::Math::AnyValue::CreateFrom<int64_t>(k)},
            {"largest", Ops::Math::AnyValue::CreateFrom<bool>(options.largest)},
            {"order", Ops::Math::AnyValue::CreateFrom<int64_t>(options.order)},
            {"indices_dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(options.indicesType)},
            {"indices_offset", Ops::Math::AnyValue::CreateFrom<int64_t>(options.offset)},
            {"output_mode", Ops::Math::AnyValue::CreateFrom<int64_t>(SelectTopK::OUTPUT_BOTH)},
            {"value_fill", Ops::Math::AnyValue::CreateFrom<float>(VALUE_FILL)},
            {"index_fill", Ops::Math::AnyValue::CreateFrom<int64_t>(INDEX_FILL)},
        },
        {1, 0, 0}, {1, 1}, &g_compileInfo, A2_VECTOR_CORES, A2_UB_BYTES, sizeof(SelectTopKTilingData));
}
} // namespace

class SelectTopKTiling : public testing::Test {
protected:
    // 调用已注册的 Host 入口，先检查结果完整性，再读取字段，避免失败时访问无效 tiling 数据。
    void RunTiling(const gert::TilingContextPara& context)
    {
        ASSERT_TRUE(ExecuteTiling(context, info_));
        ASSERT_EQ(info_.tilingDataSize, sizeof(data_));
        ASSERT_NE(info_.tilingData, nullptr);
        std::memcpy(&data_, info_.tilingData.get(), sizeof(data_));
        EXPECT_GT(info_.blockNum, 0U);
        EXPECT_LE(info_.blockNum, A2_VECTOR_CORES);
        EXPECT_EQ(info_.blockNum, data_.base.cores);
        ASSERT_EQ(info_.workspaceSizes.size(), 1U);
        EXPECT_GE(info_.workspaceSizes[0], static_cast<int64_t>(data_.workspaceBytes));
    }

    TilingInfo info_;
    SelectTopKTilingData data_{};
};

// k=1 使用单极值模板，并保留求最小值及按值排序的语义。
TEST_F(SelectTopKTiling, SelectsTop1Template)
{
    ContextOptions options;
    options.largest = false;
    const auto context = MakeContext({{OUTER_SIZE, AXIS_SIZE, 1}, {OUTER_SIZE, AXIS_SIZE, 1}}, 1, options);
    ASSERT_NO_FATAL_FAILURE(RunTiling(context));
    EXPECT_EQ(info_.tilingKey, SELECT_TOP_K_TILING_TOP1);
    EXPECT_EQ(data_.base.rows, OUTER_SIZE);
    EXPECT_EQ(data_.base.k, 1U);
    EXPECT_EQ(data_.base.largest, 0U);
    EXPECT_EQ(data_.base.sorted, 1U);
}

// 无序全 K 的 BF16 输入使用直接复制模板，不能误按旧 order 编码进入排序。
TEST_F(SelectTopKTiling, UnsortedFullAxisUsesCopyAll)
{
    ContextOptions options;
    options.dtype = ge::DT_BF16;
    options.order = 0;
    const auto context = MakeContext({{OUTER_SIZE, AXIS_SIZE, 1}, {OUTER_SIZE, AXIS_SIZE, 1}}, AXIS_SIZE, options);
    ASSERT_NO_FATAL_FAILURE(RunTiling(context));
    EXPECT_EQ(info_.tilingKey, SELECT_TOP_K_TILING_COPY_ALL);
    EXPECT_EQ(data_.base.sorted, 0U);
    EXPECT_EQ(data_.outputK, AXIS_SIZE);
    EXPECT_EQ(data_.flags & (SelectTopK::WRITE_VALUES | SelectTopK::WRITE_INDICES),
              SelectTopK::WRITE_VALUES | SelectTopK::WRITE_INDICES);
}

// FP32 非末轴视图直接推导输入/输出跨度，工作区必须容纳融合输入的完整搬运。
TEST_F(SelectTopKTiling, DerivesFusedLayoutFromInputShape)
{
    ContextOptions options;
    options.dtype = ge::DT_FLOAT;
    const auto context = MakeContext(
        {{OUTER_SIZE, FUSED_AXIS_SIZE, FUSED_INNER_SIZE}, {OUTER_SIZE, FUSED_AXIS_SIZE, FUSED_INNER_SIZE}},
        FUSED_OUTPUT_K, options);
    ASSERT_NO_FATAL_FAILURE(RunTiling(context));
    EXPECT_EQ(data_.base.rows, OUTER_SIZE * FUSED_INNER_SIZE);
    EXPECT_EQ(data_.base.axis, FUSED_AXIS_SIZE);
    EXPECT_EQ(data_.inputInner, FUSED_INNER_SIZE);
    EXPECT_EQ(data_.outputInner, FUSED_INNER_SIZE);
    EXPECT_GT(data_.inputBlock, 0U);
    EXPECT_GT(data_.outputBlock, 0U);
    constexpr uint64_t INPUT_BYTES = OUTER_SIZE * FUSED_AXIS_SIZE * FUSED_INNER_SIZE * sizeof(float);
    EXPECT_GE(data_.workspaceBytes, data_.inputWorkspaceOffset + INPUT_BYTES);
}

// 广播 begin 与逐行 end 进入区间模板；有效索引偏移和填充值分别保存，填充索引不加偏移。
TEST_F(SelectTopKTiling, RangeBoundsKeepOffsetAndFillSeparate)
{
    ContextOptions options;
    options.offset = VALID_INDEX_OFFSET;
    constexpr int64_t PADDED_K = RANGE_AXIS_SIZE + 1;
    auto context = MakeContext({{OUTER_SIZE, RANGE_AXIS_SIZE, 1}, {OUTER_SIZE, RANGE_AXIS_SIZE, 1}}, PADDED_K, options);
    context.inputTensorDesc_.push_back({{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND});
    context.inputTensorDesc_.push_back({{{OUTER_SIZE}, {OUTER_SIZE}}, ge::DT_INT32, ge::FORMAT_ND});
    context.inputInstanceNum_ = {1, 1, 1};
    ASSERT_NO_FATAL_FAILURE(RunTiling(context));
    EXPECT_EQ(info_.tilingKey, SELECT_TOP_K_TILING_RANGE_MERGE);
    constexpr uint32_t EXPECTED_BOUNDS = SelectTopK::HAS_BEGIN | SelectTopK::HAS_END | SelectTopK::BROADCAST_BEGIN |
                                         SelectTopK::RANGE;
    EXPECT_EQ(data_.flags & EXPECTED_BOUNDS, EXPECTED_BOUNDS);
    EXPECT_EQ(data_.flags & SelectTopK::BROADCAST_END, 0U);
    EXPECT_EQ(data_.outputK, PADDED_K);
    EXPECT_EQ(data_.indicesOffset, VALID_INDEX_OFFSET);
    EXPECT_EQ(data_.indexFill, INDEX_FILL);
    EXPECT_FLOAT_EQ(data_.valueFill, VALUE_FILL);
}

// order=2 是尚未支持的索引排序，必须拒绝，不能降级为无序输出。
TEST_F(SelectTopKTiling, RejectsReservedIndexOrder)
{
    ContextOptions options;
    options.order = RESERVED_INDEX_ORDER;
    const auto context = MakeContext({{OUTER_SIZE, AXIS_SIZE, 1}, {OUTER_SIZE, AXIS_SIZE, 1}}, 1, options);
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

// INT32 偏移边界应按完整原轴校验：最大合法偏移可用，再增加一位即拒绝。
TEST_F(SelectTopKTiling, ChecksInt32IndexOffsetBoundary)
{
    ContextOptions options;
    options.indicesType = ge::DT_INT32;
    options.offset = std::numeric_limits<int32_t>::max() - (AXIS_SIZE - 1);
    const gert::StorageShape shape = {{OUTER_SIZE, AXIS_SIZE, 1}, {OUTER_SIZE, AXIS_SIZE, 1}};
    ASSERT_NO_FATAL_FAILURE(RunTiling(MakeContext(shape, 1, options)));
    EXPECT_EQ(data_.indicesOffset, options.offset);
    ++options.offset;
    ExecuteTestCase(MakeContext(shape, 1, options), ge::GRAPH_FAILED);
}
