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
 * \file feeds_repeat_tiling.cpp
 * \brief
 */
#include <cstdint>
#include <string>
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "log/log.h"
#include "platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "feeds_repeat_tiling.h"

namespace optiling {

constexpr int64_t SIZE_OF_FP16 = 2;
constexpr int64_t SIZE_OF_FP32 = 4;
constexpr int64_t SIZE_OF_BF16 = 2;
constexpr int64_t SIZE_OF_INT32 = 4;
constexpr int64_t SIZE_OF_INT64 = 8;
constexpr int64_t ALIGN_NUM = 32;
constexpr int64_t ALIGN_FP16 = ALIGN_NUM / SIZE_OF_FP16;
constexpr int64_t ALIGN_FP32 = ALIGN_NUM / SIZE_OF_FP32;
constexpr int64_t ALIGN_BF16 = ALIGN_NUM / SIZE_OF_BF16;
constexpr int64_t ALIGN_INT32 = ALIGN_NUM / SIZE_OF_INT32;
constexpr int64_t ALIGN_INT64 = ALIGN_NUM / SIZE_OF_INT64;
constexpr uint32_t SPACE_USED_RATIO = 2;
constexpr uint32_t UB_BUFFER_USED = 128;
// Sum / InitOutput 等接口内部经 PopStackBuffer 从 UB 末端（arch3510 为 TMP_UB_OFFSET）弹出临时缓冲：
// Sum 需要 AlignUp(CeilDiv(length_aligned, VL), 32) 个 float；InitOutput 的突发上限为
// MAX_REPEAT_TIMES(255) * ONE_BLK_SIZE(32B) = 8160B。两者取大并向上取整到 8KB。
// 若不预留，TPipe 占满 UB 时 PopStackBuffer 的断言 endAddress > queEndAddress 失败，
// 设备侧表现为 VEC_ERROR / NO_OUTPUT / 静默错误数值。
constexpr uint32_t UB_STACK_RESERVE = 8192;
constexpr uint32_t DOUBLE_BUFFER = 2;
constexpr uint32_t TILING_KEY_100 = 100;
constexpr int64_t FEEDS_RANK_MIN = 1;
constexpr int64_t FEEDS_RANK_MAX = 8;

static void FeedsRepeatPrintParam(gert::TilingContext* context, FeedsRepeatTilingData& tiling)
{
    auto nodeName = context;
    OP_LOGD(nodeName, ">>>>>>>>>>>>>>> Start to print FeedsRepeat tiling data <<<<<<<<<<<<<<<<");
    OP_LOGD(nodeName, ">>> op [TilingData]: elem_row = %ld", tiling.get_elem_row());
    OP_LOGD(nodeName, ">>> op [TilingData]: elem_per_loop = %ld", tiling.get_elem_per_loop());
    OP_LOGD(nodeName, ">>> op [TilingData]: length = %u", tiling.get_length());
    OP_LOGD(nodeName, ">>> op [TilingData]: length_aligned = %u", tiling.get_length_aligned());
    OP_LOGD(nodeName, ">>> op [TilingData]: max_core_num = %ld", tiling.get_max_core_num());
    OP_LOGD(nodeName, ">>> op [TilingData]: core_per_group = %ld", tiling.get_core_per_group());
    OP_LOGD(nodeName, ">>> op [TilingData]: core_moreover = %ld", tiling.get_core_moreover());
    OP_LOGD(nodeName, ">>> op [TilingData]: empty_size = %ld", tiling.get_empty_size());
    OP_LOGD(nodeName, ">>> op [TilingData]: row_per_core = %ld", tiling.get_row_per_core());
    OP_LOGD(nodeName, ">>> op [TilingData]: row_left = %ld", tiling.get_row_left());
    OP_LOGD(nodeName, ">>>>>>>>>>>>>>> End print FeedsRepeat tiling data <<<<<<<<<<<<<<<<");
}

static ge::graphStatus SetTilingBatch(const gert::TilingContext* context, FeedsRepeatTilingData& tiling)
{
    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto max_core_num = ascendcPlatform.GetCoreNumAiv();
    const auto* feeds_input = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, feeds_input);
    const gert::Shape feeds_shape = feeds_input->GetStorageShape();
    // 秩校验必须先于 GetDim(0)：rank-0 输入下直接取第 0 维属于越界读
    OP_CHECK_IF(feeds_shape.GetDimNum() < static_cast<size_t>(FEEDS_RANK_MIN) ||
                    feeds_shape.GetDimNum() > static_cast<size_t>(FEEDS_RANK_MAX),
                OP_LOGE(context->GetNodeName(), "[FeedsRepeat] the rank of feeds must be in [%ld, %ld], but got %zu",
                        FEEDS_RANK_MIN, FEEDS_RANK_MAX, feeds_shape.GetDimNum()),
                return ge::GRAPH_FAILED);
    int64_t batch_num = feeds_shape.GetDim(0);
    int64_t group = 0;
    int64_t core_per_group = 0;
    int64_t core_moreover = 0;
    if (batch_num <= 0) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
            context->GetNodeName(), "feeds", std::to_string(batch_num).c_str(),
            ("the dim0 of feeds must be a positive number, but got " + std::to_string(batch_num)).c_str());
        return ge::GRAPH_FAILED;
    }
    if (batch_num < max_core_num) {
        group = batch_num;
        core_per_group = max_core_num / batch_num;
        core_moreover = max_core_num % batch_num;
    } else {
        group = max_core_num;
    }
    int64_t row_per_core = batch_num / group;
    int64_t row_left = batch_num % group;
    tiling.set_max_core_num(max_core_num);
    tiling.set_core_per_group(core_per_group);
    tiling.set_core_moreover(core_moreover);
    tiling.set_row_per_core(row_per_core);
    tiling.set_row_left(row_left);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetTilingLength(const gert::TilingContext* context, FeedsRepeatTilingData& tiling,
                                       uint32_t& tiling_key, uint32_t& length_space)
{
    const auto* feeds_input = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, feeds_input);
    const auto* times_input = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, times_input);
    const int64_t feeds_dim0 = feeds_input->GetStorageShape().GetDim(0);
    const int64_t times_numel = times_input->GetStorageShape().GetShapeSize();
    // 负维（未校验前不得转入无符号运算，否则负值回绕为巨值并使后续限长判断反转）
    if (feeds_dim0 < 0 || times_numel < 0) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context->GetNodeName(), "feeds and feeds_repeat_times",
            ("feeds dim0=" + std::to_string(feeds_dim0) +
             ", feeds_repeat_times element count=" + std::to_string(times_numel))
                .c_str(),
            "feeds dim0 and the element count of feeds_repeat_times must be non-negative");
        return ge::GRAPH_FAILED;
    }
    // 统一按元素总数判定（kernel 按元素线性读取 feeds_repeat_times，README 亦以元素总数定义合法域）
    if (feeds_dim0 != times_numel) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "feeds and feeds_repeat_times",
                                               ("feeds dim0=" + std::to_string(feeds_dim0) +
                                                ", feeds_repeat_times element count=" + std::to_string(times_numel))
                                                   .c_str(),
                                               "feeds_repeat_times element count must equal feeds dim0");
        return ge::GRAPH_FAILED;
    }
    const auto* times_desc = context->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, times_desc);
    auto dtype_length = times_desc->GetDataType();
    uint32_t elem_size = 0;
    if (dtype_length == ge::DT_INT32) {
        elem_size = SIZE_OF_INT32;
    } else if (dtype_length == ge::DT_INT64) {
        tiling_key = TILING_KEY_100; // feeds_repeat_times为int64时tiling_key百位为1
        elem_size = SIZE_OF_INT64;
    } else {
        OP_LOGE(context->GetNodeName(), "[FeedsRepeat] feeds_repeat_times dtype only support int32, int64");
        return ge::GRAPH_FAILED;
    }
    const uint64_t length = static_cast<uint64_t>(feeds_dim0);
    // 先按元素数上界拒绝，再对齐/相乘，避免 length_aligned * elem_size 在 uint64 中回绕后绕过限长校验
    const uint64_t max_length = 0xFFFFFFFFULL / static_cast<uint64_t>(elem_size);
    OP_CHECK_IF(length > max_length, OP_LOGE(context->GetNodeName(), "[FeedsRepeat] feeds_repeat_times is too large"),
                return ge::GRAPH_FAILED);
    // 按 float32 归约缓冲的 32B 对齐（ALIGN_FP32=8 个元素）统一对齐元素数，
    // 使 T2 缓冲、float 缓冲与 Sum 的 inner 同时满足 32B 对齐（int64 下按 4 元素对齐会出现 32B 非整除）
    const uint64_t length_aligned = (length + ALIGN_FP32 - 1) / ALIGN_FP32 * ALIGN_FP32;
    const uint64_t length_space64 = length_aligned * elem_size;
    OP_CHECK_IF(length_space64 > 0xFFFFFFFFULL,
                OP_LOGE(context->GetNodeName(), "[FeedsRepeat] feeds_repeat_times is too large"),
                return ge::GRAPH_FAILED);
    length_space = static_cast<uint32_t>(length_space64);
    tiling.set_length(static_cast<uint32_t>(length));
    tiling.set_length_aligned(static_cast<uint32_t>(length_aligned));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetTilingElemRow(const gert::TilingContext* context, FeedsRepeatTilingData& tiling,
                                        uint32_t& tiling_key, uint32_t& length_space)
{
    const auto* feeds_input = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, feeds_input);
    const auto* feeds_desc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, feeds_desc);
    const gert::Shape feeds_shape = feeds_input->GetStorageShape();
    const int64_t dim_num = feeds_shape.GetDimNum();
    // tiling 需独立拦截秩上下界，避免非法 rank 经 release binary 通路（无编译期 shape 检查）到达执行
    OP_CHECK_IF(dim_num < FEEDS_RANK_MIN || dim_num > FEEDS_RANK_MAX,
                OP_LOGE(context->GetNodeName(), "[FeedsRepeat] the rank of feeds must be in [%ld, %ld], but got %ld",
                        FEEDS_RANK_MIN, FEEDS_RANK_MAX, dim_num),
                return ge::GRAPH_FAILED);
    int64_t elem_row = 1;
    for (int64_t i = 1; i < dim_num; i++) {
        const int64_t dim = feeds_shape.GetDim(i);
        OP_CHECK_IF(dim < 0,
                    OP_LOGE(context->GetNodeName(),
                            "[FeedsRepeat] the dim %ld of feeds must be non-negative, but got %ld", i, dim),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(dim > 0 && elem_row > INT64_MAX / dim,
                    OP_LOGE(context->GetNodeName(), "[FeedsRepeat] the element count of one feeds row is too large"),
                    return ge::GRAPH_FAILED);
        elem_row *= dim;
    }
    OP_CHECK_IF(elem_row < 1,
                OP_LOGE(context->GetNodeName(), "[FeedsRepeat] the element count of one feeds row must be at least 1"),
                return ge::GRAPH_FAILED);
    auto dtype = feeds_desc->GetDataType();
    uint64_t max_ub_size;
    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, max_ub_size);
    int64_t elem_row_aligned = 0;
    int64_t elem_per_loop = 0;
    uint32_t align_data = 0;
    // 以 int64 计算可用 UB，避免 uint32 相乘回绕（feeds_repeat_times 缓冲 + float 归约缓冲 + 少量标量缓冲
    // + 为 Sum/InitOutput 的 PopStackBuffer 预留的末端栈空间）
    const int64_t ub_reserved = static_cast<int64_t>(length_space) * SPACE_USED_RATIO + UB_BUFFER_USED +
                                UB_STACK_RESERVE;
    const int64_t ub_avail = static_cast<int64_t>(max_ub_size) - ub_reserved;
    if (dtype == ge::DT_FLOAT) {
        tiling_key += 1; // feeds为fp32,tiling_key后两位为1
        align_data = ALIGN_FP32;
        elem_per_loop = ub_avail / SIZE_OF_FP32 / DOUBLE_BUFFER;
    } else if (dtype == ge::DT_FLOAT16) {
        tiling_key += 2; // feeds为fp16,tiling_key后两位为2
        align_data = ALIGN_FP16;
        elem_per_loop = ub_avail / SIZE_OF_FP16 / DOUBLE_BUFFER;
    } else if (dtype == ge::DT_BF16) {
        tiling_key += 3; // feeds为bf16,tiling_key后两位为3
        align_data = ALIGN_BF16;
        elem_per_loop = ub_avail / SIZE_OF_BF16 / DOUBLE_BUFFER;
    } else {
        OP_LOGE(context->GetNodeName(), "[FeedsRepeat] feeds dtype only support fp32, fp16, bf16 for now");
        return ge::GRAPH_FAILED;
    }
    // elem_row_aligned 的向上对齐含 elem_row + align_data - 1 有符号加法，elem_row 接近 INT64_MAX 时溢出，先行拒绝
    if (elem_row > INT64_MAX - static_cast<int64_t>(align_data)) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
            context->GetNodeName(), "feeds", std::to_string(elem_row).c_str(),
            ("the element count of one feeds row is too large, got " + std::to_string(elem_row)).c_str());
        return ge::GRAPH_FAILED;
    }
    elem_row_aligned = (elem_row + align_data - 1) / align_data * align_data;
    // feeds_repeat_times 与 float 归约缓冲占用 UB，另为 Sum/InitOutput 的 PopStackBuffer 预留 8KB 栈空间
    int64_t required_ub = static_cast<int64_t>(length_space) * SPACE_USED_RATIO + UB_BUFFER_USED + UB_STACK_RESERVE;
    int64_t available_ub = static_cast<int64_t>(max_ub_size);
    if (required_ub >= available_ub) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
            context->GetNodeName(), "feeds_repeat_times", std::to_string(required_ub).c_str(),
            ("feeds_repeat_times requires " + std::to_string(length_space) +
             " bytes plus fixed reserve, exceeding available UB " + std::to_string(available_ub) + " bytes")
                .c_str());
        return ge::GRAPH_FAILED;
    }
    elem_per_loop = elem_per_loop / align_data * align_data;
    elem_per_loop = elem_row_aligned > elem_per_loop ? elem_per_loop : elem_row_aligned;
    if (elem_per_loop <= 0) {
        OP_LOGE(
            context->GetNodeName(),
            "[FeedsRepeat] elem_per_loop is 0; feeds has an empty non-leading dim or element size over UB capacity");
        return ge::GRAPH_FAILED;
    }
    tiling.set_elem_row(elem_row);
    tiling.set_elem_per_loop(elem_per_loop);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4FeedsRepeat(gert::TilingContext* context)
{
    OP_LOGD(context, "Begin the tiling process for Arch35 architecture");
    OP_LOGD("FeedsRepeat tiling start");
    FeedsRepeatTilingData tiling;
    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto max_core_num = ascendcPlatform.GetCoreNumAiv();
    ge::graphStatus ret = SetTilingBatch(context, tiling);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    uint32_t tiling_key = 0;
    uint32_t length_space = 0;
    ret = SetTilingLength(context, tiling, tiling_key, length_space);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = SetTilingElemRow(context, tiling, tiling_key, length_space);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "attrs");
        return ge::GRAPH_FAILED;
    }
    const int64_t* empty_size_ptr = attrs->GetInt(0);
    if (empty_size_ptr == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "output_feeds_size");
        return ge::GRAPH_FAILED;
    }
    // output_feeds_size 直接决定输出第 0 维与 Kernel 的清零/截断上界，负值必须拒绝
    if (*empty_size_ptr < 0) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            context->GetNodeName(), "output_feeds_size",
            ("the attribute output_feeds_size must be non-negative, but got " + std::to_string(*empty_size_ptr))
                .c_str());
        return ge::GRAPH_FAILED;
    }
    tiling.set_empty_size(*empty_size_ptr);
    context->SetTilingKey(tiling_key);
    OP_LOGD(context, ">>> [FeedsRepeat] tilingKey: %u", tiling_key);
    context->SetBlockDim(max_core_num);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = sysWorkspaceSize;
    FeedsRepeatPrintParam(context, tiling);
    OP_LOGD("Tiling4FeedsRepeat tiling end");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4FeedsRepeat(gert::TilingParseContext* context)
{
    OP_LOGD("FeedsRepeat:", "TilingPrepareForFeedsRepeat start.");
    auto compileInfo = context->GetCompiledInfo<FeedsRepeatCompileInfo>();
    if (compileInfo == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "compile_info");
        return ge::GRAPH_FAILED;
    }
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "platform_info");
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->total_core_num = ascendcPlatform.GetCoreNumAiv();
    if (compileInfo->total_core_num <= 0) { // 0 negative number
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            context->GetNodeName(), "platform_info",
            ("AIV core number must be greater than 0, but got " + std::to_string(compileInfo->total_core_num)).c_str());
        return ge::GRAPH_FAILED;
    }

    uint64_t ub_size_platform = 0U; // 0, init
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size_platform);
    compileInfo->ub_size_platform = static_cast<int64_t>(ub_size_platform);
    if (compileInfo->ub_size_platform <= 0) { // 0
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            context->GetNodeName(), "platform_info",
            ("UB size must be greater than 0, but got " + std::to_string(compileInfo->ub_size_platform) + " bytes")
                .c_str());
        return ge::GRAPH_FAILED;
    }
    OP_LOGD("FeedsRepeat:", "TilingPrepareForFeedsRepeat end.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(FeedsRepeat).Tiling(Tiling4FeedsRepeat).TilingParse<FeedsRepeatCompileInfo>(TilingPrepare4FeedsRepeat);
} // namespace optiling
