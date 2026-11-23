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
 * \file test_transpose_tiling_strategy_common.h
 * \brief 策略 UT 公共辅助（每策略一个 UT 文件共用）
 *
 * 只提供测试夹具与字段级断言辅助，不含任何测试用例：
 *   - 直接构造策略所需的最小 TilingContext / 平台信息 / 算子参数；
 *   - 注册 TilingData 的字段级断言（不使用序列化字符串）。
 */

#ifndef TEST_TRANSPOSE_TILING_STRATEGY_COMMON_H
#define TEST_TRANSPOSE_TILING_STRATEGY_COMMON_H

#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "platform/platform_info.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_host/arch35/transpose_tiling_strategy.h"

using namespace Transpose;

namespace transpose_strategy_ut {

/**
 * @brief 从 TilingInfo 取实际 TilingData（e2e 用例）
 *
 * 分别用 EXPECT_NE / EXPECT_EQ 校验 data 非空与 size 正确，便于失败时区分原因；
 * 任一不满足返回 nullptr，由调用点 ASSERT_NE 终止用例（避免解引用空指针）。
 */
template <typename T>
inline const T* GetTilingData(const TilingInfo& tilingInfo)
{
    EXPECT_NE(tilingInfo.tilingData, nullptr);
    EXPECT_EQ(tilingInfo.tilingDataSize, sizeof(T));
    if (tilingInfo.tilingData == nullptr || tilingInfo.tilingDataSize != sizeof(T)) {
        return nullptr;
    }
    return reinterpret_cast<const T*>(tilingInfo.tilingData.get());
}

/**
 * @brief 从 context 取已写入的实际 TilingData（策略直调用例）
 *
 * 先用 EXPECT_NE / EXPECT_EQ 校验 raw data 非空与码流实际写入大小，便于区分失败原因；
 * 任一不满足返回 nullptr，由调用点 ASSERT_NE 终止用例。
 */
template <typename T>
inline const T* GetTilingData(gert::TilingContext* context)
{
    auto* rawTilingData = context->GetRawTilingData();
    EXPECT_NE(rawTilingData, nullptr);
    if (rawTilingData == nullptr) {
        return nullptr;
    }
    EXPECT_EQ(rawTilingData->GetDataSize(), sizeof(T));
    if (rawTilingData->GetDataSize() != sizeof(T)) {
        return nullptr;
    }
    return context->GetTilingData<T>();
}

constexpr int64_t kCoreNum = 48;
constexpr int64_t kUbSize = 253952; // 248K（与存量 UT 默认档一致）
constexpr int64_t kCacheLineSize = 256;
constexpr int64_t kUbBlockSize = 32;
constexpr int64_t kDefaultThreshold = 4000000;
constexpr uint64_t kUint32Max = std::numeric_limits<uint32_t>::max();
constexpr int64_t kHugeDim = 5000000000LL; ///< > uint32 的单个轴长度

using ShapeInfo = optiling::StrategyShapeInfo;
using TilingStatus = optiling::TilingStatus;

/* ===================== 策略上下文构造 ===================== */

inline optiling::TransposePlatformInfo MakePlatformInfo(int64_t arch = static_cast<int64_t>(Ops::Base::DAV_3510),
                                                        bool isSupportSimt = true)
{
    optiling::TransposePlatformInfo platform;
    platform.coreNum = kCoreNum;
    platform.ubSize = kUbSize;
    platform.cacheLineSize = kCacheLineSize;
    platform.ubBlockSize = kUbBlockSize;
    platform.npuArch = static_cast<NpuArch>(arch);
    platform.isSupportSimt = isSupportSimt;
    return platform;
}

inline optiling::TransposeOpParam MakeOpParam(const ShapeInfo& si, bool isRelatedTranspose = false)
{
    optiling::TransposeOpParam opParam;
    opParam.shapeInfo = si;
    opParam.isRelatedTranspose = isRelatedTranspose;
    return opParam;
}

// 构造典型简化后 shape（主类前置产出形态）：
// 维度 / 输出 shape / 总元素数 / 尾轴转置标记均由输入 shape 与 perm 推导
inline ShapeInfo MakeShape(const std::vector<int64_t>& in, const std::vector<int64_t>& perm, int64_t eleLen)
{
    ShapeInfo si;
    const int64_t dim = static_cast<int64_t>(in.size());
    si.dim = dim;
    for (int64_t i = 0; i < static_cast<int64_t>(si.reducedInShape.size()); i++) {
        si.reducedInShape[i] = (i < static_cast<int64_t>(in.size())) ? in[i] : 0;
        si.reducedPerm[i] = (i < static_cast<int64_t>(perm.size())) ? perm[i] : 0;
    }
    // 输出 shape：out[i] = in[perm[i]]；perm 与 in 不匹配的非法用例按 0 处理
    for (int64_t i = 0; i < dim && i < static_cast<int64_t>(perm.size()); i++) {
        const int64_t axis = perm[i];
        si.reducedOutShape[i] = (axis >= 0 && axis < static_cast<int64_t>(in.size())) ? in[axis] : 0;
    }
    // 总元素数：输入 shape 各轴乘积
    int64_t totalVolume = 1;
    for (const int64_t d : in) {
        totalVolume *= d;
    }
    si.totalVolumeActual = totalVolume;
    // 尾轴是否参与转置：perm[dim-1] != dim-1
    si.isLastAxisTranspose = (dim > 0 && dim <= static_cast<int64_t>(perm.size())) && (perm[dim - 1] != dim - 1);
    si.eleLenInBytes = eleLen;
    return si;
}

inline optiling::TransposeCompilerInfo MakeCompilerInfo(int64_t coreNum = kCoreNum, int64_t ubSize = kUbSize)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = coreNum;
    compileInfo.ubSize = ubSize;
    return compileInfo;
}

inline int64_t AlignUp(int64_t value, int64_t align) { return (value + align - 1) / align * align; }

inline int64_t Product(const int64_t* shape, int64_t beg, int64_t end)
{
    int64_t product = 1;
    for (int64_t i = beg; i < end; ++i) {
        product *= shape[i];
    }
    return product;
}

// 策略直调用例的占位上下文：shapeInfo 由调用方注入，context 仅提供 TilingData/Workspace 载体。
inline gert::TilingContextPara MakeDummyPara(optiling::TransposeCompilerInfo& compileInfo)
{
    static int64_t permValue[1] = {0};
    gert::TilingContextPara::TensorDescription x({{1}, {1}}, ge::DT_UINT8, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &permValue);
    gert::TilingContextPara::TensorDescription out({{1}, {1}}, ge::DT_UINT8, ge::FORMAT_ND);
    return gert::TilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
}

} // namespace transpose_strategy_ut

#endif // TEST_TRANSPOSE_TILING_STRATEGY_COMMON_H
