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
 * \file transpose_tiling_arch35.h
 * \brief Transpose Tiling 调度器定义
 *
 * 本文件定义 TransposeNddmaTiling：核心 Tiling 计算类（参数解析/校验与策略调度）。
 *
 * 各策略的常量与中间结构定义在策略自身文件内；TilingData 结构体由
 * op_kernel/arch35/transpose_tiling_data.h 以纯 struct 形式定义，Host 与 Kernel 共用，
 * 保证两侧内存布局完全一致。
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSPOSE_TILING_ARCH35_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSPOSE_TILING_ARCH35_H

#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <cstdint>
#include <vector>
#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "register/op_impl_registry.h"
#include "transpose_tiling_base.h"
#include "conversion/transpose/op_kernel/arch35/transpose_tiling_data.h"

namespace optiling {
using namespace Transpose;

/* 前置声明：策略注册上下文参数（完整定义见 transpose_tiling_strategy.h，由算子侧注册文件引入） */
struct TransposePlatformInfo;
struct TransposeOpParam;

ge::graphStatus TilingPrepareTransposeForAscendC(gert::TilingParseContext* context);

class TransposeNddmaTiling {
public:
    explicit TransposeNddmaTiling(gert::TilingContext* context) : tilingContext_(context) {};
    ge::graphStatus Init(const int64_t& coreNum, const int64_t& ubSize);
    ge::graphStatus RunTranposelTiling();
    // 关联转置：Tiling 计算与 TilingData 写入全部由 Transpose 侧完成，调用方无需持有 TilingData，
    // 直接把自己的编译信息（const 引用）传进来即可
    ge::graphStatus TilingForRelatedTranspose(gert::TilingContext* context, const TransposeCompilerInfo& compilerInfo,
                                              ShapeInfo& opInput);

private:
    template <typename T>
    bool GetPerm(const gert::Tensor* permTensor);
    void SetIsLastAxisTranspose();
    ge::graphStatus GetShapeInfo();
    ge::graphStatus CheckShapeDims();
    ge::graphStatus CheckPermValue();
    ge::graphStatus CheckShapeInfo();
    ge::graphStatus CheckB4ShapeInfo();
    ge::graphStatus CheckReducedShapeInfo();
    ge::graphStatus AdjustShapeForB4();

    // 公共前置：把完整 ShapeInfo 裁剪为策略侧精简结构，打包为只读算子参数
    TransposeOpParam PrepareStrategyContext();
    // 编译期注册类调度（策略集合与顺序见 .cpp 中的 TransposeStrategyList，顺序即优先级）
    ge::graphStatus DispatchStrategies(const TransposeOpParam& opParam);

private:
    gert::TilingContext* tilingContext_ = nullptr;

    ShapeInfo shapeInfo_;
    int64_t coreNum_ = 0;
    int64_t ubSize_ = 0;
    int64_t cacheLineSize_ = 0;
    int64_t ubBlockSize_ = 0;

    bool isRelatedTranspose_ = false;
    bool is4bit_ = false;
};
} // namespace optiling

#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSPOSE_TILING_ARCH35_H
