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
 * \file stateless_bernoulli_tiling_arch35.h
 * \brief
 */
#ifndef STATELESS_BERNOULLI_TILING_ARCH35_H
#define STATELESS_BERNOULLI_TILING_ARCH35_H

#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "../../../random_common/op_host/arch35/random_tiling_arch35.h"

namespace optiling {

class StatelessBernoulliTiling : public RandomTilingArch35 {
public:
    explicit StatelessBernoulliTiling(gert::TilingContext* context) : RandomTilingArch35(context, BuildOpConfig()) {}

protected:
    ge::graphStatus DoSimtBlockTiling() override;
    // 场景化 tilingKey 设置（挂点须在步骤 6：基类步骤 5 无条件重置 key 为 100）
    ge::graphStatus UniqueProcess() override;

private:
    static OpTilingConfig BuildOpConfig();
    // 一趟扫描状态机: 从最内维向外逐维计算 prob 步长并做连续轴合并, 段表按内→外序暂存
    ge::graphStatus ScanAndCoalesce(const gert::Shape& outputDims, const gert::Shape& probShape, int64_t segSizes[],
                                    int64_t segStrides[], int32_t& segCnt, bool& hasBroadcastDim);
    // 张量语义: prob 广播 stride 计算、可广播校验与连续轴合并, 填充 probDims/probSizes/probStrides
    ge::graphStatus ComputeProbBroadcastTable(const gert::Shape& outputDims, const gert::Shape& probShape,
                                              bool& hasBroadcastDim);
    // 场景判定中转（DoSimtBlockTiling 写、UniqueProcess 读, 仅供 tilingKey 100/101/102 判定；
    // 语义已由 tilingKey 编码, 不落 tilingData、kernel 不消费）
    bool isScalarProb_ = false;
    bool hasBroadcastDim_ = false;
};
} // namespace optiling
#endif // STATELESS_BERNOULLI_TILING_ARCH35_H
