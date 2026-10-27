/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
 * LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file maximum_grad_dag.h
 * \brief MaximumGrad 算子 DAG 计算图定义（atvoss 框架 - Broadcast 模式）
 *
 * 支持三种输出模式：
 *   OutputY1: 只输出 y1 (grad_x=True, grad_y=False)
 *   OutputY2: 只输出 y2 (grad_x=False, grad_y=True)
 *   TwoOutput: 输出 y1 和 y2 (grad_x=True, grad_y=True)
 *
 */

#ifndef MAXIMUM_GRAD_DAG_H
#define MAXIMUM_GRAD_DAG_H

#ifndef __CCE_AICORE__
#ifndef __aicore__
#define __aicore__
#endif
#endif

#include <type_traits>
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

namespace NsMaximumGrad {

using namespace Ops::Base;

// atvoss Vec::Compare 的 cmpMode 模板实参：4 = CMPMODE::GE（大于等于）
constexpr int CMP_MODE_GE = 4;

// ============================================================================
// DAG 模板：MaximumGradSingleDag（单输出，SwapBranch 区分 y1/y2）
//   y1 = grad * (x1 >= x2)        → SwapBranch=false: Select(mask, grad, 0)
//   y2 = grad * NOT(x1 >= x2)     → SwapBranch=true : Select(mask, 0, grad)
// ============================================================================
template <typename T, bool SwapBranch>
struct MaximumGradSingleDag {
    // 输入搬入（广播）
    using InputGrad = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using InputX1 = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;
    using InputX2 = Bind<Vec::CopyInBrc<T>, Placeholder::In2<T>>;

    // 比较：x1 >= x2 (GE)，y1/y2 共享
    using CompareMask = Bind<Vec::Compare<uint8_t, T, CMP_MODE_GE>, InputX1, InputX2>;

    // 常量零
    using ConstZero = MAKE_CONST(T, 0);

    // 选择：SwapBranch=false → mask=1 选 grad（y1）；true → mask=1 选 0（y2）
    using SelectVal = std::conditional_t<SwapBranch,
                                         Bind<Vec::Select<uint8_t, T, 0>, CompareMask, ConstZero, InputGrad>,
                                         Bind<Vec::Select<uint8_t, T, 0>, CompareMask, InputGrad, ConstZero>>;

    // 输出搬出：单输出
    using CopyOutVal = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, SelectVal>;

    // DAG 组装：单输出
    using Outputs = Elems<CopyOutVal>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T>
struct MaximumGradDagBoth {
    // 输入搬入（广播）
    using InputGrad = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using InputX1 = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;
    using InputX2 = Bind<Vec::CopyInBrc<T>, Placeholder::In2<T>>;

    // 比较：x1 >= x2 (GE)
    using CompareMask = Bind<Vec::Compare<uint8_t, T, CMP_MODE_GE>, InputX1, InputX2>;

    // 常量零（用于 Select）
    using ConstZero = MAKE_CONST(T, 0);

    // 选择 y1：mask=1 时选 grad，mask=0 时选 0；0表示掩码取自比较结果寄存器
    using SelectY1 = Bind<Vec::Select<uint8_t, T, 0>, CompareMask, InputGrad, ConstZero>;

    // 选择 y2：mask=1 时选 0，mask=0 时选 grad
    using SelectY2 = Bind<Vec::Select<uint8_t, T, 0>, CompareMask, ConstZero, InputGrad>;

    // 输出搬出：y1 和 y2
    using CopyOutY1 = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, SelectY1>;
    using CopyOutY2 = Bind<Vec::CopyOut<T>, Placeholder::Out1<T>, SelectY2>;

    // DAG 组装：双输出
    using Outputs = Elems<CopyOutY1, CopyOutY2>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

namespace DagTypes {

template <typename T>
using OutputY1 = typename MaximumGradSingleDag<T, false>::OpDag; // grad * (x1 >= x2)

template <typename T>
using OutputY2 = typename MaximumGradSingleDag<T, true>::OpDag; // grad * (x1 < x2)

template <typename T>
using TwoOutput = typename MaximumGradDagBoth<T>::OpDag;

} // namespace DagTypes

} // namespace NsMaximumGrad

#endif // MAXIMUM_GRAD_DAG_H
