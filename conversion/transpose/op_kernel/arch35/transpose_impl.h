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
 * \file transpose_impl.h
 * \brief 依赖 Transpose 的算子共用 Kernel 实现（DepthToSpace / SpaceToDepth / ConfusionTransposeD）
 *
 * 这些算子在 Host 侧通过 TransposeNddmaTiling::TilingForRelatedTranspose() 复用 Transpose 的
 * Tiling 计算（TilingData 也由 Transpose 侧直接写入 context），Kernel 侧复用 Transpose 的搬运
 * 策略类。TilingKey 分发、TilingData 解析、TPipe/TaskType 等全部逻辑都集中在此，各算子的
 * kernel 入口只需把 DTYPE_X 作为模板参数传入：
 *
 *   Transpose::TransposeImpl<DTYPE_X>(x, y, tiling);
 *
 * 支持的 TilingKey（见 transpose_tiling_key.h）：10000-10005，其中 10001(SIMT) 仅在
 * 支持 SIMT 的架构上编译。
 */

#ifndef TRANSPOSE_IMPL_H
#define TRANSPOSE_IMPL_H

#include "transpose_base.h"
#include "transpose_tensor_move.h"
#include "transpose_nddma_cut_1.h"
#include "transpose_nddma_cut_2.h"
#ifdef __CCE_AICORE_SUPPORT_SIMT__
#include "transpose_small_shape.h"
#endif
#include "transpose_nddma_big_dim.h"
#include "transpose_n_last.h"
#include "transpose_tiling_key.h"

namespace Transpose {
/**
 * @brief 依赖 Transpose 的算子 Kernel 主体：解析 TilingData 并按 TilingKey 分发执行
 *
 * @tparam T      数据类型（由 DTYPE_X 实例化，策略类均为按字节搬运，不感知具体数值类型）
 * @param x       输入张量 GM 地址
 * @param y       输出张量 GM 地址
 * @param tiling  Tiling 数据 GM 地址（内容为 TransposeOpTilingData）
 */
template <typename T>
__aicore__ inline void TransposeImpl(GM_ADDR x, GM_ADDR y, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY); // 纯 Vector 核任务
    REGISTER_TILING_DEFAULT(Transpose::TransposeOpTilingData);
    GET_TILING_DATA_WITH_STRUCT(Transpose::TransposeOpTilingData, tilingData, tiling);
    AscendC::TPipe pipe;

    if (TILING_KEY_IS(TK_TENSOR_MOVE)) {
        TransposeTensorMove<T> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
#ifdef __CCE_AICORE_SUPPORT_SIMT__
    } else if (TILING_KEY_IS(TK_SMALL_SHAPE)) {
        TransposeSmallShape<T> op;
        op.Init(x, y, &tilingData);
        op.Process();
#endif
    } else if (TILING_KEY_IS(TK_NDDMA_CUT_1)) {
        TransposeNddmaCut1<T> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TK_NDDMA_CUT_2)) {
        TransposeNddmaCut2<T> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TK_NDDMA_BIG_DIM)) {
        TransposeNddmaBigDim<T> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TK_N_LAST)) {
        TransposeNLast<T> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    }
}

} // namespace Transpose

#endif // TRANSPOSE_IMPL_H
