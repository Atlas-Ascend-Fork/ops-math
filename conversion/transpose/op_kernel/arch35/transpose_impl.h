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
 * 支持的 TilingKey（见 transpose_tiling_key.h）：10000-10005、10014、10011 与 10012，其中
 * 10001/10011/10012(SIMT) 仅在支持 SIMT 的架构上编译；均已迁移为普通/关联入口统一的精简
 * TilingData：10000/10002/10003/10004/10005/10014/10001/10011/10012。
 */

#ifndef TRANSPOSE_IMPL_H
#define TRANSPOSE_IMPL_H

#include "transpose_base.h"
#include "transpose_tensor_move.h"
#include "transpose_nddma_cut_1.h"
#include "transpose_nddma_cut_2.h"
#ifdef __CCE_AICORE_SUPPORT_SIMT__
#include "transpose_small_shape.h"
#include "transpose_small_shape_23d.h"
#include "transpose_small_shape_2d.h"
#endif
#include "transpose_nddma_big_dim.h"
#include "transpose_n_last.h"
#include "transpose_n_last_small.h"
#include "transpose_tiling_key.h"

namespace Transpose {
/**
 * @brief 依赖 Transpose 的算子 Kernel 主体：解析 TilingData 并按 TilingKey 分发执行
 *
 * @tparam T      数据类型（由 DTYPE_X 实例化，策略类均为按字节搬运，不感知具体数值类型）
 * @param x       输入张量 GM 地址
 * @param y       输出张量 GM 地址
 * @param tiling  Tiling 数据 GM 地址（10000/10002/10003/10004/10005/10014/10001/10011 均为
 *                普通/关联入口统一的精简结构）
 */
template <typename T>
__aicore__ inline void TransposeImpl(GM_ADDR x, GM_ADDR y, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY); // 纯 Vector 核任务
    REGISTER_NONE_TILING;

    if (TILING_KEY_IS(TK_TENSOR_MOVE)) { // 10000: 普通/关联入口统一紧凑 TilingData
        GET_TILING_DATA_PTR_WITH_STRUCT(TransposeTensorMoveTilingData, data, tiling);
        AscendC::InitSocState();
        TransposeTensorMove<T> op;
        op.Init(x, y, data);
        op.Process();
        return;
#ifdef __CCE_AICORE_SUPPORT_SIMT__
    } else if (TILING_KEY_IS(TK_SMALL_SHAPE_2D)) { // 10012: 2D perm=[1,0] UB 4x4 SIMT
        GET_TILING_DATA_PTR_WITH_STRUCT(TransposeSmallShape2DTilingData, tilingDataPtr, tiling);
        AscendC::InitSocState();
        TransposeSmallShape2D<T> op;
        op.Init(x, y, tilingDataPtr);
        op.Process();
        return;
    } else if (TILING_KEY_IS(TK_SMALL_SHAPE_23D)) { // 10011: 普通/关联入口统一二/三维 24B 精简 TilingData
        GET_TILING_DATA_PTR_WITH_STRUCT(TransposeSmallShape23DTilingData, tilingDataPtr, tiling);
        AscendC::InitSocState();
        TransposeSmallShape23D<T> op;
        op.Init(x, y, tilingDataPtr);
        op.Process();
        return;
    } else if (TILING_KEY_IS(TK_SMALL_SHAPE)) { // 10001: 普通/关联入口统一 4-8D 精简 TilingData
        GET_TILING_DATA_PTR_WITH_STRUCT(TransposeSmallShape48DTilingData, tilingDataPtr, tiling);
        AscendC::InitSocState();
        TransposeSmallShape<T> op;
        op.Init(x, y, tilingDataPtr);
        op.Process();
        return;
#endif
    } else if (TILING_KEY_IS(TK_NDDMA_CUT_1)) { // 10002: 普通/关联入口统一紧凑 TilingData
        GET_TILING_DATA_WITH_STRUCT(TransposeNddmaCut1TilingData, tilingData, tiling);
        AscendC::InitSocState();
        TransposeNddmaCut1<T> op;
        op.Init(x, y, &tilingData);
        op.Process();
        return;
    } else if (TILING_KEY_IS(TK_NDDMA_CUT_2)) { // 10003: 普通/关联入口统一紧凑 TilingData
        GET_TILING_DATA_WITH_STRUCT(TransposeNddmaCut2TilingData, tilingData, tiling);
        AscendC::InitSocState();
        TransposeNddmaCut2<T> op;
        op.Init(x, y, &tilingData);
        op.Process();
        return;
    } else if (TILING_KEY_IS(TK_NDDMA_BIG_DIM)) { // 10005: 普通/关联入口统一紧凑 TilingData，静态 Tensor 无需 TPipe
        GET_TILING_DATA_WITH_STRUCT(TransposeNddmaBigDimTilingData, tilingData, tiling);
        AscendC::InitSocState();
        TransposeNddmaBigDim<T> op;
        op.Init(x, y, &tilingData);
        op.Process();
        return;
    } else if (TILING_KEY_IS(TK_N_LAST)) { // 10004: 普通/关联入口统一紧凑 TilingData
        GET_TILING_DATA_WITH_STRUCT(TransposeNLastTilingData, tilingData, tiling);
        AscendC::InitSocState();
        TransposeNLast<T> op;
        op.Init(x, y, &tilingData);
        op.Process();
        return;
    } else if (TILING_KEY_IS(TK_N_LAST_SMALL)) { // 10014: 普通/关联入口统一紧凑 TilingData
        GET_TILING_DATA_WITH_STRUCT(TransposeNLastSmallTilingData, tilingData, tiling);
        AscendC::InitSocState();
        TransposeNLastSmall op;
        op.Init(x, y, &tilingData);
        op.Process();
        return;
    }
}

} // namespace Transpose

#endif // TRANSPOSE_IMPL_H
