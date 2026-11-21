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
 * \file transpose_apt.cpp
 * \brief Transpose 算子 Kernel 层主入口文件
 *
 * 本文件是 Transpose 算子 Kernel 侧的唯一入口，负责根据 TilingKey 将执行流分发到
 * 11 种具体的转置策略类。所有策略共享同一套 Tiling 数据结构，但使用不同的硬件加速路径。
 *
 * 分发机制：
 *   - transpose() 为全局 Kernel 入口函数，通过 TILING_KEY_IS 宏判断当前 TilingKey；
 *   - 每个 TilingKey 对应一个 Process 函数（如 TransposeNddmaCut1Process）；
 *   - Process 函数内部自行解析 TilingData，并按需创建 TPipe（不需要缓冲队列的模板不创建）；
 *   - Process 函数直接用归一化后的 DTYPE_X_ 实例化对应策略模板。
 *
 * DTYPE_X 归一化：
 *   - 策略模板均为按字节搬运，只关心元素宽度，故按 sizeof(DTYPE_X) 归一化：
 *     - 4bit → int8_t (hif4/fp4，1 字节打包)
 *     - 1B → int8_t  （int8/uint8/bool/hifloat8/fp8，以及 1 字节打包的 HIF4/FP4）
 *     - 2B → int16_t （fp16/bf16/int16/uint16）
 *     - 4B → int32_t （fp32/int32/uint32）
 *     - 8B → int64_t （fp64/int64/uint64）
 *   - hif4/fp4 仅在 B4 模板中按 fp4 语义处理；其余模板一律按 int8_t（1 字节）处理。
 */

#include <type_traits>
#include "arch35/transpose_gather.h"
#include "arch35/transpose_n_last.h"
#include "arch35/transpose_nddma_big_dim.h"
#include "arch35/transpose_nddma_cut_1.h"
#include "arch35/transpose_nddma_cut_2.h"
#include "arch35/transpose_tensor_move.h"
#include "arch35/transpose_tiling_key.h"
#include "arch35/transpose_transto5hd_021.h"
#include "arch35/transpose_transto5hd_10.h"
#include "arch35/transpose_transto5hd_10_perf.h"
#ifdef __CCE_AICORE_SUPPORT_SIMT__
#include "arch35/transpose_b4.h"
#include "arch35/transpose_small_shape.h"
#endif

using namespace Transpose;

// 按输入元素的字节宽度归一化模板参数：策略模板只按字节搬运，不感知具体数值类型。
// HIF4/FP4 为 1 字节打包类型，归入 int8_t；其它非 1/2/4/8 字节类型同样按 int8_t 兜底。
static constexpr auto X_DTYPE_SIZE_ = sizeof(DTYPE_X);
using DTYPE_X_ = std::conditional_t<
    X_DTYPE_SIZE_ == sizeof(int8_t), int8_t,
    std::conditional_t<X_DTYPE_SIZE_ == sizeof(int16_t), int16_t,
                       std::conditional_t<X_DTYPE_SIZE_ == sizeof(int32_t), int32_t,
                                          std::conditional_t<X_DTYPE_SIZE_ == sizeof(int64_t), int64_t, int8_t>>>>;

/**
 * @brief TENSOR_MOVE(10000) 策略的 Process 函数
 * 纯数据搬运场景，经 RemoveAxisV2+MergeAxisV2 后 reduced dim==1。
 * 使用 DataCopyPad 直搬，双缓冲流水线加速。
 *
 * @param x      输入张量 GM 地址
 * @param y      输出张量 GM 地址
 * @param tiling Tiling 数据 GM 地址（TransposeOpTilingData 结构）
 */
extern "C" __aicore__ inline void TransposeTensorMoveProcess(GM_ADDR x, GM_ADDR y, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(TransposeOpTilingData, tilingData, tiling);
    Transpose::TransposeTensorMove<DTYPE_X_> op;
    op.Init(x, y, &tilingData, &pipe);
    op.Process();
}

// SIMT 专用模板：对 kernel 头部开销敏感，集中前置
#ifdef __CCE_AICORE_SUPPORT_SIMT__
/**
 * @brief SMALL_SHAPE(10001) 策略的 Process 函数
 * 小数据量场景，使用 SIMT 模式（2048线程并行）直接 GM→GM 读写。
 * 每个线程独立计算输出地址，无需 UB 中转；不使用缓冲队列，故不创建 TPipe。
 *
 * @param x      输入张量 GM 地址
 * @param y      输出张量 GM 地址
 * @param tiling Tiling 数据 GM 地址（TransposeOpTilingData 结构）
 */
extern "C" __aicore__ inline void TransposeSmallShapeProcess(GM_ADDR x, GM_ADDR y, GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(TransposeOpTilingData, tilingData, tiling);
    Transpose::TransposeSmallShape<DTYPE_X_> op;
    op.Init(x, y, &tilingData);
    op.Process();
}

/**
 * @brief B4(10009) 策略的 Process 函数
 * 4bit（HIF4/FP4）尾轴转置的双切分路径：CopyIn → Cast(fp4→bf16) → 转置 → Cast(bf16→fp4) → CopyOut。
 * 该模板按 fp4 语义处理数据，不参与 DTYPE_X_ 的 int8 归一化。
 *
 * @param x      输入张量 GM 地址
 * @param y      输出张量 GM 地址
 * @param tiling Tiling 数据 GM 地址（TransposeB4TilingData 结构）
 */
extern "C" __aicore__ inline void TransposeB4Process(GM_ADDR x, GM_ADDR y, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(TransposeB4TilingData, tilingData, tiling);
    TransposeB4 op;
    op.Init(x, y, &tilingData, &pipe);
    op.Process();
}
#endif

/**
 * @brief NDDMA_CUT_1(10002) 策略的 Process 函数
 * NDDMA 5维搬运，单轴切分。适用于 outCutIndex ≤ FindOutIndex(inCutIndex) 的场景。
 * 通过 DataCopy<T, TP_NDDMA_MAX_DIM_NUM> 5维NDDMA自动转置，硬件完成维度重排。
 *
 * @param x      输入张量 GM 地址
 * @param y      输出张量 GM 地址
 * @param tiling Tiling 数据 GM 地址（TransposeOpTilingData 结构）
 */
extern "C" __aicore__ inline void TransposeNddmaCut1Process(GM_ADDR x, GM_ADDR y, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(TransposeOpTilingData, tilingData, tiling);
    Transpose::TransposeNddmaCut1<DTYPE_X_> op;
    op.Init(x, y, &tilingData, &pipe);
    op.Process();
}

/**
 * @brief NDDMA_CUT_2(10003) 策略的 Process 函数
 * NDDMA 5维搬运，双轴切分。需要同时切输入轴和输出轴，数据被划分为4种区间：
 * Main / InputTail / OutputTail / Tail，每种区间使用不同的 NDDMA 参数。
 *
 * @param x      输入张量 GM 地址
 * @param y      输出张量 GM 地址
 * @param tiling Tiling 数据 GM 地址（TransposeOpTilingData 结构）
 */
extern "C" __aicore__ inline void TransposeNddmaCut2Process(GM_ADDR x, GM_ADDR y, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(TransposeOpTilingData, tilingData, tiling);
    Transpose::TransposeNddmaCut2<DTYPE_X_> op;
    op.Init(x, y, &tilingData, &pipe);
    op.Process();
}

/**
 * @brief N_LAST(10004) 策略的 Process 函数
 * 尾轴不转置（perm[dim-1]==dim-1）且尾轴≥32元素的场景。
 * 尾轴在输入输出中保持连续，可按连续行搬移，使用双缓冲流水线加速。
 * 支持8维循环展开（loop4~7）实现高效的 CopyOut。
 *
 * @param x      输入张量 GM 地址
 * @param y      输出张量 GM 地址
 * @param tiling Tiling 数据 GM 地址（TransposeOpTilingData 结构）
 */
extern "C" __aicore__ inline void TransposeNLastProcess(GM_ADDR x, GM_ADDR y, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(TransposeOpTilingData, tilingData, tiling);
    Transpose::TransposeNLast<DTYPE_X_> op;
    op.Init(x, y, &tilingData, &pipe);
    op.Process();
}

/**
 * @brief NDDMA_BIG_DIM(10005) 策略的 Process 函数
 * 维度>5 的场景，将原始 shape 压缩到5维 NDDMA 格式进行搬运。
 * Host 侧 FlushBaseNumForBigDim() 预计算 nddmaIdx 映射，Kernel 侧据此设置 NDDMA 参数。
 *
 * @param x      输入张量 GM 地址
 * @param y      输出张量 GM 地址
 * @param tiling Tiling 数据 GM 地址（TransposeOpTilingData 结构）
 */
extern "C" __aicore__ inline void TransposeNddmaBigDimProcess(GM_ADDR x, GM_ADDR y, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(TransposeOpTilingData, tilingData, tiling);
    Transpose::TransposeNddmaBigDim<DTYPE_X_> op;
    op.Init(x, y, &tilingData, &pipe);
    op.Process();
}

/**
 * @brief GATHER(10006) 策略的 Process 函数
 * 尾轴转置+大shape场景，使用 Gather 硬件指令（RegBase路径）。
 * 根据预计算的索引数组从 UB 中按任意跨步读取数据，实现尾轴转置。
 * 预生成4组索引覆盖 main/tail 的4种组合。
 *
 * @param x      输入张量 GM 地址
 * @param y      输出张量 GM 地址
 * @param tiling Tiling 数据 GM 地址（TransposeGatherTilingData 结构，非通用结构）
 */
extern "C" __aicore__ inline void TransposeGatherProcess(GM_ADDR x, GM_ADDR y, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(TransposeGatherTilingData, tilingData, tiling);
    Transpose::TransposeGather<DTYPE_X_> op;
    op.Init(x, y, &tilingData, &pipe);
    op.Process();
}

/**
 * @brief TRANSTO5HD_10(10007) 策略的 Process 函数
 * DAV_5102(Ascend950) 特定路径：2D perm=[1,0] + 16bit数据类型。
 * 使用 TransDataTo5HD 硬件指令实现2D矩阵转置，以16行为单位进行行列互换。
 * 运行时仅 16bit dtype 会路由到该模板。
 *
 * @param x      输入张量 GM 地址
 * @param y      输出张量 GM 地址
 * @param tiling Tiling 数据 GM 地址（TransposeTransTo5HD10TilingData 结构）
 */
extern "C" __aicore__ inline void TransposeTransTo5HD10Process(GM_ADDR x, GM_ADDR y, GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(TransposeTransTo5HD10TilingData, tilingData, tiling);
    // TransDataTo5HD 支持 8/16/32bit，但该模板仅处理 16bit（tiling 只会把 16bit dtype 路由到 10007）
    if constexpr (sizeof(DTYPE_X_) == sizeof(int16_t)) {
        TPipe pipe;
        Transpose::TransposeTransTo5HD10<DTYPE_X_> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    }
}

/**
 * @brief TRANSTO5HD_021(10008) 策略的 Process 函数
 * DAV_5102(Ascend950) 特定路径：3D perm=[0,2,1]（H↔W转置，N维不变）。
 * 使用 TransDataTo5HD 硬件指令，支持 8/16/32bit 数据类型。
 * 8bit 需要高低半分别处理（evenParams/oddParams），32bit dstStride 需加倍。
 *
 * @param x      输入张量 GM 地址
 * @param y      输出张量 GM 地址
 * @param tiling Tiling 数据 GM 地址（TransposeTransTo5HD021TilingData 结构）
 */
extern "C" __aicore__ inline void TransposeTransTo5HD021Process(GM_ADDR x, GM_ADDR y, GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(TransposeTransTo5HD021TilingData, tilingData, tiling);
    // TransDataTo5HD 支持 8/16/32bit；64bit 既不会路由到该模板，也不实例化（硬件指令不支持）
    if constexpr (sizeof(DTYPE_X_) <= sizeof(int32_t)) {
        TPipe pipe;
        Transpose::TransposeTransTo5HD021<DTYPE_X_> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    }
}

/**
 * @brief TRANSTO5HD_10_PERF(10017) 策略的 Process 函数
 * 2D perm=[1,0] 特化路径：dtype 为 16bit/32bit 且 m/n 均 16 对齐。
 * 按 tile 搬入 UB 后用 TransDataTo5HD 完成 16x16 块转置，再按列搬出。
 * 使用静态 Tensor 分配，不依赖 TPipe。
 *
 * @param x      输入张量 GM 地址
 * @param y      输出张量 GM 地址
 * @param tiling Tiling 数据 GM 地址（TransposeTransTo5HD10PerfTilingData 结构）
 */
extern "C" __aicore__ inline void TransposeTransTo5HD10PerfProcess(GM_ADDR x, GM_ADDR y, GM_ADDR tiling)
{
    GET_TILING_DATA_PTR_WITH_STRUCT(TransposeTransTo5HD10PerfTilingData, tilingDataPtr, tiling);
    Transpose::TransposeTransTo5HD10Perf<DTYPE_X_> op;
    op.Init(x, y, tilingDataPtr);
    op.Process();
}

/**
 * @brief Transpose 算子 Kernel 主入口函数
 *
 * 执行流程：
 * 1. 声明任务类型为 AIV_ONLY（仅使用 Vector 核）
 * 2. 通过 TILING_KEY_IS 判断当前 TilingKey，调用对应策略的 Process 函数
 * 3. 各 Process 函数内部自行解析 TilingData、按需创建 TPipe
 *
 * 分支顺序：SIMT 路径（SMALL_SHAPE/B4）对 kernel 头部开销敏感，紧随 TENSOR_MOVE 之后。
 *
 * @param x         输入张量 GM 地址
 * @param perm      perm 数组 GM 地址（Kernel 侧不直接使用，通过 TilingData 传递）
 * @param y         输出张量 GM 地址
 * @param workspace 工作空间 GM 地址
 * @param tiling    Tiling 数据 GM 地址
 */
extern "C" __global__ __aicore__ void transpose(GM_ADDR x, GM_ADDR perm, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY); // 声明为纯 Vector 核任务，不使用 Cube 核
    // 各 TilingKey 使用不同的 TilingData 结构体，无法注册唯一的默认结构体，故按 key 分支各自解读
    REGISTER_NONE_TILING;
    if (TILING_KEY_IS(TK_TENSOR_MOVE)) { // 10000: 纯数据搬运
        TransposeTensorMoveProcess(x, y, tiling);
#ifdef __CCE_AICORE_SUPPORT_SIMT__
    } else if (TILING_KEY_IS(TK_SMALL_SHAPE)) { // 10001: SIMT 小shape直读直写
        TransposeSmallShapeProcess(x, y, tiling);
    } else if (TILING_KEY_IS(TK_B4)) { // 10009: 4bit 双切分转置
        TransposeB4Process(x, y, tiling);
#endif
    } else if (TILING_KEY_IS(TK_NDDMA_CUT_1)) { // 10002: NDDMA单轴切分
        TransposeNddmaCut1Process(x, y, tiling);
    } else if (TILING_KEY_IS(TK_NDDMA_CUT_2)) { // 10003: NDDMA双轴切分
        TransposeNddmaCut2Process(x, y, tiling);
    } else if (TILING_KEY_IS(TK_N_LAST)) { // 10004: 尾轴不转置+连续行搬移
        TransposeNLastProcess(x, y, tiling);
    } else if (TILING_KEY_IS(TK_NDDMA_BIG_DIM)) { // 10005: >5维压缩到5维NDDMA
        TransposeNddmaBigDimProcess(x, y, tiling);
    } else if (TILING_KEY_IS(TK_GATHER)) { // 10006: DataCopyGather硬件加速
        TransposeGatherProcess(x, y, tiling);
    } else if (TILING_KEY_IS(TK_TRANSTO5HD_10)) { // 10007: TransDataTo5HD 2D转置
        TransposeTransTo5HD10Process(x, y, tiling);
    } else if (TILING_KEY_IS(TK_TRANSTO5HD_021)) { // 10008: TransDataTo5HD 3D 021转置
        TransposeTransTo5HD021Process(x, y, tiling);
    } else if (TILING_KEY_IS(TK_TRANSTO5HD_10_PERF)) { // 10017: 2D TransDataTo5HD 全tile转置
        TransposeTransTo5HD10PerfProcess(x, y, tiling);
    }
}
