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
 * \file transpose_tiling_data.h
 * \brief Transpose 算子 arch35 TilingData 结构定义（Host / Kernel 共用）
 *
 * 本文件被 Host 侧 Tiling 与 Kernel 侧算子共同包含，使用纯 POD struct 描述
 * Tiling 数据：Host 侧通过 TilingContext::GetTilingData<T>() 直接写入 Tiling 码流，
 * Kernel 侧通过 GET_TILING_DATA_WITH_STRUCT 按相同结构解读，两侧布局天然一致。
 *
 * TilingKey 与结构体对应关系：
 *   10000 - 10005 : TransposeOpTilingData
 *   10006         : TransposeGatherTilingData
 *   10007         : TransposeTransTo5HD10TilingData
 *   10008         : TransposeTransTo5HD021TilingData
 *   10009         : TransposeB4TilingData
 *   10017         : TransposeTransTo5HD10PerfTilingData
 */

#ifndef _TRANSPOSE_TILING_DATA_ARCH35_H_
#define _TRANSPOSE_TILING_DATA_ARCH35_H_

#include <cstdint>

namespace Transpose {

/* TilingData 数组维度常量，Host / Kernel 侧共享 */
static constexpr int64_t TP_MAX_AXIS_NUM = 8;         ///< 通用 TilingData 最大轴数
static constexpr int64_t TP_NDDMA_MAX_DIM_NUM = 5;    ///< NDDMA 5维搬运的最大维度数
static constexpr int8_t TP_GATHER_MAX_AXIS_NUM = 8;   ///< Gather 路径 block 轴最大数量
static constexpr int8_t TP_GATHER_UB_MAX_DIM_NUM = 6; ///< Gather 路径 UB 内轴最大数量
static constexpr int32_t TP_B4_GM_LEN = 8;            ///< B4 双切分 GM 轴最大数量
static constexpr int32_t TP_B4_UB_LEN = 4;            ///< B4 双切分 UB 轴最大数量

/**
 * @brief 通用 Tiling 数据（TilingKey 10000 - 10005）
 *
 * 包含 perm 信息、切分参数、核数分配、NDDMA 5维扩展参数和 UB shape 信息。
 */
struct TransposeOpTilingData {
    int64_t permSize = 0;
    int64_t inCutIndex = 0;
    int64_t outCutIndex = 0;
    int64_t inUbFactor = 0;
    int64_t outUbFactor = 0;
    int64_t inTailFactor = 0;
    int64_t outTailFactor = 0;
    int64_t realCoreNum = 0;
    int64_t blkFactor = 0;
    int64_t blkTailFactor = 0;
    int64_t ubSize = 0;
    int64_t totalNddmaNum = 0;
    int64_t rangeMainEnd = 0;
    int64_t rangeInputTailStart = 0;
    int64_t rangeInputTailEnd = 0;
    int64_t rangeOutputTailStart = 0;
    int64_t rangeOutputTailEnd = 0;
    int64_t rangeTailStart = 0;
    int64_t rangeTailEnd = 0;
    int64_t bigDimBufferNum = 0;
    int64_t inputShape[TP_MAX_AXIS_NUM] = {};
    int64_t outputShape[TP_MAX_AXIS_NUM] = {};
    int64_t perm[TP_MAX_AXIS_NUM] = {};
    int64_t baseInShape[TP_MAX_AXIS_NUM] = {};
    int64_t baseNddmaShape[TP_NDDMA_MAX_DIM_NUM] = {};
    int64_t nddmaIdx[TP_NDDMA_MAX_DIM_NUM] = {};
    int64_t expandedPerm[TP_NDDMA_MAX_DIM_NUM] = {};
    int64_t expandedInputShape[TP_NDDMA_MAX_DIM_NUM] = {};
    int64_t expandedOutputShape[TP_NDDMA_MAX_DIM_NUM] = {};
    int64_t inUbMainSrcShape[TP_NDDMA_MAX_DIM_NUM] = {};
    int64_t inUbMainDstShape[TP_NDDMA_MAX_DIM_NUM] = {};
    int64_t inUbInputTailSrcShape[TP_NDDMA_MAX_DIM_NUM] = {};
    int64_t inUbInputTailDstShape[TP_NDDMA_MAX_DIM_NUM] = {};
    int64_t inUbOutputTailSrcShape[TP_NDDMA_MAX_DIM_NUM] = {};
    int64_t inUbOutputTailDstShape[TP_NDDMA_MAX_DIM_NUM] = {};
    int64_t inUbTailSrcShape[TP_NDDMA_MAX_DIM_NUM] = {};
    int64_t inUbTailDstShape[TP_NDDMA_MAX_DIM_NUM] = {};
};

/**
 * @brief TransDataTo5HD 2D 转置（TilingKey 10007）分核切分参数
 */
struct CoreSplitPara {
    int64_t AlignBlockFactor = 0;     ///< 对齐后每核基础 block 数
    int64_t BlockFactor = 0;          ///< 每核基础 block 数
    int64_t BlockCount = 0;           ///< block 总数
    int64_t AlignBlockTailFactor = 0; ///< 对齐后尾核额外 block 数
    int64_t BlockTailFactor = 0;      ///< 尾核额外 block 数
};

/**
 * @brief TransDataTo5HD 2D 转置（TilingKey 10007）UB 切分参数
 */
struct UbSplitPara {
    int64_t MainCoreUbAlignFactor = 0;     ///< 主核 UB 对齐因子
    int64_t MainCoreUbFactor = 0;          ///< 主核 UB 切分因子
    int64_t MainCoreUbCount = 0;           ///< 主核 UB 循环次数
    int64_t MainCoreTailUbAlignFactor = 0; ///< 主核尾块 UB 对齐因子
    int64_t MainCoreTailUbFactor = 0;      ///< 主核尾块 UB 因子
    int64_t TailCoreUbAlignFactor = 0;     ///< 尾核 UB 对齐因子
    int64_t TailCoreUbFactor = 0;          ///< 尾核 UB 切分因子
    int64_t TailCoreUbCount = 0;           ///< 尾核 UB 循环次数
    int64_t TailCoreTailUbAlignFactor = 0; ///< 尾核尾块 UB 对齐因子
    int64_t TailCoreTailUbFactor = 0;      ///< 尾核尾块 UB 因子
};

/**
 * @brief TransDataTo5HD 2D 转置 Tiling 数据（TilingKey 10007）
 */
struct TransposeTransTo5HD10TilingData {
    int64_t AvailableUbSize = 0;   ///< 可用 UB 大小（单位：Block）
    int64_t UsedCoreNum = 0;       ///< 实际使用核数
    int64_t MainCoreLoopCount = 0; ///< 主核循环次数
    int64_t TailCoreLoopCount = 0; ///< 尾核循环次数
    int64_t RLen = 0;              ///< 行数（转置前）
    int64_t CLen = 0;              ///< 列数（转置前）
    int64_t RAlignBlock = 0;       ///< 行 Block 对齐数
    int64_t CAlignBlock = 0;       ///< 列 Block 对齐数
    int64_t RAlignBlockElem = 0;   ///< 行 Block 对齐后的元素数
    int64_t CAlignBlockElem = 0;   ///< 列 Block 对齐后的元素数
    bool IsRSplit = false;         ///< 是否按行切分
    bool IsRCSplit = false;        ///< 是否行列均切分
    CoreSplitPara rSplitPara;      ///< 行方向分核参数
    CoreSplitPara cSplitPara;      ///< 列方向分核参数
    UbSplitPara rUbSplitPara;      ///< 行方向 UB 切分参数
    UbSplitPara cUbSplitPara;      ///< 列方向 UB 切分参数
};

/**
 * @brief TransDataTo5HD 3D 021 转置（TilingKey 10008）UB 切分参数
 */
struct TransposeTransTo5HD021UbSplitPara {
    int64_t UbAlignFactor = 0;     ///< UB 对齐因子
    int64_t UbFactor = 0;          ///< UB 切分因子
    int64_t UbCount = 0;           ///< UB 循环次数
    int64_t UbTailAlignFactor = 0; ///< UB 尾块对齐因子
    int64_t UbTailFactor = 0;      ///< UB 尾块因子
};

/**
 * @brief TransDataTo5HD 3D 021 转置 Tiling 数据（TilingKey 10008）
 */
struct TransposeTransTo5HD021TilingData {
    int64_t AvailableUbSize = 0;                    ///< 可用 UB 大小（单位：Block）
    int64_t UsedCoreNum = 0;                        ///< 实际使用核数
    int64_t NLen = 0;                               ///< N 维长度
    int64_t HLen = 0;                               ///< H 维长度
    int64_t WLen = 0;                               ///< W 维长度
    int64_t HAlignBlockElem = 0;                    ///< H 维 Block 对齐后的元素数
    int64_t WAlignBlockElem = 0;                    ///< W 维 Block 对齐后的元素数
    int64_t NPerCore = 0;                           ///< 每核处理的 N 数量
    int64_t NTailCore = 0;                          ///< N 方向尾核数量
    int64_t UbLoopCount = 0;                        ///< UB 循环次数
    bool UseRConv = false;                          ///< 是否使用 R 方向转换
    bool UseHSplit = false;                         ///< 是否按 H 切分
    int64_t HPerCore = 0;                           ///< 每核处理的 H 数量
    int64_t HTailCore = 0;                          ///< H 方向尾核数量
    TransposeTransTo5HD021UbSplitPara rUbSplitPara; ///< R 方向 UB 切分参数
    TransposeTransTo5HD021UbSplitPara cUbSplitPara; ///< C 方向 UB 切分参数
};

/**
 * @brief DataCopyGather 路径 Tiling 数据（TilingKey 10006）
 */
struct TransposeGatherTilingData {
    uint64_t tilingKey = 0;                                 ///< TilingKey
    uint32_t dataTensorSize = 0;                            ///< 数据 Tensor 大小（单位：B）
    uint32_t indexTensorSize = 0;                           ///< 索引 Tensor 大小（单位：B）
    uint32_t usedCoreCnt = 0;                               ///< 实际使用核数
    int8_t blkAxesCnt = 0;                                  ///< block 轴数量
    int8_t blkInUbCutPos = 0;                               ///< block 输入切分轴位置
    int8_t blkOutUbCutPos = 0;                              ///< block 输出切分轴位置
    int8_t ubAxesCnt = 0;                                   ///< UB 内轴数量
    int8_t inUbInCutPos = 0;                                ///< 输入侧切分轴在 UB 内的位置（输入视角）
    int8_t inUbOutCutPos = 0;                               ///< 输入侧切分轴在 UB 内的位置（输出视角）
    int8_t outUbInCutPos = 0;                               ///< 输出侧切分轴在 UB 内的位置（输入视角）
    int8_t outUbOutCutPos = 0;                              ///< 输出侧切分轴在 UB 内的位置（输出视角）
    int64_t blkFactor = 0;                                  ///< 每核基础 block 数
    int64_t blkTailFactor = 0;                              ///< 尾核额外 block 数
    int64_t inUbCutAxisSize = 0;                            ///< 输入切分轴完整尺寸
    int64_t outUbCutAxisSize = 0;                           ///< 输出切分轴完整尺寸
    int32_t inUbCutAxisFactor = 0;                          ///< 输入切分轴主块因子
    int32_t outUbCutAxisFactor = 0;                         ///< 输出切分轴主块因子
    int64_t axis0InSrcStride = 0;                           ///< 搬入轴0 源跨步
    int64_t axis1InSrcStride = 0;                           ///< 搬入轴1 源跨步
    int64_t axis2InSrcStride = 0;                           ///< 搬入轴2 源跨步
    int64_t axis0OutDstStride = 0;                          ///< 搬出轴0 目标跨步
    int64_t axis1OutDstStride = 0;                          ///< 搬出轴1 目标跨步
    int64_t axis2OutDstStride = 0;                          ///< 搬出轴2 目标跨步
    int64_t blkAxes[TP_GATHER_MAX_AXIS_NUM] = {};           ///< block 循环轴尺寸
    int64_t blkAxesInAOffset[TP_GATHER_MAX_AXIS_NUM] = {};  ///< block 循环轴输入偏移
    int64_t blkAxesOutAOffset[TP_GATHER_MAX_AXIS_NUM] = {}; ///< block 循环轴输出偏移
    int32_t inUbAxes[TP_GATHER_UB_MAX_DIM_NUM] = {};        ///< UB 内输入各轴尺寸
    int32_t outUbAxes[TP_GATHER_UB_MAX_DIM_NUM] = {};       ///< UB 内输出各轴尺寸
    int8_t ubPerm[TP_GATHER_UB_MAX_DIM_NUM] = {};           ///< UB 内轴的排列顺序
};

/**
 * @brief HIF4 双切分 Tiling 数据（TilingKey 10009）
 */
struct TransposeB4TilingData {
    // --- 多核（GM）循环 ---
    int32_t gmShape[TP_B4_GM_LEN] = {};     ///< 外层循环 shape
    int64_t gmInStride[TP_B4_GM_LEN] = {};  ///< 每个 gm 轴的输入跨步
    int64_t gmOutStride[TP_B4_GM_LEN] = {}; ///< 每个 gm 轴的输出跨步
    // --- UB 内层循环 ---
    int32_t ubShape[TP_B4_UB_LEN] = {};      ///< ub 循环 shape
    int64_t ubInStride[TP_B4_UB_LEN] = {};   ///< 每个 ub 轴的输入跨步
    int64_t ubOutStride[TP_B4_UB_LEN] = {};  ///< 每个 ub 轴的输出跨步
    int64_t moveUbStride[TP_B4_UB_LEN] = {}; ///< ub 轴在输出序下的输出跨步
    int32_t ubPerm[TP_B4_UB_LEN] = {};       ///< ub 轴在输出序下的 perm
    int32_t ubShapeId[TP_B4_UB_LEN] = {};    ///< 每个 ub 轴对应的输入轴索引（未使用为 -1）
    // --- 切分轴尾块信息 ---
    int32_t cutAxisIdx01 = 0;  ///< 第1个切分轴的 ub 轴索引（无则为 -1）
    int32_t cutAxisIdx02 = 0;  ///< 第2个切分轴的 ub 轴索引（无则为 -1）
    int32_t cutAxisTail01 = 0; ///< 第1个切分轴尾块大小
    int32_t cutAxisTail02 = 0; ///< 第2个切分轴尾块大小
    // --- 切分轴在输入 shape 中的索引 ---
    int32_t outCutId = 0; ///< 输出切分轴在 inShape 中的索引（无则为 -1）
    int32_t inCutId = 0;  ///< 输入切分轴在 inShape 中的索引（无则为 -1）
    // --- 轴数量 ---
    int32_t outerAxisNums = 0; ///< 使用的 gm 轴数量
    int32_t innerAxisNums = 0; ///< 使用的 ub 轴数量
    // --- 多核切分 ---
    int32_t dimsPerCore = 0;     ///< 主核外层迭代次数
    int32_t useCoreNum = 0;      ///< 实际使用核数
    int32_t dimsPerCoreTail = 0; ///< 尾核外层迭代次数
    // --- UB 预算 ---
    int32_t ubSize = 0; ///< 每 buffer 有效 UB 大小（单位：B）
    // --- Kernel 侧 perm 信息 ---
    int32_t permSize = 0;               ///< 有效 perm 数量
    int64_t perm[TP_MAX_AXIS_NUM] = {}; ///< 转置 perm
};

/**
 * @brief 2D perm=[1,0] 特化模板的精简 Tiling 数据（TilingKey 10017）
 *
 * TransDataTo5HD 全 tile 转置路径：m x n 的二维转置，按 tileM x tileN 的 tile 搬入 UB
 * 后用 16x16 块转置指令处理，再按列搬出。字段类型为 int32_t，下发前 Host 侧已保证不越界。
 */
struct TransposeTransTo5HD10PerfTilingData {
    int32_t m = 0;       ///< 输入行数
    int32_t n = 0;       ///< 输入列数
    int32_t coreNum = 0; ///< 使用核数
    int32_t tileM = 0;   ///< 单 tile 行数
    int32_t tileN = 0;   ///< 单 tile 列数
};

} // namespace Transpose

#endif // _TRANSPOSE_TILING_DATA_ARCH35_H_
