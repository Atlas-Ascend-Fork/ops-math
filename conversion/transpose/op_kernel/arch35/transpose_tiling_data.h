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
 * 字段类型约定：
 *   - 轴相关（轴索引 / 轴数 / perm / 轴位置）统一用 int8_t；
 *   - UB 大小相关（UB 字节数 / UB 切分因子 / UB 循环次数）统一用 int32_t / uint32_t；
 *   - 字段按 8B → 4B → 2B → 1B 对齐从大到小排列，减少 padding。
 *
 * TilingKey 与结构体对应关系（各入口均使用各自的精简结构）：
 *   10000         : TransposeTensorMoveTilingData（普通/关联入口统一）
 *   10001         : TransposeSmallShapeTilingData<TP_MAX_AXIS_NUM>（四~八维精简结构，普通/关联入口统一）
 *   10002         : TransposeNddmaCut1TilingData（普通/关联入口统一）
 *   10003         : TransposeNddmaCut2TilingData（普通/关联入口统一）
 *   10004         : TransposeNLastTilingData（普通/关联入口统一）
 *   10005         : TransposeNddmaBigDimTilingData（NDDMA_BIG_DIM 紧凑结构）
 *   10006         : TransposeGatherTilingData
 *   10007         : TransposeTransTo5HD10TilingData
 *   10008         : TransposeTransTo5HD021TilingData
 *   10009         : TransposeB4TilingData
 *   10014         : TransposeNLastSmallTilingData（三/四维尾轴不变统一搬运）
 *   10017         : TransposeTransTo5HD10PerfTilingData（2D TransDataTo5HD 全 tile 转置）
 *   10011         : TransposeSmallShapeTilingData<3>（二/三维 SIMT 小 shape，Host 分核精简结构）
 *   10012         : TransposeSmallShape2DTilingData（2D perm=[1,0] UB 4x4 SIMT，64KB~小 shape 阈值）
 *
 * 说明：普通入口与关联算子入口（depth_to_space/space_to_depth/confusion_transpose_d）统一使用
 *       上述精简结构，不存在通用 TilingData。
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
 * @brief NDDMA_BIG_DIM(10005) 紧凑 Tiling 数据
 *
 * 仅下发 kernel 实际消费的字段：perm 维数、输出切分轴/因子、分核参数、5 维 NDDMA
 * 映射以及 UB 缓冲信息。仅保留 kernel 实际消费的字段，去掉冗余大数组，
 * 降低 Tiling 码流与 kernel 头开销。
 */
struct TransposeNddmaBigDimTilingData {
    int64_t blkFactor = 0;                             ///< 分核：每核基础 block 数
    int64_t blkTailFactor = 0;                         ///< 分核：尾核额外 block 数
    int64_t totalNddmaNum = 1;                         ///< NDDMA 循环总次数
    int64_t inputShape[TP_MAX_AXIS_NUM] = {};          ///< 简化后输入 shape
    int64_t outputShape[TP_MAX_AXIS_NUM] = {};         ///< 简化后输出 shape
    int64_t baseInShape[TP_MAX_AXIS_NUM] = {};         ///< 原始轴输入地址基数
    int64_t baseNddmaShape[TP_NDDMA_MAX_DIM_NUM] = {}; ///< NDDMA 5 维基础 shape
    uint32_t outUbFactor = 0;                          ///< 输出主块切分因子
    uint32_t outTailFactor = 0;                        ///< 输出尾块元素数
    uint32_t ubBytes = 0;                              ///< 每 buffer UB 字节数 = ubSize / bufferNum
    uint32_t bufferNum = 1;                            ///< UB 缓冲数（1 或 2）
    int8_t permSize = 0;                               ///< reduced dim（>5 且 <=8）
    int8_t outCutIndex = 0;                            ///< 输出切分轴索引
    int8_t perm[TP_MAX_AXIS_NUM] = {};                 ///< 简化后 perm
    int8_t nddmaIdx[TP_NDDMA_MAX_DIM_NUM] = {};        ///< NDDMA 5 维索引映射
};

/**
 * @brief TK_TENSOR_MOVE 紧凑 Tiling 数据（TilingKey 10000）
 *
 * 普通 Transpose 入口与关联算子入口（depth_to_space/space_to_depth/confusion_transpose_d）
 * 统一使用：一维纯搬运做动态主块划分，主块按 512B 对齐，完整块在参与核间均分，
 * 末核额外处理唯一尾块。
 */
struct TransposeTensorMoveTilingData {
    static constexpr int64_t CACHE_LINE_BYTES = 512;  ///< 动态主块对齐粒度（字节）
    static constexpr int64_t BANK_OFFSET_BYTES = 256; ///< 双缓冲错位量（字节）

    uint64_t mainBlocksPerCore; ///< 每核基础完整主块数
    uint32_t mainBlockElements; ///< 单个主块元素数（512B 对齐）
    uint32_t extraBlockCores;   ///< 额外多分一个主块的核数
    uint32_t tailElements;      ///< 末核唯一尾块元素数
};

/**
 * @brief N_LAST(10004) 紧凑 Tiling 数据
 *
 * 仅保留 kernel 实际消费的字段：分核参数、UB 预算、输入切分轴/因子、
 * reduced 输入 shape 与 perm。轴相关字段 int8，UB 相关字段 int32。
 */
struct TransposeNLastTilingData {
    int64_t blkFactor;                   ///< 分核：每核基础 block 数
    int64_t blkTailFactor;               ///< 分核：尾核额外 block 数
    int64_t inputShape[TP_MAX_AXIS_NUM]; ///< 简化后输入 shape
    int32_t inUbFactor;                  ///< 输入主块切分因子
    int32_t inTailFactor;                ///< 输入尾块元素数
    int32_t ubSize;                      ///< UB 总字节数
    int8_t permSize;                     ///< 简化后维数
    int8_t inCutIndex;                   ///< 输入切分轴索引
    int8_t perm[TP_MAX_AXIS_NUM];        ///< 简化后 perm
};

/**
 * @brief TK_N_LAST_SMALL(10014) 三/四维尾轴不变统一搬运 Tiling 数据（72B）
 *
 * 普通入口与关联算子入口统一使用。
 */
struct TransposeNLastSmallTilingData {
    static constexpr uint32_t MAX_BLOCK_COUNT = 4095;
    static constexpr uint32_t MAX_BLOCK_BYTES = (uint32_t{1} << 21) - 1;

    uint32_t rowBytes;
    uint32_t srcTileStride;
    uint32_t dstTileStride;
    uint32_t outDstStrideBytes;
    // 单块输出的UB循环步长；为0时使用分块流程。
    uint32_t srcLoopStrideBytes;
    uint32_t tailSrcBytes;
    uint32_t bTiles;
    uint32_t cTiles;
    uint32_t tileBytes;
    // 第二块UB缓冲区的偏移，单块时为0。
    uint32_t bufferOffset;
    uint32_t blocksPerCore;
    uint32_t aTiles;
    // 三维时为0；四维按最外层输入轴拆平面，后两个步长仅四维读取。
    uint32_t srcOuterStride;
    uint32_t dstOuterStride;
    uint32_t dstAStride;
    uint16_t tileA;
    uint16_t tailA;
    uint16_t tileB;
    uint16_t tailB;
    uint16_t extraBlocks;
};

/**
 * @brief NDDMA TK_NDDMA_CUT_1(10002) 紧凑 Tiling 数据
 *
 * 仅保留单轴切分 kernel 实际消费的字段：分核/UB/切分参数与 5 维 NDDMA 扩展信息。
 */
struct TransposeNddmaCut1TilingData {
    int64_t blkFactor;                                 ///< 分核：每核基础 block 数
    int64_t blkTailFactor;                             ///< 分核：尾核额外 block 数
    int64_t expandedInputShape[TP_NDDMA_MAX_DIM_NUM];  ///< 5 维扩展输入 shape
    int64_t expandedOutputShape[TP_NDDMA_MAX_DIM_NUM]; ///< 5 维扩展输出 shape
    int64_t inUbMainSrcShape[TP_NDDMA_MAX_DIM_NUM];    ///< 主块 UB 内输入 shape
    int64_t inUbMainDstShape[TP_NDDMA_MAX_DIM_NUM];    ///< 主块 UB 内输出 shape
    int64_t inUbTailSrcShape[TP_NDDMA_MAX_DIM_NUM];    ///< 尾块 UB 内输入 shape
    int64_t inUbTailDstShape[TP_NDDMA_MAX_DIM_NUM];    ///< 尾块 UB 内输出 shape
    int32_t outUbFactor;                               ///< 输出主块切分因子
    int32_t outTailFactor;                             ///< 输出尾块元素数
    int32_t ubSize;                                    ///< UB 总字节数
    int8_t permSize;                                   ///< 简化后维数
    int8_t outCutIndex;                                ///< 输出切分轴索引
    int8_t expandedPerm[TP_NDDMA_MAX_DIM_NUM];         ///< 5 维扩展 perm
};

/**
 * @brief NDDMA TK_NDDMA_CUT_2(10003) 紧凑 Tiling 数据
 *
 * 仅保留双轴切分 kernel 实际消费的字段：分核/UB/双轴切分参数、四区间边界
 * 与 5 维 NDDMA 扩展信息。
 */
struct TransposeNddmaCut2TilingData {
    int64_t blkFactor;                                    ///< 分核：每核基础 block 数
    int64_t blkTailFactor;                                ///< 分核：尾核额外 block 数
    int64_t rangeMainEnd;                                 ///< Main 区间结束索引
    int64_t rangeInputTailStart;                          ///< InputTail 区间起始索引
    int64_t rangeInputTailEnd;                            ///< InputTail 区间结束索引
    int64_t rangeOutputTailStart;                         ///< OutputTail 区间起始索引
    int64_t rangeOutputTailEnd;                           ///< OutputTail 区间结束索引
    int64_t rangeTailStart;                               ///< Tail 区间起始索引
    int64_t rangeTailEnd;                                 ///< Tail 区间结束索引
    int64_t expandedInputShape[TP_NDDMA_MAX_DIM_NUM];     ///< 5 维扩展输入 shape
    int64_t expandedOutputShape[TP_NDDMA_MAX_DIM_NUM];    ///< 5 维扩展输出 shape
    int64_t inUbMainSrcShape[TP_NDDMA_MAX_DIM_NUM];       ///< 主块 UB 内输入 shape
    int64_t inUbMainDstShape[TP_NDDMA_MAX_DIM_NUM];       ///< 主块 UB 内输出 shape
    int64_t inUbInputTailSrcShape[TP_NDDMA_MAX_DIM_NUM];  ///< 输入尾块 UB 内输入 shape
    int64_t inUbInputTailDstShape[TP_NDDMA_MAX_DIM_NUM];  ///< 输入尾块 UB 内输出 shape
    int64_t inUbOutputTailSrcShape[TP_NDDMA_MAX_DIM_NUM]; ///< 输出尾块 UB 内输入 shape
    int64_t inUbOutputTailDstShape[TP_NDDMA_MAX_DIM_NUM]; ///< 输出尾块 UB 内输出 shape
    int64_t inUbTailSrcShape[TP_NDDMA_MAX_DIM_NUM];       ///< 全尾块 UB 内输入 shape
    int64_t inUbTailDstShape[TP_NDDMA_MAX_DIM_NUM];       ///< 全尾块 UB 内输出 shape
    int32_t inUbFactor;                                   ///< 输入主块切分因子
    int32_t outUbFactor;                                  ///< 输出主块切分因子
    int32_t inTailFactor;                                 ///< 输入尾块元素数
    int32_t outTailFactor;                                ///< 输出尾块元素数
    int32_t ubSize;                                       ///< UB 总字节数
    int8_t permSize;                                      ///< 简化后维数
    int8_t inCutIndex;                                    ///< 输入切分轴索引
    int8_t outCutIndex;                                   ///< 输出切分轴索引
    int8_t expandedPerm[TP_NDDMA_MAX_DIM_NUM];            ///< 5 维扩展 perm
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
    int32_t MainCoreUbAlignFactor = 0;     ///< 主核 UB 对齐因子
    int32_t MainCoreUbFactor = 0;          ///< 主核 UB 切分因子
    int32_t MainCoreTailUbAlignFactor = 0; ///< 主核尾块 UB 对齐因子
    int32_t MainCoreTailUbFactor = 0;      ///< 主核尾块 UB 因子
    int32_t TailCoreUbAlignFactor = 0;     ///< 尾核 UB 对齐因子
    int32_t TailCoreUbFactor = 0;          ///< 尾核 UB 切分因子
    int32_t TailCoreTailUbAlignFactor = 0; ///< 尾核尾块 UB 对齐因子
    int32_t TailCoreTailUbFactor = 0;      ///< 尾核尾块 UB 因子
};

/**
 * @brief TransDataTo5HD 2D 转置 Tiling 数据（TilingKey 10007）
 */
struct TransposeTransTo5HD10TilingData {
    int64_t MainCoreLoopCount = 0; ///< 主核循环次数
    int64_t TailCoreLoopCount = 0; ///< 尾核循环次数
    int64_t RLen = 0;              ///< 行数（转置前）
    int64_t CLen = 0;              ///< 列数（转置前）
    int64_t RAlignBlock = 0;       ///< 行 Block 对齐数
    int64_t CAlignBlock = 0;       ///< 列 Block 对齐数
    int64_t RAlignBlockElem = 0;   ///< 行 Block 对齐后的元素数
    int64_t CAlignBlockElem = 0;   ///< 列 Block 对齐后的元素数
    CoreSplitPara rSplitPara;      ///< 行方向分核参数
    CoreSplitPara cSplitPara;      ///< 列方向分核参数
    UbSplitPara rUbSplitPara;      ///< 行方向 UB 切分参数
    UbSplitPara cUbSplitPara;      ///< 列方向 UB 切分参数
    int32_t AvailableUbSize = 0;   ///< 可用 UB 大小（单位：Block）
    bool IsRSplit = false;         ///< 是否按行切分
    bool IsRCSplit = false;        ///< 是否行列均切分
};

/**
 * @brief TransDataTo5HD 3D 021 转置（TilingKey 10008）UB 切分参数
 */
struct TransposeTransTo5HD021UbSplitPara {
    int32_t UbAlignFactor = 0;     ///< UB 对齐因子
    int32_t UbFactor = 0;          ///< UB 切分因子
    int32_t UbTailAlignFactor = 0; ///< UB 尾块对齐因子
    int32_t UbTailFactor = 0;      ///< UB 尾块因子
};

/**
 * @brief TransDataTo5HD 3D 021 转置 Tiling 数据（TilingKey 10008）
 */
struct TransposeTransTo5HD021TilingData {
    int64_t NLen = 0;                               ///< N 维长度
    int64_t HLen = 0;                               ///< H 维长度
    int64_t WLen = 0;                               ///< W 维长度
    int64_t HAlignBlockElem = 0;                    ///< H 维 Block 对齐后的元素数
    int64_t WAlignBlockElem = 0;                    ///< W 维 Block 对齐后的元素数
    int64_t NPerCore = 0;                           ///< 每核处理的 N 数量
    int64_t NTailCore = 0;                          ///< N 方向尾核数量
    int64_t UbLoopCount = 0;                        ///< UB 循环次数
    int64_t HPerCore = 0;                           ///< 每核处理的 H 数量
    int64_t HTailCore = 0;                          ///< H 方向尾核数量
    TransposeTransTo5HD021UbSplitPara rUbSplitPara; ///< R 方向 UB 切分参数
    TransposeTransTo5HD021UbSplitPara cUbSplitPara; ///< C 方向 UB 切分参数
    int32_t AvailableUbSize = 0;                    ///< 可用 UB 大小（单位：Block）
    bool UseRConv = false;                          ///< 是否使用 R 方向转换
    bool UseHSplit = false;                         ///< 是否按 H 切分
};

/**
 * @brief DataCopyGather 路径 Tiling 数据（TilingKey 10006）
 *
 * 只保留 kernel 实际消费的字段（去掉冗余 tilingKey），并按 8B 对齐原则把
 * int64_t / 数组字段前置、标量与标记位后置，尽量减小 Tiling 码流与 kernel 头开销。
 */
struct TransposeGatherTilingData {
    int64_t blkFactor = 0;                                  ///< 每核基础 block 数
    int64_t blkTailFactor = 0;                              ///< 尾核额外 block 数
    int64_t inUbCutAxisSize = 0;                            ///< 输入切分轴完整尺寸
    int64_t outUbCutAxisSize = 0;                           ///< 输出切分轴完整尺寸
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
    uint32_t dataTensorSize = 0;                            ///< 数据 Tensor 大小（单位：B）
    uint32_t indexTensorSize = 0;                           ///< 索引 Tensor 大小（单位：B）
    int32_t inUbCutAxisFactor = 0;                          ///< 输入切分轴主块因子
    int32_t outUbCutAxisFactor = 0;                         ///< 输出切分轴主块因子
    int8_t blkAxesCnt = 0;                                  ///< block 轴数量
    int8_t blkInUbCutPos = 0;                               ///< block 输入切分轴位置
    int8_t blkOutUbCutPos = 0;                              ///< block 输出切分轴位置
    int8_t ubAxesCnt = 0;                                   ///< UB 内轴数量
    int8_t inUbInCutPos = 0;                                ///< 输入侧切分轴在 UB 内的位置（输入视角）
    int8_t inUbOutCutPos = 0;                               ///< 输入侧切分轴在 UB 内的位置（输出视角）
    int8_t outUbInCutPos = 0;                               ///< 输出侧切分轴在 UB 内的位置（输入视角）
    int8_t outUbOutCutPos = 0;                              ///< 输出侧切分轴在 UB 内的位置（输出视角）
    int8_t ubPerm[TP_GATHER_UB_MAX_DIM_NUM] = {};           ///< UB 内轴的排列顺序
};

/**
 * @brief HIF4 双切分 Tiling 数据（TilingKey 10009）
 */
struct TransposeB4TilingData {
    // --- int64：各轴跨步与 perm ---
    int64_t gmInStride[TP_B4_GM_LEN] = {};   ///< 每个 gm 轴的输入跨步
    int64_t gmOutStride[TP_B4_GM_LEN] = {};  ///< 每个 gm 轴的输出跨步
    int64_t ubInStride[TP_B4_UB_LEN] = {};   ///< 每个 ub 轴的输入跨步
    int64_t ubOutStride[TP_B4_UB_LEN] = {};  ///< 每个 ub 轴的输出跨步
    int64_t moveUbStride[TP_B4_UB_LEN] = {}; ///< ub 轴在输出序下的输出跨步
    // --- int32：轴长度/切分参数/核分配/UB 预算 ---
    int32_t gmShape[TP_B4_GM_LEN] = {}; ///< 外层循环 shape
    int32_t ubShape[TP_B4_UB_LEN] = {}; ///< ub 循环 shape
    int32_t cutAxisTail01 = 0;          ///< 第1个切分轴尾块大小
    int32_t cutAxisTail02 = 0;          ///< 第2个切分轴尾块大小
    int32_t dimsPerCore = 0;            ///< 主核外层迭代次数
    int32_t dimsPerCoreTail = 0;        ///< 尾核外层迭代次数
    int32_t ubSize = 0;                 ///< 每 buffer 有效 UB 大小（单位：B）
    // --- int8：轴索引/轴数/perm ---
    int8_t ubPerm[TP_B4_UB_LEN] = {};    ///< ub 轴在输出序下的 perm
    int8_t ubShapeId[TP_B4_UB_LEN] = {}; ///< 每个 ub 轴对应的输入轴索引（未使用为 -1）
    int8_t cutAxisIdx01 = 0;             ///< 第1个切分轴的 ub 轴索引（无则为 -1）
    int8_t cutAxisIdx02 = 0;             ///< 第2个切分轴的 ub 轴索引（无则为 -1）
    int8_t outCutId = 0;                 ///< 输出切分轴在 inShape 中的索引（无则为 -1）
    int8_t inCutId = 0;                  ///< 输入切分轴在 inShape 中的索引（无则为 -1）
    int8_t outerAxisNums = 0;            ///< 使用的 gm 轴数量
    int8_t innerAxisNums = 0;            ///< 使用的 ub 轴数量
    int8_t permSize = 0;                 ///< 有效 perm 数量
    int8_t perm[TP_MAX_AXIS_NUM] = {};   ///< 转置 perm（轴索引，int8）
};

/**
 * @brief 2D perm=[1,0] UB 4x4 SIMT（TilingKey 10012）的 Tiling 数据
 *
 * Host 按 128B tile 切 stripe、按 UB 预算切 chunkCols，并下发 totalWork/inPitch/bufBytes；
 * kernel 只负责当前核的 work 分配、M/N 尾块与 x0/x1/y0/y1 双缓冲流水。
 * 实际使用核数由 kernel 侧 GetBlockNum() 获取。
 */
struct TransposeSmallShape2DTilingData {
    uint32_t rows = 0;      ///< 输入行数 M（= 输出列数）
    uint32_t cols = 0;      ///< 输入列数 N（= 输出行数）
    uint32_t chunkCols = 0; ///< N 方向单 chunk 列数（Host 切分）
    uint32_t totalWork = 0; ///< 总 work item 数（stripe × chunk）
    uint32_t inPitch = 0;   ///< UB 输入行间距（元素）
    uint32_t bufBytes = 0;  ///< 单 buffer 字节数（(ubSize - DCACHE)/4，32B 对齐）
};
static_assert(sizeof(TransposeSmallShape2DTilingData) <= 24, "TransposeSmallShape2DTilingData must be <= 24B");

/**
 * @brief 2D perm=[1,0] TransDataTo5HD 全 tile 转置的精简 Tiling 数据（TilingKey 10017）
 *
 * 按 tileM x tileN 的 tile 搬入 UB 后用 16x16 块转置指令处理，再按列搬出。
 * 字段类型为 int32_t，下发前 Host 侧已保证不越界。
 */
struct TransposeTransTo5HD10PerfTilingData {
    int32_t m = 0;     ///< 输入行数
    int32_t n = 0;     ///< 输入列数
    int32_t tileM = 0; ///< 单 tile 行数
    int32_t tileN = 0; ///< 单 tile 列数
};

/**
 * @brief SIMT 小 shape 精简 Tiling 数据（2-8D，按维数容量模板化）
 *
 * DIM 为该实例承接的最大维数（数组容量）：
 *   - 10011（二/三维）用 TransposeSmallShapeTilingData<3>，24B；
 *   - 10001（四~八维）用 TransposeSmallShapeTilingData<TP_MAX_AXIS_NUM>。
 * 实际维数由 dim 字段给出（<= DIM），inputShape/perm 只取前 dim 个有效槽位，
 * 未用槽位分别置 0 与 -1。
 * 分核由 Host 按 SMALL_SHAPE 的 1024B 对齐公式算好（blkFactor/blkTailFactor）；
 * 实际使用核数不下发，kernel 侧通过 GetBlockNum() 获取（blockDim 精确等于 Host 分核结果）。
 * 输出 shape/步长/魔术数仍由 kernel 按 inputShape+perm 推导。
 */
template <int64_t DIM>
struct TransposeSmallShapeTilingData {
    static constexpr int64_t AXIS_NUM = DIM; ///< 数组容量（该实例承接的最大维数）
    uint32_t blkFactor;                      ///< 每核基础元素数（1024B 对齐）
    uint32_t blkTailFactor;                  ///< 末核元素数
    int32_t inputShape[DIM];                 ///< 简化后输入 shape（前 dim 个有效）
    int8_t perm[DIM];                        ///< 简化后 perm（前 dim 个有效，未用槽位 -1）
    uint8_t dim;                             ///< 实际简化后维数（<= DIM）
};

// 两个 Kernel 模板各自的 TilingData 具体类型（由上面的模板实例化得到）
using TransposeSmallShape23DTilingData = TransposeSmallShapeTilingData<3>;               ///< 二/三维 Kernel
using TransposeSmallShape48DTilingData = TransposeSmallShapeTilingData<TP_MAX_AXIS_NUM>; ///< 四~八维 Kernel
static_assert(sizeof(TransposeSmallShape23DTilingData) <= 24, "TransposeSmallShape23DTilingData must be <= 24B");

} // namespace Transpose

#endif // _TRANSPOSE_TILING_DATA_ARCH35_H_
