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
 * \brief Transpose Tiling 核心数据结构和策略选型定义
 *
 * 本文件定义了：
 * 1. SplitMode 枚举：各 Tiling 策略的 key 值定义
 * 2. SplitInfo/Interval/ParamInfo：Tiling 计算中间结构体
 * 3. TransposeNddmaTiling：核心 Tiling 计算类
 *
 * TilingData 结构体本身由 op_kernel/arch35/transpose_tiling_data.h 以纯 struct 形式定义，
 * Host 与 Kernel 共用同一份定义，保证两侧内存布局完全一致。
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
#include "register/tilingdata_base.h"
#include "transpose_tiling_base.h"
#include "../../op_kernel/arch35/transpose_tiling_data.h"

namespace optiling {
using namespace Transpose;
/* 核心常量定义 */
constexpr int64_t NDDMA_MAX_LOOP_NUM = 3;                   ///< NDDMA 最大循环层数
constexpr uint64_t INPUT_IDX_X = 0;                         ///< 输入 x 的索引
constexpr uint64_t OUTPUT_IDX_Y = 0;                        ///< 输出 y 的索引
constexpr uint64_t INPUT_IDX_PERM = 1;                      ///< 输入 perm 的索引
constexpr uint64_t B8_BYTES = 1;                            ///< 8bit 类型字节数
constexpr uint64_t B16_BYTES = 2;                           ///< 16bit 类型字节数
constexpr uint64_t B32_BYTES = 4;                           ///< 32bit 类型字节数
constexpr uint64_t B64_BYTES = 8;                           ///< 64bit 类型字节数
constexpr uint64_t BUFFER_NUM = 2;                          ///< 缓冲数量
constexpr int64_t BUFFER_NUM_DOUBLE_BUF = 2;                ///< 双缓冲
constexpr int64_t BUFFER_NUM_SINGLE_BUF = 1;                ///< 单缓冲
constexpr double VEC_CORE_USED_THRES_HOLD = 0.9;            ///< 核利用率阈值（低于此值调整切分因子）
constexpr int64_t MOVEALIGN_LAST_MIN_ELE = 32;              ///< N_LAST_TRANSPOSE 尾轴最小元素数
constexpr int64_t SMALL_SHAPE_SPLIT_BYTES_ALIGN_SIZE = 128; ///< SmallShape 128字节对齐
constexpr int64_t INPUT_IDX = 0;
constexpr int64_t OUTPUT_IDX = 0;
constexpr int64_t ATTR_BLOCK_SIZE_IDX = 0;
constexpr int64_t ATTR_MODE_IDX = 1;
constexpr int64_t ATTR_DEPTH_DATA_FORMAT_IDX = 2;
constexpr int64_t ATTR_SPACE_DATA_FORMAT_IDX = 1;
constexpr int64_t DIM_NUM = 4;
constexpr int64_t DIM_ZERO = 0;
constexpr int64_t DIM_ONE = 1;
constexpr int64_t DIM_TWO = 2;
constexpr int64_t DIM_THREE = 3;
constexpr int64_t DIM_FOUR = 4;
constexpr int64_t DIM_FIVE = 5;
constexpr int64_t DIM_SIX = 6;
constexpr int64_t DIM_EIGHT = 8;
constexpr int64_t HW_ALIGN = 16;        ///< HW 对齐单位（TransDataTo5HD 使用16对齐）
constexpr int64_t HW_MIN_PRODUCT = 448; ///< HW 最小乘积阈值（021 VCONV 场景）
constexpr int64_t SMALL_SHAPE_BYTES_THRES_HOLD_DAV_5102 = 1000000;      ///< DAV_5102 小shape阈值：1MB
constexpr int64_t SMALL_SHAPE_BYTES_THRES_HOLD_DAV_5102_NLAST = 400000; ///< DAV_5102 N_LAST小shape阈值：400KB
constexpr int64_t SMALL_SHAPE_BYTES_THRES_HOLD_DAV_5102_021 = 70000;    ///< DAV_5102 021小shape阈值：70KB

/**
 * @brief Tiling 策略的 Key 值枚举
 *
 */
enum class SplitMode : int64_t {
    TENSOR_MOVE = 10000,            // 仅1维，纯数据搬运
    SMALL_SHAPE = 10001,            // 小shape，SIMT模式GM→GM
    CUT_ONCE = 10002,               // NDDMA单轴切分
    CUT_TWICE = 10003,              // NDDMA双轴切分
    N_LAST_TRANSPOSE = 10004,       // 尾轴不转置+连续行搬移
    BIG_DIM = 10005,                // >5维压缩到5维NDDMA
    GATHER_TRANSPOSE = 10006,       // DataCopyGather硬件加速
    VCONV_TRANSPOSE = 10007,        // TransDataTo5HD 2D转置(DAV_5102)
    VCONV_021_TRANSPOSE = 10008,    // TransDataTo5HD 3D 021转置(DAV_5102)
    B4_DOUBLE_SPLIT = 10009,        // B4 双切分
    TRANSPOSE_2D_TRANSDATA = 10011, // 2D perm=[1,0] 特化，TransDataTo5HD 全tile转置
    NDDMA_BASE = 90000              // NDDMA基础key（非实际使用的key，用于策略选型中间状态）
};

/**
 * @brief UB 切分信息结构体
 *
 * 记录 Tiling 计算过程中的 UB 切分参数，包括输入/输出切分轴、切分因子、尾块因子等。
 *
 * 字段含义：
 *   - ubElement：UB 预算（元素数），TENSOR_MOVE/N_LAST 为 ubSize/2/eleBytes，
 *     NDDMA_BASE 取 sqrt(ubSize/eleBytes)
 *   - inUbElement/outUbElement：输入/输出侧剩余预算（切轴扫描中逐步扣除）
 *   - inUbActual/outUbActual：已整体吞入 block 的轴的容量乘积
 *   - inCutIndex/outCutIndex：输入/输出侧切分轴索引
 *   - inUbFactor/outUbFactor：主块切分因子（每 block 切分轴上搬的元素数）
 *   - inTailFactor/outTailFactor：尾块元素数（shape % factor，0 表示无尾块）
 */
struct SplitInfo {
    int64_t ubElement = 1;     ///< UB 元素预算
    int64_t inUbElement = 1;   ///< 输入侧剩余预算
    int64_t outUbElement = 1;  ///< 输出侧剩余预算
    int64_t inUbActual = 1;    ///< 输入侧已吞入轴容量乘积
    int64_t outUbActual = 1;   ///< 输出侧已吞入轴容量乘积
    int64_t inCutIndex = 0;    ///< 输入切分轴索引
    int64_t outCutIndex = 0;   ///< 输出切分轴索引
    int64_t inUbFactor = 0;    ///< 输入主块切分因子
    int64_t outUbFactor = 0;   ///< 输出主块切分因子
    int64_t inTailFactor = 0;  ///< 输入尾块元素数
    int64_t outTailFactor = 0; ///< 输出尾块元素数
};

/**
 * @brief 区间结构体（CUT_TWICE 的 4 种数据区间边界）
 */
struct Interval {
    int64_t start = 0; ///< 区间起始全局 block 索引
    int64_t end = 0;   ///< 区间结束全局 block 索引
};

struct ParamInfo {
    gert::Shape xShape;
    ge::DataType xDtype;
    int64_t blockSize;
    const char* modePtr;
    const char* dataFormatPtr;
};

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
    bool CheckB4Tiling();
    ge::graphStatus TryB4Tiling();
    ge::graphStatus AdjustShapeForB4();
    ge::graphStatus TryVCONVTiling();
    bool Is021VConvValid();
    bool Is2DPermutation();
    ge::graphStatus Fill2DTransDataTilingData();
    void FlushBaseNumForBigDim();
    void EntryTilingTemplate();
    void CalcUBSplitInfo();
    void CalcBlockSplitInfo();
    void CalcBlockSplitInfoForTensorMove();
    void CalcBlockSplitInfoForSmallShape();
    void CalcBlockSplitInfoForNLastTranspose();
    void SetRealCoreNumAndBlkFactor(int64_t coreNum);
    void CalcBlockSplitInfoForCutOnce();
    void CalcBlockSplitInfoForCutTwice();
    void CalcBlockSplitInfoForBigDim();
    ge::graphStatus FillTilingData();
    void PrintTilingData();
    void DoSplitUB();
    int64_t DoSplitUBInput(bool tryAbsorb = false);
    void DoSplitUBOutputScan(int64_t remainingTotalElment);
    int64_t FindOutIndex(int64_t index);
    bool UbOutOfBoundCheck(int64_t currentSplitIndex, int64_t currentSplitValue, bool calcIn);
    bool UbOutOfBoundCheckNLast(int64_t currentSplitIndex, int64_t currentSplitValue);
    void FindSplitFactorByMultiplesLast(int64_t currentSplitIndex, int64_t currentInShapeDim,
                                        int64_t remainingTotalElment, int64_t coreNumMultiples);
    void FindSplitFactorByRateNLast(int64_t currentSplitIndex, int64_t currentInShapeDim, int64_t remainingTotalElment);
    void FindSplitFactorByMultiplesNLast(int64_t currentSplitIndex, int64_t currentInShapeDim,
                                         int64_t remainingTotalElment, int64_t coreNumMultiples);
    void CheckInUbFactorValid(int64_t& currentSplitIndex, int64_t& currentInShapeDim, int64_t& remainingTotalElment,
                              int64_t& coreNumMultiples, int64_t* solvedTotalElment);
    void DoSplitUBBigDim();
    void NDDMADimExpand();
    void GetInUbShapeInfo();
    void GetIntervalInfo();
    void CalcInUbShapeInfoForNoNeedCut();
    void CalcInUbShapeInfoForCutOnce();
    void CalcInUbShapeInfoForCutTwice();
    void GetIntervalInfoForCutTwice();

private:
    gert::TilingContext* tilingContext_ = nullptr;

    int64_t realCoreNum_ = 0;
    int64_t tilingKey_ = 0;
    int64_t blkFactor_ = 0;
    int64_t blkTailFactor_ = 0;
    int64_t nddmaBufferNum_ = 1;
    int64_t bigDimBufferNum_ = 1;
    int64_t totalNddmaNum_ = 1;
    int64_t SMALL_SHAPE_BYTES_THRES_HOLD = 4000000;
    int64_t inputShape_[TP_MAX_AXIS_NUM] = {0};
    int64_t outputShape_[TP_MAX_AXIS_NUM] = {0};
    int64_t perm_[TP_MAX_AXIS_NUM] = {0};
    ShapeInfo shapeInfo_;
    SplitInfo splitInfo_;
    int64_t coreNum_ = 0;
    int64_t ubSize_ = 0;
    int64_t cacheLineSize_ = 0;
    int64_t ubBlockSize_ = 0;
    Interval offsetRangeMain_;
    Interval offsetRangeInputTail_;
    Interval offsetRangeOutputTail_;
    Interval offsetRangeTail_;
    int64_t baseInShape_[TP_MAX_AXIS_NUM] = {0};
    int64_t baseNddmaShape_[TP_NDDMA_MAX_DIM_NUM] = {0};
    int64_t nddmaIdx_[TP_NDDMA_MAX_DIM_NUM] = {-1};

    int64_t expandedPerm_[TP_NDDMA_MAX_DIM_NUM] = {0, 1, 2, 3, 4};
    int64_t expandedInputShape_[TP_NDDMA_MAX_DIM_NUM] = {1, 1, 1, 1, 1};
    int64_t expandedOutputShape_[TP_NDDMA_MAX_DIM_NUM] = {1, 1, 1, 1, 1};
    int64_t inUbMainSrcShape_[TP_NDDMA_MAX_DIM_NUM] = {0};
    int64_t inUbMainDstShape_[TP_NDDMA_MAX_DIM_NUM] = {0};
    int64_t inUbInputTailSrcShape_[TP_NDDMA_MAX_DIM_NUM] = {0};
    int64_t inUbInputTailDstShape_[TP_NDDMA_MAX_DIM_NUM] = {0};
    int64_t inUbOutputTailSrcShape_[TP_NDDMA_MAX_DIM_NUM] = {0};
    int64_t inUbOutputTailDstShape_[TP_NDDMA_MAX_DIM_NUM] = {0};
    int64_t inUbTailSrcShape_[TP_NDDMA_MAX_DIM_NUM] = {0};
    int64_t inUbTailDstShape_[TP_NDDMA_MAX_DIM_NUM] = {0};

    bool isRelatedTranspose_ = false;
    bool is4bit_ = false;
};
} // namespace optiling

#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSPOSE_TILING_ARCH35_H
