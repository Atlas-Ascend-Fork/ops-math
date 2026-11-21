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
 * \file transpose_tiling_strategy_gather.h
 * \brief GATHER(10006) 策略类
 *
 * 适用场景：非融合内联场景下的尾轴转置（非 4bit），
 * 使用 DataCopyGather 指令完成搬运。
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSPOSE_TILING_STRATEGY_GATHER_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSPOSE_TILING_STRATEGY_GATHER_H

#include <array>
#include <set>
#include "transpose_tiling_strategy.h"

namespace optiling {

/**
 * @brief GATHER(10006) 策略
 *
 * 准入：!isRelatedTranspose_ && shapeInfo_.isLastAxisTranspose && !is4bit_。
 * 切分成功返回 SUCCESS；UB 效率或块数不足返回 FALLBACK；TilingData 写回失败返回 FAILURE。
 */
class TransposeGatherStrategy : public TransposeTilingStrategy {
private:
    static constexpr int8_t GATHER_UB_MAX_BRW_NUM = 3; ///< UB 内最多搬入/搬出轴数

    /**
     * @brief Gather 路径 UB 切分信息结构体
     *
     * 记录进入 UB 的轴尺寸、切分轴因子、MTE 搬运跨步等
     *
     *   inUbAxes/outUbAxes：UB 内输入/输出各轴尺寸（最多 TP_GATHER_UB_MAX_DIM_NUM=6 维）
     *   ubPerm：UB 内轴的排列顺序（索引=allUbPerm 中的序号）
     *   inUbCutAxisSize/outUbCutAxisSize：输入/输出切分轴完整尺寸
     *   inUbCutAxisFactor/outUbCutAxisFactor：切分轴主块因子
     *   inUbInCutPos/outUbInCutPos：输入切分轴在 UB 轴序列中的位置
     *   inUbOutCutPos/outUbOutCutPos：输出切分轴在 UB 轴序列中的位置
     *   axis0/1/2InSrcStride：搬入 GM→UB 的循环轴源跨步
     *   axis0/1/2OutDstStride：搬出 UB→GM 的循环轴目标跨步
     */
    struct GatherUbSplitInfo {
        int64_t inUbCutAxisSize = 1;
        int64_t outUbCutAxisSize = 1;
        int64_t axis0InSrcStride = 0; // loop layer from outer to inner: 2, 1, 0
        int64_t axis1InSrcStride = 0;
        int64_t axis2InSrcStride = 0;
        int64_t axis0OutDstStride = 0;
        int64_t axis1OutDstStride = 0;
        int64_t axis2OutDstStride = 0;
        int32_t inUbCutAxisFactor = 1;
        int32_t outUbCutAxisFactor = 1;
        // for count gather index
        int32_t inUbAxes[Transpose::TP_GATHER_UB_MAX_DIM_NUM] = {1, 1, 1, 1, 1, 1}; // maybe changed accorrding to MTE2
        int32_t outUbAxes[Transpose::TP_GATHER_UB_MAX_DIM_NUM] = {1, 1, 1, 1, 1, 1};
        int8_t ubPerm[Transpose::TP_GATHER_UB_MAX_DIM_NUM] = {0xf, 0xf, 0xf,
                                                              0xf, 0xf, 0xf}; // maybe changed accorrding to MTE2
        int8_t ubAxesCnt = 0;
        // for update inUbAxes and outUbAxes
        int8_t inUbInCutPos = 0;  // include right offset to be move in burst length
        int8_t inUbOutCutPos = 0; // maybe changed accorrding to MTE2
        int8_t outUbInCutPos = 0;
        int8_t outUbOutCutPos = 0; // include right offset to be move out burst length
    };
    /**
     * @brief Gather 路径分核（块级循环）信息结构体
     *
     * 记录进入块级循环（block loop）的轴及地址跨步：
     *   blkAxes[cnt]：块级各轴循环次数
     *   blkAxesInAOffset[cnt]：块级输入地址跨步（元素数）
     *   blkAxesOutAOffset[cnt]：块级输出地址跨步（元素数）
     *   blkFactor/blkTailFactor：每核块数/尾核块数
     *   usedCoreCnt：实际使用核数
     *   blkInUbCutPos/blkOutUbCutPos：块级循环最后一轮时输入/输出切分轴要切尾块
     *     （kernel UpdateUbAxes 据此切换 main/tail 与 gather 索引 phase）
     */
    struct GatherBlockSplitInfo {
        int64_t blkAxes[Transpose::TP_GATHER_MAX_AXIS_NUM] = {1, 1, 1, 1, 1, 1, 1, 1};
        int64_t blkAxesInAOffset[Transpose::TP_GATHER_MAX_AXIS_NUM] = {1, 1, 1, 1, 1, 1, 1, 1};
        int64_t blkAxesOutAOffset[Transpose::TP_GATHER_MAX_AXIS_NUM] = {1, 1, 1, 1, 1, 1, 1, 1};
        int64_t blkFactor = 1;
        int64_t blkTailFactor = 0;
        uint32_t usedCoreCnt = 0;
        int8_t blkAxesCnt = 0;
        // to count what time to update input and output ub cut axes
        int8_t blkInUbCutPos = -1;
        int8_t blkOutUbCutPos = -1;
    };

    struct GatherUbPermInfo {
        int8_t cnt = 0;
        int8_t perm[GATHER_UB_MAX_BRW_NUM] = {0xf, 0xf, 0xf};
    };

public:
    using TransposeTilingStrategy::TransposeTilingStrategy;

    // G1：准入检查，纯判定、无副作用
    bool IsSupport() const override;

protected:
    // 主体切分
    TilingStatus DoExecute() override;
    // 写 TransposeGatherTilingData（容量不足 → FAILURE）
    bool SetTilingData() override;
    // 打印 TilingData
    void LogTilingData() override;
    uint32_t GetUsedCoreNum() const override { return blkSplitInfo_.usedCoreCnt; }
    uint64_t GetTilingKey() const override { return tilingKey_; }

private:
    int64_t CalcShapeSize(const std::array<int64_t, Transpose::TP_MAX_AXIS_NUM>& shape, int64_t beg, int64_t end);
    int64_t CalcSqrtedTensor(int64_t elemInTensor);
    void CalcTensorSize();
    void CalcInUbPerm(int64_t sqrtedTensor);
    void CalcOutUbPerm(int64_t sqrtedTensor);
    void CalcUbAxisCutFactor(int64_t elemInTensor, int64_t sqrtedTensor, bool isLastInPermLeft, bool isLastOutPermLeft,
                             const std::set<int8_t>& viceAllUbPerm);
    void CalcUbCutFactorBothLeft(int64_t elemInTensor, int64_t sqrtedTensor, int64_t outSavedElems,
                                 int64_t inSavedElems, int64_t elemPerBlock, int64_t maxCutAxisSize,
                                 int64_t maxOutCutAxisSize);
    void CalcUbCutFactorDiffLastAxis(int64_t elemInTensor, int64_t sqrtedTensor, int64_t outSavedElems,
                                     int64_t inSavedElems, int64_t elemPerBlock);
    void CalcUbCutFactorOutLeftOnly(int64_t elemInTensor, int64_t maxCutAxisSize, int64_t maxOutCutAxisSize);
    void CalcUbCutFactorInLeftOnly(int64_t elemInTensor, int64_t maxCutAxisSize);
    ge::graphStatus CalcUbAxesInfo(const int64_t (&tmpInAxes)[Transpose::TP_GATHER_MAX_AXIS_NUM],
                                   const int64_t (&tmpOutAxes)[Transpose::TP_GATHER_MAX_AXIS_NUM],
                                   const int8_t (&tmpOutPerm)[Transpose::TP_GATHER_MAX_AXIS_NUM]);
    void CompactUbAxes(const int64_t (&tmpInAxes)[Transpose::TP_GATHER_MAX_AXIS_NUM],
                       const int64_t (&tmpOutAxes)[Transpose::TP_GATHER_MAX_AXIS_NUM],
                       const int8_t (&tmpOutPerm)[Transpose::TP_GATHER_MAX_AXIS_NUM]);
    ge::graphStatus CheckUbHwConstraint();
    void MapUbCutPosToOut();
    ge::graphStatus CalcUbSplitInfo4Gather(int64_t elemInTensor, int64_t sqrtedTensor);
    void CalcUbSplitInfo4MTE();
    void CalcInSrcStrideConsecutive(int64_t dim, int8_t& axisIdx);
    void CalcInSrcStrideDefault(int64_t dim, int8_t& axisIdx);
    void CalcOutDstStride(int64_t dim, int8_t& axisIdx);
    void AdjustInUbAxesPosition();
    void ShiftInUbAxes4Borrow(int8_t outLastDimInPos, int8_t axis0Gap);
    void AdjustUbPerm4BorrowCntTwo();
    void AdjustUbPerm4BorrowCntThree();
    void AdjustUbPermCntThreeNoOverlap();
    void AdjustUbPermCntThreeOverlap();
    void AdjustUbCutAxisFactor(int32_t& axisFactor, int8_t axisFlag, int64_t elemInTensor);
    void CalcUbAxesSizes(const GatherUbPermInfo& mainUbPerm, const GatherUbPermInfo& viceUbPerm,
                         int64_t& mainUbAxesSize, int64_t& viceUbAxesSize);
    void AdjustUbCutFactorSameAxis(int32_t& axisFactor, int64_t dstInUbAxesSize, int64_t dstOutUbAxesSize,
                                   int64_t srcInUbAxesSize, int64_t srcOutUbAxesSize, int64_t elemInTensor,
                                   int64_t elemPerBlock);
    void AdjustUbCutFactorOutAxis(int32_t& axisFactor, int64_t dstInUbAxesSize, int64_t dstOutUbAxesSize,
                                  int64_t srcInUbAxesSize, int64_t srcOutUbAxesSize, int64_t elemInTensor,
                                  int64_t elemPerBlock, int8_t axisFlag);
    void AdjustUbCutFactorInAxis(int32_t& axisFactor, int64_t dstInUbAxesSize, int64_t dstOutUbAxesSize,
                                 int64_t srcInUbAxesSize, int64_t srcOutUbAxesSize, int64_t elemInTensor,
                                 int64_t elemPerBlock);
    ge::graphStatus CalcUbSplitInfo();
    ge::graphStatus CalcBlockSplitInfo();
    void CollectBlkAxes(int64_t& totalElems);
    void CalcBlkUbCutAxis(int64_t& axisFactor, int64_t cutAxisSize, int32_t cutAxisFactor, int8_t& cutPos,
                          int8_t blkAxesCnt);
    void CalcBlkAxesOffset(int8_t i, int8_t dim, int64_t axisFactor, int64_t& totalElems);
    bool CheckBC(int32_t steps);

    uint64_t tilingKey_ = static_cast<uint64_t>(TK_GATHER);
    uint32_t dataTensorSize_ = 0;
    uint32_t indexTensorSize_ = 0;
    GatherUbSplitInfo ubSplitInfo_;
    GatherBlockSplitInfo blkSplitInfo_;
    GatherUbPermInfo inUbPerm_;
    GatherUbPermInfo outUbPerm_;
    std::set<int8_t> inUbPermSet_;
    std::set<int8_t> allUbPerm_;
};

} // namespace optiling

#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSPOSE_TILING_STRATEGY_GATHER_H
