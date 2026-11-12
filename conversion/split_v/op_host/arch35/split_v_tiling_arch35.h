/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SPLIT_V_TILING_H
#define SPLIT_V_TILING_H
#include <cstdint>
#include "op_host/tiling_base_class.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

constexpr int64_t BASE_2 = 2;
constexpr int64_t MAX_CORE_COUNT = 72;
constexpr int32_t MAX_COL_OFFSET_COUNT = 128;

// 等长切分(same len)快速路径的精简tiling数据, 由tilingKey(100/101)区分, host与kernel经框架序列化传递
BEGIN_TILING_DATA_DEF(SplitVSlimTilingData)
TILING_DATA_FIELD_DEF(int64_t, ubSize);
TILING_DATA_FIELD_DEF(int64_t, gSize);
TILING_DATA_FIELD_DEF(int64_t, nBlockFactorNum);
TILING_DATA_FIELD_DEF(int64_t, mBlockFactor);
TILING_DATA_FIELD_DEF(int64_t, mBlockFactorTail);
TILING_DATA_FIELD_DEF(int64_t, mBlockCount);
TILING_DATA_FIELD_DEF(int64_t, gUBFactor);
TILING_DATA_FIELD_DEF(int64_t, gUBFactorTail);
TILING_DATA_FIELD_DEF(int64_t, gUBCount);
TILING_DATA_FIELD_DEF(int64_t, nBlockFactor);
TILING_DATA_FIELD_DEF(int64_t, nBlockFactorTail);
TILING_DATA_FIELD_DEF(int64_t, nBlockCount);
TILING_DATA_FIELD_DEF(int64_t, blockFactor);
TILING_DATA_FIELD_DEF(int64_t, blockFactorTail);
TILING_DATA_FIELD_DEF(int64_t, realCoreNum);
TILING_DATA_FIELD_DEF(int64_t, mBlockFactorNum);
END_TILING_DATA_DEF;
// tilingKey: 100 等长纯搬运模板, 101 等长UB内split模板
REGISTER_TILING_DATA_CLASS(SplitV_100, SplitVSlimTilingData)
REGISTER_TILING_DATA_CLASS(SplitV_101, SplitVSlimTilingData)
REGISTER_TILING_DATA_CLASS(Split_100, SplitVSlimTilingData)
REGISTER_TILING_DATA_CLASS(Split_101, SplitVSlimTilingData)

BEGIN_TILING_DATA_DEF(SplitVTilingData)
TILING_DATA_FIELD_DEF(int64_t, ubSize);            // asc 接口获取到的ub大小，Bytes
TILING_DATA_FIELD_DEF(int64_t, splitDim);          // attr里带的原始的split的轴, 负数转正数
TILING_DATA_FIELD_DEF(int64_t, sizeAfterSplitDim); // split轴之后所有shape的乘积, 个数
TILING_DATA_FIELD_DEF(int64_t, mBlockFactor);      // m轴block切分factor, 个数  --对应Split: mUBFactor
TILING_DATA_FIELD_DEF(int64_t, mBlockFactorTail);  // m轴切分factor尾快, 个数   --对应Split: mUBFactorTail
TILING_DATA_FIELD_DEF(int64_t, mBlockFactorNum);   // m轴主块的个数             --对应Split: mSize
TILING_DATA_FIELD_DEF(int64_t, mBlockCount);       // m轴切分块的总个数         --对应Split: mUBCount
TILING_DATA_FIELD_DEF(int64_t, gUBFactor);         // Split: Gi
TILING_DATA_FIELD_DEF(int64_t, gUBFactorTail);     // Split: GiTail
TILING_DATA_FIELD_DEF(int64_t, gSize);             // Split: gSize
TILING_DATA_FIELD_DEF(int64_t, gUBCount);          // Split: Go
TILING_DATA_FIELD_DEF(int64_t, nBlockFactor);      // n轴block切分factor, 个数  --对应Split: nUBFactor
TILING_DATA_FIELD_DEF(int64_t, nBlockFactorAlign); // n轴block切分factor对齐到32字节, 个数  --对应Split: nUBFactorAlign
TILING_DATA_FIELD_DEF(int64_t, nBlockFactorTail); // n轴切分factor尾快, 个数     --对应Split: nUBFactorTail
TILING_DATA_FIELD_DEF(int64_t,
                      nBlockFactorTailAlign); // n轴切分factor尾快对齐到32字节, 个数   --对应Split: nUBFactorTailAlign
TILING_DATA_FIELD_DEF(int64_t, nBlockFactorNum); // n轴主块的个数               --对应Split: nSize
TILING_DATA_FIELD_DEF(int64_t, nBlockCount);     // n轴切分块的总个数           --对应Split: nUBCount
TILING_DATA_FIELD_DEF(int64_t, realCoreNum);     // 真实使用的核数
TILING_DATA_FIELD_DEF(int64_t, blockFactor);     // Split: blockFactor
TILING_DATA_FIELD_DEF(int64_t, blockFactorTail); // Split: blockFactorTail
TILING_DATA_FIELD_DEF(int64_t, nSize);           // 整个n轴的大小, 个数
TILING_DATA_FIELD_DEF(int64_t, nSizeAlign);      // 整个n轴的大小, 对齐到32字节, 个数
TILING_DATA_FIELD_DEF(int64_t, pureOutIdx);      // split个数为1的纯搬运场景，真实的输出的idx
TILING_DATA_FIELD_DEF(int64_t, negIdx);          // sizeSplits为负数的索引
TILING_DATA_FIELD_DEF(int64_t, negValue);        // sizeSplits为负数的实际split轴大小
TILING_DATA_FIELD_DEF(int64_t, isInt32);         // sizeSplits是否为int32类型
TILING_DATA_FIELD_DEF(int64_t, mSize);           // 整个m轴的大小, 个数
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_COUNT,
                          nBlockSplitOffset); // 所有核在N轴上每个block起始位置在第几个Split块上, 从0开始
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_COUNT, nBlockSplitOffsetEnd); // end - start = 处理的split块数, 至少为1
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_COUNT,
                          nBlockSplitPrefixStart); // 所有核在N轴上每个block起始位置在第几个Split块上, 从0开始
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_COUNT, nBlockSplitPrefixEnd); // end - start = 处理的split块数, 至少为1
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(SplitV, SplitVTilingData)
REGISTER_TILING_DATA_CLASS(Split, SplitVTilingData)

// tilingKey=102(纯搬运模板)专用tiling数据, 仅包含 SplitVPureCopyMode 模板(op_kernel/arch35/split_v_pure_copy.h)
// 实际访问的成员, 字段顺序与 SplitVTilingData 保持一致
BEGIN_TILING_DATA_DEF(SplitVPureCopyTilingData)
TILING_DATA_FIELD_DEF(int64_t, ubSize);            // asc 接口获取到的ub大小，Bytes
TILING_DATA_FIELD_DEF(int64_t, sizeAfterSplitDim); // split轴之后所有shape的乘积, 个数
TILING_DATA_FIELD_DEF(int64_t, mBlockFactor);      // m轴block切分factor, 个数
TILING_DATA_FIELD_DEF(int64_t, mBlockFactorTail);  // m轴切分factor尾快, 个数
TILING_DATA_FIELD_DEF(int64_t, mBlockFactorNum);   // m轴主块的个数
TILING_DATA_FIELD_DEF(int64_t, gSize);             // Split: gSize
TILING_DATA_FIELD_DEF(int64_t, nBlockFactor);      // n轴block切分factor, 个数
TILING_DATA_FIELD_DEF(int64_t, nBlockFactorTail);  // n轴切分factor尾快, 个数
TILING_DATA_FIELD_DEF(int64_t, nBlockFactorNum);   // n轴主块的个数
TILING_DATA_FIELD_DEF(int64_t, nBlockCount);       // n轴切分块的总个数
TILING_DATA_FIELD_DEF(int64_t, realCoreNum);       // 真实使用的核数
TILING_DATA_FIELD_DEF(int64_t, negIdx);            // sizeSplits为负数的索引
TILING_DATA_FIELD_DEF(int64_t, negValue);          // sizeSplits为负数的实际split轴大小
TILING_DATA_FIELD_DEF(int64_t, nSize);             // 整个n轴的大小, 个数
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_COUNT,
                          nBlockSplitOffset); // 所有核在N轴上每个block起始位置在第几个Split块上, 从0开始
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_COUNT, nBlockSplitOffsetEnd); // end - start = 处理的split块数, 至少为1
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_COUNT,
                          nBlockSplitPrefixStart); // 所有核在N轴上每个block起始位置在第几个Split块上, 从0开始
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(SplitV_102, SplitVPureCopyTilingData)

// tilingKey=103(UB内split模板)专用tiling数据, 仅包含 SplitVUbSplit 模板(op_kernel/arch35/split_v_ub_split.h)
// 实际访问的成员, 字段顺序与 SplitVTilingData 保持一致
BEGIN_TILING_DATA_DEF(SplitVUbSplitTilingData)
TILING_DATA_FIELD_DEF(int64_t, ubSize);            // asc 接口获取到的ub大小，Bytes
TILING_DATA_FIELD_DEF(int64_t, sizeAfterSplitDim); // split轴之后所有shape的乘积, 个数
TILING_DATA_FIELD_DEF(int64_t, mBlockFactor);      // m轴block切分factor, 个数
TILING_DATA_FIELD_DEF(int64_t, mBlockFactorTail);  // m轴切分factor尾快, 个数
TILING_DATA_FIELD_DEF(int64_t, mBlockFactorNum);   // m轴主块的个数
TILING_DATA_FIELD_DEF(int64_t, gSize);             // Split: gSize
TILING_DATA_FIELD_DEF(int64_t, nBlockFactor);      // n轴block切分factor, 个数
TILING_DATA_FIELD_DEF(int64_t, nBlockFactorAlign); // n轴block切分factor对齐到32字节, 个数
TILING_DATA_FIELD_DEF(int64_t, nBlockFactorTail);  // n轴切分factor尾快, 个数
TILING_DATA_FIELD_DEF(int64_t, nBlockFactorNum);   // n轴主块的个数
TILING_DATA_FIELD_DEF(int64_t, nBlockCount);       // n轴切分块的总个数
TILING_DATA_FIELD_DEF(int64_t, realCoreNum);       // 真实使用的核数
TILING_DATA_FIELD_DEF(int64_t, negIdx);            // sizeSplits为负数的索引
TILING_DATA_FIELD_DEF(int64_t, negValue);          // sizeSplits为负数的实际split轴大小
TILING_DATA_FIELD_DEF(int64_t, isInt32);           // sizeSplits是否为int32类型
TILING_DATA_FIELD_DEF(int64_t, nSize);             // 整个n轴的大小, 个数
TILING_DATA_FIELD_DEF(int64_t, mSize);             // 整个m轴的大小, 个数
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_COUNT,
                          nBlockSplitOffset); // 所有核在N轴上每个block起始位置在第几个Split块上, 从0开始
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_COUNT, nBlockSplitOffsetEnd); // end - start = 处理的split块数, 至少为1
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_COUNT,
                          nBlockSplitPrefixStart); // 所有核在N轴上每个block起始位置在第几个Split块上, 从0开始
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_COUNT, nBlockSplitPrefixEnd); // end - start = 处理的split块数, 至少为1
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(SplitV_103, SplitVUbSplitTilingData)

// tilingKey=104/111/112 共用的纯标量tiling数据(无数组字段), 为三个模板实际访问成员的并集:
// 104(SplitVPureCopySpecialMode, split_v)、111(SplitVUbSplitSameLenSmallG, split/unpack)、
// 112(SplitVUbSplitSameLenDeinterleave, split/unpack)。字段顺序与 SplitVTilingData 保持一致。
BEGIN_TILING_DATA_DEF(SplitVScalarTilingData)
TILING_DATA_FIELD_DEF(int64_t, ubSize);            // asc 接口获取到的ub大小，Bytes
TILING_DATA_FIELD_DEF(int64_t, splitDim);          // attr里带的原始的split的轴, 负数转正数
TILING_DATA_FIELD_DEF(int64_t, sizeAfterSplitDim); // split轴之后所有shape的乘积, 个数
TILING_DATA_FIELD_DEF(int64_t, mBlockFactor);      // m轴block切分factor, 个数
TILING_DATA_FIELD_DEF(int64_t, mBlockFactorTail);  // m轴切分factor尾快, 个数
TILING_DATA_FIELD_DEF(int64_t, mBlockCount);       // m轴切分块的总个数
TILING_DATA_FIELD_DEF(int64_t, gUBFactor);         // Split: Gi
TILING_DATA_FIELD_DEF(int64_t, gUBFactorTail);     // Split: GiTail
TILING_DATA_FIELD_DEF(int64_t, gSize);             // Split: gSize
TILING_DATA_FIELD_DEF(int64_t, gUBCount);          // Split: Go
TILING_DATA_FIELD_DEF(int64_t, nBlockFactor);      // n轴block切分factor, 个数
TILING_DATA_FIELD_DEF(int64_t, nBlockFactorTail);  // n轴切分factor尾快, 个数
TILING_DATA_FIELD_DEF(int64_t, nBlockFactorNum);   // n轴主块的个数
TILING_DATA_FIELD_DEF(int64_t, nBlockCount);       // n轴切分块的总个数
TILING_DATA_FIELD_DEF(int64_t, realCoreNum);       // 真实使用的核数
TILING_DATA_FIELD_DEF(int64_t, blockFactor);       // Split: blockFactor
TILING_DATA_FIELD_DEF(int64_t, blockFactorTail);   // Split: blockFactorTail
TILING_DATA_FIELD_DEF(int64_t, pureOutIdx);        // split个数为1的纯搬运场景，真实的输出的idx
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(SplitV_104, SplitVScalarTilingData)
REGISTER_TILING_DATA_CLASS(SplitV_111, SplitVScalarTilingData)
REGISTER_TILING_DATA_CLASS(SplitV_112, SplitVScalarTilingData)
REGISTER_TILING_DATA_CLASS(Split_111, SplitVScalarTilingData)
REGISTER_TILING_DATA_CLASS(Split_112, SplitVScalarTilingData)

BEGIN_TILING_DATA_DEF(SplitVSIMTTilingData)
TILING_DATA_FIELD_DEF(int32_t, mSize); // 整个m轴的大小, 元素个数
TILING_DATA_FIELD_DEF(int32_t, nSize); // split轴之后所有shape的乘积, 个数
TILING_DATA_FIELD_DEF(int32_t, sizeAfterSplit);
TILING_DATA_FIELD_DEF(int32_t, splitNum);
TILING_DATA_FIELD_DEF(int32_t, realCoreNum); // 真实使用的核数
TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_COL_OFFSET_COUNT, colOffset);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(SplitV_200, SplitVSIMTTilingData)
REGISTER_TILING_DATA_CLASS(SplitV_202, SplitVSIMTTilingData)

BEGIN_TILING_DATA_DEF(SplitSIMTTilingData)
TILING_DATA_FIELD_DEF(int32_t, mSize); // 整个m轴的大小, 元素个数
TILING_DATA_FIELD_DEF(int32_t, nSize); // split轴之后所有shape的乘积, 个数
TILING_DATA_FIELD_DEF(int32_t, splitSize);
TILING_DATA_FIELD_DEF(int32_t, splitNum);
TILING_DATA_FIELD_DEF(int32_t, realCoreNum);         // 真实使用的核数
TILING_DATA_FIELD_DEF(int32_t, tensorSplitNum);      // tensor切分的块数
TILING_DATA_FIELD_DEF(int32_t, mainTensorSplitSize); // tensor切分主块大小
TILING_DATA_FIELD_DEF(int32_t, tailTensorSplitSize); // tensor切分尾块大小
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(SplitV_201, SplitSIMTTilingData)
REGISTER_TILING_DATA_CLASS(Split_201, SplitSIMTTilingData)
REGISTER_TILING_DATA_CLASS(SplitV_203, SplitSIMTTilingData)
REGISTER_TILING_DATA_CLASS(Split_203, SplitSIMTTilingData)

ge::graphStatus SplitVTilingAscendC(gert::TilingContext* context, int32_t maxCoreNum, uint32_t ubSize,
                                    int32_t isSameLen);

struct DualSplitNode {
    int64_t m = 0;     // M轴切分块数
    int64_t n = 0;     // N轴切分块数
    int64_t t = 0;     // m*n, 实际使用核数
    int64_t delta = 0; // 最大块和最小块面积的差值, 仅作观测和最后兜底

    // 以下字段由 host 侧成本估算填充, 用于以 kernel 真实 UB 循环次数为核心的候选选择
    int64_t mTimes = 0;       // 最重核在 M 方向的 UB 循环次数
    int64_t nTimes = 0;       // 最重核在 N 方向的 UB 循环次数
    int64_t maxSplitCnt = 0;  // 单个 N block 覆盖的最大 split 块数
    int64_t computeCost = 0;  // 综合计算成本 = mTimes * (maxSplitCnt + nTimes), 主排序键
    int64_t maxDataBytes = 0; // 最重核的数据量, 仅作极端失衡保护

    DualSplitNode() {};
    DualSplitNode(int64_t mV, int64_t nV, int64_t tV, int64_t deltaV) : m(mV), n(nV), t(tV), delta(deltaV) {}

    // 计算瓶颈是每核 UB 循环放大后的 split 处理量, 因此以 computeCost 为主排序键。
    // 数据量随硬件带宽提升不再是主瓶颈, 只在 ChooseBestSplitInfo 中作为极端失衡过滤, 不参与主评分。
    // 排序顺序: computeCost 小 -> 用满核(t 大) -> maxSplitCnt 小 -> maxDataBytes 小 -> n 小(保守少切 N)。
    bool operator<(const DualSplitNode& x) const
    {
        if (computeCost != x.computeCost) {
            return computeCost < x.computeCost;
        }
        if (t != x.t) {
            return t > x.t;
        }
        if (maxSplitCnt != x.maxSplitCnt) {
            return maxSplitCnt < x.maxSplitCnt;
        }
        if (maxDataBytes != x.maxDataBytes) {
            return maxDataBytes < x.maxDataBytes;
        }
        return n < x.n;
    }
};

struct SplitVCompileInfo {
    int64_t core_num;
    uint32_t maxCoreNum{0};
    uint32_t ubSizePlatform{0};
};

class SplitVTiling {
public:
    explicit SplitVTiling(gert::TilingContext* tilingContext) : context_(tilingContext) {};
    ge::graphStatus DoSplitVTiling(int32_t maxCoreNum, uint32_t ubSize);
    ge::graphStatus DoSplitTiling(int32_t maxCoreNum, uint32_t ubSize);
    ge::graphStatus DoSplitVSIMTTiling(int32_t maxCoreNum);

protected:
    virtual ge::graphStatus InitParamsSameLen(int32_t maxCoreNum, uint32_t ubSize);
    virtual ge::graphStatus GetInputParamsSameLen();

    int64_t splitDim_ = 0;
    int64_t numSplit_ = 0;
    int32_t coreNum_ = 0;
    uint32_t ubSize_ = 0;
    gert::TilingContext* context_{nullptr};
    int64_t xDtypeSize_ = 0;
    gert::Shape inputShape_;

private:
    template <typename T>
    std::string ArrayToString(const T* vec, size_t num) const;
    template <typename T>
    std::string VectorToString(const std::vector<T>& vec) const;
    ge::graphStatus ModifySizeSplitList();
    template <typename T>
    bool GetData(const gert::Tensor* tensor, std::vector<int64_t>& values) const;
    bool GetSizeSplitsList();
    ge::graphStatus GetInputParams();
    ge::graphStatus InitParams(int32_t maxCoreNum, uint32_t ubSize);
    void FuseAllShape();
    void SetPureMoveTilingMode();
    void SetBlockSplitInfo(int64_t mBlockCnt, int64_t nBlockCnt);
    void FuseInputShape();
    void CalEmptyInputTiling();
    void FindAllPossibleCutCnt(std::set<int64_t>& cutCountSet, int64_t cores) const;
    void GetAllCutInfoNode(int64_t M, int64_t N, int64_t cores, const std::set<int64_t>& cutCountSet,
                           std::vector<DualSplitNode>& splitNodeList) const;
    void EstimateSplitNodeCost(int64_t dataBytes, DualSplitNode& node) const;
    int64_t CalcMaxSplitCntPerNBlock(int64_t nBlockCnt) const;
    void ChooseBestSplitInfo(std::vector<DualSplitNode>& splitNodeList, DualSplitNode& splitInfo) const;
    void CalBlockSplitTwoAxis(int64_t rowM, int64_t colN, int64_t dataBytes, int64_t coreNum,
                              DualSplitNode& splitInfo) const;
    void CalBlockTilingParams();
    void CalcSplitLenCond(int64_t curSplitNumN, int64_t& condNum, int64_t& total) const;
    void CalcSplitUBSizeCond(int64_t curSplitNumN, int64_t& condNum, int64_t& total);
    int64_t UpdataNextBlockFactor(int64_t sizeSplitIdx, int64_t handedSplitSize, int64_t leftSplitSize,
                                  int64_t& handedBlockNumN);
    void UpdateMAxisSplitOffset();
    void CalcTilingKey(double condN, double condM, double totalCompareN, double totalCompareM);
    void SetTilingMode();
    void FillTilingData();
    void FillSlimTilingData();
    void FillPureCopyTilingData();
    void FillUbSplitTilingData();
    void FillScalarTilingData();
    void SetBlockDimAndTilingKey() const;
    void SetWorkspaceSize() const;
    // Used for split
    int64_t CeilDiv(int64_t value, int64_t factor) const;
    int64_t FloorAlign(int64_t value, int64_t factor) const;
    void FuseInputShapeSameLen();
    int64_t AdjustUbFactor(int64_t factor, int64_t alignFactor) const;
    void CalcSameLenCopyTilingInfo(int64_t halfUbEleNum, int64_t blockEleNum);
    void CalcSameLenSplitTilingInfo(int64_t halfUbEleNum, int64_t blockEleNum);
    void CalcSameLenTilingInfo();
    void FillSIMTSplitVTilingData();
    int64_t CalInputDataSize();
    void CountSplitPrefix();
    int64_t SplitPrefix(int64_t i);
    bool IsDoSplitVSIMT(int32_t maxCoreNum);
    void DoDiffLenModeSplitVSIMTTiling(int32_t maxCoreNum);
    void CalSplitSizeDiff(int32_t maxCoreNum, int64_t& maxSizeSplit, int64_t& minSizeSplit, int32_t countPerCore);

private:
    int64_t nonZeroSplitCnt_ = 0;
    int64_t pureOutIdx_ = -1;
    bool isSameLenMode_ = true;

    int64_t splitStride_ = 0; // 切分轴右侧的所有轴的乘积 --对应Split: splitDimFactor_
    int64_t fusedShape_[BASE_2] = {0};

    int64_t mSimtSize_ = 1;
    int64_t nSimtSize_ = 1;
    int64_t simtSizeAfterSplit_ = 0;
    int64_t maxFuseSizeAfterSplit_ = 0;
    int64_t minFuseSizeAfterSplit_ = 0;
    int64_t realCoreNum_ = 0; // 处理空tensor需要两个变量，realCoreNum_ 和 blockDim_分开赋值
    int64_t blockDim_ = 0;
    int64_t blockFactor_ = 0;
    int64_t blockFactorTail_ = 0;

    int64_t mBlockCount_ = 0;  // M轴方向核切分块数
    int64_t mBlockFactor_ = 0; // M轴核切分的 main_factor -- tail_factor=main_factor-1
    int64_t mBlockFactorTail_ = 0;
    int64_t mBlockFactorCount_ = 0; // M轴核切分的主块个数 -- 尾块个数=mBlockCount_ - mBlockFactorCount_
    int64_t gUBCount_ = 0;
    int64_t gUBFactor_ = 0;
    int64_t gUBFactorTail_ = 0;
    int64_t nBlockCount_ = 0;
    int64_t nBlockFactor_ = 0;
    int64_t nBlockFactorTail_ = 0;
    int64_t nBlockFactorCount_ = 0;
    int64_t tilingKey_ = 0;
    int64_t negIdx_ = -1;
    int64_t negValue_ = 0;
    int64_t isInt32_ = 0;

    int64_t nBlockSplitOffsetStart_[MAX_CORE_COUNT] = {
        0}; // 所有核在N轴上每个blockfactor起始位置在第几个Split块上, 从idx 0开始
    int64_t nBlockSplitOffsetEnd_[MAX_CORE_COUNT] = {0}; // end - start = 处理的split块数
    int64_t nBlockSplitPrefixStart_[MAX_CORE_COUNT] = {
        0}; // 所有核在N轴上每个blockfactor起始位置所在split块的split轴偏移
    int64_t nBlockSplitPrefixEnd_[MAX_CORE_COUNT] = {0}; // 所有核在N轴上每个blockfactor结束位置所在split块的split轴偏移

    SplitVTilingData tilingData_;
    SplitVSlimTilingData slimTilingData_;
    SplitVPureCopyTilingData pureCopyTilingData_;
    SplitVUbSplitTilingData ubSplitTilingData_;
    SplitVScalarTilingData scalarTilingData_;
    SplitVSIMTTilingData simtSplitVTilingData_;
    std::vector<int64_t> oriSizeSplits_;
    std::vector<int64_t> sizeSplits_; // 合轴后的
    uint32_t colOffset_[MAX_COL_OFFSET_COUNT] = {0};
};

} // namespace optiling
#endif
