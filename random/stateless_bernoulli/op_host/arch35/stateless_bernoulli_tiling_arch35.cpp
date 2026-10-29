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
 * \file stateless_bernoulli_tiling_arch35.cpp
 * \brief
 */
#include "stateless_bernoulli_tiling_arch35.h"
#include <string>
#include "log/log.h"
#include "register/op_def_registry.h"
#include "op_host/math_tiling_templates_registry.h"
#include "util/math_util.h"
#include "../../../random_common/op_host/arch35/random_tiling_base.h"

namespace optiling {
static constexpr uint16_t INPUT_IDX_SHAPE = 0;
static constexpr uint16_t INPUT_IDX_PROB = 1;
static constexpr uint16_t INPUT_IDX_SEED = 2;
static constexpr uint16_t INPUT_IDX_OFFSET = 3;
static constexpr uint16_t OUTPUT_IDX_Y = 0;
static constexpr uint16_t ATTR_IDX_PROB_MODE = 1; // def 属性声明顺序：dtype(0), probMode(1)
static constexpr int64_t DCACHE_SIZE = 128 * 1024;
static constexpr int64_t CORE_MINIEST_NUM = 256;
static constexpr int64_t OFFSET_MULTIPLE = 4;
static constexpr int64_t PROB_MODE_TENSOR = 0;
static constexpr int64_t PROB_MODE_SCALAR = 1;
constexpr uint64_t TILING_KEY_SCALAR = 100;
constexpr uint64_t TILING_KEY_TENSOR_DENSE = 101;
constexpr uint64_t TILING_KEY_TENSOR_BCAST = 102;
static constexpr int32_t MAX_BROADCAST_DIMS = 8; // probSizes/probStrides 定长

OpTilingConfig StatelessBernoulliTiling::BuildOpConfig()
{
    OpTilingConfig config;

    config.inputCheckRules = {
        {INPUT_IDX_SHAPE, {{ge::DT_INT32, ge::DT_INT64}, -1, {1}, nullptr}},
        {INPUT_IDX_PROB, {{ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16}, -1, {}, nullptr}},
        {INPUT_IDX_SEED, {{ge::DT_INT64}, 1, {}, nullptr}},
        {INPUT_IDX_OFFSET, {{ge::DT_INT64}, 1, {}, nullptr}},
    };
    config.outputCheckRules = {
        {OUTPUT_IDX_Y,
         {{ge::DT_INT8, ge::DT_UINT8, ge::DT_INT16, ge::DT_UINT16, ge::DT_INT32, ge::DT_UINT32, ge::DT_INT64,
           ge::DT_UINT64, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BOOL},
          -1,
          {},
          nullptr}},
    };

    config.getOutputSize = [](gert::TilingContext* ctx, int64_t& size) {
        return RandomUtils::GetAndCheckOutputSize<INPUT_IDX_SHAPE, OUTPUT_IDX_Y, false>(ctx, size);
    };

    config.getSeedAndOffset = [](gert::TilingContext* ctx, int64_t& seed, int64_t& offset) {
        auto seedTensor = ctx->GetRequiredInputTensor(INPUT_IDX_SEED);
        OP_CHECK_NULL_WITH_CONTEXT(ctx, seedTensor);
        const int64_t* seedVal = seedTensor->GetData<int64_t>();
        OP_CHECK_NULL_WITH_CONTEXT(ctx, seedVal);
        seed = *seedVal;
        auto offsetTensor = ctx->GetRequiredInputTensor(INPUT_IDX_OFFSET);
        OP_CHECK_NULL_WITH_CONTEXT(ctx, offsetTensor);
        const int64_t* offsetVal = offsetTensor->GetData<int64_t>();
        OP_CHECK_NULL_WITH_CONTEXT(ctx, offsetVal);
        offset = *offsetVal;
        if (offset % OFFSET_MULTIPLE != 0) {
            std::string valueStr = std::to_string(offset);
            std::string reasonMsg = "offset value must be a multiple of 4";
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ctx->GetNodeName(), "input offset", valueStr.c_str(),
                                                  reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    };

    config.kernelMode = RandomKernelMode::SIMT;
    // scalar 路径经 splitBlocks 取执行参数, T=256*min(ceil(n/256),78*8),
    // 超大 shape 时 kernelOffset 按分块累计
    config.enableSplitBlocks = true;
    // philox 每次 counter 调用产出 4 个随机分量(unroll 4), 与 kernel 侧 SIMT_STEP 对齐
    constexpr uint32_t PHILOX_UNROLL_FACTOR = 4;
    config.unrollFactor = PHILOX_UNROLL_FACTOR;
    config.DcacheSize = DCACHE_SIZE;
    config.isNeedSyncAll = false;
    return config;
}

// 一趟扫描状态机: 从最内维向外逐维计算 prob 步长并做连续轴合并
// (对标竞品 TensorIterator::coalesce_dimensions)。段表按内→外序暂存于
// segSizes/segStrides, 段数经 segCnt 传出。
ge::graphStatus StatelessBernoulliTiling::ScanAndCoalesce(const gert::Shape& outputDims, const gert::Shape& probShape,
                                                          int64_t segSizes[], int64_t segStrides[], int32_t& segCnt,
                                                          bool& hasBroadcastDim)
{
    int64_t segSize = 0; // 当前合并段 size(0=尚无段)
    int64_t segStride = 0;
    int64_t probStride = 1;
    int64_t probIdx = static_cast<int32_t>(probShape.GetDimNum()) - 1;
    segCnt = 0;
    const int32_t outDimNum = static_cast<int32_t>(outputDims.GetDimNum());
    for (int32_t i = outDimNum - 1; i >= 0; --i) {
        const int64_t outDim = outputDims.GetDim(i);
        int64_t curStride = 0;
        if (probIdx >= 0) {
            const int64_t probDim = probShape.GetDim(probIdx);
            if (probDim == 1 && outDim != 1) {
                curStride = 0; // 广播维
                hasBroadcastDim = true;
            } else if (probDim == outDim) {
                curStride = probStride; // 连续维
                probStride *= probDim;
            } else {
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context_->GetNodeName(), "prob", std::to_string(probDim).c_str(),
                    "prob shape is not broadcastable to output shape at this dimension");
                return ge::GRAPH_FAILED;
            }
            --probIdx;
        } // 前导广播维 curStride = 0
        if (segSize == 0) {
            segSize = outDim; // 开新段
            segStride = curStride;
        } else if (curStride == segSize * segStride || outDim == 1) {
            segSize *= outDim; // 并入当前段(坐标折叠, 段内侧 stride 不变)
        } else if (segSize == 1) {
            segSize = outDim; // 段 size1 坐标恒 0, 段退化为本维(stride 取本维)
            segStride = curStride;
        } else {
            segSizes[segCnt] = segSize; // 段边界: 暂存当前段(内→外序)
            segStrides[segCnt] = segStride;
            ++segCnt;
            segSize = outDim;
            segStride = curStride;
        }
    }
    if (segSize != 0) { // 落盘最外段
        segSizes[segCnt] = segSize;
        segStrides[segCnt] = segStride;
        ++segCnt;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StatelessBernoulliTiling::ComputeProbBroadcastTable(const gert::Shape& outputDims,
                                                                    const gert::Shape& probShape, bool& hasBroadcastDim)
{
    // probSizes/probStrides 定长 8: 超 8 维会越界写 tiling 并在 kernel 侧除零, 须显式校验拒绝
    const int32_t outDimNum = static_cast<int32_t>(outputDims.GetDimNum());
    const int32_t probDimNum = static_cast<int32_t>(probShape.GetDimNum());
    if (outDimNum > MAX_BROADCAST_DIMS) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "y", std::to_string(outDimNum).c_str(),
                                              "output dim num must not exceed 8 for prob broadcast addressing");
        return ge::GRAPH_FAILED;
    }
    if (probDimNum > outDimNum) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "prob", std::to_string(probDimNum).c_str(),
                                              "prob rank must not exceed output rank for broadcast");
        return ge::GRAPH_FAILED;
    }
    int64_t segSizes[MAX_BROADCAST_DIMS] = {0};
    int64_t segStrides[MAX_BROADCAST_DIMS] = {0};
    int32_t segCnt = 0;
    hasBroadcastDim = (probDimNum != outDimNum); // 前导广播或 rank 不一致即非 dense
    if (ScanAndCoalesce(outputDims, probShape, segSizes, segStrides, segCnt, hasBroadcastDim) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    simtTilingData_.probDims = segCnt;
    for (int32_t i = 0; i < segCnt; ++i) { // 反转为外→内序, 左对齐紧凑写入
        simtTilingData_.probSizes[i] = segSizes[segCnt - 1 - i];
        simtTilingData_.probStrides[i] = segStrides[segCnt - 1 - i];
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StatelessBernoulliTiling::DoSimtBlockTiling()
{
    OP_CHECK_IF((totalCoreNum_ <= 0), OP_LOGE(opName_, "totalCoreNum is %ld, must be greater than 0.", totalCoreNum_),
                return ge::GRAPH_FAILED);

    // probMode: 读取显式属性 (1=标量, 0=张量), 缺省为张量语义
    // 注意与 def 属性声明顺序耦合: dtype(0), probMode(1)（gert 属性仅支持按下标读取）。
    // 场景判定经类成员中转供 UniqueProcess 设 tilingKey, 不落 tilingData(kernel 不消费)
    const auto* probModeAttr = context_->GetAttrs()->GetAttrPointer<int64_t>(ATTR_IDX_PROB_MODE);
    isScalarProb_ = (probModeAttr != nullptr) && (*probModeAttr == PROB_MODE_SCALAR);
    // splitBlocks 仅被标量分支消费; 张量为单 launch + kernel 内自算 T, 关闭开关跳过
    // 基类分块计算, 避免极端 shape（>8 分块）误报错
    config_.enableSplitBlocks = isScalarProb_;

    auto probTensor = context_->GetRequiredInputTensor(INPUT_IDX_PROB);
    OP_CHECK_NULL_WITH_CONTEXT(context_, probTensor);
    auto outputShape = context_->GetOutputShape(OUTPUT_IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputShape);
    auto outputDims = outputShape->GetStorageShape();
    auto probShape = probTensor->GetStorageShape();

    if (isScalarProb_) {
        // 标量语义: kernel 仅读 prob 首元素, prob 的 rank/shape 与输出无关(含输出 rank 0),
        // 不做广播校验; 仅校验 def 层约束: probMode=1 时 prob 必须为单元素张量
        OP_CHECK_IF(probShape.GetShapeSize() != 1,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "prob",
                                                          std::to_string(probShape.GetShapeSize()).c_str(),
                                                          "prob must be a single-element tensor when probMode is 1"),
                    return ge::GRAPH_FAILED);
    } else {
        bool hasBroadcastDim = false;
        // 失败分支已在 ComputeProbBroadcastTable 内打具体错误日志, 此处仅透传结果
        if (ComputeProbBroadcastTable(outputDims, probShape, hasBroadcastDim) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        // 非广播（prob 与输出逐维同 shape）时 stride 寻址退化为直接索引，走 dense 快路径；
        // 判定结果经类成员中转供 UniqueProcess 设 tilingKey，不落 tilingData
        hasBroadcastDim_ = hasBroadcastDim;
    }

    // 公共尾部: 参与执行的核数
    int64_t avgPerCore = Ops::Base::CeilDiv(simtTilingData_.outputSize, totalCoreNum_);
    int64_t numOfPerCore = Ops::Base::CeilAlign(avgPerCore, CORE_MINIEST_NUM);
    int64_t usedCoreNum = Ops::Base::CeilDiv(simtTilingData_.outputSize, numOfPerCore);
    simtTilingData_.usedCoreNum = std::max(1L, std::min(usedCoreNum, totalCoreNum_));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StatelessBernoulliTiling::UniqueProcess()
{
    // 场景化 tilingKey（须在步骤 6 设置：基类 CalcTilingKeyAndWorkspace 步骤 5 无条件重置为 100）
    // 100=标量(交织), 101=张量非广播(直接索引), 102=张量广播(stride 寻址), 供 ST/日志按场景观测
    if (isScalarProb_) {
        tilingKey_ = TILING_KEY_SCALAR;
    } else if (!hasBroadcastDim_) {
        tilingKey_ = TILING_KEY_TENSOR_DENSE;
    } else {
        tilingKey_ = TILING_KEY_TENSOR_BCAST;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4StatelessBernoulli(gert::TilingContext* context)
{
    StatelessBernoulliTiling tiling(context);
    return tiling.DoTiling();
}

static ge::graphStatus TilingPrepare4StatelessBernoulli(gert::TilingParseContext* context)
{
    return RandomTilingParseArch35(context, "StatelessBernoulli");
}

IMPL_OP_OPTILING(StatelessBernoulli)
    .Tiling(Tiling4StatelessBernoulli)
    .TilingParse<RandomOperatorCompileInfo>(TilingPrepare4StatelessBernoulli)
    .TilingInputsDataDependency({INPUT_IDX_SHAPE, INPUT_IDX_SEED, INPUT_IDX_OFFSET});
} // namespace optiling
