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
 * \file radix_top_k_tiling.cpp
 * \brief
 */

#include <cstdint>
#include <limits>
#include "tiling/tiling_api.h"
#include "register/tilingdata_base.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "log/log.h"
#include "radix_top_k_tiling.h"
#include "../op_kernel/radix_top_k_struct.h"
#include "../op_kernel/radix_top_k_tiling_key.h"

namespace optiling {
namespace RadixTopK {
static constexpr uint64_t BLOCK_SIZE = 32;
static constexpr uint64_t CACHE_LINE = 512;
static constexpr uint64_t MAX_TILE_LEN = 12032;
static constexpr uint64_t LARGE_SORT_LEN = 1440000000;
static constexpr uint64_t LARGE_TILE_LEN = 10496;
static constexpr uint64_t MAX_TILE_NUM_IN_UB = 6144;
// sortLen 超过 int32 上限时切换 int64 模板（索引值域与跨核计数可达 sortLen）
static constexpr uint64_t INT32_SORT_LEN_MAX = static_cast<uint64_t>(std::numeric_limits<int32_t>::max());
// isInt64Shape_（sortLen > INT32_SORT_LEN_MAX）必须蕴含 isLargeShape_（sortLen > LARGE_SORT_LEN）：
// needWorkspace / GetWorkspaceSize 两处都写成 `isLargeShape_ || isInt64Shape_`（后者冗余）。这里用
// static_assert 把两个常量的相对大小钉死，避免以后调大 LARGE_SORT_LEN 时 i64 分支被漏掉。
static_assert(LARGE_SORT_LEN < INT32_SORT_LEN_MAX,
              "isInt64Shape_ must imply isLargeShape_: LARGE_SORT_LEN must stay below INT32_SORT_LEN_MAX");
// checkUb 中 UB 占用按「每 tile 元素数 × 系数」估算：
//   int32 索引：xBuf ping/pong(fp32) + outValue(fp32) + outIndex(int32) = 16 元素/tile；
//   int64 索引：outIndexBuf 翻倍（int64），系数 16 -> 20。
static constexpr uint64_t UB_COEF_PER_TILE = 16;
static constexpr uint64_t UB_COEF_PER_TILE_I64 = 20;
static constexpr uint64_t INPUT_X = 0;
static constexpr uint64_t NUM_VALUE_2BIT = 4;
static constexpr uint64_t BIT_PER_BYTE = 8;
static constexpr uint64_t BUFFER_NUM_IN_OUT = 4;
static constexpr uint64_t ATTR_INDEX_SORTED = 0;
static constexpr uint64_t ATTR_INDEX_DIM = 1;
static constexpr uint64_t ATTR_INDEX_LARGEST = 2;
static constexpr uint64_t OUTPUT_INDICES = 1;
static constexpr uint64_t WS_INT32_PER_TILE = 5;

// ---- SelectTopK 常量（与 kernel 同口径）----
static constexpr uint64_t STK_MAX_K = 4096;            /**< 子门控窗口上界：k ≤ 4096 */
static constexpr uint64_t STK_MIN_SORT_LEN = 12000000; /**< 子门控窗口下界：sortLen ≥ 12e6 */
static constexpr uint64_t STK_MIN_SEG_LEN = 256;       /**< 行内多核单核最小段长 = CACHE_LINE / sizeof(T) */
static constexpr uint64_t STK_SLOT_ALIGN = 64; /**< GM 归并槽位步长对齐（元素，与 kernel 同口径） */
static constexpr uint64_t STK_FILTERED_CAP = 1536; /**< 阈值命中上限（候选水位余量） */
static constexpr uint64_t STK_PAD = 256;           /**< 候选容量余量 */
static constexpr uint64_t STK_MERGE4_MAX_K = 2048; /**< 4 路归并的最大 k：4k 候选缓冲放得进 UB 才启用 */
static constexpr uint64_t STK_SINGLE_MERGE_MAX_K = 128; /**< 单核归并的最大 k：k≤128 由核 0 串行合并全部槽位 */
static constexpr uint64_t STK_UB_RESERVE = 4096;    /**< UB 预算预留（对齐/框架开销保守下界） */
static constexpr uint32_t STK_MIN_ACTIVE_CORES = 2; /**< 并行度下限（防御值） */

/** @brief 向上取整到 32 字节（与 kernel STK_ALIGN32BYTE 同口径） */
static inline uint64_t StkAlign32Byte(uint64_t v) { return (v + 31ULL) / 32ULL * 32ULL; }

/** @brief 归并槽位合并所需候选数：sides·AlignUp(k,8) + k（与 kernel MergeSlots 同口径） */
static inline uint64_t StkMergeNeed(uint64_t kValue, uint64_t sides)
{
    const uint64_t kA = (kValue + 7ULL) / 8ULL * 8ULL;
    return sides * kA + kValue;
}

/**
 * @brief SelectTopK 单帧 UB 占用核算（与 kernel InitBuffers 逐项对应，E=2、EW=4）
 */
static uint64_t SelectTopKUbUsage(uint64_t kValue, uint64_t blockSize, uint32_t numStages, uint64_t cap,
                                  uint64_t mergeArity, uint64_t mergeSides, uint64_t blockDim)
{
    // 4 路/单核归并的 fp32 暂存按合并候选数放大（与 kernel InitBuffers 同口径）
    uint64_t mergeElems = blockSize;
    if (mergeArity == 0ULL) {
        const uint64_t kA = (kValue + 7ULL) / 8ULL * 8ULL;
        uint64_t singleSides = (cap > kValue) ? ((cap - kValue) / kA) : 0ULL;
        if (singleSides + 1ULL > blockDim) {
            singleSides = blockDim - 1ULL;
        }
        mergeElems = singleSides * kA + kValue;
    } else if (mergeArity > 2ULL) {
        mergeElems = StkMergeNeed(kValue, mergeSides);
    }
    const uint64_t fp32Elems = (mergeElems > blockSize) ? mergeElems : blockSize;
    uint64_t maskBytes = StkAlign32Byte((cap + 64ULL) / 8ULL);
    uint64_t usage = 0;
    usage += static_cast<uint64_t>(numStages) * blockSize * sizeof(uint16_t); // 输入 staging（bf16，多级流水）
    usage += fp32Elems * sizeof(float);                                       // fp32 工作缓冲
    usage += cap * sizeof(float);                                             // 候选 value（fp32 尺寸双视图）
    usage += cap * sizeof(int32_t);                                           // 候选 index（0-based 轴下标）
    usage += cap * sizeof(int32_t);    // 复选二分 / EQ 掩码压实用 scratch
    usage += kValue * sizeof(float);   // LT/GT 值压实
    usage += kValue * sizeof(int32_t); // LT/GT 索引压实
    usage += 2ULL * maskBytes;         // 复选掩码（LT / EQ）
    return usage;
}

// 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
ge::graphStatus RadixTopKTiling::GetPlatformInfo()
{
    OP_LOGD(opName_, "RadixTopKTiling GetPlatformInfo.");
    auto compileInfo = static_cast<const RadixTopKCompileInfo*>(context_->GetCompileInfo());
    if (compileInfo == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "compileInfo");
        return ge::GRAPH_FAILED;
    }

    totalCoreNum_ = static_cast<uint64_t>(compileInfo->totalCoreNum);
    ubSize_ = compileInfo->ubSizePlatForm;
    OP_CHECK_IF((ubSize_ <= 0), OP_LOGE(opName_, "ub size is invalid."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 2、获取INPUT/OUTPUT/ATTR信息
ge::graphStatus RadixTopKTiling::GetShapeAttrsInfo()
{
    OP_LOGD(opName_, "RadixTopKTiling GetShapeAttrsInfo.");
    // 获取输入shape和dtype
    const gert::StorageShape* xShape = context_->GetInputShape(INPUT_X);
    if (xShape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "x");
        return ge::GRAPH_FAILED;
    }
    uint64_t batch = 1;
    uint64_t dimNum = xShape->GetShape().GetDimNum();
    for (uint64_t i = 0; i < dimNum - 1; i++) {
        batch *= xShape->GetShape().GetDim(i);
    }
    tilingData_.batch = batch;
    tilingData_.sortLen = xShape->GetShape().GetDim(dimNum - 1);

    auto tensorK = context_->GetInputTensor(1);
    if (tensorK == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "k");
        return ge::GRAPH_FAILED;
    }
    const int32_t* constDataPtr = tensorK->GetData<int32_t>();
    OP_CHECK_IF(constDataPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "k"), return ge::GRAPH_FAILED);
    int32_t kValue = static_cast<int32_t>(*constDataPtr);
    OP_CHECK_IF(
        kValue == 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "k", std::to_string(kValue).c_str(), "k must be greater than 0"),
        return ge::GRAPH_FAILED);
    tilingData_.kValue = kValue;

    auto attrs = context_->GetAttrs();
    tilingData_.sorted = *attrs->GetAttrPointer<bool>(ATTR_INDEX_SORTED);
    tilingData_.largest = *attrs->GetAttrPointer<bool>(ATTR_INDEX_LARGEST);
    isLargeShape_ = tilingData_.sortLen > LARGE_SORT_LEN;
    isInt64Shape_ = tilingData_.sortLen > INT32_SORT_LEN_MAX;
    // int64 模板输出 int64 索引，防御性校验输出 dtype 匹配（int32 组合不允许超 int32 shape）
    if (isInt64Shape_) {
        auto indicesDesc = context_->GetOutputDesc(1);
        if (indicesDesc == nullptr) {
            OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "indices");
            return ge::GRAPH_FAILED;
        }
        OP_CHECK_IF((indicesDesc->GetDataType() != ge::DT_INT64),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        opName_, "indices", Ops::Base::ToString(indicesDesc->GetDataType()).c_str(),
                        "RadixTopK sortLen exceeds int32, indices output must be int64"),
                    return ge::GRAPH_FAILED);
    }

    // dim 属性须为 -1（最后一维）或 rank-1，否则报错；aclnn 层会先行转置。
    auto attrsDim = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_DIM);
    if (attrsDim != nullptr && dimNum > 0) {
        int64_t lastDim = static_cast<int64_t>(dimNum) - 1;
        OP_CHECK_IF(((*attrsDim != -1) && (*attrsDim != lastDim)),
                    OP_LOGE(opName_, "RadixTopK dim [%ld] must be -1 or rank-1 [%ld].", *attrsDim, lastDim),
                    return ge::GRAPH_FAILED);
    }

    xDType_ = context_->GetInputDesc(INPUT_X)->GetDataType();
    xDtypeSize_ = ge::GetSizeByDataType(xDType_);

    // SelectTopK 子门控：窗口 k ≤ 4096 ∧ sortLen ≥ 12e6，且 indices 为 int32、非 int64 shape；
    // 窗口内 k ≤ sortLen、sortLen/k ≥ 100 与 dtype 由算子注册约束保证。不满足则回落 radix。
    auto indicesDesc = context_->GetOutputDesc(OUTPUT_INDICES);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indicesDesc);
    const ge::DataType indicesDType = indicesDesc->GetDataType();
    const bool inStkWindow = (tilingData_.sortLen >= STK_MIN_SORT_LEN) && (tilingData_.kValue <= STK_MAX_K);
    isSelectTopK_ = inStkWindow && (indicesDType == ge::DT_INT32) && !isInt64Shape_;

    return ge::GRAPH_SUCCESS;
}

// SelectTopK 尝试失败回落 radix 时清理 stk* 字段：保证回落口径确定（TilingKey bit4=0 时 kernel 不读这些字段）
static void ResetSelectTopKTilingData(RadixTopKTilingData& td)
{
    td.stkKLevel = 0;
    td.stkBlockSize = 0;
    td.stkCandidateCapacity = 0;
    td.stkFilteredCapacity = 0;
    td.stkNumStages = 0;
    td.stkSafeTrigger = 0;
    td.stkWarmupLen = 0;
    td.stkCoreMode = 0;
    td.stkRowsPerCore = 0;
    td.stkFormerCoreNum = 0;
    td.stkTailCoreNum = 0;
    td.stkSegLen = 0;
    td.stkFormerSegCoreNum = 0;
    td.stkTailSegCoreNum = 0;
    td.stkFullRows = 0;
    td.stkResidualRows = 0;
    td.stkMergeArity = 0;
}

// 3、计算数据切分TilingData
ge::graphStatus RadixTopKTiling::DoOpTiling()
{
    OP_LOGD(opName_, "RadixTopKTiling DoOpTiling.");
    bool isCalcSuccess = false;
    // SelectTopK 优先；UB 三档均不满足时清理 stk 字段并回落 radix 三分支
    if (isSelectTopK_) {
        isCalcSuccess = CalcSelectTopKTilingParams();
        if (!isCalcSuccess) {
            ResetSelectTopKTilingData(tilingData_);
            isSelectTopK_ = false;
        }
    }
    if (isSelectTopK_) {
        // SelectTopK 分支已自持 blockDim_ 与 coreNum，直接进入公共收尾
        OP_CHECK_IF((blockDim_ == 0), OP_LOGE(opName_, "RadixTopK SelectTopK blockDim is zero."),
                    return ge::GRAPH_FAILED);
    } else {
        if (isInt64Shape_) {
            isCalcSuccess = CalcLargeTilingParams(tilingData_.sortLen, UB_COEF_PER_TILE_I64);
        } else if (!isLargeShape_) {
            isCalcSuccess = CalcTilingParams(tilingData_.sortLen);
            if (!isCalcSuccess) {
                isLargeShape_ = true;
                isCalcSuccess = CalcLargeTilingParams(tilingData_.sortLen, UB_COEF_PER_TILE);
            }
        } else {
            isCalcSuccess = CalcLargeTilingParams(tilingData_.sortLen, UB_COEF_PER_TILE);
        }
        OP_CHECK_IF((!isCalcSuccess), OP_LOGE(opName_, "RadixTopK Tiling failed."), return ge::GRAPH_FAILED);
        // int64 模板的 coreTopK/边界计数等 int32 通道要求 kValue 不超 int32 上限
        OP_CHECK_IF((isInt64Shape_ && tilingData_.kValue > INT32_SORT_LEN_MAX),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "k", std::to_string(tilingData_.kValue).c_str(),
                                                          "kValue exceeds int32 limit for int64 shape"),
                    return ge::GRAPH_FAILED);
        // per-core 直方图计数保持 int32：per-core 元素数须 < 2^31（S < 2^31*coreNum，物理内存内恒成立，防御性校验）
        OP_CHECK_IF((isInt64Shape_ && tilingData_.formerTileNum * tilingData_.formerTileLen > INT32_SORT_LEN_MAX),
                    OP_LOGE(opName_, "RadixTopK per-core element num exceeds int32 limit."), return ge::GRAPH_FAILED);

        blockDim_ = tilingData_.formerCoreNum + tilingData_.tailCoreNum;
        tilingData_.coreNum = blockDim_;
        // 仅 UB 变体可复用 indices 内存（WS/INT64 变体的跨核暂存与之冲突）；isInt64Shape_ ⇒ isLargeShape_，||
        // 为防御性冗余
        if (isLargeShape_ || isInt64Shape_) {
            tilingData_.needWorkspace = true;
        } else {
            uint64_t wsInt32Cnt = tilingData_.coreNum * (NUM_VALUE_2BIT + 2);
            tilingData_.needWorkspace = (tilingData_.kValue <= wsInt32Cnt);
        }
    }
    SetTilingData();
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

// 4、计算高阶API的TilingData
ge::graphStatus RadixTopKTiling::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

// 5、计算TilingKey
uint64_t RadixTopKTiling::GetTilingKey() const
{
    OP_LOGD(opName_, "RadixTopKTiling GetTilingKey.");

    // isSelectTopK=1 时 isLargeShape/isInt64Shape 按编码约定恒置 0
    const bool keyLargeShape = isSelectTopK_ ? false : isLargeShape_;
    const bool keyInt64Shape = isSelectTopK_ ? false : isInt64Shape_;
    const uint64_t tilingKey = GET_TPL_TILING_KEY(tilingData_.sorted, tilingData_.largest, keyLargeShape, keyInt64Shape,
                                                  isSelectTopK_);
    OP_LOGD(opName_, "tilingKey is: [%lu]", tilingKey);
    OP_LOGD(opName_, "sorted, largest, isLargeShape, isInt64Shape, isSelectTopK is: [%d, %d, %d, %d, %d]",
            tilingData_.sorted, tilingData_.largest, keyLargeShape, keyInt64Shape, isSelectTopK_);

    return tilingKey;
}

// 6、计算Workspace 大小
ge::graphStatus RadixTopKTiling::GetWorkspaceSize()
{
    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    const uint64_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();

    if (isSelectTopK_) {
        // SelectTopK：stkCoreMode==1（行内多核）与行模式余数行段化（stkResidualRows>0）都走 GM 槽位树形归并，
        // 须按 coreNum 分配归并槽位 workspace（每核 k 个值 + k 个索引；余数行时 blockDim==coreNum，公式同）
        workspaceSize_ = sysWorkspaceSize;
        if (tilingData_.stkCoreMode == 1 || tilingData_.stkResidualRows > 0) {
            const uint64_t slotStride = (tilingData_.kValue + STK_SLOT_ALIGN - 1) / STK_SLOT_ALIGN * STK_SLOT_ALIGN;
            workspaceSize_ += 2ULL * tilingData_.coreNum * slotStride * sizeof(int32_t);
        }
        return ge::GRAPH_SUCCESS;
    }
    if (!tilingData_.needWorkspace) {
        workspaceSize_ = 0;
        return ge::GRAPH_SUCCESS;
    }
    // 注：同上，isInt64Shape_ 已被 isLargeShape_ 蕴含
    if (isLargeShape_ || isInt64Shape_) {
        workspaceSize_ = (NUM_VALUE_2BIT * tilingData_.coreNum + tilingData_.coreNum + tilingData_.coreNum +
                          WS_INT32_PER_TILE * tilingData_.totalTileNum) *
                         sizeof(int32_t);
    } else {
        workspaceSize_ = (NUM_VALUE_2BIT * tilingData_.coreNum + tilingData_.coreNum + tilingData_.coreNum) *
                         sizeof(int32_t);
    }
    workspaceSize_ += sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

// 7、保存Tiling数据
ge::graphStatus RadixTopKTiling::PostTiling()
{
    OP_LOGD(opName_, "RadixTopKTiling PostTiling.");

    // 设置workspace大小
    auto workspaces = context_->GetWorkspaceSizes(1);
    if (workspaces == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "workspaces");
        return ge::GRAPH_FAILED;
    }
    workspaces[0] = workspaceSize_;

    // 用到SyncAll，需要设置为batch mode模式
    context_->SetScheduleMode(1);

    auto res = context_->SetBlockDim(static_cast<uint32_t>(blockDim_));
    OP_CHECK_IF((res != ge::GRAPH_SUCCESS), OP_LOGE(opName_, "SetBlockDim failed."), return ge::GRAPH_FAILED);

    res = context_->SetLocalMemorySize(ubSize_);
    OP_CHECK_IF((res != ge::GRAPH_SUCCESS), OP_LOGE(opName_, "SetLocalMemorySize failed."), return ge::GRAPH_FAILED);

    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                           &tilingData_, sizeof(RadixTopKTilingData));
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(sizeof(RadixTopKTilingData));

    return ge::GRAPH_SUCCESS;
}

bool RadixTopKTiling::IsCapable() { return true; }

void RadixTopKTiling::PrintTilingData()
{
    OP_LOGD(opName_, "formerCoreNum:    %lu.", tilingData_.formerCoreNum);
    OP_LOGD(opName_, "tailCoreNum:      %lu.", tilingData_.tailCoreNum);
    OP_LOGD(opName_, "totalTileNum:     %lu.", tilingData_.totalTileNum);
    OP_LOGD(opName_, "formerTileNum:    %lu.", tilingData_.formerTileNum);
    OP_LOGD(opName_, "tailTileNum:      %lu.", tilingData_.tailTileNum);
    OP_LOGD(opName_, "formerTileLen:    %lu.", tilingData_.formerTileLen);
    OP_LOGD(opName_, "tailTileLen:      %lu.", tilingData_.tailTileLen);
    OP_LOGD(opName_, "batch:            %lu.", tilingData_.batch);
    OP_LOGD(opName_, "sortLen:          %lu.", tilingData_.sortLen);
    OP_LOGD(opName_, "kValue:           %lu.", tilingData_.kValue);
    OP_LOGD(opName_, "largest:          %d.", tilingData_.largest);
    OP_LOGD(opName_, "sorted:           %d.", tilingData_.sorted);
    OP_LOGD(opName_, "isInt64Shape:     %d.", isInt64Shape_);
    OP_LOGD(opName_, "needWorkspace:    %d.", tilingData_.needWorkspace);
    OP_LOGD(opName_, "isSelectTopK:     %d.", isSelectTopK_);
    OP_LOGD(opName_, "stkKLevel:         %u.", tilingData_.stkKLevel);
    OP_LOGD(opName_, "stkBlockSize:      %lu.", tilingData_.stkBlockSize);
    OP_LOGD(opName_, "stkCandidateCap:   %lu.", tilingData_.stkCandidateCapacity);
    OP_LOGD(opName_, "stkFilteredCap:    %lu.", tilingData_.stkFilteredCapacity);
    OP_LOGD(opName_, "stkNumStages:      %u.", tilingData_.stkNumStages);
    OP_LOGD(opName_, "stkSafeTrigger:    %lu.", tilingData_.stkSafeTrigger);
    OP_LOGD(opName_, "stkWarmupLen:      %lu.", tilingData_.stkWarmupLen);
    OP_LOGD(opName_, "stkCoreMode:       %u.", tilingData_.stkCoreMode);
    OP_LOGD(opName_, "stkRowsPerCore:    %lu.", tilingData_.stkRowsPerCore);
    OP_LOGD(opName_, "stkFormerCoreNum:  %lu.", tilingData_.stkFormerCoreNum);
    OP_LOGD(opName_, "stkTailCoreNum:    %lu.", tilingData_.stkTailCoreNum);
    OP_LOGD(opName_, "stkSegLen:         %lu.", tilingData_.stkSegLen);
    OP_LOGD(opName_, "stkFormerSegCore:  %lu.", tilingData_.stkFormerSegCoreNum);
    OP_LOGD(opName_, "stkTailSegCore:    %lu.", tilingData_.stkTailSegCoreNum);
}

/**
 * @brief SelectTopK 单遍扫描快路径 tiling；返回 false 表示 UB 三档均不满足，调用方回落 radix
 */
bool RadixTopKTiling::CalcSelectTopKTilingParams()
{
    const uint64_t sortLen = tilingData_.sortLen;
    const uint64_t kValue = tilingData_.kValue;
    const uint64_t batch = tilingData_.batch;

    // ---- ① k 档位（向上取档）：0→512 / 1→1024 / 2→4096 ----
    uint32_t kLevel = 0;
    if (kValue > 512ULL) {
        kLevel = (kValue > 1024ULL) ? 2U : 1U;
    }
    tilingData_.stkKLevel = kLevel;

    // ---- ② 核映射模式（以 totalCoreNum_ 为基准，避免与 blockDim_ 循环依赖）----
    // 行内多核：单行且段长足以摊薄归并开销。归并为 GM 槽位树形归并，
    // 层数随 log2(D) 增长，故不再按 √(S/(4k)) 限流，直接取满核数。
    const bool segMode = (batch < totalCoreNum_) && (sortLen >= totalCoreNum_ * STK_MIN_SEG_LEN);

    if (segMode) {
        tilingData_.stkCoreMode = 1;
        blockDim_ = totalCoreNum_;
    } else {
        tilingData_.stkCoreMode = 0;
        blockDim_ = (batch < totalCoreNum_) ? batch : totalCoreNum_;
        // 并行度下限防御：窗口内不可达，保留以防后续调整窗口下界
        if (blockDim_ < STK_MIN_ACTIVE_CORES) {
            OP_LOGD(opName_, "SelectTopK fallback: active cores [%u] < min [%u].", blockDim_, STK_MIN_ACTIVE_CORES);
            return false;
        }
    }
    if (blockDim_ == 0) {
        return false;
    }

    // ---- ③ 行 / 段切分公式 ----
    if (tilingData_.stkCoreMode == 1) {
        const uint64_t segLen = Ops::Base::CeilDiv(sortLen, static_cast<uint64_t>(blockDim_));
        const uint64_t formerSeg = sortLen - (segLen - 1) * static_cast<uint64_t>(blockDim_);
        tilingData_.stkSegLen = segLen;
        tilingData_.stkFormerSegCoreNum = formerSeg;
        tilingData_.stkTailSegCoreNum = static_cast<uint64_t>(blockDim_) - formerSeg;
        tilingData_.stkRowsPerCore = 0;
        tilingData_.stkFormerCoreNum = 0;
        tilingData_.stkTailCoreNum = 0;
        tilingData_.stkFullRows = 0;
        tilingData_.stkResidualRows = 0;
    } else {
        // 余数行段化：batch = q*blockDim + r（r>0）时 r 个余数行由全核协作扫，避免尾核多扛整行
        const uint64_t q = batch / static_cast<uint64_t>(blockDim_);
        const uint64_t r = batch % static_cast<uint64_t>(blockDim_);
        const bool residualSeg = (r > 0) && (sortLen >= static_cast<uint64_t>(blockDim_) * STK_MIN_SEG_LEN);
        if (residualSeg) {
            tilingData_.stkRowsPerCore = q;
            tilingData_.stkFormerCoreNum = static_cast<uint64_t>(blockDim_);
            tilingData_.stkTailCoreNum = 0;
            tilingData_.stkFullRows = q * static_cast<uint64_t>(blockDim_);
            tilingData_.stkResidualRows = r;
            tilingData_.stkSegLen = Ops::Base::CeilDiv(sortLen, static_cast<uint64_t>(blockDim_));
            const uint64_t formerSeg = sortLen - (tilingData_.stkSegLen - 1) * static_cast<uint64_t>(blockDim_);
            tilingData_.stkFormerSegCoreNum = formerSeg;
            tilingData_.stkTailSegCoreNum = static_cast<uint64_t>(blockDim_) - formerSeg;
        } else {
            const uint64_t rowsPerCore = Ops::Base::CeilDiv(batch, static_cast<uint64_t>(blockDim_));
            const uint64_t formerCores = batch - (rowsPerCore - 1) * static_cast<uint64_t>(blockDim_);
            tilingData_.stkRowsPerCore = rowsPerCore;
            tilingData_.stkFormerCoreNum = formerCores;
            tilingData_.stkTailCoreNum = static_cast<uint64_t>(blockDim_) - formerCores;
            tilingData_.stkFullRows = 0;
            tilingData_.stkResidualRows = 0;
            tilingData_.stkSegLen = 0;
            tilingData_.stkFormerSegCoreNum = 0;
            tilingData_.stkTailSegCoreNum = 0;
        }
    }

    // ---- ④ UB 自适应降级选档（严格不改写其它 stk 字段；三档均超限 ⇒ 回落）----
    static const uint32_t kTierStages[3] = {3U, 2U, 2U};
    static const uint64_t kTierBlockSize[3] = {8192ULL, 8192ULL, 4096ULL};
    // 归并叉数：k≤128 由核 0 串行合并全部槽位（只留 2 次 SyncAll）；
    // k≤2048 四路（4k 候选放得进 UB），k>2048 二叉（4k 约 246 KB 超预算）。
    const uint64_t mergeArity = (kValue <= STK_SINGLE_MERGE_MAX_K) ? 0ULL :
                                                                     ((kValue <= STK_MERGE4_MAX_K) ? 4ULL : 2ULL);
    const uint64_t mergeSides = (mergeArity == 0ULL) ? 0ULL : (mergeArity - 1ULL);
    const uint64_t ubBudget = (ubSize_ > STK_UB_RESERVE) ? (ubSize_ - STK_UB_RESERVE) : ubSize_;
    bool ubFound = false;
    for (uint32_t t = 0; (t < 3U) && !ubFound; ++t) {
        const uint64_t blockSize = kTierBlockSize[t];
        // cap 线性取值：cap ≥ safeTrigger + B 是扫描追加的硬约束，+STK_PAD 补 repeat 余量；
        // 还需覆盖归并布局 sides·AlignUp(k,8)+k（见 ⑤ 断言），不依赖 cap 为 2 的幂。
        const uint64_t scanCap = kValue + blockSize + STK_FILTERED_CAP + STK_PAD;
        const uint64_t mergeCap = StkMergeNeed(kValue, mergeSides) + STK_PAD;
        const uint64_t cap = (mergeCap > scanCap) ? mergeCap : scanCap;
        if (SelectTopKUbUsage(kValue, blockSize, kTierStages[t], cap, mergeArity, mergeSides, blockDim_) <= ubBudget) {
            tilingData_.stkNumStages = kTierStages[t];
            tilingData_.stkBlockSize = blockSize;
            tilingData_.stkFilteredCapacity = STK_FILTERED_CAP;
            tilingData_.stkCandidateCapacity = cap;
            tilingData_.stkSafeTrigger = kValue + STK_FILTERED_CAP;
            tilingData_.stkWarmupLen = blockSize;
            tilingData_.stkMergeArity = static_cast<uint32_t>(mergeArity);
            ubFound = true;
        }
    }
    if (!ubFound) {
        OP_LOGD(opName_, "SelectTopK fallback: no UB tier fits (ubSize [%lu]).", ubSize_);
        return false;
    }

    // ---- ⑤ 候选容量与触发水位断言：cap ≥ safeTrigger + B（保证一次全块追加不溢出）----
    OP_CHECK_IF((tilingData_.stkCandidateCapacity < tilingData_.stkSafeTrigger + tilingData_.stkBlockSize),
                OP_LOGE(opName_, "SelectTopK: stkCandidateCapacity [%lu] < stkSafeTrigger [%lu] + stkBlockSize [%lu].",
                        tilingData_.stkCandidateCapacity, tilingData_.stkSafeTrigger, tilingData_.stkBlockSize),
                return false);
    // 归并/复选布局断言：cap 须覆盖 sides·AlignUp(k,8)+k 与 EQ 值暂存，避免调档后静默越界。
    const uint64_t mergeNeed = StkMergeNeed(kValue, mergeSides);
    OP_CHECK_IF((tilingData_.stkCandidateCapacity < mergeNeed),
                OP_LOGE(opName_, "SelectTopK: stkCandidateCapacity [%lu] < mergeSides·AlignUp(k,8)+k [%lu].",
                        tilingData_.stkCandidateCapacity, mergeNeed),
                return false);

    tilingData_.coreNum = blockDim_;
    tilingData_.needWorkspace = false; // SelectTopK 自持 GM workspace = 0 B（仅保留系统量）
    return true;
}

void RadixTopKTiling::CalcTileDistribution(const uint64_t& dataNum, const uint64_t& formerTileLen)
{
    uint64_t totalTileNum = Ops::Base::CeilDiv(dataNum, formerTileLen);
    uint64_t formerTileNum = Ops::Base::CeilDiv(totalTileNum, totalCoreNum_);
    uint64_t tailTileNum = totalTileNum / totalCoreNum_;
    uint64_t formerCoreNum = totalTileNum % totalCoreNum_;
    uint64_t tailCoreNum = tailTileNum ? (totalCoreNum_ - formerCoreNum) : 0;
    uint64_t totalProcessedTiles = formerTileNum * formerCoreNum + tailTileNum * tailCoreNum;
    uint64_t tailTileLen = dataNum - formerTileLen * (totalProcessedTiles - 1);

    tilingData_.formerCoreNum = formerCoreNum;
    tilingData_.tailCoreNum = tailCoreNum;
    tilingData_.totalTileNum = totalTileNum;
    tilingData_.formerTileNum = formerTileNum;
    tilingData_.tailTileNum = tailTileNum;
    tilingData_.formerTileLen = formerTileLen;
    tilingData_.tailTileLen = tailTileLen;
}

template <typename CheckUbFn>
bool RadixTopKTiling::TryCalcTileDistribution(const uint64_t& dataNum, uint64_t startTileLen, uint64_t minTileLen,
                                              uint64_t step, CheckUbFn&& checkUb)
{
    for (uint64_t tileLen = startTileLen; tileLen >= minTileLen; tileLen -= step) {
        if (checkUb(tileLen)) {
            CalcTileDistribution(dataNum, tileLen);
            return true;
        }
    }
    OP_LOGD(opName_, "Cannot find suitable tileLen to do RadixTopK.");
    return false;
}

/**
 * @brief 大 shape / int64 shape 的 tile 切分（两者仅 UB 系数不同，合并为一个带参函数）
 * @param dataNum 单 core 处理的数据量
 * @param ubCoefPerTile UB 占用中「每 tile 元素数」的系数（int32 索引 16、int64 索引 20）
 */
bool RadixTopKTiling::CalcLargeTilingParams(const uint64_t& dataNum, const uint64_t& ubCoefPerTile)
{
    uint64_t dataAlign = CACHE_LINE / xDtypeSize_;
    uint64_t minTileLen = dataAlign;

    auto checkUb = [this, ubCoefPerTile](uint64_t tileLen) -> bool {
        uint64_t cmpMaskSize = tileLen / 8;
        uint64_t ubUsed = ubCoefPerTile * tileLen + cmpMaskSize + MAX_TILE_NUM_IN_UB * sizeof(int32_t) + BLOCK_SIZE;
        return ubUsed <= ubSize_;
    };

    return TryCalcTileDistribution(dataNum, LARGE_TILE_LEN, minTileLen, dataAlign, checkUb);
}

bool RadixTopKTiling::CalcTilingParams(const uint64_t& dataNum)
{
    uint64_t coreNum = totalCoreNum_;
    uint64_t dataAlign = CACHE_LINE / xDtypeSize_;
    uint64_t minTileLen = dataAlign;

    auto checkUb = [this, dataNum, coreNum](uint64_t tileLen) -> bool {
        if (tileLen == 0)
            return false;
        uint64_t totalTn = (dataNum + tileLen - 1) / tileLen;
        if (totalTn == 0)
            totalTn = 1;
        uint64_t tn = (totalTn + coreNum - 1) / coreNum;
        uint64_t tnAlign = Ops::Base::CeilAlign(tn, static_cast<uint64_t>(8));
        uint64_t tileHistSize = NUM_VALUE_2BIT * tnAlign * sizeof(int32_t);
        uint64_t cmpMaskSize = tileLen / 8;
        uint64_t tempBuf = (tileHistSize > cmpMaskSize) ? tileHistSize : cmpMaskSize;
        uint64_t ubUsed = BUFFER_NUM_IN_OUT * sizeof(int32_t) * tileLen + NUM_VALUE_2BIT * tnAlign + tempBuf +
                          BLOCK_SIZE;
        return ubUsed <= ubSize_;
    };

    uint64_t startTileLen = MAX_TILE_LEN;
    uint64_t utilCap = Ops::Base::CeilAlign(Ops::Base::CeilDiv(dataNum, coreNum), dataAlign);
    if (utilCap < startTileLen) {
        startTileLen = utilCap;
    }

    return TryCalcTileDistribution(dataNum, startTileLen, minTileLen, dataAlign, checkUb);
}

void RadixTopKTiling::SetTilingData()
{
    OP_LOGD(opName_, "RadixTopKTiling SetTilingData.");
    RadixTopKTilingData* tilingData = context_->GetTilingData<RadixTopKTilingData>();
    tilingData->formerCoreNum = tilingData_.formerCoreNum;
    tilingData->tailCoreNum = tilingData_.tailCoreNum;
    tilingData->totalTileNum = tilingData_.totalTileNum;
    tilingData->formerTileNum = tilingData_.formerTileNum;
    tilingData->tailTileNum = tilingData_.tailTileNum;
    tilingData->formerTileLen = tilingData_.formerTileLen;
    tilingData->tailTileLen = tilingData_.tailTileLen;
    tilingData->coreNum = tilingData_.coreNum;
    tilingData->batch = tilingData_.batch;
    tilingData->sortLen = tilingData_.sortLen;
    tilingData->kValue = tilingData_.kValue;
    tilingData->sorted = tilingData_.sorted;
    tilingData->largest = tilingData_.largest;
    tilingData->needWorkspace = tilingData_.needWorkspace;
    // ---- SelectTopK 字段组 ----
    tilingData->stkKLevel = tilingData_.stkKLevel;
    tilingData->stkBlockSize = tilingData_.stkBlockSize;
    tilingData->stkCandidateCapacity = tilingData_.stkCandidateCapacity;
    tilingData->stkFilteredCapacity = tilingData_.stkFilteredCapacity;
    tilingData->stkNumStages = tilingData_.stkNumStages;
    tilingData->stkSafeTrigger = tilingData_.stkSafeTrigger;
    tilingData->stkWarmupLen = tilingData_.stkWarmupLen;
    tilingData->stkCoreMode = tilingData_.stkCoreMode;
    tilingData->stkRowsPerCore = tilingData_.stkRowsPerCore;
    tilingData->stkFormerCoreNum = tilingData_.stkFormerCoreNum;
    tilingData->stkTailCoreNum = tilingData_.stkTailCoreNum;
    tilingData->stkSegLen = tilingData_.stkSegLen;
    tilingData->stkFormerSegCoreNum = tilingData_.stkFormerSegCoreNum;
    tilingData->stkTailSegCoreNum = tilingData_.stkTailSegCoreNum;
    tilingData->stkFullRows = tilingData_.stkFullRows;
    tilingData->stkResidualRows = tilingData_.stkResidualRows;
    tilingData->stkMergeArity = tilingData_.stkMergeArity;
}
} // namespace RadixTopK

static ge::graphStatus TilingRadixTopK(gert::TilingContext* context)
{
    OP_LOGD(context, "RadixTopKTiling start.");
    RadixTopK::RadixTopKTiling tilingOp(context);
    auto ret = tilingOp.DoTiling();
    OP_CHECK_IF((ret == ge::GRAPH_FAILED), OP_LOGD(context, "RadixTopKTiling tiling failed!"), return ge::GRAPH_FAILED);
    OP_LOGD(context, "RadixTopKTiling end.");

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForRadixTopK(gert::TilingParseContext* context)
{
    OP_LOGD(context, "Tiling Prepare For RadixTopK start.");
    auto compileInfo = context->GetCompiledInfo<RadixTopKCompileInfo>();
    if (compileInfo == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "compileInfo");
        return ge::GRAPH_FAILED;
    }
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "platformInfo");
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    if (compileInfo->totalCoreNum == 0) {
        OP_LOGE(context, "coreNum %d", compileInfo->totalCoreNum);
        return ge::GRAPH_FAILED;
    }
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = ubSizePlatForm;
    OP_LOGD(context, "ub_size_platform is %lu.", compileInfo->ubSizePlatForm);
    OP_LOGD(context, "Tiling Prepare For RadixTopK end.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(RadixTopK)
    .Tiling(TilingRadixTopK)
    .TilingParse<RadixTopKCompileInfo>(TilingPrepareForRadixTopK)
    .TilingInputsDataDependency({1});
} // namespace optiling
