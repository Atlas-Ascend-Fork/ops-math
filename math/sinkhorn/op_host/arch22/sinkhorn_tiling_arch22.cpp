/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file sinkhorn.cc
 * \brief
 */

#include "register/op_def_registry.h"
#include "platform/platform_info.h"
#include "platform/platform_ascendc.h"
#include "math/sinkhorn/op_host/sinkhorn_tiling.h"
#include <limits>
#include <log/log.h>
#include <tiling/platform/platform_ascendc.h>
#include <platform/platform_infos_def.h>

namespace {
constexpr static float UB_USAGE = 0.85f;
constexpr static uint32_t ROW_BLOCK_SIZE = 32;
constexpr static uint32_t MAX_TILE_ROW = 127 * 32; // 数据复制时，blockCount最大为4095，向下对齐到 127 * 32
constexpr static uint32_t WORKSPACE_HEADER_SIZE = 64; // 64B对齐
constexpr static uint32_t CACHE_LINE_SIZE = 64;
constexpr static uint32_t NARROW_COL_THRESHOLD = 16;
constexpr static uint32_t NARROW_COL_ROWS_PER_CORE = 512;
constexpr static uint32_t IO_DIM = 2;
constexpr static uint64_t MAX_COST_COL = 4168;
constexpr static uint32_t BATCH_MODE_SCHEDULE = 1; // SetScheduleMode 调度模式：全核 batch 启动，配合 kernel 侧 SyncAll
constexpr static uint32_t WORKSPACE_BLOCK_NUM = 1; // GetWorkspaceSizes 申请的 workspace 块数
constexpr static uint64_t D0_QUEUE_COUNT = 4;
constexpr static uint64_t D1_QUEUE_COUNT = 5;
// arch22 元素字节数：FP16 走 <half,half> 半精度内部，sizeOfDataType=2；FP32/BF16 内部恒 FP32，=4。
constexpr static uint64_t SIZEOF_FP32 = 4;
constexpr static uint64_t SIZEOF_FP16 = 2;

uint64_t AlignUp(uint64_t value, uint64_t alignment) { return (value + alignment - 1) / alignment * alignment; }

bool CheckedMul(uint64_t lhs, uint64_t rhs, uint64_t& result)
{
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

bool CheckedAdd(uint64_t lhs, uint64_t rhs, uint64_t& result)
{
    if (rhs > std::numeric_limits<uint64_t>::max() - lhs) {
        return false;
    }
    result = lhs + rhs;
    return true;
}

// arch22 UB 预留公式，必须与 kernel op_kernel/sinkhorn.inc::InitUB 逐字对应：
//   costIn + costOut : 2 * tileRows * colAligned * sizeOfT（无 arch35 VF +64 padding）
//   d0 x4            : 4 * AlignUp32(tileRows * sizeOfT)（32B 队列对齐）
//   d1 x5            : 5 * colAligned * sizeOfT
//   无 CPU 级联归约 scratch（Option A 行和用 ReduceSum + d0OutQueue3 作 scratch）
uint64_t CalculateUbBytesArch22(uint64_t tileRows, uint64_t colAligned, uint64_t sizeOfT)
{
    const uint64_t costBytes = 2 * tileRows * colAligned * sizeOfT;
    const uint64_t d0Bytes = D0_QUEUE_COUNT * AlignUp(tileRows * sizeOfT, ROW_BLOCK_SIZE);
    const uint64_t d1Bytes = D1_QUEUE_COUNT * colAligned * sizeOfT;
    return costBytes + d0Bytes + d1Bytes;
}
} // namespace

namespace optiling {
ge::graphStatus TilingPrepareForSinkhornArch22(gert::TilingParseContext* context);
ge::graphStatus TilingForSinkhornArch22(gert::TilingContext* context);

class SinkhornTilingArch22 {
public:
    explicit SinkhornTilingArch22(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus Init();
    ge::graphStatus RunKernelTiling();
    void TilingDataPrint();

private:
    ge::graphStatus InitWS();
    inline ge::graphStatus InitTiling();

    SinkhornTilingData tiling;

    gert::TilingContext* tilingContext = nullptr;

    // ub对齐后长度
    uint64_t totalLengthAlignedWithBlock;

    // context传递
    uint64_t tilingKey = 1;
    uint64_t numBlocks = 0;

    // tiling data传递
    uint64_t formerNum = 0;    // former 数量
    uint64_t formerRow = 0;    // former cost行数
    uint64_t formerLength = 0; // former cost总长

    uint64_t formerTileNum = 0;        // former Tile数量
    uint64_t formerLastTileRow = 0;    // fomer last Tile行数
    uint64_t formerLastTileLength = 0; // fomer last Tile长度

    uint64_t tailNum = 0;    // tail 数量
    uint64_t tailRow = 0;    // tail cost行数
    uint64_t tailLength = 0; // tail cost总长

    uint64_t tailTileNum = 0;        // tail Tile数量
    uint64_t tailLastTileRow = 0;    // tail last Tile行数
    uint64_t tailLastTileLength = 0; // tail last Tile长度

    uint64_t tileRow = 0;    // Tile行数(非Last)
    uint64_t tileLength = 0; // Tile长度(非Last)

    uint64_t totalRow = 0;        // 总行数
    uint64_t totalCol = 0;        // 总列数
    uint64_t totalColAligned = 0; // 对齐后的总列数

    float tol = 0.0001f; // 误差

    size_t userWorkspaceSize; // workspace大小

    // 求单个元素大小
    uint64_t sizeOfDataType = 1;
};

ge::graphStatus SinkhornTilingArch22::Init()
{
    OP_LOGD(tilingContext, "Tiling initialization starts.");

    auto costShape = tilingContext->GetInputShape(0);
    if (costShape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(tilingContext->GetNodeName(), "cost");
        return ge::GRAPH_FAILED;
    }
    auto costStorageShape = costShape->GetStorageShape();
    // 防御校验：aclnn 直连通路下 tiling 侧独立把关，避免非法形状进入分块计算（blockLen uint16 截断等风险）
    if (costStorageShape.GetDimNum() != IO_DIM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(tilingContext->GetNodeName(), "cost",
                                     std::to_string(costStorageShape.GetDimNum()) + "D", "2D");
        return ge::GRAPH_FAILED;
    }
    int64_t rowDim = costStorageShape.GetDim(0);
    int64_t colDim = costStorageShape.GetDim(1);
    if (rowDim < 0 || colDim < 0) {
        const std::string costDims = "(" + std::to_string(rowDim) + ", " + std::to_string(colDim) + ")";
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(tilingContext->GetNodeName(), "cost", costDims.c_str(),
                                                 "cost dims must be non-negative");
        return ge::GRAPH_FAILED;
    }
    totalRow = static_cast<uint64_t>(rowDim);
    totalCol = static_cast<uint64_t>(colDim);
    if (totalCol > MAX_COST_COL) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(tilingContext->GetNodeName(), "cost", "col=" + std::to_string(totalCol),
                                     "<= 4168");
        return ge::GRAPH_FAILED;
    }
    uint64_t totalElements = 0;
    if (!CheckedMul(totalRow, totalCol, totalElements) || totalElements > std::numeric_limits<uint32_t>::max()) {
        const std::string costShapeText = std::to_string(totalRow) + "x" + std::to_string(totalCol);
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(tilingContext->GetNodeName(), "cost", costShapeText.c_str(),
                                                  "cost element count must be <= UINT32_MAX");
        return ge::GRAPH_FAILED;
    }

    auto attrs = tilingContext->GetAttrs();
    if (attrs != nullptr && attrs->GetAttrNum() != 0) {
        const float* tolPtr = attrs->GetAttrPointer<float>(0);
        if (tolPtr != nullptr) {
            tol = *tolPtr;
        }
    }
    auto compileInfo = reinterpret_cast<const SinkhornCompileInfo*>(tilingContext->GetCompileInfo());
    if (compileInfo == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(tilingContext->GetNodeName(), "compileInfo");
        return ge::GRAPH_FAILED;
    }
    uint64_t aivNum = compileInfo->aivNum; // Vector核数量
    uint64_t ubSize = compileInfo->ubSize; // ubSize大小
    if (aivNum == 0) {
        OP_LOGE_WITHOUT_REPORT(tilingContext, "AIV core count must be greater than zero.");
        return ge::GRAPH_FAILED;
    }

    auto inputDesc = tilingContext->GetInputDesc(0);
    if (inputDesc == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(tilingContext->GetNodeName(), "inputDesc");
        return ge::GRAPH_FAILED;
    }
    uint64_t dataType = inputDesc->GetDataType();
    switch (dataType) {
        case ge::DT_FLOAT:
        case ge::DT_BF16:
            // FP32 直调与 BF16（内部恒 FP32）按 4 字节
            sizeOfDataType = SIZEOF_FP32;
            break;
        case ge::DT_FLOAT16:
            // arch22：FP16 走 <half,half> 半精度内部，按 2 字节。
            // 不变式 I1：totalColAligned 必须等于 kernel rowLengthAligned=AlignUp32(totalCol*sizeof(T))/sizeof(T)，
            // 故 sizeOfDataType 必须与 kernel sizeof(T) 成对一致（Y-A 裁定）。
            sizeOfDataType = SIZEOF_FP16;
            break;
        default:
            OP_LOGE_WITH_INVALID_INPUT_DTYPE(tilingContext->GetNodeName(), "cost", std::to_string(dataType).c_str(),
                                             "FLOAT/FLOAT16/BF16");
            return ge::GRAPH_FAILED;
    }
    tilingKey = dataType;
    if (totalRow == 0 || totalCol == 0) {
        // 空 Tensor：无数据可算，单核下发，kernel 侧直接返回
        numBlocks = 1;
        tileRow = 1;
        if (InitWS() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        OP_LOGD(tilingContext, "cost is empty (%lu, %lu), skip compute.", static_cast<unsigned long>(totalRow),
                static_cast<unsigned long>(totalCol));
        return ge::GRAPH_SUCCESS;
    }

    // 按 sizeOfDataType 对齐到 32B：FP32/BF16 为 8 个元素，FP16（Y-A 半精度内部）为 16 个元素。
    uint32_t rowAlignNum = ROW_BLOCK_SIZE / sizeOfDataType;
    totalColAligned = ((totalCol + rowAlignNum - 1) / rowAlignNum) * rowAlignNum;

    uint64_t usableUbSize = static_cast<uint64_t>(ubSize * UB_USAGE);
    // fixedBytes：tileRow 极小时的常量部分（d0 队列至少 1 行的 32B 对齐量 + 5 个 d1 队列），用于 usable 过小时拒绝
    const uint64_t fixedBytes = D0_QUEUE_COUNT * AlignUp(1 * sizeOfDataType, ROW_BLOCK_SIZE) +
                                D1_QUEUE_COUNT * totalColAligned * sizeOfDataType;
    if (usableUbSize <= fixedBytes) {
        OP_LOGE_WITHOUT_REPORT(tilingContext, "UB size %lu is too small for col %lu.",
                               static_cast<unsigned long>(usableUbSize), static_cast<unsigned long>(totalCol));
        return ge::GRAPH_FAILED;
    }

    tileRow = (usableUbSize - fixedBytes) / (2 * totalColAligned * sizeOfDataType + D0_QUEUE_COUNT * sizeOfDataType);
    while (tileRow > 0 && CalculateUbBytesArch22(tileRow, totalColAligned, sizeOfDataType) > usableUbSize) {
        --tileRow;
    }
    while (CalculateUbBytesArch22(tileRow + 1, totalColAligned, sizeOfDataType) <= usableUbSize) {
        ++tileRow;
    }
    if (tileRow == 0 || CalculateUbBytesArch22(tileRow, totalColAligned, sizeOfDataType) > usableUbSize) {
        OP_LOGE_WITHOUT_REPORT(tilingContext, "UB size %lu cannot hold one tile row for col %lu.",
                               static_cast<unsigned long>(usableUbSize), static_cast<unsigned long>(totalCol));
        return ge::GRAPH_FAILED;
    }

    // 保护处理， 数据复制时，blockCount最大为4095，向下对齐到 127 * 32
    if (tileRow > MAX_TILE_ROW) {
        tileRow = MAX_TILE_ROW;
    }

    tileLength = tileRow * totalCol;

    // block数量
    uint32_t ubNum = (totalRow + tileRow - 1) / tileRow;

    // 运行核数
    numBlocks = (ubNum > aivNum) ? aivNum : ubNum;
    if (totalCol < NARROW_COL_THRESHOLD) {
        uint64_t narrowBlocks = (totalRow + NARROW_COL_ROWS_PER_CORE - 1) / NARROW_COL_ROWS_PER_CORE;
        numBlocks = (narrowBlocks > aivNum) ? aivNum : narrowBlocks;
    }

    InitTiling();
    if (InitWS() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(tilingContext, "Tiling initialization finished.");
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus SinkhornTilingArch22::InitTiling()
{
    // 切分流程
    formerNum = totalRow % numBlocks;
    if (formerNum == 0) {
        formerNum = numBlocks;
    }
    tailNum = numBlocks - formerNum;

    formerRow = (totalRow + numBlocks - 1) / numBlocks;
    formerLength = formerRow * totalCol;
    formerTileNum = (formerRow + tileRow - 1) / tileRow;
    formerLastTileRow = formerRow % tileRow;
    if (formerLastTileRow == 0) {
        formerLastTileRow = tileRow;
    }
    formerLastTileLength = formerLastTileRow * totalCol;

    if (tailNum > 0) {
        tailRow = (totalRow - formerRow * formerNum) / tailNum; // 一定能整除
        tailLength = tailRow * totalCol;
        tailTileNum = (tailRow + tileRow - 1) / tileRow;
        tailLastTileRow = tailRow % tileRow;
        if (tailLastTileRow == 0) {
            tailLastTileRow = tileRow;
        }
        tailLastTileLength = tailLastTileRow * totalCol;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SinkhornTilingArch22::InitWS()
{
    // arch22 GM user workspace 布局，与 kernel op_kernel/sinkhorn.inc::InitWS 成对：
    //   header(64B) + d0Global(4*totalRow) + d1Block per-block(4*numBlocks*totalCol)
    //   + d1Global(4*totalCol) + d1GlobalNew(4*totalCol)
    // 区段元素恒按 FP32(4B) 计；FP16 直调时 kernel 侧 sizeof(T)=2，host 过量分配 ⇒ host>=kernel，安全。
    // 不变式 I4：numBlocks 必须等于 kernel GetBlockNum()（d1Block 区大小与 SumD1 循环次数均依赖它）。
    uint64_t workspaceBytes = WORKSPACE_HEADER_SIZE;
    uint64_t bytes = 0;
    uint64_t elements = 0;
    if (!CheckedMul(totalRow, SIZEOF_FP32, bytes) || !CheckedAdd(workspaceBytes, bytes, workspaceBytes) ||
        !CheckedMul(numBlocks, totalCol, elements) || !CheckedMul(elements, SIZEOF_FP32, bytes) ||
        !CheckedAdd(workspaceBytes, bytes, workspaceBytes) || !CheckedMul(totalCol, SIZEOF_FP32, bytes) ||
        !CheckedAdd(workspaceBytes, bytes, workspaceBytes) || !CheckedMul(totalCol, SIZEOF_FP32, bytes) ||
        !CheckedAdd(workspaceBytes, bytes, workspaceBytes)) {
        OP_LOGE_WITHOUT_REPORT(tilingContext, "workspace size calculation overflows for cost shape (%lu, %lu).",
                               static_cast<unsigned long>(totalRow), static_cast<unsigned long>(totalCol));
        return ge::GRAPH_FAILED;
    }
    if (totalRow > 0 && totalCol > 0) {
        if (workspaceBytes > std::numeric_limits<uint64_t>::max() - (CACHE_LINE_SIZE - 1)) {
            OP_LOGE_WITHOUT_REPORT(tilingContext, "workspace alignment overflows for cost shape (%lu, %lu).",
                                   static_cast<unsigned long>(totalRow), static_cast<unsigned long>(totalCol));
            return ge::GRAPH_FAILED;
        }
        workspaceBytes = AlignUp(workspaceBytes, CACHE_LINE_SIZE);
    }
    if (workspaceBytes > std::numeric_limits<size_t>::max()) {
        OP_LOGE_WITHOUT_REPORT(tilingContext, "workspace size does not fit size_t for cost shape (%lu, %lu).",
                               static_cast<unsigned long>(totalRow), static_cast<unsigned long>(totalCol));
        return ge::GRAPH_FAILED;
    }
    userWorkspaceSize = static_cast<size_t>(workspaceBytes);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SinkhornTilingArch22::RunKernelTiling()
{
    OP_LOGD(tilingContext, "Tiling calculation starts.");

    // 内核侧使用 SyncAll 全核同步，需设置 BATCH_MODE_SCHEDULE 调度模式，避免多流场景卡死
    if (tilingContext->SetScheduleMode(BATCH_MODE_SCHEDULE) != ge::GRAPH_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(tilingContext, "SetScheduleMode failed.");
        return ge::GRAPH_FAILED;
    }

    // context传递
    tilingContext->SetTilingKey(tilingKey);
    tilingContext->SetBlockDim(numBlocks);

    // tiling data传递
    tiling.set_formerNum(formerNum);
    tiling.set_formerRow(formerRow);
    tiling.set_formerLength(formerLength);

    tiling.set_formerTileNum(formerTileNum);
    tiling.set_formerLastTileRow(formerLastTileRow);
    tiling.set_formerLastTileLength(formerLastTileLength);

    tiling.set_tailNum(tailNum);
    tiling.set_tailRow(tailRow);
    tiling.set_tailLength(tailLength);

    tiling.set_tailTileNum(tailTileNum);
    tiling.set_tailLastTileRow(tailLastTileRow);
    tiling.set_tailLastTileLength(tailLastTileLength);

    tiling.set_tileRow(tileRow);
    tiling.set_tileLength(tileLength);

    tiling.set_totalRow(totalRow);
    tiling.set_totalCol(totalCol);
    tiling.set_totalColAligned(totalColAligned);

    tiling.set_tol(tol);

    auto rawTilingData = tilingContext->GetRawTilingData();
    if (rawTilingData == nullptr || rawTilingData->GetData() == nullptr ||
        rawTilingData->GetCapacity() < tiling.GetDataSize()) {
        OP_LOGE_WITHOUT_REPORT(tilingContext, "Raw tiling data buffer is null or smaller than %zu bytes.",
                               tiling.GetDataSize());
        return ge::GRAPH_FAILED;
    }
    tiling.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(tiling.GetDataSize());

    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(
        WORKSPACE_BLOCK_NUM); // 通过框架获取workspace的指针，GetWorkspaces入参所需workspace的块数。当前限制使用一块。
    if (currentWorkspace == nullptr) {
        OP_LOGE_WITHOUT_REPORT(tilingContext, "GetWorkspaceSizes returned nullptr.");
        return ge::GRAPH_FAILED;
    }
    auto compileInfo = reinterpret_cast<const SinkhornCompileInfo*>(tilingContext->GetCompileInfo());
    if (compileInfo == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(tilingContext->GetNodeName(), "compileInfo");
        return ge::GRAPH_FAILED;
    }
    if (compileInfo->sysWorkspaceSize > std::numeric_limits<size_t>::max() - userWorkspaceSize) {
        OP_LOGE_WITHOUT_REPORT(tilingContext, "system and user workspace size addition overflows size_t.");
        return ge::GRAPH_FAILED;
    }
    currentWorkspace[0] = userWorkspaceSize + compileInfo->sysWorkspaceSize;

    OP_LOGD(tilingContext, "userWorkspaceSize: %lu.", userWorkspaceSize);
    TilingDataPrint();
    OP_LOGD(tilingContext, "Tiling calculation ends.");
    return ge::GRAPH_SUCCESS;
}

void SinkhornTilingArch22::TilingDataPrint()
{
    OP_LOGD(tilingContext, "            tilingKey: %lu", tilingContext->GetTilingKey());
    OP_LOGD(tilingContext, "             numBlocks: %u", tilingContext->GetBlockDim());

    OP_LOGD(tilingContext, "            formerNum: %lu", tiling.get_formerNum());
    OP_LOGD(tilingContext, "            formerRow: %lu", tiling.get_formerRow());
    OP_LOGD(tilingContext, "         formerLength: %lu", tiling.get_formerLength());

    OP_LOGD(tilingContext, "        formerTileNum: %lu", tiling.get_formerTileNum());
    OP_LOGD(tilingContext, "    formerLastTileRow: %lu", tiling.get_formerLastTileRow());
    OP_LOGD(tilingContext, " formerLastTileLength: %lu", tiling.get_formerLastTileLength());

    OP_LOGD(tilingContext, "              tailNum: %lu", tiling.get_tailNum());
    OP_LOGD(tilingContext, "              tailRow: %lu", tiling.get_tailRow());
    OP_LOGD(tilingContext, "           tailLength: %lu", tiling.get_tailLength());

    OP_LOGD(tilingContext, "          tailTileNum: %lu", tiling.get_tailTileNum());
    OP_LOGD(tilingContext, "      tailLastTileRow: %lu", tiling.get_tailLastTileRow());
    OP_LOGD(tilingContext, "   tailLastTileLength: %lu", tiling.get_tailLastTileLength());

    OP_LOGD(tilingContext, "              tileRow: %lu", tiling.get_tileRow());
    OP_LOGD(tilingContext, "           tileLength: %lu", tiling.get_tileLength());

    OP_LOGD(tilingContext, "             totalRow: %lu", tiling.get_totalRow());
    OP_LOGD(tilingContext, "             totalCol: %lu", tiling.get_totalCol());
    OP_LOGD(tilingContext, "      totalColAligned: %lu", tiling.get_totalColAligned());

    OP_LOGD(tilingContext, "                  tol: %f", tiling.get_tol());

    OP_LOGD(tilingContext, "    userWorkspaceSize: %lu", userWorkspaceSize);
}

ge::graphStatus TilingForSinkhornArch22(gert::TilingContext* context)
{
    SinkhornTilingArch22 tilingObject(context);
    if (tilingObject.Init() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunKernelTiling();
}

ge::graphStatus TilingPrepareForSinkhornArch22(gert::TilingParseContext* context)
{
    OP_LOGD(context, "TilingPrepareForSinkhorn start.");
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "platformInfo");
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    // 将aivNum、workSpaveSize、ubSize的变量放到compileInfo中
    auto compileInfo = context->GetCompiledInfo<SinkhornCompileInfo>();
    if (compileInfo == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "compileInfo");
        return ge::GRAPH_FAILED;
    }

    compileInfo->aivNum = ascendcPlatform.GetCoreNumAiv();
    OP_LOGD(context, "compileInfo->aivNum is %lu.", compileInfo->aivNum);

    compileInfo->sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    OP_LOGD(context, "compileInfo->sysWorkspaceSize is %lu.", compileInfo->sysWorkspaceSize);

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    OP_LOGD(context, "compileInfo->ubSize is %lu.", compileInfo->ubSize);

    OP_LOGD(context, "TilingPrepareForSinkhorn end.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Sinkhorn)
    .Tiling(TilingForSinkhornArch22)
    .TilingParse<SinkhornCompileInfo>(TilingPrepareForSinkhornArch22);
} // namespace optiling
