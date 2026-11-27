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
 * \file masked_fill_range_tiling.cpp
 * \brief MaskedFillRange 算子的 tiling 实现（AscendC 接入，kernel 为 PyPTO-Pro）
 */

#include "masked_fill_range_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "graph/utils/type_utils.h"

// PyPTO 生成的头文件（顺序：tilingkey 在前，tiling 在后），路径由 pypto_include_tiling_gen()
// 注入；2674d3154 起 PyPTO 接入改为显式 include（不再 force-include），与 pdist 接入对齐
#include "MaskedFillRangeTilingKey_tilingkey.h"
#include "MaskedFillRangeTiling_tiling.h"

#include <algorithm>
#include <cstdint>
#include <iterator>

namespace optiling {
// TilingKey 的 Dtype/Win 取值，与 op_kernel/masked_fill_range.py 一致
constexpr uint64_t DTYPE_FP32 = 0;
constexpr uint64_t DTYPE_FP16 = 1;
constexpr uint64_t DTYPE_INT32 = 2;
constexpr uint64_t DTYPE_INT8 = 3;
constexpr uint64_t WIN_VECTOR_CMP = 1;  // 32-bit lane 向量比较域（Win=1）
constexpr uint64_t WIN_PREFIX_MASK = 0; // 数据 dtype 前缀掩码域（Win=0）
// Win=1 的常驻元数据单寄存器容量：64 条 int32 lane（kernel 侧 _LANES_32B 镜像）
constexpr int64_t META_SINGLE_REG_LANES = 64;
// kernel 常驻元数据 tile 的轮容量（_TILE_R）
constexpr int64_t MAX_ROUNDS = 128;
// kernel op_kernel/masked_fill_range.py 的 _TILE_D；任务切分必须与 data tile 对齐。
constexpr int64_t TILE_D = 16384;
// 元素总数上限（契约 E60003）
constexpr int64_t MAX_NUMEL = 1LL << 30;
constexpr size_t MAX_DIMS = 3;
// start/end 窗口表契约 rank（必为 2D [rounds, B]）
constexpr size_t RANK_WINDOW_TABLE = 2;
// fillMode 模式值（跨文件契约：与 op_kernel/masked_fill_range.py 的
// MaskedFillRangeTiling.fillMode 字段注释一一对应，勿单独改动）
constexpr int64_t FILL_MODE_WINDOW = 0; // 行内 clamp 窗口（1D/2D/3D axis2）
constexpr int64_t FILL_MODE_ROW_B = 1;  // 3D axis0 整片填充（k=row//repeat=n）
constexpr int64_t FILL_MODE_ROW_C = 2;  // 3D axis1 整行填充（k=row%repeat=c）
// 支持的数据类型（x/value/y 三者一致）
constexpr ge::DataType SUPPORTED_DTYPES[] = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8};

/*!
 * \brief MaskedFillRange 算子的 tiling 入口
 */
static ge::graphStatus MaskedFillRangeTilingFunc(gert::TilingContext* context)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    int64_t coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "coreNum must be positive, got %ld", coreNum), return ge::GRAPH_FAILED);

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* axisPtr = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, axisPtr);
    int64_t axis = *axisPtr;

    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    const auto& shapeX = inputX->GetStorageShape();
    const size_t dimNum = shapeX.GetDimNum();
    OP_CHECK_IF(dimNum < 1 || dimNum > MAX_DIMS, OP_LOGE(context, "input dim = %zu, should be in [1, 3]", dimNum),
                return ge::GRAPH_FAILED);
    int64_t numel = 1;
    for (size_t i = 0; i < dimNum; ++i) {
        const int64_t dimSize = shapeX.GetDim(i);
        OP_CHECK_IF(dimSize < 0, OP_LOGE(context, "input dimension %zu is negative: %ld", i, dimSize),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(dimSize != 0 && numel > MAX_NUMEL / dimSize, OP_LOGE(context, "input numel exceeds %ld", MAX_NUMEL),
                    return ge::GRAPH_FAILED);
        numel *= dimSize;
    }
    auto xDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    ge::DataType xDtype = xDesc->GetDataType();
    OP_CHECK_IF(
        std::find(std::begin(SUPPORTED_DTYPES), std::end(SUPPORTED_DTYPES), xDtype) == std::end(SUPPORTED_DTYPES),
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "x", Ops::Base::ToString(xDtype).c_str(),
                                  "FLOAT, FLOAT16, INT32 or INT8"),
        return ge::GRAPH_FAILED);
    // 下界校验先于归一化：axis 为外部 REQUIRED attr 可传任意 int64（含
    // INT64_MIN 级极端负值），先拦截 axis < -dimNum 再归一化，保证
    // axis += dimNum 不发生有符号溢出 UB（回绕值虽仍会被下方后置校验拦截，
    // 但 UB 本身不允许发生）
    OP_CHECK_IF(axis < -static_cast<int64_t>(dimNum),
                OP_LOGE(context, "axis %ld out of range for dim %zu", axis, dimNum), return ge::GRAPH_FAILED);
    if (axis < 0) {
        axis += static_cast<int64_t>(dimNum);
    }
    OP_CHECK_IF(axis < 0 || axis >= static_cast<int64_t>(dimNum),
                OP_LOGE(context, "axis %ld out of range for dim %zu", axis, dimNum), return ge::GRAPH_FAILED);

    auto inputStart = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputStart);
    const auto& shapeStart = inputStart->GetStorageShape();
    auto inputEnd = context->GetInputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputEnd);
    const auto& shapeEnd = inputEnd->GetStorageShape();
    OP_CHECK_IF(shapeStart.GetDimNum() != RANK_WINDOW_TABLE || shapeEnd.GetDimNum() != RANK_WINDOW_TABLE ||
                    shapeStart.GetDim(0) != shapeEnd.GetDim(0) || shapeStart.GetDim(1) != shapeEnd.GetDim(1),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context->GetNodeName(), "start and end",
                    ("start=" + Ops::Base::ToString(shapeStart) + ", end=" + Ops::Base::ToString(shapeEnd)).c_str(),
                    "same-shape 2D (rounds, B) tensors"),
                return ge::GRAPH_FAILED);
    auto startDesc = context->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, startDesc);
    auto endDesc = context->GetInputDesc(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, endDesc);
    OP_CHECK_IF(startDesc->GetDataType() != ge::DT_INT32 || endDesc->GetDataType() != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPE(
                    context->GetNodeName(), "start/end",
                    (Ops::Base::ToString(startDesc->GetDataType()) + ", " + Ops::Base::ToString(endDesc->GetDataType()))
                        .c_str(),
                    "both INT32"),
                return ge::GRAPH_FAILED);
    const int64_t num = shapeStart.GetDim(0);
    const int64_t winBatchDim = shapeStart.GetDim(1); // 窗口表第二维 B（winStride 取值来源）
    OP_CHECK_IF(num < 1 || winBatchDim < 1,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context->GetNodeName(), "start and end",
                    ("num=" + std::to_string(num) + ", winBatchDim=" + std::to_string(winBatchDim)).c_str(),
                    "start/end must be same-shape 2D tensors and both dims must be >= 1"),
                return ge::GRAPH_FAILED);

    auto inputValue = context->GetInputShape(3);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputValue);
    const auto& shapeValue = inputValue->GetStorageShape();
    OP_CHECK_IF(shapeValue.GetDimNum() != 1 || shapeValue.GetDim(0) != num,
                OP_LOGE(context, "value must be 1D with length start.size(0) %ld", num), return ge::GRAPH_FAILED);
    auto valueDesc = context->GetInputDesc(3);
    OP_CHECK_NULL_WITH_CONTEXT(context, valueDesc);
    OP_CHECK_IF(valueDesc->GetDataType() != xDtype,
                OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "value",
                                          Ops::Base::ToString(valueDesc->GetDataType()).c_str(),
                                          Ops::Base::ToString(xDtype).c_str()),
                return ge::GRAPH_FAILED);

    int64_t rows = 1;
    int64_t cols = 1;
    int64_t repeat = 1;
    int64_t batches = winBatchDim;
    int64_t rounds = num;
    int64_t fillMode = FILL_MODE_WINDOW;
    if (dimNum == 1) {
        rows = 1;
        cols = shapeX.GetDim(0);
        repeat = 1;
        batches = std::min(winBatchDim, rows);
        if (winBatchDim != 1) {
            // TIK 基准 1D 每轮只读 start/end[i,0]（start_ub[0]），其余窗口列忽略；
            // 告警提示潜在 shape 配置问题，不拒绝执行
            OP_LOGW(context->GetNodeName(), "start/end 1D window batch dim %ld > 1, extra window columns ignored",
                    winBatchDim);
        }
    } else if (dimNum == 2) { // 2D [rows, cols] 行视图
        rows = shapeX.GetDim(0);
        cols = shapeX.GetDim(1);
        repeat = 1;
        batches = std::min(winBatchDim, rows);
        if (winBatchDim != rows) {
            // TIK 基准按 min(B, rows) 对齐（golden _rows_fit/_mask_fit 同构）：
            // B < rows 时尾部行不填充（保持 x 原值），B > rows 时尾部窗口列丢弃。
            // proto 文档名义契约 B == rows，不等时告警提示静默部分填充风险
            OP_LOGW(context->GetNodeName(), "start/end window batch dim %ld != x rows %ld, tail rows keep x unchanged",
                    winBatchDim, rows);
        }
    } else {                                        // 3D [N, C, W]：行视图行数 = N*C
        const int64_t dim0Batch = shapeX.GetDim(0); // N：批数
        const int64_t dim1Rows = shapeX.GetDim(1);  // C：每批行数（repeat 取值来源）
        cols = shapeX.GetDim(2);
        rows = dim0Batch * dim1Rows;
        repeat = dim1Rows;
        batches = std::min(winBatchDim, dim0Batch);
        if (winBatchDim != dim0Batch) {
            OP_LOGW(context->GetNodeName(),
                    "start/end window batch dim %ld != x batch dim %ld, tail batches keep x unchanged", winBatchDim,
                    dim0Batch);
        }
        if (axis == 0) {
            fillMode = FILL_MODE_ROW_B;
        } else if (axis == 1) {
            fillMode = FILL_MODE_ROW_C;
        }
    }
    // 先完成全部输入与资源校验，再写 tiling 输出（blockDim/tilingKey/tilingData）。
    // 空 tensor（numel==0，行视图等价 rows==0 || cols==0，负维已在上方拦截）：
    // no-op 放行，对齐 TIK 基准能力不收窄（SE 报告 §3.6.1 CUDA gpu_kernel
    // numel==0 不 launch kernel 参照；基准崩溃类形状 1D(0,)/2D(R,0)/3D(0,C,W)/
    // (N,0,W) 在本实现为超集兼容）。不提前 return：窗口表 rank/shape、dtype 面、
    // value 长度、rounds×batches 容量、axis 等契约校验对本分支保持全量执行。
    // batches 下钳 1 保证下方 rounds > MAX_ROUNDS / batches 有符号整除不发生
    // 除零 UB（与下方"显式拦截不依赖隐式推导链"防御原则一致）；3D 空 middle
    // 维时 min(B, dim0Batch) 名义值保持不变（仅防 0），容量校验对空/非空分支
    // 语义一致，不按 numel 分叉校验集。
    if (rows == 0 || cols == 0) {
        batches = std::max<int64_t>(1, batches);
    } else {
        // 本分支恒有 rows/cols >= 1（负维已在上方拦截，0 维已走空 tensor 分支）：
        // 下方三处 <1 拦截在当前布局下不可达，保留为防御性双保险——显式拦截
        // 保证下方有符号整除 MAX_ROUNDS / batches 不发生除零 UB，不依赖空分支
        // 钳制或负维校验的隐式推导链（重排校验顺序即引入 UB）
        OP_CHECK_IF(cols < 1,
                    OP_LOGE_FOR_INVALID_SHAPESIZE(context->GetNodeName(), "x fill-axis", std::to_string(cols).c_str(),
                                                  "greater than 0"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(rows < 1,
                    OP_LOGE_FOR_INVALID_SHAPESIZE(context->GetNodeName(), "x leading dims",
                                                  std::to_string(rows).c_str(), "greater than 0"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(batches < 1,
                    OP_LOGE_FOR_INVALID_SHAPESIZE(context->GetNodeName(), "window batches",
                                                  std::to_string(batches).c_str(), "greater than 0"),
                    return ge::GRAPH_FAILED);
    }
    // rounds 即上文 num（num < 1 已拦截，恒 >= 1），不重复校验；除法分母
    // batches 恒 >= 1（空 tensor 分支下钳 1，非空分支显式拦截）
    OP_CHECK_IF(
        rounds > MAX_ROUNDS / batches,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context->GetNodeName(), "start/end rounds and batches",
            ("rounds=" + std::to_string(rounds) + ", batches=" + std::to_string(batches)).c_str(),
            ("rounds * batches must not exceed resident metadata capacity " + std::to_string(MAX_ROUNDS)).c_str()),
        return ge::GRAPH_FAILED);

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = 0;

    // Tiling struct and TPL key declaration are generated from op_kernel/masked_fill_range.py
    // (explicitly included above), matching the pdist integration.
    MaskedFillRangeTiling* tilingData = context->GetTilingData<MaskedFillRangeTiling>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tilingData);
    tilingData->rows = rows;
    tilingData->cols = cols;
    tilingData->rounds = rounds;
    tilingData->batches = batches;
    // GM 窗口步长用 start 第二维原值 winBatchDim（≠min 后的 batches）：kernel 窗口 (i, win) 的
    // GM flat 索引为 i*winStride+win，1D 等 batches 截断场景若误用 batches 作
    // 步长会错位读成第 0 轮的后续窗口列
    tilingData->winStride = winBatchDim;
    tilingData->repeat = repeat;
    tilingData->fillMode = fillMode;

    // 按 (row, 列块) 分发任务，避免 1D 长向量只有 rows=1 时仅启动一个 block。
    // 总任务数不超过 numel，且 numel 已限制在 2^30；空 tensor（taskCount==0）
    // kernel 按 pl.range 零跳 no-op，blockDim 下钳 1 防 SetBlockDim(0)。
    const int64_t colBlocks = (cols + TILE_D - 1) / TILE_D;
    const int64_t taskCount = rows * colBlocks;
    const int64_t workers = std::max<int64_t>(1, std::min(coreNum, taskCount));
    context->SetBlockDim(static_cast<uint32_t>(workers));
    uint64_t dtypeKey = (xDtype == ge::DT_FLOAT)   ? DTYPE_FP32 :
                        (xDtype == ge::DT_FLOAT16) ? DTYPE_FP16 :
                        (xDtype == ge::DT_INT32)   ? DTYPE_INT32 :
                                                     DTYPE_INT8;
    // Win=1 门限按窗口表实际占用：最大 gather 索引 (rounds-1)*winBatchDim+batches（B 大时
    // 窗口表跨度远超 rounds*batches，须按跨度卡容量，防越界 _TILE_R=128）。
    // 跨文件隐式契约：Win=1 仅允许 fillMode==0 且 Dtype∈{FP32, INT32}（32-bit
    // lane 向量比较域），kernel 侧 _fill_tile_cmp 对 FP16/INT8 为空实现（静默
    // y=x 直通）——若放宽本门限的 dtype 白名单或 fillMode 限制，必须同步补齐
    // op_kernel/masked_fill_range.py 的 _fill_tile_cmp 实现；VF 域不支持
    // pl.pto_assert（asc_opc 实证 F00005，见 dev record U2/9.2），kernel 侧
    // 无编译期断言兜底，本门限是该契约的唯一防线
    const uint64_t win = (fillMode == FILL_MODE_WINDOW && (dtypeKey == DTYPE_FP32 || dtypeKey == DTYPE_INT32) &&
                          (rounds - 1) * winBatchDim + batches <= META_SINGLE_REG_LANES) ?
                             WIN_VECTOR_CMP :
                             WIN_PREFIX_MASK;
    context->SetTilingKey(GET_TPL_TILING_KEY(dtypeKey, win));

    OP_LOGD(context->GetNodeName(),
            "tiling: rows=%ld cols=%ld colBlocks=%ld tasks=%ld rounds=%ld batches=%ld winStride=%ld repeat=%ld "
            "fillMode=%ld",
            rows, cols, colBlocks, taskCount, rounds, batches, winBatchDim, repeat, fillMode);
    return ge::GRAPH_SUCCESS;
}

/*!
 * \brief 解析 MaskedFillRange 的编译信息，本算子无静态 tiling 信息
 */
static ge::graphStatus TilingParseForMaskedFillRange([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// tiling 注册入口
IMPL_OP_OPTILING(MaskedFillRange)
    .Tiling(MaskedFillRangeTilingFunc)
    .TilingParse<MaskedFillRangeCompileInfo>(TilingParseForMaskedFillRange);
} // namespace optiling
