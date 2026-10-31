/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file strided_slice_assign_v2_tiling_arch35.cpp
 * \brief A5 (arch35) Tiling：S0 公共流程（合轴归一化 + 判据路由）+ T1/T2/T3 三模板 tiling
 */

#include <cstdint>
#include <string>

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "platform/platform_ascendc.h"

#include "strided_slice_assign_v2_tiling_arch35.h"

namespace optiling {
namespace {

// ---------------------------------------------------------------------------
// 小工具：整数运算
// ---------------------------------------------------------------------------
inline int64_t CeilDivPos(int64_t a, int64_t b) { return (a <= 0 || b <= 0) ? 0 : (a + b - 1) / b; }
inline int64_t AlignUpI64(int64_t v, int64_t align) { return (align <= 0) ? v : (v + align - 1) / align * align; }
inline int64_t AlignDownI64(int64_t v, int64_t align) { return (align <= 0) ? v : v / align * align; }
inline int64_t MinI64(int64_t a, int64_t b) { return (a < b) ? a : b; }
inline int64_t MaxI64(int64_t a, int64_t b) { return (a > b) ? a : b; }
inline int64_t AbsI64(int64_t v) { return (v < 0) ? -v : v; }

// floor / ceil 除法（带符号语义，用于窗口内 coord 取值范围的闭式反解）
inline int64_t FloorDivI64(int64_t a, int64_t b)
{
    int64_t q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) {
        --q;
    }
    return q;
}
inline int64_t CeilDivI64(int64_t a, int64_t b)
{
    int64_t q = a / b;
    if ((a % b != 0) && ((a < 0) == (b < 0))) {
        ++q;
    }
    return q;
}

// 一维形状渲染为 "[d0, d1, ...]"，供日志 reason 中给出期望形状
inline std::string ShapeToStr(const int64_t* dims, int64_t n)
{
    std::string str = "[";
    for (int64_t i = 0; i < n; ++i) {
        str += (i == 0) ? "" : ", ";
        str += std::to_string(dims[i]);
    }
    return str + "]";
}

inline bool RangeShapesUnequal(const int64_t* lhs, const int64_t* rhs, int64_t n)
{
    for (int64_t i = 0; i < n; ++i) {
        if (lhs[i] != rhs[i]) {
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// 常量输入读取（begin / end / strides / axes 值依赖常量，int32/int64 兼容）
// ---------------------------------------------------------------------------
inline const char* ConstInputName(size_t idxInput)
{
    static const char* const kNames[] = {"var", "input_value", "begin", "end", "strides", "axes"};
    return (idxInput < sizeof(kNames) / sizeof(kNames[0])) ? kNames[idxInput] : "const_input";
}
// begin/end/strides/axes 仅接受 DT_INT64（OpDef 中 idxDataType 唯一可选）。
ge::graphStatus ReadConstVector(gert::TilingContext* context, size_t idxInput, int64_t* out, int64_t& len)
{
    auto tensor = context->GetInputTensor(idxInput);
    OP_CHECK_NULL_WITH_CONTEXT(context, tensor);
    len = static_cast<int64_t>(tensor->GetShapeSize());
    OP_CHECK_IF(len > SSA_MAX_DIM_NUM,
                OP_LOGE_FOR_INVALID_LISTSIZE_WITH_REASON(
                    context->GetNodeName(), ConstInputName(idxInput), std::to_string(len).c_str(),
                    ("The element nums of " + std::string(ConstInputName(idxInput)) +
                     " must be less than or equal to " + std::to_string(SSA_MAX_DIM_NUM) + ".")
                        .c_str()),
                return ge::GRAPH_FAILED);
    const ge::DataType dtype = tensor->GetDataType();
    OP_CHECK_IF(dtype != ge::DT_INT64,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    context->GetNodeName(), ConstInputName(idxInput), Ops::Base::ToString(dtype).c_str(),
                    ("The dtype of " + std::string(ConstInputName(idxInput)) + " must be DT_INT64.").c_str()),
                return ge::GRAPH_FAILED);

    const int64_t* dataPtr = tensor->GetData<int64_t>();
    OP_CHECK_NULL_WITH_CONTEXT(context, dataPtr);
    for (int64_t i = 0; i < len; ++i) {
        out[i] = dataPtr[i];
    }
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// 约简数据模型（合轴归一化产物，三模板统一口径）
// ---------------------------------------------------------------------------
struct SsaReducedShape {
    int32_t dimNum = 0;                         // n'
    int64_t sliceShape[SSA_MAX_DIM_NUM] = {};   // S'[]
    int64_t axisStep[SSA_MAX_DIM_NUM] = {};     // T'[]（含符号）
    int64_t sliceCum[SSA_MAX_DIM_NUM] = {};     // S' 后缀积
    int64_t varBase = 0;                        // 切片首元素 var 线性偏移
    int64_t sliceElemNum = 0;                   // = Π S'[]
    int32_t directionId = SSA_DIR_ALL_POSITIVE; // 0=全正 1=全负 2=混合
};

/**
 * @brief 合轴归一化：自最内维向外合并「满覆盖且步长连续」的相邻轴
 * @note  连续判据 T[k-1] == S[k]·T[k]；由 S[k] >= 0 可知异号相邻对恒不满足，故合轴只发生在同号轴之间
 */
void CollapseAxes(int64_t n, const int64_t* slice, const int64_t* step, const int64_t* begin, const int64_t* varCum,
                  SsaReducedShape& red)
{
    int64_t varBase = 0;
    for (int64_t d = 0; d < n; ++d) {
        varBase += begin[d] * varCum[d + 1];
    }

    int64_t groupShape[SSA_MAX_DIM_NUM] = {};
    int64_t groupStep[SSA_MAX_DIM_NUM] = {};
    int32_t groupNum = 0;
    int64_t curShape = slice[n - 1];
    int64_t curStep = step[n - 1]; // 组内步长恒取最内维步长
    for (int64_t k = n - 1; k >= 1; --k) {
        if (step[k - 1] == slice[k] * step[k]) {
            curShape = slice[k - 1] * curShape; // 吸收轴 k-1
        } else {
            groupShape[groupNum] = curShape;
            groupStep[groupNum] = curStep;
            ++groupNum;
            curShape = slice[k - 1];
            curStep = step[k - 1];
        }
    }
    groupShape[groupNum] = curShape;
    groupStep[groupNum] = curStep;
    ++groupNum;

    red.dimNum = groupNum;
    for (int32_t i = 0; i < groupNum; ++i) {
        red.sliceShape[i] = groupShape[groupNum - 1 - i]; // emit 序为「内 → 外」，反转为轴序
        red.axisStep[i] = groupStep[groupNum - 1 - i];
    }

    int64_t cum = 1;
    for (int32_t i = groupNum - 1; i >= 0; --i) {
        cum *= red.sliceShape[i];
        red.sliceCum[i] = cum;
    }
    red.sliceElemNum = cum;
    red.varBase = varBase;

    bool hasPositive = false;
    bool hasNegative = false;
    for (int32_t i = 0; i < groupNum; ++i) {
        if (red.axisStep[i] > 0) {
            hasPositive = true;
        } else {
            hasNegative = true;
        }
    }
    red.directionId = (hasPositive && hasNegative) ? SSA_DIR_MIXED :
                                                     (hasNegative ? SSA_DIR_ALL_NEGATIVE : SSA_DIR_ALL_POSITIVE);
}

// ---------------------------------------------------------------------------
// 稀疏度 validNumInCacheLine（约简形状）
// ---------------------------------------------------------------------------
struct SsaCacheLineScan {
    int64_t n = 0;
    const int64_t* shape = nullptr;
    const int64_t* step = nullptr;
    int64_t window = 1;                       // 窗口长度（元素）= cacheLineElems
    int64_t remMin[SSA_MAX_DIM_NUM + 1] = {}; // dims>=dim 的最小 / 最大增量（含该维）
    int64_t remMax[SSA_MAX_DIM_NUM + 1] = {};
    int64_t count = 0;
};

void SsaScanWindow(SsaCacheLineScan& st, int64_t dim, int64_t partial)
{
    if (st.count > SSA_SPARSITY_THRESHOLD) {
        return; // 只需判定「是否 > 阈值」，可提前退出
    }
    if (dim == st.n) {
        ++st.count;
        return;
    }
    const int64_t t = st.step[dim];
    // 与窗口相交要求 partial + coord·t ∈ [ -remMax, window-1-remMin ]
    const int64_t lo = -st.remMax[dim + 1] - partial;
    const int64_t hi = st.window - 1 - st.remMin[dim + 1] - partial;
    int64_t c0 = 0;
    int64_t c1 = 0;
    if (t > 0) {
        c0 = CeilDivI64(lo, t);
        c1 = FloorDivI64(hi, t);
    } else {
        c0 = CeilDivI64(hi, t);
        c1 = FloorDivI64(lo, t);
    }
    c0 = MaxI64(c0, 0);
    c1 = MinI64(c1, st.shape[dim] - 1);
    for (int64_t c = c0; c <= c1; ++c) {
        SsaScanWindow(st, dim + 1, partial + c * t);
        if (st.count > SSA_SPARSITY_THRESHOLD) {
            return;
        }
    }
}

int64_t CalcValidNumInCacheLineImpl(const SsaReducedShape& red, int64_t elemSize)
{
    if (red.dimNum <= 0 || red.sliceElemNum <= 0 || elemSize <= 0) {
        return 0;
    }
    const int64_t cacheLineElems = SSA_SPARSE_WINDOW_BYTES / elemSize;
    const int64_t innerShape = red.sliceShape[red.dimNum - 1];
    const int64_t innerStep = AbsI64(red.axisStep[red.dimNum - 1]);
    if (innerStep == 1) {
        // 最内约简轴连续 → 非稀疏
        return MinI64(innerShape, cacheLineElems);
    }
    if (innerShape >= cacheLineElems) {
        // 窗口不跨内层行，闭式
        const int64_t checkNum = MinI64((innerShape - 1) * innerStep + 1, cacheLineElems);
        return CeilDivPos(checkNum, innerStep);
    }
    // 窗口跨内层行：有界枚举。步长取绝对值：窗口按首元素正向 [0, W) 建模，
    // 若保留负号则全负方向所有偏移 ≤0、仅锚点落入窗口，validNum 恒 1（方向不对称）；
    // 仅在方向非混合时求值（各维同号），取绝对值后间距结构与全正方向完全一致
    int64_t absStep[SSA_MAX_DIM_NUM] = {};
    for (int32_t d = 0; d < red.dimNum; ++d) {
        absStep[d] = AbsI64(red.axisStep[d]);
    }
    SsaCacheLineScan st;
    st.n = red.dimNum;
    st.shape = red.sliceShape;
    st.step = absStep;
    st.window = cacheLineElems;
    st.remMin[red.dimNum] = 0;
    st.remMax[red.dimNum] = 0;
    for (int64_t d = red.dimNum - 1; d >= 0; --d) {
        const int64_t span = (red.sliceShape[d] - 1) * absStep[d];
        st.remMin[d] = st.remMin[d + 1] + MinI64(span, 0);
        st.remMax[d] = st.remMax[d + 1] + MaxI64(span, 0);
    }
    SsaScanWindow(st, 0, 0);
    return MinI64(st.count, red.sliceElemNum);
}

int64_t CalcValidNumInCacheLine(gert::TilingContext* context, const SsaReducedShape& red, int64_t elemSize)
{
    const int64_t validNum = CalcValidNumInCacheLineImpl(red, elemSize);
    const int64_t cacheLineElems = (elemSize > 0) ? (SSA_SPARSE_WINDOW_BYTES / elemSize) : 0;
    OP_LOGI(context, "===== CalcValidNumInCacheLine =====");
    OP_LOGI(context, "dimNum = %d", red.dimNum);
    OP_LOGI(context, "elemSize = %ld", elemSize);
    OP_LOGI(context, "cacheLineElems = %ld", cacheLineElems);
    OP_LOGI(context, "innerShape = %ld", (red.dimNum > 0) ? red.sliceShape[red.dimNum - 1] : 0);
    OP_LOGI(context, "innerStep = %ld", (red.dimNum > 0) ? AbsI64(red.axisStep[red.dimNum - 1]) : 0);
    OP_LOGI(context, "sliceElemNum = %ld", red.sliceElemNum);
    OP_LOGI(context, "validNum = %ld", validNum);
    OP_LOGI(context, "sparsityThreshold = %d", SSA_SPARSITY_THRESHOLD);
    OP_LOGI(context, "hitSparse = %d", (validNum <= SSA_SPARSITY_THRESHOLD) ? 1 : 0);
    return validNum;
}

/** 连续段长度（约简形状单轴读取）：|T'[n'-1]| != 1 时恒为 1，避免把跨步元素误判为连续段 */
int64_t CalcContigRun(const SsaReducedShape& red)
{
    const int64_t innerStep = AbsI64(red.axisStep[red.dimNum - 1]);
    return (innerStep == 1) ? red.sliceShape[red.dimNum - 1] : 1;
}

// T2 逐段 DMA 要求每个段首 32B 对齐，故 varBase 与全部外层 axisStep（字节）均须 32B 对齐；
// 不满足则判 T3（Scatter 逐元素寻址无对齐约束）。
bool SsaContigSegsAligned32B(const SsaReducedShape& red, int64_t elemSize)
{
    if ((red.varBase * elemSize) % SSA_ALIGN_32_BYTES != 0) {
        return false;
    }
    for (int32_t d = 0; d + 1 < red.dimNum; ++d) {
        if ((AbsI64(red.axisStep[d]) * elemSize) % SSA_ALIGN_32_BYTES != 0) {
            return false;
        }
    }
    return true;
}

template <typename TilingT>
void CopyReducedArrays(const SsaReducedShape& red, TilingT& tiling)
{
    for (int32_t i = 0; i < red.dimNum; ++i) {
        tiling.sliceShape[i] = red.sliceShape[i];
        tiling.axisStep[i] = red.axisStep[i];
    }
}

// ---------------------------------------------------------------------------
// 原始输入打印（OP_LOGI，默认不输出）
// ---------------------------------------------------------------------------
void PrintShapeDims(gert::TilingContext* context, const char* name, const gert::StorageShape* shape)
{
    const gert::Shape& s = shape->GetStorageShape();
    const int64_t n = static_cast<int64_t>(s.GetDimNum());
    OP_LOGI(context, "%s.dimNum = %ld", name, n);
    for (int64_t i = 0; i < n; ++i) {
        OP_LOGI(context, "%s[%ld] = %ld", name, i, s.GetDim(i));
    }
}

void PrintRawVector(gert::TilingContext* context, const char* name, const int64_t* arr, int64_t len)
{
    OP_LOGI(context, "%s.len = %ld", name, len);
    for (int64_t i = 0; i < len; ++i) {
        OP_LOGI(context, "%s[%ld] = %ld", name, i, arr[i]);
    }
}

// ---------------------------------------------------------------------------
// TilingData 全字段打印（Host↔Kernel 契约自检；OP_LOGI，默认不输出）
// ---------------------------------------------------------------------------
template <size_t N>
void PrintTilingArray(gert::TilingContext* context, const char* name, const int64_t (&arr)[N], int32_t dimNum)
{
    for (int32_t d = 0; d < dimNum && d < static_cast<int32_t>(N); ++d) {
        OP_LOGI(context, "  %s[%d] = %ld", name, d, arr[d]);
    }
}

void PrintTilingInfo(gert::TilingContext* context, const StridedSliceAssignV2HybridTilingData& t)
{
    OP_LOGI(context, "===== TilingData T1 StridedSliceAssignV2Hybrid =====");
    OP_LOGI(context, "dimNum = %ld", t.dimNum);
    OP_LOGI(context, "sliceElemNum = %ld", t.sliceElemNum);
    OP_LOGI(context, "blockDim = %ld", t.blockDim);
    OP_LOGI(context, "elemPerCore = %ld", t.elemPerCore);
    OP_LOGI(context, "lastCoreElemNum = %ld", t.lastCoreElemNum);
    OP_LOGI(context, "varBase = %ld", t.varBase);
    PrintTilingArray(context, "sliceShape", t.sliceShape, static_cast<int32_t>(t.dimNum));
    PrintTilingArray(context, "axisStep", t.axisStep, static_cast<int32_t>(t.dimNum));
    OP_LOGI(context, "ubFactor = %d", t.ubFactor);
}

void PrintTilingInfo(gert::TilingContext* context, const StridedSliceAssignV2ContigTilingData& t)
{
    OP_LOGI(context, "===== TilingData T2 StridedSliceAssignV2Contig =====");
    OP_LOGI(context, "templateId = %d", t.templateId);
    OP_LOGI(context, "directionId = %d", t.directionId);
    OP_LOGI(context, "dimNum = %ld", t.dimNum);
    OP_LOGI(context, "sliceElemNum = %ld", t.sliceElemNum);
    OP_LOGI(context, "contigRun = %ld", t.contigRun);
    OP_LOGI(context, "segNum = %ld", t.segNum);
    OP_LOGI(context, "blockDim = %ld", t.blockDim);
    OP_LOGI(context, "chunksPerSeg = %ld", t.chunksPerSeg);
    OP_LOGI(context, "chunkLen = %ld", t.chunkLen);
    OP_LOGI(context, "chunkRem = %ld", t.chunkRem);
    OP_LOGI(context, "totalChunks = %ld", t.totalChunks);
    OP_LOGI(context, "chunksPerCore = %ld", t.chunksPerCore);
    OP_LOGI(context, "lastCoreChunks = %ld", t.lastCoreChunks);
    OP_LOGI(context, "varBase = %ld", t.varBase);
    PrintTilingArray(context, "sliceShape", t.sliceShape, static_cast<int32_t>(t.dimNum));
    PrintTilingArray(context, "axisStep", t.axisStep, static_cast<int32_t>(t.dimNum));
    PrintTilingArray(context, "segCum", t.segCum, static_cast<int32_t>(t.dimNum));
    OP_LOGI(context, "ubFactor = %d", t.ubFactor);
}

void PrintTilingInfo(gert::TilingContext* context, const StridedSliceAssignV2ScatterTilingData& t)
{
    OP_LOGI(context, "===== TilingData T3 StridedSliceAssignV2Scatter =====");
    OP_LOGI(context, "ubAxis = %d", static_cast<int32_t>(t.ubAxis));
    OP_LOGI(context, "ubFactor = %u", t.ubFactor);
    OP_LOGI(context, "ubPerCount = %u", t.ubPerCount);
    OP_LOGI(context, "ubTotalCount = %u", t.ubTotalCount);
    OP_LOGI(context, "tileElemNum = %u", t.tileElemNum);
    OP_LOGI(context, "idxDtypeSize = %d", t.idxDtypeSize);
    OP_LOGI(context, "dimNum = %ld", t.dimNum);
    OP_LOGI(context, "innerElems = %ld", t.innerElems);
    OP_LOGI(context, "rows = %ld", t.rows);
    OP_LOGI(context, "cols = %ld", t.cols);
    OP_LOGI(context, "ubSpan = %ld", t.ubSpan);
    OP_LOGI(context, "pitchUb = %ld", t.pitchUb);
    OP_LOGI(context, "varBase = %ld", t.varBase);
    PrintTilingArray(context, "sliceShape", t.sliceShape, static_cast<int32_t>(t.dimNum));
    PrintTilingArray(context, "axisStep", t.axisStep, static_cast<int32_t>(t.dimNum));
    PrintTilingArray(context, "sliceCum", t.sliceCum, static_cast<int32_t>(t.dimNum));
    PrintTilingArray(context, "rowStepVar", t.rowStepVar, static_cast<int32_t>(t.dimNum));
    PrintTilingArray(context, "blkStartVar", t.blkStartVar, static_cast<int32_t>(t.dimNum));
}

void PrintTilingInfo(gert::TilingContext* context, const StridedSliceAssignV2EmptyTilingData& t)
{
    OP_LOGI(context, "===== TilingData T4 StridedSliceAssignV2Empty =====");
    OP_LOGI(context, "blockNum = %u", t.blockNum);
}

// ---------------------------------------------------------------------------
// T4 StridedSliceAssignV2Empty：var 含 0 长度维或切片区域为空时，无元素可写，仅铺 1 核直接返回
// ---------------------------------------------------------------------------
ge::graphStatus FillEmptyTiling(gert::TilingContext* context)
{
    auto* tiling = context->GetTilingData<StridedSliceAssignV2EmptyTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->blockNum = 1;

    context->SetTilingKey(GET_TPL_TILING_KEY(SSA_TPL_EMPTY));
    context->SetBlockDim(1);
    PrintTilingInfo(context, *tiling);

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// T1 StridedSliceAssignV2Hybrid
// ---------------------------------------------------------------------------
ge::graphStatus FillHybridTiling(gert::TilingContext* context, const SsaReducedShape& red, int64_t coreNum,
                                 int64_t ubUsable, int64_t elemSize)
{
    const int64_t coreLimit = MaxI64(coreNum, 1);
    const int64_t blockDim = MinI64(MaxI64(CeilDivPos(red.sliceElemNum, SSA_MIN_ELEM_PER_CORE), 1), coreLimit);
    const int64_t elemPerCore = CeilDivPos(red.sliceElemNum, blockDim);
    const int64_t lastCoreElemNum = red.sliceElemNum - (blockDim - 1) * elemPerCore;

    // ubFactor：UB 预算扣 paramBuf 后按双缓冲均分，向 16 元素粒度（跨 dtype 保守上界）向下取整
    int64_t ubFactor = AlignDownI64((ubUsable - SSA_PARAM_BUF_BYTES) / (SSA_BUFFER_NUM * elemSize),
                                    SSA_ALIGN_UNIT_ELEMS_STEP);
    ubFactor = MaxI64(ubFactor, SSA_ALIGN_UNIT_ELEMS_STEP);

    auto* tiling = context->GetTilingData<StridedSliceAssignV2HybridTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->dimNum = red.dimNum;
    tiling->sliceElemNum = red.sliceElemNum;
    tiling->blockDim = blockDim;
    tiling->elemPerCore = elemPerCore;
    tiling->lastCoreElemNum = lastCoreElemNum;
    tiling->varBase = red.varBase;
    CopyReducedArrays(red, *tiling);
    tiling->ubFactor = static_cast<int32_t>(ubFactor);

    context->SetTilingKey(GET_TPL_TILING_KEY(SSA_TPL_HYBRID));
    context->SetBlockDim(static_cast<uint32_t>(blockDim));
    PrintTilingInfo(context, *tiling);
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// T2 StridedSliceAssignV2Contig
// ---------------------------------------------------------------------------
ge::graphStatus FillContigTiling(gert::TilingContext* context, const SsaReducedShape& red, int64_t coreNum,
                                 int64_t ubUsable, int64_t elemSize)
{
    const int64_t cacheLineElems = SSA_T2_MIN_CHUNK_BYTES / elemSize;
    const int64_t contigRun = CalcContigRun(red);
    const int64_t segNum = (contigRun > 0) ? (red.sliceElemNum / contigRun) : 0;
    OP_CHECK_IF(segNum <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "segNum", std::to_string(segNum).c_str(),
                                                      "The value of segNum must be greater than 0."),
                return ge::GRAPH_FAILED);

    const int64_t alignGran = MaxI64(SSA_ALIGN_32_BYTES / elemSize, 1);
    const int64_t perElemBytes = (red.directionId == SSA_DIR_ALL_NEGATIVE) ? (SSA_NEG_SLOT_NUM * elemSize) :
                                                                             (SSA_BUFFER_NUM * elemSize);
    int64_t ubFactor = AlignDownI64(ubUsable / perElemBytes, alignGran);
    ubFactor = MaxI64(ubFactor, alignGran);

    const int64_t coreLimit = MaxI64(coreNum, 1);
    const int64_t wantChunks = (segNum < coreLimit) ? CeilDivPos(coreLimit, segNum) : 1;
    const int64_t ubChunks = CeilDivPos(contigRun, ubFactor);
    int64_t chunksPerSeg = MaxI64(MaxI64(1, wantChunks), ubChunks);

    chunksPerSeg = MinI64(chunksPerSeg, MaxI64(1, contigRun / cacheLineElems));
    chunksPerSeg = MaxI64(chunksPerSeg, 1);
    const int64_t chunkLen = contigRun / chunksPerSeg;
    const int64_t chunkRem = contigRun % chunksPerSeg;

    const int64_t totalChunks = segNum * chunksPerSeg;
    const int64_t chunksPerCore = (totalChunks > coreLimit) ? CeilDivPos(totalChunks, coreLimit) : 1;
    const int64_t blockDim = (totalChunks == 0) ? 1 : MaxI64(CeilDivPos(totalChunks, MaxI64(chunksPerCore, 1)), 1);
    const int64_t lastCoreChunks = totalChunks - (blockDim - 1) * chunksPerCore;

    auto* tiling = context->GetTilingData<StridedSliceAssignV2ContigTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->templateId = SSA_TPL_CONTIG;
    tiling->directionId = red.directionId;
    tiling->dimNum = red.dimNum;
    tiling->sliceElemNum = red.sliceElemNum;
    tiling->contigRun = contigRun;
    tiling->segNum = segNum;
    tiling->blockDim = blockDim;
    tiling->chunksPerSeg = chunksPerSeg;
    tiling->chunkLen = chunkLen;
    tiling->chunkRem = chunkRem;
    tiling->totalChunks = totalChunks;
    tiling->chunksPerCore = chunksPerCore;
    tiling->lastCoreChunks = lastCoreChunks;
    tiling->varBase = red.varBase;
    CopyReducedArrays(red, *tiling);
    // segCum[]：段形状后缀积（基于约简形状，最内维置 1），段序号反解基
    int64_t cum = 1;
    for (int32_t j = red.dimNum - 1; j >= 0; --j) {
        const int64_t segShape = (j == red.dimNum - 1) ? 1 : red.sliceShape[j];
        cum *= segShape;
        tiling->segCum[j] = cum;
    }
    tiling->ubFactor = static_cast<int32_t>(ubFactor);

    context->SetTilingKey(GET_TPL_TILING_KEY(SSA_TPL_CONTIG));
    context->SetBlockDim(static_cast<uint32_t>(blockDim));
    PrintTilingInfo(context, *tiling);
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// T3 StridedSliceAssignV2Scatter
// ---------------------------------------------------------------------------
struct SsaScatterSolver {
    int64_t n = 0;
    const int64_t* shape = nullptr;
    const int64_t* step = nullptr;
    int64_t elemSize = 1;
    int64_t idxDtypeSize = 2; // Scatter 索引字节数（2/4，按 dtype 与 Reg::Scatter 配对）
    int64_t sL = 1;           // |axisStep[n-1]|
    int64_t c1 = 0;           // C1 = idxDtypeSize + BUFFER_NUM·sizeof(T)
    int64_t aSpan = 0;        // A = sL·sizeof(T)（var 侧单元素字节跨距，度量基准）
    int64_t gran = 1;         // GRAN = 32/sizeof(T)
    int64_t ubBudget = 0;     // UB' = UB_USABLE - fixed_ub
    int64_t maxIdxValue = 0;  // 索引位宽上限
    int64_t maxTblElems = 0;  // 索引表条目数上限
    int64_t coreNum = 1;

    int64_t InnerElems(int64_t a) const
    {
        int64_t prod = 1;
        for (int64_t d = a + 1; d < n; ++d) {
            prod *= shape[d];
        }
        return prod;
    }

    int64_t PitchUb(int64_t cols) const
    {
        const int64_t span = (cols - 1) * sL + 1;
        return AlignUpI64(span * elemSize, SSA_ALIGN_32_BYTES) / elemSize;
    }

    int64_t RowsOf(int64_t a, int64_t f) const
    {
        if (a == n - 1) {
            return 1; // k = 1：单行，无行序
        }
        return f * InnerElems(a) / shape[n - 1];
    }

    int64_t Cost(int64_t a, int64_t f) const
    {
        const int64_t cols = (a == n - 1) ? f : shape[n - 1];
        return f * InnerElems(a) * c1 + SSA_BUFFER_NUM_VAR * RowsOf(a, f) * PitchUb(cols) * elemSize;
    }

    /** Step 2 · a = n-1：cols = f、ubSpan 随 f 变 → 近似线性闭式 + 近等长回推 */
    int64_t SplitF() const
    {
        const int64_t denom = c1 + SSA_BUFFER_NUM_VAR * aSpan;
        const int64_t numer = ubBudget - SSA_BUFFER_NUM_VAR * SSA_ALIGN_32_BYTES;
        if (denom <= 0 || numer < denom) {
            return gran;
        }
        const int64_t f1 = numer / denom;
        const int64_t f2 = (aSpan > 0) ? (maxIdxValue / aSpan) : f1; // Scatter 索引位宽
        // 表在 IdxT 域内计算：a = n'-1 时 tileElemNum = f，须 <= maxTblElems 才能精确表示归纳量 i
        int64_t fub = AlignDownI64(MinI64(MinI64(f1, f2), maxTblElems), gran);
        fub = MaxI64(fub, gran);
        const int64_t m = MaxI64(CeilDivPos(shape[n - 1], fub), 1);
        return MinI64(shape[n - 1], AlignUpI64(CeilDivPos(shape[n - 1], m), gran));
    }

    /** UB 内单行实占跨度（元素）= (cols − 1)·sL + 1，cols = S[n'-1] */
    int64_t UbSpanOf() const { return (shape[n - 1] - 1) * sL + 1; }

    // DMA 各级固定计数与步长可行性（与 f 无关，不可行则该轴不能作切分轴）：
    // k ≤ 2 恒可行；k = 3/4 时 dim n'-2 落 blockCount（≤ 4095）、dim n'-3 落 loop1（< 2^21）
    bool DmaLevelsFit(int64_t a) const
    {
        const int64_t k = n - a;
        // k ≥ 3：dim n'-2 落在 blockCount，接口上限 4095
        if (k >= 3 && shape[n - 2] > SSA_MAX_DMA_BLOCK_COUNT) {
            return false;
        }
        if (k < 4) {
            return true;
        }
        // k = 4：dim n'-3 落在 loop1，须 < 2^21
        if (shape[n - 3] >= SSA_MAX_DMA_LOOP_SIZE) {
            return false;
        }
        return AbsI64(step[n - 2]) >= UbSpanOf();
    }

    /** Step 2 · a < n-1：cols = S[n-1] 与 f 无关 → pitchUb 固定 → cost 对 f 线性，一次除法即得 */
    int64_t MaxF(int64_t a) const
    {
        const int64_t inner = InnerElems(a);
        if (inner <= 0 || shape[n - 1] <= 0) {
            return 0;
        }
        // 表在 IdxT 域内闭式计算：tileElemNum = f·inner 须 <= maxTblElems（2B 索引时 65535），
        //    否则归纳量 i 在 IdxT 内回绕、Div 不再是精确值。inner 本身超限则该轴整体不可用。
        if (inner > maxTblElems) {
            return 0;
        }
        // 各级计数上限：blockCount（≤ 4095）与 loop1（< 2^21）承载的都是**与 f 无关**的维长，
        //    超限则该轴整体不可用（返回 0 让 PickAxis 止步）；k = 4 另需行步长 ≥ ubSpan
        if (!DmaLevelsFit(a)) {
            return 0;
        }
        const int64_t pitch = PitchUb(shape[n - 1]);
        // 每个 f 贡献的行数：inner 恒含因子 S[n-1]，故 a < n-1 时 rowsPerF >= 1；
        // rows = f·rowsPerF 即本任务的行维总数（= blockCount · loop1 · loop2）
        const int64_t rowsPerF = MaxI64(inner / shape[n - 1], 1);
        const int64_t perF = inner * c1 + SSA_BUFFER_NUM_VAR * rowsPerF * pitch * elemSize;
        if (perF <= 0) {
            return 0;
        }
        int64_t f = MinI64(ubBudget / perF, shape[a]);
        // 索引幅值上限：表内偏移以**元素**计（Reg::Scatter 按元素寻址），行基址 = ρ(r)·pitchUb，
        //    最大行基址 ≈ rows·pitchUb（元素）。maxIdxValue 是索引通道的**字节**宽度上限，
        //    故先右移 log2(sizeof(T)) 折到元素域，再除以行距 pitchUb（元素）得行数上限。
        //    等价于 maxIdxValue / (pitchUb·sizeof(T))：⌊⌊M/E⌋/P⌋ ≡ ⌊M/(E·P)⌋，两者恒等。
        const int64_t maxIdxRows = MaxI64((maxIdxValue / elemSize) / pitch, 1);
        f = MinI64(f, maxIdxRows / rowsPerF); // rows = f·rowsPerF <= maxIdxRows
        // ③ 表长上限：tileElemNum = f·inner <= maxTblElems
        f = MinI64(f, maxTblElems / inner);
        return MaxI64(f, 0);
    }

    /** Step 1 —— 选切分轴 a（自最内维向外贪心；UB 容得下整条最内维时才允许外扩） */
    void PickAxis(int64_t& a, int64_t& f) const
    {
        if (Cost(n - 1, shape[n - 1]) > ubBudget) {
            a = n - 1;
            f = MaxI64(SplitF(), 1);
            return;
        }
        // 基线：只切最内维（cols = f = S[n-1]，单任务单行）；f 同时受表长上限约束
        a = n - 1;
        f = MinI64(shape[n - 1], maxTblElems);
        // 逐轴外扩（至多到 n-4，一次搬运覆盖最后 4 维）；可行性由 MaxF 内 DmaLevelsFit 判定，
        // 不满足则返回 0，本循环止步。
        for (int64_t d = n - 2; d >= MaxI64(0, n - 4); --d) {
            const int64_t fd = MaxF(d);
            if (fd < 1) {
                break;
            }
            a = d;
            f = fd;
        }
        f = MaxI64(f, 1);
    }

    /** Step 3 —— 核间铺核 */
    void CountTasks(int64_t a, int64_t f, int64_t& total, int64_t& perCore, int64_t& usedCore) const
    {
        int64_t outer = 1;
        for (int64_t d = 0; d < a; ++d) {
            outer *= shape[d];
        }
        total = outer * MaxI64(CeilDivPos(shape[a], f), 1);
        perCore = (total > coreNum) ? MaxI64(CeilDivPos(total, coreNum), 1) : 1;
        usedCore = MaxI64(CeilDivPos(total, perCore), 1);
    }

    /** Step 4 —— 负载均衡（切分轴内移，任务过小即停） */
    void Balance(int64_t& a, int64_t& f, int64_t& total, int64_t& perCore, int64_t& usedCore) const
    {
        const int64_t ratio = static_cast<int64_t>(SSA_MIN_USED_CORES_RATIO * static_cast<float>(coreNum));
        while (usedCore < ratio && a < n - 1) {
            const int64_t cols = (a == n - 1) ? f : shape[n - 1];
            if (RowsOf(a, f) * PitchUb(cols) * elemSize < SSA_MIN_PER_TASK_BYTES) {
                break;
            }
            const int64_t fNext = (a + 1 == n - 1) ? SplitF() : MaxF(a + 1);
            if (fNext < 1) {
                break;
            }
            ++a;
            f = fNext;
            CountTasks(a, f, total, perCore, usedCore);
        }
    }
};

ge::graphStatus FillScatterTiling(gert::TilingContext* context, const SsaReducedShape& red, int64_t coreNum,
                                  int64_t ubUsable, int64_t elemSize)
{
    SsaScatterSolver solver;
    solver.n = red.dimNum;
    solver.shape = red.sliceShape;
    solver.step = red.axisStep;
    solver.elemSize = elemSize;
    solver.sL = AbsI64(red.axisStep[red.dimNum - 1]);
    solver.idxDtypeSize = (elemSize <= 2) ? 2 : 4;
    solver.c1 = solver.idxDtypeSize + SSA_SCATTER_BUFFER_NUM * elemSize;
    solver.aSpan = solver.sL * elemSize;
    solver.gran = MaxI64(SSA_ALIGN_32_BYTES / elemSize, 1);
    solver.ubBudget = ubUsable - SSA_PARAM_BUF_BYTES - SSA_SCATTER_BUFFER_NUM * SSA_VF_OVERREAD_BYTES -
                      SSA_VF_OVERREAD_BYTES - SSA_ALIGN_32_BYTES;
    solver.coreNum = MaxI64(coreNum, 1);
    solver.maxIdxValue = (solver.idxDtypeSize >= 4) ? SSA_INT32_MAX : 0xFFFF;
    solver.maxTblElems = (solver.idxDtypeSize >= 4) ? SSA_INT32_MAX : 0xFFFF;

    int64_t a = red.dimNum - 1;
    int64_t f = 1;
    solver.PickAxis(a, f);
    int64_t total = 0;
    int64_t perCore = 1;
    int64_t usedCore = 1;
    solver.CountTasks(a, f, total, perCore, usedCore);
    solver.Balance(a, f, total, perCore, usedCore);

    const int64_t innerElems = solver.InnerElems(a);
    const int64_t cols = (a == red.dimNum - 1) ? f : red.sliceShape[red.dimNum - 1];
    const int64_t rows = solver.RowsOf(a, f);
    const int64_t ubSpan = (cols - 1) * solver.sL + 1;
    const int64_t pitchUb = solver.PitchUb(cols);
    const int64_t tileElemNum = f * innerElems;

    auto* tiling = context->GetTilingData<StridedSliceAssignV2ScatterTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->ubAxis = static_cast<uint8_t>(a);
    tiling->ubFactor = static_cast<uint32_t>(f);
    tiling->ubPerCount = static_cast<uint32_t>(perCore);
    tiling->ubTotalCount = static_cast<uint32_t>(total);
    tiling->tileElemNum = static_cast<uint32_t>(tileElemNum);
    tiling->idxDtypeSize = static_cast<int32_t>(solver.idxDtypeSize);
    tiling->dimNum = red.dimNum;
    tiling->innerElems = innerElems;
    tiling->rows = rows;
    tiling->cols = cols;
    tiling->ubSpan = ubSpan;
    tiling->pitchUb = pitchUb;
    tiling->varBase = red.varBase;
    CopyReducedArrays(red, *tiling);
    for (int32_t j = 0; j < red.dimNum; ++j) {
        tiling->sliceCum[j] = red.sliceCum[j];
        tiling->rowStepVar[j] = 0;
        tiling->blkStartVar[j] = 0;
    }
    for (int64_t d = a; d <= static_cast<int64_t>(red.dimNum) - 2; ++d) {
        tiling->rowStepVar[d] = AbsI64(red.axisStep[d]); // d ∈ [a, n-2]
    }
    for (int64_t d = a; d <= static_cast<int64_t>(red.dimNum) - 1; ++d) {
        // 升地址序起点：负 stride 维取最低地址端
        tiling->blkStartVar[d] = (red.axisStep[d] > 0) ? 0 : (red.sliceShape[d] - 1) * red.axisStep[d];
    }

    context->SetTilingKey(GET_TPL_TILING_KEY(SSA_TPL_SCATTER));
    context->SetBlockDim(static_cast<uint32_t>(usedCore));
    PrintTilingInfo(context, *tiling);
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// axes 归位 + begin/end 归一化与 clamp
// ---------------------------------------------------------------------------
ge::graphStatus NormalizeIndices(gert::TilingContext* context, int64_t dimNum, const int64_t* varDim, int64_t* begin,
                                 int64_t* end, int64_t* strides)
{
    int64_t beginIn[SSA_MAX_DIM_NUM] = {};
    int64_t endIn[SSA_MAX_DIM_NUM] = {};
    int64_t stridesIn[SSA_MAX_DIM_NUM] = {};
    int64_t lenB = 0;
    int64_t lenE = 0;
    int64_t lenS = 0;
    OP_CHECK_IF(ReadConstVector(context, SSA_INDEX_BEGIN, beginIn, lenB) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "begin",
                                                         "Failed to get the const input of begin."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ReadConstVector(context, SSA_INDEX_END, endIn, lenE) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "end",
                                                         "Failed to get the const input of end."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ReadConstVector(context, SSA_INDEX_STRIDES, stridesIn, lenS) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "strides",
                                                         "Failed to get the const input of strides."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(lenB != lenE || lenB != lenS,
                OP_LOGE_FOR_INVALID_LISTSIZE_WITH_REASON(
                    context->GetNodeName(), "begin", std::to_string(lenB).c_str(),
                    ("The element nums of begin must be equal to that of end (" + std::to_string(lenE) +
                     ") and strides (" + std::to_string(lenS) + ").")
                        .c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(lenB > dimNum,
                OP_LOGE_FOR_INVALID_LISTSIZE_WITH_REASON(
                    context->GetNodeName(), "begin", std::to_string(lenB).c_str(),
                    ("The element nums of begin must be less than or equal to the dim num of var (" +
                     std::to_string(dimNum) + ").")
                        .c_str()),
                return ge::GRAPH_FAILED);

    // 原始常量输入打印（归位 / 归一化之前的字面值）
    PrintRawVector(context, "begin", beginIn, lenB);
    PrintRawVector(context, "end", endIn, lenE);
    PrintRawVector(context, "strides", stridesIn, lenS);

    // 缺省语义：未列轴为整轴切片
    for (int64_t i = 0; i < dimNum; ++i) {
        begin[i] = 0;
        end[i] = varDim[i];
        strides[i] = 1;
    }

    auto axesTensor = context->GetOptionalInputTensor(SSA_INDEX_AXES);
    if (axesTensor == nullptr) {
        for (int64_t i = 0; i < lenB; ++i) {
            begin[i] = beginIn[i];
            end[i] = endIn[i];
            strides[i] = stridesIn[i];
        }
    } else {
        int64_t axes[SSA_MAX_DIM_NUM] = {};
        int64_t lenA = 0;
        OP_CHECK_IF(ReadConstVector(context, SSA_INDEX_AXES, axes, lenA) != ge::GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "axes",
                                                             "Failed to get the const input of axes."),
                    return ge::GRAPH_FAILED);
        PrintRawVector(context, "axes", axes, lenA);
        OP_CHECK_IF(
            lenA != lenB,
            OP_LOGE_FOR_INVALID_LISTSIZE_WITH_REASON(
                context->GetNodeName(), "axes", std::to_string(lenA).c_str(),
                ("The element nums of axes must be equal to that of begin (" + std::to_string(lenB) + ").").c_str()),
            return ge::GRAPH_FAILED);
        bool axisUsed[SSA_MAX_DIM_NUM] = {};
        for (int64_t i = 0; i < lenA; ++i) {
            int64_t axis = axes[i];
            if (axis < 0) {
                axis += dimNum;
            }
            OP_CHECK_IF(axis < 0 || axis >= dimNum,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            context->GetNodeName(), "axes", std::to_string(axes[i]).c_str(),
                            ("The value of axes must be within the range [-" + std::to_string(dimNum) + ", " +
                             std::to_string(dimNum - 1) + "].")
                                .c_str()),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(
                axisUsed[axis],
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "axes", std::to_string(axes[i]).c_str(),
                                                      "The value of axes must not be duplicated."),
                return ge::GRAPH_FAILED);
            axisUsed[axis] = true;
            begin[axis] = beginIn[i];
            end[axis] = endIn[i];
            strides[axis] = stridesIn[i];
        }
    }

    for (int64_t i = 0; i < dimNum; ++i) {
        OP_CHECK_IF(
            strides[i] == 0,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "strides", std::to_string(strides[i]).c_str(),
                                                  "The value of strides must not be 0."),
            return ge::GRAPH_FAILED);
        const bool negative = (strides[i] < 0);
        if (begin[i] < 0) {
            begin[i] += varDim[i];
        }
        begin[i] = MinI64(MaxI64(begin[i], 0), negative ? (varDim[i] - 1) : varDim[i]);
        if (end[i] < 0) {
            end[i] += varDim[i];
        }
        // 负 stride 的 end 下界保留 -1，即「反向走到最低地址端」仍取满
        end[i] = MinI64(MaxI64(end[i], negative ? -1 : 0), varDim[i]);
    }
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// S0-0：var 形状校验（秩范围 + 各维非负）；含 0 长度维则置 varEmpty
// ---------------------------------------------------------------------------
ge::graphStatus ValidateVarShape(gert::TilingContext* context, const gert::StorageShape* varShape, int64_t dimNum,
                                 int64_t* varDim, bool& varEmpty)
{
    const size_t dimNumRaw = varShape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(
        dimNumRaw == 0 || dimNumRaw > static_cast<size_t>(SSA_MAX_DIM_NUM),
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            context->GetNodeName(), "var", (std::to_string(dimNumRaw) + "D").c_str(),
            ("The shape dim of var must be within the range [1D, " + std::to_string(SSA_MAX_DIM_NUM) + "D].").c_str()),
        return ge::GRAPH_FAILED);

    varEmpty = false;
    for (int64_t i = 0; i < dimNum; ++i) {
        varDim[i] = varShape->GetStorageShape().GetDim(i);
        OP_CHECK_IF(varDim[i] < 0,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "var",
                                                          Ops::Base::ToString(varShape->GetStorageShape()).c_str(),
                                                          "The shape of var must not be negative."),
                    return ge::GRAPH_FAILED);
        if (varDim[i] == 0) {
            varEmpty = true; // var 含 0 长度维 ⇒ 无元素可写
        }
    }
    return ge::GRAPH_SUCCESS;
}

// 切片区域是否为空（S[] 含 0 维度）
inline bool IsSliceEmpty(const int64_t* slice, int64_t dimNum)
{
    for (int64_t i = 0; i < dimNum; ++i) {
        if (slice[i] == 0) {
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// S0-5：input_value 形状校验（秩与 var 相等 + 逐维与切片形状 S[] 严格相等，不支持广播）
// ---------------------------------------------------------------------------
ge::graphStatus ValidateValueShape(gert::TilingContext* context, const gert::StorageShape* valueShape, int64_t dimNum,
                                   const int64_t* slice)
{
    const size_t valueDimNum = valueShape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(valueDimNum != static_cast<size_t>(dimNum),
                OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                    context->GetNodeName(), "var and input_value",
                    (std::to_string(dimNum) + " and " + std::to_string(valueDimNum) + "D").c_str(),
                    "The shape dims of var and input_value must be equal."),
                return ge::GRAPH_FAILED);

    int64_t valueDim[SSA_MAX_DIM_NUM] = {};
    for (int64_t i = 0; i < dimNum; ++i) {
        valueDim[i] = valueShape->GetStorageShape().GetDim(i);
    }
    OP_CHECK_IF(
        RangeShapesUnequal(valueDim, slice, dimNum),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            context->GetNodeName(), "input_value", Ops::Base::ToString(valueShape->GetStorageShape()).c_str(),
            ("The shape of input_value must be equal to the slice shape " + ShapeToStr(slice, dimNum) + ".").c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

} // namespace

// ---------------------------------------------------------------------------
// Tiling 入口（须位于 optiling 命名空间作用域：入口函数与头文件声明必须同名同实体）
// ---------------------------------------------------------------------------
ge::graphStatus Tiling4StridedSliceAssignV2Arch35(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Tiling4StridedSliceAssignV2Arch35 running.");

    // compileInfo 在预编译等路径下可能为 nullptr，不能硬失败；运行时优先经 PlatformInfo 查询，compileInfo 仅作兜底。
    auto compileInfo = reinterpret_cast<const StridedSliceAssignV2CompileInfo*>(context->GetCompileInfo());
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const int64_t coreNum = (compileInfo == nullptr) ? static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv()) :
                                                       static_cast<int64_t>(compileInfo->totalCoreNum);
    OP_CHECK_IF(
        coreNum <= 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "core_num", std::to_string(coreNum).c_str(),
                                              "The value of core_num must be greater than 0."),
        return ge::GRAPH_FAILED);
    uint64_t ubSizePlatform = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
    const int64_t ubUsable = (compileInfo == nullptr) ?
                                 static_cast<int64_t>(ubSizePlatform) :
                                 ((compileInfo->ubUsable > 0) ? compileInfo->ubUsable : compileInfo->ubSize);
    OP_CHECK_IF(
        ubUsable <= 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "ub_size", std::to_string(ubUsable).c_str(),
                                              "The value of ub_size must be greater than 0."),
        return ge::GRAPH_FAILED);

    const gert::StorageShape* varShape = context->GetInputShape(SSA_INDEX_VAR);
    const gert::StorageShape* valueShape = context->GetInputShape(SSA_INDEX_INPUT_VALUE);
    OP_CHECK_NULL_WITH_CONTEXT(context, varShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, valueShape);

    const int64_t dimNum = static_cast<int64_t>(varShape->GetStorageShape().GetDimNum());

    auto varDesc = context->GetInputDesc(SSA_INDEX_VAR);
    OP_CHECK_NULL_WITH_CONTEXT(context, varDesc);
    const ge::DataType varDtype = varDesc->GetDataType();
    // dtype 白名单校验（与 OpDef inputDataType 一致）
    OP_CHECK_IF(varDtype != ge::DT_FLOAT16 && varDtype != ge::DT_FLOAT && varDtype != ge::DT_BF16 &&
                    varDtype != ge::DT_INT32 && varDtype != ge::DT_INT64 && varDtype != ge::DT_DOUBLE &&
                    varDtype != ge::DT_INT8,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    context->GetNodeName(), "var", Ops::Base::ToString(varDtype).c_str(),
                    "dtype must be in [DT_FLOAT16, DT_FLOAT, DT_BF16, DT_INT32, DT_INT64, DT_DOUBLE, DT_INT8]"),
                return ge::GRAPH_FAILED);
    const int64_t elemSize = static_cast<int64_t>(ge::GetSizeByDataType(varDtype));

    OP_LOGI(context, "===== Raw input =====");
    OP_LOGI(context, "elemSize = %ld", elemSize);
    PrintShapeDims(context, "var", varShape);
    PrintShapeDims(context, "input_value", valueShape);

    // ---- S0-0：var 形状校验（秩 + 各维非负）+ 空 var 判定 ----
    int64_t varDim[SSA_MAX_DIM_NUM] = {};
    bool varEmpty = false;
    OP_CHECK_IF(
        ValidateVarShape(context, varShape, dimNum, varDim, varEmpty) != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "var", "Failed to validate the shape of var."),
        return ge::GRAPH_FAILED);

    // ---- var 含 0 长度维 ⇒ T4 空模板 ----
    if (varEmpty) {
        OP_LOGD(context->GetNodeName(), "empty var");
        return FillEmptyTiling(context);
    }

    // ---- ：读常量、axes 归位、归一化与 clamp（strides==0 在此报错）----
    int64_t begin[SSA_MAX_DIM_NUM] = {};
    int64_t end[SSA_MAX_DIM_NUM] = {};
    int64_t strides[SSA_MAX_DIM_NUM] = {};
    OP_CHECK_IF(NormalizeIndices(context, dimNum, varDim, begin, end, strides) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "begin/end/strides",
                                                         "Failed to normalize begin, end and strides."),
                return ge::GRAPH_FAILED);

    // ---- ：切片形状 S[]（正/负 stride 统一公式，负值截 0）----
    int64_t slice[SSA_MAX_DIM_NUM] = {};
    for (int64_t i = 0; i < dimNum; ++i) {
        const int64_t span = (strides[i] > 0) ? (end[i] - begin[i]) : (begin[i] - end[i]);
        slice[i] = CeilDivPos(span, AbsI64(strides[i]));
    }

    // ---- ：切片区域为空 ⇒ 同样退化为 空模板 ----
    if (IsSliceEmpty(slice, dimNum)) {
        OP_LOGD(context->GetNodeName(), "empty slice");
        return FillEmptyTiling(context);
    }

    // ---- S0-5：校验 input_value.shape ≡ S（逐维严格相等，不支持广播）----
    OP_CHECK_IF(ValidateValueShape(context, valueShape, dimNum, slice) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "input_value",
                                                         "Failed to validate the shape of input_value."),
                return ge::GRAPH_FAILED);

    // ---- S0-6：varCum 后缀积 + 元素步长 T[d] = strides[d]·varCum[d+1] ----
    int64_t varCum[SSA_MAX_DIM_NUM + 1] = {};
    int64_t axisStepRaw[SSA_MAX_DIM_NUM] = {};
    varCum[dimNum] = 1;
    for (int64_t d = dimNum - 1; d >= 0; --d) {
        varCum[d] = varDim[d] * varCum[d + 1];
        axisStepRaw[d] = strides[d] * varCum[d + 1];
    }

    // ---- S0-6.5：合轴归一化----
    SsaReducedShape red;
    CollapseAxes(dimNum, slice, axisStepRaw, begin, varCum, red);

    // ---- S0-7：模板选择判据（顺序判定，命中即定模板，不再回判）----
    const int64_t contigRunBytes = CalcContigRun(red) * elemSize;
    const int64_t minContigRunBytes = (red.dimNum >= 3) ? SSA_T2_MIN_CHUNK_BYTES : SSA_MIN_CONTIG_RUN_BYTES;
    const bool batched2dPossible = (red.directionId == SSA_DIR_ALL_POSITIVE) && (red.dimNum >= 2) &&
                                   (contigRunBytes * SSA_BUFFER_NUM <= ubUsable);
    const bool runBytesAligned = !batched2dPossible || (contigRunBytes % SSA_ALIGN_32_BYTES == 0);
    bool useHybrid = false;
    bool useContig = false;
    if (red.directionId == SSA_DIR_MIXED) {
        useHybrid = true;
    } else if (CalcValidNumInCacheLine(context, red, elemSize) <= SSA_SPARSITY_THRESHOLD) {
        useHybrid = true;
    } else if (contigRunBytes >= minContigRunBytes && runBytesAligned && SsaContigSegsAligned32B(red, elemSize)) {
        useContig = true;
    } else if (red.sliceElemNum <= SSA_SMALL_SHAPE_THRESHOLD) {
        useHybrid = true;
    }
    OP_LOGD(context->GetNodeName(),
            "SSA arch35 route: n'=%d sliceElemNum=%ld dir=%d contigRunBytes=%ld minRunBytes=%ld -> %s", red.dimNum,
            red.sliceElemNum, red.directionId, contigRunBytes, minContigRunBytes,
            useHybrid ? "T1" : (useContig ? "T2" : "T3"));

    // ---- S0-8：按模板计算切分参数并写 TilingData ----
    ge::graphStatus ret = ge::GRAPH_SUCCESS;
    if (useHybrid) {
        context->SetLocalMemorySize(static_cast<uint32_t>(ubUsable - SSA_SIMT_DCACHE_BYTES));
        ret = FillHybridTiling(context, red, coreNum, ubUsable - SSA_SIMT_DCACHE_BYTES, elemSize);
    } else if (useContig) {
        ret = FillContigTiling(context, red, coreNum, ubUsable, elemSize);
    } else {
        ret = FillScatterTiling(context, red, coreNum, ubUsable, elemSize);
    }
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4StridedSliceAssignV2Arch35(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepare4StridedSliceAssignV2Arch35 running.");
    auto compileInfo = context->GetCompiledInfo<StridedSliceAssignV2CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    compileInfo->totalCoreNum = static_cast<int32_t>(ascendcPlatform.GetCoreNumAiv());
    OP_CHECK_IF(compileInfo->totalCoreNum <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "core_num",
                                                      std::to_string(compileInfo->totalCoreNum).c_str(),
                                                      "The value of core_num must be greater than 0."),
                return ge::GRAPH_FAILED);

    uint64_t ubSizePlatForm = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSize = static_cast<int64_t>(ubSizePlatForm);
    compileInfo->ubUsable = compileInfo->ubSize;
    OP_CHECK_IF(compileInfo->ubSize <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "ub_size",
                                                      std::to_string(compileInfo->ubSize).c_str(),
                                                      "The value of ub_size must be greater than 0."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(StridedSliceAssignV2)
    .Tiling(Tiling4StridedSliceAssignV2Arch35)
    .TilingParse<StridedSliceAssignV2CompileInfo>(TilingPrepare4StridedSliceAssignV2Arch35)
    .TilingInputsDataDependency({2, 3, 4, 5});

} // namespace optiling
