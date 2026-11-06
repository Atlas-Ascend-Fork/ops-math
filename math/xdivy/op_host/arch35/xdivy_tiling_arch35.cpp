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
 * \file xdivy_tiling_arch35.cpp
 * \brief Xdivy Tiling — arch35 实现
 */
#include "xdivy_tiling_arch35.h"
#include "math/xdivy/op_kernel/arch35/xdivy_struct.h"
#include <algorithm>
#include <sstream>
#include <graph/utils/type_utils.h>
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/soc_spec.h"

using namespace ge;

// ============================================================
// Tiling 函数模块 — namespace xdivy
// ============================================================

namespace xdivy {

// === 算子特定常量 — 新算子必须修改 ===
constexpr int64_t kPhysNodesFp32 = 3; // 物理存活节点 P (FP32 路径, 无 Cast)
constexpr int64_t kPhysNodesFp16 = 4; // 物理存活节点 P (FP16/BF16 路径, 有 Cast)
constexpr int64_t kBufDtypeSize = 4;  // per_buf_elems 永远除 sizeof(float32) (cast.md §Tile)
constexpr size_t XDIVY_MAX_RANK = 8; // README 约束：输入 shape 维度上限，同时也是 TilingData 定长数组容量

// 溢出安全乘法：结果超出 int64_t 表示范围时返回 false
static inline bool MulChecked(int64_t left, int64_t right, int64_t& result)
{
    return !__builtin_mul_overflow(left, right, &result);
}

// 溢出安全上取整：不用 value + divisor - 1 的写法，避免相加回绕
static inline bool CeilDivChecked(int64_t value, int64_t divisor, int64_t& result)
{
    if (divisor <= 0) {
        return false;
    }
    result = value / divisor + (value % divisor == 0 ? 0 : 1);
    return true;
}

// 5.1 PadAndSqueeze

bool PadAndSqueeze(const std::vector<std::vector<int64_t>>& input_shapes,
                   const std::vector<std::vector<int64_t>>& output_shapes, std::vector<int64_t>& maximum_bro_shape,
                   std::vector<std::vector<int64_t>>& normal_input_shapes,
                   std::vector<std::vector<int64_t>>& normal_output_shapes)
{
    int64_t num_inputs = (int64_t)input_shapes.size();
    int64_t num_outputs = (int64_t)output_shapes.size();
    int64_t max_rank = 0;
    for (auto& s : input_shapes)
        max_rank = std::max(max_rank, (int64_t)s.size());
    for (auto& s : output_shapes)
        max_rank = std::max(max_rank, (int64_t)s.size());
    auto pad = [&](const std::vector<int64_t>& s) {
        std::vector<int64_t> p;
        p.assign(max_rank - (int64_t)s.size(), 1);
        p.insert(p.end(), s.begin(), s.end());
        return p;
    };
    std::vector<std::vector<int64_t>> padded_in(num_inputs), padded_out(num_outputs);
    for (int64_t i = 0; i < num_inputs; i++)
        padded_in[i] = pad(input_shapes[i]);
    for (int64_t i = 0; i < num_outputs; i++)
        padded_out[i] = pad(output_shapes[i]);
    maximum_bro_shape.clear();
    normal_input_shapes.assign(num_inputs, std::vector<int64_t>());
    normal_output_shapes.assign(num_outputs, std::vector<int64_t>());
    for (int64_t d = 0; d < max_rank; d++) {
        bool all_one = true;
        int64_t max_dim = 0;
        for (int64_t i = 0; i < num_inputs; i++) {
            if (padded_in[i][d] != 1)
                all_one = false;
            max_dim = std::max(max_dim, padded_in[i][d]);
        }
        for (int64_t i = 0; i < num_outputs; i++) {
            if (padded_out[i][d] != 1)
                all_one = false;
            max_dim = std::max(max_dim, padded_out[i][d]);
        }
        if (!all_one) {
            maximum_bro_shape.push_back(max_dim);
            for (int64_t i = 0; i < num_inputs; i++)
                normal_input_shapes[i].push_back(padded_in[i][d]);
            for (int64_t i = 0; i < num_outputs; i++)
                normal_output_shapes[i].push_back(padded_out[i][d]);
        }
    }
    if (maximum_bro_shape.empty()) {
        maximum_bro_shape.push_back(1);
        for (int64_t i = 0; i < num_inputs; i++)
            normal_input_shapes[i].push_back(1);
        for (int64_t i = 0; i < num_outputs; i++)
            normal_output_shapes[i].push_back(1);
    }
    return true;
}

// 5.2 FindSplitAxis — 从最内轴向外扫描，d_k × inner > per_buf_elems 时切分

bool FindSplitAxis(const std::vector<int64_t>& max_bro_shape, int64_t dtype_size, int64_t ub_per_core,
                   int64_t phys_nodes, SplitResult& out)
{
    int64_t per_buf_bytes = (ub_per_core / phys_nodes) & ~31LL;
    int64_t per_buf_elems = per_buf_bytes / dtype_size;
    if (per_buf_elems < 1) {
        return false; // UB 容量或物理节点配置异常，无法切分
    }
    int64_t rank = (int64_t)max_bro_shape.size();
    int64_t inner = 1;
    for (int64_t k = rank - 1; k >= 0; k--) {
        int64_t prod = 0;
        if (!MulChecked(max_bro_shape[k], inner, prod)) {
            return false; // 内层维度乘积溢出 int64_t
        }
        if (prod > per_buf_elems) {
            out.a_i = per_buf_elems / inner;
            if (!CeilDivChecked(max_bro_shape[k], out.a_i, out.a_o)) {
                return false;
            }
            int64_t rem = max_bro_shape[k] % out.a_i;
            out.a_i_tail = (rem == 0) ? out.a_i : rem;
            out.axis = k;
            return true;
        }
        if (k == 0) {
            out.axis = 0;
            out.a_i = max_bro_shape[0];
            out.a_o = 1;
            out.a_i_tail = max_bro_shape[0];
            return true;
        }
        inner = prod;
    }
    return true;
}

// 5.3 MultiCoreSplit

bool MultiCoreSplit(const std::vector<int64_t>& max_bro_shape, const SplitResult& ub_split, int64_t max_cores,
                    MultiCoreResult& out)
{
    int64_t k = ub_split.axis, outer_prod = 1;
    for (int64_t j = 0; j < k; j++) {
        if (!MulChecked(outer_prod, max_bro_shape[j], outer_prod)) {
            return false; // 切分轴外层维度乘积溢出 int64_t
        }
    }
    if (!MulChecked(outer_prod, ub_split.a_o, out.total_tiles)) {
        return false; // total_tiles 溢出 int64_t
    }
    if (out.total_tiles < 1) {
        return false; // 空 Tensor 已在入口提前返回，常规通路计划必须非空
    }
    out.num_cores = (out.total_tiles < max_cores) ? out.total_tiles : max_cores;
    if (out.num_cores < 1) {
        return false; // 防御：coreNum 异常（如 0）时直接失败，避免 0/0 整除零
    }
    out.tiles_main = out.total_tiles / out.num_cores;
    out.cores_tail = out.total_tiles % out.num_cores;
    return true;
}

// 5.5 地址偏移

bool PrecomputeInputStrides(const std::vector<int64_t>& s, std::vector<int64_t>& strides)
{
    int64_t rank = (int64_t)s.size();
    strides.assign(rank, 0);
    for (int64_t d = rank - 1; d >= 0; d--) {
        if (s[d] == 1) {
            strides[d] = 0;
            continue;
        }
        int64_t prod = 1;
        for (int64_t j = d + 1; j < rank; j++)
            prod *= s[j];
        strides[d] = prod;
    }
    return true;
}

bool PrecomputeOutputStrides(const std::vector<int64_t>& s, std::vector<int64_t>& strides)
{
    int64_t rank = (int64_t)s.size();
    strides.assign(rank, 0);
    for (int64_t d = rank - 1; d >= 0; d--) {
        if (s[d] == 1) {
            strides[d] = 0;
            continue;
        }
        int64_t prod = 1;
        for (int64_t j = d + 1; j < rank; j++)
            prod *= s[j];
        strides[d] = prod;
    }
    return true;
}

} // namespace xdivy

// ============================================================
// XdivyTiling — CANN 主线：单次归一化 → 映射 rank → 模板填充
// ============================================================

namespace optiling {

using namespace xdivy;

static std::string Arr2String(const int64_t* arr, int64_t n)
{
    std::ostringstream oss;
    oss << "[";
    if (n > 0) {
        for (int64_t i = 0; i < n - 1; ++i)
            oss << arr[i] << ",";
        oss << arr[n - 1];
    }
    oss << "]";
    return oss.str();
}

XdivyTiling::XdivyTiling(gert::TilingContext* ctx) : ctx_(ctx) {}

// 单个 Tensor 的 shape 合法性：rank 上限（README 约束 + TilingData 定长数组容量）、
// 维度非负、元素总数不溢出 int64_t。零维（空 Tensor）是合法输入，由上层空分支处理。
static ge::graphStatus CheckConcreteShape(const gert::TilingContext* context, const char* tensorName,
                                          const gert::Shape& shape)
{
    const size_t rank = shape.GetDimNum();
    OP_CHECK_IF(
        rank > XDIVY_MAX_RANK,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), tensorName, std::to_string(rank).c_str(),
                                                 "storage rank must be within [0, 8]"),
        return ge::GRAPH_FAILED);
    bool hasZeroDim = false;
    for (size_t i = 0; i < rank; ++i) {
        const int64_t dim = shape.GetDim(i);
        OP_CHECK_IF(dim < 0,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        context->GetNodeName(), tensorName, std::to_string(dim).c_str(),
                        ("storage dim[" + std::to_string(i) + "] must be non-negative").c_str()),
                    return ge::GRAPH_FAILED);
        hasZeroDim = hasZeroDim || dim == 0;
    }
    // 含 0 维时 numel 恒为 0；否则 GetShapeSize() 溢出时返回负值，代表元素总数超出 int64_t。
    OP_CHECK_IF(!hasZeroDim && shape.GetShapeSize() < 0,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context->GetNodeName(), tensorName,
                                                          Ops::Base::ToString(shape).c_str(),
                                                          "shape element count overflows int64_t"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 广播结果校验：y 必须等于 x1/x2 的逐维广播结果。非 1 维不相等即不可广播；
// 非 1 维取值保留 0（0 与 1 广播结果为 0，空 Tensor 语义）。上游 InferShape
// 已按同一语义推导，此处防御动态 shape 场景下伪造 y 绕过校验（含空 Tensor 早退）。
static ge::graphStatus CheckBroadcastResult(const gert::TilingContext* context, const gert::Shape& x1Shape,
                                            const gert::Shape& x2Shape, const gert::Shape& yShape)
{
    const size_t x1Rank = x1Shape.GetDimNum();
    const size_t x2Rank = x2Shape.GetDimNum();
    const size_t expectedRank = x1Rank > x2Rank ? x1Rank : x2Rank;
    OP_CHECK_IF(yShape.GetDimNum() != expectedRank,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    context->GetNodeName(), "y", std::to_string(yShape.GetDimNum()).c_str(),
                    ("rank must match broadcast rank " + std::to_string(expectedRank)).c_str()),
                return ge::GRAPH_FAILED);

    const size_t x1Offset = expectedRank - x1Rank;
    const size_t x2Offset = expectedRank - x2Rank;
    for (size_t i = 0; i < expectedRank; ++i) {
        const int64_t x1Dim = i < x1Offset ? 1 : x1Shape.GetDim(i - x1Offset);
        const int64_t x2Dim = i < x2Offset ? 1 : x2Shape.GetDim(i - x2Offset);
        OP_CHECK_IF(
            x1Dim != x2Dim && x1Dim != 1 && x2Dim != 1,
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                context->GetNodeName(), "x1 and x2", (std::to_string(x1Dim) + " and " + std::to_string(x2Dim)).c_str(),
                ("dimensions cannot broadcast at output dim " + std::to_string(i)).c_str()),
            return ge::GRAPH_FAILED);
        const int64_t expectedDim = x1Dim == 1 ? x2Dim : x1Dim;
        OP_CHECK_IF(
            yShape.GetDim(i) != expectedDim,
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                context->GetNodeName(), "y", std::to_string(yShape.GetDim(i)).c_str(),
                ("dim[" + std::to_string(i) + "] must equal broadcast result " + std::to_string(expectedDim)).c_str()),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// x1/x2/y dtype 一致性（README 约束）。Kernel 以 DTYPE_X1 单模板实例化两个输入，
// 混合 dtype 会被按 x1 的元素宽度误读；基线 Verifier（CheckTwoInputDtypeSame）显式拒绝。
ge::graphStatus XdivyTiling::CheckDtype() const
{
    auto x1Desc = ctx_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(ctx_, x1Desc);
    auto x2Desc = ctx_->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(ctx_, x2Desc);
    auto yDesc = ctx_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(ctx_, yDesc);
    const ge::DataType x1Dtype = x1Desc->GetDataType();
    const ge::DataType x2Dtype = x2Desc->GetDataType();
    const ge::DataType yDtype = yDesc->GetDataType();
    OP_CHECK_IF(x1Dtype != ge::DT_FLOAT && x1Dtype != ge::DT_FLOAT16 && x1Dtype != ge::DT_BF16,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(ctx_->GetNodeName(), "x1",
                                                      ge::TypeUtils::DataTypeToSerialString(x1Dtype).c_str(),
                                                      "supported dtypes are fp16, fp32 and bf16"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(x2Dtype != x1Dtype || yDtype != x1Dtype,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(ctx_->GetNodeName(), "x1, x2 and y",
                                                       (ge::TypeUtils::DataTypeToSerialString(x1Dtype) + ", " +
                                                        ge::TypeUtils::DataTypeToSerialString(x2Dtype) + " and " +
                                                        ge::TypeUtils::DataTypeToSerialString(yDtype))
                                                           .c_str(),
                                                       "x1, x2 and y must have the same dtype"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus XdivyTiling::GetShapeInfo()
{
    auto compileInfo = reinterpret_cast<const XdivyCompileInfo*>(ctx_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(ctx_, compileInfo);

    // 读输入 shape 并校验 rank/维度/元素数
    std::vector<gert::Shape> inShapes;
    for (size_t i = 0; i < ctx_->GetComputeNodeInfo()->GetInputsNum(); ++i) {
        auto shape = ctx_->GetInputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(ctx_, shape);
        gert::Shape s = shape->GetStorageShape();
        if (CheckConcreteShape(ctx_, (i == 0) ? "x1" : "x2", s) != ge::GRAPH_SUCCESS) {
            return GRAPH_FAILED;
        }
        inShapes.push_back(s);
        std::vector<int64_t> dims;
        for (size_t d = 0; d < s.GetDimNum(); ++d)
            dims.push_back(s.GetDim(d));
        raw_input_shapes_.push_back(dims);
    }

    // 读输出 shape 并校验
    std::vector<gert::Shape> outShapes;
    for (size_t i = 0; i < ctx_->GetComputeNodeInfo()->GetOutputsNum(); ++i) {
        auto shape = ctx_->GetOutputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(ctx_, shape);
        gert::Shape s = shape->GetStorageShape();
        if (CheckConcreteShape(ctx_, "y", s) != ge::GRAPH_SUCCESS) {
            return GRAPH_FAILED;
        }
        outShapes.push_back(s);
        std::vector<int64_t> dims;
        for (size_t d = 0; d < s.GetDimNum(); ++d)
            dims.push_back(s.GetDim(d));
        raw_output_shapes_.push_back(dims);
    }

    if (inShapes.size() != (size_t)kMaxInputSlots || outShapes.size() != (size_t)kMaxOutputSlots) {
        OP_LOGE(ctx_->GetNodeName(), "Xdivy requires 2 inputs and 1 output, got %zu inputs and %zu outputs",
                inShapes.size(), outShapes.size());
        return GRAPH_FAILED;
    }

    if (CheckDtype() != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }

    if (CheckBroadcastResult(ctx_, inShapes[0], inShapes[1], outShapes[0]) != ge::GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }

    auto inputDesc = ctx_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(ctx_, inputDesc);
    ge::DataType dtype = inputDesc->GetDataType();
    if (dtype == ge::DT_FLOAT) {
        dtype_size_ = 4;
        phys_nodes_ = kPhysNodesFp32;
    } else if (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) {
        dtype_size_ = 2;
        phys_nodes_ = kPhysNodesFp16;
    } else {
        OP_LOGE(ctx_->GetNodeName(), "Xdivy unsupported dtype");
        return GRAPH_FAILED;
    }

    // 广播校验通过后，输出含 0 维 ⟺ 合法空 Tensor（输入 0 维必传导至输出）
    output_empty_ = false;
    for (size_t d = 0; d < outShapes[0].GetDimNum(); ++d) {
        if (outShapes[0].GetDim(d) == 0) {
            output_empty_ = true;
            break;
        }
    }

    PadAndSqueeze(raw_input_shapes_, raw_output_shapes_, max_bro_shape_, normal_input_shapes_, normal_output_shapes_);
    rank_ = (int64_t)max_bro_shape_.size();

    OP_LOGI(ctx_->GetNodeName(), "Xdivy GetShapeInfo done rank %ld phys_nodes %ld ub %lu core %lu", rank_, phys_nodes_,
            compileInfo->ubSize, compileInfo->coreNum);

    return GRAPH_SUCCESS;
}

template <int64_t R>
ge::graphStatus XdivyTiling::DoTilingAndSet()
{
    auto* tiling = ctx_->GetTilingData<XdivyTilingData<R>>();
    OP_CHECK_NULL_WITH_CONTEXT(ctx_, tiling);

    auto* compileInfo = reinterpret_cast<const XdivyCompileInfo*>(ctx_->GetCompileInfo());
    int64_t ub_per_core = (int64_t)compileInfo->ubSize;

    // per_buf_bytes = (UB / P) & ~31, per_buf_elems = per_buf_bytes / kBufDtypeSize
    int64_t per_buf_bytes = (ub_per_core / phys_nodes_) & ~31LL;
    tiling->per_buf_bytes = per_buf_bytes;

    if (!FindSplitAxis(max_bro_shape_, kBufDtypeSize, ub_per_core, phys_nodes_, tiling->split)) {
        OP_LOGE(ctx_->GetNodeName(), "Xdivy FindSplitAxis failed: ub config invalid or shape product overflows");
        return GRAPH_FAILED;
    }
    if (!MultiCoreSplit(max_bro_shape_, tiling->split, (int64_t)compileInfo->coreNum, tiling->multicore)) {
        OP_LOGE(ctx_->GetNodeName(), "Xdivy MultiCoreSplit failed: shape product overflows or empty tiling plan");
        return GRAPH_FAILED;
    }

    int64_t num_in = (int64_t)normal_input_shapes_.size();
    int64_t num_out = (int64_t)normal_output_shapes_.size();
    std::vector<std::vector<int64_t>> in_strides(num_in), out_strides(num_out);
    for (int64_t i = 0; i < num_in; i++)
        PrecomputeInputStrides(normal_input_shapes_[i], in_strides[i]);
    for (int64_t i = 0; i < num_out; i++)
        PrecomputeOutputStrides(normal_output_shapes_[i], out_strides[i]);

    tiling->rank = rank_;
    int64_t delta = R - rank_;

    // max_bro_shape: 前补 1
    for (int64_t d = 0; d < delta; d++)
        tiling->max_bro_shape[d] = 1;
    for (int64_t d = 0; d < rank_; d++)
        tiling->max_bro_shape[d + delta] = max_bro_shape_[d];

    tiling->split.axis += delta;

    tiling->num_inputs = num_in;
    tiling->num_outputs = num_out;

    // input: 前补 shape=1 stride=0
    for (int64_t i = 0; i < num_in; i++) {
        for (int64_t d = 0; d < delta; d++) {
            tiling->input_shapes[i][d] = 1;
            tiling->input_strides[i][d] = 0;
        }
        for (int64_t d = 0; d < rank_; d++) {
            tiling->input_shapes[i][d + delta] = normal_input_shapes_[i][d];
            tiling->input_strides[i][d + delta] = in_strides[i][d];
        }
    }
    // 未使用 input slot: 全填 1/0
    for (int64_t i = num_in; i < kMaxInputSlots; i++)
        for (int64_t d = 0; d < R; d++) {
            tiling->input_shapes[i][d] = 1;
            tiling->input_strides[i][d] = 0;
        }

    // output: 前补 shape=1 stride=0
    for (int64_t i = 0; i < num_out; i++) {
        for (int64_t d = 0; d < delta; d++) {
            tiling->output_shapes[i][d] = 1;
            tiling->output_strides[i][d] = 0;
        }
        for (int64_t d = 0; d < rank_; d++) {
            tiling->output_shapes[i][d + delta] = normal_output_shapes_[i][d];
            tiling->output_strides[i][d + delta] = out_strides[i][d];
        }
    }
    for (int64_t i = num_out; i < kMaxOutputSlots; i++)
        for (int64_t d = 0; d < R; d++) {
            tiling->output_shapes[i][d] = 1;
            tiling->output_strides[i][d] = 0;
        }

    ctx_->SetBlockDim(tiling->multicore.num_cores);

    // 维测: TilingData 全部字段
    OP_LOGI(ctx_->GetNodeName(),
            "Xdivy TilingData: per_buf_bytes=%ld rank=%ld->R=%d "
            "max_bro_shape=%s "
            "split(axis=%ld a_i=%ld a_o=%ld a_i_tail=%ld) "
            "multi(cores=%ld tiles=%ld main=%ld core_tail=%ld) num_in=%ld num_out=%ld",
            tiling->per_buf_bytes, rank_, (int)R, Arr2String(tiling->max_bro_shape, R).c_str(), tiling->split.axis,
            tiling->split.a_i, tiling->split.a_o, tiling->split.a_i_tail, tiling->multicore.num_cores,
            tiling->multicore.total_tiles, tiling->multicore.tiles_main, tiling->multicore.cores_tail, num_in, num_out);

    for (int64_t i = 0; i < num_in; i++)
        OP_LOGI(ctx_->GetNodeName(), "Xdivy TilingData input[%ld]: shape=%s stride=%s", i,
                Arr2String(tiling->input_shapes[i], R).c_str(), Arr2String(tiling->input_strides[i], R).c_str());
    for (int64_t i = 0; i < num_out; i++)
        OP_LOGI(ctx_->GetNodeName(), "Xdivy TilingData output[%ld]: shape=%s stride=%s", i,
                Arr2String(tiling->output_shapes[i], R).c_str(), Arr2String(tiling->output_strides[i], R).c_str());

    return GRAPH_SUCCESS;
}

// 空 Tensor 早返回。校验（dtype/shape/broadcast）已全部通过后才允许进入本分支，
// 防止伪造的空 y 绕过检查。输出 0 元素，无数据需要搬运：blockDim = 1，
// tilingKey 选自定义空分支，kernel 侧直接返回。
ge::graphStatus XdivyTiling::HandleEmptyTensor() const
{
    OP_LOGD(ctx_->GetNodeName(), "Xdivy: empty tensor, skip kernel computation.");
    auto* tilingData = ctx_->GetTilingData<XdivyEmptyTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(ctx_, tilingData);
    tilingData->numel = 0;

    ctx_->SetBlockDim(1);
    ctx_->SetTilingKey(GET_TPL_TILING_KEY(XDIVY_RANK_4, XDIVY_USER_DEF_EMPTY));
    return GRAPH_SUCCESS;
}

ge::graphStatus XdivyTiling::RunTiling()
{
    ge::graphStatus ret = GetShapeInfo();
    if (ret != GRAPH_SUCCESS)
        return ret;

    if (output_empty_) {
        return HandleEmptyTensor();
    }

    int64_t mapped = (rank_ <= 4) ? 4 : 8;
    if (mapped == 4) {
        ret = DoTilingAndSet<4>();
        ctx_->SetTilingKey(GET_TPL_TILING_KEY(XDIVY_RANK_4, XDIVY_USER_DEF_NORMAL));
    } else {
        ret = DoTilingAndSet<8>();
        ctx_->SetTilingKey(GET_TPL_TILING_KEY(XDIVY_RANK_8, XDIVY_USER_DEF_NORMAL));
    }
    return ret;
}

static ge::graphStatus TilingFuncXdivy(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin the tiling process for Arch35 architecture");
    XdivyTiling xdivyTiling(context);
    auto ret = xdivyTiling.RunTiling();
    if (ret != GRAPH_SUCCESS)
        return ret;
    size_t* workspaces = context->GetWorkspaceSizes(1);
    workspaces[0] = 16 * 1024 * 1024; // 16MB
    return GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForXdivy(gert::TilingParseContext* context)
{
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    auto compileInfo = context->GetCompiledInfo<XdivyCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto ap = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ap.GetCoreNumAiv();
    ap.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    return GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Xdivy).Tiling(TilingFuncXdivy).TilingParse<XdivyCompileInfo>(TilingPrepareForXdivy);

} // namespace optiling
