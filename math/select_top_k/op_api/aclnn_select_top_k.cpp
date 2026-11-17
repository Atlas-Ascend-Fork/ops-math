/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_select_top_k.h"
#include "select_top_k.h"
#include "../../topk/op_api/topk.h"
#include "../../sort_with_index/op_api/sort_with_index.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include <algorithm>
#include <limits>
#include <vector>

namespace SelectTopKPolicy {
constexpr int64_t RADIX_LARGE_AXIS_MIN = 12000000;       // 第一档长轴门槛。
constexpr int64_t RADIX_LARGE_AXIS_K_RATIO = 100;        // 第一档要求 N > 100K。
constexpr int64_t RADIX_VERY_LARGE_AXIS_MIN = 100000000; // 第二档长轴门槛。
constexpr int64_t RADIX_VERY_LARGE_AXIS_K_RATIO = 50;    // 第二档要求 N > 50K。
constexpr int64_t RADIX_MIN_K = 8;                       // 与 aclnnTopk PR 14011 的包含式下限一致。

// 判断当前接口轴长范围内的 Radix 收益条件；芯片、类型及扩展语义由 L2 单独校验。
bool IsRadixShapeSupported(int64_t axis, int64_t k)
{
    if (k < RADIX_MIN_K || k > axis || axis > std::numeric_limits<int32_t>::max()) {
        return false;
    }
    if (axis >= RADIX_VERY_LARGE_AXIS_MIN) {
        return axis > RADIX_VERY_LARGE_AXIS_K_RATIO * k;
    }
    return axis >= RADIX_LARGE_AXIS_MIN && axis > RADIX_LARGE_AXIS_K_RATIO * k;
}
} // namespace SelectTopKPolicy

namespace {
constexpr int64_t MAX_INPUT_RANK = 8;    // 接口支持的最大输入维数。
constexpr int64_t MAX_FUSED_INNER = 256; // A2 融合布局的最大轴后元素数，与Host资源规划上限一致。
constexpr int64_t INNER_LAYOUT_DIM = 2;  // 三维布局 [outer, axis, inner] 的末维位置。
constexpr size_t AXIS_LAYOUT_RANK = 3;   // 非末轴输入与输出使用的中间布局维数。
constexpr int64_t MAX_ELEMENT_BYTES = sizeof(int64_t); // 地址溢出检查采用最宽的索引类型。

// 将任意输入维度拆成排序轴前、轴内和轴后三部分。
struct Layout {
    int64_t outer = 1;
    int64_t axis = 0;
    int64_t inner = 1;
    int64_t rows = 0;
    op::Shape outputShape;
};

// 接受具有普通步长的公开格式，拒绝需要特殊存储解释的私有格式。
bool IsPlainFormat(const aclTensor* tensor)
{
    // NC/NCL/NCHW 与 ND 一样按普通步长解释；分块私有格式不满足该条件。
    return !op::IsPrivateFormat(tensor->GetViewFormat()) && !op::IsPrivateFormat(tensor->GetStorageFormat());
}

// 计算非负维度乘积，在有符号地址范围内检查溢出。
bool Multiply(int64_t a, int64_t b, int64_t& result)
{
    if (a < 0 || b < 0 || (b && a > INT64_MAX / b)) {
        return false;
    }
    result = a * b;
    return true;
}

// 检查可选区间张量：INT32、公开格式，且为逐行数据或单值广播。
bool CheckBounds(const aclTensor* bounds, int64_t rows)
{
    if (bounds == nullptr) {
        return true;
    }
    return bounds->GetDataType() == op::DataType::DT_INT32 && IsPlainFormat(bounds) &&
           (bounds->GetViewShape().GetShapeSize() == rows || bounds->GetViewShape().GetShapeSize() == 1);
}

// 按输出模式验证指针、数据类型及完整输出形状。
bool CheckOutput(const aclTensor* output, bool enabled, op::DataType dtype, const op::Shape& shape)
{
    if (!enabled) {
        return output == nullptr;
    }
    return output != nullptr && output->GetDataType() == dtype && IsPlainFormat(output) &&
           output->GetViewShape() == shape;
}

// 校验接口参数并计算轴布局，同时保证偏移、填充值和地址乘积均可表示。
aclnnStatus CheckParams(const aclTensor* self, int64_t k, int64_t dim, aclDataType indicesType, int64_t order,
                        int64_t offset, int64_t mode, const aclTensor* begin, const aclTensor* end, int64_t indexFill,
                        const aclTensor* values, const aclTensor* indices, Layout& layout)
{
    if (self == nullptr) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    const auto& shape = self->GetViewShape();
    const int64_t rank = shape.GetDimNum();
    const auto dtype = self->GetDataType();
    if (rank < 1 || rank > MAX_INPUT_RANK || dim < -rank || dim >= rank || k < 0 || k > INT32_MAX ||
        (indicesType != ACL_INT32 && indicesType != ACL_INT64) ||
        (order != ACLNN_SELECT_TOP_K_ORDER_VALUE && order != ACLNN_SELECT_TOP_K_ORDER_NONE) ||
        mode < ACLNN_SELECT_TOP_K_OUTPUT_VALUES || mode > ACLNN_SELECT_TOP_K_OUTPUT_BOTH ||
        (dtype != op::DataType::DT_FLOAT && dtype != op::DataType::DT_FLOAT16 && dtype != op::DataType::DT_BF16) ||
        !IsPlainFormat(self)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "SelectTopK: invalid rank/dim/k, dtype, ND format, order or output mode");
        return ACLNN_ERR_PARAM_INVALID;
    }
    dim = dim < 0 ? dim + rank : dim;
    layout.axis = shape.GetDim(dim);
    if (layout.axis < 0 || layout.axis > INT32_MAX) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    for (int64_t i = 0; i < rank; ++i) {
        if (i < dim && !Multiply(layout.outer, shape.GetDim(i), layout.outer)) {
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (i > dim && !Multiply(layout.inner, shape.GetDim(i), layout.inner)) {
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    int64_t inputItems = 0, outputItems = 0;
    if (!Multiply(layout.outer, layout.inner, layout.rows) || !Multiply(layout.rows, layout.axis, inputItems) ||
        !Multiply(layout.rows, k, outputItems) || inputItems > INT64_MAX / MAX_ELEMENT_BYTES ||
        outputItems > INT64_MAX / MAX_ELEMENT_BYTES) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    layout.outputShape = shape;
    layout.outputShape.SetDim(dim, k);
    const int64_t maxIndex = std::max<int64_t>(layout.axis - 1, 0);
    const int64_t upper = indicesType == ACL_INT32 ? INT32_MAX : INT64_MAX;
    const int64_t lower = indicesType == ACL_INT32 ? INT32_MIN : INT64_MIN;
    if (offset < lower || offset > upper - maxIndex || indexFill < lower || indexFill > upper ||
        !CheckBounds(begin, layout.rows) || !CheckBounds(end, layout.rows) ||
        !CheckOutput(values, mode != ACLNN_SELECT_TOP_K_OUTPUT_INDICES, dtype, layout.outputShape) ||
        !CheckOutput(indices, mode != ACLNN_SELECT_TOP_K_OUTPUT_VALUES, static_cast<op::DataType>(indicesType),
                     layout.outputShape)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "SelectTopK: invalid bounds/output shape or index overflow");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

// 将形状参数交给执行器管理，并创建新的张量视图。
const aclTensor* Reshape(const aclTensor* tensor, const std::vector<int64_t>& shape, aclOpExecutor* executor)
{
    auto dims = executor->AllocIntArray(shape.data(), shape.size());
    return dims == nullptr ? nullptr : l0op::Reshape(tensor, dims, executor);
}

// 准备统一三维输入；融合路径保留 [outer, axis, inner]，其余路径整理为 [rows, axis, 1]。
const aclTensor* PrepareInput(const aclTensor* self, const Layout& layout, aclOpExecutor* executor, bool fusedInput)
{
    if (layout.axis == 0) {
        // Reshape 不接受空形状，直接创建空视图；内核仅写入请求的填充值和填充索引。
        auto empty = executor->CreateView(self, op::Shape{layout.rows, 0, 1}, self->GetViewOffset());
        return empty == nullptr ? nullptr : l0op::ReFormat(empty, op::Format::FORMAT_ND);
    }
    auto input = l0op::Contiguous(self, executor);
    if (input == nullptr) {
        return nullptr;
    }
    if (layout.inner > 1 && layout.axis > 1 && !fusedInput) {
        input = Reshape(input, {layout.outer, layout.axis, layout.inner}, executor);
        const int64_t permutation[] = {0, INNER_LAYOUT_DIM, 1};
        auto axes = executor->AllocIntArray(permutation, AXIS_LAYOUT_RANK);
        if (input == nullptr || axes == nullptr) {
            return nullptr;
        }
        input = l0op::Transpose(input, axes, executor);
        if (input == nullptr) {
            return nullptr;
        }
    }
    return fusedInput ? Reshape(input, {layout.outer, layout.axis, layout.inner}, executor) :
                        Reshape(input, {layout.rows, layout.axis, 1}, executor);
}

// 将计算结果恢复到原轴位置；融合输出已经完成搬运时只恢复形状。
const aclTensor* RestoreOutput(const aclTensor* tensor, const Layout& layout, int64_t k, aclOpExecutor* executor,
                               bool restored)
{
    if (!restored && layout.inner > 1 && k > 1) {
        tensor = Reshape(tensor, {layout.outer, layout.inner, k}, executor);
        const int64_t permutation[] = {0, INNER_LAYOUT_DIM, 1};
        auto axes = executor->AllocIntArray(permutation, AXIS_LAYOUT_RANK);
        if (tensor == nullptr || axes == nullptr) {
            return nullptr;
        }
        tensor = l0op::Transpose(tensor, axes, executor);
        if (tensor == nullptr) {
            return nullptr;
        }
    }
    const auto shape = op::ToShapeVector(layout.outputShape);
    auto dims = executor->AllocIntArray(shape.data(), shape.size());
    return dims == nullptr ? nullptr : l0op::Reshape(tensor, dims, executor);
}
// 将 aclnnTopk 的 Radix 门控限制在当前接口的标准语义内，扩展请求继续使用 SelectTopK。
bool IsRadixTopKSupported(const aclTensor* self, const Layout& layout, int64_t k, int64_t outputMode,
                          int64_t indicesOffset, const aclTensor* begin, const aclTensor* end)
{
    const auto version = op::GetCurrentPlatformInfo().GetSocVersion();
    const bool socSupported = version == op::SocVersion::ASCEND910B || version == op::SocVersion::ASCEND910_93;
    const auto dtype = self->GetDataType();
    const bool dtypeSupported = dtype == op::DataType::DT_FLOAT16 || dtype == op::DataType::DT_BF16;
    return socSupported && dtypeSupported && outputMode == ACLNN_SELECT_TOP_K_OUTPUT_BOTH && indicesOffset == 0 &&
           begin == nullptr && end == nullptr && SelectTopKPolicy::IsRadixShapeSupported(layout.axis, k);
}

// Radix 分流保持原维数；非末轴先交换排序轴与末轴，排序时同步维护原轴索引。
aclnnStatus RunRadixTopK(const aclTensor* self, int64_t k, int64_t dim, bool largest, int64_t order,
                         op::DataType indicesDType, aclTensor* valuesOut, aclTensor* indicesOut,
                         aclOpExecutor* executor)
{
    auto input = l0op::Contiguous(self, executor);
    CHECK_RET(input != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const int64_t rank = input->GetViewShape().GetDimNum();
    const int64_t positiveDim = dim < 0 ? dim + rank : dim;
    const int64_t lastDim = rank - 1;
    aclIntArray* axes = nullptr;
    if (positiveDim != lastDim) {
        std::vector<int64_t> permutation(rank);
        for (int64_t i = 0; i < rank; ++i) {
            permutation[i] = i;
        }
        std::swap(permutation[positiveDim], permutation[lastDim]);
        axes = executor->AllocIntArray(permutation.data(), rank);
        CHECK_RET(axes != nullptr, ACLNN_ERR_INNER_NULLPTR);
        input = l0op::Transpose(input, axes, executor);
        CHECK_RET(input != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    OP_LOGD("SelectTopK L2 route=Topk, radix_gate=true, dim=%ld, k=%ld, order=%ld", lastDim, k, order);
    const auto selected = l0op::Topk(input, k, lastDim, largest, order == ACLNN_SELECT_TOP_K_ORDER_VALUE, indicesDType,
                                     executor);
    const aclTensor* values = std::get<0>(selected);
    const aclTensor* indices = std::get<1>(selected);
    CHECK_RET(values != nullptr && indices != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 按值排序时同步重排 Radix 返回的原轴索引；无序输出直接透传。
    if (order == ACLNN_SELECT_TOP_K_ORDER_VALUE && indices->GetDataType() != op::DataType::DT_INT64) {
        const auto ordered = l0op::SortWithIndex(values, indices, lastDim, largest, true, executor);
        values = std::get<0>(ordered);
        indices = std::get<1>(ordered);
        CHECK_RET(values != nullptr && indices != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (axes != nullptr) {
        values = l0op::Transpose(values, axes, executor);
        indices = l0op::Transpose(indices, axes, executor);
        CHECK_RET(values != nullptr && indices != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    CHECK_RET(l0op::ViewCopy(values, valuesOut, executor) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (indices->GetDataType() != indicesDType) {
        indices = l0op::Cast(indices, indicesDType, executor);
        CHECK_RET(indices != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    CHECK_RET(l0op::ViewCopy(indices, indicesOut, executor) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}
} // namespace

// 构建完整 L2 执行计划，包含输入准备、TopK、索引转换及输出恢复。
aclnnStatus aclnnSelectTopKGetWorkspaceSize(const aclTensor* self, int64_t k, int64_t dim, bool largest,
                                            aclDataType indicesType, int64_t order, int64_t indicesOffset,
                                            int64_t outputMode, const aclTensor* beginOptional,
                                            const aclTensor* endOptional, const aclScalar* valueFill, int64_t indexFill,
                                            aclTensor* valuesOutOptional, aclTensor* indicesOutOptional,
                                            uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnSelectTopK,
                   DFX_IN(self, k, dim, largest, indicesType, order, indicesOffset, outputMode, beginOptional,
                          endOptional, valueFill, indexFill),
                   DFX_OUT(valuesOutOptional, indicesOutOptional));
    CHECK_RET(workspaceSize != nullptr && executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(valueFill == nullptr || op::CanCast(valueFill->GetDataType(), op::DataType::DT_FLOAT),
              ACLNN_ERR_PARAM_INVALID);
    Layout layout;
    auto status = CheckParams(self, k, dim, indicesType, order, indicesOffset, outputMode, beginOptional, endOptional,
                              indexFill, valuesOutOptional, indicesOutOptional, layout);
    CHECK_RET(status == ACLNN_SUCCESS, status);
    CHECK_RET(op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2201, ACLNN_ERR_PARAM_INVALID);
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    if (layout.rows == 0 || k == 0) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    if (IsRadixTopKSupported(self, layout, k, outputMode, indicesOffset, beginOptional, endOptional)) {
        status = RunRadixTopK(self, k, dim, largest, order, static_cast<op::DataType>(indicesType), valuesOutOptional,
                              indicesOutOptional, uniqueExecutor.get());
        CHECK_RET(status == ACLNN_SUCCESS, status);
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    const bool fusedOutput = outputMode == ACLNN_SELECT_TOP_K_OUTPUT_BOTH && beginOptional == nullptr &&
                             endOptional == nullptr && indicesOffset == 0 && layout.inner > 1 &&
                             layout.inner <= MAX_FUSED_INNER && k > 1 && k <= layout.axis;
    const bool fusedInput = fusedOutput && layout.axis > 1;
    auto input = PrepareInput(self, layout, uniqueExecutor.get(), fusedInput);
    CHECK_RET(input != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor* begin = beginOptional == nullptr ? nullptr : l0op::Contiguous(beginOptional, uniqueExecutor.get());
    const aclTensor* end = endOptional == nullptr ? nullptr : l0op::Contiguous(endOptional, uniqueExecutor.get());
    CHECK_RET((beginOptional == nullptr || begin != nullptr) && (endOptional == nullptr || end != nullptr),
              ACLNN_ERR_INNER_NULLPTR);
    // A2 的 INT64 转置开销较高；索引范围可表示时先用 INT32 恢复轴布局，再扩宽。
    // 无法用 INT32 表示的偏移和填充索引始终保留 INT64 路径。
    const bool widenAfterTranspose = !fusedOutput && outputMode != ACLNN_SELECT_TOP_K_OUTPUT_VALUES &&
                                     indicesType == ACL_INT64 && layout.inner > 1 && k > 1 &&
                                     indicesOffset >= INT32_MIN &&
                                     indicesOffset <= INT32_MAX - std::max<int64_t>(0, layout.axis - 1) &&
                                     indexFill >= INT32_MIN && indexFill <= INT32_MAX;
    const auto kernelIndicesType = widenAfterTranspose ? op::DataType::DT_INT32 :
                                                         static_cast<op::DataType>(indicesType);
    auto result = l0op::SelectTopK(input, begin, end, k, largest, order, kernelIndicesType, indicesOffset, outputMode,
                                   valueFill == nullptr ? 0.0f : valueFill->ToFloat(), indexFill, uniqueExecutor.get());
    if (outputMode != ACLNN_SELECT_TOP_K_OUTPUT_INDICES) {
        CHECK_RET(std::get<0>(result) != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto values = RestoreOutput(std::get<0>(result), layout, k, uniqueExecutor.get(), fusedOutput);
        CHECK_RET(values != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(l0op::ViewCopy(values, valuesOutOptional, uniqueExecutor.get()) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (outputMode != ACLNN_SELECT_TOP_K_OUTPUT_VALUES) {
        CHECK_RET(std::get<1>(result) != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto indices = RestoreOutput(std::get<1>(result), layout, k, uniqueExecutor.get(), fusedOutput);
        CHECK_RET(indices != nullptr, ACLNN_ERR_INNER_NULLPTR);
        if (widenAfterTranspose) {
            indices = l0op::Cast(indices, op::DataType::DT_INT64, uniqueExecutor.get());
            CHECK_RET(indices != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
        CHECK_RET(l0op::ViewCopy(indices, indicesOutOptional, uniqueExecutor.get()) != nullptr,
                  ACLNN_ERR_INNER_NULLPTR);
    }
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

// 在指定流上执行第一阶段生成的完整计算计划。
aclnnStatus aclnnSelectTopK(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnSelectTopK);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
