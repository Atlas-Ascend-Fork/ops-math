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
 * \file aclnn_histc.cpp
 * \brief
 */
#include "op_common/op_host/util/op_const_def.h"
#include <cmath>
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_histc.h"
#include "histogram.h"
#include "../../zero_op/op_api/zero_op.h"
#include "../../reduce_min/op_api/reduce_min.h"
#include "../../reduce_max/op_api/reduce_max.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "op_api/aclnn_check.h"
using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

constexpr size_t MAX_DIM = 8;

// 910b 等旧芯片无 bf16 kernel（def 仅 950/960DT/350 注册 bf16），bf16 只允许 regbase 平台通过校验，
// 旧芯片保持"aclnn 层干净拒绝 161002"的既有行为
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_910B = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT64, op::DataType::DT_INT32,
    op::DataType::DT_INT16, op::DataType::DT_INT8,    op::DataType::DT_UINT8};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_REGBASE = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16, op::DataType::DT_INT64,
    op::DataType::DT_INT32, op::DataType::DT_INT16,   op::DataType::DT_INT8, op::DataType::DT_UINT8};

static const std::initializer_list<op::DataType> DTYPE_INT_LIST = {op::DataType::DT_INT64, op::DataType::DT_INT32,
                                                                   op::DataType::DT_INT16, op::DataType::DT_INT8,
                                                                   op::DataType::DT_UINT8};

static const std::initializer_list<op::DataType> DTYPE_FLOAT_LIST = {op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

static bool CheckNotNull(const aclTensor* self, const aclScalar* min, const aclScalar* max, aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(out, return false);
    OP_CHECK_NULL(max, return false);
    OP_CHECK_NULL(min, return false);
    return true;
}

static bool SelfDTypeInt(op::DataType selfDType)
{
    auto it = std::find(DTYPE_INT_LIST.begin(), DTYPE_INT_LIST.end(), selfDType);
    if (it != DTYPE_INT_LIST.end()) {
        return true;
    }
    return false;
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* out)
{
    // 检查self的数据类型是否在Histogram算子的支持列表内（bf16 仅 regbase，旧芯片不含 bf16 kernel）
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    const auto& supportList = IsRegBase(npuArch) ? DTYPE_SUPPORT_LIST_REGBASE : DTYPE_SUPPORT_LIST_910B;
    OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);

    OP_CHECK_DTYPE_NOT_SUPPORT(out, supportList, return false);

    return true;
}

static bool CheckPromoteType(const aclTensor* self, const aclTensor* out, op::DataType promoteType)
{
    if (promoteType == DataType::DT_UNDEFINED) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Self dtype %s can not cast to promote dtype %s.",
                op::ToString(self->GetDataType()).GetString(), op::ToString(DataType::DT_UNDEFINED).GetString());
        return false;
    }

    // 检查self的数据类型能否转换为输出的数据类型
    OP_CHECK_RESULT_DTYPE_CAST_FAILED(promoteType, self->GetDataType(), return false);

    // 检查out的数据类型能否转换为输出的数据类型
    OP_CHECK_RESULT_DTYPE_CAST_FAILED(promoteType, out->GetDataType(), return false);

    return true;
}

static bool CheckMinMaxInfEqual(float minValue, float maxValue)
{
    if ((std::isinf(minValue) && minValue > 0 && std::isinf(maxValue) && maxValue > 0) ||
        (std::isinf(minValue) && minValue < 0 && std::isinf(maxValue) && maxValue < 0)) {
        return true;
    }
    return false;
}

static bool CheckMinMaxIsInfNan(float minValue, float maxValue)
{
    // 特殊处理 min == max == inf/-inf 的场景
    if (CheckMinMaxInfEqual(minValue, maxValue)) {
        return true;
    }

    if (std::isinf(minValue) || std::isinf(maxValue) || std::isnan(minValue) || std::isnan(maxValue)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "range of [%f, %f] is not finite", minValue, maxValue);
        return false;
    }
    return true;
}

// CUDA 语义: min/max 标量在 dispatch 入口转换为 acc_type(整数一律 int64, fp32 为 float),
// 不存在"标量超出 x dtype 值域"的问题。NPU kernel 要求 min/max 与 x 同 dtype, 标量超出 x 值域
// 会被 ConvertToTensor 静默截断/回绕(M3a: int8 输入 + ±1000 标量 -> 回绕后 min>max, 静默全零)。
// RegBase 上对标 CUDA: 把 x 与标量共同提升到能容纳标量的最窄类型再下发。
static bool ScalarOutOfIntRange(const aclScalar* scalar, double lo, double hi)
{
    // CUDA double payload -> int64 为向零截断(checked_convert), 截断后再做范围比较
    double value = std::trunc(scalar->ToDouble());
    return value < lo || value > hi;
}

static op::DataType WidenIntDtypeForScalars(const aclScalar* min, const aclScalar* max, double xLo, double xHi)
{
    if (!ScalarOutOfIntRange(min, xLo, xHi) && !ScalarOutOfIntRange(max, xLo, xHi)) {
        return op::DataType::DT_UNDEFINED; // 标量在 x 值域内, 无需提升
    }
    constexpr double kInt32Lo = -2147483648.0;
    constexpr double kInt32Hi = 2147483647.0;
    if (!ScalarOutOfIntRange(min, kInt32Lo, kInt32Hi) && !ScalarOutOfIntRange(max, kInt32Lo, kInt32Hi)) {
        return op::DataType::DT_INT32;
    }
    return op::DataType::DT_INT64;
}

static op::DataType WidenDtypeForScalars(op::DataType xDtype, const aclScalar* min, const aclScalar* max)
{
    switch (xDtype) {
        case op::DataType::DT_INT8:
            return WidenIntDtypeForScalars(min, max, -128.0, 127.0);
        case op::DataType::DT_UINT8:
            return WidenIntDtypeForScalars(min, max, 0.0, 255.0);
        case op::DataType::DT_INT16:
            return WidenIntDtypeForScalars(min, max, -32768.0, 32767.0);
        case op::DataType::DT_INT32:
            return WidenIntDtypeForScalars(min, max, -2147483648.0, 2147483647.0);
        case op::DataType::DT_FLOAT16: {
            // fp16 是 NPU 扩展支持(CUDA 不支持), 标量超 fp16 值域(65504)时提升到 fp32
            constexpr double kFp16Max = 65504.0;
            if (std::fabs(min->ToDouble()) > kFp16Max || std::fabs(max->ToDouble()) > kFp16Max) {
                return op::DataType::DT_FLOAT;
            }
            return op::DataType::DT_UNDEFINED;
        }
        case op::DataType::DT_BF16: {
            // bf16 同为 NPU 扩展支持(CUDA 不支持), 标量超 bf16 值域(~3.39e38)时提升到 fp32
            constexpr double kBf16Max = 3.3895314e38;
            if (std::fabs(min->ToDouble()) > kBf16Max || std::fabs(max->ToDouble()) > kBf16Max) {
                return op::DataType::DT_FLOAT;
            }
            return op::DataType::DT_UNDEFINED;
        }
        default:
            // DT_INT64 无更宽类型可提升; DT_FLOAT 不需要
            return op::DataType::DT_UNDEFINED;
    }
}

// CUDA checked_convert 语义: 浮点 payload 标量转换为 int64 bounds 时, |v| >= 2^63 或 NaN
// 响亮报错 ("value cannot be converted to type int64_t without overflow"), 不允许静默截断。
static bool CheckScalarConvertibleToInt64(const aclScalar* scalar)
{
    op::DataType payloadDtype = scalar->GetDataType();
    switch (payloadDtype) {
        case op::DataType::DT_INT8:
        case op::DataType::DT_INT16:
        case op::DataType::DT_INT32:
        case op::DataType::DT_INT64:
        case op::DataType::DT_UINT8:
        case op::DataType::DT_UINT16:
        case op::DataType::DT_UINT32:
        case op::DataType::DT_UINT64:
        case op::DataType::DT_BOOL:
            return true; // int payload 在 int64 内恒精确
        default:
            break; // 其余(含 DT_BFLOAT16)统一走 ToDouble 检查, 避免绕过 checked_convert 语义
    }
    double value = scalar->ToDouble();
    if (std::isnan(value) || value >= 9223372036854775808.0 || value < -9223372036854775808.0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "value cannot be converted to type int64_t without overflow");
        return false;
    }
    return true;
}

static bool CheckValueRange(int64_t bins, const aclScalar* min, const aclScalar* max, op::DataType selfDType)
{
    if (bins <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The value of bins must be greater than 0.");
        return false;
    }

    if (SelfDTypeInt(selfDType)) {
        // CUDA 语义: 整数输入的 bounds 以 int64(acc_type) 比较; RegBase 上用 ToInt64,
        // 避免 int64 大值域标量经 ToInt32 截断后比较结果错误。
        auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
        if (IsRegBase(npuArch)) {
            if (!CheckScalarConvertibleToInt64(min) || !CheckScalarConvertibleToInt64(max)) {
                return false;
            }
            int64_t minValue = min->ToInt64();
            int64_t maxValue = max->ToInt64();
            if (minValue > maxValue) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The value of max should be greater than or equal to min.");
                return false;
            }
            return true;
        }
        int32_t minValue = min->ToInt32();
        int32_t maxValue = max->ToInt32();
        if (minValue > maxValue) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The value of max should be greater than or equal to min.");
            return false;
        }
    } else {
        float minValue = min->ToFloat();
        float maxValue = max->ToFloat();
        if (minValue > maxValue) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The value of max should be greater than or equal to min.");
            return false;
        }
    }
    return true;
}

static bool CheckShape(const aclTensor* self, int64_t bins, const aclTensor* out)
{
    OP_CHECK_WRONG_DIMENSION(out, 1, return false);

    OP_CHECK_MAX_DIM(self, MAX_DIM, return false);

    int64_t outSize = 1;
    op::Shape outShape = out->GetViewShape();
    size_t outDimNum = outShape.GetDimNum();
    for (size_t idx = 0; idx < outDimNum; idx++) {
        outSize *= outShape.GetDim(idx);
    }

    if (outSize != bins) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The size of out tensor should be same as bins.");
        return false;
    }

    return true;
}

static bool NeedComputeMinMax(const aclScalar* min, const aclScalar* max, op::DataType selfDType)
{
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    if (!(npuArch == Ops::Base::DAV_2201 || IsRegBase(npuArch))) {
        return false;
    }

    if (SelfDTypeInt(selfDType)) {
        int64_t minValue = min->ToInt64();
        int64_t maxValue = max->ToInt64();
        return minValue == maxValue;
    }
    float minValue = min->ToFloat();
    float maxValue = max->ToFloat();
    if (CheckMinMaxInfEqual(minValue, maxValue)) {
        return true;
    }
    // CUDA 语义: 标量转换到 acc_type 后按精确相等判断重算; RegBase 上与 CUDA 对齐。
    // 910b(DAV_2201) 保持原有 1e-6 epsilon 判断不变。
    if (IsRegBase(npuArch)) {
        return minValue == maxValue;
    }
    return maxValue - minValue < static_cast<float>(1e-6) && maxValue - minValue > static_cast<float>(-1e-6);
}

static std::tuple<const aclTensor*, const aclTensor*> AllMinMax(const aclTensor* self, aclOpExecutor* executor)
{
    if (self->GetViewShape().GetDimNum() == 0) {
        return std::tuple<const aclTensor*, const aclTensor*>(self, self);
    }
    size_t dimDum = self->GetViewShape().GetDimNum();
    int64_t appendDim[dimDum];
    for (int64_t i = 0; i < static_cast<int64_t>(dimDum); i++) {
        appendDim[i] = i;
    }
    auto dim = executor->AllocIntArray(appendDim, dimDum);

    // 进行reducemin计算
    const aclTensor* min = l0op::ReduceMin(self, dim, false, executor);
    // 进行reducemax计算
    const aclTensor* max = l0op::ReduceMax(self, dim, false, true, executor);
    return std::tuple<const aclTensor*, const aclTensor*>(min, max);
}

static aclnnStatus CheckHistcParams(const aclTensor* self, int64_t bins, const aclScalar* min, const aclScalar* max,
                                    aclTensor* out)
{
    // 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, min, max, out), ACLNN_ERR_PARAM_NULLPTR);

    // 检查输入的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

    // 检查min, max是否为非法的inf/nan
    float minValue = min->ToFloat();
    float maxValue = max->ToFloat();
    CHECK_RET(CheckMinMaxIsInfNan(minValue, maxValue), ACLNN_ERR_PARAM_INVALID);

    // 检查bins, min, max的取值范围
    CHECK_RET(CheckValueRange(bins, min, max, self->GetDataType()), ACLNN_ERR_PARAM_INVALID);

    // 检查self和out能否做数据类型推导以及推导的数据类型能否转换为输出数据类型
    CHECK_RET(CheckPromoteType(self, out, out->GetDataType()), ACLNN_ERR_PARAM_INVALID);

    // 检查shape是否一致
    CHECK_RET(CheckShape(self, bins, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus EmptyTensor(aclTensor* out, aclOpExecutor* executor)
{
    auto outContiguous = l0op::Contiguous(out, executor);
    CHECK_RET(outContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 调用ZerosLike算子kernel
    auto zeroOut = l0op::ZerosLike(outContiguous, executor);
    CHECK_RET(zeroOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyOut = l0op::ViewCopy(zeroOut, out, executor);
    CHECK_RET(viewCopyOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static void CheckFormat(const aclTensor* self)
{
    ge::Format selfStorageFormat = self->GetStorageFormat();
    if (selfStorageFormat == ge::Format::FORMAT_FRACTAL_NZ) {
        OP_LOGW("aclnnHistc doesn't support format NZ.");
    }
}

aclnnStatus aclnnHistcGetWorkspaceSize(const aclTensor* self, int64_t bins, const aclScalar* min, const aclScalar* max,
                                       aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnHistc, DFX_IN(self, bins, min, max), DFX_OUT(out));

    // 参数检查
    auto ret = CheckHistcParams(self, bins, min, max, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    CheckFormat(self);

    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();

    if (self->IsEmpty()) {
        auto status = EmptyTensor(out, uniqueExecutor.get());
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return status;
    }

    // 将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    aclOpExecutor* executorP = uniqueExecutor.get();

    // RegBase 上对标 CUDA 的 acc_type 语义: 标量超出 x dtype 值域时, 把 x 与标量
    // 共同提升到更宽类型, 避免 ConvertToTensor 按 x dtype 静默截断/回绕
    op::DataType xEffDtype = selfContiguous->GetDataType();
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    if (IsRegBase(npuArch)) {
        op::DataType widenDtype = WidenDtypeForScalars(xEffDtype, min, max);
        if (widenDtype != op::DataType::DT_UNDEFINED) {
            selfContiguous = l0op::Cast(selfContiguous, widenDtype, executorP);
            CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
            xEffDtype = widenDtype;
        }
    }

    // min, max转tensor
    auto minTensor = executorP->ConvertToTensor(min, xEffDtype);
    CHECK_RET(minTensor != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    auto maxTensor = executorP->ConvertToTensor(max, xEffDtype);
    CHECK_RET(maxTensor != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    // 重新求self中的min max值
    if (NeedComputeMinMax(min, max, xEffDtype)) {
        auto minMaxResult = AllMinMax(selfContiguous, uniqueExecutor.get());
        minTensor = std::get<0>(minMaxResult);
        maxTensor = std::get<1>(minMaxResult);
        CHECK_RET(minTensor != nullptr && maxTensor != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    }

    float minValue = min->ToFloat();
    float maxValue = max->ToFloat();

    // 调用Histogram算子kernel
    auto HistogramCal = l0op::Histogram(selfContiguous, minTensor, maxTensor, out, bins, minValue, maxValue,
                                        uniqueExecutor.get());
    CHECK_RET(HistogramCal != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将计算结果转换成输出out的数据类型
    auto castOut = l0op::Cast(HistogramCal, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将计算结果拷贝到输出out上
    auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnHistc(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnHistc);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
