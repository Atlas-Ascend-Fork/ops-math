/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_as_strided.h"
#include "as_strided.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "op_api/aclnn_check.h"
#include "op_api/op_api_def_math.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// 根据API定义，需要列出所能支持的所有dtype，与算子README约束一致
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT64,        op::DataType::DT_UINT64,    op::DataType::DT_INT32,    op::DataType::DT_UINT32,
    op::DataType::DT_FLOAT,        op::DataType::DT_FLOAT16,   op::DataType::DT_INT8,     op::DataType::DT_UINT8,
    op::DataType::DT_BF16,         op::DataType::DT_INT16,     op::DataType::DT_UINT16,   op::DataType::DT_BOOL,
    op::DataType::DT_COMPLEX32,    op::DataType::DT_COMPLEX64, op::DataType::DT_HIFLOAT8, op::DataType::DT_FLOAT8_E5M2,
    op::DataType::DT_FLOAT8_E4M3FN};

static bool CheckPlatform()
{
    // 检查芯片架构是否支持：算子仅在 regbase 平台
    if (!IsRegBase()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "AsStrided is not supported on this device.");
        return false;
    }
    return true;
}

static bool CheckNotNull(const aclTensor* x, const aclTensor* y, const aclIntArray* size, const aclIntArray* stride)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(y, return false);
    OP_CHECK_NULL(size, return false);
    OP_CHECK_NULL(stride, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* x, const aclTensor* y)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(x, DTYPE_SUPPORT_LIST, return false);
    // as_strided只做按stride搬运，输出与输入数据类型需一致
    OP_CHECK_DTYPE_NOT_SAME(x, y, return false);
    return true;
}

static bool CheckSizeAndStride(const aclIntArray* size, const aclIntArray* stride)
{
    OP_CHECK(size->Size() == stride->Size(),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "length of size [%llu] and stride [%llu] must be the same.",
                     static_cast<unsigned long long>(size->Size()), static_cast<unsigned long long>(stride->Size())),
             return false);
    OP_CHECK(size->Size() <= MAX_SUPPORT_DIMS_NUMS,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "length of size [%llu] must not be greater than %zu.",
                     static_cast<unsigned long long>(size->Size()), MAX_SUPPORT_DIMS_NUMS),
             return false);
    for (uint64_t i = 0; i < size->Size(); i++) {
        OP_CHECK(
            (*size)[i] >= 0,
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "size[%llu] must be non-negative.", static_cast<unsigned long long>(i)),
            return false);
        OP_CHECK(
            (*stride)[i] >= 0,
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "stride[%llu] must be non-negative.", static_cast<unsigned long long>(i)),
            return false);
    }
    return true;
}

static bool CheckShape(const aclTensor* y, const aclIntArray* size, const aclIntArray* stride, int64_t storageOffset)
{
    OP_CHECK(storageOffset >= 0,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "storageOffset [%ld] must be non-negative.", storageOffset),
             return false);
    CHECK_RET(CheckSizeAndStride(size, stride), false);
    // 输出形状由size决定
    op::Shape expectShape;
    for (uint64_t i = 0; i < size->Size(); i++) {
        expectShape.AppendDim((*size)[i]);
    }
    OP_CHECK(y->GetViewShape() == expectShape,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "shape of y [%s] must be equal to size [%s].",
                     op::ToString(y->GetViewShape()).GetString(), op::ToString(expectShape).GetString()),
             return false);
    return true;
}

static void CheckFormat(const aclTensor* x, const aclTensor* y)
{
    op::Format xFormat = x->GetStorageFormat();
    if (IsPrivateFormat(xFormat)) {
        OP_LOGW("Format of x is [%s], this format may lead to precision failure", op::ToString(xFormat).GetString());
    }
    op::Format yFormat = y->GetStorageFormat();
    if (IsPrivateFormat(yFormat)) {
        OP_LOGW("Format of y is [%s], this format may lead to precision failure", op::ToString(yFormat).GetString());
    }
}

static bool CheckGatherRange(const aclTensor* x, const aclIntArray* size, const aclIntArray* stride,
                             int64_t storageOffset)
{
    // 输出为空（size含0）时不读取数据，不做边界校验（与tiling的HandleEmptyTensor语义一致）
    int64_t maxIdx = storageOffset;
    for (uint64_t i = 0; i < size->Size(); i++) {
        if ((*size)[i] == 0) {
            return true;
        }
        int64_t extent = 0;
        // gather最大下标 = storageOffset + Σ(size[d]-1)*stride[d]；乘加带int64溢出保护，
        // 溢出按下标必然越界处理（tiling侧为裸int64累加，此处前置拦截防回绕放行越界读）
        if (__builtin_mul_overflow((*size)[i] - 1, (*stride)[i], &extent)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Gather range overflow at dim %llu.", static_cast<unsigned long long>(i));
            return false;
        }
        if (__builtin_add_overflow(maxIdx, extent, &maxIdx)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Gather range overflow when accumulating max index.");
            return false;
        }
    }
    // 可读区域从x视图起点起算：存储元素数 - 视图偏移（与CreateView校正后的tiling校验口径一致）
    int64_t storageNumel = 1;
    auto storageShape = x->GetStorageShape();
    for (uint64_t i = 0; i < storageShape.GetDimNum(); i++) {
        storageNumel *= storageShape.GetDim(i);
    }
    storageNumel -= x->GetViewOffset();
    if (maxIdx >= storageNumel) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Gather range out of storage: maxIdx [%ld] must be less than storage numel [%ld].", maxIdx,
                storageNumel);
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* x, aclTensor* y, const aclIntArray* size, const aclIntArray* stride,
                               int64_t storageOffset)
{
    // 0. 检查芯片架构是否支持（平台不支持时其余检查无意义）
    CHECK_RET(CheckPlatform(), ACLNN_ERR_PARAM_INVALID);

    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(x, y, size, stride), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid(x, y), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查数据维度和形状是否支持
    CHECK_RET(CheckShape(y, size, stride, storageOffset), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查gather范围是否落在x可读存储内（带int64溢出保护）
    CHECK_RET(CheckGatherRange(x, size, stride, storageOffset), ACLNN_ERR_PARAM_INVALID);

    CheckFormat(x, y);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAsStridedGetWorkspaceSize(const aclTensor* x, const aclIntArray* size, const aclIntArray* stride,
                                           int64_t storageOffset, aclTensor* y, uint64_t* workspaceSize,
                                           aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnAsStrided, DFX_IN(x, size, stride, storageOffset), DFX_OUT(y));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(x, y, size, stride, storageOffset);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空tensor在kernel中支持，输出为空时无需下发任务
    if (y->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // x带view偏移时重设storage shape用于tiling边界校验：CreateView五参重载会在框架内部
    // 以 storageSize - viewOffset 构造新视图的存储形状（op_executor.cpp：aclnn下发时GM地址
    // 已含视图偏移，合法内存大小应为原始存储减偏移），此处传入x自身的storageShape即由框架完成扣除
    const aclTensor* xInput = x;
    if (x->GetViewOffset() != 0) {
        xInput = uniqueExecutor->CreateView(x, x->GetViewShape(), x->GetStorageShape(), x->GetViewStrides(),
                                            x->GetViewOffset());
        CHECK_RET(xInput != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // size/stride/storageOffset按算子输入要求转换为int64张量
    auto sizeTensor = uniqueExecutor->ConvertToTensor(size, op::ToOpDataType(ACL_INT64));
    CHECK_RET(sizeTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto strideTensor = uniqueExecutor->ConvertToTensor(stride, op::ToOpDataType(ACL_INT64));
    CHECK_RET(strideTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    int64_t offsetValue = storageOffset;
    auto storageOffsetTensor = uniqueExecutor->ConvertToTensor(&offsetValue, 1, op::ToOpDataType(ACL_INT64));
    CHECK_RET(storageOffsetTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* opResult = nullptr;
    if (op::IsContiguous(y) && !IsPrivateFormat(y->GetStorageFormat())) {
        // 输出连续时，直接由算子写入y
        opResult = l0op::AsStrided(xInput, y, sizeTensor, strideTensor, storageOffsetTensor, uniqueExecutor.get());
        CHECK_RET(opResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        // 输出非连续时，先物化到连续tensor，再由ViewCopy拷贝至y，支持任意stride的输出
        op::Shape outShape;
        for (uint64_t i = 0; i < size->Size(); i++) {
            outShape.AppendDim((*size)[i]);
        }
        auto denseOut = uniqueExecutor->AllocTensor(outShape, x->GetDataType());
        CHECK_RET(denseOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        opResult = l0op::AsStrided(xInput, denseOut, sizeTensor, strideTensor, storageOffsetTensor,
                                   uniqueExecutor.get());
        CHECK_RET(opResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto viewCopyResult = l0op::ViewCopy(opResult, y, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAsStrided(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    // 固定写法，调用框架能力，完成计算
    L2_DFX_PHASE_2(aclnnAsStrided);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
