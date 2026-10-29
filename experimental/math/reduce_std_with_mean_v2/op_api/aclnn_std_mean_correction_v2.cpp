/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_std_mean_correction_v2.h"
#include <bitset>
#include <algorithm>
#include "reduce_std_with_mean_v2.h"
#include "reduce_std_with_mean_v2_partial.h"
#include "reduce_std_with_mean_v2_merge.h"
#include "math/reduce_std_v2/op_api/reduce_std_v2.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/transpose.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/op_api_def_math.h"
#include "opdev/platform.h"
#include "opdev/op_log.h"
#include "op_api/level2_base_caculation_math.h"
#include "op_api/aclnn_check.h"

using namespace op;
using std::bitset;

#ifdef __cplusplus
extern "C" {
#endif

constexpr size_t MAX_MASK_NUM = 64;
static const bool invert = false;
static const float eps = 0.0f;
// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT,
                                                                                       op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* stdOut, const aclTensor* meanOut)
{
    // 检查self的数据类型是否在算子的支持列表内
    auto supportList = GetDtypeSupportListV2(ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST, ASCEND910_DTYPE_DTYPE_SUPPORT_LIST);
    OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);

    // 检查stdOut、meanOut的数据类型是否在算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(stdOut, supportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(meanOut, supportList, return false);

    // Ascend950支持输入、输出数据类型不一致
    if (IsRegBase()) {
        return true;
    }

    // 检查self的数据类型是否输出的数据类型一致
    OP_CHECK_DTYPE_NOT_MATCH(self, stdOut->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(self, meanOut->GetDataType(), return false);
    return true;
}

static bool CheckDimValid(const aclTensor* self, const aclIntArray* dim)
{
    auto selfViewShape = self->GetViewShape();
    auto selfDimNum = static_cast<int64_t>(selfViewShape.GetDimNum());
    int64_t dimMin = std::min(-1 * selfDimNum, selfDimNum - 1);
    int64_t dimMax = std::max(-1 * selfDimNum, selfDimNum - 1);
    // 0维tensor
    if (selfDimNum == 0) {
        selfDimNum = 1;
    }
    bool dimMask[64] = {false};
    // dim可以为空指针
    if (dim == nullptr) {
        return true;
    }
    // 获取dim元素
    for (size_t i = 0; i < dim->Size(); i++) {
        // dim值不能超出范围
        auto currentDim = dim->operator[](i);
        if (currentDim > dimMax || currentDim < dimMin) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "Dimension out of range (expected to be in range of [%ld, %ld], but got %ld)", dimMin, dimMax,
                    selfDimNum);
            return false;
        }
        // dim值可以为负
        if (currentDim < 0) {
            currentDim = currentDim + selfDimNum;
        }
        // dim值不能重复
        if (dimMask[currentDim]) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim %ld appears multiple times in the list of dims.", currentDim);
            return false;
        }
        dimMask[currentDim] = true;
    }
    return true;
}

static void OutInferShape(const op::Shape& selfShape, const aclIntArray* dim, bool keepDim, op::Shape& reduceShape)
{
    bitset<MAX_MASK_NUM> dimTmp = bitset<MAX_MASK_NUM>();
    for (size_t i = 0; i < dim->Size(); i++) {
        int64_t index = GetPosDimWithStd(dim->operator[](i), selfShape.GetDimNum());
        // 前序已检查， 此处如果dim不会重复
        dimTmp.set(index);
    }

    for (size_t i = 0; i < selfShape.GetDimNum(); i++) {
        if (!dimTmp[i]) {
            reduceShape.AppendDim(selfShape.GetDim(i));
        } else if (keepDim) {
            reduceShape.AppendDim(1);
        }
    }
}

static bool CheckShape(const aclTensor* self, const aclIntArray* dim, const bool keepDim, const aclTensor* stdOut,
                       const aclTensor* meanOut)
{
    OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(stdOut, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(meanOut, MAX_SUPPORT_DIMS_NUMS, return false);
    // dim可以为空指针
    if (dim == nullptr || dim->Size() == 0) {
        return true;
    }
    op::Shape reduceShape;
    OutInferShape(self->GetViewShape(), dim, keepDim, reduceShape);
    // stdOut的shape必须满足Infer shape
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(stdOut, reduceShape, return false);
    // meanOut的shape必须满足Infer shape
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(meanOut, reduceShape, return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclIntArray* dim, const bool keepdim,
                               const aclTensor* stdOut, const aclTensor* meanOut)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull3Tensor(self, stdOut, meanOut), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, stdOut, meanOut), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查reduce的轴是否合理
    CHECK_RET(CheckDimValid(self, dim), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查shape是否满足推导
    CHECK_RET(CheckShape(self, dim, keepdim, stdOut, meanOut), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclIntArray* CalcDim(const aclTensor* self, aclOpExecutor* executor)
{
    FVector<int64_t> dimVector;
    auto selfViewShape = self->GetViewShape();
    size_t selfDimNum = selfViewShape.GetDimNum();
    for (size_t i = 0; i < selfDimNum; i++) {
        dimVector.push_back(static_cast<int64_t>(i));
    }
    return executor->AllocIntArray(dimVector.data(), dimVector.size());
}

static const aclIntArray* GetDimArray(const aclTensor* self, const aclIntArray* dim, aclOpExecutor* executor)
{
    const aclIntArray* dimArray = dim;
    if (dim == nullptr || dim->Size() == 0) {
        dimArray = CalcDim(self, executor);
    }
    CHECK_RET(dimArray != nullptr, nullptr);
    return dimArray;
}

static int64_t CalcShapeProd(const aclTensor* self, const aclIntArray* dim)
{
    auto selfViewShape = self->GetViewShape();
    int64_t shapeProd = 1;
    size_t selfDimNum = selfViewShape.GetDimNum();
    if (selfDimNum != 0) {
        // dim为all reduce
        if (dim->Size() == 0) {
            for (size_t i = 0; i < selfDimNum; i++) {
                shapeProd *= selfViewShape.GetDim(i);
            }
        } else {
            for (size_t i = 0; i < dim->Size(); i++) {
                auto oriDim = dim->operator[](i);
                auto realDim = oriDim < 0 ? oriDim + selfDimNum : oriDim;
                shapeProd *= selfViewShape.GetDim(realDim);
            }
        }
    }
    return shapeProd;
}

static aclnnStatus DealEmpty(aclTensor* stdOut, aclTensor* meanOut, aclOpExecutor* executor)
{
    auto ret = CheckFillScalarShapeStdAndVar(stdOut, NAN, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = CheckFillScalarShapeStdAndVar(meanOut, NAN, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    return ACLNN_SUCCESS;
}

static aclnnStatus DealshapeProdNAN(const aclTensor* meanOpOut, aclTensor* stdOut, aclTensor* meanOut,
                                    aclOpExecutor* executor)
{
    // stdOut返回NAN
    auto ret = CheckFillScalarShapeStdAndVar(stdOut, NAN, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    // 固定写法，将计算结果拷贝到输出meanOut上，meanOut可能是非连续的tensor
    auto viewCopyMeanResult = l0op::ViewCopy(meanOpOut, meanOut, executor);
    CHECK_RET(viewCopyMeanResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus DealshapeProdINF(const aclTensor* meanOpOut, aclTensor* stdOut, aclTensor* meanOut,
                                    aclOpExecutor* executor)
{
    // stdOut返回INF
    auto ret = CheckFillScalarShapeStdAndVar(stdOut, INFINITY, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    // 固定写法，将计算结果拷贝到输出meanOut上，meanOut可能是非连续的tensor
    auto viewCopyMeanResult = l0op::ViewCopy(meanOpOut, meanOut, executor);
    CHECK_RET(viewCopyMeanResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus aclnnStdMeanCorrectionV2ImplUnify(const aclTensor* self, const aclIntArray* dim, int64_t correction,
                                                     bool keepdim, aclTensor* stdOut, aclTensor* meanOut,
                                                     uint64_t* workspaceSize, UniqueExecutor& uniqueExecutor,
                                                     aclOpExecutor** executor)
{
    bool isMeanOut = true;
    auto reduceStdOut = l0op::ReduceStdV2(self, dim, correction, keepdim, isMeanOut, uniqueExecutor.get());

    auto stdOpOut = std::get<0>(reduceStdOut);
    CHECK_RET(stdOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto castOut = l0op::Cast(stdOpOut, stdOut->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult = l0op::ViewCopy(castOut, stdOut, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto meanOpOut = std::get<1>(reduceStdOut);
    CHECK_RET(meanOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto castOut1 = l0op::Cast(meanOpOut, meanOut->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut1 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult1 = l0op::ViewCopy(castOut1, meanOut, uniqueExecutor.get());
    CHECK_RET(viewCopyResult1 != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnStdMeanCorrectionV2GetWorkspaceSize(const aclTensor* self, const aclIntArray* dim, int64_t correction,
                                                     bool keepdim, aclTensor* stdOut, aclTensor* meanOut,
                                                     uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnStdMeanCorrectionV2, DFX_IN(self, dim, correction, keepdim), DFX_OUT(stdOut, meanOut));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, dim, keepdim, stdOut, meanOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // dim为空指针
    auto dimArray = GetDimArray(self, dim, uniqueExecutor.get());
    CHECK_RET(dimArray != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 空tensor在kernel中支持
    if (self->IsEmpty()) {
        // 空tensor填充NAN
        ret = DealEmpty(stdOut, meanOut, uniqueExecutor.get());
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ret;
    }

    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto selfReformat = l0op::ReFormat(selfContiguous, Format::FORMAT_ND);
    CHECK_RET(selfReformat != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (IsRegBase()) {
        return aclnnStdMeanCorrectionV2ImplUnify(selfReformat, dimArray, correction, keepdim, stdOut, meanOut,
                                                 workspaceSize, uniqueExecutor, executor);
    }

    // Transpose: move reduce dims to innermost for contiguous kernel access
    const aclTensor* selfForKernel = selfContiguous;
    const aclIntArray* dimForKernel = dimArray;
    bool didTranspose = false;
    FVector<int64_t> invPerm; // inverse perm for restoring output shape
    {
        auto selfViewShape = selfContiguous->GetViewShape();
        int64_t rank = static_cast<int64_t>(selfViewShape.GetDimNum());
        if (rank > 1) {
            bool isReduce[64] = {false};
            int64_t numReduceDims = 0;
            for (size_t i = 0; i < dimArray->Size(); i++) {
                int64_t d = (*dimArray)[i];
                if (d < 0)
                    d += rank;
                if (d >= 0 && d < rank && !isReduce[d]) {
                    isReduce[d] = true;
                    numReduceDims++;
                }
            }
            if (numReduceDims > 0 && numReduceDims < rank) {
                bool needTranspose = false;
                for (int64_t d = 0; d < rank - numReduceDims; d++) {
                    if (isReduce[d]) {
                        needTranspose = true;
                        break;
                    }
                }
                if (needTranspose) {
                    didTranspose = true;
                    FVector<int64_t> perm;
                    for (int64_t d = 0; d < rank; d++)
                        if (!isReduce[d])
                            perm.push_back(d);
                    for (int64_t d = 0; d < rank; d++)
                        if (isReduce[d])
                            perm.push_back(d);
                    // Compute inverse perm
                    invPerm.resize(rank);
                    for (int64_t i = 0; i < rank; i++)
                        invPerm[perm[i]] = i;
                    auto permArray = uniqueExecutor->AllocIntArray(perm.data(), rank);
                    CHECK_RET(permArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
                    selfForKernel = l0op::Transpose(selfForKernel, permArray, uniqueExecutor.get());
                    CHECK_RET(selfForKernel != nullptr, ACLNN_ERR_INNER_NULLPTR);
                    selfForKernel = l0op::Contiguous(selfForKernel, uniqueExecutor.get());
                    CHECK_RET(selfForKernel != nullptr, ACLNN_ERR_INNER_NULLPTR);
                    FVector<int64_t> newDimVec;
                    for (int64_t s = 0; s < numReduceDims; s++)
                        newDimVec.push_back(rank - numReduceDims + s);
                    dimForKernel = uniqueExecutor->AllocIntArray(newDimVec.data(), numReduceDims);
                    CHECK_RET(dimForKernel != nullptr, ACLNN_ERR_INNER_NULLPTR);
                }
            }
        }
    }

    // shapeProd小于等于correction场景：std填NAN/INF，mean仍由kernel正常输出
    int64_t shapeProd = 1;
    shapeProd = CalcShapeProd(self, dimArray);
    if ((shapeProd == 1) && (shapeProd <= correction)) {
        auto reduceOut = l0op::ReduceStdWithMeanV2(selfForKernel, dimForKernel, correction, keepdim, invert, eps,
                                                   uniqueExecutor.get());
        CHECK_RET(std::get<0>(reduceOut) != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(std::get<1>(reduceOut) != nullptr, ACLNN_ERR_INNER_NULLPTR);
        const aclTensor* meanForOutput = std::get<1>(reduceOut);
        if (didTranspose && keepdim) {
            auto invPermArray = uniqueExecutor->AllocIntArray(invPerm.data(), invPerm.size());
            CHECK_RET(invPermArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
            meanForOutput = l0op::Transpose(meanForOutput, invPermArray, uniqueExecutor.get());
            CHECK_RET(meanForOutput != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
        ret = DealshapeProdNAN(meanForOutput, stdOut, meanOut, uniqueExecutor.get());
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    if ((correction > 1) && (shapeProd <= correction)) {
        auto reduceOut = l0op::ReduceStdWithMeanV2(selfForKernel, dimForKernel, correction, keepdim, invert, eps,
                                                   uniqueExecutor.get());
        CHECK_RET(std::get<0>(reduceOut) != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(std::get<1>(reduceOut) != nullptr, ACLNN_ERR_INNER_NULLPTR);
        const aclTensor* meanForOutput = std::get<1>(reduceOut);
        if (didTranspose && keepdim) {
            auto invPermArray = uniqueExecutor->AllocIntArray(invPerm.data(), invPerm.size());
            CHECK_RET(invPermArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
            meanForOutput = l0op::Transpose(meanForOutput, invPermArray, uniqueExecutor.get());
            CHECK_RET(meanForOutput != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
        ret = DealshapeProdINF(meanForOutput, stdOut, meanOut, uniqueExecutor.get());
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // two-stage cross-core reduction : when the
    // non-reduce count is small (M <= 64, cannot fill the cores) and R is
    // large (> 2048, 32B-aligned rows), R is split across K cores
    // (stage 1 = ReduceStdWithMeanV2Partial) and the per-segment
    // (n, mean, m2) triples are Chan-merged (stage 2 = ReduceStdWithMeanV2Merge).
    // Math is identical to the chunked Welford (exact partition), so precision
    // is unchanged. The non-two-stage path below is untouched.
    bool twoStageEnabled = false;
    int64_t twoStageK = 0;
    {
        auto kShape = selfForKernel->GetViewShape();
        int64_t kRank = static_cast<int64_t>(kShape.GetDimNum());
        int64_t reduceLen = 1;
        for (size_t i = 0; i < dimForKernel->Size(); i++) {
            int64_t d = (*dimForKernel)[i];
            if (d < 0)
                d += kRank;
            if (d >= 0 && d < kRank)
                reduceLen *= kShape.GetDim(d);
        }
        int64_t total = 1;
        for (int64_t d = 0; d < kRank; d++)
            total *= kShape.GetDim(d);
        int64_t nonReduce = (reduceLen > 0) ? (total / reduceLen) : 0;
        int64_t typeSize = (selfReformat->GetDataType() == op::DataType::DT_FLOAT) ? 4 : 2;
        if (nonReduce <= 64 && reduceLen > 2048 && (reduceLen * typeSize) % 32 == 0) {
            int64_t coreNum = 40;
            const op::PlatformInfo& platformInfo = op::GetCurrentPlatformInfo();
            if (platformInfo.Valid()) {
                int64_t vc = static_cast<int64_t>(platformInfo.GetVectorCoreNum());
                if (vc > 0) {
                    coreNum = vc;
                }
            }
            int64_t k = std::min(coreNum, (reduceLen + 2047) / 2048);
            if (k >= 2) {
                twoStageEnabled = true;
                twoStageK = k;
            }
        }
    }
    if (twoStageEnabled) {
        auto partial = l0op::ReduceStdWithMeanV2Partial(selfForKernel, dimForKernel, twoStageK, correction, keepdim,
                                                        invert, eps, uniqueExecutor.get());
        CHECK_RET(partial != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto reduceOut = l0op::ReduceStdWithMeanV2Merge(partial, dimForKernel, twoStageK, correction, keepdim, invert,
                                                        eps, uniqueExecutor.get());
        CHECK_RET(std::get<0>(reduceOut) != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(std::get<1>(reduceOut) != nullptr, ACLNN_ERR_INNER_NULLPTR);
        // The Merge kernel outputs are flat [M] rows in the user's dim-major
        // order; fix the view shape to the exact user-visible shape (keepdim
        // positions restored) so the ViewCopy below is shape-exact.
        const aclTensor* stdForOutput = std::get<0>(reduceOut);
        const aclTensor* meanForOutput = std::get<1>(reduceOut);
        // The Merge kernel emits float32; cast to the original input dtype.
        auto dtype = selfReformat->GetDataType();
        if (dtype != op::DataType::DT_FLOAT) {
            stdForOutput = l0op::Cast(stdForOutput, dtype, uniqueExecutor.get());
            CHECK_RET(stdForOutput != nullptr, ACLNN_ERR_INNER_NULLPTR);
            meanForOutput = l0op::Cast(meanForOutput, dtype, uniqueExecutor.get());
            CHECK_RET(meanForOutput != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
        // Fix the view shape to the exact user-visible shape (keepdim positions
        // restored) so the ViewCopy below is shape-exact.
        op::Shape reduceShape;
        OutInferShape(self->GetViewShape(), dimArray, keepdim, reduceShape);
        const_cast<aclTensor*>(stdForOutput)->SetViewShape(reduceShape);
        const_cast<aclTensor*>(meanForOutput)->SetViewShape(reduceShape);
        auto viewCopyStdResult = l0op::ViewCopy(stdForOutput, stdOut, uniqueExecutor.get());
        CHECK_RET(viewCopyStdResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto viewCopyMeanResult = l0op::ViewCopy(meanForOutput, meanOut, uniqueExecutor.get());
        CHECK_RET(viewCopyMeanResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // Welford kernel: single pass over self, outputs (std, mean) in transposed layout
    auto reduceOut = l0op::ReduceStdWithMeanV2(selfForKernel, dimForKernel, correction, keepdim, invert, eps,
                                               uniqueExecutor.get());
    CHECK_RET(std::get<0>(reduceOut) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(std::get<1>(reduceOut) != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // Inverse transpose output: after kernel on transposed data, keepdim=True cases
    // have size-1 dims at wrong positions (e.g. [7,1] instead of [1,7])
    const aclTensor* stdForOutput = std::get<0>(reduceOut);
    const aclTensor* meanForOutput = std::get<1>(reduceOut);
    if (didTranspose && keepdim) {
        auto invPermArray = uniqueExecutor->AllocIntArray(invPerm.data(), invPerm.size());
        CHECK_RET(invPermArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
        stdForOutput = l0op::Transpose(stdForOutput, invPermArray, uniqueExecutor.get());
        CHECK_RET(stdForOutput != nullptr, ACLNN_ERR_INNER_NULLPTR);
        meanForOutput = l0op::Transpose(meanForOutput, invPermArray, uniqueExecutor.get());
        CHECK_RET(meanForOutput != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    // Kernel output is already in the input dtype (no fp32 Cast needed).
    auto viewCopyStdResult = l0op::ViewCopy(stdForOutput, stdOut, uniqueExecutor.get());
    CHECK_RET(viewCopyStdResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // Mean output from Welford kernel (same dtype, no fp32 Cast needed).
    auto viewCopyMeanResult = l0op::ViewCopy(meanForOutput, meanOut, uniqueExecutor.get());
    CHECK_RET(viewCopyMeanResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnStdMeanCorrectionV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                     const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnStdMeanCorrectionV2);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
