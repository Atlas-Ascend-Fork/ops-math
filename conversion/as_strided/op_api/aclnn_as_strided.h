/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_AS_STRIDED_H_
#define OP_API_INC_AS_STRIDED_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif
/**
 * @brief aclnnAsStrided的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能：按size、stride、storage_offset从输入张量x中读取元素，物化生成形状为size的输出张量y，
 * 即 y_i = x_{storage_offset + sum(i_d * stride[d])}。
 *
 * @param [in] x: npu device侧的aclTensor，数据类型支持int64, uint64, int32, uint32, float, float16,
 * int8, uint8, bfloat16, int16, uint16, bool, complex32, complex64, hifloat8, float8_e5m2, float8_e4m3fn。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] size: 输出张量的形状，int64数组，数组长度与stride一致且不超过8。
 * @param [in] stride: stride[d]为读取x时第d维的步幅，int64数组，数组元素非负。
 * @param [in] storageOffset: 读取x时的起始存储偏移，单位为元素个数，非负整数，数据类型支持INT64。
 * @param [out] y: npu device侧的aclTensor，数据类型与x一致，形状需等于size。支持非连续的Tensor，
 * 数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnAsStridedGetWorkspaceSize(const aclTensor* x, const aclIntArray* size,
                                                     const aclIntArray* stride, int64_t storageOffset, aclTensor* y,
                                                     uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnAsStrided的第二段接口，用于执行计算。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnAsStridedGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnAsStrided(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                     const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_AS_STRIDED_H_
