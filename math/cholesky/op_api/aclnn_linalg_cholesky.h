/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LINALG_CHOLESKY_H_
#define OP_API_INC_LINALG_CHOLESKY_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ACLNN_MAX_SHAPE_RANK 8

/**
 * @brief aclnnLinalgCholesky的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能：计算一个或多个实数对称正定矩阵的Cholesky分解。
 * 计算公式：upper取值不同，分解出的三角矩阵与对应的重构关系也不同。
 * $$ upper = false: self = out \times out^{H}\ (out为下三角矩阵) $$
 * $$ upper = true:  self = out^{H} \times out\ (out为上三角矩阵) $$
 * 本接口仅支持实数输入，out^{H}即out的转置。
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 *     A[(self)] -->B([l0op::Contiguous])
 *     B --> C([l0op::Cast])
 *     C --> D([l0op::Cholesky])
 *     D --> E([l0op::Cast])
 *     E --> F([l0op::ViewCopy])
 *     F --> G[(out)]
 *     H[(upper)] --> D
 * ```
 * 其中第一个l0op::Cast仅在self的数据类型为BFLOAT16或FLOAT16时执行，将其提升至FLOAT32后再进入
 * l0op::Cholesky；FLOAT输入时该步骤跳过。第二个l0op::Cast将计算结果转换回out的数据类型。
 * AI Core算子层（OpDef、Tiling、Kernel）原生仅支持FLOAT，半精度由上述Cast实现，非Kernel原生能力。
 * self为空Tensor时不下发上述计算流程，接口直接返回成功且workspaceSize为0。
 *
 * @param [in] self: npu device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BFLOAT16。支持非连续的Tensor，
 * 数据格式支持ND。shape为[..., M, M]，最后两维必须相等，M的取值范围为[0, 8192]，维度范围为[2, 8]。
 * @param [in] upper: npu host侧的布尔型，True表示分解为上三角矩阵，False表示分解为下三角矩阵。
 * 对应算子的可选属性upper（OpDef中声明为OPTIONAL，默认值false），aclnn接口调用时需显式传入。
 * @param [in] out: npu device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BFLOAT16，且需与self一致。
 * 数据格式支持ND，shape需与self一致。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 *         ACLNN_ERR_PARAM_NULLPTR(161001)：self、out、workspaceSize、executor中存在空指针。
 *         ACLNN_ERR_PARAM_INVALID(161002)：数据类型/数据格式/shape不符合上述约束，或self的最后两维不相等，
 *         或self最后一维超过8192。
 */
ACLNN_API aclnnStatus aclnnLinalgCholeskyGetWorkspaceSize(const aclTensor* self, bool upper, aclTensor* out,
                                                          uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnLinalgCholesky的第二段接口，用于执行计算。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnLinalgCholeskyGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnLinalgCholesky(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                          const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LINALG_CHOLESKY_H_
