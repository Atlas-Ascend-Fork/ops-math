/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ACLNN_SELECT_TOP_K_H_
#define ACLNN_SELECT_TOP_K_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    ACLNN_SELECT_TOP_K_ORDER_NONE = 0,  /* 不保证有效结果的排列顺序。 */
    ACLNN_SELECT_TOP_K_ORDER_VALUE = 1, /* 有效结果按值排序。 */
    ACLNN_SELECT_TOP_K_ORDER_INDEX = 2  /* 索引排序预留值，当前不支持。 */
} aclnnSelectTopKOrder;

typedef enum {
    ACLNN_SELECT_TOP_K_OUTPUT_VALUES = 0,
    ACLNN_SELECT_TOP_K_OUTPUT_INDICES = 1,
    ACLNN_SELECT_TOP_K_OUTPUT_BOTH = 2
} aclnnSelectTopKOutputMode;

/**
 * @brief aclnnSelectTopK 的第一阶段接口，构建 A2/A3 TopK 执行计划并查询工作区大小。
 *
 * 记输入维数为 rank，排序轴长度为 N，其余各维长度的乘积为 rows。
 * 每行在裁剪到 [0, N] 后的 [begin, end) 区间内选取极值，end <= begin 时该行区间为空。
 * 输出形状为 self 的形状将 dim 维长度替换为 k；不足 k 个有效元素时，在尾部填充。
 * 本阶段记录调用者预分配的输出张量，实际数据由第二阶段写入。
 *
 * @param[in] self 输入张量，不可为 nullptr。支持公开格式和非连续张量，类型为
 *                ACL_FLOAT16、ACL_BF16 或 ACL_FLOAT；rank 为 1～8，N 不超过 INT32_MAX。
 * @param[in] k 每行请求的输出元素数，范围为 [0, INT32_MAX]；允许大于有效区间长度或 N。
 * @param[in] dim 排序轴，范围为 [-rank, rank - 1]；负数表示从末维向前计数。
 * @param[in] largest 为 true 时选取最大值，为 false 时选取最小值。
 * @param[in] indicesType 输出索引类型，仅接受 ACL_INT32 或 ACL_INT64；仅输出值时也须传入合法类型。
 * @param[in] order 有效结果的排列方式。0（ACLNN_SELECT_TOP_K_ORDER_NONE）不排序；
 *                 1（ACLNN_SELECT_TOP_K_ORDER_VALUE）按值排序，largest 为 true 时降序，否则升序；
 *                 2（ACLNN_SELECT_TOP_K_ORDER_INDEX）为索引排序预留值，当前不支持。
 * @param[in] indicesOffset 仅加到有效元素的原排序轴索引上的偏移量；不使用区间内的相对索引，
 *                         也不加到 indexFill 上。偏移后完整原轴的索引范围须能用 indicesType 表示。
 * @param[in] outputMode 输出选择：ACLNN_SELECT_TOP_K_OUTPUT_VALUES 仅输出值，
 *                      ACLNN_SELECT_TOP_K_OUTPUT_INDICES 仅输出索引，ACLNN_SELECT_TOP_K_OUTPUT_BOTH 同时输出两者。
 * @param[in] beginOptional 每行区间的起点（包含），为公开格式的 ACL_INT32 设备张量，允许非连续。
 *                         元素数为 rows 时按除 dim 外各维的原顺序逐行对应，元素数为 1 时广播到全部行；
 *                         nullptr 表示所有行从 0 开始。端点按原轴坐标解释，负值截断为 0。
 * @param[in] endOptional 每行区间的终点（不包含），类型、格式及逐行对应或广播规则与 beginOptional 相同；
 *                       nullptr 表示所有行到 N 结束。端点按原轴坐标解释，负值截断为 0。
 * @param[in] valueFill 有效元素不足 k 时的值填充标量；nullptr 表示填零。
 *                     标量须可转换为 ACL_FLOAT，实现先转为 float，再转换为 self 的类型写入。
 * @param[in] indexFill 有效元素不足 k 时的索引填充值，须能用 indicesType 表示；原样写入，不叠加偏移。
 * @param[out] valuesOutOptional 调用者预分配的值输出张量，须为公开格式、类型与 self 相同且符合上述输出形状。
 *                               输出值时不可为 nullptr；仅输出索引时必须为 nullptr。
 * @param[out] indicesOutOptional 调用者预分配的索引输出张量，须为公开格式、类型为 indicesType 且符合上述输出形状。
 *                                输出索引时不可为 nullptr；仅输出值时必须为 nullptr。
 * @param[out] workspaceSize 返回第二阶段所需的设备工作区字节数，指针不可为 nullptr。
 * @param[out] executor 返回包含完整计算流程的执行器，指针不可为 nullptr；交给 aclnnSelectTopK 执行。
 * @return 成功返回 ACLNN_SUCCESS，否则返回对应的错误码。
 */
ACLNN_API aclnnStatus aclnnSelectTopKGetWorkspaceSize(const aclTensor* self, int64_t k, int64_t dim, bool largest,
                                                      aclDataType indicesType, int64_t order, int64_t indicesOffset,
                                                      int64_t outputMode, const aclTensor* beginOptional,
                                                      const aclTensor* endOptional, const aclScalar* valueFill,
                                                      int64_t indexFill, aclTensor* valuesOutOptional,
                                                      aclTensor* indicesOutOptional, uint64_t* workspaceSize,
                                                      aclOpExecutor** executor);

/**
 * @brief aclnnSelectTopK 的第二阶段接口，在指定流上执行完整 TopK 计算。
 *
 * @param[in] workspace 调用者按第一阶段查询结果分配的设备工作区；相关设备内存须保持有效直到流执行完成。
 * @param[in] workspaceSize 工作区大小，单位为字节，使用 aclnnSelectTopKGetWorkspaceSize 返回的大小。
 * @param[in] executor aclnnSelectTopKGetWorkspaceSize 返回的执行器，包含输入、输出和计算流程信息。
 * @param[in] stream 执行计算的设备流；调用者在读取输出数据前须等待该流上的计算完成。
 * @return 成功返回 ACLNN_SUCCESS，否则返回对应的错误码。
 */
ACLNN_API aclnnStatus aclnnSelectTopK(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                      aclrtStream stream);

#ifdef __cplusplus
}
#endif
#endif
