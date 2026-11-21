/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file transpose_tiling_base.cpp
 * \brief Transpose Tiling 基础轴变换函数实现
 *
 * 本文件实现 RemoveAxisV2 和 MergeAxisV2 两个核心轴变换函数，
 * 它们是 Tiling 策略选择的前置步骤。
 *
 * RemoveAxisV2 算法：
 *   1. 压缩非单例轴，记录原始轴到新轴编号的映射
 *   2. 如果所有轴都是1，直接降为1维
 *   3. 遍历 perm，跳过单例轴，按映射生成 reducedPerm 和 reducedOutShape
 *
 * MergeAxisV2 算法：
 *   1. 按 perm 顺序遍历，检查相邻两个 perm 值是否连续
 *   2. 连续的轴合并为一个轴（shape 相乘）
 *   3. 重新排列合并后的 perm 和 shape
 *   4. 计算新的 reducedInShape 和 reducedOutShape
 */
#include "transpose_tiling_base.h"

#include <vector>
#include <algorithm>

using namespace std;

namespace optiling {
using namespace Transpose;

static void DuplicateArray(const vector<int64_t>& src, vector<int64_t>& dst, int64_t len)
{
    for (int64_t i = 0; i < len; i++) {
        dst[i] = src[i];
    }
}

static void CalcOutShape(ShapeInfo& shapeInfo)
{
    const vector<int64_t>& inShape = shapeInfo.reducedInShape;
    const vector<int64_t>& perm = shapeInfo.reducedPerm;
    vector<int64_t>& outShape = shapeInfo.reducedOutShape;
    for (int64_t i = 0; i < shapeInfo.dim; i++) {
        outShape[i] = inShape[perm[i]];
    }
}

void RemoveAxisV2(ShapeInfo& shapeInfo)
{
    int64_t dim = shapeInfo.dim;
    if (dim == 1) {
        DuplicateArray(shapeInfo.inShape, shapeInfo.reducedInShape, dim);
        DuplicateArray(shapeInfo.perm, shapeInfo.reducedPerm, dim);
        DuplicateArray(shapeInfo.outShape, shapeInfo.reducedOutShape, dim);
        return;
    }

    // 建立原始轴到删除单例维度后的新轴编号的映射。
    int64_t axisMap[TP_MAX_AXIS_NUM];
    int64_t reducedDim = 0;
    for (int64_t i = 0; i < dim; i++) {
        if (shapeInfo.inShape[i] == 1) {
            axisMap[i] = -1;
        } else {
            axisMap[i] = reducedDim;
            shapeInfo.reducedInShape[reducedDim++] = shapeInfo.inShape[i];
        }
    }

    if (reducedDim == 0) {
        shapeInfo.reducedInShape[0] = 1;
        shapeInfo.reducedPerm[0] = 0;
        shapeInfo.reducedOutShape[0] = 1;
        shapeInfo.dim = 1;
        return;
    }

    int64_t outAxis = 0;
    for (int64_t i = 0; i < dim; i++) {
        const int64_t originalAxis = shapeInfo.perm[i];
        const int64_t reducedAxis = axisMap[originalAxis];
        if (reducedAxis >= 0) {
            shapeInfo.reducedPerm[outAxis] = reducedAxis;
            shapeInfo.reducedOutShape[outAxis++] = shapeInfo.inShape[originalAxis];
        }
    }
    shapeInfo.dim = reducedDim;
}

void MergeAxisV2(ShapeInfo& shapeInfo)
{
    const int64_t dim = shapeInfo.dim;
    if (dim == 1) {
        return;
    }

    // 按输出轴顺序识别连续轴组，并在每组的输入起始轴处记录组编号。
    int64_t groupAtHead[TP_MAX_AXIS_NUM];
    std::fill_n(groupAtHead, dim, -1);
    int64_t mergedShape[TP_MAX_AXIS_NUM];
    int64_t groupCount = 0;
    for (int64_t i = 0; i < dim; ++i) {
        const int64_t axis = shapeInfo.reducedPerm[i];
        if (i > 0 && axis == shapeInfo.reducedPerm[i - 1] + 1) {
            mergedShape[groupCount - 1] *= shapeInfo.reducedInShape[axis];
        } else {
            groupAtHead[axis] = groupCount;
            mergedShape[groupCount++] = shapeInfo.reducedInShape[axis];
        }
    }

    // 按输入轴顺序遍历，直接建立输出轴组到新输入轴编号的映射。
    int64_t inputAxis = 0;
    for (int64_t axis = 0; axis < dim; ++axis) {
        const int64_t group = groupAtHead[axis];
        if (group >= 0) {
            shapeInfo.reducedInShape[inputAxis] = mergedShape[group];
            shapeInfo.reducedPerm[group] = inputAxis++;
        }
    }
    shapeInfo.dim = groupCount;
    CalcOutShape(shapeInfo);
}

} // namespace optiling
