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
 * \file transpose_tiling_base.h
 * \brief Transpose Tiling 基础数据结构和轴变换函数声明
 *
 * 本文件定义了 Tiling 计算所需的基础数据结构：
 * - ShapeInfo：shape 信息结构体，包含原始/简化后的 shape、perm、维度数等
 * - TransposeCompilerInfo：编译信息结构体，包含核数和 UB 大小
 *
 * 以及两个核心轴变换函数：
 * - RemoveAxisV2：消除 size=1 的轴（减少不必要的维度）
 * - MergeAxisV2：合并 perm 中连续的轴（降低维度数）
 *
 * 这两个函数是 Tiling 计算的前置步骤，先 Remove 再 Merge，
 * 可以有效减少维度数，使更多场景落入 NDDMA 5维搬运路径。
 */
#ifndef __TRANSPOSE_RT_V2_H__
#define __TRANSPOSE_RT_V2_H__

#include <array>
#include <vector>
#include <string>
#include <map>
#include <queue>
#include <memory>

#include "../../op_kernel/arch35/transpose_tiling_data.h"

namespace optiling {
using namespace Transpose;

/**
 * @brief Shape 信息结构体，Tiling 计算的核心数据
 *
 * 包含原始和简化后的 shape、perm 信息，以及轴变换过程中的中间状态。
 */
struct ShapeInfo {
    std::vector<int64_t> inShape;         ///< 原始输入 shape（最多8维）
    std::vector<int64_t> outShape;        ///< 原始输出 shape
    std::vector<int64_t> perm;            ///< 原始 perm 数组
    std::vector<int64_t> reducedInShape;  ///< 简化后的输入 shape（RemoveAxis+MergeAxis 后）
    std::vector<int64_t> reducedOutShape; ///< 简化后的输出 shape
    std::vector<int64_t> reducedPerm;     ///< 简化后的 perm 数组

    int64_t inShapeSize;  ///< 输入维度数
    int64_t outShapeSize; ///< 输出维度数
    int64_t permSize;     ///< perm 数组长度

    int64_t origDim;           ///< 原始维度数
    int64_t dim;               ///< 简化后的维度数
    int64_t totalVolumeActual; ///< 总元素数（简化后 shape 的乘积）
    int64_t eleLenInBytes;     ///< 每个元素的字节数
    bool isLastAxisTranspose;  ///< 最后一维是否参与转置

    ShapeInfo()
    {
        inShape.resize(TP_MAX_AXIS_NUM);
        outShape.resize(TP_MAX_AXIS_NUM);
        perm.resize(TP_MAX_AXIS_NUM);
        reducedInShape.resize(TP_MAX_AXIS_NUM);
        reducedOutShape.resize(TP_MAX_AXIS_NUM);
        reducedPerm.resize(TP_MAX_AXIS_NUM);
        Reset();
    }

    void Reset()
    {
        inShapeSize = 0;
        outShapeSize = 0;
        permSize = 0;
        origDim = 0;
        dim = 0;
        totalVolumeActual = 0;
        eleLenInBytes = 0;
        isLastAxisTranspose = false;
    }
};

/**
 * @brief 编译信息结构体，从平台信息中获取
 */
struct TransposeCompilerInfo {
    int64_t coreNum; ///< AI Vector 核数
    int64_t ubSize;  ///< UB 大小（单位：Block）

    TransposeCompilerInfo() : coreNum(0), ubSize(0) {}
};

/**
 * @brief 消除 size=1 的轴
 *
 * 遍历输入 shape，将大小为1的轴从 inShape 和 perm 中移除，
 * 并调整 perm 中其余轴的索引值。这可以减少维度数，使更多场景
 * 落入 NDDMA 5维搬运路径。
 *
 * 例如：shape=[2,1,3], perm=[2,0,1] → reducedShape=[2,3], reducedPerm=[1,0]
 *
 * @param shapeInfo [in,out] Shape 信息结构体
 */
void RemoveAxisV2(ShapeInfo& shapeInfo);

/**
 * @brief 合并 perm 中连续的轴
 *
 * 如果 perm 中相邻的两个索引是连续数字（如 perm=[0,1,2]），
 * 则将对应的维度合并为一个维度（shape 相乘）。
 * 这可以进一步减少维度数。
 *
 * 例如：shape=[2,3,4], perm=[0,1,2] → reducedShape=[24], reducedPerm=[0]
 *       shape=[2,3,4], perm=[1,2,0] → reducedShape=[2,12], reducedPerm=[1,0]
 *
 * @param shapeInfo [in,out] Shape 信息结构体
 */
void MergeAxisV2(ShapeInfo& shapeInfo);

} // namespace optiling

#endif // __TRANSPOSE_RT_V2_H__
