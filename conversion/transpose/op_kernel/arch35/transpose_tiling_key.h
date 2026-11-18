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
 * \file transpose_tiling_key.h
 * \brief Transpose 算子 TilingKey 常量定义（Kernel 侧）
 *
 */

#ifndef TRANSPOSE_TILING_KEY_H
#define TRANSPOSE_TILING_KEY_H

#define TENSOR_MOVE 10000      // 融合后仅1维，等价于纯数据搬运
#define SMALL_SHAPE 10001      // 小数据量(<阈值字节)，SIMT模式GM→GM直读直写
#define CUT_ONCE 10002         // NDDMA 5维搬运，切1个轴（输出切分轴≤输入切分轴映射）
#define CUT_TWICE 10003        // NDDMA 5维搬运，切2个轴（4种数据区间：Main/InputTail/OutputTail/Tail）
#define N_LAST_TRANSPOSE 10004 // 尾轴不转置 + 尾轴≥32元素，双缓冲流水线连续行搬移
#define BIG_DIM 10005          // 维度>5，压缩到5维NDDMA格式进行搬运
#define GATHER_TRANSPOSE 10006 // 尾轴转置+大shape，使用DataCopyGather硬件指令(RegBase路径)
#define VCONV_TRANSPOSE 10007  // 2D perm=[1,0]+16bit(DAV_5102)，TransDataTo5HD硬件指令
#define VCONV_021_TRANSPOSE 10008    // 3D perm=[0,2,1]+8/16/32bit(DAV_5102)，TransDataTo5HD硬件指令
#define B4_DUAL_CUT 10009            // B4 双切分
#define TRANSPOSE_2D_TRANSDATA 10011 // 2D perm=[1,0] 特化，TransDataTo5HD 全 tile 转置

#endif // TRANSPOSE_TILING_KEY_H
