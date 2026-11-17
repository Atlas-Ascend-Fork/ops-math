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
 * \file radix_top_k_struct.h
 * \brief Radix TopK TilingData 结构体定义
 */

#ifndef RADIX_TOPK_STRUCT_H
#define RADIX_TOPK_STRUCT_H

#pragma pack(push, 8)
/**
 * @brief Radix TopK Tiling 数据结构
 *        包含数据切分、核分配、tile 参数等编译期确定的常量信息
 */
struct RadixTopKTilingData {
    uint64_t coreNum;       /**< 总 core 数 */
    uint64_t formerCoreNum; /**< 主核数量（处理 formerTileNum 个 tile） */
    uint64_t tailCoreNum;   /**< 尾核数量（处理 tailTileNum 个 tile） */
    uint64_t totalTileNum;  /**< 总 tile 数 */
    uint64_t formerTileNum; /**< 主核 tile 数 */
    uint64_t tailTileNum;   /**< 尾核 tile 数 */
    uint64_t formerTileLen; /**< 每个 tile 的数据长度（主核/尾核常规tile） */
    uint64_t tailTileLen;   /**< 最后一个 tail tile 的数据长度 */
    uint64_t batch;         /**< batch 数（除最后一维外各维度乘积） */
    uint64_t sortLen;       /**< 排序轴长度（输入最后一维长度） */
    uint64_t kValue;        /**< topK 值 */

    bool largest;       /**< 是否求最大 k 值（true: largest, false: smallest） */
    bool sorted;        /**< 输出结果是否排序（当前仅支持 false） */
    bool needWorkspace; /**< 是否使用 workspace（false: 复用 indices 内存做暂存） */

    // SelectTopK 专用字段组（追加在末尾，现网 radix 字段偏移不变；host 与 kernel 共用）
    uint32_t stkKLevel;            /**< k 档位：0→512 / 1→1024 / 2→4096（由 kValue 向上取档） */
    uint64_t stkBlockSize;         /**< 扫描块大小 B（host 按 UB 预算选定） */
    uint64_t stkCandidateCapacity; /**< 候选缓冲容量 = k + B + filtered + 256 */
    uint64_t stkFilteredCapacity;  /**< 阈值命中上限（候选水位余量） */
    uint32_t stkNumStages;         /**< 输入 staging 流水级数（host 选定） */
    uint64_t stkSafeTrigger;       /**< 候选压缩触发水位 = k + filtered */
    uint64_t stkWarmupLen;         /**< 预热段长度（初始化阈值），≤ B 量级 */
    uint32_t stkCoreMode;          /**< 核映射模式：0 = 行-核映射；1 = 行内多核 */
    uint64_t stkRowsPerCore;       /**< 行-核映射：前核每核行数 = CeilDiv(batch, blockDim) */
    uint64_t stkFormerCoreNum;     /**< 行-核映射：处理 stkRowsPerCore 行的核数 */
    uint64_t stkTailCoreNum;       /**< 行-核映射：处理 stkRowsPerCore-1 行的核数 */
    uint64_t stkSegLen;            /**< 段长 = CeilDiv(sortLen, blockDim)，行内多核与余数行段化复用 */
    uint64_t stkFormerSegCoreNum;  /**< 段切分：处理 stkSegLen 段的核数 */
    uint64_t stkTailSegCoreNum;    /**< 段切分：处理 stkSegLen-1 段的核数 */
    uint64_t stkFullRows; /**< 余数行段化：整行分配的完整行数（余数行起始 = q*blockDim） */
    uint64_t stkResidualRows; /**< 余数行段化：需块内多核协作的余数行数 = batch % blockDim；0 = 纯整行 */
    uint32_t stkMergeArity; /**< 槽位归并叉数：0 = 核 0 串行合并（k≤128）；4 = 四路（k≤2048）；2 = 二叉（k>2048） */
};

#pragma pack(pop)

#endif
