/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SPARSE_BINCOUNT_STRUCT_H
#define SPARSE_BINCOUNT_STRUCT_H

#include <cstdint>

struct SparseBincountPrecisePartial {
    float high;
    float low;
    uint32_t flags;
    uint32_t reserved;
};

struct SparseBincountDeterministicPartial {
    uint64_t words[8];
};

struct SparseBincountExactAccumulator {
    uint64_t limb0;
    uint64_t limb1;
    uint64_t limb2;
    uint64_t limb3;
    uint64_t limb4;
    uint64_t limb5;
    bool hasNan;
    bool hasPositiveInf;
    bool hasNegativeInf;
};

#endif // SPARSE_BINCOUNT_STRUCT_H
