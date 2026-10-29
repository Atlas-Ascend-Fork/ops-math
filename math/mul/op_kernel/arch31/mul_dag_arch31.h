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
 * \file mul_dag_arch31.h
 * \brief mul dag
 */

#ifndef MUL_ARCH31_DAG_H
#define MUL_ARCH31_DAG_H

#include <cstdint>

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

namespace MulDag {
using namespace AscendC;
using namespace Ops::Base;

constexpr int CAST_MODE_NONE = 0;
constexpr int CAST_MODE_RINT = 1;
constexpr std::int32_t FF = 255;

template <class T>
struct AndFF : public Vec::ElemwiseUnaryOP<T, T> {
    __aicore__ inline AndFF(const LocalTensor<T>& dst, const LocalTensor<T>& src, uint32_t count)
    {
#ifdef __CCE_AICORE__
        AscendC::Duplicate(dst, static_cast<T>(FF), count);
        AscendC::And(dst, src, dst, count);
#endif
    }
};

// Supports float and int32 without changing Mul's computation path.
template <typename T>
struct MulOp {
    using InputX1 = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using InputX2 = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;

    using Y = Bind<Vec::Mul<T>, InputX1, InputX2>;

    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, Y>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

// Keeps the arch35 FP16 promotion path.
template <typename T>
struct MulXfp16Op {
    using InputX1 = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using InputX2 = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;
    using CastX1 = Bind<Vec::Cast<float, T, CAST_MODE_NONE>, InputX1>;
    using CastX2 = Bind<Vec::Cast<float, T, CAST_MODE_NONE>, InputX2>;

    using Y = Bind<Vec::Mul<float>, CastX1, CastX2>;
    using YB16 = Bind<Vec::Cast<T, float, CAST_MODE_RINT>, Y>;

    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, YB16>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

struct MulInt8Op {
    using InputX1 = Bind<Vec::CopyInBrc<int8_t>, Placeholder::In0<int8_t>>;
    using InputX2 = Bind<Vec::CopyInBrc<int8_t>, Placeholder::In1<int8_t>>;
    using CastX1 = Bind<Vec::Cast<int32_t, int8_t, CAST_MODE_NONE>, InputX1>;
    using CastX2 = Bind<Vec::Cast<int32_t, int8_t, CAST_MODE_NONE>, InputX2>;

    using Y = Bind<Vec::Mul<int32_t>, CastX1, CastX2>;
    using Y1 = Bind<AndFF<int32_t>, Y>;
    using Y2 = Bind<Vec::Cast<uint8_t, int32_t, CAST_MODE_NONE>, Y1>;

    using OpCopyOut = Bind<Vec::CopyOut<int8_t>, Placeholder::Out0<int8_t>, Y2>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

struct MulUint8Op {
    using InputX1 = Bind<Vec::CopyInBrc<uint8_t>, Placeholder::In0<uint8_t>>;
    using InputX2 = Bind<Vec::CopyInBrc<uint8_t>, Placeholder::In1<uint8_t>>;
    using CastX1 = Bind<Vec::Cast<uint32_t, uint8_t, CAST_MODE_NONE>, InputX1>;
    using CastX2 = Bind<Vec::Cast<uint32_t, uint8_t, CAST_MODE_NONE>, InputX2>;

    using Y = Bind<Vec::Mul<uint32_t>, CastX1, CastX2>;
    using Y1 = Bind<AndFF<uint32_t>, Y>;
    using Y2 = Bind<Vec::Cast<uint8_t, uint32_t, CAST_MODE_NONE>, Y1>;

    using OpCopyOut = Bind<Vec::CopyOut<uint8_t>, Placeholder::Out0<uint8_t>, Y2>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace MulDag

#endif // MUL_ARCH31_DAG_H
