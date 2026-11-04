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
 * \file sinkhorn_exp.inc
 * \brief
 */
#ifndef SINKHORN_KERNEL_EXP_INC_
#define SINKHORN_KERNEL_EXP_INC_

namespace AscendC {
    template<typename T, typename IT>
    __aicore__ inline void KernelSinkhorn<T, IT>::ExpCost()
    {
        for (int i = 0; i < tileNum; i++) {
            uint32_t ind = i * tileLength;
            uint32_t length = tileLength;
            if (i == tileNum - 1) {
                length = lastTileLength;
            }
            CopyInForExp(ind, length);
            ComputeForExp(length);
            CopyOutForExp<IT>(ind, length);
            // costOutQueue 跨 tile 复用，补 MTE3->MTE2 排空（批准前移项）
            event_t eventIdMte3Mte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3Mte2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3Mte2);
        }
    }

    // 通用float/half
    template<typename T, typename IT>
    __aicore__ inline void KernelSinkhorn<T, IT>::CopyInForExp(uint32_t ind, uint32_t length)
    {
        LocalTensor<T> costLocal = costInQueue.AllocTensor<T>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(costLocal, costGlobal[ind], copyParams, padParams);
        costInQueue.EnQue(costLocal);
    }

    template<typename T, typename IT>
    __aicore__ inline void KernelSinkhorn<T, IT>::ComputeForExp(uint32_t length)
    {
        LocalTensor<T> costSrcLocal = costInQueue.DeQue<T>();
        LocalTensor<T> costDstLocal = costOutQueue.AllocTensor<T>();
        DUMP_LT_2(costSrcLocal, length, "cost: ");
        Exp(costDstLocal, costSrcLocal, length);
        costOutQueue.EnQue<T>(costDstLocal);
        costInQueue.FreeTensor(costSrcLocal);
    }

    // 通用float/half
    template<typename T, typename IT>
    template<typename _IT>
    __aicore__ inline void KernelSinkhorn<T, IT>::CopyOutForExp(uint32_t ind, uint32_t length)
    {
        // costOutQueue 为 QuePosition::VECOUT，EnQue/DeQue 自带 V->MTE3 序，故此处无需再补事件
        // （与下面 bfloat16_t 特化不同：特化经 VECIN 队列中转，必须显式补 V->MTE3）。
        LocalTensor<T> costLocal = costOutQueue.DeQue<T>();
        DUMP_LT_2(costLocal, length, " exp: ");
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(T)), 0, 0, 0};
        DataCopyPad(pGlobal[ind], costLocal, copyParams);
        costOutQueue.FreeTensor(costLocal);
    }

    // 特殊处理 T = float, IT = bfloat16_t
    template<>
    __aicore__ inline void KernelSinkhorn<float, bfloat16_t>::CopyInForExp(uint32_t ind, uint32_t length)
    {
        // 从costGlobal复制到tmpLocal
        LocalTensor<bfloat16_t> tmpLocal = costOutQueue.AllocTensor<bfloat16_t>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(bfloat16_t)), 0, 0, 0};
        DataCopyPadExtParams<bfloat16_t> padParams{false, 0, 0, 0};
        DataCopyPad(tmpLocal, costGlobal[ind], copyParams, padParams);
        // DataCopyPad(PIPE_MTE2)写UB后，Cast(PIPE_V)需读取同一UB，补MTE2->V同步（批准前移项）
        event_t eventIdMte2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2V);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2V);
        costOutQueue.EnQue(tmpLocal);

        // bf16 ==> float
        tmpLocal = costOutQueue.DeQue<bfloat16_t>();
        LocalTensor<float> costLocal = costInQueue.AllocTensor<float>();
        Cast(costLocal, tmpLocal, RoundMode::CAST_NONE, length);
        costInQueue.EnQue(costLocal);
        costOutQueue.FreeTensor(tmpLocal);
    }

    // 特殊处理 T = float, IT = bfloat16_t
    template<>
    template<>
    __aicore__ inline void KernelSinkhorn<float, bfloat16_t>::CopyOutForExp<bfloat16_t>(uint32_t ind, uint32_t length)
    {
        // float ==> bf16
        LocalTensor<float> costLocal = costOutQueue.DeQue<float>();
        LocalTensor<bfloat16_t> tmpLocal = costInQueue.AllocTensor<bfloat16_t>();
        Cast(tmpLocal, costLocal, RoundMode::CAST_TRUNC, length);
        // 评审 High #1：Cast 在 PIPE_V 上写 tmpLocal，随后 DataCopyPad 在 PIPE_MTE3 上读同一 UB。
        // 中转用的是 costInQueue（QuePosition::VECIN），其 EnQue/DeQue 只携带 MTE2->V 语义，
        // 不提供 V->MTE3 序，故必须在此显式补 V->MTE3 事件，否则 MTE3 可能搬出未写完的 Cast 结果。
        // 注意：simulator 串行执行各 pipe，缺失该事件也不会失败，只有实机才暴露（见交付报告残余风险）。
        event_t eventIdVMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVMte3);
        costInQueue.EnQue(tmpLocal);

        // bf16 ==> pGlobal
        tmpLocal = costInQueue.DeQue<bfloat16_t>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(bfloat16_t)), 0, 0, 0};
        DataCopyPad(pGlobal[ind], tmpLocal, copyParams);
        costInQueue.FreeTensor(tmpLocal);
        costOutQueue.FreeTensor(costLocal);
        // tmpLocal 取自 costInQueue（QuePosition::VECIN），其 freeBufEvt 是 V_MTE2，
        // FreeTensor/AllocTensor 都不提供 MTE3->V 序。上面那条 DataCopyPad 是 MTE3 对该
        // buffer 的读，而 InitD0GlobalInWS 紧接着会把同一 buffer 重新 AllocTensor 出来、
        // 用 Duplicate(PIPE_V) 写 float 1.0；缺 MTE3->V 序时 V 的写会抢在 MTE3 读完之前
        // 落地，pGlobal 收到的就是 float32(1.0) 而不是 bf16(exp(cost))——A3 缺陷 F-001。
        // 通用 float/half 路径经 costOutQueue(VECOUT) 中转，freeBufEvt 恰为 MTE3_V，
        // 框架自动补了这道序，故只有 bf16 直调（tiling key 27）受影响。
        // 用 AllocEventID 而非 FetchEventID：后者不占用 ID，会拿到与上面 costOutQueue
        // .FreeTensor 为 MTE3_V 分配的同一个 ID，形成同流水同 ID 的连续 SetFlag（ISASI
        // 文档明列为未定义行为）。放在两次 FreeTensor 之后即保证 ID 互不相同。
        event_t eventIdMte3V = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
        SetFlag<HardEvent::MTE3_V>(eventIdMte3V);
        WaitFlag<HardEvent::MTE3_V>(eventIdMte3V);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(eventIdMte3V);
    }
} // namespace AscendC

#endif // SINKHORN_KERNEL_EXP_INC_
