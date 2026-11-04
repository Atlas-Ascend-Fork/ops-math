/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file sinkhorn_update.inc
 * \brief
 */
#ifndef SINKHORN_KERNEL_UPDATE_INC_
#define SINKHORN_KERNEL_UPDATE_INC_

namespace AscendC {
    // 通用float/half
    template<typename T, typename IT>
    template<typename _IT>
    __aicore__ inline void KernelSinkhorn<T, IT>::CopyInFromP(uint16_t row, const GlobalTensor<_IT> &pG)
    {
        // 搬入cost
        uint16_t blockLen = totalCol * sizeof(T);
        LocalTensor<T> costLocal = costInQueue.AllocTensor<T>();
        DataCopyExtParams copyParams{(uint16_t)row, blockLen, 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(costLocal, pG, copyParams, padParams);
        costInQueue.EnQue(costLocal);
    }

    // 通用float/half
    template<typename T, typename IT>
    template<typename _IT>
    __aicore__ inline void KernelSinkhorn<T, IT>::SaveP(uint16_t row, const GlobalTensor<_IT> &pG, const LocalTensor<T> &localTensor)
    {
        uint16_t blockLen = totalCol * sizeof(T);
        DataCopyExtParams copyParams{(uint16_t)row, blockLen, 0, 0, 0};
        // localTensor 由 V 写出，且经 VECIN 队列传递（EnQue/DeQue 只带 MTE2->V），MTE3 搬出前补 V->MTE3 同步（批准前移项）
        event_t eventIdVMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVMte3);
        DataCopyPad(pG, localTensor, copyParams);
    }

    template<typename T, typename IT>
    __aicore__ inline void KernelSinkhorn<T, IT>::ComputeD0(uint32_t row, LocalTensor<T> costSrcLocal, LocalTensor<T> d1InLocal)
    {
        // costOutQueue = d1 * cost: 逐行计算 Mul
        {
            LocalTensor<T> costDstLocal = costOutQueue.AllocTensor<T>();
            for (int r = 0; r < row; r++) {
                uint32_t rowIdx = r * rowLengthAligned;
                Mul(costDstLocal[rowIdx], costSrcLocal[rowIdx], d1InLocal, totalCol);
            }
            costOutQueue.EnQue(costDstLocal);
        }

        // d0OutLocal = sum
        {
            LocalTensor<T> d0OutLocal = d0OutQueue.AllocTensor<T>();
            LocalTensor<T> d0OutLocal2 = d0OutQueue2.AllocTensor<T>();
            LocalTensor<T> d0OutLocal3 = d0OutQueue3.AllocTensor<T>();
            LocalTensor<T> costDstLocal = costOutQueue.DeQue<T>();

            event_t eventId_V_S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            for (int r = 0; r < row; r++) {
                uint32_t rowIdx = r * rowLengthAligned;
                ReduceSum(d0OutLocal2, costDstLocal[rowIdx], d0OutLocal3, totalCol);
                SetFlag<HardEvent::V_S>(eventId_V_S);
                WaitFlag<HardEvent::V_S>(eventId_V_S);
                d0OutLocal.SetValue(r, d0OutLocal2.GetValue(0));
            }
            // SetValue(Scalar)写UB后，EnQue 交给后续消费前补 S->V 排空（批准前移项）
            event_t eventId_S_V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(eventId_S_V);
            WaitFlag<HardEvent::S_V>(eventId_S_V);
            d0OutQueue.EnQue(d0OutLocal);
            d0OutQueue2.FreeTensor(d0OutLocal2);
            d0OutQueue3.FreeTensor(d0OutLocal3);
            costOutQueue.FreeTensor(costDstLocal);
        }

        // + eps
        {
            LocalTensor<T> d0OutLocal2 = d0OutQueue2.AllocTensor<T>();
            LocalTensor<T> d0OutLocal3 = d0OutQueue3.AllocTensor<T>();
            LocalTensor<T> d0OutLocal = d0OutQueue.DeQue<T>();
            DUMP_LT_3(d0OutLocal, row, "d0[sum]: ");
            Adds(d0OutLocal2, d0OutLocal, (T)EPS, row);
            DUMP_LT_3(d0OutLocal2, row, "d0[eps]: ");
            PipeBarrier<PIPE_V>();
            Duplicate(d0OutLocal3, static_cast<T>((1.0f)/totalRow), row);
            PipeBarrier<PIPE_V>();
            Div(d0OutLocal, d0OutLocal3, d0OutLocal2, row);
            DUMP_LT_3(d0OutLocal, row, "d0[div]: ");
            d0OutQueue.EnQue(d0OutLocal);
            d0OutQueue3.FreeTensor(d0OutLocal3);
            d0OutQueue2.FreeTensor(d0OutLocal2);
        }
    }

    // 计算每个Tile的d1   torch.sum(d0.unsqueeze(1) * cost, 1)
    template<typename T, typename IT>
    __aicore__ inline void KernelSinkhorn<T, IT>::ComputeD1(uint32_t row, LocalTensor<T> costSrcLocal, LocalTensor<T> d0OutLocal)
    {
        LocalTensor<T> d1OutLocal = d1OutQueue.AllocTensor<T>();

        // simulator 断言：本函数按 costSrcLocal[r * rowLengthAligned]（r < row）逐行读、
        // 按 d0OutLocal.GetValue(r) 逐行标量读，二者的 UB 容量均以 tileRow 为上界。
        SINKHORN_SIM_ASSERT(static_cast<uint64_t>(row) <= tileRow, "ComputeD1: tile row count exceeds tileRow");

        DataCopyExtParams copyParams{1, static_cast<uint32_t>(totalCol * sizeof(T)), 0, 0, 0};
        SetAtomicAdd<T>();
        // 评审 High #2：d0OutLocal 由 ComputeD0 的 Div(PIPE_V) 写出，经 d0OutQueue（QuePosition::VECOUT）
        // 的 EnQue/DeQue 传到此处，只携带 V->MTE3 序；下面的 GetValue 是 PIPE_S 标量读，
        // 队列同步不覆盖 V->S，必须在首次标量读之前显式补 V->S 事件，否则可能读到未写完的向量结果。
        // 该事件一次覆盖整个循环（d0OutLocal 的全部 row 个元素均由同一批 V 指令写出）。
        // 注意：simulator 串行执行各 pipe，缺失该事件也不会失败，只有实机才暴露（见交付报告残余风险）。
        event_t eventId_V_S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventId_V_S);
        WaitFlag<HardEvent::V_S>(eventId_V_S);
        event_t eventId_MTE3_V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventId_MTE3_V);
        for (int r = 0; r < row; r++) {
            uint32_t rowIdx = r * rowLengthAligned;
            T d0 = d0OutLocal.GetValue(r);
            WaitFlag<HardEvent::MTE3_V>(eventId_MTE3_V);
            Muls(d1OutLocal, costSrcLocal[rowIdx], d0, totalCol);
            event_t eventId_V_MTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventId_V_MTE3);
            WaitFlag<HardEvent::V_MTE3>(eventId_V_MTE3);
            DataCopyPad(d1BlockInWS, d1OutLocal, copyParams);
            SetFlag<HardEvent::MTE3_V>(eventId_MTE3_V);
        }
        WaitFlag<HardEvent::MTE3_V>(eventId_MTE3_V);
        SetAtomicNone();
        d1OutQueue.FreeTensor(d1OutLocal);
    }

    template<typename T, typename IT>
    __aicore__ inline void KernelSinkhorn<T, IT>::UpdateD0()
    {
        // d0 = (1 / d0.size(0)) * 1 / (torch.sum(d1 * cost, 1) + eps)

        DataCacheClean(d1GlobalInWS);
        DataCacheClean(d1GlobalInWSNew);
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(totalCol * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

        // d1InQueue = d1
        {
            LocalTensor<T> d1InLocal = d1InQueue.AllocTensor<T>();
            DataCopyPad(d1InLocal, d1GlobalInWS, copyParams, padParams);
            d1InQueue.EnQue(d1InLocal);
        }

        LocalTensor<T> d1InLocal = d1InQueue.DeQue<T>();

        // d1BlockInWS = 0
        {
            LocalTensor<T> d1OutLocal = d1OutQueue.AllocTensor<T>();
            event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            Duplicate(d1OutLocal, static_cast<T>(0), totalCol);
            SetFlag<HardEvent::V_MTE3>(eventId);
            WaitFlag<HardEvent::V_MTE3>(eventId);
            DataCopyPad(d1BlockInWS, d1OutLocal, copyParams);
            d1OutQueue.FreeTensor(d1OutLocal);
            DataCacheClean(d1BlockInWS);
        }

        for (int t = 0; t < tileNum; t++) {
            uint32_t tileIdx = t * this->tileLength;
            uint32_t row = tileRow;
            if (t == this->tileNum - 1) {
                row = lastTileRow;
            }

            // 搬入cost
            CopyInFromP<IT>(row, pGlobal[tileIdx]);

            // 计算d0
            LocalTensor<T> costSrcLocal = costInQueue.DeQue<T>();
            ComputeD0(row, costSrcLocal, d1InLocal);

            LocalTensor<T> d0OutLocal = d0OutQueue.DeQue<T>();

            ComputeD1(row, costSrcLocal, d0OutLocal);
            DataCacheClean(d1BlockInWS);
            DUMP_LT_2(d1BlockInWS, totalCol, "d1[sum]: ");

            // copy to workspace
            {
                DataCopyExtParams copyParams{1, static_cast<uint32_t>(row * sizeof(T)), 0, 0, 0};
                DataCopyPad(d0BlockInWS[tileRow * t], d0OutLocal, copyParams);
                // d0OutLocal 由 V 写出、经 MTE3 搬出后随即被 FreeTensor 复用，补 MTE3->V 排空（批准前移项）
                event_t eventIdMte3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
                SetFlag<HardEvent::MTE3_V>(eventIdMte3V);
                WaitFlag<HardEvent::MTE3_V>(eventIdMte3V);
            }
            DUMP_LT_2(d1BlockInWS, totalCol, "d1[s01]: ");

            d0OutQueue.FreeTensor(d0OutLocal);
            costInQueue.FreeTensor(costSrcLocal);
        }

        d1InQueue.FreeTensor(d1InLocal);

        SyncAll();
        DataCacheCleanAndInvalid<uint64_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(headerInWS);
        DUMP_LT_0_2(d0BlockInWS, blockRow, "d0 : ");
    }

    template<typename T, typename IT>
    __aicore__ inline void KernelSinkhorn<T, IT>::SumD1(int block)
    {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(totalCol * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

        // copy from core level
        {
            LocalTensor<T> d1InLocal = d1InQueue.AllocTensor<T>();
            DataCopyPad(d1InLocal, d1BlockInWS[block * totalCol], copyParams, padParams);
            d1InQueue.EnQue(d1InLocal);
        }
        {
            LocalTensor<T> d1OutLocal = d1OutQueue.DeQue<T>();
            LocalTensor<T> d1InLocal = d1InQueue.DeQue<T>();
            DUMP_LT_2(d1InLocal, totalCol, " d1[   ]: ");
            Add(d1OutLocal, d1InLocal, d1OutLocal, totalCol);
            d1OutQueue.EnQue(d1OutLocal);
            d1InQueue.FreeTensor(d1InLocal);
        }
    }

    template<typename T, typename IT>
    __aicore__ inline void KernelSinkhorn<T, IT>::UpdateD1()
    {
        if (blockIdx == 0) {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(totalCol * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

            // d1OutQueue = new d1
            {
                LocalTensor<T> d1OutLocal = d1OutQueue.AllocTensor<T>();
                Duplicate(d1OutLocal, static_cast<T>(0.0f), totalCol);
                d1OutQueue.EnQue(d1OutLocal);
                for (int i = 0; i < numBlocks; i++) {
                    SumD1(i);
                }
            }

            // d1OutQueue2 = new d1 + eps
            {
                LocalTensor<T> d1OutLocal = d1OutQueue.DeQue<T>();
                DUMP_LT_2(d1OutLocal, totalCol, " d1[sum]: ");
                LocalTensor<T> d1OutLocal2 = d1OutQueue2.AllocTensor<T>();
                Adds(d1OutLocal2, d1OutLocal, (T)EPS, totalCol);
                d1OutQueue2.EnQue(d1OutLocal2);
                d1OutQueue.FreeTensor(d1OutLocal);
            }

            // d1OutQueue3 = 1.0 / totalCol
            {
                LocalTensor<T> d1OutLocal3 = d1OutQueue3.AllocTensor<T>();
                Duplicate(d1OutLocal3, static_cast<T>(1.0f / totalCol), totalCol);
                d1OutQueue3.EnQue(d1OutLocal3);
            }

            // d1InQueue = d1OutQueue3 / d1OutQueue2 =  (1.0 / totalCol) / (d1 new)
            {
                LocalTensor<T> d1InLocal = d1InQueue.AllocTensor<T>();
                LocalTensor<T> d1OutLocal2 = d1OutQueue2.DeQue<T>();
                LocalTensor<T> d1OutLocal3 = d1OutQueue3.DeQue<T>();

                Div(d1InLocal, d1OutLocal3, d1OutLocal2, totalCol);
                d1InQueue.EnQue(d1InLocal);
                d1OutQueue2.FreeTensor(d1OutLocal2);
                d1OutQueue3.FreeTensor(d1OutLocal3);
            }

            // 取出新 d1
            LocalTensor<T> d1InLocal = d1InQueue.DeQue<T>();

            // 旧 d1 搬入专用 VECIN 队列 d1OldInQueue：EnQue/DeQue 自带 MTE2->V 同步，
            // 不再复用 d1OutQueue2 跨 V/MTE2，规避手动 MTE2_V 与活跃 VECIN 队列抢 event id（报错4/5/6/7/8）
            LocalTensor<T> d1OldLocal = d1OldInQueue.AllocTensor<T>();
            DataCopyPad(d1OldLocal, d1GlobalInWS, copyParams, padParams);
            d1OldInQueue.EnQue(d1OldLocal);

            // 写新值（MTE3）：同一 d1GlobalInWS 区，旧 d1 的 MTE2 读须先完成，补 MTE2->MTE3 同步（报错9）；
            // d1InLocal 由 V 写出、d1InQueue 为 VECIN（EnQue/DeQue 只带 MTE2->V），故还需补 V->MTE3 同步（批准前移项）
            {
                event_t eventIdVMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
                SetFlag<HardEvent::V_MTE3>(eventIdVMte3);
                WaitFlag<HardEvent::V_MTE3>(eventIdVMte3);
                event_t eventIdMte2Mte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
                SetFlag<HardEvent::MTE2_MTE3>(eventIdMte2Mte3);
                WaitFlag<HardEvent::MTE2_MTE3>(eventIdMte2Mte3);
                DataCopyPad(d1GlobalInWS, d1InLocal, copyParams);
                PipeBarrier<PIPE_MTE3>();
                DataCacheClean(d1GlobalInWS);
            }

            // 清空新值
            {
                LocalTensor<T> d1OutLocal3 = d1OutQueue3.AllocTensor<T>();
                Duplicate(d1OutLocal3, static_cast<T>(0.0f), totalCol);
                event_t eventId_V_MTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
                SetFlag<HardEvent::V_MTE3>(eventId_V_MTE3);
                WaitFlag<HardEvent::V_MTE3>(eventId_V_MTE3);
                DataCopyPad(d1GlobalInWSNew, d1OutLocal3, copyParams);
                d1OutQueue3.FreeTensor(d1OutLocal3);
            }

            // 收敛判断：mean(|old_d1 - new_d1|)
            {
                DataCacheClean(d1GlobalInWSNew);
                LocalTensor<T> d1OldLocalIn = d1OldInQueue.DeQue<T>();
                LocalTensor<T> d1OutLocal3 = d1OutQueue3.AllocTensor<T>();
                LocalTensor<T> d1OutLocal = d1OutQueue.AllocTensor<T>();
                LocalTensor<T> d1OutLocal2 = d1OutQueue2.AllocTensor<T>();

                // d1_old - d1
                Sub(d1OutLocal3, d1OldLocalIn, d1InLocal, totalCol);
                DUMP_LT_3(d1OutLocal3, totalCol, " d1[sub]: ");
                PipeBarrier<PIPE_V>();
                Abs(d1OutLocal, d1OutLocal3, totalCol);
                DUMP_LT_3(d1OutLocal, totalCol, " d1[abs]: ");
                PipeBarrier<PIPE_V>();
                ReduceSum(d1OutLocal2, d1OutLocal, d1OutLocal3, totalCol);
                event_t eventId_V_S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
                SetFlag<HardEvent::V_S>(eventId_V_S);
                WaitFlag<HardEvent::V_S>(eventId_V_S);
                float tolerance = (float)(d1OutLocal2.GetValue(0)) / totalCol;
                OP_LOGD_0_1("tol[%d]: " FLOAT_FMT, loopCount, tolerance);
                // IEEE 谓词（批准前移项）：NaN 时 !(tolerance > tol) 为 true ⇒ 退出，
                // 避免 baseline `tolerance <= tol` 在 NaN 下恒 false 导致死循环（与 MAX_SINKHORN_ITER 配套）
                if (!(tolerance > tol)) {
                    // 退出循环
                    SetLoopFlag(0);
                }
                d1OutQueue.FreeTensor(d1OutLocal);
                d1OutQueue3.FreeTensor(d1OutLocal3);
                d1OutQueue2.FreeTensor(d1OutLocal2);
                d1OldInQueue.FreeTensor(d1OldLocalIn);
            }

            d1InQueue.FreeTensor(d1InLocal);
        }

        SyncAll();
        DataCacheClean(d1GlobalInWS);
        DataCacheClean(d1GlobalInWSNew);
        DUMP_LT_0_2(d1GlobalInWS, totalCol, " d1[gl ]: ");
        DUMP_LT_0_2(d1GlobalInWSNew, totalCol, " d1[gln]: ");
    }

    // 特殊处理 T = float, IT = bfloat16_t
    template<>
    template<>
    __aicore__ inline void KernelSinkhorn<float, bfloat16_t>::CopyInFromP<bfloat16_t>(uint16_t row, const GlobalTensor<bfloat16_t> &pG)
    {
        // 从pGlobal复制到tmpLocal（bf16），逐行 Cast 到 float。
        // DataCopyPad 多 block 搬入后 UB 行步长为 AlignUp32(totalCol*2)/2，与 float 侧 rowLengthAligned 不同，
        // 故按行 Cast（baseline 单次连续 Cast 仅在 totalColAligned==totalCol 时正确，此处修正为逐行）。
        uint16_t blockLen = totalCol * sizeof(bfloat16_t);
        uint32_t inputRowAligned = (static_cast<uint32_t>(blockLen) + 31) / 32 * 32 / sizeof(bfloat16_t);
        LocalTensor<bfloat16_t> tmpLocal = costOutQueue.AllocTensor<bfloat16_t>();
        DataCopyExtParams copyParams{row, blockLen, 0, 0, 0};
        DataCopyPadExtParams<bfloat16_t> padParams{false, 0, 0, 0};
        DataCopyPad(tmpLocal, pG, copyParams, padParams);
        // DataCopyPad(PIPE_MTE2)写UB后，Cast(PIPE_V)读取同一UB，补MTE2->V同步（批准前移项）
        event_t eventIdMte2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2V);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2V);
        costOutQueue.EnQue(tmpLocal);

        // bf16 ==> float
        tmpLocal = costOutQueue.DeQue<bfloat16_t>();
        LocalTensor<float> costLocal = costInQueue.AllocTensor<float>();
        for (int r = 0; r < row; r++) {
            Cast(costLocal[r * rowLengthAligned], tmpLocal[r * inputRowAligned], RoundMode::CAST_NONE, totalCol);
        }
        costInQueue.EnQue(costLocal);
        costOutQueue.FreeTensor(tmpLocal);
    }

    // 特殊处理 T = float, IT = bfloat16_t
    template<>
    template<>
    __aicore__ inline void KernelSinkhorn<float, bfloat16_t>::SaveP<bfloat16_t>(uint16_t row, const GlobalTensor<bfloat16_t> &pG, const LocalTensor<float> &localTensor)
    {
        // float ==> bf16，逐行 Cast 到 UB 行步长 outputRowAligned，再单次多 block 搬出到 GM。
        // 输出 Cast 采用 RINT（契约前移项 D3：FP16/BF16 输出 RINT）。
        uint32_t blockLen = totalCol * sizeof(bfloat16_t);
        uint32_t outputRowAligned = (blockLen + 31) / 32 * 32 / sizeof(bfloat16_t);
        LocalTensor<bfloat16_t> tmpLocal = costOutQueue.AllocTensor<bfloat16_t>();
        for (int r = 0; r < row; r++) {
            Cast(tmpLocal[r * outputRowAligned], localTensor[r * rowLengthAligned], RoundMode::CAST_RINT, totalCol);
        }
        // Cast(PIPE_V)写UB后，DataCopyPad(PIPE_MTE3)搬出前补 V->MTE3 同步（批准前移项）
        event_t eventIdVMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVMte3);
        costOutQueue.EnQue(tmpLocal);

        // bf16 ==> pG
        tmpLocal = costOutQueue.DeQue<bfloat16_t>();
        DataCopyExtParams copyParams{row, blockLen, 0, 0, 0};
        DataCopyPad(pG, tmpLocal, copyParams);
        costOutQueue.FreeTensor(tmpLocal);
    }
} // namespace AscendC

#endif // SINKHORN_KERNEL_UPDATE_INC_
