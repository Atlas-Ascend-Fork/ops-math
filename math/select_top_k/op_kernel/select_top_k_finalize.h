/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SELECT_TOP_K_FINALIZE_H_
#define SELECT_TOP_K_FINALIZE_H_
#include "select_top_k_tiling_data.h"
#include "select_top_k_base_common.h"

namespace SelectTopK {
using namespace SelectTopKBase;

// 主计算输出原轴 INT32 索引；类型转换、标量偏移和区间填充在同次内核启动内完成。
template <typename T, typename I>
class Finalize {
public:
    // 绑定边界、临时结果和输出地址，并建立收尾阶段 UB 缓冲。
    __aicore__ inline void Init(GM_ADDR begin, GM_ADDR end, GM_ADDR coreValues, GM_ADDR coreIndices, GM_ADDR values,
                                GM_ADDR indices, const SelectTopKTilingData* data, TPipe* pipe)
    {
        data_ = data;
        begin_.SetGlobalBuffer((__gm__ int32_t*)begin);
        end_.SetGlobalBuffer((__gm__ int32_t*)end);
        coreValues_.SetGlobalBuffer((__gm__ T*)coreValues);
        coreIndices_.SetGlobalBuffer((__gm__ int32_t*)coreIndices);
        values_.SetGlobalBuffer((__gm__ T*)values);
        indices_.SetGlobalBuffer((__gm__ I*)indices);
        const bool narrow = data_->indicesOffset >= INT32_MIN &&
                            data_->indicesOffset <= INT32_MAX - static_cast<int64_t>(data_->base.axis);
        if (!(data_->flags & RANGE) && narrow && data_->base.rows * data_->outputK > BLOCK) {
            block_ = Min<uint32_t>(
                FINALIZE_MAX_BLOCK,
                data_->base.ubBytes / (sizeof(T) + INDEX32_BYTES + INDEX64_BYTES) / UB_BLOCK_BYTES * UB_BLOCK_BYTES);
            block_ = block_ < BLOCK ? BLOCK : block_;
        }
        pipe->InitBuffer(buffer_, block_ * (sizeof(T) + INDEX32_BYTES + INDEX64_BYTES));
        raw_ = buffer_.Get<T>();
        index32_ = buffer_.Get<int32_t>()[block_ * sizeof(T) / INDEX32_BYTES];
        index64_ = buffer_.Get<int64_t>()[block_ * (sizeof(T) + INDEX32_BYTES) / INDEX64_BYTES];
    }

    // 按行或连续索引块完成有效结果转换与不足 K 项的填充。
    __aicore__ inline void Process()
    {
        if (!(data_->flags & RANGE)) {
            // 每行均有 K 个有效项且偏移相同，将索引视为连续数组处理，减少短行的 DMA 和转换次数。
            const uint64_t items = data_->base.rows * data_->outputK;
            for (uint64_t task = GetBlockIdx(); task < DivUp(items, block_); task += data_->base.cores) {
                const uint64_t start = task * block_;
                const uint32_t count = Min<uint64_t>(block_, items - start);
                if (data_->flags & WRITE_INDICES) {
                    WriteIndices(start, start, count, count);
                }
            }
            return;
        }
        const uint64_t blocks = DivUp(data_->outputK, block_);
        for (uint64_t task = GetBlockIdx(); task < data_->base.rows * blocks; task += data_->base.cores) {
            const uint64_t row = task / blocks;
            const uint64_t start = (task % blocks) * block_;
            const uint32_t count = Min<uint64_t>(block_, data_->outputK - start);
            const uint32_t length = RowLength(row);
            const uint64_t validK = Min<uint64_t>(length, data_->base.k);
            const uint32_t valid = start < validK ? Min<uint64_t>(count, validK - start) : 0;
            const uint64_t source = row * data_->base.k + start;
            const uint64_t destination = row * data_->outputK + start;
            if ((data_->flags & RANGE) && (data_->flags & WRITE_VALUES)) {
                WriteValues(source, destination, valid, count);
            }
            if (data_->flags & WRITE_INDICES) {
                WriteIndices(source, destination, valid, count);
            }
        }
    }

private:
    // 裁剪行区间并返回真实有效长度。
    __aicore__ inline uint32_t RowLength(uint64_t row)
    {
        int32_t begin = 0, end = static_cast<int32_t>(data_->base.axis);
        if (data_->flags & HAS_BEGIN) {
            begin = begin_.GetValue((data_->flags & BROADCAST_BEGIN) ? 0 : row);
        }
        if (data_->flags & HAS_END) {
            end = end_.GetValue((data_->flags & BROADCAST_END) ? 0 : row);
        }
        begin = begin < 0 ? 0 : Min<uint32_t>(begin, data_->base.axis);
        end = end < 0 ? 0 : Min<uint32_t>(end, data_->base.axis);
        return end > begin ? end - begin : 0;
    }

    // 将当前有效结果搬入 UB，等待后续填充或转换使用。
    template <typename U>
    __aicore__ inline void Read(const LocalTensor<U>& target, const GlobalTensor<U>& source, uint32_t count)
    {
        if (count == 0) {
            return;
        }
        Fence<HardEvent::V_MTE2>();
        DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(U)), 0, 0, 0};
        DataCopyPad(target, source, params, DataCopyPadExtParams<U>{false, 0, 0, 0});
        Fence<HardEvent::MTE2_V>();
    }

    // 保留有效值的位模式，仅对不足 K 的位置写 value_fill。
    __aicore__ inline void WriteValues(uint64_t source, uint64_t destination, uint32_t valid, uint32_t count)
    {
        T fillValue{};
        if (valid < count) {
            auto fill = index64_.template ReinterpretCast<float>();
            Duplicate(fill, data_->valueFill, count);
            PipeBarrier<PIPE_V>();
            if constexpr (std::is_same<T, float>::value) {
                DataCopy(raw_.template ReinterpretCast<uint32_t>(), fill.template ReinterpretCast<uint32_t>(),
                         static_cast<uint32_t>(DivUp(count, FLOAT_BLOCK_ELEMENTS) * FLOAT_BLOCK_ELEMENTS));
            } else {
                Cast(raw_, fill, RoundMode::CAST_RINT, count);
            }
            PipeBarrier<PIPE_V>();
            Fence<HardEvent::V_S>();
            fillValue = raw_.GetValue(0);
            Fence<HardEvent::S_V>();
        }
        Read(raw_, coreValues_[source], valid);
        if (valid < count && valid % (UB_BLOCK_BYTES / sizeof(T))) {
            // GM 到 UB 的 DMA 会写入末尾块的无效字节；仅恢复该短尾，保留其余向量填充值。
            Fence<HardEvent::MTE2_S>();
            const uint32_t end = Min<uint32_t>(count,
                                               DivUp(valid, UB_BLOCK_BYTES / sizeof(T)) * (UB_BLOCK_BYTES / sizeof(T)));
            for (uint32_t i = valid; i < end; ++i) {
                raw_.SetValue(i, fillValue);
            }
            Fence<HardEvent::S_V>();
        }
        StoreToGm(values_[destination], raw_, count);
    }

    // 仅对有效原轴索引加偏移；无效位置直接写 index_fill。
    __aicore__ inline void WriteIndices(uint64_t source, uint64_t destination, uint32_t valid, uint32_t count)
    {
        if constexpr (std::is_same<I, int32_t>::value) {
            if (valid < count) {
                Duplicate(index32_, static_cast<int32_t>(data_->indexFill), count);
                PipeBarrier<PIPE_V>();
            }
            Read(index32_, coreIndices_[source], valid);
            if (valid < count && valid % FLOAT_BLOCK_ELEMENTS) {
                Fence<HardEvent::MTE2_S>();
                const uint32_t end = Min<uint32_t>(count, DivUp(valid, FLOAT_BLOCK_ELEMENTS) * FLOAT_BLOCK_ELEMENTS);
                for (uint32_t i = valid; i < end; ++i) {
                    index32_.SetValue(i, static_cast<int32_t>(data_->indexFill));
                }
                Fence<HardEvent::S_V>();
            }
            if (valid && data_->indicesOffset != 0) {
                Adds(index32_, index32_, static_cast<int32_t>(data_->indicesOffset), valid);
                PipeBarrier<PIPE_V>();
            }
            StoreToGm(indices_[destination], index32_, count);
        } else {
            Read(index32_, coreIndices_[source], valid);
            // 结果可用 INT32 表示时采用向量转换；大 INT64 偏移使用标量运算以保持精度。
            const bool narrow = data_->indicesOffset >= INT32_MIN &&
                                data_->indicesOffset <= INT32_MAX - static_cast<int64_t>(data_->base.axis);
            if (valid && narrow) {
                if (data_->indicesOffset != 0) {
                    Adds(index32_, index32_, static_cast<int32_t>(data_->indicesOffset), valid);
                    PipeBarrier<PIPE_V>();
                }
                Cast(index64_, index32_, RoundMode::CAST_NONE, valid);
                PipeBarrier<PIPE_V>();
            }
            if (!narrow || valid < count) {
                Fence<HardEvent::V_S>();
                if (!narrow) {
                    for (uint32_t i = 0; i < valid; ++i) {
                        index64_.SetValue(i, static_cast<int64_t>(index32_.GetValue(i)) + data_->indicesOffset);
                    }
                }
                for (uint32_t i = valid; i < count; ++i) {
                    index64_.SetValue(i, data_->indexFill);
                }
                Fence<HardEvent::S_V>();
            }
            StoreToGm(indices_[destination], index64_, count);
        }
    }

    static constexpr uint32_t BLOCK = 1024;
    uint32_t block_ = BLOCK;
    const SelectTopKTilingData* data_ = nullptr;
    GlobalTensor<int32_t> begin_, end_, coreIndices_;
    GlobalTensor<T> coreValues_, values_;
    GlobalTensor<I> indices_;
    TBuf<TPosition::VECCALC> buffer_;
    LocalTensor<T> raw_;
    LocalTensor<int32_t> index32_;
    LocalTensor<int64_t> index64_;
};
} // namespace SelectTopK
#endif
