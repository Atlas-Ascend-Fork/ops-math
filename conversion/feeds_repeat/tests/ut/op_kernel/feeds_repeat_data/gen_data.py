#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""FeedsRepeat kernel UT 数据生成器。

用法：python3 gen_data.py [dtype]      # dtype ∈ {float32, float16}，默认 float32
输出：input_feeds.bin / input_feeds_repeat_times.bin / golden.bin

固定用例（与 test_feeds_repeat.cpp 中手工设置的 tiling 参数保持一致）：
  feeds              shape = (4, 10, 21)   -> length=4, elem_row=10*21=210
  feeds_repeat_times       = [2, 2, 2, 2]  -> 复制后 8 行
  output_feeds_size        = 12            -> 尾部 4 行清零（覆盖输出清零路径）
"""

import os
import sys

import numpy as np

N = 4
ROW_SHAPE = [10, 21]
REPEAT_TIMES = [2, 2, 2, 2]
OUTPUT_FEEDS_SIZE = 12

DTYPE_MAP = {"float32": np.float32, "float16": np.float16}


def gen_data(dtype):
    feeds_dtype = DTYPE_MAP[dtype]
    ori_shape = [N] + ROW_SHAPE
    feeds = np.random.uniform(-1.0, 1.0, ori_shape).astype(feeds_dtype)
    repeat_times = np.array(REPEAT_TIMES, dtype=np.int32)

    repeated = np.repeat(feeds, repeat_times, axis=0)
    golden = np.zeros([OUTPUT_FEEDS_SIZE] + ROW_SHAPE, dtype=feeds_dtype)
    copy_rows = min(repeated.shape[0], OUTPUT_FEEDS_SIZE)
    golden[:copy_rows] = repeated[:copy_rows]

    feeds.tofile("./input_feeds.bin")
    repeat_times.tofile("./input_feeds_repeat_times.bin")
    golden.tofile("./golden.bin")


def main():
    dtype = sys.argv[1] if len(sys.argv) > 1 else "float32"
    if dtype not in DTYPE_MAP:
        raise SystemExit("unsupported dtype: {} (expect float32/float16)".format(dtype))
    os.system("rm -f ./*.bin")
    gen_data(dtype)


if __name__ == "__main__":
    main()
