# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import glob
import os
import sys

import numpy as np

curr_dir = os.path.dirname(os.path.realpath(__file__))


def compare_data(golden_file_lists, output_file_lists, d_type):
    dtype_map = {"float16": np.float16, "float32": np.float32, "int32": np.int32}
    np_dtype = dtype_map[d_type]

    # 禁止空目录或文件数量不匹配时默认返回 True 造成假绿
    if not golden_file_lists or len(golden_file_lists) != len(output_file_lists):
        print("FAILED! empty or mismatched bin file list.")
        return False

    data_same = True
    for gold, out in zip(golden_file_lists, output_file_lists):
        tmp_out = np.fromfile(out, np_dtype)
        tmp_gold = np.fromfile(gold, np_dtype)
        if tmp_out.size == 0 or tmp_out.shape != tmp_gold.shape:
            print(
                f"FAILED! empty or shape mismatch: out={tmp_out.shape}, golden={tmp_gold.shape}"
            )
            data_same = False
            continue
        diff_res = np.isclose(tmp_out, tmp_gold, 0, 0, False)
        diff_idx = np.where(~diff_res)[0]
        if len(diff_idx) == 0:
            print("PASSED!")
        else:
            print("FAILED! diff count:", len(diff_idx))
            print("output:", tmp_out[diff_idx[:10]])
            print("golden:", tmp_gold[diff_idx[:10]])
            data_same = False
    return data_same


def get_file_lists(dtype):
    # 按当前测试数据类型精确过滤文件，防止不同 dtype 残留文件串用
    golden_file_lists = sorted(glob.glob(f"{curr_dir}/{dtype}*golden*.bin"))
    output_file_lists = sorted(glob.glob(f"{curr_dir}/{dtype}*output*.bin"))
    return golden_file_lists, output_file_lists


def process(d_type):
    golden_file_lists, output_file_lists = get_file_lists(d_type)
    return compare_data(golden_file_lists, output_file_lists, d_type)


if __name__ == "__main__":
    ret = process(sys.argv[1])
    sys.exit(0 if ret else 1)
