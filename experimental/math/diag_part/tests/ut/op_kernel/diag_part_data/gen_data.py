# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import os
import sys

import numpy as np


def parse_str_to_shape_list(shape_str):
    shape_str = shape_str.strip("(").strip(")")
    shape_list = [int(x) for x in shape_str.split(",")]
    return np.array(shape_list)


def gen_data_and_golden(shape_str, d_type="float16"):
    d_type_dict = {"float16": np.float16, "float32": np.float32, "int32": np.int32}
    np_type = d_type_dict[d_type]
    shape = parse_str_to_shape_list(shape_str)

    if d_type == "int32":
        input_x = np.random.randint(-100, 100, size=shape).astype(np_type)
    else:
        input_x = np.random.uniform(-10, 10, shape).astype(np_type)

    golden = np.diag(input_x).astype(np_type)

    input_x.tofile(f"{d_type}_input_x_diag_part.bin")
    golden.tofile(f"{d_type}_golden_diag_part.bin")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Param num must be 3.")
        sys.exit(1)
    # 仅清理当前数据类型前缀的历史数据，避免误删并行或同目录其他 dtype 文件
    for fname in os.listdir("."):
        if fname.startswith(f"{sys.argv[2]}_") and fname.endswith(".bin"):
            os.remove(fname)
    gen_data_and_golden(sys.argv[1], sys.argv[2])
