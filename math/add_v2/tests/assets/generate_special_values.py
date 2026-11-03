#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Generate exact floating-point bit patterns and TTK CSVs in a chosen directory."""

import argparse
import csv
import struct
from pathlib import Path

OPERATOR = "AddV2"

PATTERNS = {
    "float32": (
        "I",
        [
            0x00000001,
            0x80000001,
            0x007FFFFF,
            0x807FFFFF,
            0,
            0x80000000,
            0x7F800000,
            0xFF800000,
            0x7FC12345,
            0xFFC54321,
            0x3F800000,
            0xBF800000,
            0x40800000,
            0xC0600000,
            0x00800000,
            0x80800000,
        ],
    ),
    "float16": (
        "H",
        [
            0x0001,
            0x8001,
            0x03FF,
            0x83FF,
            0,
            0x8000,
            0x7C00,
            0xFC00,
            0x7E55,
            0xFEAB,
            0x3C00,
            0xBC00,
            0x4400,
            0xC300,
            0x0400,
            0x8400,
        ],
    ),
    "bfloat16": (
        "H",
        [
            0x0001,
            0x8001,
            0x007F,
            0x807F,
            0,
            0x8000,
            0x7F80,
            0xFF80,
            0x7FC5,
            0xFFD3,
            0x3F80,
            0xBF80,
            0x4080,
            0xC060,
            0x0080,
            0x8080,
        ],
    ),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for dtype, (code, bits) in PATTERNS.items():
        values = output / f"{dtype}_mixed_bits.bin"
        values.write_bytes(struct.pack("<" + code * len(bits), *bits))
        zeros = output / f"{dtype}_zeros.bin"
        zeros.write_bytes(bytes(values.stat().st_size))
        inputs = (str(values), str(zeros)) if OPERATOR == "AddV2" else (str(values),)
        rows.append(
            {
                "testcase_name": f"{OPERATOR}_special_{dtype}",
                "op_name": OPERATOR,
                "input_shapes": repr(((len(bits),),) * len(inputs)),
                "input_dtypes": repr((dtype,) * len(inputs)),
                "input_formats": repr(("ND",) * len(inputs)),
                "output_shapes": repr(((len(bits),),)),
                "output_dtypes": repr((dtype,)),
                "output_formats": repr(("ND",)),
                "manual_input_binaries": repr(inputs),
                "manual_golden_binaries": "()"
                if OPERATOR == "AddV2"
                else repr((str(values),)),
                "attributes": "{}",
                "remark": "subnormals, signed zeros, infinities, payload NaNs and ordinary values",
            }
        )
    with (output / "special_values.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(output / "special_values.csv")


if __name__ == "__main__":
    main()
