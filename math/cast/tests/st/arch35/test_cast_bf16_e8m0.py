# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Run against the installed Cast kernel on Ascend 950 with E8M0-enabled PyTorch."""

import unittest

try:
    import torch
    import torch_npu  # noqa: F401
except ImportError:
    torch = None


@unittest.skipUnless(
    torch is not None and hasattr(torch, "float8_e8m0fnu") and torch.npu.is_available(),
    "requires E8M0-enabled PyTorch and an Ascend NPU",
)
class TestCastBf16ToE8m0(unittest.TestCase):
    def check_cast(self, bit_patterns):
        # Compare raw output bytes: floating comparisons cannot validate NaN encodings.
        source = (
            torch.tensor(bit_patterns, dtype=torch.int32)
            .to(torch.int16)
            .view(torch.bfloat16)
        )
        expected = source.float().to(torch.float8_e8m0fnu).view(torch.uint8)
        actual = source.npu().to(torch.float8_e8m0fnu).view(torch.uint8).cpu()
        self.assertTrue(
            torch.equal(actual, expected),
            "NPU output differs from the PyTorch CPU cast",
        )

    def test_all_bf16_bit_patterns(self):
        # Both signs, all exponent/mantissa combinations, zeros, subnormals, NaNs and infinities.
        self.check_cast(list(range(1 << 16)))

    def test_rounding_boundaries_and_tails(self):
        boundaries = [
            0x0000,
            0x8000,
            0x0001,
            0x003F,
            0x0040,
            0x0041,
            0x007F,
            0x0080,
            0x3F80,
            0x3FBF,
            0x3FC0,
            0x3FC1,
            0x4040,
            0x7F3F,
            0x7F40,
            0x7F7F,
            0x7F80,
            0xFF80,
            0x7FC0,
            0xFFFF,
            0xBFC0,
        ]
        for size in (1, 63, 64, 65, 127, 128, 129, 255, 256, 257, 4095, 65535, 131073):
            with self.subTest(size=size):
                self.check_cast([boundaries[i % len(boundaries)] for i in range(size)])


if __name__ == "__main__":
    unittest.main()
