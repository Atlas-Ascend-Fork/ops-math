# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Validate actual data and verifier CLIs with CPU PyTorch BF16 as the oracle."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


def bf16_bits(tensor):
    return tensor.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)


class Bfloat16FilesTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.scripts = Path(__file__).resolve().parent

    def run_script(self, script, *args):
        return subprocess.run(
            [sys.executable, str(self.scripts / script), *map(str, args)],
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )

    def test_generator_matches_real_cpu_bfloat16_storage(self):
        for mode, value in (
            ("all_true", 5.1),
            ("all_false", 1.00390625),
            ("broadcast", 1.01171875),
        ):
            with self.subTest(mode=mode, value=value):
                completed = self.run_script(
                    "gen_data.py", "bfloat16", 2, 3, mode, value
                )
                self.assertEqual(
                    completed.returncode, 0, completed.stdout + completed.stderr
                )
                tensor = (
                    torch.arange(1, 7, dtype=torch.float32)
                    .reshape(2, 3)
                    .to(torch.bfloat16)
                )
                if mode == "broadcast":
                    mask = torch.tensor([[True, False, True]])
                else:
                    mask = torch.full((2, 3), mode == "all_true", dtype=torch.bool)
                scalar = torch.tensor([value], dtype=torch.bfloat16)
                expected = torch.where(mask, scalar, tensor)
                for name, reference in (
                    ("x", tensor),
                    ("value", scalar),
                    ("golden", expected),
                ):
                    stored = np.fromfile(self.root / f"bfloat16_{name}.bin", np.uint16)
                    np.testing.assert_array_equal(stored, bf16_bits(reference).ravel())

    def test_verifier_rejects_large_error_in_first_bfloat16_element(self):
        bf16_bits(torch.tensor([1.0, 2.0])).tofile(self.root / "bfloat16_golden.bin")
        bf16_bits(torch.tensor([8.0, 2.0])).tofile(self.root / "bfloat16_output.bin")
        completed = self.run_script("compare_data.py", "bfloat16")
        self.assertEqual(completed.returncode, 1, completed.stdout + completed.stderr)
        self.assertIn("FAILED!", completed.stdout)

    def test_verifier_accepts_matching_odd_number_of_bfloat16_elements(self):
        data = bf16_bits(torch.tensor([1.0, -2.0, 3.0]))
        data.tofile(self.root / "bfloat16_golden.bin")
        data.tofile(self.root / "bfloat16_output.bin")
        completed = self.run_script("compare_data.py", "bfloat16")
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("PASSED!", completed.stdout)

    def test_other_generated_dtypes_still_verify(self):
        for dtype, np_dtype in (
            ("float32", np.float32),
            ("int32", np.int32),
            ("uint8", np.uint8),
        ):
            with self.subTest(dtype=dtype):
                generated = self.run_script("gen_data.py", dtype, 2, 3, "all_true", 5)
                self.assertEqual(
                    generated.returncode, 0, generated.stdout + generated.stderr
                )
                golden = np.fromfile(self.root / f"{dtype}_golden.bin", np_dtype)
                np.testing.assert_array_equal(golden, np.full(6, 5, dtype=np_dtype))
                golden.tofile(self.root / f"{dtype}_output.bin")
                checked = self.run_script("compare_data.py", dtype)
                self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)


if __name__ == "__main__":
    unittest.main()
