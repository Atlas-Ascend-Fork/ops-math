#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "opgen_standalone.py"


class OperatorNameValidationTest(unittest.TestCase):
    def test_invalid_cpp_names_are_rejected_before_creating_files(self):
        for variant in ("default", "aicpu"):
            for name in ("123add", "_2add", "___"):
                with (
                    self.subTest(variant=variant, name=name),
                    tempfile.TemporaryDirectory() as folder,
                ):
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(SCRIPT),
                            "--op_type",
                            "math",
                            "--op_name",
                            name,
                            "--output_path",
                            folder,
                            "--template_variant",
                            variant,
                        ],
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                    )
                    self.assertNotEqual(0, result.returncode)
                    self.assertFalse((Path(folder) / "math" / name).exists())

    def test_valid_names_still_generate(self):
        for variant in ("default", "aicpu"):
            for name in ("add2", "_example"):
                with (
                    self.subTest(variant=variant, name=name),
                    tempfile.TemporaryDirectory() as folder,
                ):
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(SCRIPT),
                            "--op_type",
                            "math",
                            "--op_name",
                            name,
                            "--output_path",
                            folder,
                            "--template_variant",
                            variant,
                        ],
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                    )
                    self.assertEqual(
                        0, result.returncode, result.stdout + result.stderr
                    )
                    self.assertTrue(
                        (Path(folder) / "math" / name / "CMakeLists.txt").is_file()
                    )


if __name__ == "__main__":
    unittest.main()
