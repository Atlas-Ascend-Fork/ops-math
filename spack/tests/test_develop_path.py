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
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "prepare_cann_env.sh"


class DevelopPathTest(unittest.TestCase):
    def test_paths_with_and_without_spaces(self):
        source = SCRIPT.read_text(encoding="utf-8")
        function = re.search(
            r"^pull_develop_branch_local\(\) \{.*?^\}", source, re.M | re.S
        ).group()
        final_cd = re.search(r"^cd [^\n]*develop_path[^\n]*$", source, re.M).group()
        for name in ("source-tree", "source tree"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                (root / name).mkdir()
                harness = root / "run.sh"
                harness.write_text(
                    'root=$(cd "$(dirname "$0")" && pwd)\n'
                    f'develop_path="$root/{name}"\n'
                    "PACKAGE_NAME=cann-ops-math@master\n"
                    'spack() { if [ "$1" = develop ]; then printf "%s\\n" "$@" > "$root/args"; fi; return 0; }\n'
                    + function
                    + "\npull_develop_branch_local\n"
                    + 'printf "%s\\n" "$develop_path" > "$root/expected"\n'
                    + 'cd "$root"\n'
                    + final_cd
                    + '\nprintf "%s\\n" "$PWD" > "$root/actual"\n',
                    encoding="utf-8",
                )
                result = subprocess.run(
                    [os.environ.get("BASH", "bash"), str(harness)],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                )
                self.assertEqual(0, result.returncode, result.stderr)
                expected = (root / "expected").read_text(encoding="utf-8").strip()
                args = (root / "args").read_text(encoding="utf-8").splitlines()
                with self.subTest(check="develop arguments"):
                    self.assertEqual(
                        ["develop", "-p", expected, "cann-ops-math@master"], args
                    )
                with self.subTest(check="restored directory"):
                    self.assertEqual(
                        expected, (root / "actual").read_text(encoding="utf-8").strip()
                    )


if __name__ == "__main__":
    unittest.main()
