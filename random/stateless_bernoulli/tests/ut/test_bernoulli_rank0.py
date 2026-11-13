# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Host regressions for logical rank-zero tensors promoted by the test framework."""

import ast
import csv
import importlib.util
from pathlib import Path
import unittest
from unittest import mock

import numpy as np


path = Path(__file__).resolve().parents[1] / "assets/golden.py"
spec = importlib.util.spec_from_file_location("bernoulli_rank0_golden", path)
golden = importlib.util.module_from_spec(spec)
spec.loader.exec_module(golden)


class BernoulliRankZeroTest(unittest.TestCase):
    def test_geir_regression_csv_matches_golden_shape(self):
        csv_path = (
            Path(__file__).resolve().parents[1] / "st/arch35/ttk_geir_bernoulli_st.csv"
        )
        with csv_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 9)
        names = {row["testcase_name"] for row in rows}
        self.assertTrue(
            names.isdisjoint(
                {
                    "bernoulli_geir_rank0_i32_scalar",
                    "bernoulli_geir_rank0_i64_scalar",
                    "bernoulli_geir_rank0_i64_tensor",
                }
            )
        )
        for row in rows:
            with self.subTest(case=row["testcase_name"]):
                self.assertNotIn(None, row)
                shapes = ast.literal_eval(row["input_shapes"])
                declared = ast.literal_eval(row["input_ori_shapes"])
                dtypes = ast.literal_eval(row["input_dtypes"])
                ranges = ast.literal_eval(row["input_data_ranges"])
                attrs = ast.literal_eval(row["attributes"])
                inputs = [
                    np.full(s or (1,), r[0], dtype=dt)
                    for s, dt, r in zip(shapes, dtypes, ranges)
                ]
                result = golden.stateless_bernoulli_golden(
                    *inputs,
                    probMode=attrs["probMode"],
                    input_ori_shapes=declared,
                    output_dtypes=("uint8",),
                )
                self.assertEqual(
                    result.shape, ast.literal_eval(row["output_shapes"])[0]
                )

    def test_logical_rank_one_probability_is_not_promoted_to_scalar(self):
        with self.assertRaisesRegex(ValueError, "prob rank"):
            golden.stateless_bernoulli_golden(
                np.array([], np.int64),
                np.array([0.5], np.float32),
                np.array([6], np.int64),
                np.array([8], np.int64),
                input_ori_shapes=((0,), (1,), (), ()),
            )

    def test_empty_probability_cannot_broadcast_to_nonempty_output(self):
        with self.assertRaisesRegex(ValueError, "not broadcastable"):
            golden.stateless_bernoulli_golden(
                np.array([2], np.int64),
                np.array([], np.float32),
                np.array([6], np.int64),
                np.array([8], np.int64),
                input_ori_shapes=((1,), (0,), (), ()),
            )

    def test_promoted_shape_placeholder_is_not_a_dimension(self):
        for placeholder in (0, 1, 7):
            for mode in (0, 1):
                with self.subTest(placeholder=placeholder, mode=mode):
                    result = golden.stateless_bernoulli_golden(
                        np.array([placeholder], np.int64),
                        np.array([0.5], np.float32),
                        np.array([6], np.int64),
                        np.array([8], np.int64),
                        probMode=mode,
                        input_ori_shapes=((), (), (), ()),
                        output_dtypes=("uint8",),
                    )
                    self.assertEqual(result.shape, ())
                    self.assertIn(result.item(), (0, 1))

    def test_rank_one_shape_list_preserves_values(self):
        out_shape, prob = golden._kernel_shape_and_prob(
            np.array([2], np.int64), np.array([0.5], np.float32), ((1,), (1,))
        )
        self.assertEqual(out_shape, [2])
        self.assertEqual(prob.shape, (1,))

    def test_empty_shape_list_produces_scalar(self):
        result = golden.stateless_bernoulli_golden(
            np.array([], np.int64),
            np.array(0.5, np.float32),
            np.array(6, np.int64),
            np.array(8, np.int64),
            input_ori_shapes=((0,), (), (), ()),
        )
        self.assertEqual(result.shape, ())

    def test_native_rank_zero_and_xpu_share_logical_shapes(self):
        out_shape, prob = golden._kernel_shape_and_prob(
            np.array(7, np.int64), np.array(0.5, np.float32)
        )
        self.assertEqual(out_shape, [])
        self.assertEqual(prob.shape, ())
        with mock.patch.object(
            golden, "_golden_bool", return_value=np.array(True)
        ) as sample:
            golden.stateless_bernoulli_golden(
                np.array([0], np.int64),
                np.array([0.5], np.float32),
                np.array(6),
                np.array(8),
                input_ori_shapes=((), (), (), ()),
            )
        self.assertEqual(sample.call_args.kwargs["prob_shape"], ())


if __name__ == "__main__":
    unittest.main()
