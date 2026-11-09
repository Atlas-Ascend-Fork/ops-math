#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import tensorflow as tf

__spec__ = {
    "matrix_diag_part": "MatrixDiagPartKernelSpec",
    # TensorFlow E2E resolves plugin specs by api_name (tf./tensorflow. prefix).
    "tf.raw_ops.MatrixDiagPart": "MatrixDiagPartTFE2ESpec",
    "tensorflow.raw_ops.MatrixDiagPart": "MatrixDiagPartTFE2ESpec",
}


class MatrixDiagPartKernelSpec:
    """Kernel flow: numpy in, [numpy_out] out."""

    def golden(x, **kwargs):
        out = tf.raw_ops.MatrixDiagPart(input=x)
        return [out.numpy()]

    class ThirdPartyImpl:
        def __call__(self, x, **kwargs):
            print("Entering golden")
            out = tf.raw_ops.MatrixDiagPart(input=x)
            return out.numpy()

    third_party = {"tf": ThirdPartyImpl}

    tolerance = {
        "float32": {"standard": "binary_equal"},
        "float16": {"standard": "binary_equal"},
        "int32": {"standard": "binary_equal"},
        "int8": {"standard": "binary_equal"},
        "uint8": {"standard": "binary_equal"},
    }


class MatrixDiagPartTFE2ESpec:
    """TensorFlow E2E flow: tf.Tensor in, [tf.Tensor] out.

    A pure gather has no arithmetic tolerance, so the reference is compared with the
    exact ``binary_equal`` standard.
    """

    def golden(input, **kwargs):
        # TTK invokes the E2E reference by the TF API parameter name (input=...).
        out = tf.raw_ops.MatrixDiagPart(input=input)
        return [out]

    third_party = {"tf": MatrixDiagPartKernelSpec.ThirdPartyImpl}

    tolerance = {
        "float32": {"standard": "binary_equal"},
        "float16": {"standard": "binary_equal"},
        "int32": {"standard": "binary_equal"},
        "int8": {"standard": "binary_equal"},
        "uint8": {"standard": "binary_equal"},
    }
