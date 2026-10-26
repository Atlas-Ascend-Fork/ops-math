# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""AccumulateNV2 TestSpec：GEIR/Kernel、ACLNN。

数学公式: y = Σ x_i

精度策略:
- CPU Golden 使用 TensorFlow ``tf.math.accumulate_n`` 计算，作为 cross_check 的高精度真值 ``g``。
- fp16 输入统一升精度到 fp32 累加后单次 cast 回：kernel 直连路径在 Vector 管线内
  实现（Cast→Add(fp32)→Cast），aclnnSum 路径与 op_api 的 promoteFp16 路径一致。
- 三方结果 ``b`` 使用 TensorFlow XPU 原始 dtype 计算，不做 cast。
- L1 比较 NPU 相对 CPU 的误差与 TensorFlow XPU 相对 CPU 的误差之比。
- 容差按 dtype 分级声明于 TestSpec ``tolerance``（浮点 cross_check L1、整数 binary_equal L2）。
- Kernel/GEIR 直连为同 shape 契约（算子本体不支持 broadcast，不做对齐）；aclnnSum 路由
  由 op_api 的 BroadcastTo 完成对齐（shape 可不同），golden 与三方参考同步该行为。
"""

import numpy as np

__golden__ = {
    "kernel": {"accumulate_nv2": "accumulate_nv2_golden"},
    "aclnn": {"aclnnSum": "aclnn_sum_golden"},
}

__spec__ = {
    "accumulate_nv2": "AccumulateNv2KernelTestSpec",
    "aclnnSum": "AclnnSumTestSpec",
    "tf.math.accumulate_n": "TensorFlowAccumulateNv2TestSpec",
    "tf.raw_ops.AccumulateNV2": "TensorFlowAccumulateNv2TestSpec",
}


def _as_tensor_list(value):
    tensors = list(value) if isinstance(value, (list, tuple)) else [value]
    if not tensors:
        raise ValueError("AccumulateNV2 requires at least one input tensor")
    return tensors


def _tensorflow_cpu_reference(values, promote_fp16=False, broadcast=False):
    """CPU reference using the TensorFlow competitor operator.

    Route semantics:
    - kernel/GEIR direct route (``broadcast=False``): same-shape contract —
      the operator does not support broadcast and rejects mismatched shapes,
      so the reference must not silently align them. fp16 uses the promote
      reference (see ``_fp16_promote_reference``).
    - aclnnSum route (``broadcast=True``): mirrors the op_api wrapper —
      BroadcastTo alignment of broadcastable inputs, then fp16 promoted to
      fp32, accumulated, and cast back once (aclnn_sum.cpp promoteFp16
      path). float32 and integer inputs are always computed at their
      original dtype (the TF competitor's native behavior).
    """
    import tensorflow as tf

    arrays = [np.asarray(v) for v in _as_tensor_list(values)]
    if broadcast:
        # aclnnSum route only: mirror the wrapper's BroadcastTo alignment.
        # numpy performs the shape unification and expansion: TF CPU's
        # broadcast_to rejects rank>=7 shape pairs ("not supported yet"),
        # while np.broadcast_shapes/np.broadcast_to cover the operator's full
        # rank<=8 domain; the expansion is pure shape replication with no
        # arithmetic, so numpy and TF are semantically identical here.
        target_shape = np.broadcast_shapes(*[a.shape for a in arrays])
        arrays = [np.broadcast_to(a, target_shape) for a in arrays]
    tensors = [_to_tensorflow_tensor(a, tf) for a in arrays]

    input_dtype = tensors[0].dtype
    if input_dtype == tf.float16 and promote_fp16:
        # fp32 accumulation with a single cast back (target_shape is inferred
        # from the tensors: inputs are either broadcast-aligned above or
        # already share one shape).
        promoted = [tf.cast(t, tf.float32) for t in tensors]
        result = tf.math.accumulate_n(promoted, tensor_dtype=tf.float32)
        return tf.cast(result, tf.float16)
    return tf.math.accumulate_n(tensors, tensor_dtype=input_dtype)


def _is_fp16_inputs(values, input_dtypes):
    """Whether the case is fp16-origin.

    TTK passes the ORIGINAL ``input_dtypes`` (pre-elevation) to custom golden
    functions: under cross_check the framework promotes fp16 inputs to fp32
    before invoking the golden, so the array dtype alone is unreliable —
    ``input_dtypes`` is authoritative when present.
    """
    if input_dtypes is not None:

        def _flatten(items):
            for item in items:
                if isinstance(item, (list, tuple)):
                    yield from _flatten(item)
                else:
                    yield str(item)

        dtypes = list(_flatten(input_dtypes))
        return bool(dtypes) and dtypes[0] == "float16"
    return np.asarray(values[0]).dtype == np.float16


def _fp16_promote_reference(values):
    """fp16 promote reference: fp32 accumulation with a single cast back.

    Deterministic promote semantics for both invocation forms: raw fp16
    inputs (cast up, accumulate, cast back once) and TTK's Promote-elevated
    fp32 inputs (fp16→fp32 is exact, so accumulate directly and cast back
    once) — both model the kernel's fp32 intermediate accumulation in the
    Vector pipeline (Cast→Add(fp32)→Cast) and the aclnnSum promoteFp16 path.
    """
    import tensorflow as tf

    tensors = [_to_tensorflow_tensor(v, tf) for v in _as_tensor_list(values)]
    promoted = [t if t.dtype == tf.float32 else tf.cast(t, tf.float32) for t in tensors]
    result = tf.math.accumulate_n(promoted, tensor_dtype=tf.float32)
    return tf.cast(result, tf.float16)


def _to_tensorflow_tensor(value, tf):
    """Convert a CPU Golden input to TensorFlow without changing its dtype."""
    if tf.is_tensor(value):
        return value
    if hasattr(value, "detach") and hasattr(value, "__dlpack__"):
        value = value.detach().contiguous().clone()
        return tf.experimental.dlpack.from_dlpack(value.__dlpack__())
    return tf.convert_to_tensor(value)


def _to_kernel_array(value):
    """Convert a TensorFlow CPU Golden result to TTK's ndarray representation."""
    return value.numpy()


def accumulate_nv2_golden(x, **kwargs):
    """GEIR/Kernel CPU Golden.

    fp16: promote semantics — fp32 accumulation with a single cast back (see
    ``_fp16_promote_reference``), matching the kernel's fp32 intermediate
    accumulation. The fp16-origin test uses the original ``input_dtypes``
    kwarg (TTK elevates fp16 inputs to fp32 under cross_check's golden
    Promote). Other dtypes: TF competitor reference at the original dtype.
    """
    tensors = _as_tensor_list(x)
    if _is_fp16_inputs(tensors, kwargs.get("input_dtypes")):
        return [_to_kernel_array(_fp16_promote_reference(tensors))]
    return [_to_kernel_array(_tensorflow_cpu_reference(tensors, promote_fp16=False))]


def tensorflow_accumulate_nv2_golden(inputs, shape=None, tensor_dtype=None, **kwargs):
    """TF E2E CPU Golden for tf.math.accumulate_n / tf.raw_ops.AccumulateNV2.

    TF competitor requires all inputs to have the same shape (no broadcast);
    fp16 uses the promote reference, consistent with the operator's fp32
    intermediate accumulation contract.
    """
    del shape, tensor_dtype, kwargs
    return [
        _to_kernel_array(
            _tensorflow_cpu_reference(inputs, promote_fp16=True, broadcast=False)
        )
    ]


def aclnn_sum_golden(tensors, out=None, **kwargs):
    """ACLNN CPU Golden: mirrors aclnnSum's BroadcastTo + fp16→fp32 promote."""
    del out, kwargs
    return [
        _to_kernel_array(
            _tensorflow_cpu_reference(tensors, promote_fp16=True, broadcast=True)
        )
    ]


class ThirdPartyTF:
    """TensorFlow XPU competitor: runs on GPU at the original input dtype.

    TTK dispatches third_party calls to the GPU server. tf.raw_ops.AccumulateNV2
    is the true competitor GPU kernel (fp16-domain sequential accumulation).

    TF competitor requires same-shape inputs (no broadcast). This class serves
    all three TestSpecs (kernel/aclnn/TF E2E); the aclnnSum route may pass
    broadcastable different-shaped inputs (the aclnn interface layer supports
    broadcast), so inputs are broadcast-aligned here before reaching the
    competitor kernel — the comparison target is "the correct sum result",
    not "the competitor's rejection behavior". Same-shape inputs pass through
    unchanged.
    """

    def __init__(self):
        import tensorflow as tf

        self._tf = tf
        self._accumulate = tf.raw_ops.AccumulateNV2

    def __call__(self, inputs=None, x=None, tensors=None, **kwargs):
        del kwargs
        values = inputs if inputs is not None else x
        if values is None:
            values = tensors
        # Broadcast-align different-shaped inputs (aclnnSum route) to the
        # common shape before calling the competitor kernel. numpy handles
        # the full rank<=8 domain (TF CPU broadcast_to rejects rank>=7).
        arrays = [np.asarray(v) for v in _as_tensor_list(values)]
        target_shape = np.broadcast_shapes(*[a.shape for a in arrays])
        arrays = [np.broadcast_to(a, target_shape) for a in arrays]
        values = [_to_tensorflow_tensor(a, self._tf) for a in arrays]
        return [self._accumulate(inputs=list(values), shape=list(target_shape))]


_L1_FLOAT_TOLERANCE = {
    "float16": {"standard": "cross_check", "level": "L1"},
    "float32": {"standard": "cross_check", "level": "L1"},
}

_INTEGER_TOLERANCE = {
    "int32": {"standard": "binary_equal", "level": "L2"},
    "int8": {"standard": "binary_equal", "level": "L2"},
    "uint8": {"standard": "binary_equal", "level": "L2"},
}

_ACCUMULATE_NV2_TOLERANCE = {
    **_L1_FLOAT_TOLERANCE,
    **_INTEGER_TOLERANCE,
}


class AccumulateNv2KernelTestSpec:
    golden = staticmethod(accumulate_nv2_golden)
    third_party = {"tf": ThirdPartyTF}
    tolerance = _ACCUMULATE_NV2_TOLERANCE


class AclnnSumTestSpec:
    golden = staticmethod(aclnn_sum_golden)
    third_party = {"tf": ThirdPartyTF}
    tolerance = _ACCUMULATE_NV2_TOLERANCE


class TensorFlowAccumulateNv2TestSpec:
    golden = staticmethod(tensorflow_accumulate_nv2_golden)
    third_party = {"tf": ThirdPartyTF}
    tolerance = _ACCUMULATE_NV2_TOLERANCE
