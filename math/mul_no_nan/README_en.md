# MulNoNan

English | [简体中文](./README.md)

## Product Support Status

| Product                                                     | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                       |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term>     |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                              |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function Description

- Operator function: performs safe multiplication and returns 0 when x2 is 0, thereby masking the two exceptions `0 * inf = NaN` and `0 * NaN = NaN`. The semantics of floating-point types correspond to `tf.math.multiply_no_nan` of TensorFlow, and INT32 is computed with the same zero-check rule.

- Computation formula:

  $$
  y = \begin{cases}
  0, & \text{if } x2 = 0 \\
  x1 \times x2, & \text{if } x2 \neq 0
  \end{cases}
  $$

## Parameter Description

<table style="table-layout: fixed; width: 820px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 190px">
  <col style="width: 260px">
  <col style="width: 120px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter Name</th>
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data Type</th>
      <th>Data Format</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x1</td>
      <td>Input</td>
      <td>Multiplication input tensor x1 in the formula.</td>
      <td>FLOAT16, FLOAT, INT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Multiplication input tensor x2 in the formula, which is the operand for the zero check. Its shape must be broadcastable with x1.</td>
      <td>Same as x1</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output tensor y in the formula. Its shape is the unified shape of x1 and x2 after broadcasting.</td>
      <td>Same as x1</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

- x1, x2, and y must be of the same data type. Mixed data types are not supported.
- Only the x2 side is checked. `x2 = -0` also enters the zero branch (IEEE-754 `-0 == 0`). When `x2 != 0`, all IEEE behaviors are preserved as in normal multiplication (NaN / Inf propagate normally).
- The numbers of dimensions of x1 and x2 do not exceed 8. Within this range, NumPy broadcast is supported (including scalars, single-dimension broadcast, and cross-rank broadcast), and dynamic shape and dynamic rank are supported. The number of dimensions determined at runtime must also satisfy the preceding upper limit.
- <term>Ascend 950PR&950DT products</term>:
  - In GE graph mode, empty tensors that satisfy the broadcast relationship are supported. The output keeps the shape after broadcasting, with 0 elements.

## Implementation

| Layer | File | Description |
| --- | --- | --- |
| Graph prototype | `op_graph/mul_no_nan_proto.h` | `REG_OP(MulNoNan)`, two inputs and one output |
| Operator definition | `op_host/mul_no_nan_def.cpp` | `OpDef::AddConfig("ascend950", ...)` |
| InferShape | `op_host/mul_no_nan_infershape.cpp` | Reuses `Ops::Base::InferShape4Broadcast` |
| Tiling | `op_host/arch35/mul_no_nan_tiling_arch35.{h,cpp}` | Calls `Ops::Base::BroadcastBaseTiling<OpDag>` based on the dtype branch |
| DAG | `op_kernel/arch35/mul_no_nan_dag.h` | The fp32/int32 path performs native computation; the fp16/bf16 path promotes intermediate precision to fp32 |
| Struct | `op_kernel/arch35/mul_no_nan_struct.h` | `BRC_TEMP_SCH_MODE_KEY_DECL/SEL` |
| Kernel entry | `op_kernel/mul_no_nan_apt.cpp` | `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` + `BroadcastSch<schMode, OpDag>` |

### fp32 / int32 path

```pseudocode
In0/In1 -- CopyInBrc --+
                       +-- Mul(x1,x2) --+
                                        +-- Select(mask, MulRes, Zero) -- CopyOut -- Out0
            Const(0)  -- Duplicate -- Zero --+
                                             |
                       +-- Compare(NE, x2, Zero) -- mask
```

`Vec::Compare<u8, T, NE>` outputs a bitmask. `Vec::Select<u8, T, TENSOR_TENSOR>` selects elements element by element according to `mask ? MulRes : Zero`. When `x2 == 0`, even if `MulRes` is `NaN` (`0 · inf` / `0 · NaN`), the value is discarded by Select and 0 is output.

### fp16 / bf16 path

```pseudocode
In0/In1 -- CopyInBrc -- Cast(->fp32) --+
                                       +-- Mul --+
                                                 +-- Select -- Cast(->T,RINT) -- CopyOut -- Out0
            Const(0,fp32) -- Duplicate -- Zero --+
                                                 |
                              +-- Compare(NE) -- mask
```

fp16 / bf16 are promoted to fp32 as a whole for cmp/sel/mul, and `CAST_MODE_RINT` (round-to-nearest-even) is used at the end to cast back. Reasons:

1. This is consistent with the behavior of the DSL `mul_no_nan.py` falling back to fp32 when fp16 `vcmpsel` is unavailable;
2. This avoids fp16 intermediate overflow (for example, `3e4 · 3e4` first becomes inf in fp16 and is then processed by select, which introduces additional saturation uncertainty). The fp32 intermediate values have sufficient dynamic range.

## Invocation Description

| Invocation Method   | Sample Code                                                     | Description                                                         |
| ---------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| GE graph mode | [test_geir_mul_no_nan](examples/arch35/test_geir_mul_no_nan.cpp) | Invokes the MulNoNan operator by building a graph through the [operator IR](op_graph/mul_no_nan_proto.h). The sample can select a device through `--device-id=N`. |
