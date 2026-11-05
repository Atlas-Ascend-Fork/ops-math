# FusedMulAddAdd

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                       |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term>     |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                              |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function Description

- Operator function: fuses the `Mul`, `Add`, and `Add` subgraphs into a single operator, aligns the four inputs according to the NumPy broadcasting rules, and computes the multiply-add-add element-wise. This operator is commonly used in patterns such as `BatchMatmul + bias + residual`.

- Computation formula:

  $$
  y = x_1 \times x_2 + x_3 + x_4
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
      <td>FLOAT16, FLOAT, INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Multiplication input tensor x2 in the formula. Its shape must be broadcastable to x1.</td>
      <td>Same as x1</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x3</td>
      <td>Input</td>
      <td>Input tensor x3 of the first addition in the formula. Its shape must be broadcastable to x1.</td>
      <td>Same as x1</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x4</td>
      <td>Input</td>
      <td>Input tensor x4 of the second addition in the formula. Its shape must be broadcastable to x1.</td>
      <td>Same as x1</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output tensor y in the formula. Its shape is the same as that of x1.</td>
      <td>Same as x1</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

- x1, x2, x3, x4, and y must be of the same data type, and mixed data types are not supported.
- The numbers of dimensions (ranks) of x1, x2, x3, and x4 must not exceed 8. Otherwise, the check fails in the tiling phase.
- The computation order is `((x1 * x2) + x3) + x4` and cannot be exchanged.
- **x1 must be the complete output shape**: The shapes of x2, x3, and x4 can be broadcast upward to x1 according to the NumPy broadcasting rules (scalars, single-dimension broadcast, and cross-rank broadcast are supported), and the shape of output y is the same as x1. The current runtime **does not support broadcasting x1 itself upward** (that is, the scenario where x1 is smaller than the output shape, for example, x1=`[1]` and x2=`[3,4]`). Such test cases fail in the RunGraph phase.
- **Dynamic shapes containing `-1` are not supported**: If any dimension of x1, x2, x3, or x4 is `-1` (unknown dim), the scenario is not supported.

## Implementation

| Layer | File | Description |
| --- | --- | --- |
| Computation graph prototype | `op_graph/fused_mul_add_add_proto.h` | `REG_OP(FusedMulAddAdd)`, four inputs and one output |
| Operator definition | `op_host/fused_mul_add_add_def.cpp` | `OpDef::AddConfig("ascend950", ...)` |
| InferShape | `op_host/fused_mul_add_add_infershape.cpp` | Reuses `Ops::Base::InferShape4Broadcast(ctx, 4)` |
| Tiling | `op_host/arch35/fused_mul_add_add_tiling_arch35.{h,cpp}` | Calls `Ops::Base::BroadcastBaseTiling<OpDag>` in branches by dtype |
| DAG | `op_kernel/arch35/fused_mul_add_add_dag.h` | The fp32/fp16 path uses `Vec::Mul + Vec::Add + Vec::Add` with fp32 intermediate precision; the int32 path uses `Vec::Mul + Vec::Add + Vec::Add` |
| Struct | `op_kernel/arch35/fused_mul_add_add_struct.h` | `BRC_TEMP_SCH_MODE_KEY_DECL/SEL` |
| Kernel entry | `op_kernel/fused_mul_add_add_apt.cpp` | `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` + `BroadcastSch<schMode, OpDag>` |

### fp16 / fp32 path

```pseudocode
In0/In1/In2/In3 -- CopyInBrc -- Cast(->fp32) -- Vec::Mul(x1,x2) -- Vec::Add(+x3) -- Vec::Add(+x4) -- Cast(->T,RINT) -- CopyOut -- Out0
```

All inputs are first cast to fp32, and then `Vec::Mul + Vec::Add + Vec::Add` computes in three stages sequentially according to the formula
`y = x1 * x2 + x3 + x4`, and finally the result is cast back to T and written out.

### int32 path

```pseudocode
In0/In1/In2/In3 -- CopyInBrc -- Vec::Mul(x1,x2) -- Vec::Add(+x3) -- Vec::Add(+x4) -- CopyOut -- Out0
```

## Invocation Description

| Invocation Method   | Sample Code                                                     | Description                                                         |
| ---------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Graph mode | [test_geir_fused_mul_add_add](examples/arch35/test_geir_fused_mul_add_add.cpp) | Invokes the FusedMulAddAdd operator by building a graph through the [operator IR](op_graph/fused_mul_add_add_proto.h). |
