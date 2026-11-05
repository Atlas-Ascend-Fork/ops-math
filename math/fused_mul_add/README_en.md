# FusedMulAdd

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

- Operator function: fuses the `Mul` and `Add` subgraphs into a single operator, aligns the three inputs according to the NumPy broadcasting rules, and computes the multiply-add element-wise.

- Computation formula:

  $$
  y = x_1 \times x_2 + x_3
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
      <td>Multiplication input tensor x2 in the formula. Its shape must be broadcastable with x1.</td>
      <td>Same as x1</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x3</td>
      <td>Input</td>
      <td>Addition input tensor x3 in the formula. Its shape must be broadcastable with the result of x1*x2.</td>
      <td>Same as x1</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output tensor y in the formula. Its shape is the unified shape after broadcasting x1, x2, and x3.</td>
      <td>Same as x1</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

- x1, x2, x3, and y must be of the same data type, and mixed data types are not supported.
- The numbers of dimensions (ranks) of x1, x2, and x3 must not exceed 8. Otherwise, the check fails in the tiling phase.
- Any NumPy broadcasting form is supported (including scalars, single-dimension broadcast, and cross-rank broadcast), and dynamic shapes and dynamic ranks are supported.

## Implementation

| Layer | File | Description |
| --- | --- | --- |
| Computation graph prototype | `op_graph/fused_mul_add_proto.h` | `REG_OP(FusedMulAdd)`, three inputs and one output |
| Operator definition | `op_host/fused_mul_add_def.cpp` | `OpDef::AddConfig("ascend950", ...)` |
| InferShape | `op_host/fused_mul_add_infershape.cpp` | Reuses `Ops::Base::InferShape4Broadcast(ctx, 3)` |
| Tiling | `op_host/arch35/fused_mul_add_tiling_arch35.{h,cpp}` | Calls `Ops::Base::BroadcastBaseTiling<OpDag>` in branches by dtype |
| DAG | `op_kernel/arch35/fused_mul_add_dag.h` | The fp32/fp16 path uses `Vec::Mul + Vec::Add` with fp32 intermediate precision; the int32 path uses `Vec::Mul + Vec::Add` |
| Struct | `op_kernel/arch35/fused_mul_add_struct.h` | `BRC_TEMP_SCH_MODE_KEY_DECL/SEL` |
| Kernel entry | `op_kernel/fused_mul_add_apt.cpp` | `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` + `BroadcastSch<schMode, OpDag>` |

### fp16 / fp32 path

```pseudocode
In0/In1/In2 -- CopyInBrc -- Cast(->fp32) -- Vec::Mul(x1,x2) -- Vec::Add(+x3) -- Cast(->T,RINT) -- CopyOut -- Out0
```

All inputs are first cast to fp32, and then `Vec::Mul + Vec::Add` computes in two stages sequentially according to the formula
`y = x1 * x2 + x3`, and finally the result is cast back to T and written out.

> `Vec::FusedMulAdd` is not used here. The underlying implementation of this API writes back to the src2 buffer in place,
> polluting the input of the next tile when a large broadcast tensor reuses the input UB buffer across tiles, and causing
> precision errors in fp32/fp16 huge broadcast scenarios. The ordinary `Vec::Mul + Vec::Add` is used to
> ensure that the cast results derived from all placeholders are read-only.

### int32 path

```pseudocode
In0/In1/In2 -- CopyInBrc -- Vec::Mul(x1,x2) -- Vec::Add(+x3) -- CopyOut -- Out0
```

## Invocation Description

| Invocation Method   | Sample Code                                                     | Description                                                         |
| ---------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Graph mode | [test_geir_fused_mul_add](examples/arch35/test_geir_fused_mul_add.cpp) | Invokes the FusedMulAdd operator by building a graph through the [operator IR](op_graph/fused_mul_add_proto.h). |
