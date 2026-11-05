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
- <term>Ascend 950PR&950DT products</term>: Supports empty tensors (0 elements) as inputs; the output is an empty tensor with the broadcast shape (0 elements).

## Invocation Description

| Invocation Method   | Sample Code                                                     | Description                                                         |
| ---------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Graph mode | [test_geir_fused_mul_add](examples/arch35/test_geir_fused_mul_add.cpp) | Invokes the FusedMulAdd operator by building a graph through the [operator IR](op_graph/fused_mul_add_proto.h). |
