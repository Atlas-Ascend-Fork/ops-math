# Tan

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                       |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term>     |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    √     |
| <term>Atlas inference products</term>                              |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

- Operator function: computes the tangent value of the input tensor element by element, and the output tensor has the same shape as the input tensor.

- Computation formula:

  $$
  y_i = \tan(x_i)
  $$

  Here, $x_i$ denotes the $i$-th element of the input tensor `x`, and $y_i$ denotes the $i$-th element of the output tensor `y`.

## Parameter Description

<table style="table-layout: fixed; width: 910px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 200px">
  <col style="width: 200px">
  <col style="width: 170px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter Name</th>
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data Type</th>
      <th>Data Format</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>The input tensor x in the formula.</td>
      <td>FLOAT16, FLOAT, BFLOAT16, INT32, DOUBLE, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>The output tensor y in the formula, with the same shape as x.</td>
      <td>FLOAT16, FLOAT, BFLOAT16, INT32, DOUBLE, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

- <term>Ascend 950PR&950DT products</term>, <term>Atlas A3 products</term>, and <term>Atlas A2 products</term>: the AI Core supports FLOAT16, FLOAT, BFLOAT16, and INT32.
- <term>Atlas 200I/500 A2 inference products</term>, <term>Atlas inference products</term>, and <term>Atlas training products</term>: the AI Core supports FLOAT16, FLOAT, and INT32, and does not support BFLOAT16.
- DOUBLE, COMPLEX64, and COMPLEX128 are computed through the AICPU path.

## Constraint Description

- The shapes of the input and output must be the same, and the data format is ND.
- In the AI Core implementation of <term>Ascend 950PR&950DT products</term>, the output is NaN when the input is NaN or Inf. A finite large input (including an absolute value greater than or equal to $10^7$) is processed through high-precision range reduction and outputs a finite value.

## Invocation Description

| Invocation Method | Sample Code | Description |
|---|---|---|
| aclnn API | [test_aclnn_tan.cpp](./examples/test_aclnn_tan.cpp) | Invokes the Tan operator through the [aclnnTan and aclnnInplaceTan](./docs/aclnnTan&aclnnInplaceTan_en.md) APIs. |
| GE graph mode | [test_geir_tan.cpp](./examples/test_geir_tan.cpp) | Invokes the Tan operator by building a graph through the [operator IR](./op_graph/tan_proto.h). |
