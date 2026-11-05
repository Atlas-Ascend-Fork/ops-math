# Neg

English | [简体中文](./README.md)

## Product Support Status

| Product                                                     | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    √     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

- Operator function: computes the opposite number of each element in the input.

- Computation formula:
  $$
    output_i = (-1) * self_i
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
      <td>x</td>
      <td>Input</td>
      <td>Output tensor self in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT, DOUBLE, INT32, INT8, INT64, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Negated output, which is the output tensor output in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT, DOUBLE, INT32, INT8, INT64, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

None

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn invocation | [test_aclnn_neg](./examples/test_aclnn_neg.cpp) | Invokes the Neg operator through the [aclnnNeg](./docs/aclnnNeg&aclnnInplaceNeg_en.md) API.    |
| Graph-mode invocation | [test_geir_neg](./examples/test_geir_neg.cpp) | Invokes the Neg operator by building a graph through the [operator IR](./op_graph/neg_proto.h). |
