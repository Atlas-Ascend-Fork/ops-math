# Square

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    √     |
| <term>Atlas inference products</term>                              |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

- Operator function: computes the square value of each element in the input tensor.

- Computation formula:

  $$
  y = x * x = x^2
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
      <td>The input tensor x in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT, INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>The output tensor y in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT, INT32, INT64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Atlas training products</term> and <term>Atlas inference products</term>: BFLOAT16 is not supported.

## Invocation Description

| Invocation Method | Invocation Sample                                                  | Description                                                                 |
|---------|-------------------------------------------------------|--------------------------------------------------------------------|
| aclnn invocation | [test_aclnn_square](./examples/arch35/test_aclnn_square.cpp) | Invokes the Square operator through the [aclnnSquare](./docs/aclnnSquare_en.md) API. |
| Graph-mode invocation | [test_geir_square](./examples/test_geir_square.cpp)     | Invokes the Square operator by building a graph through the [operator IR](./op_graph/square_proto.h).                   |
