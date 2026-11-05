# SquareSumV1

English | [简体中文](./README.md)

## Product Support Status

| Product                                                                | Supported |
|:------------------------------------------------------------------| :------: |
| <term>Ascend 950PR&950DT products</term>                            |    √     |
| <term>Atlas A3 products</term>                      |    √     |
| <term>Atlas A2 products</term>                      |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                               |    ×     |
| <term>Atlas inference products</term>                                         |    √     |
| <term>Atlas training products</term>                                         |    √     |

## Function Description

- Operator function: computes the sum of squares of the input tensor along the specified axes.

- Computation formula:

$$
y = \sum_{i\in\text{axis}}x_i^2
$$

## Parameter Description

<table style="table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
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
      <td>FLOAT16, FLOAT, BFloat16</td>
      <td>ND</td>
    </tr>
     <tr>
      <td>y</td>
      <td>Output</td>
      <td>The output tensor y in the formula.</td>
      <td>FLOAT16, FLOAT, BFloat16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>Attribute</td>
      <td>A list of integers, specifying the axes along which the sum is computed.</td>
      <td>INT32_ARRAY</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>keep_dims</td>
      <td>Attribute</td>
      <td>Whether to keep the reduced dimensions in the output. The default value is false.</td>
      <td>bool</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

None

## Invocation Description

| Invocation Method | Invocation Sample                                                | Description                                                             |
|--------------|-----------------------------------------------------|----------------------------------------------------------------|
| Graph-mode invocation | [test_geir_square_sum_v1.cpp](examples/test_geir_square_sum_v1.cpp)   | Invokes the SquareSumV1 operator by building a graph through the [operator IR](op_graph/square_sum_v1_proto.h).                   |
