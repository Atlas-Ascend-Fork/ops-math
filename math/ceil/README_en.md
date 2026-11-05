# Ceil

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

- Operator function: returns the result of rounding up each element in the input tensor

- Computation formula:

$$
y_i =⌈x_i⌉
$$

## Parameter Description

<table class="tg" style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 50px">
  <col style="width: 70px">
  <col style="width: 120px">
  <col style="width: 300px">
  <col style="width: 50px">
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
      <td>Input x in the computation formula.</td>
      <td>FLOAT16, FLOAT, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output y in the computation formula.</td>
      <td>FLOAT16, FLOAT, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

None

## Invocation Description

| Invocation Method | Invocation Sample                                                | Description                                                             |
|--------------|-----------------------------------------------------|----------------------------------------------------------------|
| aclnn invocation | [test_aclnn_ceil.cpp](examples/test_aclnn_ceil.cpp) | Invokes the Ceil operator through the [aclnnCeil](docs/aclnnCeil&aclnnInplaceCeil_en.md) API. |
| Graph-mode invocation | [test_geir_ceil.cpp](examples/test_geir_ceil.cpp)   | Invokes the Ceil operator by building a graph through the [operator IR](op_graph/ceil_proto.h).                   |
