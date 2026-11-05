# RealDiv

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                       |    √     |
| <term>Atlas A3 products</term>       |    √     |
| <term>Atlas A2 products</term>       |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                       |    √     |
| <term>Atlas inference products</term>                               |    √     |
| <term>Atlas training products</term>                               |    √     |

## Function Description

- Operator function: returns the result of x1/x2 element-wise.

- Computation formula:

  $$
  y = x_1 / x_2
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
      <td>Input tensor x_1 in the formula</td>
      <td>FLOAT, FLOAT16, BF16, DOUBLE, UINT8, INT8,
                           UINT16, INT16, INT32, INT64, BOOL,
                           COMPLEX64, COMPLEX128</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Input tensor x2 in the formula</td>
      <td>Same as x1</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output tensor y in the formula</td>
      <td>Same as x1</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Invocation Description

| Invocation Method | Invocation Sample                                             | Description                                                                                         |
|---------|----------------------------------------------------|----------------------------------------------------------------------------------------------|
| Graph-mode invocation | [test_geir_real_div](./examples/test_geir_real_div.cpp)   | Invokes the RealDiv operator by building a graph through the [operator IR](./op_graph/real_div_proto.h)                                      |
