# DivNoNan

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                     |     √    |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    √    |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    √    |

## Function Description

- Operator function: performs the safe division computation and avoids the NaN value problem caused by a denominator of 0 during division. When the denominator is 0, 0 is returned instead of NaN.
- Computation formula:
  $$
  y = \begin{cases}
  \frac{x1}{x2}, & \text{if } x2 \neq 0 \\
  0, & \text{if } x2 = 0
  \end{cases}
  $$

## Parameter Description

<table style="table-layout: fixed; width: 1576px"><colgroup>
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
    </tr></thead>
  <tbody>
    <tr>
      <td>x1</td>
      <td>Input</td>
      <td>The input x1 in the formula, which serves as the numerator.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, INT32, UINT8, INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>The input x2 in the formula, which serves as the denominator.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, INT32, UINT8, INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>The output y in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, INT32, UINT8, INT8</td>
      <td>ND</td>
    </tr>
  </tbody></table>

  - <term>Atlas training products</term>, <term>Atlas inference products</term>, and <term>Atlas 200I/500 A2 inference products</term>: BFLOAT16 is not supported.

## Constraint Description

None

## Invocation Description

| Invocation Method   | Sample Code                                                     | Description                                                         |
| ---------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Graph mode | [test_geir_div_no_nan](examples/test_geir_div_no_nan.cpp) | Invokes the DivNoNan operator by building a graph through the [operator IR](op_graph/div_no_nan_proto.h). |
