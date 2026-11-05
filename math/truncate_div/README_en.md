# TruncateDiv

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    √     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

- API function: performs the truncation division computation, and the result is rounded toward zero

- Computation formula:

  $$
  out_i = trunc(\frac{x1_i}{x2_i})
  $$

  where `trunc` indicates rounding toward zero (truncation).

- Exception: None

## Parameter Description

<table style="table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 352px">
  <col style="width: 213px">
  <col style="width: 100px">
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
      <td>The dividend in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT, INT32, UINT8, INT8, INT64, INT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>The divisor in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT, INT32, UINT8, INT8, INT64, INT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>out in the formula, which is the truncation division result.</td>
      <td>BFLOAT16, FLOAT16, FLOAT, INT32, UINT8, INT8, INT64, INT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

- The inputs x1 and x2 must satisfy the broadcast relationship.
- When the divisor is 0, the result is undefined behavior.

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph-mode invocation | [test_geir_truncate_div](./examples/test_geir_truncate_div.cpp)   | Invokes the TruncateDiv operator by building a graph through the [operator IR](./op_graph/truncate_div_proto.h). |
