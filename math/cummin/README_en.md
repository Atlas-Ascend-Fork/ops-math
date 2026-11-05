# Cummin

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                     |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term>     |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    √     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                             |    √     |

## Function Description

- Operator function: computes the cumulative minimum of self and returns the minimum values and the corresponding indices.

- Computation formula:

  valuesOut:

  $$
  valuesOut_{i} = min(self_{1}, self_{2}, self_{3}, ...... , self_{i})
  $$

  indicesOut:

  $$
  indicesOut_{i} = argmin(self_{1}, self_{2}, self_{3}, ...... , self_{i})
  $$

## Parameter Description

<table style="table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 290px">
  <col style="width: 450px">
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
      <td>x</td>
      <td>Input</td>
      <td>self in the formula, which is the input Tensor.</td>
      <td>FLOAT16, FLOAT, DOUBLE, UINT8, INT8, INT16, INT32, INT64, BOOL, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>valuesOut in the formula, which is the cumulative minimum of x.</td>
      <td>FLOAT16, FLOAT, DOUBLE, UINT8, INT8, INT16, INT32, INT64, BOOL, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>argmin</td>
      <td>Output</td>
      <td>indicesOut in the formula, which is the index corresponding to y.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>Attribute</td>
      <td>Dimension to process</td>
      <td>int64_t</td>
      <td>-</td>
    </tr>

  </tbody></table>

## Constraint Description

None

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn invocation | [test_aclnn_cummin](./examples/test_aclnn_cummin.cpp) | Invokes the Cummin operator through the [aclnnCummin](./docs/aclnnCummin_en.md) API.    |
