# Trunc

English | [简体中文](./README.md)

## Product Support Status

| Product                                | Supported |
|-----------------------------------| :----: |
| <term>Ascend 950PR&950DT products</term>                             |     √     |
| <term>Atlas A3 products</term>     |    √       |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |     ×     |
| <term>Atlas inference products</term>                             |   √     |
| <term>Atlas training products</term>                              |   √     |

## Function Description

- Operator function: rounds each element of the input tensor and discards the decimal part.
- Computation formula:

$$
out_i = \text{trunc}(input_i)
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>Input parameter for the trunc computation, which is input_i in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, INT8, INT32, UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output tensor after the trunc computation, which is out_i in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, INT8, INT32, UINT8</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## Constraint Description

None

## Invocation Description

| Invocation Method   | Invocation Sample                                                                                     | Description                                                                         |
| ---------- | ------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------- |
| aclnn invocation  | [test_aclnn_trunc](./examples/test_aclnn_trunc.cpp)                                         | Invokes the Trunc operator through the [aclnnTrunc](./docs/aclnnTrunc&aclnnInplaceTrunc_en.md) or [aclnnInplaceTrunc](./docs/aclnnTrunc&aclnnInplaceTrunc_en.md) API. |
