# Dot

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

- Interface function: computes the dot product of two one-dimensional tensors.

- Computation formula:

$$
self = [x_{1}, x_{2}, ..., x_{n}]
$$

$$
tensor = [y_{1}, y_{2}, ..., y_{n}]
$$

$$
out = x_{1}*y_{1} + x_{2}*y_{2} + ... + x_{n}*y_{n}
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
      <td>self</td>
      <td>Input</td>
      <td>The input tensor involved in the dot product computation, which is self in the formula. Its data type is the same as those of tensor and out, and its shape is the same as that of tensor.</td>
      <td>BFLOAT16, FLOAT16, FLOAT, INT8, INT32, UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>tensor</td>
      <td>Input</td>
      <td>The input tensor involved in the dot product computation, which is tensor in the formula. Its data type is the same as those of self and out, and its shape is the same as that of self.</td>
      <td>BFLOAT16, FLOAT16, FLOAT, INT8, INT32, UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>The dot product computation result, which is the output out in the formula. Its data type is the same as those of self and tensor.</td>
      <td>BFLOAT16, FLOAT16, FLOAT, INT8, INT32, UINT8</td>
      <td>ND</td>
    </tr>
  </tbody>
  </table>

  - <term>Atlas inference products</term> and <term>Atlas training products</term>: the BFLOAT16 data type is not supported.

## Constraint Description

None

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_dot](./examples/test_aclnn_dot.cpp) | Invokes the Dot operator through the [aclnnDot](./docs/aclnnDot_en.md) API. |
