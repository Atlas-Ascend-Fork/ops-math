# Muls

English | [简体中文](./README.md)

## Product Support Status

| Product                                                     | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |     ×     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

- Operator function: multiplies each element of the input tensor by a scalar variable.

- Computation formula:

$$y = x \times value$$

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
      <td>Input parameter for the Mul computation, which is x in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, INT16, INT32, INT64, COMPLEX32, COMPLEX64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>value</td>
      <td>Input attribute</td>
      <td>Scalar input parameter for the Mul computation, which is value in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, INT16, INT32, INT64, COMPLEX32, COMPLEX64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Element-wise product result of the tensors, which is y in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, INT16, INT32, INT64, COMPLEX32, COMPLEX64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

None

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_muls](./examples/test_aclnn_muls.cpp) | Invokes the Muls operator through the [aclnnMuls](./docs/aclnnMuls&aclnnInplaceMuls_en.md) API. |
