# TanhGrad

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
| ---- | :----:|
| <term>Ascend 950PR&950DT products</term>                       |    √     |
| <term>Atlas A3 products</term>       |    √     |
| <term>Atlas A2 products</term>       |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                       |    ×     |
| <term>Atlas inference products</term>                               |    √     |
| <term>Atlas training products</term>                               |    √     |

## Function Description

- Operator function: backward implementation of Tanh.

- Computation formula:

$$
d = tanh(x)= (\frac{e^{x} - {e^{-x}}}{e^{x} + {e^{-x}}})  \tag{1}
$$
$$
dy = 1 -tan(x)^2  \tag{2}
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
      <td>y</td>
      <td>Input</td>
      <td>Output of the forward computation.</td>
      <td>FLOAT16, FLOAT, DOUBLE, BFLOAT16, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dy</td>
      <td>Input</td>
      <td>Gradient of the previous operator in the backward pass.</td>
      <td>FLOAT16, FLOAT, DOUBLE, BFLOAT16, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>complex_conj</td>
      <td>Optional attribute</td>
      <td>Indicates whether to perform the conjugate operation on the complex type.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>z</td>
      <td>Output</td>
      <td>Output of the backward computation.</td>
      <td>FLOAT16, FLOAT, DOUBLE, BFLOAT16, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

None

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_tanh_backward](./examples/test_aclnn_tanh_backward.cpp) | Invokes the Abs operator through the [aclnnTanhBackward](./docs/aclnnTanhBackward_en.md) API. |
