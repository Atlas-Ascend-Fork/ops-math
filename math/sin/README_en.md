# Sin

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    √     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

- Operator function: performs the sine operation on each element of the input matrix and outputs the result.

- Computation formula:

$$
out_{i}=sin(input_{i})
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
      <td>Input parameter for the sin computation, which is input_i in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output parameter for the sin computation, which is out_i in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Atlas 200I/500 A2 inference products, Atlas inference products, and Atlas training products: BFLOAT16 is not supported.

## Constraint Description

- <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: the accuracy requirement is met when the input data of the FLOAT, FLOAT16, and BFLOAT16 types is in the range [-10^7, 10^7]. The accuracy cannot be guaranteed if the value is out of the range. Use the CPU for computation.
- <term>Atlas inference products</term>: the accuracy requirement is met when the input data of the FLOAT and FLOAT16 types is in the range [-65504, 65504]. The accuracy cannot be guaranteed if the value is out of the range. Use the CPU for computation.

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn invocation | [test_aclnn_sin](./examples/test_aclnn_sin.cpp) | Invokes the Sin operator through the [aclnnSin](./docs/aclnnSin&aclnnInplaceSin_en.md) API. |
