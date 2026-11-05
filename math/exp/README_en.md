# Exp

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |   ×    |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×    |

## Function Description

- Operator function: returns a new tensor in which each element is the exponential of the corresponding element in the input tensor.
- Computation formula:

$$
out_{i} = e^{self_{i}}
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
      <td>Input parameter for the exp computation, which is self_i in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>base</td>
      <td>Optional attribute</td>
      <td><ul><li>Specifies the base of the exponential operation.</li><li>The default value is -1.0.</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>Optional attribute</td>
      <td><ul><li>Specifies the scaling coefficient.</li><li>The default value is 1.0.</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>shift</td>
      <td>Optional attribute</td>
      <td><ul><li>Specifies the offset.</li><li>The default value is 0.0.</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output parameter for the exp computation, which is out_i in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

None

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_exp](./examples/test_aclnn_exp.cpp) | Invokes the Exp operator through the [aclnnExp&aclnnInplaceExp](./docs/aclnnExp&aclnnInplaceExp_en.md) API. |
