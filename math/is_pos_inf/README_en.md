# IsPosInf

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    ×     |

## Function Description

- Operator function: determines which elements in a tensor are positive infinity, that is, +inf.
- Computation formula:

  $$
  out_i=(input_i == +inf)
  $$

- Examples:
  - If x=[9, 6, +inf], the result of isNegInf(x) is [False, False, True].

## Parameter Description

<table style="table-layout: fixed; width: 920px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 290px">
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
      <td>x</td>
      <td>Input</td>
      <td>Input parameter for the determination, which is input_i in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output parameter for the determination, which is out_i in the formula.</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Invocation Description

| Invocation Method | Invocation Sample                                             | Description                                                                                         |
|---------|----------------------------------------------------|----------------------------------------------------------------------------------------------|
| aclnn invocation | [test_aclnn_isposinf](./examples/test_aclnn_isposinf.cpp) | Invokes the IsPosInf operator through the [aclnnIsPosInf](./docs/aclnnIsPosInf_en.md) API  |
